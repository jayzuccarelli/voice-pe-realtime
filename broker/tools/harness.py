"""End-to-end reliability harness for the Domus realtime broker.

Drives the broker exactly like the Voice PE firmware does — streams PCM16 /
24 kHz / mono speech up the WebSocket in 20 ms frames with trailing silence —
then collects the spoken reply, transcribes it, and asserts on content,
turn-taking, and latency. No hardware needed; test speech is synthesized with
OpenAI TTS and replies are transcribed with Whisper.

Targets a RUNNING broker. Defaults to ws://127.0.0.1:8766 on purpose so it does
NOT fight the live puck on :8765 (the broker is single-client — connecting
kicks whoever is already on). Spin an isolated broker on 8766 to run this.

    OPENAI_API_KEY=... python -m broker.tools.harness [ws://host:port] [--soak N]

Exit code is non-zero if any scenario fails, so it drops straight into CI /
`make check`.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import websockets

RATE = 24000
CHUNK = int(RATE * 0.02) * 2  # 20 ms of PCM16
KEY = os.environ["OPENAI_API_KEY"]
SILENCE_1S = b"\x00\x00" * RATE


# ----------------------------------------------------------------------------
# OpenAI TTS / STT helpers (plain HTTP, no extra deps — matches existing tools)
# ----------------------------------------------------------------------------
def _send(req: urllib.request.Request, attempts: int = 4) -> bytes:
    """POST with retry — OpenAI's audio endpoints occasionally blip (429/5xx,
    even a transient 404). A flaky API call must not fail a scenario."""
    delay = 1.0
    for n in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if n == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def synth(text: str, voice: str = "alloy") -> bytes:
    """Text -> raw PCM16/24k/mono, the broker's exact input format."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({
            "model": "gpt-4o-mini-tts", "voice": voice,
            "input": text, "response_format": "pcm",
        }).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    return _send(req)


def _wav(pcm: bytes) -> bytes:
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", n) + pcm)


def transcribe(pcm: bytes) -> str:
    """PCM16/24k -> text via Whisper. '' if there's no speech."""
    if len(pcm) < RATE:  # < 0.5s, nothing worth sending
        return ""
    boundary = "----domusharness"
    body = bytearray()
    for key, val in (("model", "whisper-1"), ("response_format", "text")):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\""
                 f"\r\n\r\n{val}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             "filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += _wav(pcm) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions", data=bytes(body),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return _send(req).decode().strip()


# ----------------------------------------------------------------------------
# Conversation client
# ----------------------------------------------------------------------------
@dataclass
class Reply:
    audio: bytes
    texts: list[str] = field(default_factory=list)
    first_audio_ms: float | None = None

    @property
    def seconds(self) -> float:
        return len(self.audio) / 2 / RATE

    @property
    def got_audio(self) -> bool:
        return len(self.audio) > RATE  # > 0.5s of speech

    def transcript(self) -> str:
        return transcribe(self.audio)


class Client:
    def __init__(self, ws):
        self.ws = ws
        self._send_done = 0.0

    async def _stream(self, pcm: bytes) -> None:
        for i in range(0, len(pcm), CHUNK):
            await self.ws.send(pcm[i:i + CHUNK])
            await asyncio.sleep(0.02)

    async def _collect(self, idle: float = 2.5, max_wait: float = 25.0) -> Reply:
        audio = bytearray()
        texts: list[str] = []
        first: float | None = None
        start = time.monotonic()
        while True:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=idle)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            if isinstance(msg, (bytes, bytearray)):
                if first is None:
                    first = time.monotonic()
                audio.extend(msg)
            else:
                texts.append(msg)
            if time.monotonic() - start > max_wait:
                break
        first_ms = (first - self._send_done) * 1000 if first else None
        return Reply(bytes(audio), texts, first_ms)

    async def ask(self, text: str, voice: str = "alloy") -> Reply:
        """Speak a turn (speech + 1s trailing silence) and collect the reply."""
        await self._stream(synth(text, voice))
        await self._stream(SILENCE_1S)
        self._send_done = time.monotonic()
        return await self._collect()

    async def listen(self, seconds: float, with_silence: bool = True) -> Reply:
        """Hold the line (optionally streaming silence) and capture anything
        the broker pushes unprompted. Used to prove it stays quiet."""
        if with_silence:
            await self._stream(b"\x00\x00" * int(RATE * seconds))
        self._send_done = time.monotonic()
        return await self._collect(idle=seconds, max_wait=seconds + 2)


@asynccontextmanager
async def session(url: str):
    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        yield Client(ws)


# ----------------------------------------------------------------------------
# Scenarios — each returns (passed: bool, detail: str, latency_ms: float | None)
# ----------------------------------------------------------------------------
async def s_capital(url):
    async with session(url) as c:
        r = await c.ask("In one short sentence, what is the capital of France?")
        t = r.transcript().lower()
        ok = r.got_audio and "paris" in t
        return ok, f"reply={t!r}", r.first_audio_ms


async def s_follow_up(url):
    async with session(url) as c:
        await c.ask("What is the capital of France?")
        r = await c.ask("Roughly how many people live there?")
        t = r.transcript().lower()
        ok = r.got_audio and any(w in t for w in ("million", "two", "2", "paris"))
        return ok, f"follow-up reply={t!r}", r.first_audio_ms


async def s_weather_tool(url):
    async with session(url) as c:
        r = await c.ask("What's the weather like right now?")
        t = r.transcript().lower()
        ok = r.got_audio and any(
            w in t for w in ("degree", "temperature", "weather", "cloud",
                             "sun", "rain", "wind", "humid", "warm", "cold"))
        return ok, f"weather reply={t!r}", r.first_audio_ms


async def s_no_reply_to_silence(url):
    async with session(url) as c:
        await c.ask("What is the capital of France?")  # real turn first
        quiet = await c.listen(3.5)  # then 3.5s of silence
        ok = not quiet.got_audio
        return ok, f"audio_after_silence={quiet.seconds:.1f}s (want 0)", None


async def s_no_ghost_on_connect(url):
    async with session(url) as c:
        quiet = await c.listen(2.5)  # silence before any speech
        ok = not quiet.got_audio
        return ok, f"ghost_audio={quiet.seconds:.1f}s (want 0)", None


async def s_reconnect(url):
    async with session(url) as c:
        r1 = await c.ask("What is the capital of France?")
    if not r1.got_audio:
        return False, "first session got no audio", r1.first_audio_ms
    async with session(url) as c:  # fresh connection — the fjfricke #9 bug
        r2 = await c.ask("In one short sentence, what is two plus two?")
    t = r2.transcript().lower()
    ok = r2.got_audio and any(w in t for w in ("four", "4"))
    return ok, f"reconnect reply={t!r}", r2.first_audio_ms


SCENARIOS = {
    "capital_qa": s_capital,
    "follow_up_multiturn": s_follow_up,
    "weather_ha_tool": s_weather_tool,
    "no_reply_to_silence": s_no_reply_to_silence,
    "no_ghost_on_connect": s_no_ghost_on_connect,
    "reconnect": s_reconnect,
}


async def run(url: str, soak: int = 1) -> int:
    print(f"== Domus broker reliability harness -> {url} ==")
    if soak > 1:
        print(f"   soak mode: {soak} rounds")
    results: list[tuple[str, bool, str, float | None]] = []
    latencies: list[float] = []
    for rnd in range(soak):
        for name, fn in SCENARIOS.items():
            label = f"{name}#{rnd + 1}" if soak > 1 else name
            t0 = time.monotonic()
            try:
                ok, detail, lat = await fn(url)
            except Exception as e:  # noqa: BLE001 — a crash IS a failed scenario
                ok, detail, lat = False, f"EXCEPTION {type(e).__name__}: {e}", None
            dt = time.monotonic() - t0
            if lat is not None:
                latencies.append(lat)
            latstr = f"{lat:.0f}ms 1st-audio" if lat is not None else "—"
            print(f"[{'PASS' if ok else 'FAIL'}] {label:<26} {latstr:<16} {dt:4.1f}s  {detail}")
            results.append((label, ok, detail, lat))

    passed = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n{passed}/{len(results)} scenarios passed")
    if latencies:
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        print(f"first-audio latency: p50={p50:.0f}ms  p95={p95:.0f}ms  "
              f"min={min(latencies):.0f}  max={max(latencies):.0f}  n={len(latencies)}")
    return 0 if passed == len(results) else 1


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    url = args[0] if args else "ws://127.0.0.1:8766"
    soak = 1
    if "--soak" in sys.argv:
        soak = int(sys.argv[sys.argv.index("--soak") + 1])
    raise SystemExit(asyncio.run(run(url, soak)))


if __name__ == "__main__":
    main()
