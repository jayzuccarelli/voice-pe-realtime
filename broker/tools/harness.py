"""End-to-end reliability harness for the voice-pe realtime broker.

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

import array
import asyncio
import json
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import deque
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
    boundary = "----voicepeharness"
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


async def s_background_rejection(url):
    # After a real turn, stream speech that is NOT addressed to the assistant
    # (TV / news / side conversation). It should stay silent (wait_for_user),
    # not answer the room. Approximates the "responds to the TV" failure.
    async with session(url) as c:
        r1 = await c.ask("In one short sentence, what is the capital of France?")
        if not r1.got_audio:
            return False, "first (addressed) turn got no audio", r1.first_audio_ms
        bg = ("And coming up next on the evening news at eleven, the city council "
              "voted today to approve the new downtown transit budget. More on that "
              "story after the break.")
        r2 = await c.ask(bg, voice="echo")
        t2 = r2.transcript().lower() if r2.got_audio else ""
        ok = not r2.got_audio
        return ok, f"bg_reply={r2.seconds:.1f}s want0 {t2[:50]!r}", None


async def s_challenge_follow_up(url):
    # Both real turns the TV test swallowed (2026-07-01) were follow-ups to
    # the bot's own answer. A bare challenge ("are you sure?") has no topic
    # words at all — the hardest addressed-speech case for the background
    # gate. The bot must answer, not wait_for_user it away.
    async with session(url) as c:
        r1 = await c.ask("In one short sentence, what is the capital of France?")
        if not r1.got_audio:
            return False, "setup turn got no audio", r1.first_audio_ms
        r2 = await c.ask("Are you sure about that?")
        t = r2.transcript().lower() if r2.got_audio else ""
        return r2.got_audio, f"challenge reply={t!r}", r2.first_audio_ms


async def s_tv_line_after_answer(url):
    # Counter-metric to the follow-up bias in BACKGROUND_GUIDANCE: a
    # conversational TV line in a different voice right after the bot
    # answers — the exact false-accept from the TV test ("It's good, huh?",
    # 2026-07-01 20:49:56). Want silence. Pre-declared tradeoff: swallowing
    # a real follow-up is worse than answering a TV line, so under --soak
    # this scenario is allowed to be the flakier of the pair — but it must
    # not fail while challenge_follow_up also fails.
    async with session(url) as c:
        r1 = await c.ask("In one short sentence, what is the capital of France?")
        if not r1.got_audio:
            return False, "setup turn got no audio", r1.first_audio_ms
        tv = ("Oh man, it's good, huh? I told you this show gets better. "
              "Hang on, I'm gonna grab another drink before it starts.")
        r2 = await c.ask(tv, voice="echo")
        t2 = r2.transcript().lower() if r2.got_audio else ""
        ok = not r2.got_audio
        return ok, f"tv_reply={r2.seconds:.1f}s want0 {t2[:50]!r}", None


async def s_mid_speech_disconnect(url):
    # Drop the connection mid-word (WiFi blip / firmware auto-stop), leaving
    # uncommitted audio in OpenAI's input buffer, then reconnect. Gate for
    # the on-disconnect buffer clear: the orphaned audio must not come back
    # as a ghost turn (the stray 'Bye.' answered before the real question,
    # TV test 2026-07-01 20:49), and the session must still answer.
    pcm = synth("What is the capital of France? Also, could you tell me")
    half = (len(pcm) // 2) // CHUNK * CHUNK
    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        await Client(ws)._stream(pcm[:half])  # hard drop mid-word, no silence
    async with session(url) as c:
        quiet = await c.listen(3.0)
        if quiet.got_audio:
            t = quiet.transcript().lower()
            return False, f"ghost reply after reconnect: {t[:60]!r}", None
        r = await c.ask("In one short sentence, what is two plus two?")
        t = r.transcript().lower() if r.got_audio else ""
        ok = r.got_audio and any(w in t for w in ("four", "4"))
        return ok, f"post-drop reply={t!r}", r.first_audio_ms


# ----------------------------------------------------------------------------
# Barge-in scenarios (NCC_GATE=on brokers ONLY — see BARGEIN_SCENARIOS below).
# Simulates the acoustic loop the puck will produce once the firmware mic
# gate is removed: while the bot's reply streams down, an attenuated delayed
# copy streams back up as "mic" audio (synthetic AEC residual). The NCC gate
# must eat the echo (no self-trigger) but pass a real voice shouting over it
# (barge-in interrupts).
# ----------------------------------------------------------------------------
def _scale(pcm: bytes, num: int, den: int) -> bytes:
    a = array.array("h", pcm)
    for i in range(len(a)):
        # Clamp — gain > 1 (the barge path uses 3/2) overflows int16 on loud
        # samples, and array('h') assignment raises OverflowError.
        a[i] = max(-32768, min(32767, a[i] * num // den))
    return a.tobytes()


def _mix(a_pcm: bytes, b_pcm: bytes) -> bytes:
    a = array.array("h", a_pcm)
    b = array.array("h", b_pcm)
    for i in range(min(len(a), len(b))):
        s = a[i] + b[i]
        a[i] = max(-32768, min(32767, s))
    return a.tobytes()


async def _echo_loop(url: str, barge: bool):
    """Full-duplex synthetic-echo session.

    Returns (reply_seconds, ghost: bool, interrupted: bool, barge_ms).
    """
    question = synth("Count slowly from one to fifteen, one number at a time, nothing else.")
    barge_pcm = _scale(synth("Stop! Stop talking now. Please stop.", voice="echo"), 3, 2) if barge else b""

    reply = bytearray()
    last_rx = [0.0]
    interrupted = [False]
    echo_q: deque[bytes] = deque()
    stop = asyncio.Event()

    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        c = Client(ws)
        await c._stream(question)

        async def receiver() -> None:
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    return
                if isinstance(msg, (bytes, bytearray)):
                    if not reply:
                        # acoustic-path delay: echo starts ~100ms behind
                        for _ in range(5):
                            echo_q.append(bytes(CHUNK))
                    reply.extend(msg)
                    last_rx[0] = time.monotonic()
                    att = _scale(bytes(msg), 3, 5)  # loud residual (0.6x)
                    for i in range(0, len(att) - CHUNK + 1, CHUNK):
                        echo_q.append(att[i : i + CHUNK])
                elif "interrupt" in str(msg):
                    interrupted[0] = True

        async def sender() -> None:
            barge_pos = 0
            barge_started = [None]
            # Absolute 20ms clock: a per-tick sleep(0.02) drifts a few ms per
            # chunk and walks the synthetic echo out of the gate's lag-search
            # window after ~4s. Real device echo lag is fixed; the rig's
            # must be too.
            t0 = time.monotonic()
            n = 0
            while not stop.is_set():
                chunk = echo_q.popleft() if echo_q else bytes(CHUNK)
                if barge and len(reply) > 2 * RATE * 2 and barge_pos < len(barge_pcm):
                    if barge_started[0] is None:
                        barge_started[0] = time.monotonic()
                        sender.barge_t0 = barge_started[0]
                    chunk = _mix(chunk, barge_pcm[barge_pos : barge_pos + CHUNK])
                    barge_pos += CHUNK
                await ws.send(chunk)
                n += 1
                await asyncio.sleep(max(0.0, t0 + n * 0.02 - time.monotonic()))

        rx = asyncio.create_task(receiver())
        tx = asyncio.create_task(sender())
        t0 = time.monotonic()
        ghost = False
        barge_ms = None
        try:
            if barge:
                # success = broker flushes the device (interrupt frame) soon
                # after the barge utterance starts
                while time.monotonic() - t0 < 40:
                    await asyncio.sleep(0.1)
                    if interrupted[0]:
                        break
                if interrupted[0] and getattr(sender, "barge_t0", None):
                    barge_ms = (time.monotonic() - sender.barge_t0) * 1000
            else:
                # wait for the reply to finish (2.5s idle), then 3 more
                # seconds of echo-free line: any NEW audio is a self-trigger
                while time.monotonic() - t0 < 40:
                    await asyncio.sleep(0.25)
                    if reply and time.monotonic() - last_rx[0] > 2.5:
                        break
                settled = len(reply)
                await asyncio.sleep(3.0)
                ghost = len(reply) - settled > RATE  # >0.5s of new audio
        finally:
            stop.set()
            for t in (rx, tx):
                t.cancel()
    return len(reply) / 2 / RATE, ghost, interrupted[0], barge_ms


async def s_ncc_echo_no_selftrigger(url):
    reply_s, ghost, _, _ = await _echo_loop(url, barge=False)
    ok = reply_s > 3 and not ghost
    return ok, f"reply={reply_s:.1f}s ghost_after_echo={ghost}", None


async def s_ncc_echo_barge(url):
    reply_s, _, interrupted, barge_ms = await _echo_loop(url, barge=True)
    ok = interrupted and barge_ms is not None and barge_ms < 2000
    lat = f"{barge_ms:.0f}ms" if barge_ms is not None else "n/a"
    return ok, f"interrupted={interrupted} barge_latency={lat} reply={reply_s:.1f}s", None


SCENARIOS = {
    "capital_qa": s_capital,
    "follow_up_multiturn": s_follow_up,
    "weather_ha_tool": s_weather_tool,
    "no_reply_to_silence": s_no_reply_to_silence,
    "no_ghost_on_connect": s_no_ghost_on_connect,
    "reconnect": s_reconnect,
    "background_rejection": s_background_rejection,
    "challenge_follow_up": s_challenge_follow_up,
    "tv_line_after_answer": s_tv_line_after_answer,
    "mid_speech_disconnect": s_mid_speech_disconnect,
}

# Run ONLY against an NCC_GATE=on broker (--bargein or --only <name>). In
# off/shadow modes the mic is silence-fed during playback, so the barge
# scenario cannot pass by design.
BARGEIN_SCENARIOS = {
    "ncc_echo_no_selftrigger": s_ncc_echo_no_selftrigger,
    "ncc_echo_barge": s_ncc_echo_barge,
}

ALL_SCENARIOS = {**SCENARIOS, **BARGEIN_SCENARIOS}


async def run(url: str, soak: int = 1, only: str | None = None) -> int:
    print(f"== voice-pe broker reliability harness -> {url} ==")
    if soak > 1:
        print(f"   soak mode: {soak} rounds")
    if only:
        scenarios = {only: ALL_SCENARIOS[only]}
    elif "--bargein" in sys.argv:
        scenarios = BARGEIN_SCENARIOS
    else:
        scenarios = SCENARIOS
    results: list[tuple[str, bool, str, float | None]] = []
    latencies: list[float] = []
    for rnd in range(soak):
        for name, fn in scenarios.items():
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
    flag_values = {
        sys.argv.index(f) + 1 for f in ("--soak", "--only") if f in sys.argv
    }
    args = [
        a for i, a in enumerate(sys.argv)
        if i > 0 and i not in flag_values and not a.startswith("--")
    ]
    url = args[0] if args else "ws://127.0.0.1:8766"
    soak = 1
    if "--soak" in sys.argv:
        soak = int(sys.argv[sys.argv.index("--soak") + 1])
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
        if only not in ALL_SCENARIOS:
            raise SystemExit(
                f"unknown scenario {only!r}; one of: {', '.join(ALL_SCENARIOS)}"
            )
    raise SystemExit(asyncio.run(run(url, soak, only)))


if __name__ == "__main__":
    main()
