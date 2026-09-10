"""Canonical verifier for the realtime broker.

Closes the loop with NO device and NO being home: synthesizes spoken
questions, streams them into the live broker exactly as the Voice PE
firmware would, transcribes the spoken reply, and asserts on content.

Runs from anywhere the broker is reachable (host 8765). Exit 0 = green.

Usage: python tools/check.py [ws://host:8765]
Env: OPENAI_API_KEY (auto-loaded from broker/.env if unset).
"""
import asyncio
import json
import os
import struct
import sys
import urllib.request

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "..", ".env")
RATE = 24000

# (spoken question, [any-of accepted substrings in the transcribed reply]).
# The broker keeps conversational context for a full session (~50 min), so
# probes must be robust to prior history: each is framed "ignore prior
# context" and asserts on a token that appears in the answer regardless of
# phrasing. Deterministic + HA-independent so the check is signal, not flake.
# Each case asks for a FULL SENTENCE on purpose. The Live engine answers a
# bare factual question with a single word, which is about 300 ms of audio,
# and transcribing 300 ms is a coin flip: real replies of "Earth." came back
# as 'art' and 'ers.', and "Paris." as 'parrot'. Measured with
# tools/probe-style energy analysis, the audio was not clipped (there was
# ~4 s of leading silence and a clean 300 ms of speech), so the flake was in
# the transcription, not the broker. Asking for a sentence removes the
# artifact and is closer to how the puck is actually used.
CASES = [
    (
        (
            "New question, ignore anything before: in one full sentence, "
            "what is the capital of France?"
        ),
        ["paris"],
    ),
    (
        (
            "New question, ignore anything before: in one full sentence, "
            "what planet do humans live on?"
        ),
        ["earth"],
    ),
]


def load_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    with open(ENV) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit("OPENAI_API_KEY not set and not found in broker/.env")


KEY = load_key()

# How long a reply may go quiet before we call it finished. The Live engine
# can speak a short holding phrase, fall silent while the delegated backend
# works, then come back with the answer: a 3s idle window cut those replies
# off mid-thought and the check read the filler as the answer. Bounded by
# REPLY_MAX_WAIT so a genuinely dead broker still fails fast enough.
REPLY_IDLE = float(os.environ.get("REPLY_IDLE_SECS", "8.0"))
REPLY_MAX_WAIT = float(os.environ.get("REPLY_MAX_WAIT_SECS", "45.0"))
WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"


def _post(url: str, data: bytes, headers: dict, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def synth_pcm(text: str) -> bytes:
    body = json.dumps({
        "model": "gpt-4o-mini-tts", "voice": "alloy",
        "input": text, "response_format": "pcm",
    }).encode()
    return _post("https://api.openai.com/v1/audio/speech", body,
                 {"Authorization": f"Bearer {KEY}",
                  "Content-Type": "application/json"})


def wav_bytes(pcm: bytes) -> bytes:
    n = len(pcm)
    head = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
    head += struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
    head += b"data" + struct.pack("<I", n)
    return head + pcm


def transcribe(pcm: bytes) -> str:
    # multipart/form-data with the wav + model field.
    boundary = "----brokercheck"
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="model"\r\n\r\ngpt-4o-transcribe\r\n')
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="file"; filename="r.wav"\r\n'
                 "Content-Type: audio/wav\r\n\r\n")
    body = parts[0].encode() + parts[1].encode() + wav_bytes(pcm) + \
        f"\r\n--{boundary}--\r\n".encode()
    raw = _post("https://api.openai.com/v1/audio/transcriptions", body,
                {"Authorization": f"Bearer {KEY}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(raw).get("text", "")


async def ask(question: str) -> bytes:
    speech = synth_pcm(question)
    async with websockets.connect(WS_URL, max_size=None) as ws:
        chunk = int(RATE * 0.02) * 2
        for i in range(0, len(speech), chunk):
            await ws.send(speech[i:i + chunk])
            await asyncio.sleep(0.02)
        for i in range(0, RATE * 2, chunk):  # 1s trailing silence
            await ws.send(b"\x00\x00" * (chunk // 2))
            await asyncio.sleep(0.02)
        out = bytearray()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + REPLY_MAX_WAIT
        while True:
            # Cap each wait by whatever is left of the total, so a silent
            # broker cannot overrun REPLY_MAX_WAIT by a whole idle window.
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=min(REPLY_IDLE, remaining))
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            if isinstance(msg, bytes):
                out.extend(msg)
    return bytes(out)


def _listening(url: str, timeout: float = 2.0) -> bool:
    """Whether anything is accepting connections at this ws:// URL.

    `make check` is wired into a hook, so it runs whenever a shell lands in
    this repo, including when no broker is up. Without this probe that case
    surfaced as a 20-line ConnectionRefusedError traceback out of
    websockets, which reads like the code is broken rather than "nothing is
    running" — four times in one session before anyone fixed it.
    """
    import socket  # local: only needed for this preflight

    rest = url.split("://", 1)[-1].split("/", 1)[0]
    host, _, port = rest.rpartition(":")
    try:
        target = (host or "127.0.0.1", int(port))
    except ValueError:
        return True  # unparseable: let the real connection report it
    try:
        with socket.create_connection(target, timeout):
            return True
    except OSError:
        return False


async def main() -> int:
    print(f"check: broker={WS_URL}  cases={len(CASES)}")
    if not _listening(WS_URL):
        print(f"  no broker is listening on {WS_URL}, so there is nothing to check.")
        print("  Start one, or point this at one that is already up:")
        print("    live puck's broker:   make check WS=ws://127.0.0.1:8765")
        print("    isolated dev broker:  see 'Two engines' in README.md")
        return 1
    failures = 0
    for question, accept in CASES:
        audio = await ask(question)
        if not audio:
            print(f"  FAIL  {question!r}\n        no audio returned")
            failures += 1
            continue
        reply = transcribe(audio).lower()
        ok = any(a in reply for a in accept)
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {question!r}\n        reply={reply!r}")
        if not ok:
            print(f"        expected any of {accept}")
            failures += 1
    print(f"\n{'GREEN' if failures == 0 else 'RED'}: "
          f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
