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
CASES = [
    ("New question, ignore anything before: what is the capital of France?",
     ["paris"]),
    ("New question, ignore anything before: what planet do humans live on?",
     ["earth"]),
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
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                out.extend(msg)
    return bytes(out)


async def main() -> int:
    print(f"check: broker={WS_URL}  cases={len(CASES)}")
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
