"""Hardware-free test client for the Domus Realtime broker.

Mimics what the Voice PE firmware does: stream PCM16/24k/mono speech up the
WebSocket, then receive the response audio back as binary frames. No device
needed. Test speech is synthesized via OpenAI's plain TTS (pcm format =
24k/16-bit/mono, the broker's exact input format).

Usage: OPENAI_API_KEY=... python client.py [ws://host:8080] ["question text"]
"""
import asyncio
import json
import os
import struct
import sys
import urllib.request

import websockets

WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8080"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else (
    "Hi! In one short sentence, what is the capital of France?"
)
RATE = 24000
KEY = os.environ["OPENAI_API_KEY"]


def synth_pcm(text: str) -> bytes:
    """OpenAI TTS -> raw PCM16 24k mono bytes."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            "input": text,
            "response_format": "pcm",
        }).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def write_wav(path: str, pcm: bytes) -> None:
    n = len(pcm)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n))
        f.write(pcm)


async def main() -> None:
    print(f"synthesizing test utterance: {QUESTION!r}")
    speech = synth_pcm(QUESTION)
    write_wav("input.wav", speech)
    print(f"  got {len(speech)} bytes of input PCM "
          f"({len(speech)/2/RATE:.1f}s); wrote input.wav")

    print(f"connecting to broker at {WS_URL}")
    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("  connected; streaming speech in 20ms chunks")
        chunk = int(RATE * 0.02) * 2  # 20ms of PCM16
        for i in range(0, len(speech), chunk):
            await ws.send(speech[i:i + chunk])
            await asyncio.sleep(0.02)
        # Trailing silence so server_vad detects end-of-speech.
        silence = b"\x00\x00" * int(RATE * 1.0)
        for i in range(0, len(silence), chunk):
            await ws.send(silence[i:i + chunk])
            await asyncio.sleep(0.02)
        print("  speech sent; collecting response audio (idle-stop 3s)")

        out = bytearray()
        frames = 0
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                out.extend(msg)
                frames += 1
            else:
                print(f"  text frame: {msg[:200]}")

    print(f"\nRESULT: received {frames} audio frames, {len(out)} bytes "
          f"({len(out)/2/RATE:.1f}s of speech)")
    if out:
        write_wav("response.wav", bytes(out))
        print("wrote response.wav")
    else:
        print("NO AUDIO RETURNED — check broker logs")


if __name__ == "__main__":
    asyncio.run(main())
