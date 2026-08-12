"""Hardware-free barge-in test for the voice-pe broker.

Streams a question, lets the bot start a long reply, then streams an
INTERRUPTING utterance mid-reply. Proves the broker-side barge-in chain:
device-mic audio (mid-response) -> OpenAI server VAD -> Pipecat interrupt ->
_BotPlaybackGate sends {"type":"interrupt"} (which THIS client receives
as a text frame). No Voice PE device, no human voice needed.

Does NOT test echo/self-trigger (acoustic) or the firmware gate (device-side).

Usage: OPENAI_API_KEY=... python test_bargein.py [ws://127.0.0.1:8766]
"""
import asyncio
import json
import os
import sys
import time
import urllib.request

import websockets

WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8766"
RATE = 24000
KEY = os.environ["OPENAI_API_KEY"]
CHUNK = int(RATE * 0.02) * 2  # 20ms PCM16

Q1 = ("Please count slowly out loud from one all the way to fifty, "
      "saying one number at a time. Do not stop.")
INTERRUPTOR = "Stop! Stop counting! What is two plus two?"


def synth(text: str, voice: str = "echo") -> bytes:
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({
            "model": "gpt-4o-mini-tts", "voice": voice,
            "input": text, "response_format": "pcm",
        }).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


async def stream(ws, pcm: bytes) -> None:
    for i in range(0, len(pcm), CHUNK):
        await ws.send(pcm[i:i + CHUNK])
        await asyncio.sleep(0.02)


async def main() -> None:
    print("synthesizing test utterances...")
    q1 = synth(Q1)
    interruptor = synth(INTERRUPTOR)
    print(f"  q1={len(q1)/2/RATE:.1f}s  interruptor={len(interruptor)/2/RATE:.1f}s")

    state = {"interrupt_frame": False, "text_frames": [],
             "frames_before": 0, "frames_after": 0,
             "first_audio_t": None, "interrupt_sent_t": None,
             "interrupt_frame_t": None}
    first_audio = asyncio.Event()
    interrupt_sent = asyncio.Event()

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print(f"connected to {WS_URL}")

        async def recv():
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    break
                if isinstance(msg, (bytes, bytearray)):
                    if not first_audio.is_set():
                        state["first_audio_t"] = time.monotonic()
                        first_audio.set()
                    if interrupt_sent.is_set():
                        state["frames_after"] += 1
                    else:
                        state["frames_before"] += 1
                else:
                    print(f"  <-- TEXT FRAME: {msg[:200]}")
                    state["text_frames"].append(msg)
                    # Only count interrupt frames that arrive AFTER we sent
                    # the interruptor: an early one is a spurious boundary
                    # flush (a bug), not a successful barge-in.
                    if "interrupt" in msg and interrupt_sent.is_set():
                        state["interrupt_frame"] = True
                        state["interrupt_frame_t"] = time.monotonic()

        recv_task = asyncio.create_task(recv())

        print("streaming Q1 + 0.8s trailing silence (to end the turn)...")
        await stream(ws, q1)
        await stream(ws, b"\x00\x00" * int(RATE * 0.8))

        print("waiting for bot to start replying...")
        await asyncio.wait_for(first_audio.wait(), timeout=20)
        print("  bot is talking; letting it run 1.2s, then INTERRUPTING")
        await asyncio.sleep(1.2)

        state["interrupt_sent_t"] = time.monotonic()
        interrupt_sent.set()
        await stream(ws, interruptor)
        print("  interruptor sent; observing reaction for 6s")
        await asyncio.sleep(6)
        recv_task.cancel()

    print("\n===== RESULT =====")
    print(f"interrupt text frame received: {state['interrupt_frame']}")
    print(f"audio frames before interrupt: {state['frames_before']}")
    print(f"audio frames after interrupt:  {state['frames_after']}")
    if state["interrupt_frame_t"] and state["interrupt_sent_t"]:
        dt = state["interrupt_frame_t"] - state["interrupt_sent_t"]
        print(f"latency interruptor->interrupt frame: {dt*1000:.0f} ms")
    print(f"all text frames: {state['text_frames']}")
    verdict = "PASS" if state["interrupt_frame"] else "FAIL (no interrupt frame)"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
