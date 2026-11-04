"""A stand-in puck: stream one tape to a relay over its websocket at real time and report what came back.

Mimics the firmware's two habits that matter to the relay: 20 ms chunks,
and a microphone that sends nothing while audio is playing or for 500 ms
after. Prints when audio came back, how far ahead of real time it ran
(the puck's playback queue holds ten chunks), and whether the relay hung
up on its own.

    broker/.venv-live/bin/python evals/puck_client.py ws://127.0.0.1:8768/ tape.wav
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave

import aiohttp

RATE = 24000
FRAME = RATE // 50 * 2  # 20 ms
MIC_CLOSED_S = 0.5  # the firmware's is_bot_speaking() window


async def main(url: str, path: str) -> None:
    with wave.open(path) as w:
        pcm = w.readframes(w.getnframes())
    got: list[tuple[float, int]] = []
    disconnect_at = None
    last_rx = -10.0
    async with aiohttp.ClientSession() as session, session.ws_connect(url, max_msg_size=0) as ws:
        t0 = time.monotonic()

        async def receive() -> None:
            nonlocal disconnect_at, last_rx
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    got.append((time.monotonic() - t0, len(msg.data)))
                    last_rx = time.monotonic()
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"{time.monotonic() - t0:5.1f}s text frame {msg.data}")
                    if "disconnect" in msg.data:
                        disconnect_at = time.monotonic() - t0
                        return
                else:
                    return

        receiver = asyncio.create_task(receive())
        dropped = 0
        for i in range(len(pcm) // FRAME):
            if receiver.done():
                break
            await asyncio.sleep(max(0.0, t0 + i * 0.02 - time.monotonic()))
            if time.monotonic() - last_rx < MIC_CLOSED_S:
                dropped += 1
                continue
            await ws.send_bytes(pcm[i * FRAME:(i + 1) * FRAME])
        print(f"{time.monotonic() - t0:5.1f}s tape sent ({len(pcm) / 2 / RATE:.1f}s, {dropped * 0.02:.1f}s dropped while audio played)")
        try:
            await asyncio.wait_for(receiver, timeout=40)
        except asyncio.TimeoutError:
            print("no disconnect within 40s after the tape; the relay did not hang up")
    # Bursts of audio: start time, length, and how far ahead of real time the chunks ran.
    bursts: list[list[tuple[float, int]]] = []
    for t, n in got:
        if bursts and t - bursts[-1][-1][0] < 0.5:
            bursts[-1].append((t, n))
        else:
            bursts.append([(t, n)])
    for b in bursts:
        start, total = b[0][0], sum(n for _, n in b) / 2 / RATE
        leads = [(sum(n for _, n in b[:k]) / 2 / RATE) - (t - start) for k, (t, _) in enumerate(b)]
        gaps = [t - b[k - 1][0] for k, (t, _) in enumerate(b) if k and t - b[k - 1][0] > 0.15]
        print(f"audio burst at {start:5.1f}s: {total:.1f}s of audio, lead {min(leads):+.2f}s..{max(leads):+.2f}s, "
              f"{len(gaps)} inter-chunk gaps >150ms{(' (max %.2fs)' % max(gaps)) if gaps else ''}")
    print(f"total {sum(n for _, n in got) / 2 / RATE:.1f}s of audio in {len(got)} chunks; "
          f"disconnect at {disconnect_at if disconnect_at is None else round(disconnect_at, 1)}s")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
