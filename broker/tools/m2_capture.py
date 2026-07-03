"""M2 echo-residual capture server.

Stands in for the broker: accepts the puck's WebSocket connection and
records every inbound binary frame (raw PCM16/24k/mono mic audio) to disk,
sending NOTHING back. Because no bot audio is ever sent, the firmware's
mic gate never engages and the mic streams for the whole session.

Run it on the port the puck targets (the live broker must be stopped
first — see M2_RUNBOOK.md), wake the puck, and play the reference clip
through the puck's media_player via m2_play_ref.py. What lands here is
the XMOS ch1 post-AEC residual of that playback — the real signal the
NCC gate would see.

    docker stop voicepe
    uv run --with websockets tools/m2_capture.py --port 8765

Each connection produces m2_captures/capture_<n>.pcm plus a .meta.jsonl
of per-frame arrival times for alignment. Ctrl-C to stop; restart the
live broker afterwards.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import math
import time
from pathlib import Path

import websockets

RATE = 24000


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--out-dir", default="m2_captures")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    # Resume numbering after existing captures — a fresh run must never
    # overwrite takes recorded by a previous one mid-calibration-session.
    counter = max(
        (int(p.stem.split("_")[1]) for p in out.glob("capture_*.pcm")
         if p.stem.split("_")[1].isdigit()),
        default=0,
    )

    async def handle(ws) -> None:
        nonlocal counter
        counter += 1
        pcm_path = out / f"capture_{counter}.pcm"
        meta_path = out / f"capture_{counter}.meta.jsonl"
        print(f"[m2] device connected -> {pcm_path}")
        t0 = time.monotonic()
        nbytes = 0
        last_report = 0.0
        with pcm_path.open("wb") as pcm, meta_path.open("w") as meta:
            try:
                async for msg in ws:
                    t = time.monotonic() - t0
                    if isinstance(msg, bytes):
                        pcm.write(msg)
                        meta.write(json.dumps({"t": round(t, 4), "n": len(msg)}) + "\n")
                        nbytes += len(msg)
                        if t - last_report >= 1.0 and msg:
                            samples = array.array("h", msg)
                            rms = int(math.sqrt(sum(s * s for s in samples) / len(samples)))
                            print(f"[m2]   t={t:5.1f}s  {nbytes/2/RATE:5.1f}s audio  rms={rms}")
                            last_report = t
                    else:
                        print(f"[m2]   text frame: {msg!r}")
            except websockets.ConnectionClosed:
                pass
        print(f"[m2] device disconnected: {nbytes/2/RATE:.1f}s captured in {pcm_path}")

    async with websockets.serve(handle, args.host, args.port, max_size=None):
        print(f"[m2] capture server on ws://{args.host}:{args.port} — wake the puck when ready")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
