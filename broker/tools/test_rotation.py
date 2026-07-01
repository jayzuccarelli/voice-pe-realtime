"""Proves the broker rotates the OpenAI Realtime session SEAMLESSLY mid-call.

OpenAI caps a Realtime session at ~60 min and treats expiry as fatal — the bug
fjfricke/ha-openai-realtime crashes on (#8). Our broker rotates proactively
before the cap and rebuilds the session under a still-connected device. This
test forces that path: start a broker with a tiny MAX_SESSION_SECONDS, hold ONE
device connection open across several rotations, and assert every turn still
answers (and context survives).

Run against a broker started with e.g. MAX_SESSION_SECONDS=12:
    OPENAI_API_KEY=... python broker/tools/test_rotation.py [ws://127.0.0.1:8766] [gap_s]
"""
from __future__ import annotations

import asyncio
import sys

from harness import session

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8766"
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0  # > broker MAX_SESSION_SECONDS


async def main() -> int:
    turns = [
        ("What is the capital of France?", ("paris",)),
        ("Roughly how many people live there?", ("million", "paris", "two", "2")),
        ("In one short sentence, what is two plus two?", ("four", "4")),
    ]
    passed = 0
    async with session(URL) as c:
        for i, (q, expect) in enumerate(turns):
            if i > 0:
                print(f"  ...holding the line {GAP:.0f}s to force a session rotation...")
                await asyncio.sleep(GAP)
            r = await c.ask(q)
            t = r.transcript().lower()
            ok = r.got_audio and any(w in t for w in expect)
            print(f"[{'PASS' if ok else 'FAIL'}] turn {i + 1}: {t!r}")
            passed += ok
    print(f"\n{passed}/{len(turns)} turns survived rotation")
    return 0 if passed == len(turns) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
