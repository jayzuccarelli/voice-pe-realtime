"""Replay tapes through the relay's cutter alone, at a given chunk size, and print the cuts.

Used once to find the puck's real chunk size by matching production's
"utterance: Xs of speech at mic Ys" lines. No network, no model.

    broker/.venv-live/bin/python evals/cutter_replay.py <chunk_ms> tape.wav ...
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "x")
os.environ.setdefault("HA_MCP_URL", "http://127.0.0.1:1/mcp_server/sse")
os.environ.setdefault("HA_TOKEN", "x")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minimal"))

import relay  # noqa: E402


def main() -> None:
    chunk_ms = int(sys.argv[1])
    frame = relay.RATE * chunk_ms // 1000 * 2
    for path in sys.argv[2:]:
        conv = relay.Conversation(None, None, None, [])
        cuts = []
        conv.on_utterance = lambda pcm: None
        orig = relay.log.info
        relay.log.info = lambda fmt, *a: cuts.append(fmt % a) if fmt.startswith("utterance") else None
        with wave.open(path) as w:
            pcm = w.readframes(w.getnframes())
        for i in range(0, len(pcm), frame):
            f = pcm[i:i + frame]
            conv.track_utterance(f, relay.rms(f))
        relay.log.info = orig
        print(f"{chunk_ms}ms {Path(path).name[5:-4]}: " + "; ".join(c.replace("utterance: ", "").replace(" of speech at mic ", "@") for c in cuts))


if __name__ == "__main__":
    main()
