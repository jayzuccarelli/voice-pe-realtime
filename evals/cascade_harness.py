"""Replay recorded puck tapes through minimal/relay.py against the fake house.

Every tape in TAPES goes frame by frame through the relay's own cutter,
transcriber, guard, model and tool calls, with the house swapped for
evals/fake_house_server.py and the voice replaced by a log line. Scored
against evals/cascade_labels.json: did the expected action run, did an
unexpected one run, did it speak at all, did it claim an action no tool did.

    broker/.venv-live/bin/python evals/fake_house_server.py &        # :8791
    HA_MCP_URL=http://127.0.0.1:8791/mcp_server/sse HA_TOKEN=fake \
        broker/.venv-live/bin/python evals/cascade_harness.py [tape ...]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import wave
from pathlib import Path

os.environ.setdefault("HA_MCP_URL", "http://127.0.0.1:8791/mcp_server/sse")
os.environ.setdefault("HA_TOKEN", "fake")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minimal"))

from dotenv import load_dotenv  # noqa: E402

if os.environ.get("ENV_FILE"):  # the key, from a checkout that has broker/.env; never the real house
    load_dotenv(os.environ["ENV_FILE"])

import aiohttp  # noqa: E402
import relay  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.sse import sse_client  # noqa: E402

HOUSE = os.environ["HA_MCP_URL"].split("/mcp_server")[0]
TAPES = Path(os.environ.get("TAPES", "/tmp/claude/puck-tapes"))
LABELS = json.loads((Path(__file__).with_name("cascade_labels.json")).read_text())
FRAME = relay.RATE // 50 * 2  # 20 ms, the puck's real chunk (see evals/cutter_replay.py)
CLAIMS = re.compile(r"\b(is|are) (now )?(off|on|locked|unlocked)\b|\bturned\b|\bturning\b", re.I)


class Puck:
    closed = False

    async def send_bytes(self, pcm: bytes) -> None:
        pass


async def replay(path: Path, http: aiohttp.ClientSession) -> dict:
    said: list[str] = []
    heard: list[str] = []
    rejected: list[str] = []
    t0 = time.monotonic()
    async with http.post(f"{HOUSE}/reset", json={}) as r:
        r.raise_for_status()
    async with sse_client(os.environ["HA_MCP_URL"]) as (rd, wr), ClientSession(rd, wr) as ha:
        await ha.initialize()
        tools = [relay.chat_tool(t) for t in (await ha.list_tools()).tools] + [relay.END_TOOL]
        conv = relay.Conversation(Puck(), ha, http, tools)

        async def tts(text: str) -> bytes:
            return bytes(relay.RATE // 10 * 2)

        async def say(text: str) -> None:
            if text:
                said.append(text)

        conv.tts = tts
        conv.say = say
        with wave.open(str(path)) as w:
            pcm = w.readframes(w.getnframes())
        for i in range(0, len(pcm), FRAME):
            frame = pcm[i:i + FRAME]
            conv.track_utterance(frame, relay.rms(frame))
            if conv.turns:
                await asyncio.sleep(0)  # let a finished turn log in order
        if conv.utterance is not None and conv.voiced_s >= relay.MIN_SPEECH_S:
            conv.on_utterance(bytes(conv.utterance))
        await asyncio.gather(*conv.turns, return_exceptions=True)
        for user, _ in zip(conv.history[::2], conv.history[1::2]):
            heard.append(user["content"])
        if relay._ack_pcm is None:
            relay._ack_pcm = b""
    async with http.get(f"{HOUSE}/executions") as r:
        executions = await r.json()
    acted = [(e["name"], json.dumps(e["arguments"]).lower()) for e in executions
             if relay.ACTION_TOOLS.match(e["name"])]
    return {"tape": path.name, "secs": round(time.monotonic() - t0, 1), "heard": heard,
            "rejected": rejected, "said": said, "acted": acted}


def score(label: dict, out: dict) -> tuple[str, str]:
    expect = label.get("expect")
    acted = out["acted"]
    fabricated = any(CLAIMS.search(s) for s in out["said"]) and not acted
    if any(s.startswith("Sorry, something went wrong") for s in out["said"]):
        return "FAIL", "turn failed: " + " | ".join(out["said"])[:70]
    if expect == "none":
        if acted:
            return "FAIL", "acted on nothing: " + "; ".join(f"{n}{a}" for n, a in acted)
        if fabricated:
            return "FAIL", "claimed an action: " + " | ".join(out["said"])
        return "PASS", "quiet" if not out["said"] else "spoke: " + " | ".join(out["said"])[:60]
    if expect == "answer":
        if acted:
            return "FAIL", "acted on a question: " + "; ".join(f"{n}{a}" for n, a in acted)
        return ("PASS", " | ".join(out["said"])[:70]) if out["said"] else ("FAIL", "no answer")
    missing = [f"{tool} {must}" for tool, must in expect
               if not any(re.fullmatch(tool, n) and re.search(must, a) for n, a in acted)]
    extra = [f"{n}{a}" for n, a in acted
             if not any(re.fullmatch(tool, n) and re.search(must, a) for tool, must in expect)]
    if missing:
        return "FAIL", "missing " + ", ".join(missing) + (" | said: " + " | ".join(out["said"])[:50] if out["said"] else " | silent")
    if fabricated:
        return "FAIL", "fabricated: " + " | ".join(out["said"])
    if extra:
        return "FAIL", "unasked action: " + "; ".join(extra)[:90]
    return "PASS", "; ".join(f"{n}{a}" for n, a in acted)[:70] + " | " + " | ".join(out["said"])[:40]


async def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    relay.log.setLevel(logging.INFO)
    tapes = [Path(a) for a in argv] or sorted(p for p in TAPES.glob("*.wav") if p.name in LABELS)
    tally: dict[str, int] = {}
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), trust_env=True) as http:
        for path in tapes:
            label = LABELS.get(path.name, {"expect": "none"})
            relay.log.info("=== %s  (%s)", path.name, label.get("said") or label.get("note", ""))
            try:
                out = await replay(path, http)
            except Exception as exc:  # noqa: BLE001 - one broken tape must not end the run
                out = {"tape": path.name, "secs": 0, "heard": [], "rejected": [], "said": [], "acted": [], "error": repr(exc)}
            verdict, why = score(label, out) if "error" not in out else ("ERROR", out["error"])
            kind = "command" if isinstance(label["expect"], list) else label["expect"]
            tally[f"{kind} {verdict}"] = tally.get(f"{kind} {verdict}", 0) + 1
            rows.append(f"{verdict:5} {kind:7} {path.name[5:-4]}  {why}")
            relay.log.info("--- %s: %s", verdict, why)
    print()
    print("\n".join(rows))
    print()
    print(json.dumps(tally, indent=1))
    Path(os.environ.get("HARNESS_OUT", "/tmp/claude/cascade_harness.txt")).write_text("\n".join(rows) + "\n" + json.dumps(tally) + "\n")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
