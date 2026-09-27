"""Voice PE puck <-> GPT-Live with Home Assistant tools, and nothing else.

The whole assistant is gpt-live-1: it hears, decides, delegates to its
backend, and speaks. This file only does what the puck cannot do itself:

- carry the puck's audio to Live and Live's audio back (PCM16, 24 kHz);
- run the backend's tool calls against Home Assistant over MCP and hand the
  results back to Live;
- end the conversation, because the puck never hangs up on its own: after
  FOLLOWUP_S of quiet once the assistant has spoken, and at HARD_CAP_S
  whatever happens, so a stuck session cannot keep billing.

No transcript checks, no output holds, no retries, no memory: the eval
(evals/gate.py) showed each of those made the broker slower and less
accurate than Live on its own.

    broker/.venv-live/bin/python minimal/relay.py     # listens on :8767
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv(Path(__file__).resolve().parent.parent / "broker" / ".env")

PORT = int(os.environ.get("MINIMAL_PORT", "8767"))
LIVE_URL = os.environ.get("LIVE_URL", "wss://api.openai.com/v1/live/sessions")
LIVE_MODEL = os.environ.get("LIVE_MODEL", "gpt-live-1")
BACKEND_MODEL = os.environ.get("LIVE_BACKEND_MODEL", "gpt-5.4-mini")
VOICE = os.environ.get("LIVE_VOICE", "cedar")
API_KEY = os.environ["OPENAI_API_KEY"]
HA_MCP_URL = os.environ["HA_MCP_URL"]
HA_TOKEN = os.environ["HA_TOKEN"]
RATE = 24000
FOLLOWUP_S = 6.0
FIRST_REPLY_S = 15.0
HARD_CAP_S = 180.0

INSTRUCTIONS = os.environ.get(
    "INSTRUCTIONS",
    "You are Atriensis, the household steward for this smart home. Always respond in English. "
    "Be concise, warm, and natural, like a capable, unflappable butler. You can control the home "
    "with the available tools; when asked to do something, just do it and confirm in one short sentence.",
)
BACKEND_INSTRUCTIONS = (
    "You are the backend of a home voice assistant. Each message is the recent voice conversation "
    "as a transcript; work out what is being asked and do it. Use the Home Assistant tools to "
    "control the home and read live state. Reply with the verified result in one short "
    "conversational sentence the assistant can say aloud, with no Markdown and no JSON, and never "
    "claim an action completed without a tool result confirming it."
)

log = logging.getLogger("minimal")


def live_tool(tool) -> dict:
    """An MCP tool as a Live backend function: optional fields become nullable, all listed."""
    params = copy.deepcopy(tool.inputSchema or {"type": "object", "properties": {}})
    props = params.setdefault("properties", {})
    required = set(params.get("required", []))
    for name, schema in list(props.items()):
        if name not in required:
            props[name] = {"anyOf": [schema, {"type": "null"}]}
    params["required"] = list(props)
    params["additionalProperties"] = False
    return {"type": "function", "name": tool.name, "description": tool.description or tool.name,
            "parameters": params}


def session_start(tools: list[dict]) -> dict:
    return {
        "type": "session.start",
        "session": {
            "model": LIVE_MODEL,
            "instructions": INSTRUCTIONS,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": VOICE}},
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": BACKEND_MODEL,
                    "instructions": BACKEND_INSTRUCTIONS,
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "reasoning": {"effort": "none"},
                    "text": {"verbosity": "low"},
                },
            },
        },
    }


class Conversation:
    """One wake: from the puck connecting to the relay closing it."""

    def __init__(self, puck: web.WebSocketResponse, live, ha: ClientSession) -> None:
        self.puck, self.live, self.ha = puck, live, ha
        self.started = time.monotonic()
        self.heard_until = self.started  # when the puck finishes playing what it has been sent
        self.spoke = False
        self.pending: dict[str, list[asyncio.Task]] = {}
        self.busy = 0  # tool calls in flight

    async def run(self) -> None:
        tasks = [asyncio.create_task(c) for c in (self.from_puck(), self.from_live(), self.clock())]
        _, rest = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in rest:
            t.cancel()
        await asyncio.gather(*rest, return_exceptions=True)

    async def from_puck(self) -> None:
        async for msg in self.puck:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await self.live.send_json({"type": "session.input_audio.append",
                                           "audio": base64.b64encode(msg.data).decode()})
            elif msg.type == aiohttp.WSMsgType.TEXT:
                log.info("puck says %s", msg.data)
        log.info("puck hung up")

    async def from_live(self) -> None:
        async for msg in self.live:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            kind = event.get("type")
            if kind != "session.output_audio.delta":
                inner = (event.get("event") or {}).get("type", "")
                log.debug("live event %s %s", kind, inner)
            if kind == "session.output_audio.delta":
                pcm = base64.b64decode(event["delta"])
                now = time.monotonic()
                self.heard_until = max(self.heard_until, now) + len(pcm) / 2 / RATE
                self.spoke = True
                await self.puck.send_bytes(pcm)
            elif kind == "session.input_transcript.delta":
                self.heard_until = max(self.heard_until, time.monotonic())
                log.info("heard: %s", event.get("delta", "").strip())
            elif kind == "session.output_transcript.delta":
                log.info("said: %s", event.get("delta", "").strip())
            elif kind == "response.event":
                await self.on_backend(event.get("event") or {})
            elif kind in ("error", "session.closed"):
                log.info("live %s: %s", kind, json.dumps(event)[:300])
                if kind == "session.closed":
                    return

    async def on_backend(self, inner: dict) -> None:
        kind = inner.get("type")
        item = inner.get("item") or {}
        if kind == "response.output_item.done" and item.get("type") == "function_call" \
                and item.get("status") == "completed":
            rid = inner.get("response_id") or ""
            self.pending.setdefault(rid, []).append(asyncio.create_task(self.call(item)))
        elif kind == "response.completed":
            rid = (inner.get("response") or {}).get("id", "")
            calls = self.pending.pop(rid, None) or self.pending.pop("", None)
            if not calls:
                return
            for call_id, output in await asyncio.gather(*calls):
                await self.live.send_json({
                    "type": "response.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id,
                             "output": json.dumps(output, ensure_ascii=False)},
                })
            await self.live.send_json({"type": "response.create"})

    async def call(self, item: dict) -> tuple[str, dict]:
        name = item.get("name", "")
        args = {k: v for k, v in json.loads(item.get("arguments") or "{}").items() if v is not None}
        self.busy += 1
        try:
            result = await self.ha.call_tool(name, args)
            text = "".join(getattr(c, "text", "") for c in result.content)
            output = {"error": text} if result.isError else {"result": text}
        except Exception as exc:  # noqa: BLE001 - a failed call is reported to the model, not raised
            output = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.busy -= 1
            self.heard_until = max(self.heard_until, time.monotonic())
        log.info("tool %s(%s) -> %s", name, json.dumps(args), json.dumps(output)[:200])
        return item.get("call_id", ""), output

    async def clock(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if now - self.started > HARD_CAP_S:
                log.info("hard cap reached")
                return
            quiet = now - self.heard_until
            if not self.busy and not self.pending and quiet > (FOLLOWUP_S if self.spoke else FIRST_REPLY_S):
                log.info("quiet for %.0fs, ending the conversation", quiet)
                return


async def handle(request: web.Request) -> web.WebSocketResponse:
    puck = web.WebSocketResponse(max_msg_size=0)
    await puck.prepare(request)
    log.info("wake from %s", request.remote)
    try:
        async with sse_client(HA_MCP_URL, headers={"Authorization": f"Bearer {HA_TOKEN}"}) as (r, w), \
                ClientSession(r, w) as ha:
            await ha.initialize()
            tools = [live_tool(t) for t in (await ha.list_tools()).tools]
            async with aiohttp.ClientSession() as http, http.ws_connect(
                LIVE_URL, headers={"Authorization": f"Bearer {API_KEY}"}, heartbeat=20, max_msg_size=0
            ) as live:
                await live.send_json(session_start(tools))
                first = await live.receive_json()
                if first.get("type") != "session.started":
                    log.error("live refused the session: %s", json.dumps(first)[:400])
                    return puck
                await Conversation(puck, live, ha).run()
                await live.send_json({"type": "session.close"})
    except Exception:
        log.exception("conversation failed")
    finally:
        await puck.close()
        log.info("conversation over")
    return puck


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.setLevel(os.environ.get("MINIMAL_LOG", "INFO"))
    app = web.Application()
    app.router.add_get("/{tail:.*}", handle)
    web.run_app(app, host="0.0.0.0", port=PORT, print=lambda *_: log.info("listening on :%d", PORT))


if __name__ == "__main__":
    main()
