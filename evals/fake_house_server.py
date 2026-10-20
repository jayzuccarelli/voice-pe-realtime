"""The fake house, served as a Home Assistant MCP endpoint for the broker under test.

The broker reaches Home Assistant only through HA_MCP_URL, so pointing that
here puts the broker in the same fake house raw GPT-Live is scored in, with
the same 24 tool schemas, and nothing reaches the real one. Beside the MCP
SSE endpoint it has a small control API the harness uses around each
scenario:

    POST /reset       {"initial_state": {...}}  start a scenario
    GET  /snapshot    final state, same shape as House.snapshot()
    GET  /executions  every tool call since the last reset

    broker/.venv-live/bin/python evals/fake_house_server.py   # :8791
    HA_MCP_URL=http://127.0.0.1:8791/mcp_server/sse HA_TOKEN=fake  (for the broker)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

sys.path.insert(0, str(Path(__file__).resolve().parent / "gpt_live_evals"))
from assistants.smart_home.house import TOOLS_FILE, House  # noqa: E402

PORT = 8791
TOOLS = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
house = House()
server = Server("fake-home-assistant")
sse = SseServerTransport("/mcp_server/messages/")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in TOOLS]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    result = house.execute(name, arguments or {}, call_id=f"mcp_{len(house.executions) + 1}")
    return [types.TextContent(type="text", text=json.dumps(result))]


async def handle_sse(request: Request) -> Response:
    async with sse.connect_sse(request.scope, request.receive, request._send) as (read, write):
        await server.run(read, write, server.create_initialization_options())
    return Response()


async def reset(request: Request) -> JSONResponse:
    global house
    body = await request.json()
    house = House(initial_state=body.get("initial_state") or {})
    return JSONResponse({"ok": True})


async def snapshot(request: Request) -> JSONResponse:
    return JSONResponse(house.snapshot())


async def executions(request: Request) -> JSONResponse:
    return JSONResponse(house.executions)


app = Starlette(routes=[
    Route("/mcp_server/sse", handle_sse),
    Mount("/mcp_server/messages/", app=sse.handle_post_message),
    Route("/reset", reset, methods=["POST"]),
    Route("/snapshot", snapshot),
    Route("/executions", executions),
])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
