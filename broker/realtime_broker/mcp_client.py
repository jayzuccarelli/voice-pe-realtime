"""Home Assistant control via the Model Context Protocol.

HA's built-in MCP Server integration exposes its tools over SSE at
`/mcp_server/sse` (the streamable-HTTP path returns 404), so we use Pipecat's
SSE transport. Tools are fetched once at startup and registered on the
OpenAI Realtime service so the model can call them (turn on lights, etc.).
"""

from __future__ import annotations

import logging

from pipecat.services.mcp_service import MCPClient, SseServerParameters

logger = logging.getLogger(__name__)


class ReconnectingMCPClient(MCPClient):
    """An MCP client that notices when its session has quietly died.

    The SSE session to Home Assistant can go away while the broker stays
    up and the model keeps its 24 tools. Pipecat logs the failure, returns
    an empty result and carries on, so every tool call fails in
    milliseconds without ever reaching Home Assistant, and the model
    apologises for something it never attempted: "I'm sorry, I couldn't
    turn those lights off just now", for two days, after hearing the
    request perfectly (2026-09-24). Nothing reconnected, because from the
    broker's side nothing had crashed.

    An empty result is that failure's only signature here, so it is worth
    one reconnect and one retry. A tool that genuinely had nothing to say
    costs a reconnect it did not need, which is a few hundred
    milliseconds, once.
    """

    async def _call_tool_text(self, session, function_name, arguments) -> str:
        text = await super()._call_tool_text(session, function_name, arguments)
        if text:
            return text
        logger.warning(
            "MCP tool %s came back empty; reconnecting to Home Assistant and retrying",
            function_name,
        )
        try:
            await self.close()
        except Exception as exc:  # noqa: BLE001 - already torn down is fine
            logger.debug("MCP close during reconnect: %s", exc)
        try:
            await self.start()
        except Exception:
            logger.exception("Could not reconnect to Home Assistant")
            return text
        return await super()._call_tool_text(self._ensure_connected(), function_name, arguments)


async def connect(url: str, token: str) -> MCPClient:
    """Build an MCP client pointed at HA's SSE endpoint."""
    logger.info("Connecting to Home Assistant MCP at %s", url)
    return ReconnectingMCPClient(
        server_params=SseServerParameters(
            url=url,
            headers={"Authorization": f"Bearer {token}"},
        )
    )
