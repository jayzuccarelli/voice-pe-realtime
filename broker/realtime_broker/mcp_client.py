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


async def connect(url: str, token: str) -> MCPClient:
    """Build an MCP client pointed at HA's SSE endpoint."""
    logger.info("Connecting to Home Assistant MCP at %s", url)
    return MCPClient(
        server_params=SseServerParameters(
            url=url,
            headers={"Authorization": f"Bearer {token}"},
        )
    )
