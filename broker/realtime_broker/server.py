"""WebSocket audio broker: bridges a Voice PE device to OpenAI Realtime.

One long-lived process. A device connects over `ws://`, streams PCM up, and
plays PCM back; in between sits a Pipecat pipeline whose brain is an OpenAI
Realtime session (optionally able to control Home Assistant via MCP tools).

The OpenAI session is persistent for the process lifetime, so conversation
context carries across the device's per-turn reconnects for free. (OpenAI
caps a session at 60 min; transparent reconnect-on-expiry is a known TODO.)
"""

from __future__ import annotations

import logging

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from . import mcp_client
from .agent import build_agent
from .config import Config
from .serializer import RawPCMSerializer

logger = logging.getLogger(__name__)


async def run(config: Config) -> None:
    """Build the pipeline and serve until cancelled."""
    mcp = None
    if config.ha_control_enabled:
        mcp = await mcp_client.connect(config.ha_mcp_url, config.ha_token)
    else:
        logger.info("Home Assistant control disabled (HA_MCP_URL/HA_TOKEN unset)")

    service = await build_agent(config, mcp)

    transport = WebsocketServerTransport(
        host=config.ws_host,
        port=config.ws_port,
        params=WebsocketServerParams(
            serializer=RawPCMSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    aggregator = LLMContextAggregatorPair(LLMContext())
    pipeline = Pipeline(
        [
            transport.input(),
            aggregator.user(),
            service,
            aggregator.assistant(),
            transport.output(),
        ]
    )

    @transport.event_handler("on_client_connected")
    async def _on_connect(_transport, client):  # noqa: ANN001
        logger.info("Device connected: %s", getattr(client, "remote_address", client))

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, client, *args):  # noqa: ANN001
        logger.info("Device disconnected")

    task = PipelineTask(pipeline, idle_timeout_secs=None, cancel_on_idle_timeout=False)
    logger.info("Broker listening on ws://%s:%d", config.ws_host, config.ws_port)
    await PipelineRunner().run(task)
