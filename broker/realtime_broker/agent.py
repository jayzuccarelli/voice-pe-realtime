"""Builds the OpenAI Realtime LLM service for a session.

Configures voice, system prompt, server-side VAD, and — if Home Assistant
control is enabled — registers the HA MCP tools so the model can act on the
home. The audio/session shape here is the GA Realtime API (post-Aug 2025):
voice lives under audio.output, format is implied by the 24 kHz PCM contract.
"""

from __future__ import annotations

import logging

from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from .config import Config

logger = logging.getLogger(__name__)


async def build_agent(config: Config, mcp: MCPClient | None) -> OpenAIRealtimeLLMService:
    """Create a configured OpenAI Realtime service, with HA tools if available."""
    tools: list[dict] = []
    tools_schema = None
    if mcp is not None:
        tools_schema = await mcp.get_tools_schema()
        for fn in tools_schema.standard_tools:
            tools.append(
                {
                    "type": "function",
                    "name": fn.name,
                    "description": fn.description,
                    "parameters": {
                        "type": "object",
                        "properties": fn.properties,
                        "required": fn.required,
                    },
                }
            )
        logger.info("Loaded %d Home Assistant tools", len(tools))

    session = SessionProperties(
        instructions=config.instructions,
        audio=AudioConfiguration(
            input=AudioInput(
                turn_detection=TurnDetection(
                    type="server_vad",
                    threshold=config.vad_threshold,
                    prefix_padding_ms=config.vad_prefix_padding_ms,
                    silence_duration_ms=config.vad_silence_duration_ms,
                )
            ),
            output=AudioOutput(voice=config.voice),
        ),
        tools=tools or None,
    )

    service = OpenAIRealtimeLLMService(
        api_key=config.openai_api_key,
        model=config.model,
        session_properties=session,
        start_audio_paused=False,
    )

    if mcp is not None and tools_schema is not None:
        await mcp.register_tools_schema(tools_schema, service)
        logger.info("Registered %d Home Assistant tool handlers", len(tools))

    return service
