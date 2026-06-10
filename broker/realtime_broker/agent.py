"""Builds the OpenAI Realtime LLM service for a session.

Configures voice, system prompt, server-side VAD, and the available tools:
the Home Assistant MCP tools (if HA control is enabled) plus two custom broker
tools — `get_weather` (live HA weather, which HA's MCP doesn't surface) and
`end_conversation` (clean "ok, bye" stop). Handlers for the custom tools are
registered by the server (they need HA access / the device connection).
"""

from __future__ import annotations

import logging

from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from .config import Config

logger = logging.getLogger(__name__)

# Custom broker tools, registered with handlers by the server.
CUSTOM_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get the current local weather (conditions, temperature, humidity, "
            "wind) from Home Assistant. Call this whenever the user asks about "
            "the weather or outdoor conditions."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "play_music",
        "description": (
            "Play music on a speaker via Music Assistant (Spotify). ALWAYS use "
            "this for any request to play music, a song, artist, album, genre, or "
            "playlist (e.g. 'play some jazz', 'play Miles Davis on the Den'). Do "
            "NOT use the generic media search tool for music. Pass what to play as "
            "`query` and the speaker name (e.g. 'Den') as `speaker`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to play, e.g. 'relaxing jazz'"},
                "speaker": {
                    "type": "string",
                    "description": "Speaker name, e.g. 'Den'. Optional; defaults to the main speaker.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "end_conversation",
        "description": (
            "End the conversation and stop listening. Call this when the user "
            "says goodbye, bye, stop, that's all, thanks that's it, or otherwise "
            "signals they are done."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


async def build_agent(config: Config, mcp: MCPClient | None) -> OpenAIRealtimeLLMService:
    """Create a configured OpenAI Realtime service, with HA + custom tools."""
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

    tools.extend(CUSTOM_TOOLS)

    session = SessionProperties(
        instructions=config.instructions,
        audio=AudioConfiguration(
            input=AudioInput(
                turn_detection=TurnDetection(
                    type="server_vad",
                    threshold=config.vad_threshold,
                    prefix_padding_ms=config.vad_prefix_padding_ms,
                    silence_duration_ms=config.vad_silence_duration_ms,
                ),
                # Far-field mic: filter speaker bleed / room noise BEFORE VAD,
                # so the threshold can stay low enough to hear a normal voice
                # without the bot's own output tripping it (choppiness).
                noise_reduction=InputAudioNoiseReduction(type="far_field"),
                # DEBUG: surface what OpenAI thinks the user said so we can
                # diagnose self-trigger / "janky" behavior from broker logs.
                # whisper-1: gpt-4o-transcribe yielded zero transcription
                # events on gpt-realtime-2.
                transcription=InputAudioTranscription(model="whisper-1"),
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
        logger.info("Registered %d Home Assistant tool handlers", len(tools_schema.standard_tools))

    return service
