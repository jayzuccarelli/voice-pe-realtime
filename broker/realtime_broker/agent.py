"""Builds the OpenAI Realtime LLM service for a session.

Configures voice, system prompt, server-side VAD, and the available tools:
the Home Assistant MCP tools (if HA control is enabled) plus two custom broker
tools, `get_weather` (live HA weather, which HA's MCP doesn't surface) and
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
    InputAudioTranscription,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.processors.aggregators.llm_context import LLMContext

from .config import Config

logger = logging.getLogger(__name__)


class VoicePERealtimeService(OpenAIRealtimeLLMService):
    """Realtime service tuned for a server-VAD voice device.

    Upstream's _handle_context treats the FIRST context frame as conversation
    setup: it replays the context as conversation items and issues a bare
    response.create. With server_vad the audio commit already created the user
    item and auto-created the response, so that double-fires, OpenAI rejects
    it (conversation_already_has_active_response) and Pipecat treats any error
    event as fatal, killing the session's receive loop. Conversation state
    lives server-side here; the only thing context frames must deliver is new
    tool results (_process_completed_function_calls sends them and triggers
    its own response.create).
    """

    async def _handle_context(self, context: LLMContext) -> None:
        self._context = context
        self._llm_needs_conversation_setup = False
        await self._process_completed_function_calls(send_new_results=True)

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
    {
        "type": "function",
        "name": "wait_for_user",
        "description": (
            "Call this ONLY when the most recent audio is clearly NOT addressed "
            "to you: TV or other media dialogue, or a conversation between other "
            "people. If you just answered and the next utterance could be a "
            "follow-up, reaction, or challenge to your answer ('are you sure?', "
            "'okay, and what about...'), it IS addressed to you, answer it "
            "instead of calling this. Calling it means stay silent and keep "
            "listening. Produce no spoken reply when you call it."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# Appended to the configured persona instructions. This device is far-field and
# its mic hears the whole room (TV, other people), so the model must gate on
# whether speech is actually addressed to it: the OpenAI-recommended pattern
# for rejecting non-addressed speech (there is no speaker separation at the API
# layer).
BACKGROUND_GUIDANCE = (
    " IMPORTANT: You are a far-field home assistant; your microphone picks up the "
    "whole room. Only respond to speech clearly addressed to you. If the audio is "
    "a TV or other media, a side conversation between other people, background "
    "chatter, or any speech not directed at you, call the wait_for_user function "
    "and stay silent. Do not narrate that you are waiting. One strong exception: "
    "right after you answer, the next utterance is usually the same user "
    "following up. A follow-up question, reaction, or challenge to what you just "
    "said ('are you sure?', 'okay, and...', 'what about tomorrow?') is addressed "
    "to you even when it does not name you, answer it. When torn between "
    "answering a plausible follow-up and staying silent, answer: a wrongly "
    "ignored user must repeat themselves, which is worse than a wrongly "
    "answered TV line."
)


def build_audio_input(config: Config, threshold: float | None) -> AudioInput:
    """Full audio.input block for session.create AND mid-session updates.

    Always send the complete block: session.update may replace nested objects
    wholesale, so a partial update could silently drop noise reduction or
    transcription.

    threshold=None disables turn detection entirely (serialized as
    turn_detection: null), which also discards the server's VAD state,
    used to drop a speech-in-progress segment after a device disconnect.
    """
    return AudioInput(
        turn_detection=(
            TurnDetection(
                type="server_vad",
                threshold=threshold,
                prefix_padding_ms=config.vad_prefix_padding_ms,
                silence_duration_ms=config.vad_silence_duration_ms,
            )
            if threshold is not None
            else False
        ),
        # No server-side noise reduction: the device streams the XMOS
        # noise-suppressed, no-AGC mic tap (firmware "NS tap, ch1"), which is
        # already clean but quiet. Re-applying OpenAI's far_field reduction on
        # top scrubbed that quiet speech to nothing, so server_vad never fired
        # (device connects, audio flows, no transcript, no reply). Leaving NR
        # off lets the quiet-but-clean tap reach the VAD intact.
        noise_reduction=None,
        # Optionally transcribe each user turn so broker logs show what OpenAI
        # heard (self-trigger / "janky" diagnosis). Off by default: it bills a
        # Whisper pass per turn. whisper-1 because gpt-4o-transcribe yielded
        # zero transcription events on gpt-realtime.
        transcription=(
            InputAudioTranscription(model="whisper-1")
            if config.debug_transcription
            else None
        ),
    )


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
        instructions=config.instructions + BACKGROUND_GUIDANCE,
        audio=AudioConfiguration(
            input=build_audio_input(config, config.vad_threshold),
            output=AudioOutput(voice=config.voice),
        ),
        tools=tools or None,
    )

    service = VoicePERealtimeService(
        api_key=config.openai_api_key,
        model=config.model,
        session_properties=session,
        start_audio_paused=False,
    )

    if mcp is not None and tools_schema is not None:
        await mcp.register_tools_schema(tools_schema, service)
        logger.info("Registered %d Home Assistant tool handlers", len(tools_schema.standard_tools))

    return service
