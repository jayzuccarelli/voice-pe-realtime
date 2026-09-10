"""Builds the OpenAI Live (gpt-live-1) service for a session.

The live model is the *frontend*: it owns the spoken conversation, listens
while it speaks, and decides on its own when to stop talking. Anything that
needs a tool or real reasoning it *delegates* to a backend model. We use
Responses delegation, so OpenAI hosts the backend and the function calls it
makes come back here to run against Home Assistant over MCP: one API key,
one vendor, and the existing tool handlers work unchanged.

Two behaviours the Realtime engine needed and this one does not:

- No server VAD. The live model detects turns itself; there is no threshold
  to raise while the bot speaks, no adaptive-VAD dance, no VAD state to
  reset between wakes.
- No `wait_for_user`. That tool existed to suppress a reply the Realtime
  turn model had already committed to. The live model chooses whether to
  speak at all, so the far-field guidance below is instruction-only.
"""

from __future__ import annotations

import logging

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMSettings

from .config import Config

logger = logging.getLogger(__name__)


class VoicePELiveService(OpenAILiveLLMService):
    """Live service whose billed session is bound to a device connection.

    Upstream opens its websocket in `setup()` and starts the session on the
    first context frame, then keeps both for the process's lifetime. That
    suits a call; it does not suit a wall puck that wakes for 20 seconds at
    a time, because the live model bills per minute of session, not per
    minute of speech. So the transport is left connected (free) while the
    *session* is started and closed around each wake.

    It also counts the delegated backend responses currently in flight, so
    the broker can tell "the model has finished answering" from "the model
    paused mid-answer while the backend thinks". Nothing else can: the live
    model streams continuous audio and speaks in bursts, so a gap in speech
    looks identical either way.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._live_delegations_in_flight = 0

    @property
    def delegation_in_flight(self) -> bool:
        """Whether a delegated backend response is still being worked on."""
        return self._live_delegations_in_flight > 0

    async def _handle_evt_response(self, evt) -> None:
        """Count delegated responses as they open and close.

        Tracked here rather than read off upstream's `_pending_responses`,
        which only drops an entry for a response that made function calls: a
        delegated response that just answers is never removed, so its length
        reads as "forever in flight" after the first delegation.
        """
        inner = getattr(evt, "inner_type", None)
        if inner == "response.created":
            self._live_delegations_in_flight += 1
        elif inner in ("response.completed", "response.incomplete", "response.failed"):
            # Floor at zero: a reconnect can deliver a completion whose
            # matching creation belonged to a session that is already gone.
            self._live_delegations_in_flight = max(0, self._live_delegations_in_flight - 1)
        await super()._handle_evt_response(evt)

    async def begin_live_session(self) -> None:
        """Open a billed session for a freshly connected device."""
        if self._session_started:
            return
        self._live_delegations_in_flight = 0
        # _connect() early-returns on a non-None socket even when it is dead,
        # so always tear the old one down first.
        await self._disconnect()
        await self._connect()
        self._needs_session_config = True
        if self._context is not None:
            await self._send_session_config()

    async def end_live_session(self) -> None:
        """Close the billed session when the device goes away.

        Every step is attempted even if an earlier one raises. The steps are
        ordered nice-to-have first and load-bearing last: failing to close
        out turns costs a transcript, failing to disconnect leaves a billed
        session open, so an exception in the first must not skip the last.
        Nothing upstream guards this: the framework's own cleanup only runs
        when the whole worker shuts down, and by then the meter has been
        running for however long the puck has been idle.
        """
        try:
            try:
                await self._close_open_turns()
            finally:
                await self._close_session()
        finally:
            try:
                await self._disconnect()
            finally:
                self._needs_session_config = True
                self._live_delegations_in_flight = 0


# Appended to the configured persona instructions. The device is far-field
# and hears the whole room, so the model is told what not to answer. Unlike
# the Realtime engine there is no tool to call for this: a full-duplex model
# that decides to stay quiet simply does not speak.
BACKGROUND_GUIDANCE = (
    " IMPORTANT: You are a far-field home assistant; your microphone picks up the "
    "whole room. Only respond to speech clearly addressed to you. If the audio is "
    "a TV or other media, a side conversation between other people, or background "
    "chatter, stay silent and keep listening. Do not narrate that you are waiting. "
    "One strong exception: right after you answer, the next utterance is usually "
    "the same user following up. A follow-up question, reaction, or challenge to "
    "what you just said ('are you sure?', 'okay, and...', 'what about tomorrow?') "
    "is addressed to you even when it does not name you. Answer it. When torn "
    "between answering a plausible follow-up and staying silent, answer: a wrongly "
    "ignored user must repeat themselves, which is worse than a wrongly answered "
    "TV line."
)

# Told to the frontend model only. Task knowledge lives in the backend
# prompt; this is about conversation and when to hand off.
DELEGATION_GUIDANCE = (
    " You have a backend that holds the smart-home tools and does the careful "
    "thinking. Answer directly, without delegating, anything you already know: "
    "chit-chat, questions about this conversation, and ordinary general "
    "knowledge. Delegate only what you cannot answer from your own knowledge: "
    "controlling the home (lights, music, TV, scenes), reading live state "
    "(weather, whether something is on, what is playing), and genuine lookups. "
    "Hand off as soon as you know the request is for the backend, keep the "
    "conversation going while it works, and relay the result when it lands. "
    "Ignore results the conversation has already moved past. Never make the "
    "user wait in silence: if a hand-off is taking a moment, say so briefly."
)

WEATHER_TOOL = FunctionSchema(
    name="get_weather",
    description=(
        "Get the current local weather (conditions, temperature, humidity, wind) "
        "from Home Assistant. Call this whenever the user asks about the weather "
        "or outdoor conditions."
    ),
    properties={},
    required=[],
)

MUSIC_TOOL = FunctionSchema(
    name="play_music",
    description=(
        "Play music on a speaker via Music Assistant (Spotify). ALWAYS use this "
        "for any request to play music, a song, artist, album, genre, or playlist "
        "(e.g. 'play some jazz', 'play Miles Davis on the Den'). Do NOT use the "
        "generic media search tool for music. Pass what to play as `query` and the "
        "speaker name (e.g. 'Den') as `speaker`."
    ),
    properties={
        "query": {"type": "string", "description": "What to play, e.g. 'relaxing jazz'"},
        "speaker": {
            "type": "string",
            "description": "Speaker name, e.g. 'Den'. Optional; defaults to the main speaker.",
        },
    },
    required=["query"],
)

END_TOOL = FunctionSchema(
    name="end_conversation",
    description=(
        "End the conversation and stop listening. Call this when the user says "
        "goodbye, bye, stop, that's all, thanks that's it, or otherwise signals "
        "they are done."
    ),
    properties={},
    required=[],
)

CUSTOM_TOOLS = [WEATHER_TOOL, MUSIC_TOOL, END_TOOL]


def _scrub_floats(node, path: str) -> None:
    """Strip float literals out of a JSON-Schema fragment, in place.

    The Live API rejects `session.start` outright if any delegated tool
    schema contains a float: it answers `Invalid AVAS session_data: Type is
    not JSON serializable: decimal.Decimal`, which is the server parsing our
    JSON numbers as Decimal and then failing to re-encode them. Minimal
    repro: one tool with `{"type": "number", "minimum": 0.5}` fails, the same
    tool with `minimum: 1` starts fine (verified against gpt-live-1,
    2026-09-10). It bites us because Home Assistant's MCP server reports
    numeric bounds as floats, so `tv_remote` alone (channel 1.0-9999.0,
    repeats 1.0-20.0) took the whole session down.

    A whole float is rewritten as the same integer, which is lossless.
    Anything genuinely fractional is dropped: losing one bound on a tool
    argument costs far less than a session that cannot start, and the
    property keeps its type.
    """
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, float) and not isinstance(value, bool):
                if value.is_integer():
                    node[key] = int(value)
                else:
                    logger.warning(
                        "Dropping fractional %s=%s at %s (Live API rejects float "
                        "literals in tool schemas)",
                        key,
                        value,
                        path,
                    )
                    del node[key]
            else:
                _scrub_floats(value, f"{path}.{key}")
    elif isinstance(node, list):
        # Rebuilt rather than edited in place: a fractional float inside a
        # list (an `enum` of allowed values, say) has to come out too, and
        # deleting while enumerating skips elements. Losing one enum value
        # narrows what the model may pass; leaving it in means no session at
        # all, so out it goes.
        kept = []
        for i, value in enumerate(node):
            if isinstance(value, float) and not isinstance(value, bool):
                if value.is_integer():
                    kept.append(int(value))
                else:
                    logger.warning(
                        "Dropping fractional list value %s at %s[%d] (Live API "
                        "rejects float literals in tool schemas)",
                        value,
                        path,
                        i,
                    )
                continue
            _scrub_floats(value, f"{path}[{i}]")
            kept.append(value)
        node[:] = kept


async def build_live_tools(mcp: MCPClient | None) -> ToolsSchema:
    """Home Assistant's MCP tools plus the three broker-local ones.

    The tools go in the pipeline's LLMContext, which is where the Live
    adapter reads them from to configure the delegated backend model.
    """
    standard: list = []
    if mcp is not None:
        ha = await mcp.get_tools_schema()
        standard.extend(ha.standard_tools)
        logger.info("Loaded %d Home Assistant tools", len(ha.standard_tools))
    standard.extend(CUSTOM_TOOLS)
    # In place, so the handler registration keyed on these same objects and
    # the schema the adapter serializes cannot drift apart.
    for tool in standard:
        properties = getattr(tool, "_properties", None)
        if isinstance(properties, dict):
            _scrub_floats(properties, f"tool:{getattr(tool, 'name', '?')}")
    return ToolsSchema(standard_tools=standard)


def build_live_agent(config: Config) -> VoicePELiveService:
    """Create the Live service with an OpenAI-hosted backend model."""
    return VoicePELiveService(
        api_key=config.openai_api_key,
        settings=VoicePELiveService.Settings(
            model=config.live_model,
            voice=config.voice,
            system_instruction=config.instructions + BACKGROUND_GUIDANCE + DELEGATION_GUIDANCE,
        ),
        delegation=VoicePELiveService.ResponsesDelegation(
            settings=OpenAIResponsesLLMSettings(
                model=config.live_backend_model,
                system_instruction=(
                    "You are the backend of a home voice assistant. Each message is "
                    "the recent voice conversation as a transcript; work out what is "
                    "being asked and do it. The transcript may contain transcription "
                    "errors; use the most likely intent. Use the Home Assistant tools "
                    "to control the home and read live state. Reply with the verified "
                    "result in one short conversational sentence the assistant can say "
                    "aloud, with no Markdown and no JSON, and never claim an action "
                    "completed without a tool result confirming it."
                ),
            ),
        ),
    )
