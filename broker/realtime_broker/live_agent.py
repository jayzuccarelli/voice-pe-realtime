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

import array
import logging
import math
from collections import deque

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import InputAudioRawFrame
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

    #: Terminal states for a delegated backend response.
    _RESPONSE_DONE = ("response.completed", "response.incomplete", "response.failed")

    #: How much speech to hold while the billed session is opening. Opening
    #: costs ~3s (our websocket handshake, then the API's own session.start),
    #: and a puck connects *because* the user just said the wake word, so the
    #: question is spoken entirely inside that window. Bounded so an unusually
    #: slow open replays recent speech rather than a stale backlog.
    _MAX_PRESTART_AUDIO_SECONDS = 8.0

    #: Length of the rolling window the gate measures level over.
    _GATE_WINDOW_SECONDS = 0.1
    #: What gated frames are scaled by: 40 dB down, not digital zero. This is
    #: the figure the replay experiments validated against the real device.
    _GATE_ATTENUATION = 0.01

    def __init__(
        self, *, input_gate_rms: float = 0.0, input_gate_hold_ms: float = 250.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Ids of delegated responses still open, for the CURRENT session only.
        self._live_open_responses: set[str] = set()
        # Mic audio captured while the session was still opening, oldest first.
        self._prestart_audio: deque[InputAudioRawFrame] = deque()
        self._prestart_seconds = 0.0
        # Input noise gate. 0 disables it. The decision is made on a rolling
        # 100 ms window, not the 20 ms frame: frame-level RMS dips under the
        # threshold on every consonant and micro-pause and the gate then
        # chops words into fragments. 100 ms is what the replay experiments
        # validated.
        self._input_gate_rms = float(input_gate_rms)
        self._gate_hold_seconds = float(input_gate_hold_ms) / 1000.0
        self._gate_quiet_seconds = 0.0
        self._gate_window: deque[tuple[float, int, float]] = deque()  # (sum sq, n, secs)
        self._gate_window_seconds = 0.0
        self._gate_passed = 0
        self._gate_attenuated = 0
        logger.info(
            "Live input gate: rms<%.0f attenuated after %.0f ms quiet (0 = off)",
            self._input_gate_rms,
            self._gate_hold_seconds * 1000,
        )

    def _gate_input(self, frame: InputAudioRawFrame) -> InputAudioRawFrame:
        """Attenuate frames that carry only the room's noise floor.

        gpt-live-1 does its own turn detection, and on this device it never
        opens a turn: the puck's mic path carries a constant floor of about
        -40 dBFS (mains hum and a device tone) under and around the speech,
        and the model treats that as "no one is talking" no matter how loud
        the words on top of it are. The same recording with its between-word
        floor pulled down 40 dB gets answered; clean synthetic speech, which
        falls to digital zero between words, always did. Level, bandwidth,
        leading silence and the persona's far-field instruction were each
        ruled out by replaying the device's own capture with one change at a
        time (2026-09-14).

        So the floor is removed here, before the audio reaches the model,
        with a hold so word tails and mid-sentence pauses are not chopped.
        """
        if self._input_gate_rms <= 0:
            return frame
        samples = array.array("h")
        samples.frombytes(frame.audio[: len(frame.audio) // 2 * 2])
        if not samples:
            return frame
        secs = self._frame_seconds(frame)
        frame_sq = float(sum(s * s for s in samples))
        self._gate_window.append((frame_sq, len(samples), secs))
        self._gate_window_seconds += secs
        while self._gate_window_seconds > self._GATE_WINDOW_SECONDS and len(self._gate_window) > 1:
            _, _, old = self._gate_window.popleft()
            self._gate_window_seconds -= old
        total_sq = sum(w[0] for w in self._gate_window)
        total_n = sum(w[1] for w in self._gate_window) or 1
        # Fast attack, slow release: the frame alone opens the gate, so a
        # word's first milliseconds are never clipped while the window still
        # holds the quiet before it; the window keeps it open across the
        # dips inside a word.
        frame_rms = math.sqrt(frame_sq / len(samples))
        window_rms = math.sqrt(total_sq / total_n)
        rms = max(frame_rms, window_rms)
        if rms >= self._input_gate_rms:
            self._gate_quiet_seconds = 0.0
            self._gate_passed += 1
            return frame
        self._gate_quiet_seconds += self._frame_seconds(frame)
        if self._gate_quiet_seconds <= self._gate_hold_seconds:
            self._gate_passed += 1
            return frame
        self._gate_attenuated += 1
        quiet = array.array("h", (int(s * self._GATE_ATTENUATION) for s in samples))
        return InputAudioRawFrame(
            audio=quiet.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )

    @property
    def delegation_in_flight(self) -> bool:
        """Whether a delegated backend response is still being worked on."""
        return bool(self._live_open_responses)

    @staticmethod
    def _response_key(evt) -> str | None:
        """Identify the delegated response an envelope belongs to."""
        event = getattr(evt, "event", None) or {}
        response = event.get("response") or {}
        key = response.get("id")
        if key:
            return str(key)
        # Fall back to the delegation the envelope was wrapped in, which the
        # API carries even when the response snapshot is empty.
        for attr in ("delegation_id", "item_id", "event_id"):
            value = getattr(evt, attr, None)
            if value:
                return str(value)
        return None

    async def _handle_evt_response(self, evt) -> None:
        """Track which delegated responses are open, by id.

        Tracked here rather than read off upstream's `_pending_responses`,
        which only drops an entry for a response that made function calls: a
        delegated response that just answers is never removed, so its length
        reads as "forever in flight" after the first delegation.

        Ids rather than a counter, and cleared per session, because a plain
        count is not safe across a session boundary. A completion from the
        previous session can arrive after the next one has already opened a
        delegation, and decrementing a shared counter would release the new
        session's hold and let the watcher hang up mid-answer. An id that was
        never opened in this session is simply not in the set, so a late
        arrival from a dead session is ignored instead of stealing a
        decrement.
        """
        inner = getattr(evt, "inner_type", None)
        key = self._response_key(evt)
        if inner == "response.created":
            if key is not None:
                self._live_open_responses.add(key)
        elif inner in self._RESPONSE_DONE:
            if key is not None:
                self._live_open_responses.discard(key)
            elif self._live_open_responses:
                # No id to match on. Release one hold rather than hold
                # forever; the pending-reply grace and the hard cap still
                # bound the session either way.
                self._live_open_responses.pop()
        await super()._handle_evt_response(evt)

    @staticmethod
    def _frame_seconds(frame: InputAudioRawFrame) -> float:
        """Wall-clock duration of one PCM16 frame."""
        rate = getattr(frame, "sample_rate", 0) or 0
        channels = getattr(frame, "num_channels", 1) or 1
        if not rate:
            return 0.0
        return len(frame.audio) / float(rate * channels * 2)

    def _clear_prestart_audio(self) -> None:
        self._prestart_audio.clear()
        self._prestart_seconds = 0.0

    async def _send_user_audio(self, frame: InputAudioRawFrame) -> None:
        """Hold mic audio that arrives before the session is live.

        Upstream drops it ("dropping input audio until the session has
        started"). On a call that is harmless, because the human is still
        saying hello into an already-open session. On a wake-word puck it is
        the entire request: the device connects *because* the user just
        spoke, so the question lands inside the ~3s the session takes to
        open. Dropping it makes a wake sound like the chime followed by
        nothing, the model having come alive to silence and hung up with
        zero turns.
        """
        frame = self._gate_input(frame)
        if not self._session_started:
            self._prestart_audio.append(frame)
            self._prestart_seconds += self._frame_seconds(frame)
            while self._prestart_audio and self._prestart_seconds > self._MAX_PRESTART_AUDIO_SECONDS:
                self._prestart_seconds -= self._frame_seconds(self._prestart_audio.popleft())
            return
        await super()._send_user_audio(frame)

    async def _handle_evt_session_started(self, evt) -> None:
        """Start the session, then replay what the user said while it opened."""
        await super()._handle_evt_session_started(evt)
        if not self._prestart_audio:
            return
        held = list(self._prestart_audio)
        seconds = self._prestart_seconds
        self._clear_prestart_audio()
        self._describe_prestart_audio(held, seconds)
        for frame in held:
            await super()._send_user_audio(frame)

    def _describe_prestart_audio(self, held, seconds: float) -> None:
        """Log what was actually captured, and keep a copy to listen to.

        A flush that reports the right duration still tells us nothing about
        whether the user's voice is in it. Peak amplitude separates "we held
        3s of the user asking a question" from "we held 3s of near-silence",
        which are the same line in the log otherwise.
        """
        import array
        import wave

        samples = array.array("h")
        for frame in held:
            try:
                samples.frombytes(frame.audio)
            except ValueError:
                pass
        peak = max((abs(s) for s in samples), default=0)
        rates = sorted({getattr(f, "sample_rate", 0) for f in held})
        logger.info(
            "Flushing %.1fs of speech captured while the session opened "
            "(%d frames, rate(s)=%s, peak=%d/32767)",
            seconds,
            len(held),
            rates,
            peak,
        )
        if not samples:
            return
        try:
            path = f"/tmp/claude/prestart-{int(seconds * 1000)}ms.wav"
            with wave.open(path, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rates[-1] if rates else 16000)
                out.writeframes(samples.tobytes())
            logger.info("Wrote captured audio to %s", path)
        except OSError as exc:  # debug aid only; never break a session for it
            logger.warning("Could not write captured audio: %s", exc)

    async def begin_live_session(self) -> None:
        """Open a billed session for a freshly connected device."""
        if self._session_started:
            return
        # A fresh session owns no delegations. Clearing here is what makes a
        # late completion from the previous session harmless.
        self._live_open_responses.clear()
        self._clear_prestart_audio()
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
                self._live_open_responses.clear()
                self._clear_prestart_audio()
                if self._input_gate_rms > 0:
                    logger.info(
                        "Live input gate this session: %d frames passed, %d attenuated",
                        self._gate_passed,
                        self._gate_attenuated,
                    )
                self._gate_passed = self._gate_attenuated = 0
                self._gate_quiet_seconds = 0.0
                self._gate_window.clear()
                self._gate_window_seconds = 0.0


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
    guidance = (BACKGROUND_GUIDANCE if config.live_far_field_guidance else "") + DELEGATION_GUIDANCE
    return VoicePELiveService(
        api_key=config.openai_api_key,
        input_gate_rms=config.live_input_gate_rms,
        input_gate_hold_ms=config.live_input_gate_hold_ms,
        settings=VoicePELiveService.Settings(
            model=config.live_model,
            voice=config.live_voice or config.voice,
            system_instruction=config.instructions + guidance,
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
