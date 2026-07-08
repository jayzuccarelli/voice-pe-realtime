"""WebSocket audio broker: bridges a Voice PE device to OpenAI Realtime.

One long-lived process. A device connects over `ws://`, streams PCM up, and
plays PCM back; in between sits a Pipecat pipeline whose brain is an OpenAI
Realtime session (optionally able to control Home Assistant via MCP tools).

OpenAI caps a Realtime session at 60 min and treats expiry as fatal, so the
broker runs the pipeline in a supervised loop: it proactively rotates the
session before the cap (and rebuilds after any session death), rebinding a
fresh transport each time. Within a session, context carries for free; the
device is turn-based, so a rotation between turns is invisible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from websockets.protocol import State

from pipecat.services.llm_service import FunctionCallResultProperties
from pipecat.services.openai.realtime import events as oai_events

from . import mcp_client
from .agent import build_agent, build_audio_input
from .config import Config
from .serializer import RawPCMSerializer

logger = logging.getLogger(__name__)


class _UserTranscriptLogger(FrameProcessor):
    """DEBUG: log what OpenAI thinks the user said (requires input transcription).

    Consumes the TranscriptionFrame instead of re-pushing it. If one reaches
    the user aggregator upstream, the aggregator emulates VAD (a spurious
    pipeline interruption) and then pushes a context frame that makes the
    service double-fire response.create — OpenAI rejects it
    (conversation_already_has_active_response) and Pipecat treats any error
    event as fatal, silently killing the session's receive loop. Nothing
    upstream of here needs the transcript, so log it and stop it.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.UPSTREAM:
            logger.info("USER TRANSCRIPT: %r", frame.text)
            return
        await self.push_frame(frame, direction)


class _BotPlaybackGate(FrameProcessor):
    """Gate barge-in flushes and the OpenAI VAD threshold on DEVICE playback.

    Two jobs, both keyed to whether the device speaker is audibly playing bot
    speech:

    1. Adaptive VAD: the XMOS AEC leaves enough residual of the bot's own
       voice in the mic feed to trip server_vad at any threshold a
       normal-volume user can also cross (measured: bleed trips 0.6, user is
       inaudible at 0.7+). So: sensitive threshold while idle, strict while
       the bot has the floor. Barge-in still works — it just needs a slightly
       raised voice.

    2. Barge-in flush: on a real interruption, tell the device to drop its
       speaker queue ({"type":"interrupt"}) so the cutoff is crisp. Pipecat
       also emits InterruptionFrame at turn boundaries when nothing is
       playing; flushing then would mask the first 500ms of the next reply
       (the firmware's INTERRUPT_IGNORE_AUDIO_MS window), so flush only while
       playback is live.

    Why model playback instead of using BotStarted/StoppedSpeakingFrame:
    OpenAI delivers TTS audio faster than realtime, the output transport
    paces it to the device at 1x, and the device buffers up to ~740ms more
    (10-chunk send queue + 3x100ms ring buffers). BotStopped fires when the
    SEND queue drains, while the speaker may still be audible. Summing the
    duration of the audio frames we forward gives the actual time the speaker
    goes quiet.
    """

    def __init__(self, service, config: Config, get_ws) -> None:  # noqa: ANN001
        super().__init__()
        self._service = service
        self._config = config
        self._get_ws = get_ws
        self._playback_end = 0.0  # event-loop time when the speaker goes quiet
        self._threshold = config.vad_threshold
        self._restore_task: asyncio.Task | None = None
        self._reset_task: asyncio.Task | None = None  # in-flight reset_vad
        # Max (playback_end - now) seen during the current reply: how far the
        # forwarded-audio schedule runs ahead of the speaker. OpenAI delivers
        # faster than realtime, so anything correlating mic audio against
        # forwarded TTS (the planned echo gate) must index by this schedule,
        # not by arrival time. Logged per reply to size that misalignment.
        self._max_backlog = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            now = asyncio.get_running_loop().time()
            duration = len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)
            self._playback_end = max(self._playback_end, now) + duration
            self._max_backlog = max(self._max_backlog, self._playback_end - now)
            await self._set_threshold(self._config.vad_threshold_speaking)
            if self._restore_task is None:
                self._restore_task = self.create_task(self._restore_when_quiet())
        elif isinstance(frame, InterruptionFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._maybe_flush_device()
        elif isinstance(frame, (CancelFrame, EndFrame)):
            if self._restore_task is not None:
                task, self._restore_task = self._restore_task, None
                await self.cancel_task(task)
        await self.push_frame(frame, direction)

    async def _restore_when_quiet(self) -> None:
        delay = self._config.vad_release_delay_ms / 1000
        loop = asyncio.get_running_loop()
        while True:
            wait = self._playback_end + delay - loop.time()
            if wait <= 0:
                break
            await asyncio.sleep(wait)
        # Clear BEFORE the await below: an audio frame racing the restore then
        # starts a fresh restore task and re-raises the threshold after us.
        self._restore_task = None
        if self._max_backlog > 0:
            logger.info("playback: max send-ahead backlog %.2fs this reply", self._max_backlog)
            self._max_backlog = 0.0
        await self._set_threshold(self._config.vad_threshold)

    async def _maybe_flush_device(self) -> None:
        now = asyncio.get_running_loop().time()
        if now >= self._playback_end:
            return  # nothing audibly playing; boundary interruption, not barge-in
        ws = self._get_ws()
        if ws is None:
            return
        try:
            await ws.send('{"type":"interrupt"}')
        except Exception:  # noqa: BLE001
            logger.exception("barge-in: failed to signal device")
            return
        self._playback_end = now  # device drops its queue on the flush
        logger.info("barge-in: signaled device to flush speaker")

    async def _set_threshold(self, threshold: float) -> None:
        if threshold == self._threshold:
            return
        self._threshold = threshold
        if self._reset_task is not None:
            # A VAD reset is in flight: sending server_vad config now would
            # re-enable turn detection before the reset's buffer clear and
            # resurrect the ghost turn it exists to kill. Just record the
            # threshold; the reset's final re-enable applies it.
            logger.info("adaptive VAD: threshold -> %s (deferred, reset in flight)", threshold)
            return
        await self._send_vad(threshold)
        logger.info("adaptive VAD: threshold -> %s", threshold)

    async def _send_vad(self, threshold: float | None) -> None:
        await self._service.send_client_event(
            oai_events.SessionUpdateEvent(
                session=oai_events.SessionProperties(
                    audio=oai_events.AudioConfiguration(
                        input=build_audio_input(self._config, threshold)
                    )
                )
            )
        )

    async def reset_vad(self, drain: float = 0.4) -> None:
        """Discard server VAD state left over from a dead connection.

        Clearing the input buffer removes the audio BYTES but not the VAD
        state machine: when the device vanishes mid-utterance, server_vad
        still holds speech-started, and once post-reconnect silence gives it
        its window it commits whatever is buffered — the tail fragment that
        drained from the pipeline after the clear, or nothing at all — and
        auto-creates a response. The model greets the ghost turn ("I'm here
        when you're ready"; soak 2026-07-02, 5/5 then 2/6 with a delayed
        clear alone). Disabling turn detection makes the server drop the
        pending segment; then let the stale pipeline tail drain (`drain`
        seconds — 0 on connect, when any tail drained long ago), wipe the
        buffer, and re-enable at the current threshold.

        Single-flight: if a reset is already in flight, joining callers
        no-op — the running one clears and re-enables for everyone. In
        particular a fast reconnect's on-connect reset must NOT preempt the
        disconnect reset mid-drain, or the clear fires before the stale
        tail lands and the ghost returns. _set_threshold defers its
        session.update while a reset holds the floor. Safe window: a
        reconnect needs a full wake-word ceremony, so no genuine speech
        arrives within `drain` of a disconnect.
        """
        if self._reset_task is not None:
            return
        task = asyncio.current_task()
        self._reset_task = task
        try:
            await self._send_vad(None)
            if drain:
                await asyncio.sleep(drain)
            await self._service.send_client_event(oai_events.InputAudioBufferClearEvent())
            await self._send_vad(self._threshold)
        except Exception:  # noqa: BLE001
            logger.exception("VAD reset failed")
        finally:
            if self._reset_task is task:
                self._reset_task = None


class _MicInputGate(FrameProcessor):
    """Feed OpenAI silence while the device speaker is audibly playing.

    The device streams its mic continuously (open for barge-in), but the XMOS
    AEC leaves a residual of the bot's own voice in that feed. With no
    server-side noise reduction (removed because it scrubbed the quiet NS-tap
    mic to nothing), that residual is loud enough for server_vad to read as
    user speech, so the bot answers its own echo in a runaway loop. While the
    speaker is playing — tracked by the playback gate's buffer-accurate model,
    plus the VAD release margin — replace the incoming mic audio with silence
    so OpenAI has nothing to trigger on. This makes the assistant turn-based;
    real barge-in needs echo cancellation, not just an open mic.
    """

    def __init__(self, gate: "_BotPlaybackGate", config: Config) -> None:
        super().__init__()
        self._gate = gate
        self._config = config

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            now = asyncio.get_running_loop().time()
            release = self._config.vad_release_delay_ms / 1000
            if now < self._gate._playback_end + release:
                frame = InputAudioRawFrame(
                    audio=bytes(len(frame.audio)),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
        await self.push_frame(frame, direction)


# Fixed allowance from client connect until the first committed user turn. A
# silent false wake (TV says the wake word, nobody follows up) would otherwise
# stream mic audio to OpenAI until session rotation: the firmware's 10s
# auto-stop only arms after the first bot audio chunk, which a silent wake
# never produces. Internal constant on purpose — a safety floor, not a knob.
_INITIAL_GRACE_SECONDS = 10.0


class _TurnHygiene(FrameProcessor):
    """Bound how long one wake keeps the conversation open (JAY-84).

    TV speech can keep re-tripping server_vad and spiral a single wake into a
    minutes-long open-mic session. Two structural bounds, both broker-side
    (the close reuses end_conversation's {"type":"disconnect"} control frame,
    so the firmware plays the existing end cue and re-arms the wake word;
    zero firmware changes):

    - Follow-up window: after each reply the user gets
      config.followup_window_seconds — measured from when the SPEAKER goes
      quiet (the playback gate's clock + VAD release), not response.done — to
      take another turn. Expiry with nothing in flight -> disconnect. A real
      follow-up starts a fresh window after its reply, so natural
      conversation is untouched.
    - Turn budget: config.max_turns_per_wake committed user turns per WS
      connection; once reached, disconnect as soon as the current reply
      finishes playing (never mid-speech). Caps the model-fooled-by-TV
      spiral even when each TV line lands inside the window.

    Either knob at 0 disables that bound (both 0 = exactly the pre-hygiene
    behavior) — the no-redeploy rollback lever. The OpenAI session stays
    alive either way; context carries to the next wake as today.

    Must sit between the service and the assistant context aggregator: the
    aggregator CONSUMES LLMFullResponseEndFrame and FunctionCallsStartedFrame
    (llm_response_universal.py) instead of re-pushing them, so downstream of
    it a response end is invisible.
    """

    def __init__(self, gate: "_BotPlaybackGate", config: Config, get_ws) -> None:  # noqa: ANN001
        super().__init__()
        self._gate = gate  # single source of truth for the playback clock
        self._config = config
        self._get_ws = get_ws
        self._device_connected = False
        self._connect_time = 0.0
        self._turns = 0  # committed user turns this WS connection
        self._user_speaking = False  # UserStarted seen without UserStopped
        self._awaiting_response = False  # a response should still arrive/finish
        self._awaiting_since = 0.0  # when awaiting was last (re)armed
        self._tool_followup = False  # a tool call will trigger a 2nd response
        self._watch_task: asyncio.Task | None = None  # single-flight closer
        self._gen = 0  # bumps per device connect; stale watchers self-identify
        self._conn_ws = None  # the socket this connection's state belongs to

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            # server_vad committed the turn and auto-creates the response.
            self._user_speaking = False
            self._awaiting_response = True
            self._awaiting_since = asyncio.get_running_loop().time()
            self._tool_followup = False  # fresh response cycle; drop stale state
            self._turns += 1
        elif isinstance(frame, FunctionCallsStartedFrame) and direction == FrameDirection.DOWNSTREAM:
            # Keyed on FunctionCallsStartedFrame, not FunctionCallInProgressFrame:
            # in-progress is broadcast from a function-runner TASK
            # (llm_service._run_parallel_function_calls), so it can land after
            # response.done's LLMFullResponseEndFrame; started is pushed inline
            # from the receive loop, strictly before it.
            if any(fc.function_name != "wait_for_user" for fc in frame.function_calls):
                # HA tools / get_weather / play_music trigger a second
                # response.create for the verbalization — stay awaiting until
                # THAT response ends. wait_for_user has run_llm=False (no
                # follow-up response), so an ignored TV line clears awaiting
                # on this response's end; playback_end is already past, so
                # the window expires promptly instead of being extended.
                self._tool_followup = True
                self._awaiting_since = asyncio.get_running_loop().time()
        elif isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._tool_followup:
                self._tool_followup = False  # verbalization response still coming
                self._awaiting_since = asyncio.get_running_loop().time()
            else:
                self._awaiting_response = False
        elif isinstance(frame, (CancelFrame, EndFrame)):
            if self._watch_task is not None:
                task, self._watch_task = self._watch_task, None
                await self.cancel_task(task)
        await self.push_frame(frame, direction)

    def on_device_connect(self) -> None:
        # Fresh wake = fresh budget and window. The connect handler runs for
        # the NEW client before a kicked stale socket's disconnect fires (and
        # that handler's stale guard returns early), so old-connection
        # teardown can't clobber this state.
        self._device_connected = True
        self._connect_time = asyncio.get_running_loop().time()
        self._turns = 0
        self._user_speaking = False
        self._awaiting_response = False
        self._tool_followup = False
        self._gen += 1
        # Stamp the socket this connection's state belongs to (pipecat swaps
        # transport._websocket before this handler's task runs) — the watcher
        # re-checks it before sending, so a tick that interleaves between the
        # swap and this reset can't fire the OLD connection's expiry at the
        # NEW socket and kill a fresh wake.
        self._conn_ws = self._get_ws()
        if self._watch_task is not None:
            # Always cancel-and-recreate rather than reuse: an old watcher can
            # be parked in ws.send on a half-dead kicked socket for seconds —
            # skipping creation here would leave the new connection with NO
            # watcher once it finishes (window + budget silently off).
            task, self._watch_task = self._watch_task, None
            asyncio.create_task(self.cancel_task(task))
        if self._config.followup_window_seconds > 0 or self._config.max_turns_per_wake > 0:
            self._watch_task = self.create_task(self._watch())

    async def on_device_disconnect(self) -> None:
        self._device_connected = False
        if self._watch_task is not None:
            task, self._watch_task = self._watch_task, None
            await self.cancel_task(task)

    async def _watch(self) -> None:
        # ~0.5s tick; single-flight like the gate's _restore_task. Exits (and
        # clears itself) after signaling the close: the firmware closes the WS
        # in response, and the next wake's on_device_connect starts a fresh
        # watcher. end_conversation needs no special-casing — its disconnect
        # makes the device drop, and on_device_disconnect cancels us.
        loop = asyncio.get_running_loop()
        release = self._config.vad_release_delay_ms / 1000
        window = self._config.followup_window_seconds
        budget = self._config.max_turns_per_wake
        my_gen = self._gen  # a newer connection's watcher supersedes us
        while True:
            await asyncio.sleep(0.5)
            if self._gen != my_gen:
                return  # superseded; the new watcher owns _watch_task now
            now = loop.time()
            if self._awaiting_response and now - self._awaiting_since > 60.0:
                # Fail-open guard: OpenAI can kill a response without a
                # response.done reaching the pipeline, and a hung tool's
                # verbalization response may never materialize — either would
                # hold the window open forever. 60s is far above any real
                # response/tool latency (play_music's HA path ~20s); don't
                # shorten it below that.
                logger.warning("turn hygiene: response overdue >60s; clearing awaiting")
                self._awaiting_response = False
            if not self._device_connected or self._user_speaking or self._awaiting_response:
                continue
            quiet_since = self._gate._playback_end + release  # speaker + echo tail
            if budget > 0 and self._turns >= budget and now >= quiet_since:
                reason = f"turn budget reached ({self._turns}/{budget})"
            elif window > 0 and now >= max(
                quiet_since, self._connect_time + _INITIAL_GRACE_SECONDS
            ) + window:
                # max() folds in the initial grace: before the first committed
                # turn playback_end is stale (or zero), and a silent false
                # wake must still get bounded — grace + window — instead of
                # streaming mic audio to OpenAI until rotation.
                reason = f"follow-up window expired ({self._turns} turns)"
            else:
                continue
            ws = self._get_ws()
            if ws is None:
                continue  # transport already dropped it; disconnect event cleans up
            if ws is not self._conn_ws:
                continue  # socket swapped mid-tick (re-wake); fresh state incoming
            try:
                await ws.send('{"type":"disconnect"}')
            except Exception:  # noqa: BLE001
                logger.exception("turn hygiene: failed to signal device")
            else:
                logger.info("turn hygiene: %s; disconnecting device", reason)
            if self._gen == my_gen:
                # Only clear our own handle — a re-wake during the send above
                # already replaced _watch_task with the new watcher's.
                self._watch_task = None
            return


async def run(config: Config) -> None:
    """Serve forever, rotating the OpenAI session before its 60-min cap."""
    mcp = None
    if config.ha_control_enabled:
        mcp = await mcp_client.connect(config.ha_mcp_url, config.ha_token)
    else:
        logger.info("Home Assistant control disabled (HA_MCP_URL/HA_TOKEN unset)")

    logger.info("Broker listening on ws://%s:%d", config.ws_host, config.ws_port)
    while True:
        try:
            await _serve_session(config, mcp)
        except Exception:  # noqa: BLE001
            logger.exception("Session crashed; rebuilding")
        await asyncio.sleep(0.5)  # let the socket fully release before rebind


def _fetch_weather(config: Config) -> str:
    """Live weather from HA (HA's MCP doesn't surface the weather domain)."""
    if not config.ha_control_enabled:
        return "Weather is unavailable; Home Assistant is not connected."
    base = config.ha_mcp_url.split("/mcp_server")[0]
    try:
        req = urllib.request.Request(
            f"{base}/api/states/{config.weather_entity}",
            headers={"Authorization": f"Bearer {config.ha_token}"},
        )
        st = json.load(urllib.request.urlopen(req, timeout=8))
        a = st.get("attributes", {})
        unit = a.get("temperature_unit", "\u00b0F")
        parts = [f"condition {st.get('state')}"]
        if a.get("temperature") is not None:
            parts.append(f"{a['temperature']}{unit}")
        if a.get("humidity") is not None:
            parts.append(f"humidity {a['humidity']}%")
        if a.get("wind_speed") is not None:
            parts.append(f"wind {a['wind_speed']} {a.get('wind_speed_unit', '')}".strip())
        return "Current weather: " + ", ".join(parts) + "."
    except Exception as e:  # noqa: BLE001
        logger.warning("weather fetch failed: %s", e)
        return "Sorry, I couldn't get the weather right now."


def _start_music(config: Config, query: str, speaker: str | None) -> str:
    """Play `query` on a Music Assistant speaker via music_assistant.play_media.

    HA's generic media-search tool refuses MA players (they don't advertise the
    SEARCH_MEDIA feature), so we call MA's own service, which searches Spotify
    internally. Resolves the spoken speaker name to an MA media_player entity
    (only those accept this service); falls back to config.music_player.
    """
    if not config.ha_control_enabled:
        return "Music is unavailable; Home Assistant is not connected."
    base = config.ha_mcp_url.split("/mcp_server")[0]
    hdr = {"Authorization": f"Bearer {config.ha_token}", "Content-Type": "application/json"}
    # Only MA media_player entities accept music_assistant.play_media. Get their
    # ids via a template, then read friendly names from /api/states.
    tmpl = (
        "{{ integration_entities('music_assistant') "
        "| select('match','media_player') | list | to_json }}"
    )
    players: dict = {}  # entity_id -> friendly_name
    try:
        req = urllib.request.Request(
            base + "/api/template", data=json.dumps({"template": tmpl}).encode(), headers=hdr
        )
        ma_ids = set(json.loads(urllib.request.urlopen(req, timeout=8).read()))
        states = json.load(
            urllib.request.urlopen(
                urllib.request.Request(base + "/api/states", headers=hdr), timeout=8
            )
        )
        for s in states:
            if s["entity_id"] in ma_ids:
                players[s["entity_id"]] = s.get("attributes", {}).get("friendly_name")
    except Exception as e:  # noqa: BLE001
        logger.warning("MA player lookup failed: %s", e)

    avail = ", ".join(sorted(n for n in players.values() if n)) or "no speakers"
    if speaker:
        sp = speaker.lower()
        target = next(
            (eid for eid, name in players.items() if name and (name.lower() in sp or sp in name.lower())),
            None,
        )
        if target is None:
            # A speaker was named but didn't match — don't silently play on the
            # wrong one; tell the user what's available.
            return f"I couldn't find a speaker called {speaker}. Available: {avail}."
    else:
        target = config.music_player  # no speaker named -> default
    if not target:
        return f"Which speaker should I use? Available: {avail}."
    try:
        body = json.dumps({"entity_id": target, "media_id": query}).encode()
        urllib.request.urlopen(
            urllib.request.Request(
                base + "/api/services/music_assistant/play_media", data=body, headers=hdr
            ),
            timeout=20,
        )
        return f"Playing {query}."
    except Exception as e:  # noqa: BLE001
        logger.warning("play_music failed: %s", e)
        return "Sorry, I couldn't start the music."


async def _serve_session(config: Config, mcp) -> None:  # noqa: ANN001
    """Run one OpenAI session until it dies or reaches max age, then tear down."""
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

    async def _get_weather(params):  # noqa: ANN001
        # Run the blocking HA fetch off the event loop so it can't stall audio.
        weather = await asyncio.to_thread(_fetch_weather, config)
        await params.result_callback(weather)

    async def _end_conversation(params):  # noqa: ANN001
        await params.result_callback("Okay, goodbye!")
        ws = getattr(transport.input(), "_websocket", None)
        if ws is not None:
            try:
                await ws.send('{"type":"disconnect"}')
            except Exception:  # noqa: BLE001
                logger.exception("end_conversation: failed to signal device")

    async def _play_music(params):  # noqa: ANN001
        args = params.arguments or {}
        msg = await asyncio.to_thread(
            _start_music, config, args.get("query", ""), args.get("speaker")
        )
        await params.result_callback(msg)

    async def _wait_for_user(params):  # noqa: ANN001
        # Non-addressed speech (TV, side conversation, background). Acknowledge
        # the call but suppress the follow-up response (run_llm=False) so the bot
        # stays silent and keeps listening instead of replying to the room.
        await params.result_callback(
            "", properties=FunctionCallResultProperties(run_llm=False)
        )

    service.register_function("get_weather", _get_weather)
    service.register_function("end_conversation", _end_conversation)
    service.register_function("play_music", _play_music)
    service.register_function("wait_for_user", _wait_for_user)

    gate = _BotPlaybackGate(
        service, config, lambda: getattr(transport.input(), "_websocket", None)
    )
    mic_gate = _MicInputGate(gate, config)
    hygiene = _TurnHygiene(
        gate, config, lambda: getattr(transport.input(), "_websocket", None)
    )
    aggregator = LLMContextAggregatorPair(LLMContext())
    pipeline = Pipeline(
        [
            transport.input(),
            mic_gate,
            aggregator.user(),
            # The realtime service pushes user TranscriptionFrames UPSTREAM
            # (Pipecat >= 0.0.92); the logger must sit between the aggregator
            # and the service so it can intercept (and consume) them before
            # the aggregator reacts to them — see _UserTranscriptLogger.
            _UserTranscriptLogger(),
            service,
            # Turn hygiene must observe the service's raw downstream frames;
            # the assistant aggregator below consumes response-end and
            # function-call frames — see _TurnHygiene.
            hygiene,
            aggregator.assistant(),
            gate,
            transport.output(),
        ]
    )

    loop = asyncio.get_running_loop()
    device_connected = False
    idle_since = loop.time()

    @transport.event_handler("on_client_connected")
    async def _on_connect(_transport, client):  # noqa: ANN001
        nonlocal device_connected
        device_connected = True
        logger.info("Device connected: %s", getattr(client, "remote_address", client))
        # The device opens a fresh websocket per wake, but the OpenAI session
        # is reused for context. Anything left from the previous connection —
        # uncommitted buffer audio AND a speech-in-progress VAD segment —
        # would surface as a ghost turn before the real question (stray
        # 'Bye.', TV test 2026-07-01). Full reset, no drain: any stale
        # pipeline tail finished draining while no device was connected, and
        # the reset's few client events land well before wake-ack ends and
        # real speech starts.
        asyncio.create_task(gate.reset_vad(drain=0.0))
        hygiene.on_device_connect()

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, client, *args):  # noqa: ANN001
        nonlocal device_connected, idle_since
        # When a new connection kicks a lingering old one (re-wake after a
        # WiFi blip: the dead socket never closed), Pipecat swaps the
        # transport's websocket to the NEW client before the old handler
        # exits and fires this event for the OLD one. The device is still
        # here — don't flag it disconnected, and above all don't reset VAD
        # and clear the buffer while the user's real question streams in.
        # The on-connect reset already dealt with the stale state.
        current = getattr(transport.input(), "_websocket", None)
        if current is not None and current is not client:
            logger.info("Stale connection closed; device still connected")
            return
        device_connected = False
        idle_since = loop.time()
        logger.info("Device disconnected")
        # A disconnect mid-utterance leaves server VAD holding a
        # speech-in-progress segment (plus mic-stream tail still draining
        # through the pipeline). A buffer clear alone can't kill it — see
        # _BotPlaybackGate.reset_vad, which disables and re-enables turn
        # detection to drop the segment before it becomes a ghost turn
        # (stray 'Bye.' in the TV test 2026-07-01; 'I'm here when you're
        # ready' in the soak 2026-07-02).
        asyncio.create_task(gate.reset_vad())
        # Cancel the hygiene watcher for a REAL disconnect only — the stale
        # guard above already returned for a kicked old socket, so it can't
        # cancel the new connection's window/budget state.
        await hygiene.on_device_disconnect()

    task = PipelineTask(pipeline, idle_timeout_secs=None, cancel_on_idle_timeout=False)
    runner_task = asyncio.create_task(PipelineRunner().run(task))
    deadline = loop.time() + config.max_session_seconds
    try:
        # Whichever comes first: the session dies on its own, it ages out, or
        # the OpenAI socket dies underneath it. OpenAI drops idle Realtime
        # sockets well before our max-age rotation, and Pipecat keeps the
        # stale handle and pumps audio into it — the device hears silence
        # until the next rotation. Poll the socket and rotate the moment it
        # goes dead (websockets' keepalive flips state within ~40s).
        #
        # Both age-based rotations only fire while no device is connected:
        # rotating mid-conversation kicks the device off (it happened live,
        # 2026-06-10 23:46). A conversation can outlive max age by at most
        # its own length; OpenAI's 60-min hard cap is the backstop, and the
        # dead-socket check below catches that.
        while True:
            now = loop.time()
            if not device_connected:
                if now >= deadline:
                    logger.info(
                        "Session reached max age (%ds); rotating",
                        config.max_session_seconds,
                    )
                    break
                if now - idle_since >= config.idle_refresh_seconds:
                    logger.info(
                        "Idle session refresh (no device for %ds); rotating",
                        int(now - idle_since),
                    )
                    break
            timeout = min(15.0, deadline - now) if now < deadline else 15.0
            try:
                await asyncio.wait_for(asyncio.shield(runner_task), timeout)
                logger.warning("Pipeline ended on its own; rebuilding session")
                break
            except asyncio.TimeoutError:
                pass
            # Two death modes: the socket itself dies (idle drop — keepalive
            # flips the state within ~40s), or Pipecat's receive loop exits on
            # an OpenAI error event while the socket stays OPEN (brain-dead
            # session: audio goes in, nothing comes back).
            ws = getattr(service, "_websocket", None)
            receiver = getattr(service, "_receive_task", None)
            if (
                ws is None
                or ws.state is not State.OPEN
                or (receiver is not None and receiver.done())
            ):
                logger.warning(
                    "OpenAI session dead (ws state=%s, receiver done=%s); rotating now",
                    getattr(ws, "state", None),
                    receiver.done() if receiver is not None else None,
                )
                break
    finally:
        if not runner_task.done():
            await task.cancel()
        # Await the runner to fully finish so the websocket server releases the
        # port before the next session rebinds — otherwise rotation can hit an
        # intermittent "address already in use".
        try:
            await asyncio.wait_for(asyncio.shield(runner_task), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Runner slow to stop after cancel; forcing")
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("Runner teardown error")
