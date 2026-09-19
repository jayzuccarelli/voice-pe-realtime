"""WebSocket audio broker, GPT-Live engine: bridges a Voice PE to gpt-live-1.

Same device contract as the Realtime engine (raw PCM16/24k up and down, JSON
control frames), different brain. The live model is full-duplex: it hears the
user while it speaks and stops on its own, so the machinery the Realtime
engine needed to fake that is gone here. No adaptive VAD, no mic gate feeding
silence during playback, no barge-in flush, no ghost-turn VAD reset, no
session rotation.

What replaces it is smaller and blunter. The live model bills per minute of
open *session*, and the firmware's 10-second auto-stop keys on speaker audio
that a continuous stream keeps alive, so the broker owns the meter: a session
opens when the device connects and closes when it goes away, with a follow-up
window, a turn budget and a hard cap as backstops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.workers.runner import WorkerRunner

from . import mcp_client
from .config import Config
from .live_agent import build_live_agent, build_live_tools
from .serializer import RawPCMSerializer
from .server import _fetch_weather, _start_music

logger = logging.getLogger(__name__)

# How long the watcher will hold off on the follow-up window and the turn
# budget waiting for a reply that has been asked for but has not started.
# This only has to bridge the gap from a committed question to either the
# first word or the first sign of a delegation, measured at a few seconds:
# long thinking is covered by the delegation hold instead, so this does not
# need to be generous. Keeping it short matters because a false wake (a TV
# line the model correctly ignores) commits a turn and never produces speech,
# and every second of that hold is billed.
_REPLY_OVERDUE_SECONDS = 12.0

# Output frames quieter than this (RMS, 16-bit) are the model's idle stream,
# not speech, and are not sent to the device once the reply is over.
_OUTPUT_SILENCE_RMS = 50.0
# How long after the last audible frame quiet frames are still forwarded.
# Pauses inside a reply (a comma, a breath) fall under the threshold too,
# and dropping them punches holes in the stream: the puck's speaker runs dry
# at each one and the voice breaks up (heard 2026-09-15). The mic stays muted
# for this long after the last word, which the firmware's own 500 ms rule
# nearly does anyway.
_OUTPUT_SILENCE_HOLD_SECONDS = 0.8
# Audio held back at the start of each reply before any of it is sent. The
# live model produces speech at exactly playback speed, so without this the
# device plays each chunk the moment it lands and has nothing in reserve:
# measured at the broker, a 4.7 s reply arrived with 13 gaps the device's
# buffer could not cover, and words broke in the middle (2026-09-15). Held
# audio becomes that reserve; the whole reply then plays from 0.4 s behind.
_OUTPUT_PREROLL_SECONDS = 0.4
# Quiet needed after muted filler before the model may be heard again.
_MUTE_RELEASE_SECONDS = 0.3


class _OutputSilenceFilter(FrameProcessor):
    """Send the device only the audio the model actually speaks.

    gpt-live-1 streams output continuously, silence included. The firmware
    treats any speaker audio in the last 500 ms as "the bot is speaking" and
    drops mic data for as long as that holds (voice_assistant_websocket.cpp,
    on_microphone_data_ / is_bot_speaking), a guard written for a turn-based
    model that only ever sent audio while talking. Forward the idle stream
    and the puck goes deaf the moment the session opens: the model then only
    ever hears what was captured before session start, which is the wake
    chime. That was the whole on-device failure (2026-09-14).

    Dropping silent frames is safe on the device side between replies: the
    firmware keeps its own audio chain warm with silence. Inside a reply
    they are kept, so the stream the speaker plays has no holes.
    """

    def __init__(self, output_hold=None) -> None:
        super().__init__()
        self._dropped = 0
        self._sent = 0
        self._last_sent_at = 0.0
        self._last_loud_at = 0.0
        self._burst_frames = 0
        self._preroll: list[OutputAudioRawFrame] = []
        self._preroll_seconds = 0.0
        # (hold, keep): see VoicePELiveService.output_hold.
        self._output_hold = output_hold or (lambda: (False, False))
        self._muting = False
        self._muted_frames = 0
        self._recover: list = []

    def _admit(self, frame, rms: float, now: float) -> list:
        """The frames to send on for this one: none, it, or it plus the
        answer's start that was held back a moment too long.

        While the backend runs a tool the live model fills the silence, and
        what it says has nothing to do with the outcome: "Sure, resetting
        it" to a time question, "Sure, switching it on" during a turn-off,
        "How can I help?" with the lights already going off (2026-09-18/19).
        Told in the prompt not to, it does it anyway, so it is dropped here.
        A mute only ever begins at the start of an utterance, never inside a
        word; it lifts the moment the hold does, and whatever was held back
        after the backend finished is sent first, since that was the answer.
        """
        hold, keep = self._output_hold()
        loud = rms >= _OUTPUT_SILENCE_RMS
        if not self._muting:
            starts_utterance = now - self._last_loud_at >= _MUTE_RELEASE_SECONDS
            if not (hold and loud and starts_utterance):
                if loud:
                    self._last_loud_at = now
                return [frame]
            self._muting = True
            self._recover = []
            logger.info("output: holding the model back (not heard yet, or backend working)")
        if loud:
            self._last_loud_at = now
        if not hold:
            self._muting = False
            recovered, self._recover = self._recover, []
            logger.info(
                "output: released; dropped %d frames, recovered %d of the answer",
                self._muted_frames,
                len(recovered),
            )
            self._muted_frames = 0
            return recovered + [frame]
        if keep:
            self._recover.append(frame)
        else:
            self._muted_frames += len(self._recover) + 1
            self._recover = []
        return []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            rms = _rms16(frame.audio)
            now = asyncio.get_running_loop().time()
            for out in self._admit(frame, rms, now):
                await self._forward(out, rms if out is frame else _rms16(out.audio), now, direction)
            return
        await self.push_frame(frame, direction)

    async def _forward(self, frame, rms: float, now: float, direction: FrameDirection) -> None:
        if rms < _OUTPUT_SILENCE_RMS:
            if now - self._last_loud_at > _OUTPUT_SILENCE_HOLD_SECONDS:
                self._dropped += 1
                return
        else:
            self._last_loud_at = now
        if now - self._last_sent_at > 1.0 and not self._preroll:
            # Start of a burst of audio to the device. Anything here that
            # is not the model talking is what mutes the puck's mic.
            if self._burst_frames:
                logger.info("output: previous burst was %d frames", self._burst_frames)
            logger.info("output: burst starts, rms %.0f (%d dropped as silence since last)", rms, self._dropped)
            self._burst_frames = 0
            self._dropped = 0
        self._burst_frames += 1
        self._sent += 1
        if self._burst_frames <= 1 or self._preroll:
            self._preroll.append(frame)
            self._preroll_seconds += len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)
            if self._preroll_seconds < _OUTPUT_PREROLL_SECONDS:
                return
            held, self._preroll = self._preroll, []
            self._preroll_seconds = 0.0
            self._last_sent_at = now
            for f in held:
                await self.push_frame(f, direction)
            return
        self._last_sent_at = now
        await self.push_frame(frame, direction)


def _rms16(pcm: bytes) -> float:
    import array
    import math

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class _LiveHygiene(FrameProcessor):
    """Own the billing meter for one wake.

    The live model streams output continuously and the firmware's auto-stop
    keys on received speaker audio, so nothing on the device will ever end a
    session. Every bound here is therefore load-bearing, not a nicety:

    - Follow-up window: once the bot stops speaking and the user is not
      talking, the user has `followup_window_seconds` to take another turn.
    - Turn budget: `max_turns_per_wake` user turns per connection, closed as
      soon as the current reply finishes.
    - Hard cap: `max_live_session_seconds` from connect, honoured whatever
      the conversation is doing. This is the fuse: with a $0.05/min meter a
      stuck session is the failure that costs money, so it fires even
      mid-sentence.

    `end_conversation` also lands here, deferred until the bot has finished
    saying goodbye (the tool result comes back long before the words do).
    """

    def __init__(self, config: Config, get_ws, is_delegating=None) -> None:
        super().__init__()
        self._config = config
        self._get_ws = get_ws
        # Returns True while a delegated backend response is in flight.
        self._is_delegating = is_delegating or (lambda: False)
        self._connected = False
        self._connect_time = 0.0
        self._quiet_since = 0.0  # when the bot last stopped speaking
        self._bot_speaking = False
        self._user_speaking = False
        self._reply_pending = False  # question committed, reply not started
        self._reply_pending_since = 0.0
        self._bot_spoke = False  # any reply at all this session
        self._turns = 0
        self._close_requested = False  # end_conversation fired
        self._watch: asyncio.Task | None = None
        self._gen = 0
        self.on_close = None  # async callable() set by the server

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._bot_spoke = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._reply_pending = False
            self._quiet_since = asyncio.get_running_loop().time()
            if self._close_requested:
                # The goodbye has now actually been spoken.
                await self._close("end_conversation")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, FunctionCallInProgressFrame):
            # A tool is running on the user's behalf: the reply is owed from
            # now, under the same grace as any other, so the session neither
            # hangs up mid-action nor waits out a slow tool on the meter.
            self._reply_pending = True
            self._reply_pending_since = asyncio.get_running_loop().time()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._turns += 1
            # A reply is owed but has not started. Nobody is speaking during
            # this gap, and it is long: the model may hand off to the backend
            # and think for seconds before its first word. Without this flag
            # the watcher would judge the gap against a _quiet_since from
            # before the question and hang up on the user mid-thought, or
            # close on the turn budget the instant the last question landed.
            self._reply_pending = True
            self._reply_pending_since = asyncio.get_running_loop().time()
        elif isinstance(frame, (CancelFrame, EndFrame)) and self._watch is not None:
            task, self._watch = self._watch, None
            await self.cancel_task(task)
        await self.push_frame(frame, direction)

    def request_close(self) -> None:
        """end_conversation: hang up once the bot stops speaking."""
        self._close_requested = True

    def on_prestart_replayed(self) -> None:
        """The pre-start replay has reached the model: the first-reply clock
        starts now. The hard cap still runs from the connect."""
        if self._reply_pending and not self._bot_spoke:
            self._reply_pending_since = asyncio.get_running_loop().time()

    def on_device_connect(self) -> None:
        loop = asyncio.get_running_loop()
        self._connected = True
        self._connect_time = loop.time()
        self._quiet_since = loop.time()
        self._bot_speaking = False
        self._user_speaking = False
        self._bot_spoke = False
        # The device connected because the wake word fired, so a question is
        # about to be asked and a reply is owed. Start the session in that
        # state rather than on the follow-up clock: the clock ran from the
        # chime and expired at 6s with "0 turns", which on this device was
        # while the question was still being spoken or the backend was still
        # fetching the answer (every wake on 2026-09-13/14 died that way).
        # Zero turns is the normal count here: the user-turn frames this
        # counter relies on are not emitted for the device's audio.
        self._reply_pending = True
        self._reply_pending_since = loop.time()
        self._turns = 0
        self._close_requested = False
        self._gen += 1
        if self._watch is not None:
            task, self._watch = self._watch, None
            asyncio.create_task(self.cancel_task(task))
        self._watch = self.create_task(self._watch_loop())

    async def on_device_disconnect(self) -> None:
        self._connected = False
        if self._watch is not None:
            task, self._watch = self._watch, None
            await self.cancel_task(task)

    async def _watch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        window = self._config.followup_window_seconds
        budget = self._config.max_turns_per_wake
        cap = self._config.max_live_session_seconds
        my_gen = self._gen
        while True:
            await asyncio.sleep(0.5)
            if self._gen != my_gen or not self._connected:
                return
            now = loop.time()
            # The fuse ignores every other piece of state on purpose.
            if cap > 0 and now - self._connect_time >= cap:
                await self._close(f"hard cap reached ({cap}s)")
                return
            if self._bot_speaking or self._user_speaking:
                continue
            if self._is_delegating():
                # The model paused mid-answer to let the backend work, and it
                # will speak again when the result lands. Treating that pause
                # as the end of the reply is what truncated answers
                # ("humans live on" instead of "humans live on Earth"), so
                # hold the window open and restart it from the moment the
                # backend finishes.
                self._quiet_since = now
                continue
            if self._reply_pending:
                # Fail-open: a reply that never arrives (a hung tool, a
                # response the API dropped without telling us) must not hold
                # the meter open until the hard cap. Well above any real
                # delegation round trip, which measured a few seconds.
                if now - self._reply_pending_since <= _REPLY_OVERDUE_SECONDS:
                    continue
                if not self._bot_spoke:
                    # Nothing was ever answered: a false wake, or a question
                    # the model declined. Do not also run out a follow-up
                    # window on top; every second is billed.
                    await self._close(f"no reply within {_REPLY_OVERDUE_SECONDS:.0f}s of wake")
                    return
                logger.warning(
                    "live hygiene: reply overdue >%.0fs; releasing the pending hold",
                    _REPLY_OVERDUE_SECONDS,
                )
                self._reply_pending = False
                self._quiet_since = now
            if budget > 0 and self._turns >= budget:
                await self._close(f"turn budget reached ({self._turns}/{budget})")
                return
            if window > 0 and now - self._quiet_since >= window:
                await self._close(f"follow-up window expired ({self._turns} turns)")
                return

    async def _close(self, reason: str) -> None:
        ws = self._get_ws()
        if ws is not None:
            try:
                await ws.send('{"type":"disconnect"}')
            except Exception:
                logger.exception("live hygiene: failed to signal device")
        logger.info("live hygiene: %s; disconnecting device", reason)
        if self.on_close is not None:
            await self.on_close()


class _Health:
    """What a watchdog needs to know, without starting a billed session.

    A websocket probe is what a watchdog reaches for, but the broker answers
    one by opening a session: on a per-minute meter that is a real cost every
    few minutes, forever. A port check is free and proves nothing, which is
    the hole this fills: the failure actually seen on this deployment was the
    pipeline rebuilding twice a second behind a port that stayed open
    (2026-09-14), and that shows up here as a rebuild count rather than as a
    refused connection.
    """

    #: Rebuilds inside the window past which the broker is judged wedged.
    _REBUILD_LIMIT = 5
    _REBUILD_WINDOW_SECONDS = 300.0

    def __init__(self) -> None:
        self.started = time.time()
        self.serving = False
        self.last_device = 0.0
        self.last_error = ""
        self._rebuilds: deque[float] = deque()

    def note_rebuild(self, error: str) -> None:
        self._rebuilds.append(time.time())
        self.last_error = error[:200]
        self._expire()

    def _expire(self) -> None:
        now = time.time()
        while self._rebuilds and now - self._rebuilds[0] > self._REBUILD_WINDOW_SECONDS:
            self._rebuilds.popleft()

    def snapshot(self) -> dict:
        self._expire()
        looping = len(self._rebuilds) >= self._REBUILD_LIMIT
        return {
            "ok": bool(self.serving and not looping),
            "engine": "live",
            "serving": self.serving,
            "uptime_seconds": round(time.time() - self.started),
            "recent_rebuilds": len(self._rebuilds),
            "seconds_since_last_wake": (
                round(time.time() - self.last_device) if self.last_device else None
            ),
            "last_error": self.last_error,
        }


async def _serve_health(config: Config, health: _Health) -> None:
    """Serve the snapshot on /health. Never touches the voice path."""
    from aiohttp import web

    async def handler(_request):
        snap = health.snapshot()
        return web.json_response(snap, status=200 if snap["ok"] else 503)

    app = web.Application()
    app.router.add_get("/health", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", config.live_health_port).start()
    logger.info("Health endpoint on http://0.0.0.0:%d/health", config.live_health_port)


async def run_live(config: Config) -> None:
    """Serve forever on the Live engine, one billed session per wake."""
    if not config.ha_control_enabled:
        logger.info("Home Assistant control disabled (HA_MCP_URL/HA_TOKEN unset)")

    logger.info("Live broker listening on ws://%s:%d", config.ws_host, config.ws_port)
    health = _Health()
    if config.live_health_port:
        try:
            await _serve_health(config, health)
        except OSError as exc:
            logger.warning("Health endpoint unavailable: %s", exc)
    while True:
        # A fresh MCP client per attempt: the pipeline closes the one it was
        # given when it ends, and rebuilding on the closed client raised
        # "MCPClient is not connected" twice a second, forever (2026-09-14).
        mcp = None
        try:
            if config.ha_control_enabled:
                mcp = await mcp_client.connect(config.ha_mcp_url, config.ha_token)
                # Pipecat's MCPClient is a managed connection now: constructing
                # it is not enough, the SSE session has to be started before
                # tools exist.
                await mcp.start()
            await _serve_live(config, mcp, health)
        except Exception as exc:
            logger.exception("Live session crashed; rebuilding")
            health.note_rebuild(f"{type(exc).__name__}: {exc}")
        finally:
            health.serving = False
            if mcp is not None:
                try:
                    await mcp.close()
                except Exception as exc:  # noqa: BLE001 - already torn down is fine
                    logger.debug("MCP client close: %s", exc)
        await asyncio.sleep(0.5)  # let the socket fully release before rebind


async def _serve_live(config: Config, mcp, health: _Health | None = None) -> None:
    service = build_live_agent(config)
    tools = await build_live_tools(mcp)

    serializer = RawPCMSerializer()
    transport = WebsocketServerTransport(
        host=config.ws_host,
        port=config.ws_port,
        params=WebsocketServerParams(
            serializer=serializer,
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )
    def get_ws():
        return getattr(transport.input(), "_websocket", None)

    hygiene = _LiveHygiene(config, get_ws, lambda: service.delegation_in_flight)
    service.on_prestart_replayed = hygiene.on_prestart_replayed

    async def _get_weather(params):
        await params.result_callback(await asyncio.to_thread(_fetch_weather, config))

    async def _play_music(params):
        args = params.arguments or {}
        msg = await asyncio.to_thread(
            _start_music, config, args.get("query", ""), args.get("speaker")
        )
        await params.result_callback(msg)

    async def _end_conversation(params):
        # Only arms the close: the model has not said goodbye yet, and
        # hanging up now would cut the word off mid-air.
        await params.result_callback("Okay, goodbye!")
        hygiene.request_close()

    service.register_function("get_weather", _get_weather)
    service.register_function("play_music", _play_music)
    service.register_function("end_conversation", _end_conversation)
    if mcp is not None:
        await mcp.register_tools_schema(tools, service)

    context = LLMContext(tools=tools)
    aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [
            transport.input(),
            aggregator.user(),
            service,
            # Between the service and the output transport: the output
            # transport pushes BotStarted/StoppedSpeakingFrame upstream from
            # here, which is how the meter knows the speaker went quiet.
            hygiene,
            # Last before the wire: the device must never receive the model's
            # idle silence, or its firmware mutes the mic (see the class).
            _OutputSilenceFilter(output_hold=service.output_hold),
            transport.output(),
            aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=None,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    session_lock = asyncio.Lock()

    async def _end_session() -> None:
        # Closing waits on the server draining in-flight work, so a fast
        # re-wake must queue behind it rather than race it.
        async with session_lock:
            await service.end_live_session()

    hygiene.on_close = _end_session

    @transport.event_handler("on_client_connected")
    async def _on_connect(_transport, client):
        logger.info("Device connected: %s", getattr(client, "remote_address", client))
        if health is not None:
            health.last_device = time.time()
        # Each wake starts clean. The aggregator keeps every turn of every
        # earlier wake, and the live model, seeded with them, answers them
        # again before hearing anything: a 7 AM wake got last night's "It's
        # 10:28 PM" and a stale trivia answer, and with the model already
        # talking the transcription fallback stood down (2026-09-15).
        # Memory across wakes wants a dated, tool-result-free transcript,
        # not the raw history.
        async with session_lock:
            # Under the lock: end_live_session closes the previous wake's
            # turns, and those closing frames append to this same context.
            context.set_messages([])
            await service.begin_live_session()
        hygiene.on_device_connect()
        # Seeds the context, which is what configures and starts the session.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, client, *args):
        current = get_ws()
        if current is not None and current is not client:
            logger.info("Stale connection closed; device still connected")
            return
        logger.info("Device disconnected")
        await hygiene.on_device_disconnect()
        await _end_session()

    if health is not None:
        health.serving = True
    refresh = config.live_socket_refresh_seconds
    keepalive = (
        asyncio.create_task(_keep_socket_warm(service, refresh)) if refresh > 0 else None
    )
    try:
        await runner.run()
    finally:
        if keepalive is not None:
            keepalive.cancel()


async def _keep_socket_warm(service, every: float) -> None:
    """Replace the idle OpenAI connection before OpenAI drops it."""
    while True:
        await asyncio.sleep(every)
        try:
            await service.refresh_idle_socket()
        except Exception as exc:  # noqa: BLE001 - a failed refresh must not end the pipeline
            logger.warning("Live: idle connection refresh failed: %s", exc)
