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

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
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

    def __init__(self, config: Config, get_ws) -> None:
        super().__init__()
        self._config = config
        self._get_ws = get_ws
        self._connected = False
        self._connect_time = 0.0
        self._quiet_since = 0.0  # when the bot last stopped speaking
        self._bot_speaking = False
        self._user_speaking = False
        self._turns = 0
        self._close_requested = False  # end_conversation fired
        self._watch: asyncio.Task | None = None
        self._gen = 0
        self.on_close = None  # async callable() set by the server

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._quiet_since = asyncio.get_running_loop().time()
            if self._close_requested:
                # The goodbye has now actually been spoken.
                await self._close("end_conversation")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._turns += 1
        elif isinstance(frame, (CancelFrame, EndFrame)) and self._watch is not None:
            task, self._watch = self._watch, None
            await self.cancel_task(task)
        await self.push_frame(frame, direction)

    def request_close(self) -> None:
        """end_conversation: hang up once the bot stops speaking."""
        self._close_requested = True

    def on_device_connect(self) -> None:
        loop = asyncio.get_running_loop()
        self._connected = True
        self._connect_time = loop.time()
        self._quiet_since = loop.time()
        self._bot_speaking = False
        self._user_speaking = False
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


async def run_live(config: Config) -> None:
    """Serve forever on the Live engine, one billed session per wake."""
    mcp = None
    if config.ha_control_enabled:
        mcp = await mcp_client.connect(config.ha_mcp_url, config.ha_token)
        # Pipecat's MCPClient is a managed connection now: constructing it is
        # not enough, the SSE session has to be started before tools exist.
        await mcp.start()
    else:
        logger.info("Home Assistant control disabled (HA_MCP_URL/HA_TOKEN unset)")

    logger.info("Live broker listening on ws://%s:%d", config.ws_host, config.ws_port)
    while True:
        try:
            await _serve_live(config, mcp)
        except Exception:
            logger.exception("Live session crashed; rebuilding")
        await asyncio.sleep(0.5)  # let the socket fully release before rebind


async def _serve_live(config: Config, mcp) -> None:
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

    hygiene = _LiveHygiene(config, get_ws)

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
        async with session_lock:
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

    await runner.run()
