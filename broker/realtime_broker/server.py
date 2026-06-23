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
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
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

    Why model playback instead of using BotStarted/StoppedSpeakingFrame: the
    output transport paces audio to the device at 2x realtime and BotStopped
    fires when the SEND queue drains — for an N-second reply the device still
    holds up to N/2 seconds in its own buffer at that moment. Summing the
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

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            now = asyncio.get_running_loop().time()
            duration = len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)
            self._playback_end = max(self._playback_end, now) + duration
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
        await self._service.send_client_event(
            oai_events.SessionUpdateEvent(
                session=oai_events.SessionProperties(
                    audio=oai_events.AudioConfiguration(
                        input=build_audio_input(self._config, threshold)
                    )
                )
            )
        )
        logger.info("adaptive VAD: threshold -> %s", threshold)


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
            f"{base}/api/states/weather.forecast_home",
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

    service.register_function("get_weather", _get_weather)
    service.register_function("end_conversation", _end_conversation)
    service.register_function("play_music", _play_music)

    gate = _BotPlaybackGate(
        service, config, lambda: getattr(transport.input(), "_websocket", None)
    )
    aggregator = LLMContextAggregatorPair(LLMContext())
    pipeline = Pipeline(
        [
            transport.input(),
            aggregator.user(),
            # The realtime service pushes user TranscriptionFrames UPSTREAM
            # (Pipecat >= 0.0.92); the logger must sit between the aggregator
            # and the service so it can intercept (and consume) them before
            # the aggregator reacts to them — see _UserTranscriptLogger.
            _UserTranscriptLogger(),
            service,
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

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnect(_transport, client, *args):  # noqa: ANN001
        nonlocal device_connected, idle_since
        device_connected = False
        idle_since = loop.time()
        logger.info("Device disconnected")

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
