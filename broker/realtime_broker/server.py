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
    runner_task = asyncio.create_task(PipelineRunner().run(task))
    try:
        # Whichever comes first: the session dies on its own, or it ages out.
        await asyncio.wait_for(asyncio.shield(runner_task), config.max_session_seconds)
        logger.warning("Pipeline ended on its own; rebuilding session")
    except asyncio.TimeoutError:
        logger.info("Session reached max age (%ds); rotating", config.max_session_seconds)
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
