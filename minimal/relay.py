"""Voice PE puck <-> GPT-Live with Home Assistant tools, and nothing else.

The whole assistant is gpt-live-1: it hears, decides, delegates to its
backend, and speaks. This file only does what the puck cannot do itself:

- carry the puck's audio to Live and Live's audio back (PCM16, 24 kHz);
- run the backend's tool calls against Home Assistant over MCP and hand the
  results back to Live;
- end the conversation, because the puck never hangs up on its own: after
  FOLLOWUP_S of quiet once the assistant has spoken, and at HARD_CAP_S
  whatever happens, so a stuck session cannot keep billing.

No transcript checks, no output holds, no retries, no memory: the eval
(evals/gate.py) showed each of those made the broker slower and less
accurate than Live on its own.

    broker/.venv-live/bin/python minimal/relay.py     # listens on :8767
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import copy
import json
import logging
import os
import struct
import time
import urllib.request
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv(Path(__file__).resolve().parent.parent / "broker" / ".env")

PORT = int(os.environ.get("MINIMAL_PORT", "8767"))
LIVE_URL = os.environ.get("LIVE_URL", "wss://api.openai.com/v1/live/sessions")
LIVE_MODEL = os.environ.get("LIVE_MODEL", "gpt-live-1")
BACKEND_MODEL = os.environ.get("LIVE_BACKEND_MODEL", "gpt-5.4-mini")
VOICE = os.environ.get("LIVE_VOICE", "cedar")
API_KEY = os.environ["OPENAI_API_KEY"]
HA_MCP_URL = os.environ["HA_MCP_URL"]
HA_TOKEN = os.environ["HA_TOKEN"]
RATE = 24000
# Seconds of quiet after the assistant speaks before the conversation ends.
# 6 s closed on a re-ask that came 7 s later (2026-09-29).
FOLLOWUP_S = 10.0
# Mic level that counts as someone talking; the quietest real question on
# tape (far across the room) peaks around 300.
VOICE_RMS = 250.0
FIRST_REPLY_S = 15.0
HARD_CAP_S = 180.0
# The production broker's proven values for telling Live's speech from its
# silence (broker/realtime_broker/live_server.py).
SILENCE_RMS = 50.0
SILENCE_HOLD_S = 0.8
# How long Live's audio is held before it goes to the puck: the slack that
# absorbs network jitter so playback is not choppy.
PREROLL_S = 0.4
# The handoff. Measured on Jay's own puck recordings with no relay at all
# (evals/gate.py --voice jay): gpt-live-1 hears his longer requests 10/10
# and his short ones ("What time is it?", "Put Netflix on.") 0/15, at any
# level, with or without noise. It exposes no input setting. The same audio
# reads correctly through gpt-4o-transcribe every time, so when the mic
# clearly carried speech and Live produced no transcript, that text goes
# to Live's backend as the user's words. Live does everything else.
HANDOFF_MODEL = "gpt-4o-transcribe"
HANDOFF_QUIET_S = 0.7  # this much quiet ends an utterance
HANDOFF_MIN_SPEECH_S = 0.3  # shorter bursts are noise
# Mic level that opens an utterance for the handoff. Lower than VOICE_RMS:
# Jay's quiet "Put Netflix on." from across the room sits at 150-260, the
# room's floor at 30-60. Anything opened here is still judged by the
# transcriber before Live sees a word of it.
UTTERANCE_RMS = 120.0
HANDOFF_GRACE_S = 1.0  # how long after the utterance Live gets to transcribe it
HANDOFF_LEAD_FRAMES = 20  # 0.4 s kept from before speech opened: the words said right after the chime
HANDOFF_MEAN_LOGPROB = -0.15  # below this the transcriber was guessing
HANDOFF_MIN_LOGPROB = -0.5

INSTRUCTIONS = os.environ.get(
    "INSTRUCTIONS",
    "You are Atriensis, the household steward for this smart home. Always respond in English. "
    "Be concise, warm, and natural, like a capable, unflappable butler. You can control the home "
    "with the available tools; when asked to do something, just do it and confirm in one short sentence.",
) + (
    # Without this Live answered "what time is it" at once with an invented
    # time ("11:34 p.m.", "3:45") and never asked the backend (2026-09-28).
    " You have no clock: the time and date always come from your backend, never from a guess."
)
BACKEND_INSTRUCTIONS = (
    "You are the backend of a home voice assistant. Each message is the recent voice conversation "
    "as a transcript; work out what is being asked and do it. Use the Home Assistant tools to "
    "control the home and read live state. Reply with the verified result in one short "
    "conversational sentence the assistant can say aloud, with no Markdown and no JSON, and never "
    "claim an action completed without a tool result confirming it."
)

log = logging.getLogger("minimal")
# MINIMAL_TAPE=<dir> saves what the puck sent on each wake as a WAV: the one
# way to tell a bad microphone from a bad relay when Live mishears.
TAPE_DIR = os.environ.get("MINIMAL_TAPE", "")


def save_tape(pcm: bytearray) -> None:
    import wave

    path = Path(TAPE_DIR) / f"puck-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(pcm))
    log.info("tape: %.1fs of puck audio in %s", len(pcm) / 2 / RATE, path)


def wav(pcm: bytes) -> bytes:
    head = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    head += struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
    return head + b"data" + struct.pack("<I", len(pcm)) + pcm


def transcribe(pcm: bytes) -> tuple[str, float, float]:
    """The words, with the model's mean and lowest token log-probability.

    Every transcriber writes a sentence when given noise (16 s of room tone
    came back as "Honestly, Carl, if you give a shit."). The log-probs tell
    it apart: Jay's real requests score mean >= -0.05 and lowest >= -0.16,
    that invented one -0.27 and -0.94.
    """
    boundary = uuid.uuid4().hex
    body = b""
    fields = (("model", HANDOFF_MODEL), ("language", "en"), ("response_format", "json"), ("include[]", "logprobs"))
    for name, value in fields:
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav(pcm) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        result = json.loads(r.read())
    logprobs = [t["logprob"] for t in result.get("logprobs") or []]
    if not logprobs:
        return result["text"].strip(), 0.0, 0.0
    return result["text"].strip(), sum(logprobs) / len(logprobs), min(logprobs)


def rms(pcm: bytes) -> float:
    samples = memoryview(pcm).cast("h") if len(pcm) % 2 == 0 else memoryview(pcm[:-1]).cast("h")
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 if len(samples) else 0.0


def live_tool(tool) -> dict:
    """An MCP tool as a Live backend function: optional fields become nullable, all listed."""
    params = copy.deepcopy(tool.inputSchema or {"type": "object", "properties": {}})
    props = params.setdefault("properties", {})
    required = set(params.get("required", []))
    for name, schema in list(props.items()):
        if name not in required:
            props[name] = {"anyOf": [schema, {"type": "null"}]}
    params["required"] = list(props)
    params["additionalProperties"] = False
    return {"type": "function", "name": tool.name, "description": tool.description or tool.name,
            "parameters": params}


def session_start(tools: list[dict]) -> dict:
    return {
        "type": "session.start",
        "session": {
            "model": LIVE_MODEL,
            "instructions": INSTRUCTIONS,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": VOICE}},
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": BACKEND_MODEL,
                    "instructions": BACKEND_INSTRUCTIONS,
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "reasoning": {"effort": "none"},
                    "text": {"verbosity": "low"},
                },
            },
        },
    }


class Conversation:
    """One wake: from the puck connecting to the relay closing it."""

    def __init__(self, puck: web.WebSocketResponse, live, ha: ClientSession) -> None:
        self.puck, self.live, self.ha = puck, live, ha
        self.started = time.monotonic()
        self.heard_until = self.started  # when the puck finishes playing what it has been sent
        self.spoke = False
        self.pending: dict[str, list[asyncio.Task]] = {}
        self.busy = 0  # tool calls in flight
        self.tape = bytearray()  # everything the puck sent, for MINIMAL_TAPE
        self.last_loud = 0.0  # when Live's audio last carried speech
        self.outgoing: collections.deque[tuple[float, bytes]] = collections.deque()  # (send at, pcm)
        # The handoff's view of the mic: the utterance being spoken, and when
        # Live last showed it had heard anything.
        self.recent: collections.deque[bytes] = collections.deque(maxlen=HANDOFF_LEAD_FRAMES)
        self.utterance: bytearray | None = None
        self.utterance_started = 0.0
        self.mic_s = 0.0  # seconds of mic audio received so far
        self.voiced_s = 0.0
        self.quiet_s = 0.0
        self.live_heard_at = 0.0
        self.backend_started_at = 0.0  # Live delegated on its own
        self.answer_words = 0  # words Live has spoken since the utterance opened
        self.handoffs: set[asyncio.Task] = set()

    async def run(self) -> None:
        tasks = [asyncio.create_task(c)
                 for c in (self.from_puck(), self.from_live(), self.speaker(), self.clock())]
        _, rest = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in rest:
            t.cancel()
        await asyncio.gather(*rest, return_exceptions=True)
        # Whatever is still queued is the end of the last sentence.
        for _, pcm in self.outgoing:
            with contextlib.suppress(ConnectionError):
                await self.puck.send_bytes(pcm)

    async def speaker(self) -> None:
        """Play Live's audio to the puck PREROLL_S late, at Live's own pace.

        Live streams speech in real time, so a network hiccup on the way
        in became a hole in the puck's playback: the voice came out choppy
        (2026-09-30). Holding each chunk for PREROLL_S before sending gives
        that much slack to absorb, and the puck plays it back seamlessly.
        """
        while True:
            if not self.outgoing:
                await asyncio.sleep(0.01)
                continue
            due, pcm = self.outgoing[0]
            if (wait := due - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            self.outgoing.popleft()
            await self.puck.send_bytes(pcm)

    async def from_puck(self) -> None:
        async for msg in self.puck:
            if msg.type == aiohttp.WSMsgType.BINARY:
                self.tape.extend(msg.data)
                # The user is talking even when Live does not transcribe it.
                # Keyed only on Live's transcript, the relay hung up on "dude,
                # I asked you a question" mid-sentence (2026-09-29).
                level = rms(msg.data)
                if level >= VOICE_RMS:
                    self.heard_until = max(self.heard_until, time.monotonic())
                self.track_utterance(msg.data, level >= UTTERANCE_RMS)
                await self.live.send_json({"type": "session.input_audio.append",
                                           "audio": base64.b64encode(msg.data).decode()})
            elif msg.type == aiohttp.WSMsgType.TEXT:
                log.info("puck says %s", msg.data)
        log.info("puck hung up")

    async def from_live(self) -> None:
        async for msg in self.live:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            kind = event.get("type")
            if kind != "session.output_audio.delta":
                inner = (event.get("event") or {}).get("type", "")
                log.debug("live event %s %s", kind, inner)
            if kind == "session.output_audio.delta":
                pcm = base64.b64decode(event["delta"])
                now = time.monotonic()
                # Live streams silence without end once it has spoken. The
                # puck keeps its mic muted while anything plays, so relaying
                # that silence left it deaf after the first answer and kept
                # every wake open to the hard cap (2026-09-28). Quiet frames
                # just after speech are kept: they are commas and breaths.
                if rms(pcm) < SILENCE_RMS and now - self.last_loud > SILENCE_HOLD_S:
                    continue
                if rms(pcm) >= SILENCE_RMS:
                    self.last_loud = now
                self.heard_until = max(self.heard_until, now + PREROLL_S) + len(pcm) / 2 / RATE
                self.spoke = True
                self.outgoing.append((now + PREROLL_S, pcm))
            elif kind == "session.input_transcript.delta":
                self.heard_until = self.live_heard_at = max(self.heard_until, time.monotonic())
                log.info("heard: %s", event.get("delta", "").strip())
            elif kind == "session.output_transcript.delta":
                self.answer_words += len(event.get("delta", "").split())
                log.info("said: %s", event.get("delta", "").strip())
            elif kind == "response.event":
                await self.on_backend(event.get("event") or {})
            elif kind in ("error", "session.closed"):
                log.info("live %s: %s", kind, json.dumps(event)[:300])
                if kind == "session.closed":
                    return

    def track_utterance(self, pcm: bytes, loud: bool) -> None:
        """Cut the mic into utterances; each one that ends gets a handoff check."""
        now = time.monotonic()
        secs = len(pcm) / 2 / RATE
        self.mic_s += secs
        self.recent.append(pcm)
        if self.utterance is None:
            # The first half second of mic audio is the wake chime, not the
            # user. Counted in audio, not wall time: the puck streams from
            # the wake while the relay is still connecting, so the opening
            # second arrives in one burst.
            if loud and self.mic_s > 0.6:
                self.utterance = bytearray(b"".join(self.recent))
                self.utterance_started = now
                self.voiced_s = self.quiet_s = 0.0
                self.answer_words = 0
            return
        self.utterance.extend(pcm)
        if loud:
            self.voiced_s += secs
            self.quiet_s = 0.0
            return
        self.quiet_s += secs
        if self.quiet_s >= HANDOFF_QUIET_S:
            done, started, voiced = bytes(self.utterance), self.utterance_started, self.voiced_s
            self.utterance = None
            if voiced >= HANDOFF_MIN_SPEECH_S:
                # Kept in self.handoffs: an unreferenced task can be garbage
                # collected mid-await, and this one sleeps first.
                task = asyncio.create_task(self.handoff(done, started))
                self.handoffs.add(task)
                task.add_done_callback(self.handoffs.discard)

    async def handoff(self, pcm: bytes, started: float) -> None:
        await asyncio.sleep(HANDOFF_GRACE_S)
        if self.live_heard_at > started:
            return  # Live heard it itself
        if self.backend_started_at > started:
            # Live made something of the audio and delegated on its own. Let
            # that finish: if it produced a real answer the words are not
            # needed, and a second request on top gave "How may I help...
            # on Thursday" (2026-09-30). If it came back with nothing to
            # say, hand off as usual.
            for _ in range(20):
                await asyncio.sleep(0.2)
                if not self.pending and not self.busy:
                    break
            await asyncio.sleep(1.0)
            if self.answer_words >= 4 or self.live_heard_at > started:
                return
        try:
            text, mean_lp, min_lp = await asyncio.to_thread(transcribe, pcm)
        except Exception as exc:  # noqa: BLE001 - a lost handoff is a lost turn, not a crash
            log.info("handoff: transcription failed: %s", exc)
            return
        if self.live_heard_at > started:
            return
        if len(text.split()) < 2 or mean_lp < HANDOFF_MEAN_LOGPROB or min_lp < HANDOFF_MIN_LOGPROB:
            log.info("handoff: not speech (%r, mean %.2f, min %.2f)", text, mean_lp, min_lp)
            return
        log.info("handoff: Live heard nothing; the user said %r", text)
        self.heard_until = max(self.heard_until, time.monotonic() + 3)
        # Straight to the backend as the user's words, the way Live's own
        # transcript would have gone: it acts and Live speaks the result.
        await self.live.send_json({
            "type": "response.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        })
        await self.live.send_json({"type": "response.create"})

    async def on_backend(self, inner: dict) -> None:
        kind = inner.get("type")
        item = inner.get("item") or {}
        if kind == "response.created":
            self.backend_started_at = time.monotonic()
        if kind == "response.output_item.done" and item.get("type") == "function_call" \
                and item.get("status") == "completed":
            rid = inner.get("response_id") or ""
            self.pending.setdefault(rid, []).append(asyncio.create_task(self.call(item)))
        elif kind == "response.completed":
            rid = (inner.get("response") or {}).get("id", "")
            calls = self.pending.pop(rid, None) or self.pending.pop("", None)
            if not calls:
                return
            for call_id, output in await asyncio.gather(*calls):
                await self.live.send_json({
                    "type": "response.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id,
                             "output": json.dumps(output, ensure_ascii=False)},
                })
            await self.live.send_json({"type": "response.create"})

    async def call(self, item: dict) -> tuple[str, dict]:
        name = item.get("name", "")
        args = {k: v for k, v in json.loads(item.get("arguments") or "{}").items() if v is not None}
        self.busy += 1
        try:
            result = await self.ha.call_tool(name, args)
            text = "".join(getattr(c, "text", "") for c in result.content)
            output = {"error": text} if result.isError else {"result": text}
        except Exception as exc:  # noqa: BLE001 - a failed call is reported to the model, not raised
            output = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.busy -= 1
            self.heard_until = max(self.heard_until, time.monotonic())
        log.info("tool %s(%s) -> %s", name, json.dumps(args), json.dumps(output)[:200])
        return item.get("call_id", ""), output

    async def clock(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if now - self.started > HARD_CAP_S:
                log.info("hard cap reached")
                return
            quiet = now - self.heard_until
            if not self.busy and not self.pending and quiet > (FOLLOWUP_S if self.spoke else FIRST_REPLY_S):
                log.info("quiet for %.0fs, ending the conversation", quiet)
                return


async def handle(request: web.Request) -> web.WebSocketResponse:
    puck = web.WebSocketResponse(max_msg_size=0)
    await puck.prepare(request)
    log.info("wake from %s", request.remote)
    try:
        async with sse_client(HA_MCP_URL, headers={"Authorization": f"Bearer {HA_TOKEN}"}) as (r, w), \
                ClientSession(r, w) as ha:
            await ha.initialize()
            tools = [live_tool(t) for t in (await ha.list_tools()).tools]
            async with aiohttp.ClientSession() as http, http.ws_connect(
                LIVE_URL, headers={"Authorization": f"Bearer {API_KEY}"}, heartbeat=20, max_msg_size=0
            ) as live:
                await live.send_json(session_start(tools))
                first = await live.receive_json()
                if first.get("type") != "session.started":
                    log.error("live refused the session: %s", json.dumps(first)[:400])
                    return puck
                conversation = Conversation(puck, live, ha)
                await conversation.run()
                await live.send_json({"type": "session.close"})
                if TAPE_DIR:
                    save_tape(conversation.tape)
    except Exception:
        log.exception("conversation failed")
    finally:
        # The firmware only goes back to idle on an explicit disconnect
        # message. A bare close reads to it as a dropped link: it tries to
        # reconnect, and the next wake chimes and never connects (2026-09-28).
        if not puck.closed:
            with contextlib.suppress(ConnectionError):
                await puck.send_str('{"type":"disconnect"}')
        await puck.close()
        log.info("conversation over")
    return puck


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.setLevel(os.environ.get("MINIMAL_LOG", "INFO"))
    app = web.Application()
    app.router.add_get("/{tail:.*}", handle)
    web.run_app(app, host="0.0.0.0", port=PORT, print=lambda *_: log.info("listening on :%d", PORT))


if __name__ == "__main__":
    main()
