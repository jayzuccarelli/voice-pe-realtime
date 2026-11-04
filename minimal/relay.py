"""The "chain" relay: wake word -> transcriber -> text model with Home Assistant tools -> speech.

Two relays have served the puck. "live" (minimal/relay.py up to commit
e5b2559, kept beside the deploy as relay.py.bak_live_e5b2559) put
gpt-live-1 in charge of hearing, deciding and speaking. "chain", this file,
is the plain chained design from OpenAI's voice-agent guide, chosen on
2026-11-04 after the live version said "Sure, I'll take care of that" to a
Spotify request and did nothing. See minimal/VERSIONS.md.

The puck streams its microphone. The relay cuts that into utterances, has a
transcriber read each one, hands the words to a text model that can act only
through Home Assistant's tools, and speaks the outcome:

- the confirmation of an action is built from the tool result, never from
  the model's words, so "the lights are off" cannot be said unless a tool
  reported it (gpt-live-1 said it without acting on 4 of 14 commands,
  2026-10-20..29, and "the front door is now locked" to "actually nothing");
- "On it." plays the moment a command is read, before the model is asked,
  so the acknowledgement comes before the action instead of after it;
- an answer that needs no action is the model's text, spoken as is.

Until 2026-10-30 gpt-live-1 was the ear and the brain. On this puck it
produced a transcript for about one utterance in five at any length (of
twelve identical "What time is it?" wakes it heard one), so the transcriber
was already carrying the conversation as a fallback. This file makes that
the design. The puck's firmware closes the mic while audio plays, so
nothing is lost on interruptions: there were none to lose.

The relay still does what the puck cannot: keep the puck on the line
through a slow tool, and hang up after FOLLOWUP_S of quiet or at
HARD_CAP_S whatever happens.

    broker/.venv-live/bin/python minimal/relay.py     # listens on :8767
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import re
import struct
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv(Path(__file__).resolve().parent.parent / "broker" / ".env")

VERSION = "chain"
PORT = int(os.environ.get("MINIMAL_PORT", "8767"))
BRAIN_MODEL = os.environ.get("LIVE_BACKEND_MODEL", "gpt-5.4-mini")
# Two transcribers read every utterance and the better-scoring English
# reading wins. One alone is a coin flip on a short far-field clip: the same
# "Turn the living room lights on." came back as Afrikaans, then as
# 'livingroom lights on.' at mean -0.80, on consecutive replays (2026-10-30).
# gpt-transcribe (July 2026) is the more accurate successor of gpt-4o-transcribe.
STT_MODELS = tuple(os.environ.get("STT_MODELS", "gpt-transcribe,gpt-4o-mini-transcribe").split(","))
TTS_MODEL = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")
VOICE = os.environ.get("LIVE_VOICE", "cedar")
TZ = ZoneInfo(os.environ.get("HOUSE_TZ", "America/New_York"))
API_KEY = os.environ["OPENAI_API_KEY"]
HA_MCP_URL = os.environ["HA_MCP_URL"]
HA_TOKEN = os.environ["HA_TOKEN"]
RATE = 24000
CHUNK = RATE // 10 * 2  # 100 ms of audio to the puck; its playback queue holds ten
FOLLOWUP_S = 10.0
FIRST_REPLY_S = 15.0
HARD_CAP_S = 180.0
# How far ahead of its playback the puck is kept: the slack that absorbs
# network jitter so speech is not choppy. Audio is paced at real time after
# that; a whole reply sent at once overran the puck's heap (11 KB free
# during playback, 2026-10-19).
PREROLL_S = 0.4
# The cutter, measured in production since 2026-09-27 and unchanged here. A
# quiet "Put Netflix on." from across the room sits at 150-260 RMS, the
# room's floor at 30-60; with the TV on the room sat at 500-800 for a whole
# conversation and a fixed threshold never closed an utterance, so speech
# must also be FLOOR_RATIO above the quietest 100 ms of the last 3 s.
UTTERANCE_RMS = 120.0
FLOOR_RATIO = 2.5
FLOOR_FRAMES = 150  # 3 s
QUIET_S = 0.7  # this much quiet ends an utterance
MIN_SPEECH_S = 0.3  # shorter bursts are noise
LEAD_FRAMES = 20  # 0.4 s kept from before speech opened: the words said right after the chime
MAX_UTTERANCE_S = 8.0  # an utterance this long is read as it stands
# Frame counts are in puck chunks of 20 ms (one microphone callback each;
# the firmware's AUDIO_SEND_INTERVAL_MS = 100 is nominal). Measured by
# replaying tapes at several sizes against production's cut times
# (evals/cutter_replay.py, 2026-10-30): 20 ms matches to the tenth.
# Every transcriber writes a sentence when given noise (16 s of room tone
# came back as "Honestly, Carl, if you give a shit."; a chair scrape as
# 'Schön.'). Mean token log-probability tells them apart: of 33 wakes
# 2026-10-20..29 every real sentence scored mean >= -0.53 and every
# invented one <= -0.59 or came out in another alphabet. The lowest single
# token is NOT used: a long clean command always has one unlikely word in
# it ("Turn Netflix on, living room TV and turn the lights on in the living
# room." was thrown away at min -3.04 while scoring mean -0.30).
GUARD_MIN_WORDS = 2
GUARD_MEAN_LOGPROB = -0.6
GUARD_ASCII = 0.9  # share of letters that must be plain ASCII: language=en is a hint, not a rule
# No vocabulary "prompt" for the transcriber: given one, it returned the
# prompt itself as the transcript of a half-second of room noise, at a
# log-probability that passed the guard, and seven tools fired in reply
# to "the what?" (harness, 2026-10-30).
HOUSE_AREAS = os.environ.get("HOUSE_AREAS", "the living room, the bedroom, the office and the kitchen")
ACK = "On it."
# Only a command gets the acknowledgement; a question gets its answer, room
# talk that slipped through the guard gets nothing.
COMMAND_RE = re.compile(
    r"\b(turn|switch|put|play|pause|resume|set|dim|mute|unmute|stop|start|open|close|lock|unlock|"
    r"netflix|youtube|spotify|lights?|lamp|tv|television|playstation|ps5|volume)\b", re.I)
INSTRUCTIONS = os.environ.get(
    "INSTRUCTIONS",
    "You are Atriensis, the household steward for this smart home. Always respond in English. "
    "Be concise, warm, and natural, like a capable, unflappable butler.",
) + (
    " Each message is one request transcribed from a far-field microphone. The transcriber garbles "
    "the opening words of a command and keeps the end, so the device and the LAST state word are the "
    "request: 'Don't be leaving the lights off', 'Don't put the living room lights off', 'I'm leaving "
    "the lights off' and 'Remember lights off' all mean 'Turn the lights off' (off, never on); "
    "'Fernando PlayStation 5' is 'Turn on the PlayStation 5'; 'Netflix on out there' is 'Put Netflix "
    f"on'. The rooms are {HOUSE_AREAS}; 'living' means the living room, and 'the TV' with no room "
    "means the living room TV, where this microphone is. Whenever a device of this "
    "home is named with on or off, or you are asked to play, put on, open or pause something, act on "
    "it. An app or show named without such a verb is not a request. When the speaker corrects "
    "themselves ('the TV, uh, the light'), only the correction counts. Never act on a device, app or "
    "room that was not named. To turn a TV off or on use HassTurnOff or HassTurnOn on the media "
    "player, not the remote. You can act on the home only through the tools: when asked to do "
    "something, call the tool and write nothing. When asked a question, answer in one short spoken "
    "sentence with no Markdown. If you called no tool, never say that anything was done. If the words "
    "are only the wake word ('Hey Mycroft', 'Minecraft', 'Michael'), or name nothing in the home and "
    "ask you nothing (overheard conversation, a fragment, a TV line), reply with exactly: (nothing)"
)
log = logging.getLogger("minimal")
# MINIMAL_TAPE=<dir> saves what the puck sent on each wake as a WAV: the one
# way to tell a bad microphone from a bad relay when a request is misheard.
TAPE_DIR = os.environ.get("MINIMAL_TAPE", "")
OPENAI = "https://api.openai.com/v1"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


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


def transcribe(pcm: bytes, model: str) -> tuple[str, float]:
    """The words, with the model's mean token log-probability."""
    boundary = uuid.uuid4().hex
    body = b""
    fields = (("model", model), ("language", "en"), ("response_format", "json"), ("include[]", "logprobs"))
    for name, value in fields:
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav(pcm) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{OPENAI}/audio/transcriptions", data=body,
        headers={**AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        result = json.loads(r.read())
    logprobs = [t["logprob"] for t in result.get("logprobs") or []]
    return result["text"].strip(), (sum(logprobs) / len(logprobs)) if logprobs else 0.0


def is_speech(text: str, mean_lp: float) -> bool:
    letters = [c for c in text if c.isalpha()]
    ascii_share = sum(c.isascii() for c in letters) / len(letters) if letters else 0.0
    return len(text.split()) >= GUARD_MIN_WORDS and mean_lp >= GUARD_MEAN_LOGPROB and ascii_share >= GUARD_ASCII


def rms(pcm: bytes) -> float:
    samples = memoryview(pcm).cast("h") if len(pcm) % 2 == 0 else memoryview(pcm[:-1]).cast("h")
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 if len(samples) else 0.0


def chat_tool(tool) -> dict:
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or tool.name,
        "parameters": tool.inputSchema or {"type": "object", "properties": {}}}}


def bare(name: str) -> str:
    """Tool names arrive as "assist__HassTurnOn" once HA's MCP Server entry serves
    more than one API (Assist + Memory since 2026-10-31)."""
    return name.rsplit("__", 1)[-1]


ACTION_TOOLS = re.compile(r"^(Hass(?!.*Get)|tv_)")  # tools that change the house; the rest read it or keep memory

MEMORY_HEADER = 'Follow these instructions for tools from "memory":'


async def memory_instructions(ha: ClientSession) -> str:
    """The Memory part of HA's merged LLM API prompt: how to use the memory tools and
    the current index. Empty when the MCP Server entry has no Memory API ticked."""
    try:
        prompts = (await ha.list_prompts()).prompts
    except Exception as exc:  # noqa: BLE001 - a server without prompts (the eval's fake house) means no memory
        log.info("no memory prompt: %s", exc)
        return ""
    if not prompts:
        return ""
    text = (await ha.get_prompt(prompts[0].name)).messages[0].content.text
    at = text.find(MEMORY_HEADER)
    return "" if at < 0 else text[at:].strip()


def confirmation(acted: list[tuple[str, dict, dict]]) -> str:
    """What to say about the actions that ran, from their results alone."""
    said = []
    for name, args, output in acted:
        name = bare(name)
        if "error" in output:
            said.append("That didn't work.")
            continue
        try:
            result = json.loads(output.get("result") or "{}")
        except ValueError:
            result = {}
        if result.get("success") is False:
            said.append("That didn't work.")
            continue
        if "response_type" not in result and "result" in result:
            result = result["result"]  # the eval's fake house wraps HA's shape
        if isinstance(result, str):  # a custom tool's own words ("Launched Netflix on the living room TV")
            said.append(result.strip().rstrip(".") + ".")
            continue
        data = result.get("data") or {}
        names = list(dict.fromkeys(e.get("name", "") for e in data.get("success") or [] if e.get("name")))
        if result.get("response_type") == "error" or (not names and data.get("failed")):
            said.append("That didn't work.")
        elif names and name in ("HassTurnOff", "HassTurnOn"):
            state = "off" if name.endswith("Off") else "on"
            if len(names) > 3:  # every lamp and strip in the house is not a sentence
                kind = "lights" if args.get("domain") == ["light"] else "devices"
                said.append(f"{len(names)} {kind} {state}.")
            else:
                said.append(f"{', '.join(names)} {state}.")
        elif names:
            said.append(f"{', '.join(names)}: done.")
        else:
            said.append(((result.get("speech") or {}).get("plain") or {}).get("speech") or "Done.")
    return " ".join(said)


class Conversation:
    """One wake: from the puck connecting to the relay closing it."""

    def __init__(self, puck: web.WebSocketResponse, ha: ClientSession, http: aiohttp.ClientSession,
                 tools: list[dict], memory: str = "") -> None:
        self.puck, self.ha, self.http, self.tools = puck, ha, http, tools
        self.memory = memory  # HA's memory instructions and index, refreshed after a memory write
        self.started = time.monotonic()
        self.heard_until = self.started  # when the puck finishes playing what it has been sent
        self.spoke = False
        self.busy = 0  # turns in flight: transcribing, thinking, or a tool running
        self.tape = bytearray()  # everything the puck sent, for MINIMAL_TAPE
        self.play_end = 0.0  # when the audio sent so far finishes playing on the puck
        self.cursor = 0.0  # when the audio queued so far finishes playing
        self.outgoing: collections.deque[tuple[float, bytes]] = collections.deque()  # (send at, pcm)
        self.history: list[dict] = []  # this wake's turns, so "and the bedroom too" has context
        # The cutter's view of the mic.
        self.recent: collections.deque[bytes] = collections.deque(maxlen=LEAD_FRAMES)
        self.smooth: collections.deque[float] = collections.deque(maxlen=5)  # 100 ms of levels
        self.levels: collections.deque[float] = collections.deque(maxlen=FLOOR_FRAMES)
        self.utterance: bytearray | None = None
        self.mic_s = 0.0  # seconds of mic audio received so far
        self.voiced_s = 0.0
        self.quiet_s = 0.0
        self.turns: set[asyncio.Task] = set()

    async def run(self) -> None:
        tasks = [asyncio.create_task(c) for c in (self.from_puck(), self.speaker(), self.clock())]
        _, rest = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in rest:
            t.cancel()
        await asyncio.gather(*rest, return_exceptions=True)
        # Whatever is still queued is the end of the last sentence.
        for _, pcm in self.outgoing:
            with contextlib.suppress(ConnectionError):
                await self.puck.send_bytes(pcm)

    async def speaker(self) -> None:
        """Send queued audio when it is due; each chunk is due PREROLL_S before it plays."""
        while True:
            if not self.outgoing:
                await asyncio.sleep(0.01)
                continue
            due, pcm = self.outgoing[0]
            if (wait := due - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            self.outgoing.popleft()
            self.play_end = max(self.play_end, time.monotonic()) + len(pcm) / 2 / RATE
            await self.puck.send_bytes(pcm)

    def queue(self, pcm: bytes) -> None:
        """Schedule audio at real time from the end of whatever is already queued."""
        self.cursor = max(time.monotonic(), self.play_end, self.cursor)
        for i in range(0, len(pcm), CHUNK):
            chunk = pcm[i:i + CHUNK]
            self.outgoing.append((self.cursor - PREROLL_S, chunk))
            self.cursor += len(chunk) / 2 / RATE
        self.spoke = True
        self.heard_until = max(self.heard_until, self.cursor)

    async def from_puck(self) -> None:
        async for msg in self.puck:
            if msg.type == aiohttp.WSMsgType.BINARY:
                self.tape.extend(msg.data)
                self.track_utterance(msg.data, rms(msg.data))
                # The user is talking: keep the wake open. Only an utterance
                # with real speech in it counts: a single loud frame did, and
                # a room at ~150 RMS kept a wake open 104 s (2026-10-19).
                if self.utterance is not None and self.voiced_s >= MIN_SPEECH_S:
                    self.heard_until = max(self.heard_until, time.monotonic())
            elif msg.type == aiohttp.WSMsgType.TEXT:
                log.info("puck says %s", msg.data)
        log.info("puck hung up")

    def track_utterance(self, pcm: bytes, level: float) -> None:
        """Cut the mic into utterances; each one that ends with speech in it becomes a turn."""
        secs = len(pcm) / 2 / RATE
        self.mic_s += secs
        if self.mic_s > 0.6:
            self.recent.append(pcm)  # never the chime: it sends the transcriber into other languages
        # The room's level: the quietest tenth of a second in the last three.
        # Speech has such gaps between words; a chime or a sentence does not
        # drag it up, the TV between words does.
        self.smooth.append(level)
        self.levels.append(sum(self.smooth) / len(self.smooth))
        floor = min(self.levels)
        loud = level >= max(UTTERANCE_RMS, FLOOR_RATIO * floor)
        if self.utterance is None:
            # The first half second of mic audio is the wake chime, not the
            # user. Counted in audio, not wall time: the puck streams from
            # the wake while the relay is still connecting, so the opening
            # second arrives in one burst.
            if loud and self.mic_s > 0.6:
                self.utterance = bytearray(b"".join(self.recent))
                self.voiced_s = self.quiet_s = 0.0
            return
        self.utterance.extend(pcm)
        if loud:
            self.voiced_s += secs
            self.quiet_s = 0.0
        else:
            self.quiet_s += secs
        if self.quiet_s >= QUIET_S or len(self.utterance) >= MAX_UTTERANCE_S * RATE * 2:
            done, voiced = bytes(self.utterance), self.voiced_s
            self.utterance = None
            log.info("utterance: %.1fs of speech at mic %.1fs%s", voiced, self.mic_s,
                     "" if voiced >= MIN_SPEECH_S else " (too short)")
            if voiced >= MIN_SPEECH_S:
                self.on_utterance(done)

    def on_utterance(self, pcm: bytes) -> None:
        # Kept in self.turns: an unreferenced task can be garbage collected mid-await.
        task = asyncio.create_task(self.turn(pcm))
        self.turns.add(task)
        task.add_done_callback(self.turns.discard)

    async def hear(self, pcm: bytes) -> tuple[str, float] | None:
        """The best reading of an utterance that passes the guard, or None."""
        readings = await asyncio.gather(*(asyncio.to_thread(transcribe, pcm, m) for m in STT_MODELS),
                                        return_exceptions=True)
        best = None
        for model, reading in zip(STT_MODELS, readings):
            if isinstance(reading, BaseException):
                log.info("%s failed: %s", model, reading)
                continue
            text, mean_lp = reading
            log.debug("%s: %r (mean %.2f)", model, text, mean_lp)
            if is_speech(text, mean_lp) and (best is None or mean_lp > best[1]):
                best = (text, mean_lp)
        if best is None:
            log.info("not speech (%s)", "; ".join(
                f"{r[0]!r} {r[1]:.2f}" for r in readings if not isinstance(r, BaseException)))
        return best

    async def turn(self, pcm: bytes) -> None:
        self.busy += 1
        try:
            t0 = time.monotonic()
            heard = await self.hear(pcm)
            if heard is None:
                return
            text, mean_lp = heard
            log.info("heard: %s (mean %.2f, %.1fs)", text, mean_lp, time.monotonic() - t0)
            if COMMAND_RE.search(text):
                await self.ack()
            answer, acted = await self.think(text)
            spoken = confirmation(acted) if acted else answer
            if spoken.strip().lower().startswith("(nothing)"):
                spoken = ""
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": spoken or "(nothing)"}]
            await self.say(spoken)
        except Exception:
            log.exception("turn failed")
            await self.say("Sorry, something went wrong.")
        finally:
            self.busy -= 1

    async def think(self, text: str) -> tuple[str, list[tuple[str, dict, dict]]]:
        """Ask the model; run its tool calls; stop at the first action, or return its answer."""
        now = datetime.now(TZ).strftime("%A %B %-d, %-I:%M %p")
        system = f"{INSTRUCTIONS} It is now {now}." + (f"\n\n{self.memory}" if self.memory else "")
        messages = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": text}]
        acted: list[tuple[str, dict, dict]] = []
        for _ in range(3):
            t0 = time.monotonic()
            async with self.http.post(f"{OPENAI}/chat/completions", headers=AUTH, json={
                "model": BRAIN_MODEL, "messages": messages, "tools": self.tools,
                "tool_choice": "auto", "parallel_tool_calls": True,
            }) as r:
                data = await r.json()
            if "choices" not in data:
                raise RuntimeError(f"chat refused: {json.dumps(data)[:300]}")
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            log.info("thought %.1fs: %s", time.monotonic() - t0,
                     ", ".join(c["function"]["name"] for c in calls) or repr((msg.get("content") or "")[:80]))
            if not calls:
                return (msg.get("content") or "").strip(), acted
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            outputs = await asyncio.gather(*(self.call(c["function"]["name"], c["function"]["arguments"]) for c in calls))
            for c, output in zip(calls, outputs):
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(output, ensure_ascii=False)})
                if ACTION_TOOLS.match(bare(c["function"]["name"])):
                    acted.append((c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"), output))
                elif bare(c["function"]["name"]) in ("memory_save", "memory_delete") and "error" not in output:
                    # The index in the prompt is from the wake; after a write the
                    # model otherwise looks for names that no longer exist.
                    self.memory = await memory_instructions(self.ha)
            if acted:
                return "", acted
        return "", acted

    async def call(self, name: str, arguments: str) -> dict:
        args = {k: v for k, v in json.loads(arguments or "{}").items() if v is not None}
        # "Can you turn the lights off?" became HassTurnOff(domain=[light, switch])
        # with no name or area: every exposed switch went off, the server's
        # own outlet among them (2026-10-10).
        # No domain at all is refused too: device_class "outlet" alone matches the same plugs.
        domain = args.get("domain") or []
        if bare(name) in ("HassTurnOn", "HassTurnOff") and (not domain or "switch" in domain) \
                and not (args.get("name") or args.get("area") or args.get("floor")):
            output = {"error": "Refused: switches are never turned on or off all at once. For lights "
                               "use domain [\"light\"] alone; for a switch, give its name or area."}
            log.info("tool %s(%s) -> %s", name, json.dumps(args), json.dumps(output)[:200])
            return output
        try:
            result = await self.ha.call_tool(name, args)
            text = "".join(getattr(c, "text", "") for c in result.content)
            output = {"error": text} if result.isError else {"result": text}
        except Exception as exc:  # noqa: BLE001 - a failed call is reported, not raised
            output = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.heard_until = max(self.heard_until, time.monotonic())
        log.info("tool %s(%s) -> %s", name, json.dumps(args), json.dumps(output)[:200])
        return output

    async def tts(self, text: str) -> bytes:
        async with self.http.post(f"{OPENAI}/audio/speech", headers=AUTH, json={
            "model": TTS_MODEL, "voice": VOICE, "input": text, "response_format": "pcm",
        }) as r:
            if r.status != 200:
                raise RuntimeError(f"tts {r.status}: {(await r.text())[:200]}")
            return await r.read()

    async def ack(self) -> None:
        global _ack_pcm
        if _ack_pcm is None:
            _ack_pcm = await self.tts(ACK)
        log.info("said: %s", ACK)
        self.queue(_ack_pcm)

    async def say(self, text: str) -> None:
        """Speak text, queueing the voice as it streams in so the first word is not held for the last."""
        if not text:
            return
        log.info("said: %s", text)
        buf = bytearray()
        async with self.http.post(f"{OPENAI}/audio/speech", headers=AUTH, json={
            "model": TTS_MODEL, "voice": VOICE, "input": text, "response_format": "pcm",
        }) as r:
            if r.status != 200:
                raise RuntimeError(f"tts {r.status}: {(await r.text())[:200]}")
            async for part in r.content.iter_chunked(CHUNK):
                buf += part
                whole = len(buf) // CHUNK * CHUNK
                if whole:
                    self.queue(bytes(buf[:whole]))
                    del buf[:whole]
        if buf:
            self.queue(bytes(buf))

    async def clock(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if now - self.started > HARD_CAP_S:
                log.info("hard cap reached")
                return
            # The puck hangs up after 10 s without audio; a slow tool (the TV
            # took 8 s to switch off) ended the wake before the answer
            # (2026-10-22). Silence while waiting keeps it on the line. Not
            # while the user is talking: the puck drops mic audio whenever
            # speaker audio arrived recently, so silence then made it deaf
            # (2026-10-31).
            if self.busy and self.utterance is None and now - self.play_end > 2:
                self.outgoing.append((now, bytes(CHUNK)))
            quiet = now - self.heard_until
            if not self.busy and quiet > (FOLLOWUP_S if self.spoke else FIRST_REPLY_S):
                log.info("quiet for %.0fs, ending the conversation", quiet)
                return


_ack_pcm: bytes | None = None


async def handle(request: web.Request) -> web.WebSocketResponse:
    puck = web.WebSocketResponse(max_msg_size=0)
    await puck.prepare(request)
    woke = time.monotonic()
    log.info("wake from %s", request.remote)
    try:
        async with sse_client(HA_MCP_URL, headers={"Authorization": f"Bearer {HA_TOKEN}"}) as (r, w), \
                ClientSession(r, w) as ha, \
                aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=True) as http:
            await ha.initialize()
            tools = [chat_tool(t) for t in (await ha.list_tools()).tools]
            memory = await memory_instructions(ha)
            log.info("session ready %.2fs after the wake", time.monotonic() - woke)
            conversation = Conversation(puck, ha, http, tools, memory)
            await conversation.run()
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
    web.run_app(app, host="0.0.0.0", port=PORT,
                print=lambda *_: log.info("relay '%s' listening on :%d (%s + %s -> %s -> %s)",
                                          VERSION, PORT, *STT_MODELS[:2], BRAIN_MODEL, TTS_MODEL))


if __name__ == "__main__":
    main()
