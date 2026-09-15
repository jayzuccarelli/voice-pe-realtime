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
import asyncio
import logging
import math
import os
import time
from collections import deque

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMSettings
from websockets.exceptions import ConnectionClosed

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

    async def push_error(self, error_msg: str = "", exception=None, **kwargs) -> None:
        """An idle socket closing is not an error worth ending the pipeline for.

        The OpenAI socket is held open between wakes so the next one pays
        only session.start. OpenAI closes it after a couple of idle hours;
        upstream treats any close outside its own teardown as permanent,
        the pipeline ends, and the device is dead until a restart (observed
        2026-09-14 after ~2h20 idle). With no session running there is
        nothing to lose: drop the socket and let the next wake reconnect.
        """
        if isinstance(exception, ConnectionClosed) and not self._session_started:
            logger.info("Idle OpenAI socket closed (%s); reconnecting on the next wake", exception)
            self._websocket = None
            return
        await super().push_error(error_msg, exception=exception, **kwargs)

    #: Terminal states for a delegated backend response.
    _RESPONSE_DONE = ("response.completed", "response.incomplete", "response.failed")

    #: How much speech to hold while the billed session is opening. Opening
    #: costs ~3s (our websocket handshake, then the API's own session.start),
    #: and a puck connects *because* the user just said the wake word, so the
    #: question is spoken entirely inside that window. Bounded so an unusually
    #: slow open replays recent speech rather than a stale backlog.
    _MAX_PRESTART_AUDIO_SECONDS = 8.0

    #: What gated frames are scaled by: 40 dB down, not digital zero. This is
    #: the figure the replay experiments validated against the real device.
    _GATE_ATTENUATION = 0.01
    #: Silero runs at 16 kHz on 512-sample chunks; anything at or above this
    #: confidence counts as speech.
    _GATE_VAD_RATE = 16000
    _GATE_VAD_CONFIDENCE = 0.5
    #: Frames held back so the decision for a frame can use audio slightly
    #: after it. Silero needs ~32 ms of audio before it reports an onset; two
    #: 20 ms frames of look-ahead mean a word's first syllable is never the
    #: casualty of that delay. 40 ms of added mic latency is inaudible.
    _GATE_LOOKAHEAD_FRAMES = 2
    #: Fallback when Silero is unavailable: level over a rolling window.
    _GATE_WINDOW_SECONDS = 0.1

    def __init__(
        self,
        *,
        input_gate_rms: float = 0.0,
        input_gate_hold_ms: float = 250.0,
        input_gate_vad: bool = True,
        flush_pace: float = 1.0,
        input_gain_db: float = 0.0,
        fallback_transcription: bool = True,
        fallback_model: str = "gpt-4o-mini-transcribe",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._flush_pace = float(flush_pace)
        self._input_gain = 10 ** (float(input_gain_db) / 20.0)
        if input_gain_db:
            logger.info("Live input gain: %+.1f dB", input_gain_db)
        # Audio waiting to go to the model at real-time pace once the session
        # is live: what was captured while it opened, then anything that
        # arrives while that is still draining, so order is preserved and
        # live frames are never interleaved with the replay.
        self._flush_queue: deque[InputAudioRawFrame] = deque()
        self._flush_task = None
        self._draining = False
        # Called once the pre-start replay has fully reached the model.
        self.on_prestart_replayed = None
        # Whether the model has reported hearing the user at all this session.
        self._user_turn_seen = False
        # Debug aid: everything sent to the model this session, written to a
        # WAV at session end so a silent wake can be transcribed and heard.
        self._session_tape = bytearray() if os.environ.get("LIVE_SESSION_TAPE") else None
        # Ids of delegated responses still open, for the CURRENT session only.
        self._live_open_responses: set[str] = set()
        # Mic audio captured while the session was still opening, oldest first.
        self._prestart_audio: deque[InputAudioRawFrame] = deque()
        self._prestart_seconds = 0.0
        # Input noise gate. 0 disables it. Speech is decided by Silero VAD,
        # which is trained on exactly this kind of noisy far-field audio; a
        # level threshold is the fallback, and it is a poor one here because
        # the device's floor and its quieter speech overlap in level.
        self._input_gate_rms = float(input_gate_rms)
        self._gate_hold_seconds = float(input_gate_hold_ms) / 1000.0
        self._gate_quiet_seconds = 0.0
        self._gate_pending: deque[InputAudioRawFrame] = deque()
        self._gate_vad = None
        self._gate_resampler = None
        self._gate_vad_buf = bytearray()
        self._gate_vad_conf = 0.0
        self._gate_window: deque[tuple[float, int, float]] = deque()  # (sum sq, n, secs)
        self._gate_window_seconds = 0.0
        self._gate_passed = 0
        self._gate_attenuated = 0
        # Silero is fed once per frame and read by both the gate and the
        # transcription fallback, so it is built whenever it is available.
        try:
            from pipecat.audio.utils import create_stream_resampler
            from pipecat.audio.vad.silero import SileroVADAnalyzer

            self._gate_vad = SileroVADAnalyzer(sample_rate=self._GATE_VAD_RATE)
            # The constructor stores the rate; the model only sees it
            # once this is called (the transport normally does it).
            self._gate_vad.set_sample_rate(self._GATE_VAD_RATE)
            self._gate_resampler = create_stream_resampler()
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "Silero VAD unavailable (%s); level gate only, no transcription fallback", exc
            )
        self._gate_use_vad = input_gate_vad and self._gate_vad is not None
        logger.info(
            "Live input gate: %s, floor attenuated 40 dB after %.0f ms quiet",
            "off"
            if self._input_gate_rms <= 0
            else ("Silero VAD" if self._gate_use_vad else f"level rms<{self._input_gate_rms:.0f}"),
            self._gate_hold_seconds * 1000,
        )
        # Transcription fallback: see _track_fallback_segment.
        self._fallback_enabled = bool(fallback_transcription) and self._gate_vad is not None
        self._fallback_model = fallback_model
        self._fallback_task = None
        self._openai = None
        self._reset_fallback()
        logger.info(
            "Live transcription fallback: %s",
            f"on ({fallback_model})" if self._fallback_enabled else "off",
        )

    async def _vad_speech(self, frame: InputAudioRawFrame) -> bool:
        """Whether the newest audio is speech, by Silero. Fed exactly once per frame."""
        pcm = await self._gate_resampler.resample(frame.audio, frame.sample_rate, self._GATE_VAD_RATE)
        self._gate_vad_buf += pcm
        need = self._gate_vad.num_frames_required() * 2
        while len(self._gate_vad_buf) >= need:
            chunk = bytes(self._gate_vad_buf[:need])
            del self._gate_vad_buf[:need]
            # Silero hands back a 1-element array, not a scalar.
            conf = self._gate_vad.voice_confidence(chunk)
            self._gate_vad_conf = float(getattr(conf, "flat", [conf])[0])
        return self._gate_vad_conf >= self._GATE_VAD_CONFIDENCE

    def _level_speech(self, frame: InputAudioRawFrame) -> bool:
        """Whether the newest audio is speech, by level alone."""
        samples = array.array("h")
        samples.frombytes(frame.audio[: len(frame.audio) // 2 * 2])
        if not samples:
            return False
        secs = self._frame_seconds(frame)
        frame_sq = float(sum(s * s for s in samples))
        self._gate_window.append((frame_sq, len(samples), secs))
        self._gate_window_seconds += secs
        while self._gate_window_seconds > self._GATE_WINDOW_SECONDS and len(self._gate_window) > 1:
            _, _, old = self._gate_window.popleft()
            self._gate_window_seconds -= old
        total_sq = sum(w[0] for w in self._gate_window)
        total_n = sum(w[1] for w in self._gate_window) or 1
        # Fast attack on the frame, slow release on the window.
        rms = max(math.sqrt(frame_sq / len(samples)), math.sqrt(total_sq / total_n))
        return rms >= self._input_gate_rms

    async def _gate_input(
        self, frame: InputAudioRawFrame, speech: bool
    ) -> list[InputAudioRawFrame]:
        """Attenuate frames that carry only the room's noise floor.

        gpt-live-1 does its own turn detection, and on this device it never
        opens a turn: the puck's mic path carries a constant floor of about
        -40 dBFS (mains hum and a device tone) under and around the speech,
        and the model treats that as "no one is talking" no matter how loud
        the words on top of it are. The same recording with its between-word
        floor pulled down 40 dB gets answered; clean synthetic speech, which
        falls to digital zero between words, always did. Level, bandwidth,
        leading silence and reverb were each ruled out by replaying the
        device's own capture with one change at a time (2026-09-14).

        So the floor is removed here, before the audio reaches the model.
        Returns the frames now ready to send: none while the look-ahead
        fills, then one per call.
        """
        if self._input_gate_rms <= 0:
            return [frame]
        if not self._gate_use_vad:
            speech = self._level_speech(frame)
        self._gate_pending.append(frame)
        if len(self._gate_pending) <= self._GATE_LOOKAHEAD_FRAMES:
            return []
        out = self._gate_pending.popleft()
        if speech:
            self._gate_quiet_seconds = 0.0
            self._gate_passed += 1
            return [out]
        self._gate_quiet_seconds += self._frame_seconds(out)
        if self._gate_quiet_seconds <= self._gate_hold_seconds:
            self._gate_passed += 1
            return [out]
        self._gate_attenuated += 1
        samples = array.array("h")
        samples.frombytes(out.audio[: len(out.audio) // 2 * 2])
        quiet = array.array("h", (int(s * self._GATE_ATTENUATION) for s in samples))
        return [
            InputAudioRawFrame(
                audio=quiet.tobytes(),
                sample_rate=out.sample_rate,
                num_channels=out.num_channels,
            )
        ]

    async def _open_turn(self, role: str) -> None:
        if role == "user":
            self._user_turn_seen = True
        else:
            self._model_spoke = True
        await super()._open_turn(role)

    async def _end_turn(self, role: str) -> None:
        # What the model heard and what it said, one line each: the two
        # halves of "did the request reach OpenAI and did an answer come back".
        turn = self._user_turn if role == "user" else self._assistant_turn
        if turn.open and turn.text.strip():
            logger.info("Live %s said: %r", "user" if role == "user" else "model", turn.text.strip())
        await super()._end_turn(role)

    #
    # Transcription fallback
    #

    _FALLBACK_BUFFER_SECONDS = 12.0
    _FALLBACK_START_SECONDS = 0.1  # speech this long opens the utterance
    _FALLBACK_QUIET_SECONDS = 0.9  # quiet this long closes it
    _FALLBACK_MAX_SECONDS = 6.0  # a TV never goes quiet; a question is shorter than this
    _FALLBACK_TAIL_SECONDS = 0.5
    _FALLBACK_MODEL_GRACE_SECONDS = 0.6

    def _reset_fallback(self) -> None:
        self._fallback_done = False
        self._model_spoke = False
        self._fb_audio = bytearray()
        self._fb_rate = 0
        self._fb_origin = 0.0  # seconds trimmed off the front of _fb_audio
        self._fb_seconds = 0.0  # seconds of mic audio seen this session
        self._fb_speech_seconds = 0.0
        self._fb_quiet_seconds = 0.0
        self._fb_seg_start = None
        self._sent_seconds = 0.0  # mic audio actually sent to the model this session

    def _track_fallback_segment(
        self, frame: InputAudioRawFrame, speech: bool
    ) -> tuple[float, float] | None:
        """Follow the first thing said after the wake; return its bounds once it ends.

        gpt-live-1 decides on its own when someone has spoken to it, and on
        this device's real far-field audio it often decides nobody has: a
        recording Whisper transcribes word-perfect gets "I didn't catch
        that" from the live model itself (2026-09-15), with no input
        transcript and no turn. There is no input-side setting to change
        that. So the first utterance after the wake is tracked here with
        Silero, and if the model has not opened a turn on it by the time the
        speaker goes quiet, the words go to it as text instead.

        Once per wake, and only the first utterance: the wake word gated
        exactly one request, and anything after it (a TV, the room) is not
        one. Follow-ups after a reply stay the model's job.
        """
        if not self._fallback_enabled or self._fallback_done:
            return None
        secs = self._frame_seconds(frame)
        if not self._fb_rate:
            self._fb_rate = frame.sample_rate
        self._fb_audio += frame.audio
        self._fb_seconds += secs
        max_bytes = int(self._FALLBACK_BUFFER_SECONDS * self._fb_rate * 2)
        if len(self._fb_audio) > max_bytes:
            drop = len(self._fb_audio) - max_bytes
            del self._fb_audio[:drop]
            self._fb_origin += drop / float(self._fb_rate * 2)
        if speech:
            self._fb_speech_seconds += secs
            self._fb_quiet_seconds = 0.0
            if (
                self._fb_seg_start is None
                and self._fb_speech_seconds >= self._FALLBACK_START_SECONDS
            ):
                self._fb_seg_start = self._fb_seconds - self._fb_speech_seconds
        else:
            self._fb_speech_seconds = 0.0
            if self._fb_seg_start is not None:
                self._fb_quiet_seconds += secs
        if self._fb_seg_start is None:
            return None
        if self._fb_quiet_seconds >= self._FALLBACK_QUIET_SECONDS:
            self._fallback_done = True
            return (self._fb_seg_start, self._fb_seconds - self._fb_quiet_seconds)
        if self._fb_seconds - self._fb_seg_start >= self._FALLBACK_MAX_SECONDS:
            self._fallback_done = True
            return (self._fb_seg_start, self._fb_seconds)
        return None

    def _fallback_slice(self, end: float) -> bytes:
        """Everything heard from the wake to just after the utterance.

        From the wake, not from where Silero first fired: at its default
        confidence it flags about half of a short far-field question, and a
        transcriber given the lead-in as well loses no words to that. The
        wake word gated this one request, so nothing before it is in here.
        """
        bps = self._fb_rate * 2
        b = end + self._FALLBACK_TAIL_SECONDS - self._fb_origin
        return bytes(self._fb_audio[: int(b * bps)])

    def _model_took_the_turn(self) -> bool:
        return self._user_turn_seen or self._model_spoke or self.delegation_in_flight

    async def _run_fallback(self, start: float, end: float) -> None:
        # The model hears the same audio at real-time pace, a little behind
        # the mic while the opening seconds replay: wait until it has the
        # whole utterance, then give it a moment to open the turn itself.
        deadline = asyncio.get_running_loop().time() + 5.0
        while (
            self._sent_seconds < end + self._FALLBACK_TAIL_SECONDS
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
        await asyncio.sleep(self._FALLBACK_MODEL_GRACE_SECONDS)
        if self._model_took_the_turn():
            logger.debug("Live fallback: the model opened the turn itself")
            return
        text = await self._transcribe(self._fallback_slice(end))
        if self._model_took_the_turn():
            logger.info("Live fallback: heard %r, but the model took the turn meanwhile", text)
            return
        if len(text.split()) < 2:
            logger.info("Live fallback: heard %r; too little to act on", text)
            return
        logger.info("Live fallback: heard %r; handing it to the model as text", text)
        from pipecat.services.openai.live import events

        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        await self.send_client_event(events.ResponseItemCreateEvent(item=item))
        await self.send_client_event(events.ResponseCreateEvent())

    async def _transcribe(self, pcm: bytes) -> str:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self._fb_rate)
            out.writeframes(pcm)
        if self._openai is None:
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI()
        try:
            result = await self._openai.audio.transcriptions.create(
                model=self._fallback_model, file=("wake.wav", buf.getvalue())
            )
        except Exception as exc:  # noqa: BLE001 - a failed backstop must not end the session
            logger.warning("Live fallback: transcription failed: %s", exc)
            return ""
        return (result.text or "").strip()

    async def _stop_fallback(self) -> None:
        task, self._fallback_task = self._fallback_task, None
        if task is not None and not task.done():
            await self.cancel_task(task, timeout=1.0)

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
        frame = self._apply_input_gain(frame)
        speech = await self._vad_speech(frame) if self._gate_vad is not None else False
        segment = self._track_fallback_segment(frame, speech)
        if segment is not None:
            self._fallback_task = self.create_task(self._run_fallback(*segment))
        for ready in await self._gate_input(frame, speech):
            if not self._session_started:
                self._prestart_audio.append(ready)
                self._prestart_seconds += self._frame_seconds(ready)
                while (
                    self._prestart_audio
                    and self._prestart_seconds > self._MAX_PRESTART_AUDIO_SECONDS
                ):
                    self._prestart_seconds -= self._frame_seconds(self._prestart_audio.popleft())
                continue
            if self._draining:
                # The replay of the opening seconds is still going out; queue
                # behind it rather than jumping the line, or the model hears
                # the question with live silence spliced between its frames.
                self._flush_queue.append(ready)
                continue
            await self._send_to_model(ready)

    def _apply_input_gain(self, frame: InputAudioRawFrame) -> InputAudioRawFrame:
        """Scale mic audio toward the level the model's turn detector expects.

        Far-field speech from across the room arrives well under it; see
        Config.live_input_gain_db for the measurement. Samples are clipped
        at full scale rather than wrapped.
        """
        if self._input_gain == 1.0:
            return frame
        samples = array.array("h")
        samples.frombytes(frame.audio[: len(frame.audio) // 2 * 2])
        if not samples:
            return frame
        g = self._input_gain
        louder = array.array("h", (max(-32768, min(32767, int(s * g))) for s in samples))
        return InputAudioRawFrame(
            audio=louder.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )

    async def _send_to_model(self, frame: InputAudioRawFrame) -> None:
        self._sent_seconds += self._frame_seconds(frame)
        if self._session_tape is not None:
            self._session_tape += frame.audio
        await super()._send_user_audio(frame)

    def _write_session_tape(self) -> None:
        if not self._session_tape:
            return
        import wave

        path = f"/tmp/claude/session-{int(time.time())}.wav"
        try:
            with wave.open(path, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(24000)
                out.writeframes(bytes(self._session_tape))
            logger.info(
                "Session tape: %.1fs written to %s", len(self._session_tape) / 48000.0, path
            )
        except OSError as exc:
            logger.warning("Could not write session tape: %s", exc)
        self._session_tape.clear()

    async def _handle_evt_session_started(self, evt) -> None:
        """Start the session, then replay what the user said while it opened."""
        await super()._handle_evt_session_started(evt)
        if not self._prestart_audio:
            return
        held = list(self._prestart_audio)
        seconds = self._prestart_seconds
        self._clear_prestart_audio()
        self._describe_prestart_audio(held, seconds)
        # Paced, not dumped: the model's turn detector runs on a real-time
        # stream, and a question that arrives in one instant is not a turn.
        # Drained by its own task so this handler, which runs on the receive
        # loop, returns at once and server events keep flowing meanwhile.
        self._flush_queue.extend(held)
        self._draining = True
        self._flush_task = self.create_task(self._drain_flush_queue())

    async def _drain_flush_queue(self) -> None:
        try:
            while self._flush_queue:
                frame = self._flush_queue.popleft()
                await self._send_to_model(frame)
                if self._flush_pace > 0:
                    await asyncio.sleep(self._frame_seconds(frame) / self._flush_pace)
        finally:
            self._draining = False
            # Anything that slipped in between the last pop and the flag
            # clearing goes out now, still in order.
            while self._flush_queue:
                await self._send_to_model(self._flush_queue.popleft())
            # The model has now heard everything said before the session
            # opened; the wait for its first reply starts here, not at the
            # wake, or a long replay would eat the reply's time.
            if self.on_prestart_replayed is not None:
                self.on_prestart_replayed()

    async def _stop_flush(self) -> None:
        task, self._flush_task = self._flush_task, None
        self._draining = False
        self._flush_queue.clear()
        if task is not None and not task.done():
            await self.cancel_task(task, timeout=1.0)

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
        if not samples or self._session_tape is None:
            # Microphone audio is only ever written to disk when the
            # operator opted in with LIVE_SESSION_TAPE.
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
        self._user_turn_seen = False
        self._clear_prestart_audio()
        await self._stop_flush()
        await self._stop_fallback()
        self._reset_fallback()
        # Reuse the socket when it is alive: the handshake is the bulk of the
        # ~3s a cold open costs, and everything the user says during that
        # wait has to be replayed later. The socket is free to hold open; only
        # the session bills. _connect() early-returns on a non-None socket
        # even when it is dead, so a dead one is torn down first.
        socket_dead = self._websocket is not None and (
            self._receive_task is None or self._receive_task.done()
        )
        if socket_dead:
            await self._disconnect()
        if self._websocket is None:
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
        await self._stop_flush()
        await self._stop_fallback()
        if self._session_tape is not None:
            self._write_session_tape()
        try:
            try:
                await self._close_open_turns()
            finally:
                await self._close_session()
        finally:
            try:
                await self._disconnect()
                # Reconnect now, while nobody is waiting, so the next wake
                # pays only session.start (~0.7s) and not the handshake.
                await self._connect()
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
                self._gate_pending.clear()
                self._gate_vad_buf.clear()
                self._gate_vad_conf = 0.0


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

# The opposite bias from BACKGROUND_GUIDANCE, for a wake-word device. The
# wake word is consumed on the device, so the model never hears itself
# addressed; without this it has only distant, noisy audio and no reason to
# believe anyone is talking to it. The Live API has no input-side knob for
# this (no turn-detection or noise-reduction setting exists in session.start),
# so the prompt is the only lever.
WAKE_GUIDANCE = (
    " The user has just said your wake word, so they are speaking to you. They "
    "are across the room, through a far-field microphone, so their voice may "
    "sound quiet, distant or noisy. Treat what you hear right after the session "
    "starts as a request addressed to you and answer it promptly. Do not wait to "
    "hear your name, and do not stay silent because the audio is imperfect."
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
    guidance = (
        (BACKGROUND_GUIDANCE if config.live_far_field_guidance else "")
        + (WAKE_GUIDANCE if config.live_wake_guidance else "")
        + DELEGATION_GUIDANCE
    )
    return VoicePELiveService(
        api_key=config.openai_api_key,
        input_gate_rms=config.live_input_gate_rms,
        input_gate_hold_ms=config.live_input_gate_hold_ms,
        input_gate_vad=config.live_input_gate_vad,
        flush_pace=config.live_flush_pace,
        input_gain_db=config.live_input_gain_db,
        fallback_transcription=config.live_fallback_transcription,
        fallback_model=config.live_fallback_model,
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
