"""Regression tests for the Live engine's session-lifetime bounds.

No network, no broker, no device: these drive `_LiveHygiene` and the
delegation counter directly, because the bug they cover only shows up in the
timing between a question and a reply, which is awkward to provoke through
the audio harness.

    python3 broker/tools/test_live_hygiene.py
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import time
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipecat.frames.frames import InputAudioRawFrame

from realtime_broker.live_agent import (
    VoicePELiveService,
    echoes_vocabulary,
    transcripts_agree,
)
from realtime_broker.live_server import _LiveHygiene, _OutputSilenceFilter


def _response_evt(inner: str, response_id: str | None):
    """A minimal stand-in for a wrapped Responses lifecycle envelope."""
    event = {"response": {"id": response_id}} if response_id else {}
    return types.SimpleNamespace(
        inner_type=inner,
        type="response.event",
        event=event,
        delegation_id=None,
        item_id=None,
        event_id=None,
    )


class _MemoryOnly(VoicePELiveService):
    """The cross-wake memory without any of the websocket machinery."""

    def __init__(self, turns=8, minutes=60.0):
        self._memory = collections.deque()
        self._memory_turns = turns
        self._memory_seconds = minutes * 60.0
        self._wake_turns = []
        self._wake_acted = False


def test_memory_digest_carries_speech_and_nothing_else():
    """Recent turns come back as relative-time text; stale and empty ones do not.

    Ages are relative on purpose: an absolute clock reading carried into the
    next wake is a fact the model repeats long after it stopped being true,
    which is exactly how "it's still 9:01" happened.
    """
    svc = _MemoryOnly(turns=4, minutes=30)
    now = time.time()
    svc._remember("user", "what time is it")
    svc._remember("assistant", "Mm-hmm.")  # an acknowledgement, not content
    svc._remember("assistant", "It's 9:01 PM on Tuesday.")
    svc._remember("user", "what time is it")  # repeat of the last user line, kept
    svc._commit_wake_memory()
    digest = svc._memory_digest()
    assert "Mm-hmm" not in digest, digest
    assert "what time is it" in digest and "9:01 PM" in digest, digest
    assert "just now" in digest, digest
    assert "looked up again" in digest, digest

    # Older than the window, and beyond the turn cap, both drop out.
    svc._memory[0] = (now - 3600, "user", "yesterday's question")
    assert "yesterday's question" not in svc._memory_digest()
    for i in range(6):
        svc._remember("user", f"question {i}")
    svc._commit_wake_memory()
    assert len(svc._memory) == 4, svc._memory
    assert "question 0" not in svc._memory_digest()

    assert _MemoryOnly(turns=0)._memory_digest() == "", "memory off must produce nothing"
    print("PASS: the digest carries what was said, relative and bounded")


def test_a_wake_that_acted_is_never_remembered():
    """A command in memory is a command the model carries out again.

    A "what time is it" called the TV's turn-off twice because the wake
    before had asked for it (2026-09-19). Lookups keep their turns;
    anything that acted on the house leaves none behind.
    """
    svc = _MemoryOnly()
    svc._remember("user", "turn the living room TV off")
    svc._remember("assistant", "Done, the living room TV is off.")
    svc._wake_acted = True
    svc._commit_wake_memory()
    assert not svc._memory, svc._memory
    assert svc._memory_digest() == ""

    svc._remember("user", "what is the capital of France")
    svc._remember("assistant", "Paris is the capital.")
    svc._commit_wake_memory()
    assert "France" in svc._memory_digest()
    assert "TV" not in svc._memory_digest()

    assert VoicePELiveService._is_read_only_tool("GetDateTime")
    assert VoicePELiveService._is_read_only_tool("get_weather")
    assert not VoicePELiveService._is_read_only_tool("HassTurnOff")
    assert not VoicePELiveService._is_read_only_tool("tv_remote")
    print("PASS: commands never reach memory; questions do")


def test_transcripts_agree_filters_garbage():
    """Two transcripts of the same clip must agree before the backstop acts.

    Taken from real passes over one far-field request: the two good ones
    agree whatever their punctuation, and the garbage one agrees with
    neither.
    """
    good_a = "Turn the living room TV off."
    good_b = "turn the living room tv off"
    garbage = "Drogadmeni group TVApps."
    italian = "Trova di vincitivi."
    assert transcripts_agree(good_a, good_b)
    assert not transcripts_agree(good_a, garbage)
    assert not transcripts_agree(good_a, italian)
    assert not transcripts_agree(good_a, "")
    assert transcripts_agree("What time is it?", "what time is it,")
    # Mostly the same words, opposite command: must not agree.
    assert not transcripts_agree("turn the living room TV off", "turn the living room TV on")
    assert not transcripts_agree("set the thermostat to 20", "set the thermostat to 70")
    assert not transcripts_agree("lock the front door", "unlock the front door")
    assert transcripts_agree("Turn the TV off.", "turn the tv off")
    print("PASS: the backstop only acts when two transcribers agree")


class _FallbackOnly(VoicePELiveService):
    """The transcription-fallback segment tracker without the websocket machinery."""

    def __init__(self):
        self._fallback_enabled = True
        self._open_function_calls = {}
        self._request_known = False
        self._answer_text_started = False
        self._reset_fallback()


def _mic_frame(ms: int = 20) -> InputAudioRawFrame:
    return InputAudioRawFrame(
        audio=bytes(24000 * 2 * ms // 1000), sample_rate=24000, num_channels=1
    )


async def test_every_utterance_is_bounded_for_checking():
    """Each thing said is bounded on its own, the follow-ups included.

    A blip shorter than 100 ms does not open one, a pause shorter than
    700 ms does not close one. The first is cut from the wake, since Silero
    flags only about half of a short far-field question; a follow-up is cut
    from just before itself, so it is not read together with everything
    already said and answered.
    """
    svc = _FallbackOnly()

    def feed(speech: bool, ms: int):
        result = None
        for _ in range(ms // 20):
            got = svc._track_fallback_segment(_mic_frame(), speech)
            if got is not None:
                result = got
        return result

    assert feed(False, 500) is None  # room noise before the question
    assert feed(True, 60) is None  # a blip shorter than the attack
    assert feed(False, 200) is None
    assert feed(True, 1500) is None  # the question
    assert feed(False, 400) is None  # a mid-sentence pause
    assert feed(True, 500) is None  # ...the question continues
    first = feed(False, 1000)  # quiet: the utterance is over
    assert first is not None, "the utterance never closed"
    start, end = first
    assert abs(start - 0.76) < 0.03 and abs(end - 3.16) < 0.03, first
    assert svc._utterances == 1
    # The first is read from the wake, so nothing of it can be missed.
    assert len(svc._fallback_slice(start, end)) == int((end + 0.5) * 48000)

    # A follow-up a few seconds later is bounded too, and read alone.
    assert feed(False, 3000) is None
    assert feed(True, 1200) is None
    second = feed(False, 1000)
    assert second is not None, "the follow-up was never bounded"
    s2, e2 = second
    assert s2 > end, second
    assert svc._utterances == 2
    clip = svc._fallback_slice(s2, e2)
    assert len(clip) == int((e2 + 0.5) * 48000) - int((s2 - 0.5) * 48000), "cut around the follow-up"
    assert len(clip) < int((e2 + 0.5) * 48000), "not the whole conversation"
    print("PASS: every utterance is bounded, the first from the wake")


async def test_fallback_gives_up_on_speech_that_never_stops():
    """A TV does not go quiet; the utterance is closed at the cap rather than never."""
    svc = _FallbackOnly()
    seg = None
    for _ in range(8000 // 20):
        got = svc._track_fallback_segment(_mic_frame(), True)
        if got is not None:
            seg = got
    assert seg is not None, "an endless utterance was never closed"
    assert abs(seg[1] - seg[0] - 6.0) < 0.03, seg
    print("PASS: an utterance that never goes quiet is closed at the cap")


def test_filler_is_muted_while_the_backend_works_and_the_answer_is_not():
    """Filler is dropped, the answer is heard whole, and no word is cut.

    Built from the orders measured on 2026-09-19: filler while the backend
    works; the answer arriving all loud after silence; the answer starting
    inside the settle moment after the backend finished.
    """
    f = _OutputSilenceFilter.__new__(_OutputSilenceFilter)
    state = {"v": (False, False)}
    f._output_hold = lambda: state["v"]
    f._muting = False
    f._muted_frames = 0
    f._run, f._runs, f._run_keep = [], 0, False
    f._passed_tape = f._dropped_tape = None
    f._last_loud_at = 0.0
    LOUD = 2000.0
    idle, working, settling, unheard = (False, False), (True, False), (True, True), (True, False)

    t = 10.0
    assert f._admit("a", LOUD, t) == ["a"], "speech with nothing pending passes"

    state["v"] = working
    t += 0.5
    assert f._admit("filler1", LOUD, t) == [], "filler while the backend works is dropped"
    t += 0.1
    assert f._admit("filler2", LOUD, t) == [], "the rest of it too"

    # The answer starts inside the settle moment, then the hold lifts.
    state["v"] = settling
    t += 0.5
    assert f._admit("ans1", LOUD, t) == [], "held a moment..."
    t += 0.02
    assert f._admit("ans2", LOUD, t) == [], "...still held..."
    state["v"] = idle
    t += 0.02
    assert f._admit("ans3", LOUD, t) == ["ans1", "ans2", "ans3"], "...then replayed whole"

    # The answer starts while the backend still says it is working, and the
    # release lands mid-word: the words already spoken are not lost.
    state["v"] = working
    t += 1.0
    assert f._admit("filler", LOUD, t) == [], "filler dropped"
    t += 0.5  # a gap, then the answer begins while still held
    assert f._admit("a1", LOUD, t) == []
    t += 0.02
    assert f._admit("a2", LOUD, t) == []
    state["v"] = idle
    t += 0.02
    assert f._admit("a3", LOUD, t) == ["a1", "a2", "a3"], "the answer is sent whole"

    # Not heard yet, and the check then confirms the model heard right: what
    # it said while the check ran is the answer, and is sent, not lost.
    unverified = (True, True)
    state["v"] = unverified
    t += 1.0
    assert f._admit("maybe1", LOUD, t) == []
    t += 0.02
    assert f._admit("maybe2", LOUD, t) == []
    state["v"] = idle
    t += 0.02
    assert f._admit("maybe3", LOUD, t) == ["maybe1", "maybe2", "maybe3"], "confirmed: kept"

    # Not heard yet, and the check corrects it: what it said was an answer to
    # something that was never asked, and is thrown away.
    state["v"] = unverified
    t += 1.0
    assert f._admit("wrong1", LOUD, t) == []
    state["v"] = unheard  # the correction goes in; keep turns off
    t += 0.02
    assert f._admit("wrong2", LOUD, t) == []
    state["v"] = idle
    t += 3.0  # nothing at all arrives in between
    assert f._admit("answer", LOUD, t) == ["answer"], "corrected: only the real answer passes"

    # The hold arrives while the model is mid-word: the word is not cut.
    state["v"] = working
    t += 0.05
    assert f._admit("mid", LOUD, t) == ["mid"], "must not mute mid-utterance"
    print("PASS: filler dropped, answers whole, no word cut")


class _HoldOnly(VoicePELiveService):
    """The output-hold state without the websocket machinery."""

    def __init__(self):
        self._request_known = False
        self._answer_text_started = False
        self._live_heard_at = None
        self._held_since = None
        self._live_open_responses = set()
        self._response_started = {}
        self._backstop_at = None
        self._current_response_at = 0.0
        self._last_response_done = 0.0
        self._user_turn_seen = False


async def test_hold_lifts_when_the_answer_text_starts():
    """Replays the backend events of a real "what time is it" (2026-09-19).

    Held while unheard and through the tool call; lifted the moment the
    backend starts writing the answer, before it has formally finished.
    """
    s = _HoldOnly()

    async def evt(inner, key):
        VoicePELiveService._follow_response(s, types.SimpleNamespace(
            inner_type=inner, type="response.event",
            event={"response": {"id": key}} if key else {},
            delegation_id=None, item_id=key, event_id=None,
        ))

    assert s.output_hold()[0], "held before the request is heard"
    s._request_known = True
    await evt("response.created", "r1")
    assert s.output_hold() == (True, False), "held while the backend calls a tool"
    await evt("response.completed", "r1")
    await evt("response.created", "r2")
    assert s.output_hold()[0], "still held: no answer text yet"
    await evt("response.content_part.added", "item2")
    assert s.output_hold() == (False, False), "answer text: the model may speak"
    print("PASS: the hold lifts when the backend starts writing the answer")


class _TrackerOnly(_HoldOnly):
    """The real delegation tracker, without the websocket machinery."""

    async def _handle_evt_response(self, evt):
        # The parent chain would need a live session; the tracking is ours.
        self._follow_response(evt)


async def test_delegation_tracker_opens_and_closes():
    svc = _TrackerOnly()
    assert not svc.delegation_in_flight
    await svc._handle_evt_response(_response_evt("response.created", "resp_a"))
    assert svc.delegation_in_flight
    await svc._handle_evt_response(_response_evt("response.created", "resp_b"))
    await svc._handle_evt_response(_response_evt("response.completed", "resp_a"))
    assert svc.delegation_in_flight, "one of two closed should still hold"
    await svc._handle_evt_response(_response_evt("response.failed", "resp_b"))
    assert not svc.delegation_in_flight, "all closed should release"
    print("PASS: delegation tracker opens and closes by response id")


async def test_stale_completion_cannot_release_a_new_session():
    """The bug a plain counter would have: a late completion stealing a hold.

    Session A opens a delegation and goes away. Session B opens its own. A's
    completion then arrives late. With a counter that decrement would release
    B's hold and let the watcher hang up mid-answer; keyed by id and cleared
    per session, it is ignored.
    """
    svc = _TrackerOnly()
    await svc._handle_evt_response(_response_evt("response.created", "resp_old"))
    svc._live_open_responses.clear()  # what begin_live_session() does
    await svc._handle_evt_response(_response_evt("response.created", "resp_new"))
    assert svc.delegation_in_flight
    await svc._handle_evt_response(_response_evt("response.completed", "resp_old"))
    assert svc.delegation_in_flight, "a stale completion released the new session's hold"
    await svc._handle_evt_response(_response_evt("response.completed", "resp_new"))
    assert not svc.delegation_in_flight
    print("PASS: a stale completion cannot release the new session's hold")


async def test_unidentifiable_completion_does_not_hold_forever():
    """An envelope with no id must not wedge the hold open."""
    svc = _TrackerOnly()
    await svc._handle_evt_response(_response_evt("response.created", "resp_a"))
    await svc._handle_evt_response(_response_evt("response.completed", None))
    assert not svc.delegation_in_flight, "an id-less completion left the hold stuck"
    # And one with nothing open is harmless.
    await svc._handle_evt_response(_response_evt("response.completed", None))
    assert not svc.delegation_in_flight
    print("PASS: an id-less completion releases a hold instead of wedging it")


def _hygiene(*, window=1.0, budget=0, cap=0, delegating=lambda: False):
    """A `_LiveHygiene` with its state set up, bypassing pipeline wiring.

    FrameProcessor.__init__ wants a live pipeline, and none of these tests
    push frames, so the attributes the watcher reads are set directly.
    """
    h = _LiveHygiene.__new__(_LiveHygiene)
    h._config = types.SimpleNamespace(
        followup_window_seconds=window,
        max_turns_per_wake=budget,
        max_live_session_seconds=cap,
    )
    h._get_ws = lambda: None
    h._is_delegating = delegating
    h._connected = True
    loop_time = asyncio.get_event_loop().time()
    h._connect_time = loop_time
    h._quiet_since = loop_time
    h._bot_speaking = False
    h._user_speaking = False
    h._reply_pending = False
    h._reply_pending_since = 0.0
    h._bot_spoke = True
    h._turns = 1
    h._close_requested = False
    h._watch = None
    h._gen = 0
    h.on_close = None
    h.closed = []

    async def _close(reason):
        h.closed.append(reason)

    h._close = _close
    return h


async def test_delegation_holds_the_window():
    """A pause while the backend works must not be read as a finished reply.

    This is the truncation bug: the model says part of an answer, goes quiet
    to delegate, and the follow-up window expires during the pause, so the
    device is disconnected before the rest of the sentence arrives.
    """
    delegating = {"v": True}
    h = _hygiene(window=1.0, delegating=lambda: delegating["v"])
    task = asyncio.create_task(_LiveHygiene._watch_loop(h))
    await asyncio.sleep(2.5)  # well past the 1s window
    assert not h.closed, f"closed mid-delegation: {h.closed}"
    delegating["v"] = False
    await asyncio.sleep(2.5)
    assert h.closed, "never closed after the delegation finished"
    assert "follow-up window" in h.closed[0], h.closed[0]
    task.cancel()
    print("PASS: delegation holds the follow-up window open, then it closes")


async def test_window_closes_when_idle():
    """With nothing in flight, the window still closes on time."""
    h = _hygiene(window=1.0)
    task = asyncio.create_task(_LiveHygiene._watch_loop(h))
    await asyncio.sleep(2.0)
    assert h.closed, "idle session was never closed"
    task.cancel()
    print("PASS: idle session closes on the follow-up window")


async def test_pending_reply_defers_then_fails_open():
    """A committed question defers the window, but not forever.

    A reply that never arrives (an ignored TV line, a dropped response) must
    not pin a per-minute meter open.
    """
    h = _hygiene(window=0.5)
    h._reply_pending = True
    h._reply_pending_since = asyncio.get_event_loop().time()
    task = asyncio.create_task(_LiveHygiene._watch_loop(h))
    await asyncio.sleep(1.5)
    assert not h.closed, f"closed while a reply was owed: {h.closed}"
    # Pretend the overdue threshold has passed.
    h._reply_pending_since -= 100.0
    await asyncio.sleep(1.5)
    assert h.closed, "never failed open on an overdue reply"
    task.cancel()
    print("PASS: pending reply defers the window, then fails open")


async def test_fresh_wake_waits_for_the_first_reply():
    """A new session starts with a reply owed, and closes if none ever comes.

    The device connects because the wake word fired, so the question is
    still being spoken when the old follow-up clock (started at the chime)
    ran out at 6s with "0 turns"; every wake on the device died that way.
    The user-turn frames the turn counter relies on never fire for the
    device's audio, so zero turns is the normal count and cannot be used.
    """
    h = _hygiene(window=0.5)
    h._reply_pending = True
    h._reply_pending_since = asyncio.get_event_loop().time()
    h._bot_spoke = False
    task = asyncio.create_task(_LiveHygiene._watch_loop(h))
    await asyncio.sleep(1.5)  # well past the 0.5s window
    assert not h.closed, f"hung up before the first reply could come: {h.closed}"
    h._reply_pending_since -= 100.0  # overdue, and nothing was ever answered
    await asyncio.sleep(1.5)
    assert h.closed and "no reply" in h.closed[0], h.closed
    task.cancel()
    print("PASS: a fresh wake waits for the first reply, then closes if none comes")


async def test_hard_cap_fires_regardless():
    """The cost fuse ignores speech and delegation state."""
    h = _hygiene(window=0, cap=1, delegating=lambda: True)
    h._bot_speaking = True
    h._user_speaking = True
    h._reply_pending = True
    h._reply_pending_since = asyncio.get_event_loop().time()
    task = asyncio.create_task(_LiveHygiene._watch_loop(h))
    await asyncio.sleep(2.0)
    assert h.closed, "hard cap did not fire"
    assert "hard cap" in h.closed[0], h.closed[0]
    task.cancel()
    print("PASS: hard cap fires through speech, delegation and a pending reply")


async def test_turn_budget_waits_for_the_reply():
    """The budget must not hang up the instant the last question lands."""
    h = _hygiene(window=0, budget=1)
    h._reply_pending = True
    h._reply_pending_since = asyncio.get_event_loop().time()
    task = asyncio.create_task(_LiveHygiene._watch_loop(h))
    await asyncio.sleep(1.5)
    assert not h.closed, f"budget closed before the reply: {h.closed}"
    h._reply_pending = False
    await asyncio.sleep(1.0)
    assert h.closed and "turn budget" in h.closed[0], h.closed
    task.cancel()
    print("PASS: turn budget waits for the owed reply")




async def test_a_short_reply_is_not_held_waiting_for_more():
    """A reply shorter than the reserve still goes out, on its own clock.

    The model can stop sending audio altogether after a short answer, so a
    reserve released only when the next frame arrives would hold "Done."
    for ever: it is released by a timer the filter owns.
    """
    f = _OutputSilenceFilter.__new__(_OutputSilenceFilter)
    f._dropped = f._sent = f._burst_frames = 0
    f._last_sent_at = f._last_loud_at = 0.0
    f._preroll, f._preroll_seconds, f._preroll_task = [], 0.0, None
    sent, timers = [], []

    async def _push(frame, direction):
        sent.append(frame)

    def _create_task(coro, *a, **kw):
        timers.append(coro)  # the deadline, fired by hand below
        return None

    f.push_frame = _push
    f.create_task = _create_task

    class _Frame:
        num_channels, sample_rate = 1, 24000
        audio = b"\x10\x00" * 480  # 20 ms, loud

    t = 100.0
    for _ in range(10):  # 200 ms of speech, far short of the reserve
        await f._forward(_Frame(), 2000.0, t, None)
        t += 0.02
    assert not sent, "held while the reserve fills"
    assert len(timers) == 1, "a deadline is armed when the reserve starts filling"

    await f._flush_preroll(None)  # what the deadline does when it fires
    assert len(sent) == 10, f"the whole short reply goes out: {len(sent)}"
    assert not f._preroll
    for coro in timers:
        coro.close()
    print("PASS: a short reply is not held waiting for more")


def test_no_accidental_overrides_of_pipecat():
    """Every name we share with pipecat's Live service is a deliberate override.

    A helper that happens to share a name with a pipecat method silently
    replaces it: `_track_response` did, pipecat got None where it expected
    a key, and every tool call crashed the receive loop (2026-09-19). A new
    override must be added here on purpose.
    """
    from pipecat.services.openai.live.llm import OpenAILiveLLMService

    ours = {n for n in VoicePELiveService.__dict__ if not n.startswith("__")}
    parent = {n for c in OpenAILiveLLMService.__mro__ for n in c.__dict__}
    intended = {
        "_abc_impl",
        "_end_turn",
        "_handle_evt_response",
        "_handle_evt_session_started",
        "_invocation_params",
        "_open_turn",
        "_run_function_call",
        "_send_session_config",
        "_send_user_audio",
        "push_error",
    }
    unexpected = (ours & parent) - intended
    assert not unexpected, f"shadows a pipecat method by accident: {sorted(unexpected)}"
    print("PASS: no accidental overrides of pipecat")


def test_the_prompt_is_not_mistaken_for_a_request():
    """A transcriber handed silence reads its own prompt back.

    Both transcribers get the same vocabulary prompt, so this is the one
    mishearing they can agree on, and what they agree on is a command:
    left alone it would turn the lights off on an empty room.
    """
    echo = (
        "Turn the living room lights off, put Netflix on the TV, what time is it, "
        "what's the weather, play music in the den, set a timer, how's it going."
    )
    assert echoes_vocabulary(echo)
    assert echoes_vocabulary(echo.lower().replace(".", ""))
    # What someone actually says is one of those, not the list.
    assert not echoes_vocabulary("turn the living room lights off")
    assert not echoes_vocabulary("what time is it")
    assert not echoes_vocabulary("put netflix on the tv and turn the lights off")
    print("PASS: the prompt read back is not mistaken for a request")


async def test_nothing_is_said_into_a_wake_that_is_over():
    """A check that comes back after the wake ended says nothing.

    The mic runs for a few seconds after the session closes, so an
    utterance can close, and come back read, with nobody to say it to.
    Sending it anyway put an event on a closed session, OpenAI answered
    "the first Live event must be session.start", and the service was torn
    down for the rest of the night (2026-09-20).
    """
    svc = _FallbackOnly()
    said = []

    async def _append(*a, **kw):
        said.append(a)

    svc._send_context_append = _append
    svc._utterance_acted = False

    svc._device_present = False
    await svc._unreadable(first=True)
    assert not said, "spoke into a closed session"

    svc._device_present = True
    await svc._unreadable(first=True)
    assert said, "stayed silent with the device still there"

    # And a queued check is not even read back once the wake is over.
    read = []

    async def _verify(*a, **kw):
        read.append(a)

    svc._verify_utterance = _verify
    svc._verifying = False
    svc._device_present = False
    await svc._check_utterances((0.0, 1.0))
    assert not read, "transcribed for a wake that had ended"
    print("PASS: nothing is said into a wake that is over")



async def test_a_tool_that_never_answers_does_not_mute_the_model():
    """Every function call gets exactly one output, whatever the tool does.

    The MCP call is awaited with no timeout, so a Home Assistant tool that
    takes a request and never returns leaves the call open for the life of
    the socket, which is kept warm between wakes. The API then refuses
    every `response.create`, and the model is mute for that wake and for
    every wake after it (2026-09-22).
    """
    svc = _FallbackOnly()
    svc._device_present = True
    svc._TOOL_ANSWER_SECONDS = 0.02
    answered = []

    async def _out(call_id, output):
        answered.append((call_id, output))
        svc._open_function_calls.pop(call_id, None)

    svc._send_function_call_output = _out

    svc._open_function_calls["stuck"] = "response-1"
    await svc._answer_if_stuck("stuck")
    assert [c for c, _ in answered] == ["stuck"], "the stuck call was never answered"
    assert "error" in answered[0][1]
    assert not svc._open_function_calls, "the call is closed once answered"

    # A call the tool answered itself is not answered a second time: the
    # API takes one output per call and rejects the second.
    answered.clear()
    await svc._answer_if_stuck("already-done")
    assert not answered

    # And nothing is written into a wake that is over.
    svc._open_function_calls["late"] = "response-2"
    svc._device_present = False
    await svc._answer_if_stuck("late")
    assert not answered, "wrote into a closed session"
    assert "late" not in svc._open_function_calls, "left a call to poison the next wake"

    # The backstop waits for open calls, and gives up rather than hanging.
    svc._TOOL_ANSWER_SECONDS = 0.0
    svc._open_function_calls["never"] = "response-3"
    await svc._await_open_calls()
    print("PASS: a tool that never answers does not mute the model")



async def test_it_never_apologises_over_its_own_answer():
    """"I didn't catch that" is never said on top of an answer.

    The check failing to read the audio back says nothing about whether
    the model heard it. It answered "Humans live on Earth.", the two
    transcribers then disagreed with each other, and it apologised over
    the top of its own correct answer (2026-09-25).
    """
    svc = _FallbackOnly()
    svc._device_present = True
    svc._utterance_acted = False
    said = []

    async def _append(*a, **kw):
        said.append(a)

    svc._send_context_append = _append

    svc._answer_text_started = True
    await svc._unreadable(first=True)
    assert not said, "apologised over its own answer"

    # With nothing answered and nothing done, it still speaks up.
    svc._answer_text_started = False
    await svc._unreadable(first=True)
    assert said, "stayed silent when it really had not caught it"
    print("PASS: it never apologises over its own answer")



async def main():
    """Run every test in the file, in the order they are written.

    Named one by one, a test written below the entry point is never run:
    two of these sat there passing in name only until the file was read
    (2026-09-20).
    """
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        result = fn()
        if inspect.isawaitable(result):
            await result
    print("\nall live-hygiene tests passed")


if __name__ == "__main__":
    asyncio.run(main())
