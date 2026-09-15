"""Regression tests for the Live engine's session-lifetime bounds.

No network, no broker, no device: these drive `_LiveHygiene` and the
delegation counter directly, because the bug they cover only shows up in the
timing between a question and a reply, which is awkward to provoke through
the audio harness.

    python3 broker/tools/test_live_hygiene.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipecat.frames.frames import InputAudioRawFrame

from realtime_broker.live_agent import VoicePELiveService
from realtime_broker.live_server import _LiveHygiene


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


class _FallbackOnly(VoicePELiveService):
    """The transcription-fallback segment tracker without the websocket machinery."""

    def __init__(self):
        self._fallback_enabled = True
        self._reset_fallback()


def _mic_frame(ms: int = 20) -> InputAudioRawFrame:
    return InputAudioRawFrame(
        audio=bytes(24000 * 2 * ms // 1000), sample_rate=24000, num_channels=1
    )


async def test_fallback_tracks_only_the_first_utterance():
    """The first thing said after the wake is bounded once; the room after it is not.

    A blip shorter than 100 ms does not open it, a pause shorter than 900 ms
    does not close it, and once it has closed nothing else (a TV, a second
    remark) is ever handed to the model as text: the wake word gated one
    request.
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
    seg = feed(False, 1000)  # quiet: the utterance is over
    assert seg is not None, "the utterance never closed"
    start, end = seg
    assert abs(start - 0.76) < 0.03 and abs(end - 3.16) < 0.03, seg
    assert len(svc._fallback_slice(end)) == int((end + 0.5) * 48000)  # from the wake, not Silero
    assert feed(True, 1500) is None and feed(False, 1000) is None, "a second utterance was bounded"
    print("PASS: only the first utterance after the wake is bounded for transcription")


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


class _TrackerOnly(VoicePELiveService):
    """The delegation tracker without any of the websocket machinery."""

    def __init__(self):
        self._live_open_responses = set()

    async def _handle_evt_response(self, evt):
        # Skip the parent chain, which would need a live session.
        inner = getattr(evt, "inner_type", None)
        key = self._response_key(evt)
        if inner == "response.created":
            if key is not None:
                self._live_open_responses.add(key)
        elif inner in self._RESPONSE_DONE:
            if key is not None:
                self._live_open_responses.discard(key)
            elif self._live_open_responses:
                self._live_open_responses.pop()


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


async def main():
    await test_delegation_tracker_opens_and_closes()
    await test_stale_completion_cannot_release_a_new_session()
    await test_unidentifiable_completion_does_not_hold_forever()
    await test_delegation_holds_the_window()
    await test_window_closes_when_idle()
    await test_fresh_wake_waits_for_the_first_reply()
    await test_pending_reply_defers_then_fails_open()
    await test_hard_cap_fires_regardless()
    await test_turn_budget_waits_for_the_reply()
    print("\nall live-hygiene tests passed")


if __name__ == "__main__":
    asyncio.run(main())
