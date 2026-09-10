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

from realtime_broker.live_server import _LiveHygiene  # noqa: E402


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
    await test_delegation_holds_the_window()
    await test_window_closes_when_idle()
    await test_pending_reply_defers_then_fails_open()
    await test_hard_cap_fires_regardless()
    await test_turn_budget_waits_for_the_reply()
    print("\nall live-hygiene tests passed")


if __name__ == "__main__":
    asyncio.run(main())
