"""Make the broker look like GPT-Live to OpenAI's eval harness.

The harness talks to a GPT-Live endpoint; the broker only talks to a Voice PE
puck: raw PCM16 at 24 kHz over a WebSocket, where connecting is the wake.
This sits between them on :8790 and speaks just enough of the Live v3 session
protocol for the harness to grade the broker exactly as it grades raw Live:

- session.start is answered with session.started, and a device connection
  to the broker is opened (the wake).
- session.input_audio.append frames go to the broker as raw PCM.
- Broker audio comes back as session.output_audio.delta.
- Each side's speech is transcribed once it goes quiet and sent as
  session.{input,output}_transcript.delta with its timeline, since the
  harness finds turns from transcripts and the broker sends none.
- The first tool call the fake house logs is reported as a delegation, so
  "did it hand off to the backend" is scored from what actually happened.

Tool calls and final state are read by the harness from the fake house
directly (assistants/smart_home/remote.py), never relayed through here.

    broker/.venv-live/bin/python evals/broker_adapter.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import time
import urllib.request
import uuid
from array import array

import websockets
import contextlib

PORT = 8790
BROKER_WS = "ws://127.0.0.1:8766"
HOUSE = "http://127.0.0.1:8791"
RATE = 24000
SPEECH_RMS = 300
# The broker streams its speaker continuously, silence included; GPT-Live only
# sends audio while it is talking, and the harness treats any output audio as
# the assistant still holding the floor. So only voiced broker frames are
# relayed, and an unending silent tail cannot keep a turn open forever.
VOICED_RMS = 150
INPUT_QUIET_MS = 700
OUTPUT_QUIET_MS = 1000
DELEGATION_IDLE_S = 1.5

ENV = pathlib.Path(__file__).resolve().parent.parent / "broker" / ".env"
KEY = next(
    line.split("=", 1)[1].strip().strip('"')
    for line in ENV.read_text().splitlines()
    if line.startswith("OPENAI_API_KEY=")
)


def rms(pcm: bytes) -> float:
    samples = array("h", pcm[: len(pcm) - len(pcm) % 2])
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 if samples else 0.0


def wav(pcm: bytes) -> bytes:
    import struct

    n = len(pcm)
    head = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
    head += struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
    return head + b"data" + struct.pack("<I", n) + pcm


def transcribe(pcm: bytes) -> str:
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in (("model", "gpt-4o-transcribe"), ("language", "en")):
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav(pcm) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["text"].strip()


def house_calls() -> int:
    with urllib.request.urlopen(HOUSE + "/executions", timeout=5) as r:
        return len(json.loads(r.read()))


def last_call_summary() -> str:
    with urllib.request.urlopen(HOUSE + "/executions", timeout=5) as r:
        last = json.loads(r.read())[-1]
    return f"{last['name']} {last['status']}: {json.dumps(last['output'])[:300]}"


class Segment:
    """One stretch of speech on one side, timed on the session clock."""

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.start_ms: int | None = None
        self.end_ms = 0
        self.last_voice_at = 0.0


class Session:
    def __init__(self, client) -> None:
        self.client = client
        self.t0 = time.monotonic()
        self.id = f"broker_{uuid.uuid4().hex[:12]}"
        self.send_lock = asyncio.Lock()
        self.pending: set[asyncio.Task] = set()
        self.closed = False

    def now_ms(self) -> int:
        return round((time.monotonic() - self.t0) * 1000)

    async def emit(self, event: dict) -> None:
        async with self.send_lock:
            await self.client.send(json.dumps(event))

    def spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def report(self, kind: str, seg: Segment, end_ms: int) -> None:
        pcm, start_ms = bytes(seg.pcm), seg.start_ms or 0
        try:
            text = await asyncio.to_thread(transcribe, pcm)
        except Exception as exc:  # noqa: BLE001 - a lost transcript must not end the session
            print(f"[{self.id}] transcription failed: {exc}")
            return
        if text and not self.closed:
            print(f"[{self.id}] {kind.split('.')[1].split('_')[0]:6} {start_ms:>6}..{end_ms:<6} {text}")
            await self.emit({"type": kind, "start_ms": start_ms, "end_ms": end_ms, "delta": text,
                             "event_id": f"event_{uuid.uuid4().hex[:16]}"})


async def handle(client) -> None:
    start = json.loads(await client.recv())
    if start.get("type") != "session.start":
        await client.close(code=1002, reason="expected session.start")
        return
    s = Session(client)
    session_info = {"id": s.id, "model": start.get("session", {}).get("model", "broker"),
                    "instructions": start.get("session", {}).get("instructions", "")}
    await s.emit({"type": "session.started", "client_event_id": start.get("event_id"), "session": session_info})
    baseline_calls = await asyncio.to_thread(house_calls)
    broker = await websockets.connect(BROKER_WS, max_size=None)
    print(f"[{s.id}] wake: connected to the broker")

    inp, out = Segment(), Segment()
    in_samples = 0

    async def uplink() -> None:
        nonlocal inp, in_samples
        async for raw in client:
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "session.close":
                return
            if kind != "session.input_audio.append":
                continue
            pcm = base64.b64decode(event["audio"])
            with contextlib.suppress(websockets.ConnectionClosed):
                await broker.send(pcm)
            at_ms = round(in_samples * 1000 / RATE)
            in_samples += len(pcm) // 2
            if rms(pcm) >= SPEECH_RMS:
                if inp.start_ms is None:
                    inp.start_ms = at_ms
                inp.last_voice_at = time.monotonic()
            if inp.start_ms is not None:
                inp.pcm.extend(pcm)
                if (time.monotonic() - inp.last_voice_at) * 1000 >= INPUT_QUIET_MS:
                    s.spawn(s.report("session.input_transcript.delta", inp, at_ms - INPUT_QUIET_MS))
                    inp = Segment()

    out_end_ms = 0  # where the harness has placed the end of our output audio

    async def downlink() -> None:
        nonlocal out, out_end_ms
        try:
            async for msg in broker:
                if not isinstance(msg, bytes) or s.closed or rms(msg) < VOICED_RMS:
                    continue
                # Mirror the harness's own placement: a chunk plays at the later
                # of "now" and the end of the previous one. Transcript timings
                # are taken from this, so the turn it projects ends exactly
                # where it heard the last audio, which is its completion test.
                start = max(out_end_ms, s.now_ms())
                out_end_ms = start + round(len(msg) / 2 * 1000 / RATE)
                if out.start_ms is None:
                    out.start_ms = start
                out.end_ms = out_end_ms
                out.pcm.extend(msg)
                out.last_voice_at = time.monotonic()
                await s.emit({"type": "session.output_audio.delta", "delta": base64.b64encode(msg).decode()})
        except websockets.ConnectionClosed:
            pass
        print(f"[{s.id}] the broker closed the device connection")

    async def watch() -> None:
        nonlocal out
        # A delegation, as GPT-Live reports one: opened with a response id,
        # then response.created / response.completed wrapped in response.event.
        # The harness holds the turn open while a delegation is active, so it
        # must be closed: here, once the house has seen no new tool call for
        # DELEGATION_IDLE_S. Without the close every tool scenario timed out.
        delegation_id = f"deleg_{s.id}"
        response_id = f"resp_{s.id}"
        calls_seen, last_call_at, open_ = baseline_calls, 0.0, False

        async def lifecycle(kind: str, status: str) -> None:
            await s.emit({"type": "response.event", "delegation_id": delegation_id,
                          "event_id": f"event_{uuid.uuid4().hex[:16]}",
                          "event": {"type": kind, "response": {"id": response_id, "status": status, "output": []}}})

        while not s.closed:
            await asyncio.sleep(0.2)
            if out.start_ms is not None and (time.monotonic() - out.last_voice_at) * 1000 >= OUTPUT_QUIET_MS:
                s.spawn(s.report("session.output_transcript.delta", out, out.end_ms))
                out = Segment()
            calls = await asyncio.to_thread(house_calls)
            if calls > calls_seen:
                calls_seen, last_call_at = calls, time.monotonic()
                if not open_ and last_call_at:
                    open_ = True
                    await s.emit({"type": "session.delegation.created", "offset_ms": s.now_ms(),
                                  "event_id": f"event_{uuid.uuid4().hex[:16]}",
                                  "delegation": {"id": delegation_id, "type": "delegation",
                                                 "response_id": response_id, "target": "responses"}})
                    await lifecycle("response.created", "in_progress")
                # The backend's returned text, sent when the call lands: the
                # fake house answers instantly, which is when a real backend
                # would hand its result back. The harness only accepts an
                # assistant turn that starts after returned text as the reply.
                await s.emit({"type": "response.event", "delegation_id": delegation_id,
                              "event_id": f"event_{uuid.uuid4().hex[:16]}",
                              "event": {"type": "response.output_text.done", "response_id": response_id,
                                        "item_id": f"msg_{s.id}_{calls}",
                                        "text": await asyncio.to_thread(last_call_summary)}})
            if open_ and time.monotonic() - last_call_at >= DELEGATION_IDLE_S:
                open_ = False
                await lifecycle("response.completed", "completed")

    tasks = [asyncio.create_task(downlink()), asyncio.create_task(watch())]
    try:
        await uplink()
    except websockets.ConnectionClosed:
        pass
    finally:
        s.closed = True
        for t in tasks:
            t.cancel()
        await broker.close()
        if s.pending:
            await asyncio.wait(s.pending, timeout=5)
        with contextlib.suppress(websockets.ConnectionClosed):
            await s.emit({"type": "session.closed", "reason": "close_requested", "session": session_info,
                          "usage": {"seconds": round(s.now_ms() / 1000, 1)}})
        print(f"[{s.id}] closed after {s.now_ms()} ms")


async def main() -> None:
    async with websockets.serve(handle, "127.0.0.1", PORT, max_size=None):
        print(f"broker adapter on ws://127.0.0.1:{PORT} -> {BROKER_WS}, house {HOUSE}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
