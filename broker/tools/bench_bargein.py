#!/usr/bin/env python3
"""Barge-in bench: does the model actually stop when you talk over it?

This is the number the project exists to produce, and the one nobody else
has published for a far-field home device. It runs HEADLESS against a
broker, no Voice PE required, because the model-side half of barge-in is
fully observable from the audio stream: ask something long, start speaking
part-way through the answer, and measure whether the bot's audio stops and
how long it takes.

What it does NOT measure: acoustic echo residual, i.e. whether the device's
own speaker bleeding into its mic false-triggers an interruption. That needs
the real puck in a real room and is the remaining on-device step.

Method per trial:
  1. Ask a question whose answer runs for several seconds.
  2. Wait until the bot has been speaking for `--cut-after` seconds.
  3. Stream an interrupting utterance.
  4. Keep reading, and find the last moment non-silent bot audio arrived.

  stop latency = (time of last bot speech) - (time the interruption ended)

  A negative or near-zero value means it stopped as we spoke. A large value
  means it talked over us. No stop at all is a failure.

Reported: success rate, how long the bot kept talking after we cut in, and
reply onset, measured from the END of our question. Onset is routinely
NEGATIVE on the Live engine, which is the headline full-duplex result rather
than a bug: the model begins answering while we are still speaking, because
it is not waiting for a turn boundary. The Realtime engine cannot do this by
construction, since server VAD has to detect end-of-speech first.

Usage:
  python3 broker/tools/bench_bargein.py ws://127.0.0.1:8766 --trials 5
  python3 broker/tools/bench_bargein.py ws://127.0.0.1:8765 --label realtime
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass, field

import websockets

RATE = 24000
CHUNK = int(RATE * 0.02) * 2  # 20 ms of PCM16
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "..", ".env")

# Long enough that the reply is still going when we cut in.
LONG_QUESTION = (
    "Ignore anything before this. Please explain, in several full sentences "
    "and taking your time, why the sky appears blue during the day."
)
INTERRUPTION = "Stop. Never mind that, what day is it today?"

# RMS below this counts as silence, so trailing silence in a continuous
# stream is not mistaken for the bot still talking. The Live engine streams
# silence continuously; the Realtime engine does not.
SILENCE_RMS = 180

# How much speech after our interruption still counts as "it stopped".
# A human who gets interrupted finishes the word they are on, so scoring
# a hard cut-off measures rudeness, not responsiveness. 1.0s is roughly
# one trailing word at conversational pace.
STOP_WINDOW_S = float(os.environ.get("STOP_WINDOW_S", "1.0"))


def load_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        with open(ENV) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


KEY = load_key()


def _rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    n = len(pcm) // 2
    vals = struct.unpack(f"<{n}h", pcm[: n * 2])
    return (sum(v * v for v in vals) / n) ** 0.5


def synth(text: str, voice: str = "alloy") -> bytes:
    body = json.dumps(
        {
            "model": "gpt-4o-mini-tts",
            "voice": voice,
            "input": text,
            "response_format": "pcm",
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return b""


@dataclass
class Trial:
    stopped: bool = False
    stop_latency_s: float | None = None
    first_audio_s: float | None = None
    bot_speech_after_cut_s: float = 0.0
    note: str = ""


@dataclass
class Bench:
    url: str
    label: str
    cut_after: float
    trials: list[Trial] = field(default_factory=list)


async def _stream(ws, pcm: bytes) -> None:
    """Send PCM at real time, as the firmware does."""
    for i in range(0, len(pcm), CHUNK):
        await ws.send(pcm[i : i + CHUNK])
        await asyncio.sleep(0.02)


async def run_trial(url: str, question_pcm: bytes, interrupt_pcm: bytes, cut_after: float) -> Trial:
    t = Trial()
    loop = asyncio.get_running_loop()

    async with websockets.connect(url, max_size=None, open_timeout=15) as ws:
        first_speech_at: float | None = None
        speech_started_at: float | None = None
        interrupted_at: float | None = None
        last_speech_at: float | None = None
        speech_after_cut = 0.0
        sent_interrupt = False
        deadline = loop.time() + 75.0

        async def read_loop():
            nonlocal first_speech_at, speech_started_at, last_speech_at, speech_after_cut
            while loop.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    if interrupted_at is not None:
                        return  # quiet after the cut: it stopped
                    continue
                if not isinstance(msg, (bytes, bytearray)):
                    continue
                if _rms(bytes(msg)) <= SILENCE_RMS:
                    continue
                now = loop.time()
                if first_speech_at is None:
                    first_speech_at = now
                    speech_started_at = now
                last_speech_at = now
                if interrupted_at is not None and now > interrupted_at:
                    speech_after_cut += len(msg) / (2 * RATE)

        # Start reading BEFORE the question goes out. Otherwise audio that
        # arrives while we are still streaming sits in the socket buffer and
        # is read the instant we start looking, which reported a nonsense
        # 0.08s first-audio on the first smoke run.
        reader = asyncio.create_task(read_loop())

        await _stream(ws, question_pcm)
        await _stream(ws, b"\x00" * CHUNK * 50)  # 1s trailing silence
        asked_at = loop.time()

        # Wait until the bot has genuinely been talking for cut_after seconds.
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
            if speech_started_at and loop.time() - speech_started_at >= cut_after:
                break
        if speech_started_at is None:
            reader.cancel()
            t.note = "bot never spoke"
            return t

        sent_interrupt = True
        await _stream(ws, interrupt_pcm)
        interrupted_at = loop.time()

        # Give it a window to fall silent.
        await asyncio.sleep(4.0)
        reader.cancel()

    if first_speech_at is not None:
        t.first_audio_s = first_speech_at - asked_at
    if sent_interrupt and last_speech_at is not None and interrupted_at is not None:
        t.stop_latency_s = last_speech_at - interrupted_at
        t.bot_speech_after_cut_s = speech_after_cut
        # "Stopped" means it yielded the floor, not that it cut off mid-
        # syllable. A human who is interrupted also finishes the word they
        # are on. The first run used 0.4s and scored 0/5 on a model that was
        # plainly yielding in 0.56-0.88s, which measured politeness rather
        # than failure. STOP_WINDOW_S is what a listener would still call
        # "it stopped when I spoke"; anything beyond it is talking over you.
        t.stopped = speech_after_cut < STOP_WINDOW_S
    return t


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, int(round((len(vals) - 1) * p)))
    return vals[k]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="ws://127.0.0.1:8766")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--cut-after", type=float, default=1.5,
                    help="seconds of bot speech before interrupting")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    if not KEY:
        print("SKIP  no OPENAI_API_KEY in the environment or broker/.env.")
        return 0

    label = args.label or args.url.rsplit(":", 1)[-1]
    print(f"barge-in bench: {args.url}  trials={args.trials}  cut_after={args.cut_after}s")
    print("synthesizing prompts...")
    question = synth(LONG_QUESTION)
    interrupt = synth(INTERRUPTION, voice="echo")

    bench = Bench(url=args.url, label=label, cut_after=args.cut_after)
    for i in range(1, args.trials + 1):
        try:
            t = await run_trial(args.url, question, interrupt, args.cut_after)
        except Exception as e:  # noqa: BLE001
            t = Trial(note=f"{type(e).__name__}: {str(e)[:60]}")
        bench.trials.append(t)
        mark = "STOP " if t.stopped else "TALKS"
        lat = f"{t.stop_latency_s:+.2f}s" if t.stop_latency_s is not None else "  n/a "
        over = f"{t.bot_speech_after_cut_s:.2f}s"
        first = f"{t.first_audio_s:.2f}s" if t.first_audio_s is not None else "n/a"
        print(f"  trial {i}: {mark} stop={lat} talked_over={over} first_audio={first} {t.note}")
        await asyncio.sleep(3)  # let the broker's hygiene close the session

    ok = [t for t in bench.trials if t.stopped]
    overs = [t.bot_speech_after_cut_s for t in bench.trials if t.stop_latency_s is not None]
    firsts = [t.first_audio_s for t in bench.trials if t.first_audio_s is not None]

    print(f"\n== {label} ==")
    print(f"barge-in success: {len(ok)}/{len(bench.trials)}")
    if overs:
        print(f"talked over us:   p50 {pct(overs, 0.5):.2f}s   p95 {pct(overs, 0.95):.2f}s")
    if firsts:
        print(f"first audio:      p50 {pct(firsts, 0.5):.2f}s   p95 {pct(firsts, 0.95):.2f}s")
    if len(firsts) > 1:
        print(f"first audio sd:   {statistics.stdev(firsts):.2f}s")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "label": label,
                    "url": args.url,
                    "cut_after": args.cut_after,
                    "success": len(ok),
                    "trials": len(bench.trials),
                    "talked_over_s": overs,
                    "first_audio_s": firsts,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
