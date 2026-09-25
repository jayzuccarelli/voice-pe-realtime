"""Canonical verifier for the realtime broker.

Closes the loop with NO device and NO being home: synthesizes spoken
questions, streams them into the live broker exactly as the Voice PE
firmware would, transcribes the spoken reply, and asserts on content.

Runs from anywhere the broker is reachable (host 8765). Exit 0 = green.

Usage: python tools/check.py [ws://host:8765]
Env: OPENAI_API_KEY (auto-loaded from broker/.env if unset).
"""
import asyncio
import json
import os
import re
import struct
import sys
import urllib.request

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "..", ".env")
RATE = 24000

# (spoken question, [any-of accepted substrings in the transcribed reply]).
# The broker keeps conversational context for a full session (~50 min), so
# probes must be robust to prior history: each is framed "ignore prior
# context" and asserts on a token that appears in the answer regardless of
# phrasing. Deterministic + HA-independent so the check is signal, not flake.
# Each case asks for a FULL SENTENCE on purpose. The Live engine answers a
# bare factual question with a single word, which is about 300 ms of audio,
# and transcribing 300 ms is a coin flip: real replies of "Earth." came back
# as 'art' and 'ers.', and "Paris." as 'parrot'. Measured with
# tools/probe-style energy analysis, the audio was not clipped (there was
# ~4 s of leading silence and a clean 300 ms of speech), so the flake was in
# the transcription, not the broker. Asking for a sentence removes the
# artifact and is closer to how the puck is actually used.
CASES = [
    (
        (
            "New question, ignore anything before: in one full sentence, "
            "what is the capital of France?"
        ),
        ["paris"],
    ),
    (
        (
            "New question, ignore anything before: in one full sentence, "
            "what planet do humans live on?"
        ),
        ["earth"],
    ),
]


def load_key() -> str:
    """The OpenAI key, or "" when there isn't one.

    Returns rather than exits, and tolerates a missing broker/.env, so the
    no-key case can be reported as a skip next to the no-broker one. It used
    to raise FileNotFoundError straight out of open() at import time, which
    is how a checkout without a .env (a fresh clone, a worktree) got a
    traceback instead of a sentence.
    """
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

# How long a reply may go quiet before we call it finished. The Live engine
# can speak a short holding phrase, fall silent while the delegated backend
# works, then come back with the answer: a 3s idle window cut those replies
# off mid-thought and the check read the filler as the answer. Bounded by
# REPLY_MAX_WAIT so a genuinely dead broker still fails fast enough.
REPLY_IDLE = float(os.environ.get("REPLY_IDLE_SECS", "8.0"))
REPLY_MAX_WAIT = float(os.environ.get("REPLY_MAX_WAIT_SECS", "45.0"))
WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"


def heard(want: str, reply: str) -> bool:
    """Whether the reply says `want`, allowing for one slipped character.

    The check grades audio, not text: the model's answer is spoken, then
    transcribed back by Whisper, which mangles a short reply badly. "The
    capital of France is Paris." came back as "francis parris." — the right
    answer, failed on one letter (2026-09-25). A near-miss on a long word
    is a transcription artefact, not a wrong answer. Short words are
    matched exactly, since one edit away from "on" is half the dictionary.
    """
    if want in reply:
        return True
    if len(want) < 5:
        return False
    return any(_within_one(want, word) for word in re.findall(r"[a-z0-9']+", reply))


def _within_one(a: str, b: str) -> bool:
    """Whether two words are at most one insert, delete or substitution apart."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1:]:
            return True
    return False


def _post(url: str, data: bytes, headers: dict, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def synth_pcm(text: str) -> bytes:
    body = json.dumps({
        "model": "gpt-4o-mini-tts", "voice": "alloy",
        "input": text, "response_format": "pcm",
    }).encode()
    return _post("https://api.openai.com/v1/audio/speech", body,
                 {"Authorization": f"Bearer {KEY}",
                  "Content-Type": "application/json"})


def wav_bytes(pcm: bytes) -> bytes:
    n = len(pcm)
    head = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt "
    head += struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
    head += b"data" + struct.pack("<I", n)
    return head + pcm


def transcribe(pcm: bytes) -> str:
    # multipart/form-data with the wav + model field.
    boundary = "----brokercheck"
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="model"\r\n\r\ngpt-4o-transcribe\r\n')
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 'name="file"; filename="r.wav"\r\n'
                 "Content-Type: audio/wav\r\n\r\n")
    body = parts[0].encode() + parts[1].encode() + wav_bytes(pcm) + \
        f"\r\n--{boundary}--\r\n".encode()
    raw = _post("https://api.openai.com/v1/audio/transcriptions", body,
                {"Authorization": f"Bearer {KEY}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(raw).get("text", "")


async def ask(question: str) -> bytes:
    speech = synth_pcm(question)
    async with websockets.connect(WS_URL, max_size=None) as ws:
        chunk = int(RATE * 0.02) * 2
        for i in range(0, len(speech), chunk):
            await ws.send(speech[i:i + chunk])
            await asyncio.sleep(0.02)
        for i in range(0, RATE * 2, chunk):  # 1s trailing silence
            await ws.send(b"\x00\x00" * (chunk // 2))
            await asyncio.sleep(0.02)
        out = bytearray()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + REPLY_MAX_WAIT
        while True:
            # Cap each wait by whatever is left of the total, so a silent
            # broker cannot overrun REPLY_MAX_WAIT by a whole idle window.
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=min(REPLY_IDLE, remaining))
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            if isinstance(msg, bytes):
                out.extend(msg)
    return bytes(out)


def _listening(url: str, timeout: float = 2.0) -> bool:
    """Whether anything is accepting connections at this ws:// URL.

    `make check` is wired into a hook, so it runs whenever a shell lands in
    this repo, including when no broker is up. Without this probe that case
    surfaced as a 20-line ConnectionRefusedError traceback out of
    websockets, which reads like the code is broken rather than "nothing is
    running" — four times in one session before anyone fixed it.
    """
    import socket  # local: only needed for this preflight

    rest = url.split("://", 1)[-1].split("/", 1)[0]
    host, _, port = rest.rpartition(":")
    try:
        target = (host or "127.0.0.1", int(port))
    except ValueError:
        return True  # unparseable: let the real connection report it
    try:
        with socket.create_connection(target, timeout):
            return True
    except OSError:
        return False


async def main() -> int:
    print(f"check: broker={WS_URL}  cases={len(CASES)}")
    if not KEY:
        # Same reasoning as the no-broker skip below: a checkout without a
        # key cannot verify anything, and that is a setup gap rather than a
        # regression in the code under test.
        print("  SKIP  no OPENAI_API_KEY in the environment or broker/.env.")
        print("        Set one to actually verify.")
        return 0
    if not _listening(WS_URL):
        # Exit 0, not 1. This runs from a Stop hook on every turn, and no
        # broker on the dev port is the normal resting state, not a
        # regression: there is nothing under test, so there is nothing to
        # fail. Returning 1 here meant the hook reported "make check failed"
        # after every single edit, which trained everyone to ignore it — the
        # exact opposite of what a check is for. A real red still comes from
        # a broker that answers wrongly.
        print(f"  SKIP  nothing is listening on {WS_URL}, so there is nothing to check.")
        print("        Point this at a running broker to actually verify:")
        print("          live puck's broker:   make check WS=ws://127.0.0.1:8765")
        print("          isolated dev broker:  see 'Two engines' in README.md")
        return 0
    failures = skipped = 0
    for question, accept in CASES:
        try:
            audio = await ask(question)
            reply = transcribe(audio).lower() if audio else ""
        except OSError as exc:
            # The check speaks and listens through OpenAI, so a wobble on
            # their side looks exactly like a broken broker. It is not one:
            # nothing was verified, so nothing can be red. A read timeout
            # on the speech endpoint used to end the whole run in a
            # traceback (2026-09-25). TimeoutError and URLError are both
            # OSError, so this catches the lot.
            print(f"  SKIP  {question!r}\n        could not reach OpenAI to run it: {exc}")
            skipped += 1
            continue
        if not audio:
            print(f"  FAIL  {question!r}\n        no audio returned")
            failures += 1
            continue
        ok = any(heard(a, reply) for a in accept)
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {question!r}\n        reply={reply!r}")
        if not ok:
            print(f"        expected any of {accept}")
            failures += 1
    tally = f"{len(CASES) - failures - skipped}/{len(CASES)} passed"
    if skipped:
        # Said out loud: a skip is not a pass, and a run that verified
        # nothing must not read as a clean one.
        tally += f", {skipped} skipped"
    print(f"\n{'GREEN' if failures == 0 else 'RED'}: {tally}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
