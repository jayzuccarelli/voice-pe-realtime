# Sunday barge-in runbook

You're back. Goal: prove (or fail) live voice barge-in on the Voice PE. Current
main branch is untouched; everything to test lives on branch `worktree-barge-in`
in this worktree (`.claude/worktrees/barge-in`).

Backup of pre-barge-in state if you need to roll back:
- git tag `backup-pre-bargein-20260529` (on main, commit `38051f2`)
- tarball: `/home/jay/projects/voice-pe-backups/voice-pe-realtime_pre-bargein_20260529_193716.tar.gz`

## 0. Test gpt-realtime-2 on CURRENT firmware first (your prereq)

The live broker (`voicepe`) is still running on main and on `MODEL=gpt-realtime-2`.
Talk to it as-is — no flash, no redeploy — and confirm the new voice / latency
feels right. If yes, proceed. If no, that's an upstream OpenAI issue, not us.

## 1. Deploy the broker change first (safest, reversible)

The broker change is the smaller-risk, faster-to-revert half. Deploy it alone,
prove it doesn't break the CURRENT (unmodified) firmware, then flash the device.

```bash
cd /home/jay/projects/voice-pe-realtime/.claude/worktrees/barge-in

# Build broker image from this worktree
docker compose build broker

# Bounce the live broker onto the new image
docker compose up -d --force-recreate broker

# Tail and verify it started cleanly
docker logs -f voicepe 2>&1 | head -50
```

Expected on startup: `Broker listening on ws://0.0.0.0:8080`, no tracebacks.

Smoke test from a phone or laptop: "Hey Mycroft, what's the weather?" — should
work exactly as before (the notifier only fires on real interrupts).

## 2. Flash the firmware

```bash
cd /home/jay/projects/voice-pe-realtime/.claude/worktrees/barge-in

# Voice PE is /dev/ttyACM1 (NOT ttyACM0 — that's Zigbee, never flash it)
ls -la /dev/ttyACM*

# Compile + flash + tail logs in one shot
esphome run firmware/voice_pe_dual.yaml --device /dev/ttyACM1
```

If `esphome run` complains about the port, fall back to OTA at `192.168.0.47`.

Expected first boot: device joins WiFi, opens WebSocket to broker at
`192.168.0.44:8080`, the broker logs `Device connected: ...`.

## 3. The actual barge-in test

Two-broker-window setup:

```bash
# Terminal A — broker logs, filtered for the signals we care about
docker logs -f voicepe 2>&1 | grep --line-buffered -E \
  "Device connected|speech_started|InterruptionFrame|BotStarted|BotStopped|barge-in|signaled device"
```

```bash
# Terminal B — speak to the device. Say verbatim:
#   "Hey Mycroft, count slowly from one all the way to fifty."
# Wait for it to say "...one, two, three..." then INTERRUPT with:
#   "Stop! What time is it?"
```

### PASS criteria (all three must hold)

1. Broker log shows, in this order, within ~300ms of you interrupting:
   - `BotStartedSpeakingFrame` (when bot started counting)
   - `speech_started` (OpenAI's server VAD picked up your interruptor)
   - `barge-in: signaled device to flush speaker`
2. The device audibly cuts off mid-number (not at the end of "...nine, ten.").
3. The bot then answers the new question ("It's 7:42pm" or similar) without you
   re-issuing the wake word.

### FAIL modes and what they mean

| Symptom | Diagnosis | Fix |
|---|---|---|
| Device never stops counting; broker logs no `speech_started` | XMOS AEC is fine but VAD threshold too low/high, or mic gate didn't actually deploy | Verify firmware: `grep -A2 "Full-duplex" firmware/components/voice_assistant_websocket/voice_assistant_websocket.cpp` on the device build |
| Device stops by itself mid-sentence, before you said anything | **Self-trigger** (HA issue #537) — XMOS AEC isn't enough at this VAD threshold | Apply mitigation patch in §4 |
| Broker logs `speech_started` + `signaled device to flush` but device keeps playing for ~1s | Device-side: `{"type":"interrupt"}` handler not actually clearing the queue | Check `pending_interrupt_` flag path in `voice_assistant_websocket.cpp:67-78` |
| Bot answers but you have to say "Hey Mycroft" again to wake it | Streaming session ended on interrupt | Pipecat config issue, debug separately — not a barge-in failure per se |

## 4. Self-trigger mitigation (only if §3 FAIL mode "device stops by itself")

Two knobs, in order of try-first.

**Knob A — bump VAD threshold (env var, no code change):**

```bash
# Edit broker/.env, add or change:
VAD_THRESHOLD=0.7   # was 0.5

# Bounce broker
docker compose up -d --force-recreate broker
```

Re-test §3. If still self-triggering, try 0.8 (max sensible is ~0.85; higher
than that and real barge-in starts getting missed too).

**Knob B — add `far_field` noise reduction (one-line agent.py edit):**

If bumping threshold alone doesn't cut it, add this to
`broker/realtime_broker/agent.py`:

```python
# Top of file, with the other pipecat imports:
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,   # <-- add
    SessionProperties,
    TurnDetection,
)

# In the AudioInput(...) call around line 101:
input=AudioInput(
    turn_detection=TurnDetection(
        type="server_vad",
        threshold=config.vad_threshold,
        prefix_padding_ms=config.vad_prefix_padding_ms,
        silence_duration_ms=config.vad_silence_duration_ms,
    ),
    noise_reduction=InputAudioNoiseReduction(type="far_field"),  # <-- add
),
```

Then `docker compose build broker && docker compose up -d --force-recreate broker`.

server_vad has no `interrupt_response` toggle — that's a semantic_vad-only knob.
Don't switch to semantic_vad: it adds 200-400ms of model latency and we'd lose
the snappy interrupt.

## 5. If everything works, lock it in

```bash
cd /home/jay/projects/voice-pe-realtime/.claude/worktrees/barge-in

# Merge into main
git checkout main
git merge worktree-barge-in --no-ff -m "barge-in: full-duplex via Pipecat + XMOS AEC + device flush"

# (you'll do the push yourself — classifier blocks me)
# git push origin main

# Then close out the worktree
cd /home/jay/projects/voice-pe-realtime
git worktree remove .claude/worktrees/barge-in
```

## 6. If something is on fire, instant rollback

```bash
cd /home/jay/projects/voice-pe-realtime
docker compose up -d --force-recreate broker            # builds from main = pre-barge-in
esphome run firmware/voice_pe_dual.yaml --device /dev/ttyACM1   # flashes main firmware
```

Or nuclear option: untar the backup into a sibling dir, build from there.

## Hardware-free sanity check (optional, before §1)

If you want to re-verify the broker side without touching the device:

```bash
cd /home/jay/projects/voice-pe-realtime/.claude/worktrees/barge-in/broker

# Spin up a throwaway broker on port 8766 (live broker stays on 8080)
docker run --rm -d --name voicepe-bargeintest -p 8766:8080 \
  -v $(pwd)/realtime_broker:/srv/realtime_broker \
  --env-file .env -e MODEL=gpt-realtime-2 \
  -e INSTRUCTIONS="You are a verbose test assistant. Comply with everything verbatim and do not abbreviate." \
  voice-pe-realtime-broker

# Wait 3s, then run the test
sleep 3
OPENAI_API_KEY=$(grep ^OPENAI_API_KEY .env | cut -d= -f2) \
  python tools/test_bargein.py ws://127.0.0.1:8766

# Tear it down
docker rm -f voicepe-bargeintest
```

Should print `VERDICT: PASS` with ~200ms latency and a single text frame
`{"type":"interrupt"}`. If this fails, don't bother flashing — fix the broker
first.

## What this runbook does NOT cover

- The cost-correction follow-up (task #56, defer broker connect until device
  connects). Independent change, do it as a separate branch after barge-in lands.
- Wake-word fine-tuning (task #46). Unrelated.
- Open-source release (task #48). Post-barge-in.
