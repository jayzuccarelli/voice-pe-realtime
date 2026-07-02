# M2: real echo-residual calibration (no flash)

Measures the number the whole barge-in plan pivots on: the NCC between what
the puck's speaker plays and what the XMOS post-AEC mic tap (ch1 @ 24x)
still hears of it. Synthetic sims said echo NCC 0.80+ vs user speech
0.11–0.47, but no AEC was applied there — these are the first real numbers.

**Someone must be home** (it plays audio in the room) and takes ~10 minutes.
No firmware change, no flash, fully reversible.

## Why the mic stays open

Playback goes through the puck's HA `media_player` path. The broker never
sends a binary frame, so `is_bot_speaking()` stays false and the firmware
mic gate never closes — the mic streams the whole time the clip plays.

## Steps

All from `broker/` on vaio. Terminal A:

```sh
docker stop voicepe                      # frees :8765 for the capture server
uv run --with websockets tools/m2_capture.py --port 8765
```

Terminal B, once A says it's listening:

```sh
export HA_TOKEN=...                      # and OPENAI_API_KEY on first run (TTS)
# 1. Wake the puck ("Hey Jarvis"), then IMMEDIATELY:
uv run tools/m2_play_ref.py --entity media_player.home_assistant_voice_0aa2f8_media_player
```

Watch terminal A: RMS lines should jump while the clip plays. The firmware
auto-stops the session 10s after wake if nothing is sent back, so wake →
play must be quick; re-wake and repeat for a second take if the session
drops mid-clip.

Takes to collect (each is one wake + one capture file):

1. **Echo take** ×2: clip playing at normal volume. The measurement.
2. **Max-volume echo take** ×1: puck volume at 100%. Worst case.
3. **User-speech control** ×1: no playback, speak a few sentences from
   couch distance. NCC vs ref must be ~baseline (sanity: the gate wouldn't
   eat real users).
4. **Quiet-room control** ×1: no playback, no speech, ~10s. Noise floor.

Then restore the live broker: `docker start voicepe` (verify with a wake).

## Analysis

```sh
uv run --with numpy tools/m2_analyze.py m2_captures/ref.wav m2_captures/capture_1.pcm
```

Decision (pre-declared in the plan, don't re-litigate in the moment):

| p50 NCC (active windows) | Meaning | Action |
|---|---|---|
| ≥ 0.6 | gate can see the echo | build M1 NCC gate |
| 0.3 – 0.6 | gray zone | STOP: volume cap → `aec_corr_factor` DFU → software AEC |
| < 0.3, mic RMS ≈ silent baseline | AEC buries the echo at this volume | re-check the max-volume take before concluding; if still buried, the gate may only need to be an RMS guard |

Verify entity id first if unsure: it must be the puck
(`home_assistant_voice_0aa2f8`), not another speaker.
