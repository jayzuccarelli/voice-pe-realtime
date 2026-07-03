# voice-pe-realtime

Self-hosted, real-time speech-to-speech for the [Home Assistant Voice PE](https://www.home-assistant.io/voice-pe/) — talk to your home with ChatGPT-Voice-style latency, and have it actually *do* things.

Instead of the turn-based `wake → STT → LLM → TTS` Assist pipeline, the Voice PE streams audio straight to a small server-side **broker** that holds an **OpenAI Realtime** session and controls Home Assistant over **MCP**. One round trip, natural voice, real actions.

Say **"Hey Mycroft"**, then talk.

## Architecture

```
┌──────────────┐   PCM 24k over ws://   ┌──────────────────┐   WebSocket   ┌─────────────────┐
│  Voice PE     │ ─────────────────────► │  Broker          │ ────────────► │ OpenAI Realtime │
│  (ESP32-S3)   │ ◄───────────────────── │  (Pipecat, this) │ ◄──────────── │  speech↔speech  │
│  wake + audio │     PCM 24k back       │                  │               └─────────────────┘
└──────────────┘                        │     │ MCP / SSE
                                         │     ▼
                                         │  ┌──────────────────┐
                                         └─►│ Home Assistant   │  turn on lights, play music, …
                                            │ MCP Server       │
                                            └──────────────────┘
```

The canonical pattern: **the agent runs server-side; the device is a thin full-duplex audio pipe.** Secrets never touch the device, the model is swappable (Pipecat abstracts it), and one broker can serve multiple devices.

## Why

The stock Voice PE pipeline runs STT → LLM → TTS sequentially — a latency floor that feels clunky next to ChatGPT Voice. Routing audio through a persistent Realtime session collapses that to a single round trip with a natural voice, while MCP gives the model first-class control of the home.

## Reliability

Speech-to-speech on a $59 puck is easy to demo and hard to keep up. This repo treats robustness as the feature:

- **Session rotation** — OpenAI caps a Realtime session at ~60 min and treats expiry as fatal. The broker rotates the session *before* the cap (and rebuilds after any death) under a still-connected device, so long-lived pucks never drop. Proven continuous across forced rotations.
- **Idle refresh** — a stale idle session (socket open, silently dead) is refreshed proactively.
- **Turn hygiene** — a device that vanishes mid-utterance (Wi-Fi blip, session timeout) leaves OpenAI's server VAD holding a speech-in-progress segment that would come back as a ghost turn on the next wake. Clearing the input buffer isn't enough (the bytes go, the VAD state doesn't); the broker disables and re-enables turn detection on disconnect to drop the segment for real. Background speech (a TV, a side conversation) is gated with the OpenAI-recommended `wait_for_user` pattern — with an explicit follow-up bias, so "are you sure about that?" right after an answer gets answered instead of ignored.
- **A real test harness** — `make check` drives the broker end-to-end exactly like the firmware (streams PCM, transcribes the spoken reply, asserts content + first-audio latency). No hardware needed.

```bash
cd broker && OPENAI_API_KEY=... make check          # 10 scenarios, pass/fail + p50/p95 latency
cd broker && OPENAI_API_KEY=... make soak N=20      # repeat for flake/latency
```

Scenarios: basic Q&A, multi-turn context, HA tool call, no-reply-to-silence, no-ghost-on-connect, reconnect, background-speech rejection, follow-up-challenge after an answer, TV-line-after-answer (false-accept counter-metric), mid-speech disconnect (ghost-turn regression). `--only <name>` runs one scenario. Point it at an isolated broker (`WS=ws://127.0.0.1:8766`) so it never kicks a live device.

## Quick start (broker)

```bash
cd broker
cp .env.example .env      # set OPENAI_API_KEY (+ HA_MCP_URL / HA_TOKEN for home control)
docker build -t voicepe-realtime:dev .
docker run --rm --network host --env-file .env voicepe-realtime:dev
```

Or via compose from the repo root: `docker compose up -d --build`.

### Home Assistant control (optional)

1. Enable the **MCP Server** integration in HA.
2. Create a long-lived access token (HA → profile → Security → Long-lived access tokens).
3. Set `HA_MCP_URL=http://<ha>:8123/mcp_server/sse` and `HA_TOKEN=<token>` in `.env`.

The broker fetches HA's tools at startup and registers them on the Realtime session, so the model can call `HassTurnOn`, `HassLightSet`, etc. It also ships custom tools for weather, Music Assistant playback, and clean end-of-conversation.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `MODEL` | `gpt-realtime` | Realtime model |
| `VOICE` | `marin` | Realtime voice |
| `INSTRUCTIONS` | generic | system prompt / persona |
| `WS_HOST` / `WS_PORT` | `0.0.0.0` / `8765` | where the device connects |
| `HA_MCP_URL` / `HA_TOKEN` | — | enable HA control (both required) |
| `MUSIC_PLAYER` | — | default Music Assistant speaker |
| `VAD_*` | sane defaults | OpenAI server-VAD tuning |
| `MAX_SESSION_SECONDS` | `3000` | rotate before the 60-min cap |
| `IDLE_REFRESH_SECONDS` | `600` | refresh a stale idle session |

## Firmware

The Voice PE runs ESPHome firmware that streams PCM to this broker. See [`firmware/`](firmware/). Flashing replaces the stock firmware; back up first (`esptool read_flash`) — the ESP32-S3 ROM bootloader makes bricking effectively impossible.

## Status & roadmap

- ✅ Real-time speech-to-speech, server-side
- ✅ Home Assistant control via MCP (SSE) + weather / music tools
- ✅ Session rotation before the 60-min Realtime cap (no dropouts)
- ✅ Background-speech gating (`wait_for_user`)
- ✅ End-to-end reliability harness (`make check`)
- ⏳ **Smart routing** — one wake word, fast local intents handled on-device, everything else escalated to the LLM (the elegant form of "local + cloud")
- ⏳ **Barge-in** — true open-mic interruption using the Voice PE's hardware AEC (experimental; the acoustic self-trigger loop is the open problem — an echo-residual calibration rig ships in `broker/tools/`, see `M2_RUNBOOK.md`)
- ⏳ **Beamforming** — tap the XMOS array's focused channel to reject off-axis room noise (a TV, another speaker)

## Prior art & focus

Others have put OpenAI Realtime on the Voice PE — see [NOTICE.md](NOTICE.md) for credits. This project's focus is **reliability** (proven session rotation, a real test harness), **deep HA tool integration**, and a roadmap toward smart local/cloud routing and barge-in. Honest limitation: the OpenAI Realtime API has no speaker separation, so in a loud room the assistant can still pick up other voices — the real fix for that is device-side beamforming (on the roadmap).

MIT licensed.
