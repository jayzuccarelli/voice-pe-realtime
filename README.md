# voice-pe-realtime

Self-hosted, real-time speech-to-speech for the [Home Assistant Voice PE](https://www.home-assistant.io/voice-pe/): talk to your home with ChatGPT-Voice-style latency, and have it actually *do* things.

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

The stock Voice PE pipeline runs STT → LLM → TTS sequentially, a latency floor that feels clunky next to ChatGPT Voice. Routing audio through a persistent Realtime session collapses that to a single round trip with a natural voice, while MCP gives the model first-class control of the home.

## Two engines

The broker can drive the puck with either OpenAI voice model. `ENGINE` picks
one; both speak the same device protocol, so switching is a restart, never a
reflash, and switching back is the rollback.

| | `ENGINE=realtime` (default) | `ENGINE=live` |
|---|---|---|
| Model | `gpt-realtime` | `gpt-live-1` |
| Turn taking | server VAD, one turn at a time | full duplex: the model hears you while it speaks |
| Barge-in | needs a raised voice; the mic is fed silence during playback so the bot cannot answer its own echo | the model yields on its own, ~2.7x faster than Realtime, though not instantly ([measured](#barge-in-measured)) |
| Tools | called directly by the voice model | delegated to a backend model (`LIVE_BACKEND_MODEL`) that runs the same Home Assistant tools |
| Billing | per token | per minute of open session |

The Live engine is why the broker owns session lifetime. `gpt-live-1` bills
per minute of session and streams audio continuously, which keeps the
firmware's 10-second auto-stop from ever firing, so a session is opened when
the device connects and closed when it disconnects, with a follow-up window,
a turn budget and `MAX_LIVE_SESSION_SECONDS` as a hard cost fuse.

It needs pipecat's main branch (see `broker/requirements-live.txt`) and
Python 3.11+, so it does not run in the container built from
`requirements.lock`. Run it against the isolated dev port:

```bash
cd broker
uv venv --python 3.12 .venv-live
uv pip install --python .venv-live/bin/python -r requirements-live.txt
ENGINE=live WS_PORT=8766 .venv-live/bin/python -m realtime_broker
make check WS=ws://127.0.0.1:8766      # from the repo root
```

### Known API bug: no floats in delegated tool schemas

`session.start` fails with `Invalid AVAS session_data: Type is not JSON
serializable: decimal.Decimal` if any delegated tool schema contains a float
literal. One tool with `{"type": "number", "minimum": 0.5}` is enough; the
same tool with `minimum: 1` starts fine. Home Assistant's MCP server reports
numeric bounds as floats, so a single script with a numeric range takes the
whole session down. The broker works around it by rewriting whole floats as
integers and dropping fractional bounds (`live_agent._scrub_floats`).

## Barge-in, measured

Full duplex is the reason to switch engines, so it is measured rather than
asserted. `broker/tools/bench_bargein.py` asks a long question, cuts in
1.5 s into the answer, and times how long the bot keeps talking. It runs
**headless against a broker, no Voice PE required**, because the model-side
half of barge-in is fully observable from the audio stream.

5 trials per engine, `STOP_WINDOW_S=1.0`:

| engine | yields <1 s | time to last word (p50) | audio after cut (p50) | reply onset (p50) |
|---|---|---|---|---|
| `gpt-live-1` | 1/4 | **1.48 s** | 0.56 s | 0.25 s |
| `gpt-realtime` | 0/3 | **3.99 s** | 3.24 s | 0.48 s |

**Live yields the floor about 2.7x faster.** That is the real difference, and
it is large enough to feel.

What this does **not** show: neither engine reliably goes quiet within a
second. If you want "stops the instant you speak", neither qualifies yet on
this test. Reply onset also goes *negative* on some Live trials, meaning it
begins answering while you are still speaking, which Realtime cannot do by
construction since server VAD must detect end-of-speech first.

```bash
python3 broker/tools/bench_bargein.py ws://127.0.0.1:8766 --trials 5 --label gpt-live-1
python3 broker/tools/bench_bargein.py ws://127.0.0.1:8765 --trials 5 --label gpt-realtime
```

Caveats worth stating plainly: small n (4 and 3 scored runs after excluding
dropped connections and replies that ended before the cut), synthesized TTS
interruptions rather than a human voice in a room, and **acoustic echo
residual is unmeasured** — whether the device's own speaker bleeding into its
mic false-triggers an interruption needs the real puck in a real room, and it
is the one number a browser-tab demo cannot produce.

## Reliability

Speech-to-speech on a $59 puck is easy to demo and hard to keep up. This repo treats robustness as the feature:

- **Session rotation**: OpenAI caps a Realtime session at ~60 min and treats expiry as fatal. The broker rotates the session *before* the cap (and rebuilds after any death) under a still-connected device, so long-lived pucks never drop. Proven continuous across forced rotations.
- **Idle refresh**: a stale idle session (socket open, silently dead) is refreshed proactively.
- **Turn hygiene**: a device that vanishes mid-utterance (Wi-Fi blip, session timeout) leaves OpenAI's server VAD holding a speech-in-progress segment that would come back as a ghost turn on the next wake. Clearing the input buffer isn't enough (the bytes go, the VAD state doesn't); the broker disables and re-enables turn detection on disconnect to drop the segment for real. Background speech (a TV, a side conversation) is gated with the OpenAI-recommended `wait_for_user` pattern, with an explicit follow-up bias, so "are you sure about that?" right after an answer gets answered instead of ignored.
- **A real test harness**: `make check` drives the broker end-to-end exactly like the firmware (streams PCM, transcribes the spoken reply, asserts content + first-audio latency). No hardware needed.

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
| `OPENAI_API_KEY` | none | required |
| `ENGINE` | `realtime` | `realtime` or `live` (see [Two engines](#two-engines)) |
| `MODEL` | `gpt-realtime` | Realtime model |
| `VOICE` | `marin` | Realtime voice |
| `INSTRUCTIONS` | generic | system prompt / persona |
| `WS_HOST` / `WS_PORT` | `0.0.0.0` / `8765` | where the device connects |
| `HA_MCP_URL` / `HA_TOKEN` | none | enable HA control (both required) |
| `MUSIC_PLAYER` | none | default Music Assistant speaker |
| `VAD_*` | sane defaults | OpenAI server-VAD tuning |
| `FOLLOWUP_WINDOW_SECONDS` | `6.0` | after each reply, how long the mic stays open for a follow-up before the broker disconnects the device (wake word re-arms) |
| `MAX_TURNS_PER_WAKE` | `8` | user turns allowed per wake, so TV speech can't spiral a session |
| `MAX_SESSION_SECONDS` | `3000` | rotate before the 60-min cap |
| `IDLE_REFRESH_SECONDS` | `600` | refresh a stale idle session |
| `LIVE_MODEL` | `gpt-live-1` | Live engine: the full-duplex frontend model |
| `LIVE_BACKEND_MODEL` | `gpt-5.4-mini` | Live engine: the model the frontend delegates tools and reasoning to |
| `MAX_LIVE_SESSION_SECONDS` | `180` | Live engine cost fuse: hard cap on one wake, honoured mid-sentence. `0` disables it, and a stuck session then bills until someone notices |

Turn hygiene (`FOLLOWUP_WINDOW_SECONDS` / `MAX_TURNS_PER_WAKE`): set either to `0` to disable that bound. Setting both to `0` restores the old unbounded behavior, which is the no-redeploy rollback lever.

## Firmware

The Voice PE runs ESPHome firmware that streams PCM to this broker. See [`firmware/`](firmware/). Flashing replaces the stock firmware; back up first (`esptool read_flash`): the ESP32-S3 ROM bootloader makes bricking effectively impossible.

## Status & roadmap

- ✅ Real-time speech-to-speech, server-side
- ✅ Home Assistant control via MCP (SSE) + weather / music tools
- ✅ Session rotation before the 60-min Realtime cap (no dropouts)
- ✅ Background-speech gating (`wait_for_user`)
- ✅ End-to-end reliability harness (`make check`)
- ⏳ **Smart routing**: one wake word, fast local intents handled on-device, everything else escalated to the LLM (the elegant form of "local + cloud")
- ⏳ **Barge-in**: true open-mic interruption using the Voice PE's hardware AEC (experimental; the acoustic self-trigger loop is the open problem, an echo-residual calibration rig ships in `broker/tools/`)
- ⏳ **Beamforming**: tap the XMOS array's focused channel to reject off-axis room noise (a TV, another speaker)

MIT licensed. Attribution in [NOTICE.md](NOTICE.md).
