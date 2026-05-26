# voice-pe-realtime

Self-hosted, real-time speech-to-speech for the [Home Assistant Voice PE](https://www.home-assistant.io/voice-pe/) — talk to your home with ChatGPT-Voice-style latency, and have it actually *do* things.

Instead of the turn-based `wake → STT → LLM → TTS` Assist pipeline, the Voice PE streams audio straight to a small server-side **broker** that holds an **OpenAI Realtime** session and controls Home Assistant over **MCP**. One round trip, natural voice, real actions.

## Architecture

```
┌──────────────┐   PCM 24k over ws://   ┌──────────────────┐   WebSocket   ┌─────────────────┐
│  Voice PE     │ ─────────────────────► │  Broker          │ ────────────► │ OpenAI Realtime │
│  (ESP32-S3)   │ ◄───────────────────── │  (Pipecat, this) │ ◄──────────── │  speech↔speech  │
│  wake + audio │     PCM 24k back       │                  │               └─────────────────┘
└──────────────┘                        │     │ MCP / SSE                              
                                         │     ▼                                        
                                         │  ┌──────────────────┐                        
                                         └─►│ Home Assistant   │  turn on lights, etc.  
                                            │ MCP Server       │                        
                                            └──────────────────┘                        
```

The canonical pattern: **the agent runs server-side; the device is a thin full-duplex audio pipe.** Secrets never touch the device, the model is swappable (Pipecat abstracts it), and one broker can serve multiple devices.

## Why

The stock Voice PE pipeline runs STT → LLM → TTS sequentially — a latency floor that feels clunky next to ChatGPT Voice. Routing audio through a persistent Realtime session collapses that to a single round trip with a natural voice, while MCP gives the model first-class control of the home.

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

The broker fetches HA's tools at startup and registers them on the Realtime session, so the model can call `HassTurnOn`, `HassLightSet`, etc.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `MODEL` | `gpt-realtime` | Realtime model |
| `VOICE` | `marin` | Realtime voice |
| `INSTRUCTIONS` | generic | system prompt / persona |
| `WS_HOST` / `WS_PORT` | `0.0.0.0` / `8765` | where the device connects |
| `HA_MCP_URL` / `HA_TOKEN` | — | enable HA control (both required) |
| `VAD_*` | sane defaults | OpenAI server-VAD tuning |

## Firmware

The Voice PE runs ESPHome firmware that streams PCM to this broker. See [`firmware/`](firmware/). Flashing replaces the stock firmware; back up first (`esptool read_flash`) — the ESP32-S3 ROM bootloader makes bricking effectively impossible.

## Status

- ✅ Realtime conversation, server-side
- ✅ Home Assistant control via MCP (SSE)
- ⏳ Firmware flash (per-device)
- ✅ Session rotation before the 60-min Realtime cap (no dropouts)
- ⏳ Open-mic barge-in (true duplex, using the Voice PE's hardware AEC)

## Credits

Built on prior art — see [NOTICE.md](NOTICE.md). MIT licensed.
