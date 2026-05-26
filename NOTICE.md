# Attribution

This project's architecture and firmware were informed by, and the firmware
component derives from, **[fjfricke/ha-openai-realtime](https://github.com/fjfricke/ha-openai-realtime)**
(MIT License) — a proof-of-concept that first demonstrated bridging a Home
Assistant Voice PE to the OpenAI Realtime API over WebSocket.

The broker here is a substantial rewrite (clean module structure, SSE-based
Home Assistant MCP control, persistent-session context, reproducible build),
but credit for the original approach belongs to that project.

Built on:
- [Pipecat](https://github.com/pipecat-ai/pipecat) — voice-agent framework (BSD-2-Clause)
- [ESPHome](https://esphome.io) — device firmware
- [Home Assistant](https://www.home-assistant.io) — home automation + MCP Server
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
