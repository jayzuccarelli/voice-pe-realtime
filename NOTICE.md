# Attribution

The ESPHome firmware component in
`firmware/components/voice_assistant_websocket/` derives from
**[fjfricke/ha-openai-realtime](https://github.com/fjfricke/ha-openai-realtime)**
(MIT License): the proof-of-concept that first bridged a Home Assistant
Voice PE to the OpenAI Realtime API over WebSocket. That project's MIT license
and copyright are preserved in
`firmware/components/voice_assistant_websocket/LICENSE`.

The device YAML in `firmware/voice_pe_dual.yaml` is derived from
**[esphome/home-assistant-voice-pe](https://github.com/esphome/home-assistant-voice-pe)**
(`home-assistant-voice.yaml`, Copyright (c) 2019 ESPHome), modified to route the
wake word to the broker instead of the stock Assist pipeline. The ESPHome License
applies MIT terms to non-C++ files such as YAML; its full text, including the MIT
permission notice, is preserved in `firmware/LICENSE.esphome`.

The broker is an independent rewrite (clean module structure, SSE-based Home
Assistant MCP control, persistent-session context, reproducible build), but
the original bridging approach came from that project.

Built on:
- [Pipecat](https://github.com/pipecat-ai/pipecat): voice-agent framework (BSD-2-Clause)
- [ESPHome](https://esphome.io): device firmware
- [Home Assistant](https://www.home-assistant.io): home automation + MCP Server
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
