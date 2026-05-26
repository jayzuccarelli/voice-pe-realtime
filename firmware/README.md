# Firmware

ESPHome firmware for the Home Assistant Voice PE (ESP32-S3) that streams audio to the realtime broker.

## `voice_pe_dual.yaml` — dual-mode (recommended)

Built on the **official** [`home-assistant-voice-pe`](https://github.com/esphome/home-assistant-voice-pe) 26.5.0 firmware, so the stock experience is fully preserved, with one wake word rerouted:

| Wake word | Goes to |
|---|---|
| **Hey Jarvis**, **Okay Nabu** | stock Home Assistant Assist pipeline |
| **Hey Mycroft** | the realtime broker (`voice_assistant_websocket`) |

The realtime path taps the same mic (`i2s_mics` — already echo-cancelled by the XMOS chip) and plays back through the media speaker. On "Hey Mycroft" it stops wake-word detection, opens the broker WebSocket, and streams; on disconnect it re-arms wake-word detection.

How it's wired (all additive edits to the official YAML):
- `external_components` adds the local `voice_assistant_websocket` component
- a `voice_assistant_websocket:` block points at `i2s_mics` + `media_resampling_speaker` + the broker URL
- `on_wake_word_detected` branches on the matched phrase
- the original Assist-start logic is extracted into a `start_ha_assist` script

## Setup

```bash
cp secrets.yaml.example secrets.yaml   # set wifi_ssid, wifi_password, broker_url, api_encryption_key
docker run --rm -v "$PWD":/config ghcr.io/esphome/esphome:latest compile voice_pe_dual.yaml
```

## Flash (device on USB-C to this machine)

```bash
./flash.sh                 # or ./flash.sh /dev/ttyUSB0
```

It backs up the stock firmware first, then flashes. After reboot the device joins WiFi and connects to the broker. Adopt it in Home Assistant (ESPHome integration) with the key from `secrets.yaml` so the HA Assist side authenticates.

## Status

Compiles clean (`firmware.factory.bin`). Runtime behavior — the mic/speaker handoff between Assist and realtime modes, and audio routing through the media resampler — is validated on-device after flashing; tune `voice_assistant_websocket`'s speaker target there if needed.

## Credit

The `components/voice_assistant_websocket/` ESPHome component derives from [fjfricke/ha-openai-realtime](https://github.com/fjfricke/ha-openai-realtime) (MIT). See `../NOTICE.md`.
