# Firmware

ESPHome firmware for the Home Assistant Voice PE (ESP32-S3) that streams audio to the realtime broker.

## `voice_pe_dual.yaml` — single wake word

Built on the **official** [`home-assistant-voice-pe`](https://github.com/esphome/home-assistant-voice-pe) firmware, with one wake word routed to the realtime broker:

| Wake word | Goes to |
|---|---|
| **Hey Mycroft** | the realtime broker (`voice_assistant_websocket`) |

On "Hey Mycroft" the device plays a wake cue, opens the broker WebSocket, and streams PCM from the same mic (`i2s_mics` — echo-cancelled by the XMOS chip); on disconnect it re-arms wake-word detection and plays an end cue. Audio plays back through the media speaker.

How it's wired (additive edits to the official YAML):
- `external_components` adds the local `voice_assistant_websocket` component
- a `voice_assistant_websocket:` block points at `i2s_mics` + `media_resampling_speaker` + the broker URL
- `on_wake_word_detected` starts the websocket session

> **Dual-mode variant** — an earlier build routed **Hey Jarvis / Okay Nabu** to the stock HA Assist pipeline *and* **Hey Mycroft** to the broker (local + realtime on one device). That's preserved at git tag `dual-2wake-v0` / branch `archive/dual-2wake`. v1 ships single-wake for a cleaner default; the elegant form of the hybrid (one wake word + smart local/cloud routing) is on the roadmap.

## Setup

```bash
cp secrets.yaml.example secrets.yaml   # set wifi_ssid, wifi_password, broker_url, api_encryption_key
docker run --rm -v "$PWD":/config ghcr.io/esphome/esphome:latest compile voice_pe_dual.yaml
```

## Flash (device on USB-C to this machine)

```bash
./flash.sh                 # or ./flash.sh /dev/ttyUSB0
```

It backs up the stock firmware first, then flashes. After reboot the device joins WiFi and connects to the broker. Adopt it in Home Assistant (ESPHome integration) with the key from `secrets.yaml`.

## Status

Compiles clean (`firmware.factory.bin`) and runs on-device. Known limitation: the far-field mic captures the whole room, so in a loud room the assistant can pick up a TV or another speaker — the broker mitigates this with the `wait_for_user` gate, and device-side beamforming (tapping the XMOS focused channel) is on the roadmap.

## Credit

The `components/voice_assistant_websocket/` ESPHome component derives from [fjfricke/ha-openai-realtime](https://github.com/fjfricke/ha-openai-realtime) (MIT). See `../NOTICE.md`.
