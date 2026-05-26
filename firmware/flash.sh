#!/usr/bin/env bash
# Flash the dual-mode firmware to a Voice PE connected to THIS machine via USB-C.
# Backs up the stock firmware first (your undo button), then flashes.
#
#   ./flash.sh [serial-port]      # default /dev/ttyACM0
#
# Recovery if anything goes wrong: re-flash stock-voice-pe-backup.bin, or use
# the web flasher (https://web.esphome.io). The ESP32-S3 ROM bootloader can't
# be bricked by flashing.
set -euo pipefail

IMG=ghcr.io/esphome/esphome:latest
BIN=.esphome/build/home-assistant-voice/.pioenvs/home-assistant-voice/firmware.factory.bin
PORT="${1:-/dev/ttyACM0}"

[ -e "$PORT" ] || { echo "No device at $PORT. Plug in the Voice PE, or pass the right port (try: ls /dev/ttyACM* /dev/ttyUSB*)."; exit 1; }
[ -f "$BIN" ]  || { echo "Missing $BIN — compile first: docker run --rm -v \$PWD:/config $IMG compile voice_pe_dual.yaml"; exit 1; }

echo ">> [1/2] Backing up stock firmware (16 MB) -> stock-voice-pe-backup.bin"
docker run --rm --device="$PORT" -v "$PWD":/config "$IMG" \
  python -m esptool --chip esp32s3 --port "$PORT" read_flash 0 0x1000000 /config/stock-voice-pe-backup.bin

echo ">> [2/2] Flashing dual-mode firmware"
docker run --rm --device="$PORT" -v "$PWD":/config "$IMG" \
  python -m esptool --chip esp32s3 --port "$PORT" write_flash 0x0 "/config/$BIN"

echo ">> Done. The device reboots, joins ARRIS-9AFD, and connects to the broker."
echo ">> Then in Home Assistant: Settings -> Devices & Services -> ESPHome will discover"
echo ">> the device; adopt it with the key in secrets.yaml so 'Hey Jarvis' (HA Assist) works."
echo ">> Test: say 'Hey Mycroft' (realtime) and 'Hey Jarvis' (HA Assist)."
