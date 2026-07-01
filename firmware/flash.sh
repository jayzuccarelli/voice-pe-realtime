#!/usr/bin/env bash
# Flash the single-wake firmware to a Voice PE connected to THIS machine via USB-C.
# Backs up the stock firmware first (your undo button), then flashes.
#
#   ./flash.sh <serial-port>      # e.g. ./flash.sh /dev/ttyACM1
#
# Find the port: the ESP32-S3 shows in `lsusb` as "Espressif USB JTAG/serial".
# Recovery if anything goes wrong: re-flash stock-voice-pe-backup.bin, or use
# the web flasher (https://web.esphome.io). The ESP32-S3 ROM bootloader can't
# be bricked by flashing.
set -euo pipefail

IMG=ghcr.io/esphome/esphome:latest
BIN=.esphome/build/home-assistant-voice/.pioenvs/home-assistant-voice/firmware.factory.bin
PORT="${1:-/dev/ttyACM0}"
RUN=(docker run --rm --device="$PORT" -v "$PWD":/config --entrypoint python "$IMG" -m esptool --port "$PORT")

[ -e "$PORT" ] || { echo "No device at $PORT (try: lsusb; ls /dev/ttyACM* /dev/ttyUSB*)"; exit 1; }
[ -f "$BIN" ]  || { echo "Missing $BIN — compile first."; exit 1; }

echo ">> Verifying it's an ESP32-S3 (not some other serial device)..."
"${RUN[@]}" flash_id | grep -q "ESP32-S3" || { echo "Not an ESP32-S3 on $PORT — aborting."; exit 1; }

echo ">> [1/2] Backing up stock firmware (16 MB) -> stock-voice-pe-backup.bin"
"${RUN[@]}" --baud 460800 read_flash 0 0x1000000 /config/stock-voice-pe-backup.bin

echo ">> [2/2] Flashing firmware"
"${RUN[@]}" --baud 460800 write_flash 0x0 "/config/$BIN"

echo ">> Done. Device reboots, joins WiFi, connects to the broker."
echo ">> Adopt it in Home Assistant (ESPHome integration) with the key in secrets.yaml."
echo ">> Test: say 'Hey Mycroft', then talk (routes to the realtime broker)."
