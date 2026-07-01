# voice-pe realtime broker — reliability harness.
#
# `make check` drives the broker end-to-end (synthesizes speech, streams it like
# the firmware, transcribes the reply, asserts content + latency). Needs
# OPENAI_API_KEY in the environment and a RUNNING broker.
#
# WS defaults to :8766 (an ISOLATED broker), NOT the live puck's :8765 — the
# broker is single-client, so pointing this at :8765 kicks the device.
#
#   make check                       # against ws://127.0.0.1:8766
#   make check WS=ws://<broker-ip>:8765    # against the live broker (kicks puck)
#   make soak N=10                   # repeat the suite N times for flake/latency

WS ?= ws://127.0.0.1:8766
N  ?= 5

check:
	uv run --with websockets tools/harness.py $(WS)

soak:
	uv run --with websockets tools/harness.py $(WS) --soak $(N)

.PHONY: check soak
