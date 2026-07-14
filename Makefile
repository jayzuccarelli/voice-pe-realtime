# Canonical verify loop for voice-pe-realtime.
# `make check` streams synthesized speech into a broker, transcribes the
# spoken reply, and asserts on content — no Voice PE device, no being home.
# Defaults to the ISOLATED dev broker on 8766: check.py's own default (8765)
# is the LIVE broker, and hook-fired checks were kicking the puck mid-use
# and billing OpenAI every run. Target live deliberately with
# WS=ws://127.0.0.1:8765.
WS ?= ws://127.0.0.1:8766
.PHONY: check
check:
	@python3 broker/tools/check.py $(WS)
