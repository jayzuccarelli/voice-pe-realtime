# Canonical verify loop for voice-pe-realtime.
# `make check` streams synthesized speech into the live broker, transcribes the
# spoken reply, and asserts on content — no Voice PE device, no being home.
# Override the target with WS=ws://host:8765 (default: local broker on host).
.PHONY: check
check:
	@python3 broker/tools/check.py $(WS)
