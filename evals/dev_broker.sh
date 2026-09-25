#!/bin/bash
# The broker under test: a dev Live broker on :8766, wired to the fake house
# on :8791 so nothing reaches the real Home Assistant. Extra settings as
# KEY=VAL arguments (e.g. LIVE_OUTPUT_HOLD=0). Replaces any running dev
# broker; the production container runs a bare "python", so it is untouched.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
for p in $(pgrep -f 'venv-live/bin/python -m realtime_broke[r]'); do kill "$p"; done
sleep 1
cd "$here/../broker"
export ENGINE=live WS_PORT=8766 LIVE_HEALTH_PORT=8776
export HA_MCP_URL=http://127.0.0.1:8791/mcp_server/sse HA_TOKEN=fake
# Each scenario is its own conversation: no memory carried across wakes.
export LIVE_MEMORY_TURNS=0 LIVE_VOICE=cedar
export INSTRUCTIONS="You are Atriensis, the household steward for this smart home. Always respond in English. Be concise, warm, and natural — like a capable, unflappable butler. You can control the home with the available tools; when asked to do something, just do it and confirm in one short sentence."
for kv in "$@"; do export "$kv"; done
exec .venv-live/bin/python -m realtime_broker >> /tmp/claude/live-dev.log 2>&1
