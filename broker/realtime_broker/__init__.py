"""Realtime voice broker for Home Assistant Voice PE.

A thin server-side agent that holds an OpenAI Realtime session and bridges
raw PCM audio to/from a Voice PE device over WebSocket, with optional Home
Assistant control via MCP.
"""

__version__ = "0.1.0"
