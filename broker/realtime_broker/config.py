"""Runtime configuration, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All broker settings. Construct with `Config.from_env()`."""

    openai_api_key: str
    model: str = "gpt-realtime"
    voice: str = "marin"
    instructions: str = "You are a helpful voice assistant."

    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    # Home Assistant control via MCP (optional). Both must be set to enable it.
    ha_mcp_url: str | None = None
    ha_token: str | None = None

    # OpenAI server-side VAD turn detection.
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 500

    @property
    def ha_control_enabled(self) -> bool:
        return bool(self.ha_mcp_url and self.ha_token)

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        return cls(
            openai_api_key=api_key,
            model=os.environ.get("MODEL", "gpt-realtime"),
            voice=os.environ.get("VOICE", "marin"),
            instructions=os.environ.get("INSTRUCTIONS", cls.instructions),
            ws_host=os.environ.get("WS_HOST", "0.0.0.0"),
            ws_port=int(os.environ.get("WS_PORT", "8765")),
            ha_mcp_url=os.environ.get("HA_MCP_URL") or None,
            ha_token=os.environ.get("HA_TOKEN") or None,
            vad_threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            vad_prefix_padding_ms=int(os.environ.get("VAD_PREFIX_PADDING_MS", "300")),
            vad_silence_duration_ms=int(os.environ.get("VAD_SILENCE_DURATION_MS", "500")),
        )
