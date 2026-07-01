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
    # Default Music Assistant media_player for play_music when no speaker is named.
    music_player: str | None = None
    # HA weather entity the get_weather tool reads (varies per install).
    weather_entity: str = "weather.forecast_home"
    # Bill an OpenAI Whisper transcription of each user turn for debug logging.
    # Off by default: it adds per-turn cost and is only useful for development.
    debug_transcription: bool = False

    # OpenAI server-side VAD turn detection.
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 500
    # Adaptive VAD: while the bot is speaking, residual echo of its own voice
    # (XMOS AEC leaves it at roughly 0.6-0.7 equivalent at the mic) would trip
    # the idle threshold and chop every reply. Raise the bar while the bot has
    # the floor; barge-in then just needs a slightly raised voice. The release
    # delay covers the device's ~1s speaker buffer, which keeps bleeding echo
    # after the broker has finished sending audio.
    vad_threshold_speaking: float = 0.85
    vad_release_delay_ms: int = 1200

    # OpenAI caps a Realtime session at 60 min. Proactively rotate a bit before
    # that so the broker never hits the fatal expiry. The device is turn-based,
    # so a rotation between turns is invisible.
    max_session_seconds: int = 3000  # 50 min

    # An idle Realtime session goes stale server-side WITHOUT the socket dying:
    # a 47-min-old session accepted audio and returned nothing while ws.state
    # stayed OPEN (2026-06-10). Refresh the session whenever no device has been
    # connected for this long — free, and invisible to the user.
    idle_refresh_seconds: int = 600  # 10 min

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
            music_player=os.environ.get("MUSIC_PLAYER") or None,
            weather_entity=os.environ.get("WEATHER_ENTITY", cls.weather_entity),
            debug_transcription=os.environ.get("DEBUG_TRANSCRIPTION", "").lower()
            in ("1", "true", "yes"),
            vad_threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
            vad_prefix_padding_ms=int(os.environ.get("VAD_PREFIX_PADDING_MS", "300")),
            vad_silence_duration_ms=int(os.environ.get("VAD_SILENCE_DURATION_MS", "500")),
            vad_threshold_speaking=float(os.environ.get("VAD_THRESHOLD_SPEAKING", "0.85")),
            vad_release_delay_ms=int(os.environ.get("VAD_RELEASE_DELAY_MS", "1200")),
            max_session_seconds=int(os.environ.get("MAX_SESSION_SECONDS", "3000")),
            idle_refresh_seconds=int(os.environ.get("IDLE_REFRESH_SECONDS", "600")),
        )
