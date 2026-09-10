"""Runtime configuration, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _non_negative(name: str, default: int) -> int:
    """Read an int env var that must not be negative.

    Guards the Live cost fuse in particular: `_LiveHygiene` treats any cap
    that is not greater than zero as "no cap", so a stray minus sign would
    quietly disable the one thing bounding a per-minute meter, while the
    documentation promises that only 0 does that. Refuse to start instead.
    """
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0 (0 disables the bound), got {value}")
    return value


@dataclass(frozen=True)
class Config:
    """All broker settings. Construct with `Config.from_env()`."""

    openai_api_key: str
    model: str = "gpt-realtime"
    # Which brain serves the device: "realtime" (gpt-realtime, turn-based with
    # a stack of VAD/echo workarounds) or "live" (gpt-live-1, full duplex).
    # The engines share the device protocol, so this is the rollback lever:
    # flip it back and restart, no reflash.
    engine: str = "realtime"
    live_model: str = "gpt-live-1"
    # The model the live frontend delegates tool use and reasoning to.
    live_backend_model: str = "gpt-5.4-mini"
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

    # OpenAI server-side VAD turn detection. 800ms end-of-turn silence (up
    # from OpenAI's 500ms default): with a TV on, 500ms cut real questions at
    # mid-sentence pauses and the fragments read as background speech. Costs
    # +0.3s response latency per turn.
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 800
    # Adaptive VAD: while the bot is speaking, residual echo of its own voice
    # (XMOS AEC leaves it at roughly 0.6-0.7 equivalent at the mic) would trip
    # the idle threshold and chop every reply. Raise the bar while the bot has
    # the floor; barge-in then just needs a slightly raised voice. The release
    # delay covers the device's speaker buffer (hard-bounded at ~740ms:
    # 10-chunk send queue + 3x100ms rings), which keeps bleeding echo after
    # the broker has finished sending audio.
    vad_threshold_speaking: float = 0.85
    vad_release_delay_ms: int = 1200

    # Turn hygiene: bound how long one wake keeps the conversation open so TV
    # speech can't spiral a session for minutes. After each reply the user has
    # followup_window_seconds (measured from when the SPEAKER goes quiet, not
    # response.done) to take another turn before the broker disconnects the
    # device and the wake word re-arms; max_turns_per_wake caps committed user
    # turns per WS connection. Either set to 0 disables that bound (both 0 =
    # exactly the pre-hygiene behavior): the no-redeploy rollback lever.
    followup_window_seconds: float = 6.0
    max_turns_per_wake: int = 8

    # OpenAI caps a Realtime session at 60 min. Proactively rotate a bit before
    # that so the broker never hits the fatal expiry. The device is turn-based,
    # so a rotation between turns is invisible.
    max_session_seconds: int = 3000  # 50 min

    # Live engine only. Billing is per minute of open session and nothing on
    # the device can end one (the firmware auto-stop keys on speaker audio,
    # which a continuous stream keeps alive), so this cap is a cost fuse, not
    # a UX bound: it fires even mid-sentence. 0 disables it, which means a
    # stuck session bills until someone notices.
    max_live_session_seconds: int = 180  # 3 min

    # An idle Realtime session goes stale server-side WITHOUT the socket dying:
    # a 47-min-old session accepted audio and returned nothing while ws.state
    # stayed OPEN (2026-06-10). Refresh the session whenever no device has been
    # connected for this long: free, and invisible to the user.
    idle_refresh_seconds: int = 600  # 10 min

    @property
    def ha_control_enabled(self) -> bool:
        return bool(self.ha_mcp_url and self.ha_token)

    @classmethod
    def from_env(cls) -> Config:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        return cls(
            openai_api_key=api_key,
            model=os.environ.get("MODEL", "gpt-realtime"),
            engine=os.environ.get("ENGINE", cls.engine).lower(),
            live_model=os.environ.get("LIVE_MODEL", cls.live_model),
            live_backend_model=os.environ.get("LIVE_BACKEND_MODEL", cls.live_backend_model),
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
            vad_silence_duration_ms=int(os.environ.get("VAD_SILENCE_DURATION_MS", "800")),
            vad_threshold_speaking=float(os.environ.get("VAD_THRESHOLD_SPEAKING", "0.85")),
            vad_release_delay_ms=int(os.environ.get("VAD_RELEASE_DELAY_MS", "1200")),
            followup_window_seconds=float(os.environ.get("FOLLOWUP_WINDOW_SECONDS", "6.0")),
            max_turns_per_wake=int(os.environ.get("MAX_TURNS_PER_WAKE", "8")),
            max_session_seconds=int(os.environ.get("MAX_SESSION_SECONDS", "3000")),
            idle_refresh_seconds=int(os.environ.get("IDLE_REFRESH_SECONDS", "600")),
            max_live_session_seconds=_non_negative(
                "MAX_LIVE_SESSION_SECONDS", cls.max_live_session_seconds
            ),
        )
