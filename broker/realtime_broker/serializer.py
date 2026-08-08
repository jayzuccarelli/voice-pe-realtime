"""WebSocket frame serializer for raw PCM audio.

The device speaks the simplest possible protocol: binary frames are raw
PCM16 / 24 kHz / mono audio, both directions. Text frames are device->broker
control messages (currently {"type": "interrupt"}, sent on a mid-session
wake word) and are dispatched to `on_control` instead of the pipeline.
"""

from __future__ import annotations

import json
import logging

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # OpenAI Realtime requires >= 24 kHz PCM both ways.


class RawPCMSerializer(FrameSerializer):
    """Treats binary WebSocket messages as raw PCM16/24k/mono audio.

    `on_control` (assigned by the server after the pipeline exists) receives
    parsed device control frames; it runs inline in the transport's receive
    loop, so it observes the same ordering the device sent.
    """

    on_control = None  # async callable(dict) | None

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.BINARY

    async def deserialize(self, message: bytes | str) -> InputAudioRawFrame | None:
        if not isinstance(message, bytes):
            if isinstance(message, str) and self.on_control is not None:
                try:
                    data = json.loads(message)
                except ValueError:
                    logger.warning("Dropping malformed control frame: %r", message[:200])
                    return None
                if not isinstance(data, dict):
                    logger.warning("Dropping non-object control frame: %r", message[:200])
                    return None
                # An exception here would kill the transport receive loop and
                # with it the session's audio input; log and carry on instead.
                try:
                    await self.on_control(data)
                except Exception:
                    logger.exception("Control frame handler failed")
            return None
        if len(message) % 2 != 0:
            logger.warning("Dropping odd-length audio frame (%d bytes)", len(message))
            return None
        return InputAudioRawFrame(audio=message, sample_rate=SAMPLE_RATE, num_channels=1)

    async def serialize(self, frame: Frame) -> bytes:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return b""
