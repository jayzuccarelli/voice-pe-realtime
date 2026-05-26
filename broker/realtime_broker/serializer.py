"""WebSocket frame serializer for raw PCM audio.

The device speaks the simplest possible protocol: binary frames are raw
PCM16 / 24 kHz / mono audio, both directions. (Text frames, if any, are
control messages handled by the transport, not here.)
"""

from __future__ import annotations

import logging

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # OpenAI Realtime requires >= 24 kHz PCM both ways.


class RawPCMSerializer(FrameSerializer):
    """Treats binary WebSocket messages as raw PCM16/24k/mono audio."""

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.BINARY

    async def deserialize(self, message: bytes) -> InputAudioRawFrame | None:
        if not isinstance(message, bytes):
            return None
        if len(message) % 2 != 0:
            logger.warning("Dropping odd-length audio frame (%d bytes)", len(message))
            return None
        return InputAudioRawFrame(audio=message, sample_rate=SAMPLE_RATE, num_channels=1)

    async def serialize(self, frame: Frame) -> bytes:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return b""
