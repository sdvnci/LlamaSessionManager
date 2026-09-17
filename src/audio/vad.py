"""Voice activity detection for pre-filtering transcription buffers.

Meetings contain long stretches of silence or ambient noise.  Transcribing
those buffers wastes GPU time and invites whisper hallucinations, so every
buffer is screened with WebRTC's VAD before it reaches the model.
"""
from collections.abc import Iterator
from typing import Final

import numpy as np
from numpy.typing import NDArray
from webrtcvad import Vad  # type: ignore[import-untyped]

# Whisper decodes everything to 16 kHz mono float32, which is exactly what
# WebRTC VAD expects (converted to 16-bit PCM before scoring).
VAD_SAMPLE_RATE: Final[int] = 16000
VALID_VAD_FRAME_MS: Final[tuple[int, ...]] = (10, 20, 30)
VALID_VAD_MODES: Final[tuple[int, ...]] = (0, 1, 2, 3)


class VoiceActivityDetector:
    """Screens float32 mono buffers for speech using WebRTC VAD.

    ``mode`` selects the VAD aggressiveness: 0 is the least aggressive
    (most permissive) and 3 the most aggressive.  Higher modes reject more
    ambient noise.  Mode 2 — the default — ignores white noise below
    roughly -30 dBFS in testing.

    ``min_speech_ratio`` is the minimum fraction of frames that must be
    flagged as speech for :meth:`contains_speech` to return ``True``.  The
    comparison is strict, so the default ``0.0`` means "any detected speech
    counts" while an all-silent buffer is still rejected.
    """

    __slots__ = ("_vad", "frame_ms", "min_speech_ratio")

    def __init__(
        self,
        mode: int = 2,
        frame_ms: int = 30,
        min_speech_ratio: float = 0.0,
    ) -> None:
        if mode not in VALID_VAD_MODES:
            raise ValueError(f"VAD mode must be one of {VALID_VAD_MODES}, got {mode}")
        if frame_ms not in VALID_VAD_FRAME_MS:
            raise ValueError(
                f"VAD frame length must be one of {VALID_VAD_FRAME_MS}, got {frame_ms}"
            )
        if not 0.0 <= min_speech_ratio <= 1.0:
            raise ValueError(
                f"min_speech_ratio must lie within [0, 1], got {min_speech_ratio}"
            )
        self._vad = Vad(mode)
        self.frame_ms = frame_ms
        self.min_speech_ratio = min_speech_ratio

    @staticmethod
    def _to_pcm16(audio: NDArray[np.float32]) -> bytes:
        """Convert a -1..1 float32 waveform to 16-bit little-endian PCM."""
        return (
            (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False).tobytes()
        )

    def frames(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> Iterator[bool]:
        """Yield a speech/non-speech verdict per fixed-length frame."""
        frame_samples = sample_rate * self.frame_ms // 1000
        frame_bytes = frame_samples * 2  # 16-bit PCM
        pcm = self._to_pcm16(audio)
        for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[offset : offset + frame_bytes]
            yield self._vad.is_speech(frame, sample_rate)

    def speech_ratio(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> float:
        """Return the fraction of frames flagged as speech (0.0 for empty)."""
        verdicts = list(self.frames(audio, sample_rate))
        if not verdicts:
            return 0.0
        return sum(verdicts) / len(verdicts)

    def contains_speech(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> bool:
        """True when the buffer holds enough speech to be worth transcribing."""
        return self.speech_ratio(audio, sample_rate) > self.min_speech_ratio
