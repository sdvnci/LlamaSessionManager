"""Public API for the audio module — transcription, workers, and types."""

from ._types import (
    AudioMimeTypeEnum,
    AudioProcessingEvent,
    AudioProcessState,
    EventKind,
    SubtitleFormat,
    TranscriptArtifact,
    TranscriptionAudioSource,
    TranscriptionJob,
    TranscriptProgress,
    TranscriptSegment,
)
from .subtitles import SubtitleWriter, format_timestamp, segment_to_cue
from .vad import VoiceActivityDetector
from .whispering import AudioSourceType, TranscriptionService
from .workers import BackgroundWhisperProcessPool, TranscriptionWorker

__all__ = (
    "AudioSourceType",
    "AudioMimeTypeEnum",
    "AudioProcessingEvent",
    "AudioProcessState",
    "BackgroundWhisperProcessPool",
    "EventKind",
    "SubtitleFormat",
    "SubtitleWriter",
    "TranscriptArtifact",
    "TranscriptProgress",
    "TranscriptSegment",
    "TranscriptionAudioSource",
    "TranscriptionJob",
    "TranscriptionService",
    "TranscriptionWorker",
    "VoiceActivityDetector",
    "format_timestamp",
    "segment_to_cue",
)
