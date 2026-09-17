from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast, overload

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from ..common import ARTIFACTDIR, CreateStamped, MimeTypeEnum, UIDTagged

AudioExtensions = Literal[
    "mp3",
    "wav",
    "ogg",
    "flac",
    "aac",
    "m4a",
    "opus",
    "webm",
    "wma",
    "aiff",
    "amr",
    "midi",
    "mid",
    "raw",
]

SubtitleFormat = Literal["vtt", "srt"]


class AudioMimeTypeEnum(MimeTypeEnum):
    """
    Common MIME types for audio files.
    """

    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    OGG = "audio/ogg"
    FLAC = "audio/flac"
    AAC = "audio/aac"
    M4A = "audio/mp4"
    OPUS = "audio/opus"
    WEBM = "audio/webm"
    WMA = "audio/x-ms-wma"
    AIFF = "audio/aiff"
    AMR = "audio/amr"
    MIDI = "audio/midi"
    RAW = "audio/raw"

    @property
    def extension(self) -> AudioExtensions:
        return cast(AudioExtensions, self.value.lstrip("audio/"))

    @overload
    @classmethod
    def extension_matcher(
        cls: type["AudioMimeTypeEnum"], mime_type: AudioExtensions
    ) -> "AudioMimeTypeEnum": ...
    @overload
    @classmethod
    def extension_matcher(
        cls: type["AudioMimeTypeEnum"], mime_type: str
    ) -> "AudioMimeTypeEnum": ...
    @classmethod
    def extension_matcher(
        cls: type["AudioMimeTypeEnum"], mime_type: str
    ) -> "AudioMimeTypeEnum":
        match mime_type:
            case "mp3":
                return cls.MP3
            case "wav":
                return cls.WAV
            case "ogg":
                return cls.OGG
            case "flac":
                return cls.FLAC
            case "aac":
                return cls.AAC
            case "m4a" | "mp4":
                return cls.M4A
            case "opus":
                return cls.OPUS
            case "webm":
                return cls.WEBM
            case "wma":
                return cls.WMA
            case "aiff":
                return cls.AIFF
            case "amr":
                return cls.AMR
            case "midi" | "mid":
                return cls.MIDI
            case "raw":
                return cls.RAW
            case _:
                raise ValueError(f"Unsupported audio MIME type: {mime_type}")


class AudioProcessState(StrEnum):
    """Lifecycle of a transcription job. Use ``phase`` for fine stages."""

    CREATED = "created"
    QUEUED = "queued"
    DECODING = "decoding"
    TRANSCRIBING = "transcribing"
    WRITING = "writing"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @cached_property
    def is_terminal(self) -> bool:
        return self in (
            AudioProcessState.COMPLETED,
            AudioProcessState.FAILED,
            AudioProcessState.CANCELLED,
        )


class EventKind(StrEnum):
    STATUS = "status"
    PROGRESS = "progress"
    SEGMENT = "segment"
    WARNING = "warning"
    ERROR = "error"
    COMPLETION = "completion"


class TranscriptSegmentJSON(TypedDict):
    sequence: int
    content: str
    created_at: str
    start: float | None
    end: float | None


class TranscriptSegment(UIDTagged):
    """
    A timestamped transcript segment.
    A segment's ordering is given by the `sequence` number.
    A segment spans from second `start` to `end` in the recording.
    """

    sequence: int
    content: str
    start: float | None = None
    end: float | None = None
    created_at: Annotated[
        datetime,
        Field(default_factory=lambda: datetime.now(tz=UTC)),
        PlainSerializer(lambda v: v.strftime("%Y-%m-%d %H:%M:%S"), return_type=str),
    ]


@dataclass(slots=True, frozen=True)
class TranscriptProgress:
    """One source chunk's transcription outcome plus pipeline position.

    ``segment`` is ``None`` when the chunk produced no text (VAD-filtered or
    silent); ``fraction`` is the portion of the recording processed so far
    (``0.0``–``1.0``) and advances regardless, so callers can render a live
    percentage.
    """

    segment: TranscriptSegment | None
    fraction: float
    processed_seconds: float
    duration_seconds: float


class TranscriptionAudioSource(UIDTagged):
    """
    Some source of audio data to be transcribed. This can be a file path, raw bytes, or a URL.
    Attributes:
        - `source_path`: Optional file path to the audio source.
        - `source_bytes`: Raw audio data in bytes.
        - `source_name`: A human-readable name for the audio source.
        - `source_mime_type`: The MIME type of the audio source, defaulting to `AudioMimeTypeEnum.RAW`.
        - `source_size_bytes`: The size of the audio source in bytes.
    """

    source_path: Path | None = None
    source_bytes: bytes | None = None
    source_name: str | None = None
    source_mime_type: AudioMimeTypeEnum = AudioMimeTypeEnum.RAW
    source_size_bytes: int | None = None

    def get_source(self) -> Path | bytes | None:
        return self.source_path if self.source_path else self.source_bytes


class TranscriptionJob(UIDTagged):
    """
    A request to transcribe an audio source. This is the input to the transcription system.
    A request is associated with a session, and may optionally specify an audio source.
    Attributes:
    - `session_uid`: The unique identifier of the session this request belongs to.
    - `transcript_path`: The file path where the final transcript will be saved.
    - `segments_path`: The file path where the transcript segments will be saved.
    - `metadata_path`: The file path where the transcript metadata will be saved.
    - `subtitle_path`: The file path where the transcript subtitles will be saved.
    - `subtitle_format`: The subtitle container format (``"vtt"`` or ``"srt"``).
    - `source`: An optional `TranscriptionAudioSource` containing the audio data to be transcribed.
    - `immediate`: A flag indicating whether the transcription should be processed immediately or can be queued for later processing.
    """

    session_uid: str
    subtitle_format: SubtitleFormat = "vtt"
    state: AudioProcessState = AudioProcessState.CREATED
    duration_seconds: float | None = None
    processed_seconds: float | None = None
    source: TranscriptionAudioSource | None = None
    immediate: bool = False

    class Reference(UIDTagged):
        path: Path
        duration: float | None
        summary_path: Path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TranscriptionJob):
            return NotImplemented
        return self.uid == other.uid

    @property
    def artifact_dir(self) -> Path:
        return ARTIFACTDIR / f"{self.session_uid}" / f"{self.uid}"

    @property
    def segments_path(self) -> Path:
        return self.artifact_dir / "segments.json"

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "metadata.json"

    @property
    def transcript_path(self) -> Path:
        return self.artifact_dir / "transcript.txt"

    @property
    def subtitle_path(self) -> Path:
        return self.artifact_dir / f"transcript.{self.subtitle_format}"

    @property
    def summary_path(self) -> Path:
        return self.artifact_dir / "summary.txt"

    @property
    def reference(self) -> Reference:
        """
        Render a completed transcript into a short, context-friendly note.

        Deliberately omits the full transcript body so chat history stays small;
        the model is told where to load it from on demand.
        """

        return self.Reference(
            uid=self.uid,
            path=self.transcript_path,
            duration=self.duration_seconds,
            summary_path=self.summary_path,
        )


class AudioProcessingEvent(UIDTagged, CreateStamped):
    """
    An event emitted during the audio processing lifecycle.
    This can be used for logging, monitoring, or real-time updates.
    """

    session_uid: str
    job_uid: str
    kind: EventKind
    state: AudioProcessState
    sequence: int
    message: str | None = None
    progress: float | None = None
    processed_seconds: float | None = None
    duration_seconds: float | None = None


class TranscriptArtifact(BaseModel):
    """
    File-backed transcript persistence for Transcription information.
    This is where we store/read segments and metadata.
    (JSONL), and a metadata snapshot. One ``TranscriptArtifact`` wraps one bundle.

    Use this sparingly, as this does many calls to disk.
    """

    model_config = ConfigDict(slots=True)  # type: ignore
    record: TranscriptionJob

    def write_metadata(self) -> None:
        self.record.metadata_path.write_text(
            dumps(self.record.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def ensure_exists(self) -> None:
        """Create the artifact directory and empty transcript/segment files."""
        Path(self.record.artifact_dir).mkdir(parents=True, exist_ok=True)
        self.record.transcript_path.touch(exist_ok=True)
        self.record.segments_path.touch(exist_ok=True)
        self.write_metadata()

    def append_segment(self, segment: TranscriptSegment) -> None:
        """Write a single segment as one JSON line — call as each segment arrives."""
        with self.record.segments_path.open("a", encoding="utf-8") as f:
            f.write(dumps(segment.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")

    def append_segments(self, segments: list[TranscriptSegment]) -> None:
        """Write a batch of segments (convenience wrapper)."""
        for seg in segments:
            self.append_segment(seg)

    def append_text(self, text: str) -> None:
        if not text:
            return

        with self.record.transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"{text.strip()}\n")

    def read_text(self) -> str:
        if not self.record.transcript_path.exists():
            return ""

        return self.record.transcript_path.read_text(encoding="utf-8")

    def segments(self) -> Iterator[TranscriptSegment]:
        if not self.record.segments_path.exists():
            return

        yield from (
            TranscriptSegment(**loads(line.strip()))
            for line in self.record.segments_path.open("r", encoding="utf-8")
        )

    def finalize(self) -> None:
        """Write a completion marker and final metadata snapshot."""
        self.record.metadata_path.write_text(
            dumps(
                {"state": self.record.state.value, "finalized": True}
                | self.record.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def __bytes__(self) -> bytes:
        return self.read_text().encode("utf-8")

    def __hash__(self) -> int:
        return hash(sha256(bytes(self)).hexdigest())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TranscriptArtifact):
            return False

        return self.record == other.record
