"""SRT and WebVTT serialisation for transcript segments.

The same content stored in ``segments.json`` can be encoded as subtitle
cues that video players and end users consume directly.  Cue timestamps are
derived from the segment's ``start``/``end`` float seconds.
"""

from pathlib import Path
from typing import Final

from ._types import SubtitleFormat, TranscriptSegment


def format_timestamp(seconds: float | None, *, srt: bool = False) -> str:
    """Render float seconds as ``HH:MM:SS.mmm`` (VTT) or ``HH:MM:SS,mmm`` (SRT).

    ``None`` and negative values render as zero.
    """
    total_ms = 0 if seconds is None else max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def segment_to_cue(
    segment: TranscriptSegment,
    *,
    fmt: SubtitleFormat,
    index: int,
) -> str:
    """Render one segment as an SRT or WebVTT cue block (no trailing blank)."""
    # SRT indexes cues sequentially from 1; VTT uses the (possibly gapped)
    # segment sequence number as an optional cue identifier.
    identifier = str(index) if fmt == "srt" else str(segment.sequence)
    return "\n".join(
        (
            identifier,
            f"{format_timestamp(segment.start, srt=fmt == 'srt')} --> "
            f"{format_timestamp(segment.end, srt=fmt == 'srt')}",
            segment.content.strip(),
        )
    )

@final
class SubtitleWriter:
    """Incrementally writes transcript segments to an SRT or WebVTT file.

    WebVTT files start with a ``WEBVTT`` header; SRT files have none.  Cues
    are flushed one at a time so partial transcripts survive a crash,
    mirroring the incremental ``segments.json`` writes.
    """

    __slots__: Final[tuple[str, ...]] = ("path", "format", "_index", "_has_cue")

    def __init__(self, path: Path, fmt: SubtitleFormat = "vtt") -> None:
        self.path = path
        self.format: SubtitleFormat = fmt
        self._index = 1
        self._has_cue = False

    def open(self) -> None:
        """Create the file and write the format header (WebVTT only)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "WEBVTT\n\n" if self.format == "vtt" else "", encoding="utf-8"
        )

    def append(self, segment: TranscriptSegment) -> None:
        """Append one cue to the subtitle file."""
        with self.path.open("a", encoding="utf-8") as f:
            if self._has_cue:
                f.write("\n")
            f.write(segment_to_cue(segment, fmt=self.format, index=self._index))
            f.write("\n")
        self._index += 1
        self._has_cue = True

    def append_many(self, segments: list[TranscriptSegment]) -> None:
        """Append a batch of cues (convenience wrapper)."""
        for segment in segments:
            self.append(segment)

    @property
    def next_index(self) -> int:
        """The 1-based cue index the next appended segment will receive."""
        return self._index
