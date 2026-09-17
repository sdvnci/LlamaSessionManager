import json
from pathlib import Path
from typing import Any, Final

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.audio import AudioMimeTypeEnum
from src.messaging import MessageSession
from src.textual import TextualMimeTypeEnum

CWD: Path = Path(__file__).parent.resolve()
STATIC_DIR: Final[Path] = CWD / "static"
VIEW_DIR: Final[Path] = STATIC_DIR / "views"
CSS_DIR: Final[Path] = STATIC_DIR / "css"
JS_DIR: Final[Path] = STATIC_DIR / "js"


def safe_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


async def save_upload(
    directory: Path,
    file: FileStorage,
) -> Path:
    """Save an upload to *directory*, returning its path."""
    name = secure_filename(file.filename or "upload.bin")
    dest = directory / name
    if not dest.exists():
        await file.save(dest)  # type: ignore[func-returns-value, misc]
        return dest

    stem = Path(name).stem
    ext = Path(name).suffix
    suffix = 1
    while True:
        dest = directory / f"{stem}({suffix}){ext}"
        if not dest.exists():
            await file.save(dest)  # type: ignore[func-returns-value, misc]
            return dest
        suffix += 1


async def save_raw_upload(
    directory: Path,
    filename: str,
    stream: Any,
) -> Path:
    """Stream a raw (non-multipart) request body straight to disk.

    Bypasses multipart parsing entirely — the client sends the file bytes
    verbatim as the request body (e.g. ``curl --data-binary @file``) with the
    original name in the ``X-Filename`` header.  This is the lowest-overhead
    upload path and is ideal for large audio files on a LAN.
    """
    name = secure_filename(filename or "upload.bin")
    dest = directory / name
    if dest.exists():
        stem = Path(name).stem
        ext = Path(name).suffix
        suffix = 1
        while True:
            candidate = directory / f"{stem}({suffix}){ext}"
            if not candidate.exists():
                dest = candidate
                break
            suffix += 1

    with dest.open("wb") as f:
        async for chunk in stream:
            f.write(chunk)
    return dest


def build_combined_context(
    transcript_paths: list[Path],
    other_paths: list[Path],
    instruction: str,
    sess: MessageSession,
) -> list:
    """Build the sequence of messages that form the final LLM turn.

    Transcripts use ``build_transcript_context_message`` (the library
    helper that packages the full transcript text alongside an
    instruction).  Non-audio files are fed through ``parse_content``.
    """
    messages: list = []

    # Transcripts: one message each via the library helper.
    for tp in transcript_paths:
        if not tp.exists():
            continue
        text = tp.read_text(encoding="utf-8")
        if not text.strip():
            continue
        messages.append(
            MessageSession.build_transcript_context_message(tp, instruction)
        )

    # Non-audio files: one message bundling all of them via parse_content.
    if other_paths:
        messages.append(sess.parse_content([instruction, *other_paths]))

    # If there were no transcripts and no files, just the instruction.
    if not messages:
        messages.append(sess.parse_content(instruction))

    return messages


def split_by_type(paths: list[Path | None]) -> tuple[list[Path], list[Path]]:
    """Split paths into ``(audio, non_audio)`` via ``AudioMimeTypeEnum``."""
    audio: list[Path] = []
    other: list[Path] = []
    for p in paths:
        if p is None:
            continue

        if AudioMimeTypeEnum.is_valid(p.suffix):
            audio.append(p)
            continue

        if TextualMimeTypeEnum.is_valid(p.suffix):
            other.append(p)

    return audio, other
