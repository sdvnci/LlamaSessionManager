"""Chat and transcription API — single endpoint for mixed-media context.

Relies entirely on the library's own primitives:
  - ``SessionManager`` for pool lifecycle and transcription submission.
  - ``_dispatch_events`` for automatic compact-reference ingestion.
  - ``build_transcript_context_message`` and ``parse_content`` for
    constructing the final LLM turn.

Endpoints
---------
GET  /transcription-enabled  reports whether transcription is enabled.
POST /chat/            mixed media → LLM (audio is transcribed first).
POST /transcribe/      audio → transcript → LLM summarisation.
POST /transcribeliteral/   audio → transcript + artifacts only (no LLM).
POST /ocr/             image → text via a vision model.
POST /document/        document → sectioned OCR.
WS   /ws/<session_uid> live stage/percent stream for transcription jobs.

Transcription capability is surfaced consistently on every transcription
touching body (handoffs, the web-socket snapshot/errors, and disabled-403s)
via a ``transcription_enabled`` field mirroring ``TRANSCRIPTION_ENABLED`` —
see ``GET /transcription-enabled``.

Transcription flow
------------------
The transcription endpoints are **asynchronous by default**.  Submit audio
and get back a handoff immediately:

    POST /transcribe/            (or /transcribeliteral/)

    {
        "session_uid": "<uuid>",
        "jobs": [{"job_uid": "…", "source_name": "meeting.mp3",
                  "transcript_path": "artifacts/…", "subtitle_path": "artifacts/…"}],
        "ws_url": "/ws/<session_uid>",
        "files_ingested": [],
        "async": true
    }

Connect a WebSocket to ``ws_url`` for queued/transcribing/percent updates
and a final ``done`` message (which carries the LLM ``reply`` for
/transcribe/).  Then download the artifacts and close the socket::

    GET /transcribe/artifact/artifacts/<session>/<job>/transcript.vtt
    GET /transcribeliteral/artifacts/<session>/<job>/transcript.vtt

Pass ``"wait": true`` in the payload (or ``?wait=1`` on a raw-body
/transcribeliteral/ request) to keep the old blocking behaviour, where the
HTTP response itself carries the transcript text and, for /transcribe/, the
``reply``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from logging import getLogger
from pathlib import Path
from typing import Any, Coroutine, Final, cast

from quart import Blueprint, jsonify, request, send_file, websocket
from quart.datastructures import FileStorage
from quart.wrappers import Response

from ..audio import AudioMimeTypeEnum, SubtitleFormat, TranscriptionJob
from ..common import ARTIFACTDIR
from ..common.settings import (
    transcription_capability,
    transcription_disabled_message,
    transcription_enabled,
)
from ..imaging import ImageMimeTypeEnum, OCREnabledDocumentEnum
from ..imaging.ocr import tesseract_ocr
from ..messaging import MessageSession, SessionConfig, SessionManager
from .hub import TranscriptionHub
from .utils import (
    build_combined_context,
    safe_json,
    save_raw_upload,
    save_upload,
    split_by_type,
)

LOGGER = getLogger("server.routes")

CWD = Path(__name__).parent.resolve()
TMP = CWD / "tmp"
UPLOADS = TMP / "uploads"
UPLOADS.mkdir(exist_ok=True, parents=True)

# Project root for computing relative paths in responses.
PROJECT_ROOT: Final[Path] = ARTIFACTDIR.parent

CHATBP = Blueprint(
    "Generalized Chatting",
    __name__,
    url_prefix="/chat",
)

OCRBP = Blueprint(
    "Generalized OCR",
    __name__,
    url_prefix="/ocr",
)

TRANSBP = Blueprint(
    "Transcription",
    __name__,
    url_prefix="/transcribe",
)

DOCOCRBP = Blueprint("Document-Specific OCR", __name__, url_prefix="/document")

LITERALBP = Blueprint(
    "Literal Transcription",
    __name__,
    url_prefix="/transcribeliteral",
)

# Root-level endpoints that are not scoped under a sub-path (e.g. feature
# status flags).
SETTINGSBP = Blueprint("Server Settings", __name__)

DEFAULT_INSTRUCTION: Final[str] = "Process the attached content."
DEFAULT_SUMMARY_MODEL: Final[str] = "tinyllama:latest"
DEFAULT_WHISPER_MODEL: Final[str] = "large-v3"
# Prefer the deployment's OLLAMA_HOST (docker-compose sets it to
# host.docker.internal); fall back to localhost for direct runs.
DEFAULT_HOST: Final[str] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TRANSCRIPT_PREVIEW_CHARS: Final[int] = 500
VALID_SUBTITLE_FORMATS: Final[tuple[str, ...]] = ("vtt", "srt")
ARTIFACT_MIME_BY_SUFFIX: Final[dict[str, str]] = {
    ".txt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
    ".srt": "application/x-subrip",
    ".json": "application/json",
}


def _parse_subtitle_format(payload: dict[str, Any]) -> SubtitleFormat:
    """Validate the payload's ``subtitle_format`` value (default ``"vtt"``)."""
    value = payload.get("subtitle_format", "vtt")
    if value not in VALID_SUBTITLE_FORMATS:
        raise ValueError(
            f"subtitle_format must be one of {VALID_SUBTITLE_FORMATS}, got {value!r}"
        )
    return cast(SubtitleFormat, value)


def _resolve_artifact(relpath: str) -> Path | None:
    """Resolve an artifact path, accepting project- or artifact-relative paths.

    Returns ``None`` unless the path stays inside ``ARTIFACTDIR`` and names
    an existing file, so ``..`` traversal cannot escape the artifact root.
    """
    artifact_root = ARTIFACTDIR.resolve()
    for base in (PROJECT_ROOT, ARTIFACTDIR):
        candidate = (base / relpath).resolve()
        if candidate.is_relative_to(artifact_root) and candidate.is_file():
            return candidate
    return None


async def _await_transcriptions(job_uids: list[str]) -> None:
    """Block until every transcription job reaches a terminal state.

    Jobs are removed from the registry once the dispatcher handles their
    terminal event, so ``None`` means "already done".
    """
    pending = set(job_uids)
    while pending:
        for uid in list(pending):
            job = await SessionManager.get_transcription(uid)
            if job is None or job.state.is_terminal:
                pending.discard(uid)
        if pending:
            await asyncio.sleep(1.0)


def _error_payload(
    source: str,
    etype: str,
    code: str,
    message: str,
    session_uid: str | None = None,
) -> tuple[Response, int]:
    """Render an error response in the same shape as the other routes."""
    status = 400 if etype == "argument" else 500
    return (
        jsonify(
            {
                "errors": [
                    {
                        "error_source": source,
                        "error_type": etype,
                        "error_code": code,
                        "message": message,
                    }
                ],
                "session_uid": session_uid,
            }
        ),
        status,
    )


def _feature_disabled_payload(
    source: str,
    message: str,
) -> tuple[Response, int]:
    """Render a 403 response for a feature that is disabled server-wide.

    Embeds the current transcription capability so the rejection body alone
    tells a feature-aware client the actual server state.
    """
    return (
        jsonify(
            {
                "errors": [
                    {
                        "error_source": source,
                        "error_type": "feature",
                        "error_code": "feature_disabled",
                        "message": message,
                    }
                ],
                "session_uid": None,
                **transcription_capability(),
            }
        ),
        403,
    )


async def _submit_transcription_jobs(
    sess: MessageSession,
    audio_paths: list[Path],
    whisper_model: str,
    subtitle_format: SubtitleFormat,
) -> list[TranscriptionJob]:
    """Queue one transcription job per audio file, bound to ``sess``."""
    jobs: list[TranscriptionJob] = []
    for audio_path in audio_paths:
        suffix = audio_path.suffix.lower().lstrip(".")
        try:
            mime = AudioMimeTypeEnum.from_extension(suffix)
        except ValueError:
            mime = AudioMimeTypeEnum.RAW

        job = await SessionManager.submit_transcription(
            session_uid=sess.uid,
            source=audio_path,
            source_name=audio_path.name,
            source_mime_type=mime,
            subtitle_format=subtitle_format,
            model_name=whisper_model,
        )
        jobs.append(job)
    return jobs


def _job_handoff(job: TranscriptionJob) -> dict[str, Any]:
    """The per-job slice of an async handoff response."""
    return {
        "job_uid": job.uid,
        "source_name": job.source.source_name if job.source else None,
        "transcript_path": str(job.transcript_path.relative_to(PROJECT_ROOT)),
        "subtitle_path": str(job.subtitle_path.relative_to(PROJECT_ROOT)),
    }


def _transcribe_preview_results(
    transcript_paths: list[Path],
    subtitle_paths: list[Path],
) -> list[dict[str, str | int]]:
    """Build the (truncated-preview) transcript list shared by responses."""
    results: list[dict[str, str | int]] = []
    for tp, sp in zip(transcript_paths, subtitle_paths):
        if tp.exists():
            raw = tp.read_text(encoding="utf-8")
            truncated = (
                raw
                if len(raw) <= TRANSCRIPT_PREVIEW_CHARS
                else raw[:TRANSCRIPT_PREVIEW_CHARS] + "..."
            )
            results.append(
                {
                    "path": str(tp.relative_to(PROJECT_ROOT)),
                    "subtitle_path": str(sp.relative_to(PROJECT_ROOT)),
                    "text": truncated,
                    "full_length": len(raw),
                }
            )
    return results


def _literal_transcript_payload(
    jobs: list[TranscriptionJob],
) -> list[dict[str, object]]:
    """Full-text transcript list for /transcribeliteral/ responses."""
    transcripts: list[dict[str, object]] = []
    for job in jobs:
        tp, segp, metap = job.transcript_path, job.segments_path, job.metadata_path
        text = tp.read_text(encoding="utf-8") if tp.exists() else ""

        state: str | None = None
        if metap.exists():
            try:
                state = json.loads(metap.read_text(encoding="utf-8")).get("state")
            except (json.JSONDecodeError, OSError):
                state = None

        # The worker never mutates the record it finalises, so the state in
        # metadata.json is stale (always "queued"). The transcript file on
        # disk is the truthful completion signal.
        state = "completed" if text.strip() else (state or job.state.value)

        transcripts.append(
            {
                "source_name": job.source.source_name if job.source else None,
                "job_uid": job.uid,
                "path": str(tp.relative_to(PROJECT_ROOT)) if tp.exists() else None,
                "subtitle_path": (
                    str(job.subtitle_path.relative_to(PROJECT_ROOT))
                    if job.subtitle_path.exists()
                    else None
                ),
                "text": text,
                "full_length": len(text),
                "segments_path": (
                    str(segp.relative_to(PROJECT_ROOT)) if segp.exists() else None
                ),
                "metadata_path": (
                    str(metap.relative_to(PROJECT_ROOT)) if metap.exists() else None
                ),
                "duration_seconds": job.duration_seconds,
                "state": state or job.state.value,
            }
        )
    return transcripts


async def _close_session_after(
    pipeline: Coroutine[Any, Any, Any], session_uid: str
) -> None:
    """Run a detached pipeline, report failures, and always release the session."""
    try:
        await pipeline
    except Exception as exc:
        LOGGER.exception(
            "transcription pipeline failed for session %s", session_uid[:8]
        )
        await TranscriptionHub.publish(
            session_uid,
            {
                "kind": "error",
                "session_uid": session_uid,
                "message": f"Transcription pipeline failed: {exc}",
            },
        )
    finally:
        await SessionManager.close_session(session_uid)


async def _run_transcribe_pipeline(
    sess: MessageSession,
    jobs: list[TranscriptionJob],
    other_paths: list[Path],
    instruction: str,
    keep_alive: str | None,
    think: Any,
    format: Any,
) -> dict[str, Any]:
    """Wait for transcription, summarise with the LLM, and publish over the WS."""
    transcript_paths = [j.transcript_path for j in jobs]
    subtitle_paths = [j.subtitle_path for j in jobs]
    if jobs:
        await _await_transcriptions([j.uid for j in jobs])

    transcript_results = _transcribe_preview_results(transcript_paths, subtitle_paths)

    await TranscriptionHub.publish(
        sess.uid,
        {
            "kind": "transcripts_ready",
            "session_uid": sess.uid,
            "transcripts": transcript_results,
        },
    )

    # Build the final turn and send to the LLM.
    for msg in build_combined_context(transcript_paths, other_paths, instruction, sess):
        await sess.add_message(msg)

    reply = await sess.send(
        stream=False, think=think, format=format, keep_alive=keep_alive
    )
    reply_text = reply.message.content or ""

    # Persist the summary alongside each job's artifacts so it can be
    # downloaded with the transcript later.
    for job in jobs:
        try:
            job.summary_path.write_text(reply_text, encoding="utf-8")
        except OSError:
            LOGGER.warning("Could not write summary for job %s", job.uid[:8])

    files_ingested = [p.name for p in other_paths]
    result: dict[str, Any] = {
        "reply": reply_text,
        "transcripts": transcript_results,
        "files_ingested": files_ingested,
        "session_uid": sess.uid[:8],
    }
    await TranscriptionHub.publish(
        sess.uid,
        {
            "kind": "done",
            "session_uid": sess.uid,
            "reply": reply_text,
            "transcripts": transcript_results,
            "files_ingested": files_ingested,
        },
    )
    return result


async def _run_literal_pipeline(
    sess: MessageSession,
    jobs: list[TranscriptionJob],
    other_paths: list[Path],
) -> dict[str, Any]:
    """Wait for Whisper-only transcription, then publish the artifacts over WS."""
    if jobs:
        await _await_transcriptions([j.uid for j in jobs])

    transcripts = _literal_transcript_payload(jobs)
    files_ignored = [p.name for p in other_paths]
    await TranscriptionHub.publish(
        sess.uid,
        {
            "kind": "done",
            "session_uid": sess.uid,
            "transcripts": transcripts,
            "files_ignored": files_ignored,
            "llm_involved": False,
        },
    )
    return {
        "transcripts": transcripts,
        "files_ignored": files_ignored,
        "llm_involved": False,
        "session_uid": sess.uid,
    }



async def transcription_socket(session_uid: str) -> None:
    """Stream transcription progress for one session over a WebSocket.

    Client flow: POST /transcribe (or /transcribeliteral) → get ``session_uid``
    → connect here → receive stage/percent updates → final ``done`` message
    (plus the LLM ``reply`` for /transcribe) → download artifacts → close.
    """
    known = await SessionManager.get_session(session_uid)
    capability = transcription_capability()
    if known is None:
        await websocket.send_json(
            {
                "kind": "error",
                "code": "unknown_session",
                "session_uid": session_uid,
                **capability,
                "message": (
                    "Session not found — it may have already finished "
                    "and been evicted."
                ),
            }
        )
        return

    queue = await TranscriptionHub.subscribe(session_uid)
    relay = asyncio.create_task(_relay_ws_queue(queue, websocket))
    try:
        # Late joiners see where every job stands right away.  Annotation of
        # the live capability keeps late/feature-aware clients in sync even
        # when they connect right at a toggle boundary.
        snapshot = await TranscriptionHub.snapshot(session_uid)
        await websocket.send_json({**snapshot, **capability})
        await relay
    finally:
        relay.cancel()
        await TranscriptionHub.unsubscribe(session_uid, queue)


async def _relay_ws_queue(queue: asyncio.Queue[str], ws: Any) -> None:
    """Pump JSON strings from a subscriber queue onto one WebSocket."""
    while True:
        message = await queue.get()
        try:
            await ws.send(message)
        except Exception:
            # The client went away mid-stream; stop relaying.  Cancellation
            # still propagates (CancelledError is not an Exception).
            return


@SETTINGSBP.route("/transcription-enabled", methods=["GET"])
async def transcription_enabled_status() -> tuple[Response, int]:
    """Report whether audio transcription is enabled on this server.

    Mirrors the ``TRANSCRIPTION_ENABLED`` environment variable at request
    time so clients can cheaply discover whether the Whisper worker is
    available before attempting a /transcribe/ call.
    """
    return jsonify(transcription_capability()), 200


"""
The following are routes for one-off services.
Transcribing, generating, or reading with OCR in a one-off message.
A session is spun up and force evicted all within the context of
the request and its response.
"""


@TRANSBP.route("/artifact/<path:relpath>", methods=["GET"])
async def download_artifact(relpath: str) -> tuple[Response, int]:
    """Serve a transcription artifact (transcript, segments, or subtitles).

    Accepts the project-relative ``path``/``subtitle_path`` values returned by
    the transcribe and chat endpoints (e.g.
    ``artifacts/<session>/<job>/transcript.vtt``), or the shorter
    artifact-relative form (``<session>/<job>/transcript.vtt``).
    """
    target = _resolve_artifact(relpath)
    if target is None:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/transcribe/artifact/",
                            "error_type": "system",
                            "error_code": "artifact_not_found",
                            "message": f"No such artifact: {relpath}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            404,
        )

    mimetype = ARTIFACT_MIME_BY_SUFFIX.get(target.suffix)
    return await send_file(target, mimetype=mimetype, conditional=True), 200


@TRANSBP.route("/", methods=["GET", "POST"])
async def transcribe() -> tuple[Response, int]:
    """Audio → transcript → optional LLM summarisation.

    By default this returns **immediately** (202) with the ``session_uid`` and
    a ``ws_url``; connect a WebSocket to ``/ws/<session_uid>`` for live
    stage/percent updates and the final summarisation.  Pass ``"wait": true``
    in the payload to keep the old blocking behaviour, where the response
    itself carries ``reply`` and ``transcripts``.
    """

    if not transcription_enabled():
        return _feature_disabled_payload(
            "/transcribe/", transcription_disabled_message()
        )

    try:
        files = (await request.files).getlist("files")

        payload = safe_json((await request.form).get("payload"))
        instruction = payload.get("instruction", DEFAULT_INSTRUCTION)
        llm_model = payload.get("model", DEFAULT_SUMMARY_MODEL)
        whisper_model = payload.get("whisper", DEFAULT_WHISPER_MODEL)
        host = payload.get("host", DEFAULT_HOST)
        context_length = payload.get("context_length", 8192)
        keep_alive = payload.get("keep_alive", "1m")
        think = payload.get("think")
        format = payload.get("format")
        subtitle_format = _parse_subtitle_format(payload)
        wait = bool(payload.get("wait", False))
    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/transcribe/",
                            "error_type": "argument",
                            "error_code": "field_value_required",
                            "message": str(exc),
                        }
                    ],
                    "session_uid": None,
                }
            ),
            400,
        )

    try:
        UPLOADS.mkdir(exist_ok=True, parents=True)
        saved: list[Path | None] = [await save_upload(UPLOADS, f) for f in files]
        audio_paths, other_paths = split_by_type(saved)
    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/transcribe/",
                            "error_type": "system",
                            "error_code": "attachment_not_supported",
                            "message": f"Error during saving attachments to temp dir::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    try:
        cfg = SessionConfig(
            model=llm_model, name="api-upload", host=host, context_length=context_length
        )
        sess = await SessionManager.create_session(config=cfg)
        jobs = await _submit_transcription_jobs(
            sess, audio_paths, whisper_model, subtitle_format
        )
    except Exception as exc:
        return _error_payload(
            "/transcribe/",
            "system",
            "unknown",
            f"Error during session::{str(exc)}",
        )

    pipeline = _run_transcribe_pipeline(
        sess, jobs, other_paths, instruction, keep_alive, think, format
    )

    if wait:
        try:
            result = await pipeline
        except Exception as exc:
            await SessionManager.evict_session(sess.uid, save=False)
            return _error_payload(
                "/transcribe/",
                "system",
                "unknown",
                f"Error during session::{str(exc)}",
                session_uid=sess.uid[:8],
            )
        await SessionManager.evict_session(sess.uid, save=False)
        return jsonify(result), 200

    # Async: hand the session UID back now; progress streams over the WS.
    asyncio.create_task(_close_session_after(pipeline, sess.uid))
    return (
        jsonify(
            {
                "session_uid": sess.uid,
                "jobs": [_job_handoff(job) for job in jobs],
                "ws_url": f"/ws/{sess.uid}",
                "files_ingested": [p.name for p in other_paths],
                "async": True,
                **transcription_capability(),
            }
        ),
        202,
    )


@LITERALBP.route("/", methods=["POST"])
async def transcribe_literal() -> tuple[Response, int]:
    """Transcribe audio with Whisper only — no LLM turn is ever sent.

    Two upload modes:

    * multipart/form-data — ``files`` field(s), optional ``payload`` JSON
      with a ``whisper`` model name and ``subtitle_format`` ("vtt"/"srt",
      drop-in replacement for /transcribe/).
    * raw body — send the file bytes verbatim with ``Content-Type:
      application/octet-stream``, the original name in the ``X-Filename``
      header, and optional ``?whisper=<model>`` and
      ``?subtitle_format=<vtt|srt>`` query params.  This path streams
      straight to disk with no multipart parsing — the fastest way to move
      large files in (e.g. ``curl --data-binary @meeting.mp3``).

    Two response modes:

    * default (async) — returns immediately (202) with ``session_uid`` and
      ``ws_url``; connect to ``/ws/<session_uid>`` for stage/percent updates
      and a final ``done`` message carrying the artifact paths.
    * ``wait`` — multipart payload ``{"wait": true}`` or raw-body query
      ``?wait=1`` restores the old blocking behaviour: the response itself
      contains the full transcript text and artifact paths.  There is no
      ``reply`` field in either mode — nothing was ever sent to a model.
    """

    if not transcription_enabled():
        return _feature_disabled_payload(
            "/transcribeliteral/", transcription_disabled_message()
        )

    try:
        content_type = (request.content_type or "").lower()
        files: list[FileStorage] = []
        if content_type.startswith("multipart/"):
            files = (await request.files).getlist("files")
            payload = safe_json((await request.form).get("payload"))
            whisper_model = payload.get("whisper", DEFAULT_WHISPER_MODEL)
            subtitle_format = _parse_subtitle_format(payload)
            wait = bool(payload.get("wait", False))
            raw_mode = False
        else:
            whisper_model = request.args.get("whisper", DEFAULT_WHISPER_MODEL)
            subtitle_format = _parse_subtitle_format(dict(request.args))
            wait = (request.args.get("wait", "0") or "0").lower() in (
                "1",
                "true",
                "yes",
            )
            raw_mode = True

        if raw_mode and request.content_length == 0:
            raise RuntimeError("Request body is empty.")
    except Exception as exc:
        return _error_payload(
            "/transcribeliteral/", "argument", "field_value_required", str(exc)
        )

    try:
        UPLOADS.mkdir(exist_ok=True, parents=True)
        saved: list[Path | None]
        if raw_mode:
            filename = request.headers.get("X-Filename") or "upload.bin"
            saved = [await save_raw_upload(UPLOADS, filename, request.body)]
            if saved[0] is None or saved[0].stat().st_size == 0:
                raise ValueError("Request body is empty.")
        else:
            saved = [await save_upload(UPLOADS, f) for f in files]

        audio_paths, other_paths = split_by_type(saved)
    except ValueError as exc:
        return _error_payload(
            "/transcribeliteral/", "argument", "field_value_required", str(exc)
        )
    except Exception as exc:
        return _error_payload(
            "/transcribeliteral/",
            "system",
            "attachment_not_supported",
            f"Error during saving attachments to temp dir::{str(exc)}",
        )

    if not audio_paths:
        return (
            jsonify(
                {
                    "transcripts": [],
                    "files_ignored": [p.name for p in saved if isinstance(p, Path)],
                    "llm_involved": False,
                    "session_uid": None,
                    "errors": [
                        {
                            "error_source": "/transcribeliteral/",
                            "error_type": "argument",
                            "error_code": "attachment_not_supported",
                            "message": "No audio files received.",
                        }
                    ],
                }
            ),
            400,
        )

    try:
        # A session is used only as a job container for artifact paths; its
        # model/host are never contacted because no LLM turn is ever sent.
        cfg = SessionConfig(
            model="whisper-only", name="literal-transcription", host=DEFAULT_HOST
        )
        sess = await SessionManager.create_session(config=cfg)
        jobs = await _submit_transcription_jobs(
            sess, audio_paths, whisper_model, subtitle_format
        )
    except Exception as exc:
        return _error_payload(
            "/transcribeliteral/",
            "system",
            "unknown",
            f"Error during session::{str(exc)}",
        )

    pipeline = _run_literal_pipeline(sess, jobs, other_paths)

    if wait:
        try:
            result = await pipeline
        except Exception as exc:
            await SessionManager.evict_session(sess.uid, save=False)
            return _error_payload(
                "/transcribeliteral/",
                "system",
                "unknown",
                f"Error during session::{str(exc)}",
                session_uid=sess.uid[:8],
            )
        await SessionManager.evict_session(sess.uid, save=False)
        return jsonify(result), 200

    asyncio.create_task(_close_session_after(pipeline, sess.uid))
    return (
        jsonify(
            {
                "session_uid": sess.uid,
                "jobs": [_job_handoff(job) for job in jobs],
                "ws_url": f"/ws/{sess.uid}",
                "files_ignored": [p.name for p in other_paths],
                "llm_involved": False,
                "async": True,
                **transcription_capability(),
            }
        ),
        202,
    )


@LITERALBP.route("/artifacts/<session_uid>/<job_uid>/<path:filename>", methods=["GET"])
async def literal_artifact(
    session_uid: str, job_uid: str, filename: str
) -> Response | tuple[Response, int]:
    """Download a completed artifact (transcript.txt, segments.json, ...)."""
    target = _resolve_artifact(f"{session_uid}/{job_uid}/{filename}")
    if target is None:
        return jsonify({"error": "Artifact not found."}), 404
    mimetype = ARTIFACT_MIME_BY_SUFFIX.get(target.suffix)
    return await send_file(target, mimetype=mimetype, conditional=True)


@OCRBP.route("/", methods=["POST"])
async def ocr() -> tuple[Response, int]:
    """
    Accept files for OCR. PDFs are rasterised to PNGs and swapped in-place.
    """

    try:
        files = (await request.files).getlist("files")
        if not files:
            raise RuntimeError("No files under 'files' field.")

        payload = safe_json((await request.form).get("payload"))
        instruction = cast(str, payload.get("instruction", DEFAULT_INSTRUCTION))
        llm_model = cast(str, payload.get("model", DEFAULT_SUMMARY_MODEL))
        host = payload.get("host", DEFAULT_HOST)
        context_length = payload.get("context_length", None)
        think = payload.get("think", False)
        format = payload.get("format", None)
        temperature = payload.get("temperature", None)
        num_predict = payload.get("num_predict", None)
        keep_alive = payload.get("keep_alive", None)

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/ocr/",
                            "error_type": "argument",
                            "error_code": "field_value_required",
                            "message": str(exc),
                        }
                    ],
                    "session_uid": None,
                }
            ),
            400,
        )

    try:
        UPLOADS.mkdir(exist_ok=True, parents=True)
        saved: list[Path | None] = [await save_upload(UPLOADS, f) for f in files]

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/ocr/",
                            "error_type": "system",
                            "error_code": "attachment_not_supported",
                            "message": f"Error during saving attachments to temp dir::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    try:
        # Rasterise PDFs to PNGs alongside the uploads, then swap them out.
        for i in reversed(range(len(saved))):
            p = saved[i]
            if p is None:  # Skip non-converted files
                continue
            if p.suffix.lower() != ".pdf":
                continue
            reader = ImageMimeTypeEnum.PDF.reader
            if reader is None:
                continue
            png, err = reader(p, UPLOADS)
            if err is not None:
                return jsonify({"error": f"Failed to rasterise {p.name}: {err}"}), 500
            saved[i] = png

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/chat/",
                            "error_type": "system",
                            "error_code": "attachment_not_supported",
                            "message": f"Error during pdf rasterization::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    try:
        cfg = SessionConfig(
            model=llm_model,
            name="ocr-upload",
            host=host,
            context_length=context_length,
            temperature=temperature,
            num_predict=num_predict,
        )

        async with SessionManager.scoped_session(
            config=cfg, save_on_exit=False
        ) as sess:
            msg = sess.parse_content(
                [instruction, *[s for s in saved if s is not None]]
            )
            await sess.add_message(msg)

            reply = await sess.send(
                stream=False, think=think, format=format, keep_alive=keep_alive
            )

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/chat/",
                            "error_type": "system",
                            "error_code": "unknown",
                            "message": f"Error during session::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "reply": reply.message.content,
                "files_ingested": [p.name for p in saved if isinstance(p, Path)],
                "session_uid": sess.uid[:8],
            }
        ),
        200,
    )


@CHATBP.route("/", methods=["POST"])
async def chat() -> tuple[Response, int]:
    """Accept mixed media, transcribe audio, then feed everything to the LLM."""

    try:
        files = (await request.files).getlist("files")
        payload = safe_json((await request.form).get("payload"))
        instruction = payload.get("instruction", DEFAULT_INSTRUCTION)
        llm_model = payload.get("model", DEFAULT_SUMMARY_MODEL)
        whisper_model = payload.get("whisper", DEFAULT_WHISPER_MODEL)
        host = payload.get("host", DEFAULT_HOST)
        context_length = payload.get("context_length")
        think = payload.get("think", None)
        format = payload.get("format", None)
        temperature = payload.get("temperature", None)
        num_predict = payload.get("num_predict", None)
        keep_alive = payload.get("keep_alive", None)
        subtitle_format = _parse_subtitle_format(payload)

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/chat/",
                            "error_type": "argument",
                            "error_code": "field_value_required",
                            "message": str(exc),
                        }
                    ],
                    "session_uid": None,
                }
            ),
            400,
        )

    try:
        UPLOADS.mkdir(exist_ok=True, parents=True)
        saved: list[Path | None] = [await save_upload(UPLOADS, f) for f in files]
        audio_paths, other_paths = split_by_type(saved)
    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/chat/",
                            "error_type": "system",
                            "error_code": "attachment_not_supported",
                            "message": f"Error during saving attachments to temp dir::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    # Chat is a mixed-media endpoint: when transcription is disabled, audio
    # attachments cannot be handled — reject them up front and let text/other
    # media chat proceed without ever warming the Whisper worker.
    if not transcription_enabled() and audio_paths:
        return _feature_disabled_payload(
            "/chat/", transcription_disabled_message()
        )

    try:
        cfg = SessionConfig(
            model=llm_model, name="api-upload", host=host, context_length=context_length
        )

        async with SessionManager.scoped_session(
            config=cfg, save_on_exit=False
        ) as sess:
            # Warm the Whisper worker only when transcription is on; this is
            # what pre-allocates the model's GPU memory. When disabled it is
            # skipped entirely so nothing is loaded.
            if transcription_enabled():
                await SessionManager.start_background_pool(model_name=whisper_model)

            # Submit each audio file.  The dispatcher auto-ingests a compact
            # reference note into this session on terminal completion.
            job_uids: list[str] = []
            transcript_paths: list[Path] = []
            subtitle_paths: list[Path] = []
            for audio_path in audio_paths:
                suffix = audio_path.suffix.lower().lstrip(".")
                try:
                    mime = AudioMimeTypeEnum.from_extension(suffix)
                except ValueError:
                    mime = AudioMimeTypeEnum.RAW

                job = await SessionManager.submit_transcription(
                    session_uid=sess.uid,
                    source=audio_path,
                    source_name=audio_path.name,
                    source_mime_type=mime,
                    subtitle_format=subtitle_format,
                )
                job_uids.append(job.uid)
                transcript_paths.append(job.transcript_path)
                subtitle_paths.append(job.subtitle_path)

            # Wait for all transcription jobs to finish.
            if job_uids:
                pending = set(job_uids)
                while pending:
                    for uid in list(pending):
                        j = await SessionManager.get_transcription(uid)
                        if j is None or j.state.is_terminal:
                            pending.discard(uid)
                            continue

                    await asyncio.sleep(1.0)

            # Build the final turn and send to the LLM.
            for msg in build_combined_context(
                transcript_paths, other_paths, instruction, sess
            ):
                await sess.add_message(msg)

            reply = await sess.send(
                stream=False, think=think, format=format, keep_alive=keep_alive
            )

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/chat/",
                            "error_type": "system",
                            "error_code": "unknown",
                            "message": f"Error during session::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    # Build response: include actual transcript text, not just audio filenames.
    transcript_results: list[dict[str, str | int]] = []
    for tp, sp in zip(transcript_paths, subtitle_paths):
        if tp.exists():
            raw = tp.read_text(encoding="utf-8")
            truncated = (
                raw
                if len(raw) <= TRANSCRIPT_PREVIEW_CHARS
                else raw[:TRANSCRIPT_PREVIEW_CHARS] + "…"
            )
            transcript_results.append(
                {
                    "path": str(tp.relative_to(PROJECT_ROOT)),
                    "subtitle_path": str(sp.relative_to(PROJECT_ROOT)),
                    "text": truncated,
                    "full_length": len(raw),
                }
            )

    return (
        jsonify(
            {
                "reply": reply.message.content,
                "transcripts": transcript_results,
                "files_ingested": [p.name for p in other_paths],
                "session_uid": sess.uid[:8],
            }
        ),
        200,
    )


@DOCOCRBP.route("/", methods=["POST"])
async def dococr() -> tuple[Response, int]:
    """
    Perform ocr on a specifc document type with pre-defined sections.
    Each section has ocr performed on it, and has an attached at a class level, a summarization to specifc the
    JSON format to get the information out as.
    Each one is done in a separate session, and brought back together afterwards.
    """
    try:
        files: list[FileStorage] = (await request.files).getlist("files")
        if not files:
            raise RuntimeError("No files under 'files' field.")

        payload = safe_json((await request.form).get("payload"))
        llm_model = cast(str, payload.get("model", "deepseek-ocr"))
        host = payload.get("host", DEFAULT_HOST)
        context_length = payload.get("context_length", None)
        doctype = cast(str | None, payload.get("document_type"))
        think = payload.get("think", False)
        fmt = payload.get("format")
        temperature = payload.get("temperature", 0.1)
        repeat_penalty = payload.get("repeat_penalty", 1.2)
        num_predict = payload.get("num_predict", 512)
        instruction = cast(
            str,
            payload.get(
                "instruction",
                "Transcribe ALL visible text. Output ONLY plain text. Never use HTML, tables, markdown, or formatting. One line per row of text.",
            ),
        )

        if doctype is None:
            return (
                jsonify(
                    {
                        "errors": [
                            {
                                "error_source": "/document/",
                                "error_type": "argument",
                                "error_code": "field_value_required",
                                "message": 'Callers must specify document type via the "doctype" argument',
                            }
                        ],
                        "session_uid": None,
                    }
                ),
                400,
            )

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/document/",
                            "error_type": "argument",
                            "error_code": "field_value_required",
                            "message": str(exc),
                        }
                    ],
                    "session_uid": None,
                }
            ),
            400,
        )

    try:
        UPLOADS.mkdir(exist_ok=True, parents=True)
        to_store = [f for f in files]
        if len(to_store) > 1:
            return (
                jsonify(
                    {
                        "errors": [
                            {
                                "error_source": "/document/",
                                "error_type": "system",
                                "error_code": "multiple_attachments_not_supported",
                                "message": "Error during saving attachments to temp dir",
                            }
                        ],
                        "session_uid": None,
                    }
                ),
                400,
            )
        if len(to_store) == 0:
            return (
                jsonify(
                    {
                        "errors": [
                            {
                                "error_source": "/document/",
                                "error_type": "system",
                                "error_code": "multiple_attachments_not_supported",
                                "message": "Error during saving attachments to temp dir",
                            }
                        ],
                        "session_uid": None,
                    }
                ),
                400,
            )

        saved: Path = await save_upload(UPLOADS, to_store[0])
        imtype = ImageMimeTypeEnum.from_extension(saved.suffix)
        reader = imtype.reader
        if reader is None:
            raise RuntimeError("Failed to rasterize image attachment")
        png, err = reader(saved, UPLOADS, limit=2)
        if err is not None:
            raise RuntimeError("Failed to rasterize image attachment")
        if png is None:
            raise RuntimeError("Failed to rasterize image attachment")
        saved = png
        doc_dir = UPLOADS / Path(saved).stem
        doc_dir.mkdir(exist_ok=True)

    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/document/",
                            "error_type": "system",
                            "error_code": "attachment_not_supported",
                            "message": f"Error during saving attachments to temp dir::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    try:
        docenumtype = OCREnabledDocumentEnum.from_string(doctype)
        slicer_cls = docenumtype.cls
        slicerobj = slicer_cls(image=saved)  # type: ignore[arg-type]
        segments = slicerobj.segment_document()
        images = [segment[1] for segment in segments]
        sections = slicerobj.sections
    except Exception as exc:
        return (
            jsonify(
                {
                    "errors": [
                        {
                            "error_source": "/document/",
                            "error_type": "system",
                            "error_code": "slicing_failed",
                            "message": f"Error during document slicing::{str(exc)}",
                        }
                    ],
                    "session_uid": None,
                }
            ),
            500,
        )

    prog_indices = [i for i, s in enumerate(sections) if s.programmatic]
    llm_indices = [i for i, s in enumerate(sections) if not s.programmatic]

    result: dict[str, str] = {}
    session_uids: list[str] = []

    if prog_indices:
        prog_imagepaths: dict[int, Path] = {}
        for idx in prog_indices:
            label = sections[idx].label or f"section_{idx}"
            p = doc_dir / f"{label}{Path(saved).suffix}"
            images[idx].save(str(p), dpi=(72, 72))
            prog_imagepaths[idx] = p

        async def _run_tesseract(idx: int) -> None:
            label = sections[idx].label or f"section_{idx}"
            try:
                text = await asyncio.to_thread(tesseract_ocr, prog_imagepaths[idx])
            except Exception:
                text = ""
            result[label] = text.strip()

        await asyncio.gather(*(_run_tesseract(i) for i in prog_indices))

    if llm_indices:
        try:
            llm_imagepaths: dict[int, Path] = {}
            for idx in llm_indices:
                label = sections[idx].label or f"section_{idx}"
                p = doc_dir / f"{label}{Path(saved).suffix}"
                images[idx].save(str(p), dpi=(72, 72))
                llm_imagepaths[idx] = p
            cfg = SessionConfig(
                model=llm_model,
                name="ocr-upload",
                host=host,
                context_length=context_length,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                num_predict=num_predict,
                system=(
                    "You are a precise OCR transcription engine. "
                    "Output ONLY the text visible in the image as plain text. "
                    "ABSOLUTELY NO HTML tags, NO markdown, NO tables, NO formatting of any kind. "
                    "Plain text characters only. No code fences. No commentary. "
                    "NEVER add citations, legal references, source attributions, or any text not in the image. "
                    "If you output anything not present in the image you have failed."
                ),
            )
            sessions_map: dict[int, MessageSession] = {}
            for idx in llm_indices:
                sess = await SessionManager.create_session(cfg)
                sessions_map[idx] = sess

            for idx in llm_indices:
                sess = sessions_map[idx]
                img_path = llm_imagepaths[idx]
                assert isinstance(
                    img_path, Path
                ), f"llm_imagepaths[{idx}] is {type(img_path).__name__}, expected Path"
                assert img_path.exists(), f"Slice image missing: {img_path}"
                msg = sess.parse_content(
                    [sections[idx].prompt or instruction, img_path]
                )
                await sess.add_message(msg)

            replies = []
            for idx in llm_indices:
                replies.append(
                    await sessions_map[idx].send(stream=False, think=think, format=fmt)
                )
                await asyncio.sleep(0.3)

            for idx, reply in zip(llm_indices, replies):
                label = sections[idx].label or f"section_{idx}"
                raw = (reply.message.content or "").strip()
                # Strip HTML tags and markdown that the OCR model may output
                raw = re.sub(r"<[^>]+>", " ", raw)
                raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)  # bold
                raw = re.sub(r"__([^_]+)__", r"\1", raw)
                raw = re.sub(r"#{1,6}\s*", "", raw)  # headings
                # Strip base64 image references
                raw = re.sub(r"\[ref\d+\]:\s*data:image[^]]*\]", "", raw)
                raw = re.sub(r"<\|im_end\|>", "", raw)
                raw = re.sub(r"\|\s*-+\s*\|", " ", raw)  # markdown table separators
                raw = re.sub(r"\|\s*:-+\s*\|", " ", raw)  # markdown alignment rows
                # Strip meta prefixes the model sometimes adds
                raw = re.sub(r"^\[Document Title\]\s*", "", raw)
                raw = re.sub(r"^Transcription\s*", "", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
                # Filter out standalone "None" artifacts from empty HTML cells
                while True:
                    new_raw = raw.replace(" None ", " ").replace("NoneNone", "").strip()
                    # Also catch None adjacent to punctuation/other text: (S)NoneJALOUSIE
                    new_raw = re.sub(r"(\S)None(\S)", r"\1 \2", new_raw)
                    new_raw = new_raw.strip()
                    if new_raw == raw:
                        break
                    raw = new_raw
                # Clean leading/trailing None
                raw = re.sub(r"^None\s+", "", raw)
                raw = re.sub(r"\s+None$", "", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
                result[label] = raw
                session_uids.append(sessions_map[idx].uid[:8])

            await asyncio.gather(
                *(
                    SessionManager.evict_session(s.uid, save=False)
                    for s in sessions_map.values()
                )
            )
        except Exception as exc:
            return (
                jsonify(
                    {
                        "errors": [
                            {
                                "error_source": "/document/",
                                "error_type": "system",
                                "error_code": "llm_ocr_failed",
                                "message": (
                                    f"Error during LLM OCR::{str(exc)} "
                                    f"[saved={type(saved).__name__}, "
                                    f"images={type(images).__name__}, "
                                    f"llm_indices={llm_indices}]"
                                ),
                            }
                        ],
                        "session_uid": None,
                    }
                ),
                500,
            )

    return (
        jsonify(
            {
                "reply": result,
                "files_ingested": [saved.name],
                "session_uid": session_uids,
            }
        ),
        200,
    )
