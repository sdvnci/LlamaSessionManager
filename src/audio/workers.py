"""
Defs for Long-lived transcription workers and the background subsystem that drives them.

The worker runs in its own process, owns one Whisper model, pulls
``TranscriptionJobRequest``s off an IPC queue, and emits ``AudioProcessEvent``s.
It never touches sockets or live sessions — the main process owns those.
"""

import logging as _logging
import traceback
from asyncio import get_running_loop
from logging import getLogger
from multiprocessing import get_context
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any, Final, final

from ..common import Sentinel
from ._types import (
    AudioProcessingEvent,
    AudioProcessState,
    EventKind,
    TranscriptArtifact,
    TranscriptionJob,
)
from .subtitles import SubtitleWriter
from .whispering import TranscriptionService

_logging.basicConfig(
    level=_logging.DEBUG,
    format="%(asctime)s [worker] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

_STOP = Sentinel()


def _worker_entry(model_name: str, job_queue: Queue, event_queue: Queue) -> None:
    """Module-level entry point for the worker subprocess (must be picklable)."""
    try:
        TranscriptionWorker(model_name).run(job_queue, event_queue)
    except Exception:
        traceback.print_exc()


@final
class TranscriptionWorker:
    __slots__: Final[tuple[str, ...]] = (
        "_sequence",
        "_service",
    )

    logger = getLogger("audio.worker")

    def __init__(self: "TranscriptionWorker", model: str) -> None:
        self._service = TranscriptionService(model_name=model)
        self._sequence: int = 0

    def _emit(
        self,
        event_queue: Queue[AudioProcessingEvent | Sentinel],
        job: TranscriptionJob,
        kind: EventKind,
        state: AudioProcessState,
        message: str | None = None,
        progress: float | None = None,
        processed_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        event_queue.put(
            AudioProcessingEvent(  # type: ignore[call-arg]
                kind=kind,
                job_uid=job.uid,
                session_uid=job.session_uid,
                sequence=self._sequence,
                state=state,
                message=message,
                progress=progress,
                processed_seconds=processed_seconds,
                duration_seconds=duration_seconds,
            )
        )

        self._sequence += 1

    def handle_job(
        self: "TranscriptionWorker",
        job: TranscriptionJob,
        event_queue: Queue[AudioProcessingEvent | Sentinel],
    ) -> None:
        artifact = TranscriptArtifact(record=job)
        artifact.ensure_exists()
        subtitle_writer = SubtitleWriter(job.subtitle_path, job.subtitle_format)
        subtitle_writer.open()
        self._emit(
            event_queue=event_queue,
            job=job,
            kind=EventKind.STATUS,
            state=AudioProcessState.QUEUED,
            message="Processing started",
            progress=0.0,
        )

        source = job.source
        if source is None:
            self._emit(
                event_queue,
                job=job,
                state=AudioProcessState.FAILED,
                kind=EventKind.ERROR,
                message="No audio source provided",
            )
            return

        source_source = source.get_source()
        if source_source is None:
            self._emit(
                event_queue,
                job=job,
                state=AudioProcessState.FAILED,
                kind=EventKind.ERROR,
                message=f"Failed to retrieve audio source: {source}",
            )
            return

        duration_seconds: float | None = None
        try:
            for progress in self._service.iter_transcribed_progress(
                source_source, job.uid
            ):
                duration_seconds = progress.duration_seconds
                if progress.segment is not None:
                    segment = progress.segment
                    artifact.append_text(segment.content)
                    artifact.append_segment(segment)
                    subtitle_writer.append(segment)
                    self._emit(
                        event_queue=event_queue,
                        job=job,
                        state=AudioProcessState.TRANSCRIBING,
                        kind=EventKind.SEGMENT,
                        message=(
                            f"handle_job::Segment::{segment.sequence}"
                            f"::{segment.content}::{segment.end}"
                        ),
                        progress=progress.fraction,
                        processed_seconds=progress.processed_seconds,
                        duration_seconds=progress.duration_seconds,
                    )
                else:
                    # VAD-filtered or silent chunk: no text, but the position
                    # in the recording still advances — report it so progress
                    # bars keep moving.
                    self._emit(
                        event_queue=event_queue,
                        job=job,
                        state=AudioProcessState.TRANSCRIBING,
                        kind=EventKind.PROGRESS,
                        message="Skipped silent chunk",
                        progress=progress.fraction,
                        processed_seconds=progress.processed_seconds,
                        duration_seconds=progress.duration_seconds,
                    )

        except Exception as exc:
            self.logger.exception(
                f"handle_job::exception during transcription for job {job.uid[:8]}::{exc}"
            )
            self._emit(
                event_queue=event_queue,
                job=job,
                state=AudioProcessState.FAILED,
                kind=EventKind.ERROR,
                message=f"Transcription failed: {exc}",
            )
            return

        artifact.finalize()

        self._emit(
            event_queue=event_queue,
            job=job,
            state=AudioProcessState.COMPLETED,
            kind=EventKind.STATUS,
            message=f"handle_job::Completed transcription for job {job.uid[:8]}",
            progress=1.0,
            processed_seconds=duration_seconds,
            duration_seconds=duration_seconds,
        )

    def run(
        self,
        jobs: Queue[TranscriptionJob | Sentinel],
        events: Queue[AudioProcessingEvent | Sentinel],
    ) -> None:
        while True:
            job = jobs.get()
            if isinstance(job, Sentinel):
                return
            self.handle_job(job, events)
            self._service.release_cuda_cache()


@final
class BackgroundWhisperProcessPool:
    __slots__ = (
        "_logger",
        "_model_name",
        "_ctx",
        "_process",
        "_job_queue",
        "_event_queue",
        "_running",
    )

    def __init__(self, model_name: str = "base") -> None:
        self._logger = getLogger("audio.worker")
        self._model_name = model_name
        self._ctx: SpawnContext = get_context("spawn")
        self._process: BaseProcess | None = None
        self._job_queue: Any = None
        self._event_queue: Any = None
        self._running = False

    @staticmethod
    async def _run(func: Any, *args: Any) -> Any:
        loop = get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    async def start(self) -> None:
        if self._running:
            return

        self._job_queue = self._ctx.Queue()
        self._event_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_entry,
            args=(self._model_name, self._job_queue, self._event_queue),
            daemon=True,
        )

        self._process.start()
        self._running = True
        self._logger.info(f"Background process {self._process.pid} started")

    async def stop(self, timeout: float = 10.0) -> None:
        if not self._running or self._process is None:
            return
        self._running = False
        await self._run(self._job_queue.put, _STOP)
        await self._run(self._process.join, timeout)
        if self._process.is_alive():
            self._process.terminate()
        self._logger.info("BackgroundWhisperProcessPool stopped.")

    async def submit(self, job: TranscriptionJob) -> None:
        await self._run(self._job_queue.put, job)

    async def next_event(self, poll_interval: float = 0.5) -> AudioProcessingEvent:
        """Await the next worker event, polling so the caller stays cancellable."""
        while True:
            try:
                return await self._run(self._event_queue.get, True, poll_interval)
            except Empty:
                continue

    async def snapshot(self) -> dict[str, object]:
        alive = self._process is not None and self._process.is_alive()
        return {"running": self._running, "alive": alive, "model": self._model_name}
