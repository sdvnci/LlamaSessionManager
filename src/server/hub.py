"""Fan-out of transcription events to session-bound WebSocket subscribers.

The worker subprocess never touches sockets — it only emits
``AudioProcessingEvent``s over its IPC queue.  The dispatcher (running in the
server's event loop) hands every event to ``TranscriptionHub.handle_event``,
which serialises it and enqueues a copy for each subscribed connection.

Slow subscribers shed the oldest buffered message first: progress events are
disposable, and the buffer is large enough that a healthy client always keeps
up with the terminal event.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

from ..audio import AudioProcessingEvent, TranscriptionJob
from ..messaging import SessionManager

MAX_QUEUE_DEPTH: int = 1024


class TranscriptionHub:
    """Session-keyed registry of WebSocket subscriber queues."""

    _subscribers: ClassVar[dict[str, set[asyncio.Queue[str]]]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def subscribe(cls, session_uid: str) -> asyncio.Queue[str]:
        """Register a subscriber queue for ``session_uid`` and return it."""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE_DEPTH)
        async with cls._lock:
            cls._subscribers.setdefault(session_uid, set()).add(queue)
        return queue

    @classmethod
    async def unsubscribe(cls, session_uid: str, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber queue, dropping the bucket when empty."""
        async with cls._lock:
            subscribers = cls._subscribers.get(session_uid)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                cls._subscribers.pop(session_uid, None)

    @classmethod
    async def publish(cls, session_uid: str, payload: dict[str, Any]) -> int:
        """Enqueue a JSON payload for every subscriber; return the fan-out count."""
        message = json.dumps(payload, ensure_ascii=False)
        async with cls._lock:
            subscribers = set(cls._subscribers.get(session_uid, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Shed the oldest buffered message; progress is disposable and
                # the newest state is always the most interesting.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(message)
        return len(subscribers)

    @classmethod
    async def handle_event(cls, event: AudioProcessingEvent) -> None:
        """Fan a worker event out to the owning session's subscribers."""
        progress = event.progress
        await cls.publish(
            event.session_uid,
            {
                "kind": event.kind.value,
                "session_uid": event.session_uid,
                "job_uid": event.job_uid,
                "state": event.state.value,
                "sequence": event.sequence,
                "message": event.message,
                "progress": progress,
                "percent": round(progress * 100, 1) if progress is not None else None,
                "processed_seconds": event.processed_seconds,
                "duration_seconds": event.duration_seconds,
            },
        )

    @classmethod
    async def snapshot(cls, session_uid: str) -> dict[str, Any]:
        """Current state of every live transcription job in a session.

        Sent on WebSocket connect so late joiners immediately see where each
        job stands instead of waiting for the next event.
        """
        jobs = await SessionManager.list_transcriptions_for_session(session_uid)
        return {
            "kind": "snapshot",
            "session_uid": session_uid,
            "jobs": [cls._job_snapshot(job) for job in jobs],
        }

    @staticmethod
    def _job_snapshot(job: TranscriptionJob) -> dict[str, Any]:
        progress: float | None = None
        if job.processed_seconds is not None and job.duration_seconds:
            progress = min(job.processed_seconds / job.duration_seconds, 1.0)
        source = job.source
        return {
            "job_uid": job.uid,
            "state": job.state.value,
            "source_name": source.source_name if source else None,
            "progress": progress,
            "percent": round(progress * 100, 1) if progress is not None else None,
            "processed_seconds": job.processed_seconds,
            "duration_seconds": job.duration_seconds,
        }


__all__ = ("TranscriptionHub",)
