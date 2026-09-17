from asyncio import CancelledError, Lock, Queue, Task, create_task
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from logging import Logger, getLogger
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    ClassVar,
    Final,
    Literal,
    Sequence,
    Union,
)
from uuid import uuid4

from ollama import Message, Tool

from ..audio import (
    AudioMimeTypeEnum,
    AudioProcessingEvent,
    AudioProcessState,
    AudioSourceType,
    BackgroundWhisperProcessPool,
    SubtitleFormat,
    TranscriptionAudioSource,
    TranscriptionJob,
)
from ..common import EXPORTDIR, QueuedMessage
from ..common.settings import (
    transcription_disabled_message,
    transcription_enabled,
)
from ._messaging import MessageSession
from ._types import SessionConfig

# Threshold policy (placeholders, tune per deployment).
IMMEDIATE_MAX_DURATION_SECONDS: Final[float] = 300.0
IMMEDIATE_MAX_INPUT_BYTES: Final[int] = 25 * 1024 * 1024
MAX_INLINE_TRANSCRIPT_CHARS: Final[int] = 8000
TRANSCRIPT_QUEUE: Final[str] = "transcripts"
LOGGER: Final[Logger] = getLogger("messaging.sessions")


class SessionManager:
    """
    Stateless class-level registry of ``MessageSession`` objects and named queues.

    All interaction is via classmethods; never instantiate this class directly.

    Pool operations
    ------------------
    ``create_session``    Register a new session in the pool.
    ``get_session``       Look up a session by UID.
    ``close_session``     Decrement ref-count; evict when it reaches zero.
    ``evict_session``     Force-remove and close a session immediately.
    ``scoped_session``    Context manager: create → yield → save → evict.

    Queue operations
    ------------------
    ``create_queue``      Create (or return existing) named queue.
    ``get_queue``         Look up a queue by name.
    ``drop_queue``        Remove a named queue.
    ``produce``           Wrap content in a ``QueuedMessage`` and enqueue it.
    ``consume``           Dequeue the next ``QueuedMessage``.
    ``queue_size``        Non-blocking peek at queue depth.
    """

    __slots__ = ()

    _pool: ClassVar[dict[str, MessageSession]] = {}
    _pool_lock: ClassVar[Lock] = Lock()

    _queues: ClassVar[dict[str, Queue[QueuedMessage]]] = {}
    _queue_lock: ClassVar[Lock] = Lock()

    _transcriptions: ClassVar[dict[str, TranscriptionJob]] = {}
    _transcription_lock: ClassVar[Lock] = Lock()
    _background_pool: ClassVar["BackgroundWhisperProcessPool | None"] = None
    _dispatcher_task: ClassVar[Task[None] | None] = None
    _event_consumers: ClassVar[
        list[Callable[[AudioProcessingEvent], Awaitable[None]]]
    ] = []

    @classmethod
    async def create_session(
        cls: type["SessionManager"],
        config: SessionConfig,
        system_prompts: Sequence[Message] | None = None,
        tools: Sequence[Union[Tool, Callable[..., Any]]] | None = None,
    ) -> MessageSession:
        """
        Construct a ``MessageSession``, call ``start()``, register it in the
        pool, and return it.

        The caller is responsible for calling ``close_session`` (or using
        ``scoped_session``) to release the reference when done.
        """

        sess = MessageSession(config=config, system_prompts=system_prompts, tools=tools)
        await sess.start()
        async with cls._pool_lock:
            cls._pool[sess.uid] = sess
            LOGGER.info(f"create_session:::{sess.uid[:8]} registered.")
        return sess

    @classmethod
    async def get_session(
        cls: type["SessionManager"], uid: str
    ) -> MessageSession | None:
        """Return the session registered under ``uid``, or ``None``."""
        async with cls._pool_lock:
            return cls._pool.get(uid)

    @classmethod
    async def close_session(cls: type["SessionManager"], uid: str) -> None:
        """
        Decrement the session's reference count.
        Evicts it from the pool when the count reaches zero.
        """
        async with cls._pool_lock:
            sess = cls._pool.get(uid)
            if sess is None:
                LOGGER.warning(f"close_session:::No session {uid[:8]} in pool.")
                return
            await sess.close()
            if sess.ref_count == 0:
                del cls._pool[uid]
                LOGGER.info(f"close_session:::{uid[:8]} evicted (ref_count=0).")

    @classmethod
    async def evict_session(
        cls: type["SessionManager"], uid: str, save: bool = False
    ) -> None:
        """
        Forcibly remove a session from the pool and close it immediately,
        regardless of its reference count.

        Pass ``save=True`` to persist the session's history before eviction.
        """
        async with cls._pool_lock:
            sess = cls._pool.pop(uid, None)
            if sess is None:
                LOGGER.warning(f"evict_session:::No session {uid[:8]} in pool.")
                return
        if save:
            await sess.save()
        await sess.close(force=True)
        LOGGER.info(f"evict_session:::{uid[:8]} force-evicted.")

    @classmethod
    @asynccontextmanager
    async def scoped_session(
        cls: type["SessionManager"],
        config: SessionConfig,
        system_prompts: Sequence[Message] | None = None,
        tools: Sequence[Union[Tool, Callable[..., Any]]] | None = None,
        save_on_exit: bool = True,
        export_dir: Path = EXPORTDIR,
    ) -> AsyncGenerator[MessageSession, None]:
        """
        Async context manager: create a session, register it, yield it, then
        save (optionally) and evict on exit — even on exception.

        This is the recommended entry point for all session usage.

        Usage::

            async with SessionManager.scoped_session(config=cfg) as sess:
                msg = sess.parse_content("Analyse this image", Path("img.png"))
                await sess.add_message(msg)
                response = await sess.send()
                await sess.add_message(response.message)
        """
        sess = await cls.create_session(config, system_prompts, tools)
        try:
            yield sess
        finally:
            if save_on_exit:
                await sess.save(export_dir=export_dir)
            await cls.evict_session(sess.uid, save=False)

    @classmethod
    async def create_queue(
        cls: type["SessionManager"], name: str, maxsize: int = 0
    ) -> Queue[QueuedMessage]:
        """
        Create a named queue with an optional capacity cap.

        ``maxsize=0`` (default) means unbounded.
        If a queue with this name already exists it is returned unchanged.
        """
        async with cls._queue_lock:
            if name not in cls._queues:
                cls._queues[name] = Queue(maxsize=maxsize)
                LOGGER.info(f"create_queue:::'{name}' created (maxsize={maxsize}).")
            return cls._queues[name]

    @classmethod
    async def get_queue(
        cls: type["SessionManager"], name: str
    ) -> "Queue[QueuedMessage] | None":
        """Return the queue registered under ``name``, or ``None``."""
        async with cls._queue_lock:
            return cls._queues.get(name)

    @classmethod
    async def drop_queue(cls: type["SessionManager"], name: str) -> None:
        """Remove a named queue.  Any messages still in it are discarded."""
        async with cls._queue_lock:
            cls._queues.pop(name, None)
            LOGGER.info(f"drop_queue:::'{name}' dropped.")

    @classmethod
    async def produce(
        cls: type["SessionManager"],
        source_session_uid: str,
        content: str,
        queue_name: str,
        role: Literal["user", "assistant", "system", "tool"] = "assistant",
        image_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Wrap ``content`` in a ``QueuedMessage`` envelope and push it onto
        a named queue.

        Typically called right after ``session.send()`` to forward the model's
        response downstream::

            response = await sess.send()
            await SessionManager.produce(
                sess.uid, response.message.content, "pipeline"
            )

        Returns the ``id`` of the enqueued message.
        """
        queue = await cls.get_queue(queue_name)
        if queue is None:
            raise KeyError(
                f"No queue named '{queue_name}'. "
                "Call SessionManager.create_queue() first."
            )

        msg_id = str(uuid4())
        envelope: QueuedMessage = {
            "id": msg_id,
            "source_session_uid": source_session_uid,
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if image_paths:
            envelope["image_paths"] = image_paths
        if metadata:
            envelope["metadata"] = metadata

        await queue.put(envelope)
        LOGGER.debug(
            f"produce:::{msg_id[:8]} enqueued on '{queue_name}' "
            f"(qsize={queue.qsize()})."
        )
        return msg_id

    @classmethod
    async def consume(
        cls: type["SessionManager"],
        queue_name: str,
        block: bool = True,
    ) -> QueuedMessage | None:
        """
        Pop the next message off a named queue.

        - ``block=True`` (default): waits until a message is available.
        - ``block=False``: returns ``None`` immediately if the queue is empty.

        Typical consumer loop::

            while True:
                envelope = await SessionManager.consume("pipeline")
                if envelope is None:
                    break
                async with SessionManager.scoped_session(config=consumer_cfg) as sess:
                    msg = sess.parse_content(envelope["content"])
                    await sess.add_message(msg)
                    result = await sess.send()
        """
        queue = await cls.get_queue(queue_name)
        if queue is None:
            raise KeyError(f"No queue named '{queue_name}'.")

        if not block and queue.empty():
            return None

        envelope = await queue.get()
        LOGGER.debug(
            f"consume:::{envelope['id'][:8]} dequeued from '{queue_name}' "
            f"(qsize={queue.qsize()})."
        )
        return envelope

    @classmethod
    def queue_size(cls: type["SessionManager"], name: str) -> int:
        """
        Return the current depth of a named queue without blocking.
        Returns 0 if the queue does not exist.
        """
        q = cls._queues.get(name)
        return q.qsize() if q is not None else 0

    @classmethod
    async def pool_snapshot(
        cls: type["SessionManager"],
    ) -> dict[str, dict[str, Any]]:
        """
        Return a lightweight snapshot of all live sessions, keyed by UID.
        Useful for monitoring and debugging.
        """
        async with cls._pool_lock:
            return {
                uid: {
                    "model": sess.model,
                    "name": sess.config.name,
                    "ref_count": sess.ref_count,
                }
                for uid, sess in cls._pool.items()
            }

    @classmethod
    async def start_background_pool(
        cls, model_name: str = "base"
    ) -> BackgroundWhisperProcessPool:
        """Start the background transcription worker and its event dispatcher.

        Spawning the worker loads the Whisper model into memory, so this is
        refused outright when transcription is disabled (see
        :func:`~src.common.settings.transcription_enabled`) to avoid the
        pre-allocation cost when it can never be used.
        """
        if not transcription_enabled():
            raise RuntimeError(transcription_disabled_message())
        if cls._background_pool is not None:
            return cls._background_pool

        pool = BackgroundWhisperProcessPool(model_name)
        await pool.start()
        cls._background_pool = pool
        cls._dispatcher_task = create_task(cls._dispatch_events())
        LOGGER.info("start_background_pool:::worker '%s' ready.", model_name)
        return pool

    @classmethod
    async def stop_background_pool(cls) -> None:
        """Stop the dispatcher and the background worker process."""
        pool = cls._background_pool
        if pool is None:
            return
        cls._background_pool = None
        if cls._dispatcher_task is not None:
            cls._dispatcher_task.cancel()
            with suppress(CancelledError):
                await cls._dispatcher_task
            cls._dispatcher_task = None
        await pool.stop()
        LOGGER.info("stop_background_pool:::done.")

    @classmethod
    async def submit_transcription(
        cls,
        session_uid: str,
        source: AudioSourceType,
        source_name: str | None = None,
        source_mime_type: AudioMimeTypeEnum = AudioMimeTypeEnum.RAW,
        immediate: bool | None = None,
        model_name: str = "base",
        subtitle_format: SubtitleFormat = "vtt",
    ) -> TranscriptionJob:
        """Register and submit a transcription job, returning its live record.

        ``source`` is a path or in-memory bytes. When ``immediate`` is ``None``
        the size and duration thresholds decide inline vs background handling.
        ``subtitle_format`` selects the subtitle container (``"vtt"`` or
        ``"srt"``) written alongside the transcript.
        """
        pool = await cls.start_background_pool(model_name)
        if pool is None:
            raise RuntimeError(
                "submit_transcription::Call SessionManager.start_background_pool() first."
            )

        job = cls._build_job(
            session_uid,
            source,
            source_name,
            source_mime_type,
            immediate,
            subtitle_format,
        )
        # Auto-determine immediate when caller does not specify.
        if immediate is None:
            job.immediate = cls._is_immediate_job(job)
        async with cls._transcription_lock:
            cls._transcriptions[job.uid] = job
            job.state = AudioProcessState.QUEUED
        await pool.submit(job)
        LOGGER.info(
            f"submit_transcription::Queued job {job.uid[:8]} for session {session_uid[:8]}"
        )
        return job

    @staticmethod
    def _is_immediate_job(job: TranscriptionJob) -> bool:
        """Decide inline vs background using duration then file size."""
        if job.duration_seconds is not None:
            return job.duration_seconds <= IMMEDIATE_MAX_DURATION_SECONDS
        if job.source is not None and job.source.source_size_bytes is not None:
            return job.source.source_size_bytes <= IMMEDIATE_MAX_INPUT_BYTES
        return False

    @staticmethod
    def _build_job(
        session_uid: str,
        source: AudioSourceType,
        source_name: str | None = None,
        source_mime_type: AudioMimeTypeEnum = AudioMimeTypeEnum.RAW,
        immediate: bool | None = None,
        subtitle_format: SubtitleFormat = "vtt",
    ) -> TranscriptionJob:

        source_path: Path | None = None
        source_bytes: bytes | None = None
        source_size_bytes: int | None = None

        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(
                    f"_build_job::source file not found: {source_path}"
                )
            source_name = source_name or source_path.name
            source_size_bytes = source_path.stat().st_size
        else:
            source_bytes = bytes(source)  # type: ignore
            source_size_bytes = len(source_bytes)

        tsource = TranscriptionAudioSource(
            source_path=source_path,
            source_bytes=source_bytes,
            source_size_bytes=source_size_bytes,
            source_mime_type=source_mime_type,
            source_name=source_name,
        )

        record = TranscriptionJob(
            session_uid=session_uid,
            state=AudioProcessState.CREATED,
            source=tsource,
            subtitle_format=subtitle_format,
            immediate=immediate if immediate is not None else False,
        )

        return record

    @classmethod
    async def get_transcription(cls, job_id: str) -> TranscriptionJob | None:
        """Return the live record for ``job_id`` (gone once terminally handled)."""
        async with cls._transcription_lock:
            return cls._transcriptions.get(job_id)

    @classmethod
    async def list_transcriptions_for_session(
        cls, session_uid: str
    ) -> list[TranscriptionJob]:
        async with cls._transcription_lock:
            return [
                r for r in cls._transcriptions.values() if r.session_uid == session_uid
            ]

    @classmethod
    async def _dispatch_events(cls) -> None:
        """Drain worker events into ``handle_transcription_event`` until stopped."""
        pool = cls._background_pool
        assert pool is not None, "_dispatch_events started without a pool"
        while cls._background_pool is pool:
            event = await pool.next_event()  # type: ignore[union-attr]
            async with cls._transcription_lock:
                if event.job_uid not in cls._transcriptions:
                    LOGGER.debug(
                        "_dispatch_events:::job %s already evicted; skipping event %s",
                        event.job_uid[:8],
                        event.kind.value,
                    )
                    continue
            await cls.handle_transcription_event(event)

    @classmethod
    def register_event_consumer(
        cls: type["SessionManager"],
        consumer: Callable[[AudioProcessingEvent], Awaitable[None]],
    ) -> None:
        """Register an async callback invoked for every transcription event.

        Used by the server layer to fan worker events out to session-bound
        WebSocket subscribers.  Registration is idempotent; consumers run in
        registration order, outside the transcription lock.
        """
        if consumer not in cls._event_consumers:
            cls._event_consumers.append(consumer)

    @classmethod
    async def handle_transcription_event(cls, event: AudioProcessingEvent) -> None:
        """Apply a worker event to its record, then fan it out to consumers.

        Record updates (including ingest/evict on terminal events) happen
        under the transcription lock; consumers run outside it so a slow
        subscriber can never stall job-state bookkeeping.
        """
        async with cls._transcription_lock:
            record = cls._transcriptions.get(event.job_uid)
            if record is None:
                return
            # Update live state so callers polling get_transcription() see progress.
            record.state = event.state
            if event.processed_seconds is not None:
                record.processed_seconds = event.processed_seconds
            if event.duration_seconds is not None:
                record.duration_seconds = event.duration_seconds
            if event.state.is_terminal:
                await cls.ingest_transcription_into_session(record)
                cls._transcriptions.pop(event.job_uid, None)

        for consumer in tuple(cls._event_consumers):
            with suppress(Exception):
                await consumer(event)

    @classmethod
    async def ingest_transcription_into_session(cls, record: TranscriptionJob) -> None:
        """Deliver a compact completion note to the session, live or queued.

        Never appends the raw transcript to history; only a reference note.
        """
        sess = await cls.get_session(record.session_uid)
        if sess is not None:
            await sess.add_message(MessageSession.compact_transcription(record))
            LOGGER.info(
                f"ingest_transcription_into_session:::{record.uid[:8]} appended to live {record.session_uid[:8]}.",
            )
            return

        await cls.create_queue(TRANSCRIPT_QUEUE)
        await cls.produce(
            record.session_uid,
            MessageSession.compact_transcription(record).content or "",
            TRANSCRIPT_QUEUE,
            role="system",
            metadata=cls.transcript_metadata(record),
        )
        LOGGER.info(
            f"ingest_transcription_into_session:::{record.uid[:8]} queued for offline {record.session_uid[:8]}.",
        )

    @staticmethod
    def transcript_metadata(record: TranscriptionJob) -> dict[str, Any]:
        """Build a metadata dict for a completed transcript.

        The ``inline_eligible`` flag uses the transcript's character count
        (not raw byte size) against ``MAX_INLINE_TRANSCRIPT_CHARS``.
        """
        transcript_path = record.transcript_path
        transcript_len = (
            len(transcript_path.read_text(encoding="utf-8"))
            if transcript_path.exists()
            else 0
        )
        return {
            "artifact_kind": "transcript",
            "job_id": record.uid,
            "transcript_path": transcript_path,
            "segments_path": record.segments_path,
            "subtitle_path": record.subtitle_path,
            "subtitle_format": record.subtitle_format,
            "summary_path": record.summary_path,
            "source_name": record.source.source_name if record.source else None,
            "inline_eligible": transcript_len <= MAX_INLINE_TRANSCRIPT_CHARS,
        }
