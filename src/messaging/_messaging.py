from asyncio import Lock
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from inspect import iscoroutinefunction
from json import dumps
from logging import DEBUG, ERROR, INFO, WARNING, Logger, getLogger
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Literal,
    Mapping,
    Sequence,
    get_args,
    overload,
)
from uuid import uuid4

from ollama import AsyncClient, ChatResponse, Image, Message, Tool
from pydantic import BaseModel, ConfigDict

from ..audio import TranscriptionJob
from ..common import EXPORTDIR, ensure_directory
from ..imaging._types import ImageExtensions
from ..textual._types import TextualMimeTypeEnum
from ._types import SessionConfig, SessionExport


class ParsedResponse(BaseModel):
    """
    Structured result of a single assistant turn.

    Attributes:
        content:
            Final text content from the assistant.  Empty string if the model
            only issued tool calls with no prose.
        thinking:
            Chain-of-thought produced by a thinking model (``None`` when thinking
            is disabled or the model does not support it).
        tool_calls:
            All tool calls requested by the model in this turn.  Empty list if
            none were made.  Dispatch these with ``send_with_tools()``.
        message:
            The fully-assembled ``Message`` ready to append to history::

                parsed = await collect_stream(await sess.send(stream=True))
                await sess.add_message(parsed.message)

        done_reason:
            Why generation stopped (``"stop"``, ``"tool_calls"``, ``"length"``,
            etc.).  ``None`` for individual stream chunks.
    """

    model_config = ConfigDict(slots=True)  # type: ignore
    content: str
    thinking: str | None
    tool_calls: list[Message.ToolCall]
    message: Message
    done_reason: str | None

    @staticmethod
    def from_response(response: ChatResponse) -> "ParsedResponse":
        """
        Wrap a non-streaming ``ChatResponse`` in a ``ParsedResponse``.

        The returned ``message`` is the raw response message — append it to
        history directly::

            response = await sess.send(stream=False)
            parsed   = parse_response(response)
            print(parsed.thinking)       # chain-of-thought, if any
            print(parsed.content)        # final answer
            await sess.add_message(parsed.message)
        """
        msg = response.message
        return ParsedResponse(
            content=msg.content or "",
            thinking=msg.thinking,
            tool_calls=list(msg.tool_calls or []),
            message=msg,
            done_reason=response.done_reason,
        )

    @staticmethod
    async def from_stream(stream: AsyncIterator[ChatResponse]) -> "ParsedResponse":
        """
        Drain a streaming ``ChatResponse`` iterator and assemble a
        ``ParsedResponse``.

        Accumulates content and thinking chunks into complete strings, collects
        any tool calls from the final chunk, and builds a single ``Message``
        suitable for appending to history::

            stream = await sess.send(stream=True)
            parsed = await collect_stream(stream)

            # stream to the terminal while collecting
            # (if you need live output, iterate manually and pass the assembled
            #  Message to add_message yourself — see note below)
            print(parsed.content)
            await sess.add_message(parsed.message)

        Note
        ----
        If you want to *display* chunks while they arrive **and** keep history,
        iterate the stream yourself and accumulate manually::

            full_content  = ""
            full_thinking = ""
            async for chunk in await sess.send(stream=True):
                delta = chunk.message.content or ""
                full_content += delta
                print(delta, end="", flush=True)
            await sess.add_message(Message(role="assistant", content=full_content))
        """
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[Message.ToolCall] = []
        done_reason: str | None = None

        async for chunk in stream:
            if chunk.message.content:
                content_parts.append(chunk.message.content)
            if chunk.message.thinking:
                thinking_parts.append(chunk.message.thinking)
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)
            if chunk.done:
                done_reason = chunk.done_reason

        content = "".join(content_parts)
        thinking = "".join(thinking_parts) or None

        assembled = Message(
            role="assistant",
            content=content,
            thinking=thinking,
            tool_calls=tool_calls or None,
        )

        return ParsedResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            message=assembled,
            done_reason=done_reason,
        )


class MessageSession:
    """
    Async chat session wrapping a single ``ollama.AsyncClient``.

    Stores conversation history separately from system prompts; system prompts
    are prepended at ``send()``-time so they are never part of the history you
    iterate over, save, or queue.

    Attributes are only accessible via properties; no public mutators.  All
    mutation goes through ``async`` methods that hold ``_lock`` while writing.
    """

    __slots__ = (
        "_config",
        "_tools",
        "_client",
        "_uid",
        "_history",
        "_system_prompts",
        "_ref_count",
        "_lock",
        "_logger",
    )

    def __init__(
        self: "MessageSession",
        config: SessionConfig,
        system_prompts: Sequence[Message] | None = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
    ) -> None:
        self._uid: str = str(uuid4())
        self._config: SessionConfig = config
        self._client: AsyncClient = AsyncClient(
            host=config.host,
            timeout=config.timeout,
        )
        self._history: list[Message] = []
        self._system_prompts: list[Message] = list(system_prompts or [])
        self._tools: list[Tool | Callable[..., Any]] = list(tools or [])
        self._ref_count: int = 0
        self._lock: Lock = Lock()
        self._logger: Logger = getLogger(f"MessageSession_{self._uid}")

    def _log(self, message: str, level: int = INFO) -> None:
        self._logger.log(level, f"{message}")

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def config(self) -> SessionConfig:
        return self._config

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def ref_count(self) -> int:
        return self._ref_count

    @property
    def in_use(self) -> bool:
        return self._ref_count >= 1

    async def export(self) -> SessionExport:
        async with self._lock:
            cfg = self._config
            return SessionExport(
                uid=self.uid,
                timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S"),
                name=cfg.name,
                model=cfg.model,
                description=cfg.description,
                system=cfg.system,
                messages=[m.model_dump(exclude_none=True) for m in self._history],
            )

    async def save(self, export_dir: Path = EXPORTDIR) -> Path:
        """
        Serialise and write this session to a JSON file under ``export_dir``.
        Returns the path that was written.
        """
        if not ensure_directory(export_dir):
            raise RuntimeError("Could not ensure export directory's existence")
        data = await self.export()
        out = export_dir / f"session_{self._uid}_{data['timestamp']}.json"
        out.write_text(dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        self._log(f"save → {out}", DEBUG)
        return out

    async def start(self) -> None:
        """Increment the reference count, marking this session as in-use."""
        async with self._lock:
            self._ref_count += 1
            self._log(f"START (ref_count={self._ref_count})", DEBUG)

    async def close(self, force: bool = False) -> None:
        """
        Decrement the reference count.  When it reaches zero (or ``force=True``),
        close the underlying ``AsyncClient`` and release connections.
        """
        async with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            if self._ref_count == 0 or force:
                self._ref_count = 0
                await self._client.close()
                self._log("closed.", DEBUG)

    async def history(self) -> tuple[Message, ...]:
        """Return a shallow copy of the conversation history (no system prompts)."""
        async with self._lock:
            return tuple(m.model_copy(deep=True) for m in self._history)

    async def add_message(self, message: Message) -> None:
        async with self._lock:
            self._history.append(message)
            self._log(f"add_message role={message.role!r}", DEBUG)

    async def add_messages(self, messages: Sequence[Message]) -> None:
        async with self._lock:
            self._history.extend(messages)
            self._log(f"add_messages count={len(messages)}", DEBUG)

    async def prepend_message(self, message: Message) -> None:
        """Insert a message at the start of history (after system prompts)."""
        async with self._lock:
            self._history.insert(0, message)
            self._log(f"prepend_message role={message.role!r}", DEBUG)

    async def get_message(self, index: int) -> Message:
        async with self._lock:
            if index < 0 or index >= len(self._history):
                raise IndexError(
                    f"Message index {index} out of range "
                    f"(history length {len(self._history)})."
                )
            return self._history[index]

    async def clear_history(self) -> None:
        """Wipe conversation history; system prompts are preserved."""
        async with self._lock:
            self._history.clear()
            self._log("clear_history: history cleared.", DEBUG)

    def _full_messages(self) -> list[Message]:
        """
        Concatenate system prompts + history into the final list sent to Ollama.
        Called inside ``send()``; does NOT acquire the lock (caller must hold it
        or guarantee no concurrent writes).
        """
        return [*self._system_prompts, *self._history]

    def parse_content(
        self,
        content: str | Path | Sequence[str | Path],
        role: Literal["user", "assistant"] = "user",
    ) -> Message:
        """
        Parse a `Message` from raw content; strings, file paths or a mix of.
        Supported:
            - ``str``          → inline text content
            - ``Path`` to image (.png / .jpg / .jpeg / .gif / .bmp / .webp)
                               → passed directly as Ollama ``images``
                                 (Ollama handles base-64 encoding internally)
            - ``Path`` to text file (.txt / .md / .csv / .json / …)
                               → content read and injected as text
            - ``Path`` to ``.pdf``
                               → text extracted via ``pypdf`` (optional dep)
            - ``Path`` to ``.docx``
                               → text extracted via ``python-docx`` (optional dep)
            - ``list[str | Path]``
                               → any combination of the above; text parts are
                                 joined with blank lines, images are collected
                                 into the ``images`` field

        Examples:
            ```python
            # Plain text
            msg = session.parse_content("Summarise the following report.")

            # Single image
            msg = session.parse_content(Path("chart.png"))

            # Mixed: text prompt + image + document
            msg = session.parse_content([
                "What trends do you see?",
                Path("sales_chart.png"),
                Path("annual_report.pdf"),
            ])
            ```
        """

        items: list[str | Path]
        match content:
            case str() | Path():
                items = [content]
            case Iterable():
                items = list(content)
            case _:
                raise TypeError("Unsupported content type")

        text_segments: list[str] = []
        image_paths: list[Image] = []

        for item in items:
            if isinstance(item, str):
                text_segments.append(item)
                continue

            suffix = item.suffix.lower().lstrip(".")
            if suffix in get_args(ImageExtensions):
                image_paths.append(Image(value=item))
                continue

            if not TextualMimeTypeEnum.is_valid(suffix):
                # Unknown type, try to read as UTF-8 text, warn on failure.
                reader = TextualMimeTypeEnum.PLAIN.reader
                assert (
                    reader is not None
                ), "parse_content: The default TextualMimeTypeEnum plaintext file reader is None. This should not be!"
                textcontent, textexc = reader(item)
                if textexc is not None:
                    self._log(
                        f"parse_content: Error reading file content of {item} |-> {textexc}",
                        WARNING,
                    )

                    continue

                text_segments.append(textcontent)
                continue

            textmime = TextualMimeTypeEnum.from_extension(suffix)
            reader = textmime.reader
            if reader is None:
                # The item is a text file but one we do not support
                # We should log this and move on
                self._log(
                    f"parse_content: Could not read {item} file with mime {textmime.value}",
                    WARNING,
                )

                continue

            textcontent, textexc = reader(item)
            if textexc is not None:
                self._log(
                    f"parse_content: Error reading file content of {item} |-> {textexc}",
                    WARNING,
                )
                continue

            text_segments.append(textcontent)

        return Message(
            role=role, content="\n\n".join(text_segments), images=image_paths or None
        )

    @staticmethod
    def compact_transcription(record: TranscriptionJob) -> Message:
        """Build a compact system-facing notification that a transcript is ready.

        The returned ``Message`` uses ``role="system"`` so the model treats it as
        a factual notice rather than user input.  It contains the transcript
        reference (path, duration, summary path) serialised as JSON but never
        the raw transcript body.
        """
        ref = record.reference
        refdict = ref.model_dump(mode="json")
        msgcontent = dumps(refdict, indent=2)
        return Message(role="system", content=msgcontent)

    @staticmethod
    def build_transcript_context_message(
        transcript_path: Path,
        instruction: str,
    ) -> Message:
        """Build a user message that injects a transcript alongside an instruction.

        Reads the transcript file from disk and packages it with the caller's
        instruction so the model can summarise, extract action items, translate,
        or otherwise transform the spoken content.

        The message is suitable for ``add_message`` followed by ``send()``::

            msg = sess.build_transcript_context_message(
                Path("artifacts/.../transcript.txt"),
                "Summarise the key decisions from this meeting.",
            )
            await sess.add_message(msg)
            response = await sess.send()
        """
        transcript_text = transcript_path.read_text(encoding="utf-8")
        content = (
            f"The following is a transcript of an audio recording:\n\n"
            f"--- TRANSCRIPT START ---\n"
            f"{transcript_text}\n"
            f"--- TRANSCRIPT END ---\n\n"
            f"{instruction}"
        )
        return Message(role="user", content=content)

    @overload
    async def send(
        self,
        stream: Literal[True],
        format: str | Mapping[str, Any] | None = None,
        think: bool | Literal["low", "medium", "high"] | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse: ...

    @overload
    async def send(
        self,
        stream: Literal[False] = False,
        format: str | Mapping[str, Any] | None = None,
        think: bool | Literal["low", "medium", "high"] | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse: ...

    async def send(
        self,
        stream: bool = False,
        format: str | Mapping[str, Any] | None = None,
        think: bool | Literal["low", "medium", "high"] | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse | AsyncIterator[ChatResponse]:
        """
        Send the current history (with system prompts prepended) to Ollama.

        The assistant's response is **not** appended to history automatically —
        the caller decides what to keep.  Typical pattern::

            user_msg = session.parse_content("Explain this chart", Path("chart.png"))
            await session.add_message(user_msg)

            response = await session.send()
            await session.add_message(response.message)   # keep in history

        For streaming::

            user_msg = session.parse_content("Tell me a story.")
            await session.add_message(user_msg)

            full = ""
            async for chunk in await session.send(stream=True):
                full += chunk.message.content or ""

            await session.add_message(Message(role="assistant", content=full))

        ``format`` accepts a JSON schema dict (structured output), ``"json"``
        (unstructured JSON mode), or ``None`` (default free-form text).
        """
        full_messages = self._full_messages()
        options = self._config.to_options() or None

        kwargdict: dict[str, Any] = dict(
            model=self._config.model,
            messages=full_messages,
            stream=stream,
            tools=self._tools or None,
            options=options,
            keep_alive=keep_alive,
        )
        if format is not None:
            kwargdict["format"] = format
        if think is not None:
            kwargdict["think"] = think

        self._log(
            f"send stream={stream} model={self._config.model!r} "
            f"messages={len(full_messages)}",
            DEBUG,
        )
        return await self._client.chat(**kwargdict)

    async def send_with_tools(
        self,
        tool_registry: dict[str, Callable[..., Any]],
        max_rounds: int = 10,
        format: "str | Mapping[str, Any] | None" = None,
        think: "bool | Literal['low', 'medium', 'high'] | None" = None,
        keep_alive: "float | str | None" = None,
    ) -> ParsedResponse:
        """
            Agentic tool-call loop: send → dispatch tool calls → send again until
            the model stops requesting tools or ``max_rounds`` is reached.

            Each intermediate round's assistant message and tool-result messages
            are appended to history automatically so the model has full context on
            every subsequent turn.  The **final** assistant response is *not*
            appended — the caller controls that, consistent with ``send()``.

            Tool functions may be sync or async.  Return values are coerced to
            ``str`` before being sent back as tool-result messages.  Exceptions are
            caught, logged, and forwarded to the model as error strings so it can
            attempt to recover gracefully.

        Args:
            tool_registry:
                ``{function_name: callable}`` mapping.  The names must match
                exactly what the model sends in ``tool_calls[].function.name``.
            max_rounds:
                Safety cap on the dispatch loop.  Raises no error on exhaustion —
                instead sends one final time and returns whatever the model says.

            Usage—sync tools::

                def get_weather(city: str) -> str:
                    return f"Sunny, 22°C in {city}"

                parsed = await sess.send_with_tools({"get_weather": get_weather})
                await sess.add_message(parsed.message)
                print(parsed.content)

            Usage—async tools::

                async def search_db(query: str) -> str:
                    rows = await db.fetch(query)
                    return str(rows)

                parsed = await sess.send_with_tools({"search_db": search_db})
        """
        for round_num in range(max_rounds):
            response = await self.send(
                stream=False,
                format=format,
                think=think,
                keep_alive=keep_alive,
            )
            parsed = ParsedResponse.from_response(response)

            if not parsed.tool_calls:
                # Model is done — no more tool calls requested.
                return parsed

            self._log(
                f"send_with_tools round={round_num + 1} "
                f"dispatching {len(parsed.tool_calls)} call(s): "
                + ", ".join(tc.function.name for tc in parsed.tool_calls),
                DEBUG,
            )

            # Append the assistant turn (carries the tool_calls field so the
            # model remembers what it asked for on the next round).
            await self.add_message(parsed.message)

            # Dispatch each call and feed results back as tool-role messages.
            for tc in parsed.tool_calls:
                fn_name = tc.function.name
                fn_args = dict(tc.function.arguments)
                fn = tool_registry.get(fn_name)

                if fn is None:
                    result = f"Error: tool '{fn_name}' is not registered."
                    self._log(f"send_with_tools: unknown tool '{fn_name}'", WARNING)
                else:
                    try:
                        raw = (
                            await fn(**fn_args)
                            if iscoroutinefunction(fn)
                            else fn(**fn_args)
                        )
                        result = str(raw)
                    except Exception as exc:
                        result = f"Error calling '{fn_name}': {exc}"
                        self._log(
                            f"send_with_tools: '{fn_name}' raised {exc!r}",
                            ERROR,
                        )

                await self.add_message(
                    Message(role="tool", content=result, tool_name=fn_name)
                )

        # Safety valve: max_rounds exhausted without a clean stop.
        self._log(
            f"send_with_tools: max_rounds={max_rounds} exhausted, "
            "returning final send.",
            WARNING,
        )
        return ParsedResponse.from_response(await self.send(stream=False))

    @classmethod
    @asynccontextmanager
    async def session(
        cls,
        config: SessionConfig,
        system_prompts: Sequence[Message] | None = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        save_on_exit: bool = True,
    ) -> AsyncGenerator["MessageSession", None]:
        """
        Async context manager for a single, self-contained session.

        Starts the session on entry, optionally saves on exit, always closes.
        For multi-session workloads use ``SessionManager.scoped_session``
        instead so the session is registered in the pool.

        Usage::

            async with MessageSession.session(config=cfg) as sess:
                await sess.add_message(sess.parse_content("Hello!"))
                response = await sess.send()
                print(response.message.content)
        """
        sess = cls(config=config, system_prompts=system_prompts, tools=tools)
        await sess.start()
        try:
            yield sess
        finally:
            if save_on_exit:
                await sess.save()
            await sess.close()
