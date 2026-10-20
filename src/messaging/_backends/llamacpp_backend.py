"""llama.cpp backend - talks to ``llama-server``'s OpenAI-compatible API.

llama-server (the server binary shipped with `llama.cpp
<https://github.com/ggml-org/llama.cpp>`_) exposes an OpenAI-style
``/v1/chat/completions`` endpoint.  This backend translates the messaging
layer's Ollama-shaped request/response objects onto that endpoint so the
rest of the codebase keeps working unchanged.

Supported translations
----------------------
- ``messages`` (text, images, thinking, tool calls/results)
- ``tools`` (``Tool`` objects and Python callables via Google-style docstrings)
- ``options`` (temperature, top_p, top_k, penalties, num_predict)
- ``format`` (``"json"`` and JSON-schema structured output)
- streaming and non-streaming responses, mapped back to
  :class:`ollama.ChatResponse`

Known limitations (llama-server constraints, not bugs)
-----------------------------------------------------
- ``num_ctx`` is fixed at server launch (``--ctx-size``); per-request
  ``context_length`` cannot be honoured and is dropped with a debug log.
- ``think`` toggling is not exposed on the OpenAI-compat endpoint; thinking
  is configured at server launch (chat-template level).
- ``keep_alive`` is a no-op: llama-server keeps the model resident until the
  process exits.
"""

import json
import mimetypes
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence, final

import httpx
from ollama import ChatResponse, Image, Message, Tool
from ollama._utils import convert_function_to_tool  # type: ignore[import-untyped]

LOGGER = getLogger("messaging.backends.llamacpp")

# Ollama ``options`` keys -> llama-server /v1/chat/completions body keys.
_OPTION_MAP: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "repeat_penalty": "repeat_penalty",
    "num_predict": "n_predict",
}

# Options accepted by Ollama but not expressible per-request on llama-server.
_UNSUPPORTED_OPTIONS: tuple[str, ...] = ("num_ctx",)


@final
class LlamaCppBackend:
    """Backend speaking llama-server's OpenAI-compatible HTTP API."""

    __slots__ = ("_host", "_client", "_pending_tool_ids")

    def __init__(self, host: str, timeout: float = 600.0) -> None:
        self._host: str = host.rstrip("/")
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout)
        )
        # tool_call ids for the tool results following the last assistant
        # message that requested tool calls (see _convert_messages).
        self._pending_tool_ids: list[str] = []

    async def chat(self, **kwargs: Any) -> ChatResponse | AsyncIterator[ChatResponse]:
        """Run a chat request; returns a response or an async chunk iterator."""
        body = self._build_body(**kwargs)
        if kwargs.get("stream"):
            return self._stream_chat(body)
        return await self._chat_once(body)

    async def close(self) -> None:
        await self._client.aclose()

    def _build_body(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        stream: bool = False,
        tools: list[Tool | Any] | None = None,
        options: dict[str, Any] | None = None,
        format: str | Mapping[str, Any] | None = None,
        think: Any = None,
        keep_alive: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": stream,
        }
        if tools:
            body["tools"] = self._convert_tools(tools)
        if options:
            body.update(self._convert_options(options))
        if format is not None:
            body["response_format"] = self._convert_format(format)
        if think is not None:
            LOGGER.debug(
                "llamacpp: 'think' is not exposed on /v1/chat/completions; "
                "ignoring request hint (configure thinking at server launch)."
            )
        if keep_alive is not None:
            LOGGER.debug(
                "llamacpp: 'keep_alive' is a no-op; llama-server keeps the "
                "model resident until the process exits."
            )
        return body

    def _convert_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Translate Ollama ``Message`` objects to OpenAI-style messages."""
        self._pending_tool_ids = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            converted.append(self._convert_message(message))
        return converted

    def _convert_message(self, message: Message) -> dict[str, Any]:
        if message.role == "system":
            return {"role": "system", "content": message.content or ""}

        if message.role == "tool":
            # Ollama pairs tool results with the previous assistant tool
            # calls by order; OpenAI requires the explicit tool_call_id.
            tool_call_id = (
                self._pending_tool_ids.pop(0)
                if self._pending_tool_ids
                else message.tool_name or "tool"
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": message.content or "",
            }

        if message.role == "assistant":
            out: dict[str, Any] = {"role": "assistant"}
            if message.content:
                out["content"] = message.content
            if message.thinking:
                out["reasoning_content"] = message.thinking
            if message.tool_calls:
                out["tool_calls"] = []
                for index, call in enumerate(message.tool_calls):
                    tool_call_id = f"call_{index}_{call.function.name}"
                    self._pending_tool_ids.append(tool_call_id)
                    out["tool_calls"].append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": json.dumps(dict(call.function.arguments)),
                            },
                        }
                    )
            return out

        # Default: user message (optionally multimodal).
        if not message.images:
            return {"role": "user", "content": message.content or ""}
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"type": "text", "text": message.content})
        for image in message.images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image)},
                }
            )
        return {"role": "user", "content": parts}

    def _convert_tools(self, tools: list[Tool | Any]) -> list[dict[str, Any]]:
        """Translate ``Tool`` objects / callables to OpenAI tool definitions."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            spec = (
                convert_function_to_tool(tool)
                if callable(tool)
                else Tool.model_validate(tool)
            )
            converted.append(
                {
                    "type": "function",
                    "function": spec.function.model_dump(exclude_none=True),
                }
            )
        return converted

    def _convert_options(self, options: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {}
        for key, value in options.items():
            if value is None:
                continue
            if key in _OPTION_MAP:
                body[_OPTION_MAP[key]] = value
            elif key in _UNSUPPORTED_OPTIONS:
                LOGGER.debug(
                    "llamacpp: option %r is not per-request capable; "
                    "ignoring (set it at llama-server launch).",
                    key,
                )
        return body

    def _convert_format(self, format: str | Mapping[str, Any]) -> dict[str, Any]:
        """Translate Ollama's ``format`` argument to OpenAI ``response_format``."""
        if isinstance(format, str) and format == "json":
            return {"type": "json_object"}
        if isinstance(format, Mapping):
            return {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": dict(format)},
            }
        LOGGER.warning(
            "llamacpp: unsupported format %r; sending without structured output.",
            format,
        )
        return {"type": "text"}

    def _image_data_url(self, image: Image) -> str:
        """Encode an Ollama ``Image`` as a base64 data URL."""
        value = image.value
        if isinstance(value, bytes):
            raw, mime = value, "image/png"
        else:
            path = Path(value)
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64,{self._b64(raw)}"

    @staticmethod
    def _b64(raw: bytes) -> str:
        from base64 import b64encode

        return b64encode(raw).decode("ascii")

    async def _chat_once(self, body: dict[str, Any]) -> ChatResponse:
        response = await self._client.post(
            f"{self._host}/v1/chat/completions", json=body
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_chunk(data)

    async def _stream_chat(self, body: dict[str, Any]) -> AsyncIterator[ChatResponse]:
        tool_fragments: dict[int, dict[str, Any]] = {}
        async with self._client.stream(
            "POST", f"{self._host}/v1/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                chunk = json.loads(payload)
                if not chunk.get("choices"):
                    # e.g. usage-only events; nothing to surface.
                    continue
                parsed, tool_fragments = self._parse_stream_chunk(chunk, tool_fragments)
                yield parsed

    def _parse_stream_chunk(
        self, chunk: dict[str, Any], tool_fragments: dict[int, dict[str, Any]]
    ) -> tuple[ChatResponse, dict[int, dict[str, Any]]]:
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        content = delta.get("content") or ""
        thinking = delta.get("reasoning_content") or ""

        # Accumulate tool-call fragments (OpenAI streams them piecewise).
        for fragment in delta.get("tool_calls") or []:
            index = int(fragment.get("index", 0))
            slot = tool_fragments.setdefault(index, {"name": "", "arguments_parts": []})
            function = fragment.get("function") or {}
            if function.get("name"):
                slot["name"] = function["name"]
            if function.get("arguments"):
                slot["arguments_parts"].append(function["arguments"])

        # Emit the assembled tool calls on the final chunk only, matching
        # how Ollama surfaces tool_calls in streaming responses.
        tool_calls: list[Message.ToolCall] | None = None
        if finish:
            tool_calls = self._assemble_tool_calls(tool_fragments) or None
            tool_fragments = {}

        return (
            ChatResponse(
                model=chunk.get("model"),
                created_at=datetime.now(UTC).isoformat(),
                message=Message(
                    role="assistant",
                    content=content or None,
                    thinking=thinking or None,
                    tool_calls=tool_calls,
                ),
                done=finish is not None,
                done_reason=self._map_finish_reason(finish) if finish else None,
            ),
            tool_fragments,
        )

    def _parse_chunk(self, data: dict[str, Any]) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        done_reason = self._map_finish_reason(choice.get("finish_reason"))

        tool_calls: list[Message.ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        name=function.get("name", ""),
                        arguments=arguments,
                    )
                )
            )

        return ChatResponse(
            model=data.get("model"),
            created_at=datetime.now(UTC).isoformat(),
            message=Message(
                role="assistant",
                content=(message.get("content") or None),
                thinking=(message.get("reasoning_content") or None),
                tool_calls=tool_calls or None,
            ),
            done=True,
            done_reason=done_reason,
        )

    @staticmethod
    def _assemble_tool_calls(
        fragments: dict[int, dict[str, Any]],
    ) -> list[Message.ToolCall]:
        calls: list[Message.ToolCall] = []
        for index in sorted(fragments):
            fragment = fragments[index]
            arguments_raw = "".join(fragment.get("arguments_parts", []))
            try:
                arguments = json.loads(arguments_raw or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        name=fragment.get("name", ""),
                        arguments=arguments,
                    )
                )
            )
        return calls

    @staticmethod
    def _map_finish_reason(reason: str | None) -> str | None:
        if reason is None:
            return None
        # llama-server uses "tool_calls" exactly like Ollama for tool stops.
        return reason
