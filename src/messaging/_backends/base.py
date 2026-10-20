"""Shared backend protocol, kind resolution, and factory."""

import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, Protocol

if TYPE_CHECKING:
    from .._types import SessionConfig
    from ollama import ChatResponse


#: Backend identifiers accepted by ``SessionConfig.backend`` and ``LLM_BACKEND``.
BackendKind = Literal["ollama", "llamacpp"]

BACKEND_KINDS: tuple[str, ...] = ("ollama", "llamacpp")

#: Env var selecting the process-wide default backend.
BACKEND_ENV: str = "LLM_BACKEND"

#: Fallback backend when neither the session config nor the env var choose one.
DEFAULT_BACKEND: BackendKind = "ollama"


def resolve_backend_kind(value: str | None) -> BackendKind:
    """Resolve a backend identifier to a known kind.

    ``None`` falls back to the ``LLM_BACKEND`` environment variable and then
    ``"ollama"``.  Unknown explicit values raise ``ValueError`` so typos fail
    loudly instead of silently serving from the wrong backend.
    """
    if value is None:
        value = os.environ.get(BACKEND_ENV)
    if value is None or value == "":
        return DEFAULT_BACKEND
    value = value.strip().lower()
    if value not in BACKEND_KINDS:
        raise ValueError(
            f"Unknown LLM backend {value!r}; expected one of {BACKEND_KINDS}."
        )
    return value  # type: ignore[return-value]


class ChatBackend(Protocol):
    __slots__ = ()

    """Minimal chat interface every backend must provide.

    ``chat`` accepts the same kwargs :class:`ollama.AsyncClient.chat` takes
    (``model``, ``messages``, ``stream``, ``tools``, ``options``, ``format``,
    ``think``, ``keep_alive``) and returns Ollama-shaped results:
    a single :class:`ollama.ChatResponse` when ``stream`` is false, or an
    async iterator of :class:`ollama.ChatResponse` chunks when true.
    """

    async def chat(
        self, **kwargs: Any
    ) -> "ChatResponse | AsyncIterator[ChatResponse]": ...

    async def close(self) -> None:
        """Release any underlying HTTP connections."""


def create_backend(config: "SessionConfig") -> ChatBackend:
    """Instantiate the backend selected by ``config``."""
    from .ollama_backend import OllamaBackend
    from .llamacpp_backend import LlamaCppBackend

    kind = resolve_backend_kind(config.backend)
    if kind == "ollama":
        return OllamaBackend(host=config.host, timeout=config.timeout)
    return LlamaCppBackend(host=config.host, timeout=config.timeout)
