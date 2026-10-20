"""Pluggable LLM backends for :class:`~src.messaging.MessageSession`.

A backend translates the messaging layer's Ollama-shaped chat API
(``messages``/``tools``/``options``/``stream`` kwargs and
:class:`ollama.ChatResponse` results) onto a concrete inference server:

- ``ollama``   - the official Ollama server (default, unchanged behaviour).
- ``llamacpp`` - a `llama.cpp <https://github.com/ggml-org/llama.cpp>`_
                 ``llama-server`` process exposing its OpenAI-compatible
                 ``/v1/chat/completions`` endpoint.

Sessions pick their backend via ``SessionConfig.backend`` (falling back to
the ``LLM_BACKEND`` environment variable, then ``"ollama"``).
"""

from .base import (
    BACKEND_KINDS,
    BackendKind,
    DEFAULT_BACKEND,
    create_backend,
    resolve_backend_kind,
)
from .ollama_backend import OllamaBackend
from .llamacpp_backend import LlamaCppBackend

__all__ = (
    "BACKEND_KINDS",
    "BackendKind",
    "DEFAULT_BACKEND",
    "create_backend",
    "resolve_backend_kind",
    "OllamaBackend",
    "LlamaCppBackend",
)
