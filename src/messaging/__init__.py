"""Public API for the messaging module - chat sessions, orchestration, and config."""

from ._backends import (
    BACKEND_KINDS,
    BackendKind,
    create_backend,
    resolve_backend_kind,
)
from ._messaging import MessageSession, ParsedResponse
from ._models import ModelRegistry, get_model_registry
from ._sessions import SessionManager
from ._types import SessionConfig, SessionExport

__all__ = (
    "BACKEND_KINDS",
    "BackendKind",
    "create_backend",
    "get_model_registry",
    "MessageSession",
    "ModelRegistry",
    "ParsedResponse",
    "resolve_backend_kind",
    "SessionConfig",
    "SessionExport",
    "SessionManager",
)
