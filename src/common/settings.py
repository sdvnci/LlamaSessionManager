"""Runtime feature flags read from the environment.

Keep this module dependency-free and importable from anywhere in the
project — it is consumed by both the messaging layer (which decides whether
to spin up the Whisper worker) and the HTTP routes (which decide whether a
transcription request may proceed).
"""

from __future__ import annotations

import os

# Env var name controlling whether audio transcription is available at all.
TRANSCRIPTION_ENABLED_ENV: str = "TRANSCRIPTION_ENABLED"

# Truthy values recognised.  Crucially, an unset/empty variable is treated
# as ``False``: transcription (and its Whisper GPU pre-allocation) is OFF by
# default so Ollama keeps the cards it needs for OCR/chat.  The host must
# opt in explicitly with ``TRANSCRIPTION_ENABLED=true``.
_TRANSCRIPTION_TRUE: frozenset[str] = frozenset({"1", "true", "yes", "on", "enabled"})


def _parse_bool(value: str | None) -> bool:
    """Interpret an env value as a boolean.  Defaults to ``False``."""
    if value is None:
        return False
    return value.strip().lower() in _TRANSCRIPTION_TRUE


def transcription_enabled() -> bool:
    """Return ``True`` when audio transcription is enabled for this server.

    Controlled by the ``TRANSCRIPTION_ENABLED`` environment variable and
    **disabled by default**.  Because this server only has limited VRAM and
    Ollama must keep OCR/chat models resident, transcription (and its Whisper
    GPU pre-allocation) only engages when the variable is set to a truthy
    value such as ``"1"``/``"true"``/``"yes"``/``"on"``.
    """
    return _parse_bool(os.environ.get(TRANSCRIPTION_ENABLED_ENV))


def transcription_capability() -> dict[str, bool]:
    """The serialisable capability block shared by every transcription surface.

    Endpoints and WebSocket messages that deal with transcription embed this
    so that a single field (``transcription_enabled``) is the one source of
    truth reflected everywhere for feature-aware clients.
    """
    return {"transcription_enabled": transcription_enabled()}


def transcription_disabled_message() -> str:
    """A human-readable reason sent back when transcription is disabled."""
    return (
        "Audio transcription is disabled on this server. "
        f"Set {TRANSCRIPTION_ENABLED_ENV}=true to enable it."
    )
