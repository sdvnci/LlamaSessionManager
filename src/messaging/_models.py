"""Logical -> backend-specific model name resolution.

Both backends serve the same underlying GGUF files, but they refer to them
differently: Ollama uses tags (``gpt-oss-20b-bench``) while llama-server uses
the alias/filename given at launch (``gpt-oss-20b``).  A registry file maps
the logical names used in API requests onto the right identifier per backend.

Registry format (JSON)::

    {
      "gpt-oss-20b": {
        "ollama": "gpt-oss-20b-bench",
        "llamacpp": "gpt-oss-20b"
      },
      "deepseek-ocr": "deepseek-ocr"
    }

A plain string value means the same name works on both backends.  Unknown
logical names pass through unchanged, so a registry is optional - set it up
only for models whose names differ between backends.

Location: ``MODEL_REGISTRY`` env var, else ``artifacts/model-registry.json``.
Loaded lazily once per process; restart the server to pick up edits.
"""

from json import loads
from logging import getLogger
from pathlib import Path
from typing import Any, final

from ..common import ARTIFACTDIR
from ._backends import BackendKind

LOGGER = getLogger("messaging.models")

#: Env var pointing at the registry JSON file.
MODEL_REGISTRY_ENV: str = "MODEL_REGISTRY"

#: Default registry location (host-mounted ``artifacts`` in Docker).
DEFAULT_REGISTRY_PATH: Path = ARTIFACTDIR / "model-registry.json"


@final
class ModelRegistry:
    """Maps logical model names to backend-specific identifiers."""

    __slots__ = ("_entries",)

    def __init__(self, entries: dict[str, Any]) -> None:
        self._entries: dict[str, Any] = dict(entries)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ModelRegistry":
        """Load a registry from ``path``; missing/invalid files yield an empty registry."""
        import os

        if path is None:
            path = os.environ.get(MODEL_REGISTRY_ENV, DEFAULT_REGISTRY_PATH)
        path = Path(path)
        if not path.exists():
            LOGGER.debug("Model registry %s not found; using pass-through.", path)
            return cls({})
        try:
            data: Any = loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning("Could not read model registry %s: %s", path, exc)
            return cls({})
        if not isinstance(data, dict):
            LOGGER.warning("Model registry %s is not a JSON object; ignoring.", path)
            return cls({})
        return cls(data)

    def resolve(self, name: str, backend: BackendKind) -> str:
        """Return the backend-specific model id for a logical ``name``."""
        entry = self._entries.get(name)
        if entry is None:
            return name
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return str(entry.get(backend) or entry.get("default") or name)
        LOGGER.warning(
            "Registry entry for %r has unsupported type %s; passing through.",
            name,
            type(entry).__name__,
        )
        return name


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Return the process-wide registry, loading it on first use."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.load()
    return _registry
