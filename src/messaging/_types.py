from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict


@dataclass(slots=True, frozen=True)
class SessionConfig:
    """
    Frozen, hashable configuration for a MessageSession.
    Analogous to ConnectionConfig in AsyncShipStation.

    Passed to the MessageSession constructor and owned for its lifetime.
    Can be loaded from a JSON config dict via ``from_dict()``.

    Example JSON config file::

        {
            "model": "qwen2.5-coder:7b",
            "name": "ORCA",
            "description": "A medical assistant chatbot",
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.4,
            "frequency_penalty": 0.4,
            "repeat_penalty": 1.0,
            "context_length": 40960,
            "system": "You are a helpful and precise assistant ..."
        }
    """

    model: str
    name: str | None = None
    description: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    context_length: int | None = None
    num_predict: int | None = None
    system: str | None = None
    host: str = "http://localhost:11434"
    timeout: float = 600.0

    def __hash__(self) -> int:
        raw = (
            f"{self.model}:{self.name}:{self.system}:"
            f"{self.temperature}:{self.top_p}:{self.top_k}:{self.host}"
        )
        digest = int.from_bytes(sha256(raw.encode()).digest()[:8], "big", signed=True)
        return -2 if digest == -1 else digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionConfig):
            return NotImplemented
        return (
            self.model == other.model
            and self.name == other.name
            and self.system == other.system
            and self.temperature == other.temperature
            and self.top_p == other.top_p
            and self.top_k == other.top_k
            and self.host == other.host
        )

    def to_options(self) -> dict[str, Any]:
        """
        Build the Ollama ``options`` dict for a chat request, omitting ``None`` values.
        Keys map directly to Ollama's ``Options`` TypedDict.
        """
        opts: dict[str, Any] = {}
        if self.temperature is not None:
            opts["temperature"] = self.temperature
        if self.top_p is not None:
            opts["top_p"] = self.top_p
        if self.top_k is not None:
            opts["top_k"] = self.top_k
        if self.presence_penalty is not None:
            opts["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            opts["frequency_penalty"] = self.frequency_penalty
        if self.repeat_penalty is not None:
            opts["repeat_penalty"] = self.repeat_penalty
        if self.context_length is not None:
            opts["num_ctx"] = self.context_length
        if self.num_predict is not None:
            opts["num_predict"] = self.num_predict
        return opts

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        """
        Build a ``SessionConfig`` from a plain dict, e.g. loaded from a JSON
        config file. Unknown keys are silently ignored.
        """
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_json(cls, path: Path) -> "SessionConfig":
        """
        Load a ``SessionConfig`` directly from a JSON file.
        If the file contains a top-level ``"models"`` list, the first entry
        whose ``"name"`` or ``"model"`` field matches is used; otherwise the
        root object is used directly.
        """
        from json import loads

        data: dict[str, Any] = loads(path.read_text(encoding="utf-8"))
        if "models" in data:
            # Multi-model file — caller should iterate themselves, but
            # we handle the single-entry case gracefully.
            entries: list[dict[str, Any]] = data["models"]
            if not entries:
                raise ValueError(f"No models found in {path}")
            data = entries[0]
        return cls.from_dict(data)


class SessionExport(TypedDict):
    """Serialisable snapshot of a session for persistence or replay."""

    uid: str
    timestamp: str
    name: str | None
    model: str
    description: str | None
    system: str | None
    messages: list[dict[str, Any]]
