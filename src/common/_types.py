from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    NotRequired,
    Self,
    TypedDict,
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


class Sentinel: ...


class UIDTagged(ABC, BaseModel):
    """
    Base Pydantic model for any classes containing a `uid` attr, and using `__slots__`.
    """

    model_config = ConfigDict(slots=True)  # type: ignore
    uid: str = Field(default_factory=lambda: str(uuid4()))

    def __hash__(self) -> int:
        return hash(str(self.uid))


class CreateStamped(ABC, BaseModel):
    """
    Base Pydantic model for any classes containing a `created_at` attr, and using `__slots__`.
    """

    model_config = ConfigDict(slots=True)  # type: ignore
    created_at: Annotated[
        datetime,
        Field(default_factory=lambda: datetime.now(tz=UTC)),
        PlainSerializer(lambda v: v.strftime("%Y-%m-%d %H:%M:%S"), return_type=str),
    ]


class UpdateStamped(ABC, BaseModel):
    """
    Base Pydantic model for any classes containing a `updated_at` attr, and using `__slots__`.
    """

    model_config = ConfigDict(slots=True)  # type: ignore
    updated_at: Annotated[
        datetime,
        Field(default_factory=lambda: datetime.now(tz=UTC)),
        PlainSerializer(lambda v: v.strftime("%Y-%m-%d %H:%M:%S"), return_type=str),
    ]


class QueuedMessage(TypedDict):
    """
    Message envelope that flows through a producer → consumer pipeline.
    Compatibility with Ollama's Message dicts.
    ``image_paths`` contains absolute path strings that the consuming session
    should resolve to ``Path`` objects and pass to ``parse_content``.
    """

    id: str
    source_session_uid: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: str
    image_paths: NotRequired[list[str]]  # absolute paths; resolved to Path by consumer
    metadata: NotRequired[dict[str, Any]]  # arbitrary routing / tagging data


class MimeTypeEnum(str, Enum):
    @classmethod
    @abstractmethod
    def extension_matcher(cls: type[Self], mime_type: str) -> Self:
        """
        Match the extensions to the enum members.
        Raise a ValueError if no  match is found.
        """
        raise NotImplementedError(
            "Extension matcher has not been implemented for this Enum class."
        )

    @classmethod
    def from_extension(cls: type[Self], mime_type: str) -> Self:
        to_match = mime_type.lower().strip().replace(" ", "").replace(".", "")
        return cls.extension_matcher(to_match)

    @classmethod
    def is_valid(cls: type[Self], mime_type: str) -> bool:
        try:
            cls.from_extension(mime_type)
            return True
        except Exception as _:
            return False

    @property
    def reader(
        self: Self,
    ) -> (
        Callable[[Path | BytesIO], tuple[Any, Exception | None]]
        | Callable[[Path, Path], tuple[Path | None, Exception | None]]
        | None
    ):
        raise NotImplementedError("Readers are not defined for this Mime Enum Type")
