from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Iterable, cast

from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator

from src.imaging._types import ImageMimeTypeEnum


@dataclass(slots=True, frozen=True)
class DocumentSlice:
    """
    A slice of a document with usable information which we want to exctract via ocr.
    The crop box is given by the start and end cartesian points, and the
    prompt is that given to the summarization llm to arrange the information extracted
    via ocr from documents of this kind.
    Usage:
    ```python
    parcel_slice = DocumentSlice(
        label="parcel information",
            programmatic=True,
        start=(725, 122),
        end=(None, 260),
        prompt=\"Look for the combined block+parcel value appearing twice, under the labels 'No.' and 'Parcel No.' ...\"
    )
    ```
    """

    label: str = ""
    prompt: str = ""
    dpi: int = 72
    start: tuple[int, int] = (0, 0)
    end: tuple[int | None, int | None] = (0, 0)
    programmatic: bool = False


class OCREnabledDocument(BaseModel, ABC):
    """
    Base class for Documents for which we want to extract information.
    """

    model_config = ConfigDict(slots=True, arbitrary_types_allowed=True)  # type: ignore
    _label: ClassVar[str] = ""
    _sections: ClassVar[list[DocumentSlice]] = []

    image: Image.Image

    @field_validator("image", mode="before")
    def validate_image(cls, value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            img = cast(Image.Image, value)
        elif isinstance(value, Path):
            if not value.exists():
                raise RuntimeError(f"Document image {value} does not exist")
            ImageMimeTypeEnum.from_extension(value.suffix.lower().lstrip("."))
            img = Image.open(value)
        else:
            raise RuntimeError(
                f"Image value of type {type(value)} is not valid for OCR Image Slicer."
            )

        # Store the DPI scale factor so segment_document can adjust coordinates.
        # Slice coordinates are defined for a 72-DPI reference image.
        try:
            dpi_info = img.info.get("dpi")
            if dpi_info is not None:
                dpi_x = dpi_info[0] if isinstance(dpi_info, tuple) else dpi_info
                img.info["_scale"] = dpi_x / 72.0
            else:
                img.info["_scale"] = 1.0
        except Exception:
            img.info["_scale"] = 1.0

        return img

    def segment_document(self) -> list[tuple[str, Image.Image]]:
        scale: float = float(self.image.info.get("_scale", 1.0))
        segments: list[tuple[str, Image.Image]] = []
        for section in self.sections:
            sx = int(section.start[0] * scale)
            sy = int(section.start[1] * scale)
            ex = int((section.end[0] or (self.image.width / scale)) * scale)
            ey = int((section.end[1] or (self.image.height / scale)) * scale)
            segments.append(
                (
                    section.prompt,
                    self.image.crop((sx, sy, ex, ey)),
                )
            )

        return segments

    def __init_subclass__(
        cls, label: str, slices: Iterable[DocumentSlice], **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls._label = label
        cls._sections = list(slices)

    @property
    def sections(self) -> list[DocumentSlice]:
        return self._sections


class LandRegisterOCRDocument(
    OCREnabledDocument,
    label="LandRegister",
    slices=[
        # --- registration ---
        DocumentSlice(
            label="registration",
            programmatic=False,
            start=(0, 122),
            end=(320, 275),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this land register title section. Raw text only."
            ),
        ),
        # --- appurtenances ---
        DocumentSlice(
            label="appurtenances",
            programmatic=False,
            start=(310, 122),
            end=(725, 275),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this lease summary section. Include Lessor, Lessee, Rent, Term. Raw text only."
            ),
        ),
        # --- parcel information ---
        DocumentSlice(
            label="parcel information",
            programmatic=False,
            start=(710, 120),
            end=(None, 275),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this parcel details section. Raw text only."
            ),
        ),
        # --- proprietorship instruments ---
        DocumentSlice(
            label="proprietorship instruments",
            programmatic=False,
            start=(0, 315),
            end=(270, 610),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this instrument table section. Raw text only."
            ),
        ),
        # --- proprietorship personas ---
        DocumentSlice(
            label="proprietorship personas",
            programmatic=False,
            start=(255, 315),
            end=(845, 610),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this ownership/proprietor section. Raw text only."
            ),
        ),
        # --- incumbrance instruments ---
        DocumentSlice(
            label="incumbrance instruments",
            programmatic=False,
            start=(0, 730),
            end=(325, None),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this encumbrance instrument table. Raw text only."
            ),
        ),
        # --- incumbrance particulars ---
        DocumentSlice(
            label="incumbrance particulars",
            programmatic=False,
            start=(310, 730),
            end=(780, None),
            prompt=(
                "Act as a pure OCR engine. Transcribe every character exactly as it appears "
                "in this encumbrance particulars section. Raw text only."
            ),
        ),
    ],
):
    """
    A Land Register style OCR-enabled document
    """

    ...


class OCREnabledDocumentEnum(StrEnum):
    LAND_REGISTER = "register"

    @classmethod
    def from_string(cls, value: str) -> "OCREnabledDocumentEnum":
        wrong = "_- .,:;{}[]=+\\|/?`~()*&^%$#@!"
        for chr in wrong:
            value = value.replace(chr, "")
        match value.strip().lower():
            case "register" | "landregister" | "landreg" | "lreg":
                return cls.LAND_REGISTER
            case _:
                raise RuntimeError(
                    f"{value} does not match onto a known document category"
                )

    @property
    def cls(self) -> type[OCREnabledDocument]:
        match self:
            case OCREnabledDocumentEnum.LAND_REGISTER:
                return LandRegisterOCRDocument
            case _:
                raise RuntimeError(
                    f"Could not match {self.value} to a known document category."
                )


__all__ = (
    "DocumentSlice",
    "OCREnabledDocument",
    "LandRegisterOCRDocument",
    "OCREnabledDocumentEnum",
)
