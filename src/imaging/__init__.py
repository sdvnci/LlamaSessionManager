"""Public API for the imaging module - image MIME types."""

from ._types import ImageMimeTypeEnum, downscale_image
from .slicer import (
    DocumentSlice,
    LandRegisterOCRDocument,
    OCREnabledDocument,
    OCREnabledDocumentEnum,
)

__all__ = (
    "ImageMimeTypeEnum",
    "DocumentSlice",
    "OCREnabledDocument",
    "LandRegisterOCRDocument",
    "OCREnabledDocumentEnum",
    "downscale_image",
)
