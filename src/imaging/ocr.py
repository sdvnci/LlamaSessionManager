"""
Tesseract-based OCR for document image slices.

Provides a reliable, hallucination-free text extraction path that runs
locally without an LLM.  Use this for the OCR pass; reserve LLM calls
for the downstream summarisation / JSON-structuring step.
"""

from pathlib import Path
from typing import cast

from PIL import Image, ImageFilter


def tesseract_ocr(
    image: Image.Image | Path,
    *,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    config: str = "",
) -> str:
    """
    Extract text from *image* using Tesseract.

    The image is preprocessed (greyscale + sharpening) before OCR to
    improve accuracy on low-resolution document scans.

    Parameters
    ----------
    image:
        PIL ``Image`` or path to an image file.
    lang:
        Tesseract language code (``"eng"``, ``"eng+fra"``, etc.).
    psm:
        Page-segmentation mode.  Default ``6`` = "assume a uniform block
        of text".  Other useful values: ``3`` = fully automatic,
        ``4`` = single column.
    oem:
        OCR engine mode.  ``3`` = default (LSTM + legacy).
    config:
        Additional Tesseract config flags (e.g. ``"--dpi 300"``).

    Returns
    -------
    The extracted text, with leading / trailing whitespace stripped.
    An empty string is returned if no text was found.

    Raises
    ------
    RuntimeError
        If ``pytesseract`` is not installed or Tesseract is not on PATH.
    """
    try:
        import pytesseract  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "pytesseract is required for OCR — pip install pytesseract "
            "and install the Tesseract engine: "
            "https://github.com/tesseract-ocr/tesseract"
        ) from None

    # Preprocess: greyscale + sharpen for better accuracy on document scans.
    if isinstance(image, Path):
        img = cast(Image.Image, Image.open(image))
    else:
        img = image

    img = img.convert("L")  # greyscale
    img = img.filter(ImageFilter.SHARPEN)

    custom = f"--psm {psm} --oem {oem}"
    if config:
        custom = f"{custom} {config}"

    try:
        text: str = pytesseract.image_to_string(img, lang=lang, config=custom)
    except Exception as exc:
        raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc

    return text.strip()


__all__ = ("tesseract_ocr",)
