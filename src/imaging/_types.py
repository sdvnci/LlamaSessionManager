from pathlib import Path
from typing import Any, Literal, Protocol, final, overload

from deskew import determine_skew
from numpy import array
from PIL import Image

from ..common import MimeTypeEnum

ImageExtensions = Literal["png", "jpg", "jpeg", "gif", "bmp", "webp"]

# Target resolution for images sent to a vision model during OCR.  Scanned
# documents rasterised at 300 DPI produce ~20 MP images that turn into
# thousands of vision tokens and dominate latency; 72 DPI is more than enough
# to read printed/handwritten registry text and slashes the token count.
OCR_TARGET_DPI: int = 72

# When an image carries no DPI metadata (e.g. a camera photo or an upload
# saved without a dpi tag), fall back to capping its longest edge.  This is a
# 72-DPI-equivalent budget for a ~14 inch page.
OCR_MAX_LONG_EDGE: int = 1000


class PDFRasteriser(Protocol):
    """Callable that rasterises a PDF to PNG, optionally accepting extra kwargs."""

    def __call__(
        self,
        path: Path,
        out_dir: Path,
        dpi: int = 300,
        **kwargs: Any,
    ) -> tuple[Path | None, Exception | None]: ...


def unskew_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    angle = determine_skew(array(gray))
    return image.rotate(float(angle)) if angle is not None else image


def downscale_image(
    image: Image.Image,
    target_dpi: int = OCR_TARGET_DPI,
    max_long_edge: int = OCR_MAX_LONG_EDGE,
) -> Image.Image:
    """Reduce an image to a standard OCR resolution before sending to a vision model.

    Scanned pages are often rasterised at 300 DPI, which yields 20+ megapixel
    images and thousands of vision tokens - the dominant OCR latency cost.
    This helper shrinks an image to ``target_dpi`` (72 by default) using the
    image's DPI metadata, and falls back to capping the longest edge for
    images that carry no DPI tag.  Images already at or below the target are
    returned unchanged.
    """
    dpi = image.info.get("dpi")
    dpi_x = dpi[0] if isinstance(dpi, (tuple, list)) and dpi else dpi
    try:
        dpi_x = float(dpi_x) if dpi_x else None
    except (TypeError, ValueError):
        dpi_x = None

    if dpi_x and dpi_x > target_dpi:
        scale = target_dpi / dpi_x
        w = max(1, round(image.width * scale))
        h = max(1, round(image.height * scale))
        return image.resize((w, h), Image.Resampling.LANCZOS)

    long_edge = max(image.width, image.height)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        w = max(1, round(image.width * scale))
        h = max(1, round(image.height * scale))
        return image.resize((w, h), Image.Resampling.LANCZOS)

    return image


def _pdf_to_png(
    path: Path, out_dir: Path, dpi: int = 300, **kwargs
) -> tuple[Path | None, Exception | None]:
    """Rasterise an entire PDF to a single tall PNG via pymupdf + Pillow.

    All pages are stacked vertically into one image written to *out_dir*.
    Returns ``(png_path, None)`` on success or ``(None, exc)`` on failure.
    """

    limit = kwargs.get("limit", None)
    if limit is not None and not isinstance(limit, int):
        raise RuntimeError(f"Limit can only be None or int, not {type(limit)}")

    max_dim = kwargs.get("max_dim", None)
    if max_dim is not None and not isinstance(max_dim, int):
        raise RuntimeError(f"max_dim can only be None or int, not {type(max_dim)}")

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError:
        return None, ImportError(
            "pymupdf is required for PDF rasterisation - pip install pymupdf"
        )

    try:
        doc = fitz.open(str(path))
        page_images: list[Image.Image] = []
        count = 0
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_images.append(unskew_image(img))
            count += 1
            if limit is not None:
                if count == limit:
                    break
        doc.close()

        if not page_images:
            return None, ValueError(f"PDF has no pages: {path.name}")

        width = max(im.width for im in page_images)
        height = sum(im.height for im in page_images)
        combined = Image.new("RGB", (width, height), color=(255, 255, 255))

        y = 0
        for im in page_images:
            # centre narrower pages horizontally
            x = (width - im.width) // 2
            combined.paste(im, (x, y))
            y += im.height

        dest = out_dir / f"{path.stem}.png"

        # Cap the long edge: very tall stacks (2+ pages at 300 DPI) produce
        # images whose token count blows past vision models' limits and
        # makes them emit garbage.  Vision models resize internally anyway.
        if max_dim is not None and max_dim < max(combined.size):
            combined.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        combined.save(str(dest), format="PNG", dpi=(dpi, dpi))
        return dest, None
    except Exception as exc:
        return None, exc


@final
class ImageMimeTypeEnum(MimeTypeEnum):
    """
    Common MIME types for image files.  Some types carry a ``reader`` that
    can convert non-pixel formats (e.g. PDF) into raster images consumable
    by Ollama vision models.
    """

    PNG = "image/png"
    JPG = "image/jpg"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    BMP = "image/bmp"
    WEBP = "image/webp"
    PDF = "application/pdf"

    @overload
    @classmethod
    def extension_matcher(
        cls: type["ImageMimeTypeEnum"], mime_type: ImageExtensions
    ) -> "ImageMimeTypeEnum": ...

    @overload
    @classmethod
    def extension_matcher(
        cls: type["ImageMimeTypeEnum"], mime_type: str
    ) -> "ImageMimeTypeEnum": ...

    @classmethod
    def extension_matcher(
        cls: type["ImageMimeTypeEnum"], mime_type: str
    ) -> "ImageMimeTypeEnum":
        match mime_type:
            case "png":
                return cls.PNG
            case "jpg":
                return cls.JPG
            case "jpeg":
                return cls.JPEG
            case "gif":
                return cls.GIF
            case "bmp":
                return cls.BMP
            case "webp":
                return cls.WEBP
            case "pdf":
                return cls.PDF
            case _:
                raise ValueError(f"Unknown image extension: {mime_type}")

    @property
    def reader(
        self,
    ) -> PDFRasteriser | None:
        """Return a converter for types that need rasterisation before use.

        PDF -> single tall PNG (requires *out_dir*); all other image types are
        already pixel data and return ``None``.
        """
        if self == ImageMimeTypeEnum.PDF:
            return _pdf_to_png
        return None
