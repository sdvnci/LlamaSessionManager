from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def read_plaintext_text_file(
    path: Path | BytesIO,
) -> tuple[str, Exception | None]:
    if isinstance(path, Path):
        try:
            return path.read_text(), None
        except Exception as exc:
            return "", exc

    if isinstance(path, BytesIO):
        try:
            return path.read().decode("utf-8"), None
        except Exception as exc:
            return "", exc

    return "", TypeError(f"Expected Path or BytesIO, got {type(path).__name__}")


def read_docx_file(path: Path | BytesIO) -> tuple[str, Exception | None]:
    try:
        document = Document(str(path) if isinstance(path, Path) else path)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return text, None
    except Exception as exc:
        return "", exc


def read_pdf_file(path: Path | BytesIO) -> tuple[str, Exception | None]:
    try:
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages), None
    except Exception as exc:
        return "", exc
