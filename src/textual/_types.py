from io import BytesIO
from pathlib import Path
from typing import Callable, Literal, Self, final, overload

from ..common._types import MimeTypeEnum

TextualExtensions = Literal[
    "txt",
    "md",
    "rst",
    "csv",
    "json",
    "xml",
    "html",
    "htm",
    "py",
    "ts",
    "js",
    "go",
    "c",
    "cpp",
    "yaml",
    "toml",
    # ---- MS Word ----
    "doc",
    "dot",
    "docx",
    "dotx",
    "docm",
    "dotm",
    # ---- MS Excel ----
    "xls",
    "xlt",
    "xla",
    "xlsx",
    "xltx",
    "xlsm",
    "xltm",
    "xlam",
    "xlsb",
    # ---- MS PowerPoint ----
    "ppt",
    "pot",
    "pps",
    "ppa",
    "pptx",
    "potx",
    "ppsx",
    "ppam",
    "pptm",
    "potm",
    "ppsm",
    # ---- MS Access ----
    "mdb",
]


@final
class TextualMimeTypeEnum(MimeTypeEnum):
    """
    Common MIME types for textual data.
    """

    PLAIN = "text/plain"
    HTML = "text/html"
    PDF = "application/pdf"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    XML = "application/xml"
    CSV = "text/csv"
    TSV = "text/tab-separated-values"
    YAML = "application/x-yaml"
    TOML = "application/toml"
    # ---- MS Word ----
    MSWORD = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOTX = "application/vnd.openxmlformats-officedocument.wordprocessingml.template"
    DOCM = "application/vnd.ms-word.document.macroEnabled.12"
    DOTM = "application/vnd.ms-word.template.macroEnabled.12"
    # ---- MS Excel ----
    MSEXCEL = "application/vnd.ms-excel"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    XLTX = "application/vnd.openxmlformats-officedocument.spreadsheetml.template"
    XLSM = "application/vnd.ms-excel.sheet.macroEnabled.12"
    XLTM = "application/vnd.ms-excel.template.macroEnabled.12"
    XLAM = "application/vnd.ms-excel.addin.macroEnabled.12"
    XLSB = "application/vnd.ms-excel.sheet.binary.macroEnabled.12"
    # ---- MS PowerPoint ----
    MSPOWERPOINT = "application/vnd.ms-powerpoint"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    POTX = "application/vnd.openxmlformats-officedocument.presentationml.template"
    PPSX = "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
    PPAM = "application/vnd.ms-powerpoint.addin.macroEnabled.12"
    PPTM = "application/vnd.ms-powerpoint.presentation.macroEnabled.12"
    POTM = "application/vnd.ms-powerpoint.template.macroEnabled.12"
    PPSM = "application/vnd.ms-powerpoint.slideshow.macroEnabled.12"
    # ---- MS Access ----
    MSACCESS = "application/vnd.ms-access"

    @overload
    @classmethod
    def extension_matcher(
        cls: type[Self], mime_type: TextualExtensions
    ) -> "TextualMimeTypeEnum": ...

    @overload
    @classmethod
    def extension_matcher(cls: type[Self], mime_type: str) -> "TextualMimeTypeEnum": ...

    @classmethod
    def extension_matcher(cls: type[Self], mime_type: str) -> "TextualMimeTypeEnum":
        match mime_type:
            case "txt" | "rst" | "py" | "ts" | "js" | "go" | "c" | "cpp":
                return cls.PLAIN
            case "md":
                return cls.MARKDOWN
            case "html" | "htm":
                return cls.HTML
            case "json":
                return cls.JSON
            case "xml":
                return cls.XML
            case "csv":
                return cls.CSV
            case "yaml" | "yml":
                return cls.YAML
            case "toml":
                return cls.TOML
            case "pdf":
                return cls.PDF
            # ---- MS Word ----
            case "doc" | "dot":
                return cls.MSWORD
            case "docx":
                return cls.DOCX
            case "dotx":
                return cls.DOTX
            case "docm":
                return cls.DOCM
            case "dotm":
                return cls.DOTM
            # ---- MS Excel ----
            case "xls" | "xlt" | "xla":
                return cls.MSEXCEL
            case "xlsx":
                return cls.XLSX
            case "xltx":
                return cls.XLTX
            case "xlsm":
                return cls.XLSM
            case "xltm":
                return cls.XLTM
            case "xlam":
                return cls.XLAM
            case "xlsb":
                return cls.XLSB
            # ---- MS PowerPoint ----
            case "ppt" | "pot" | "pps" | "ppa":
                return cls.MSPOWERPOINT
            case "pptx":
                return cls.PPTX
            case "potx":
                return cls.POTX
            case "ppsx":
                return cls.PPSX
            case "ppam":
                return cls.PPAM
            case "pptm":
                return cls.PPTM
            case "potm":
                return cls.POTM
            case "ppsm":
                return cls.PPSM
            # ---- MS Access ----
            case "mdb":
                return cls.MSACCESS
            case _:
                raise ValueError(f"Unsupported textual extension: {mime_type}")

    @property
    def reader(
        self: Self,
    ) -> Callable[[Path | BytesIO], tuple[str, Exception | None]] | None:
        from .tools import read_docx_file, read_pdf_file, read_plaintext_text_file

        match self:
            case (
                TextualMimeTypeEnum.PLAIN
                | TextualMimeTypeEnum.MARKDOWN
                | TextualMimeTypeEnum.HTML
                | TextualMimeTypeEnum.JSON
                | TextualMimeTypeEnum.XML
                | TextualMimeTypeEnum.CSV
                | TextualMimeTypeEnum.TSV
                | TextualMimeTypeEnum.YAML
                | TextualMimeTypeEnum.TOML
            ):
                return read_plaintext_text_file
            case TextualMimeTypeEnum.PDF:
                return read_pdf_file
            case (
                TextualMimeTypeEnum.MSWORD
                | TextualMimeTypeEnum.DOCX
                | TextualMimeTypeEnum.DOTX
                | TextualMimeTypeEnum.DOCM
                | TextualMimeTypeEnum.DOTM
            ):
                return read_docx_file
            case (
                TextualMimeTypeEnum.MSEXCEL
                | TextualMimeTypeEnum.XLSX
                | TextualMimeTypeEnum.XLTX
                | TextualMimeTypeEnum.XLSM
                | TextualMimeTypeEnum.XLTM
                | TextualMimeTypeEnum.XLAM
                | TextualMimeTypeEnum.XLSB
                | TextualMimeTypeEnum.MSPOWERPOINT
                | TextualMimeTypeEnum.PPTX
                | TextualMimeTypeEnum.POTX
                | TextualMimeTypeEnum.PPSX
                | TextualMimeTypeEnum.PPAM
                | TextualMimeTypeEnum.PPTM
                | TextualMimeTypeEnum.POTM
                | TextualMimeTypeEnum.PPSM
                | TextualMimeTypeEnum.MSACCESS
            ):
                return None
            case _:
                return None
