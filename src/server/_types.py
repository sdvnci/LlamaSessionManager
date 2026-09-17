from typing import Literal, TypedDict

ErrorTypes = Literal[
    "business_rules",
    "validation",
    "security",
    "system",
    "integrations",
    "arguments",
]

ErrorCodes = Literal[
    "field_conflict",
    "field_value_required",
    "forbidden",
    "attachment_not_supported",
    "rate_limit_exceeded",
    "request_body_required",
    "unauthorized",
    "unknown",
    "verification_failure",
]


class Error(TypedDict):
    error_source: str
    error_type: ErrorTypes
    error_code: ErrorCodes
    message: str


class ErrorResponse:
    errors: list[Error]
    session_uid: str | None
