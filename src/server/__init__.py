from quart import Quart

from ..messaging import SessionManager
from .hub import TranscriptionHub
from .routes import (
    CHATBP,
    DOCOCRBP,
    LITERALBP,
    OCRBP,
    SETTINGSBP,
    TRANSBP,
    transcription_socket,
)


def create_app() -> Quart:
    app = Quart(__name__)

    # Allow large media uploads (e.g. meeting recordings).
    # Quart's default is 16 MiB, which trips a 413 on anything bigger.
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GiB

    app.register_blueprint(CHATBP)
    app.register_blueprint(OCRBP)
    app.register_blueprint(TRANSBP)
    app.register_blueprint(DOCOCRBP)
    app.register_blueprint(LITERALBP)
    app.register_blueprint(SETTINGSBP)

    # Transcription progress WebSocket — one connection per session UID.
    app.add_websocket("/ws/<session_uid>", view_func=transcription_socket)

    # Fan worker events out to session-bound WebSocket subscribers.  This is
    # the only place the hub is wired in; registration is idempotent.
    SessionManager.register_event_consumer(TranscriptionHub.handle_event)

    return app


__all__ = (
    "CHATBP",
    "LITERALBP",
    "OCRBP",
    "SETTINGSBP",
    "create_app",
)
