"""Ollama backend - thin wrapper over :class:`ollama.AsyncClient`.

This preserves the existing behaviour exactly; all kwargs flow through to the
official Ollama chat API untouched.
"""

from typing import Any, AsyncIterator, Union, final

from ollama import AsyncClient, ChatResponse


@final
class OllamaBackend:
    """Backend speaking Ollama's native HTTP API."""

    __slots__ = ("_client",)

    def __init__(self, host: str, timeout: float = 600.0) -> None:
        self._client: AsyncClient = AsyncClient(host=host, timeout=timeout)

    async def chat(
        self, **kwargs: Any
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        return await self._client.chat(**kwargs)

    async def close(self) -> None:
        await self._client.close()
