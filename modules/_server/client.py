"""Async HTTP client for the remote OpenAI-compatible backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from modules._server.config import ProxyConfig

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def _strip_vendor(data: dict) -> dict:
    """Remove vendor-specific fields (e.g. nvext) that must not reach the client."""
    data.pop("nvext", None)
    return data


class OpenAIClient:
    def __init__(self, config: ProxyConfig) -> None:
        self._base = config.url
        self._headers = {"Authorization": f"Bearer {config.key}"} if config.key else {}
        self._client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_models(self) -> dict:
        """GET /v1/models — public endpoint, no auth required."""
        r = await self._client.get(f"{self._base}/models")
        r.raise_for_status()
        return _strip_vendor(r.json())

    async def chat(self, body: dict) -> dict:
        """POST /v1/chat/completions (non-streaming)."""
        r = await self._client.post(
            f"{self._base}/chat/completions",
            json=body,
            headers=self._headers,
        )
        r.raise_for_status()
        return _strip_vendor(r.json())

    @asynccontextmanager
    async def stream_chat(self, body: dict):
        """POST /v1/chat/completions (streaming).

        Yields an async iterator of SSE lines as strings.
        Use as: async with client.stream_chat(body) as lines: ...
        """
        async with self._client.stream(
            "POST",
            f"{self._base}/chat/completions",
            json=body,
            headers={**self._headers, "Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            yield response.aiter_lines()

    async def open_stream_chat(self, body: dict) -> httpx.Response:
        """Begin a streaming POST /v1/chat/completions.

        Returns the raw httpx.Response with the stream open.
        The caller MUST call response.aclose() when done (use try/finally).
        Raises httpx.HTTPStatusError if the remote returns a non-2xx status.
        """
        request = self._client.build_request(
            "POST",
            f"{self._base}/chat/completions",
            json=body,
            headers={**self._headers, "Accept": "text/event-stream"},
        )
        response = await self._client.send(request, stream=True)
        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            raise httpx.HTTPStatusError(
                f"Remote error {response.status_code}: {error_body.decode(errors='replace')}",
                request=request,
                response=response,
            )
        return response

    async def embeddings(self, body: dict) -> dict:
        """POST /v1/embeddings."""
        r = await self._client.post(
            f"{self._base}/embeddings",
            json=body,
            headers=self._headers,
        )
        r.raise_for_status()
        return _strip_vendor(r.json())
