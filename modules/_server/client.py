"""Async HTTP client for the remote OpenAI-compatible backend."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

from modules._server.config import ProxyConfig

logger = logging.getLogger("ooproxy")

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

# HTTP status codes that are safe to retry (server-side transient failures).
_RETRY_ON_STATUS = frozenset({429, 502, 503, 504})

# Network-level exceptions that are safe to retry.
_RETRY_ON_EXC = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)

# Maximum number of retry attempts (not counting the first try).
RETRY_MAX = 3

# Base delay in seconds for exponential back-off: attempt 0→1s, 1→2s, 2→4s.
RETRY_BASE_DELAY = 1.0


def _backoff_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Return how long to wait before the next attempt.

    Respects the upstream ``Retry-After`` header when present.
    Otherwise uses exponential back-off with ±10 % jitter.
    """
    if response is not None:
        after = response.headers.get("Retry-After", "")
        try:
            return max(0.0, float(after))
        except ValueError:
            pass
    base = RETRY_BASE_DELAY * (2 ** attempt)
    jitter = base * 0.1 * (2 * random.random() - 1)
    return base + jitter


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_body(direction: str, body: dict) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    raw = json.dumps(body, ensure_ascii=False)
    preview = raw[:600] + ("…" if len(raw) > 600 else "")
    logger.debug("upstream %s %s", direction, preview)


# ---------------------------------------------------------------------------
# Vendor-field stripping
# ---------------------------------------------------------------------------

# Top-level keys injected by specific providers that must not leak to clients.
_VENDOR_KEYS = frozenset({"nvext", "x_groq", "x_request_id"})


def _strip_vendor(data: dict) -> dict:
    """Remove provider-specific top-level keys that must not reach the client."""
    for key in _VENDOR_KEYS:
        data.pop(key, None)
    return data


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenAIClient:
    def __init__(self, config: ProxyConfig) -> None:
        self._base = config.url
        # Always include auth headers — some providers require them on /models too.
        self._headers: dict[str, str] = (
            {"Authorization": f"Bearer {config.key}"} if config.key else {}
        )
        self._client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal unified request methods
    # ------------------------------------------------------------------

    async def _post_json(self, path: str, body: dict, extra_headers: dict | None = None) -> dict:
        """POST *path* with JSON body, retry on transient failures, return parsed JSON.

        Centralises: auth headers, Accept header, request/response logging,
        latency measurement, retry/back-off, and vendor-field stripping.
        """
        url = f"{self._base}/{path.lstrip('/')}"
        headers = {**self._headers, "Accept": "application/json", **(extra_headers or {})}
        _log_body("→", body)
        last_exc: Exception | None = None
        for attempt in range(RETRY_MAX + 1):
            t0 = time.perf_counter()
            try:
                r = await self._client.post(url, json=body, headers=headers)
                latency_ms = (time.perf_counter() - t0) * 1000
                if r.status_code >= 400:
                    r.raise_for_status()   # raises HTTPStatusError
                data = _strip_vendor(r.json())
                logger.debug("upstream ← %.0fms status=%d", latency_ms, r.status_code)
                _log_body("←", data)
                return data
            except httpx.HTTPStatusError as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_exc = exc
                if attempt < RETRY_MAX and exc.response.status_code in _RETRY_ON_STATUS:
                    delay = _backoff_delay(attempt, exc.response)
                    logger.warning(
                        "upstream POST %s status=%d attempt=%d/%.0fs → retry in %.1fs",
                        path, exc.response.status_code, attempt + 1, latency_ms, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            except _RETRY_ON_EXC as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_exc = exc
                if attempt < RETRY_MAX:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "upstream POST %s %s attempt=%d/%.0fms → retry in %.1fs",
                        path, type(exc).__name__, attempt + 1, latency_ms, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]  — loop always raises before this

    async def _open_stream(self, path: str, body: dict, extra_headers: dict | None = None) -> httpx.Response:
        """Open a streaming POST to *path*, retry on transient failures before the stream starts.

        Returns the raw ``httpx.Response`` with the stream open.
        The caller MUST call ``response.aclose()`` when done.
        Once the response headers are received the stream is live — retries only
        happen on connection/header-level failures, not mid-stream.
        """
        url = f"{self._base}/{path.lstrip('/')}"
        headers = {**self._headers, "Accept": "text/event-stream", **(extra_headers or {})}
        _log_body("→", body)
        last_exc: Exception | None = None
        for attempt in range(RETRY_MAX + 1):
            t0 = time.perf_counter()
            try:
                request = self._client.build_request("POST", url, json=body, headers=headers)
                response = await self._client.send(request, stream=True)
                latency_ms = (time.perf_counter() - t0) * 1000
                if response.status_code >= 400:
                    error_body = await response.aread()
                    await response.aclose()
                    raise httpx.HTTPStatusError(
                        f"Remote error {response.status_code}: {error_body.decode(errors='replace')}",
                        request=request,
                        response=response,
                    )
                logger.debug("upstream stream ← %.0fms status=%d", latency_ms, response.status_code)
                return response
            except httpx.HTTPStatusError as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_exc = exc
                if attempt < RETRY_MAX and exc.response.status_code in _RETRY_ON_STATUS:
                    delay = _backoff_delay(attempt, exc.response)
                    logger.warning(
                        "upstream stream %s status=%d attempt=%d/%.0fms → retry in %.1fs",
                        path, exc.response.status_code, attempt + 1, latency_ms, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            except _RETRY_ON_EXC as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_exc = exc
                if attempt < RETRY_MAX:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "upstream stream %s %s attempt=%d/%.0fms → retry in %.1fs",
                        path, type(exc).__name__, attempt + 1, latency_ms, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_models(self) -> dict:
        """GET /v1/models."""
        url = f"{self._base}/models"
        headers = {**self._headers, "Accept": "application/json"}
        t0 = time.perf_counter()
        r = await self._client.get(url, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        logger.debug("upstream GET /models ← %.0fms", latency_ms)
        return _strip_vendor(r.json())

    async def chat(self, body: dict) -> dict:
        """POST /v1/chat/completions (non-streaming)."""
        return await self._post_json("chat/completions", body)

    @asynccontextmanager
    async def stream_chat(self, body: dict) -> AsyncIterator:
        """POST /v1/chat/completions (streaming) as a context manager.

        Yields an async iterator of SSE lines as strings.
        Use as: ``async with client.stream_chat(body) as lines: ...``
        """
        response = await self._open_stream("chat/completions", body)
        try:
            yield response.aiter_lines()
        finally:
            await response.aclose()

    async def open_stream_chat(self, body: dict) -> httpx.Response:
        """Begin a streaming POST /v1/chat/completions.

        Returns the raw ``httpx.Response`` with the stream open.
        The caller MUST call ``response.aclose()`` when done (use try/finally).
        """
        return await self._open_stream("chat/completions", body)

    async def embeddings(self, body: dict) -> dict:
        """POST /v1/embeddings."""
        return await self._post_json("embeddings", body)
