"""Async HTTP client for the remote OpenAI-compatible backend."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import urljoin

import httpx

from modules._server.config import ProxyConfig
from modules._server.endpoint_profiles import EndpointProfile, resolve_endpoint_profile

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


def _preview_text(text: str, limit: int = 600) -> str:
    compact = text.replace("\r", "\\r").replace("\n", "\\n")
    return compact[:limit] + ("…" if len(compact) > limit else "")


def _log_stream_line(line: str | bytes) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if isinstance(line, bytes):
        text = line.decode("utf-8", errors="replace")
    else:
        text = line
    logger.debug("upstream SSE ← %s", _preview_text(text))


class _TracedStreamResponse:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def aiter_lines(self):
        async for line in self._response.aiter_lines():
            _log_stream_line(line)
            yield line

    async def aclose(self) -> None:
        await self._response.aclose()

    def __getattr__(self, name: str):
        return getattr(self._response, name)


# ---------------------------------------------------------------------------
# Vendor-field stripping
# ---------------------------------------------------------------------------

# Top-level keys injected by specific providers that must not leak to clients.
_VENDOR_KEYS = frozenset({"nvext", "x_groq", "x_request_id"})


def _strip_vendor(data: object) -> object:
    """Remove provider-specific top-level keys that must not reach the client."""
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    for key in _VENDOR_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _normalize_model_entry(entry: dict) -> dict:
    normalized = dict(entry)
    normalized.setdefault("object", "model")
    return normalized


def _normalize_ollama_tags_payload(data: dict) -> dict:
    items = []
    for entry in data.get("models", []):
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model") or entry.get("name") or "").strip()
        if not model_id:
            continue
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        normalized = {
            "id": model_id,
            "object": "model",
            "created": None,
            "owned_by": "ollama",
        }
        optional_fields = {
            "modified_at": entry.get("modified_at"),
            "digest": entry.get("digest"),
            "family": details.get("family"),
            "families": details.get("families"),
            "format": details.get("format"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "parent_model": details.get("parent_model"),
            "capabilities": entry.get("capabilities"),
        }
        for key, value in optional_fields.items():
            if value not in (None, "", []):
                normalized[key] = value
        items.append(normalized)
    return {"object": "list", "data": items}


def _parse_iso_timestamp(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return int(datetime.fromisoformat(text).timestamp())
    except ValueError:
        return None


def _path_text(entry: object, path: str) -> str:
    value = _extract_json_path(entry, path)
    return str(value).strip() if value not in (None, "") else ""


def _path_timestamp(entry: object, path: str) -> int | None:
    value = _extract_json_path(entry, path)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return _parse_iso_timestamp(value)


def _path_value(entry: object, path: str) -> object:
    return _extract_json_path(entry, path)


def _normalize_profile_object_list_payload(data: dict, profile: EndpointProfile) -> dict:
    items_path = profile.models_items_path or "items"
    raw_items = _extract_json_path(data, items_path)
    if not isinstance(raw_items, list):
        raise TypeError(f"Unsupported model-list payload at {items_path!r}: {type(raw_items).__name__}")

    items = []
    id_path = profile.models_fields.get("id", "")
    created_path = profile.models_fields.get("created", "")
    modified_at_path = profile.models_fields.get("modified_at", "")
    context_length_path = profile.models_fields.get("context_length", "")
    parent_model_path = profile.models_fields.get("parent_model", "")

    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        model_id = _path_text(entry, id_path)
        if not model_id:
            continue

        capabilities: list[str] = []
        capability_cfg = profile.models_capabilities or {}
        embedding_when = capability_cfg.get("embedding_when")
        if isinstance(embedding_when, dict):
            kind_value = _path_text(entry, str(embedding_when.get("path") or ""))
            expected_value = str(embedding_when.get("equals") or "")
            if kind_value and kind_value == expected_value:
                capabilities.append("embedding")

        completion_when = capability_cfg.get("completion_when_any_present")
        if not capabilities and isinstance(completion_when, list):
            if any(_path_value(entry, str(path)) not in (None, "", [], {}) for path in completion_when):
                capabilities.append("completion")

        tools_when = capability_cfg.get("tools_when_truthy")
        if isinstance(tools_when, str) and bool(_path_value(entry, tools_when)):
            if "completion" not in capabilities:
                capabilities.append("completion")
            capabilities.append("tools")

        if not capabilities and capability_cfg.get("default_embedding"):
            capabilities.append("embedding")

        normalized = {
            "id": model_id,
            "object": "model",
        }
        created = _path_timestamp(entry, created_path) if created_path else None
        if created is not None:
            normalized["created"] = created
        if profile.models_owned_by:
            normalized["owned_by"] = profile.models_owned_by
        optional_fields = {
            "modified_at": _path_value(entry, modified_at_path) if modified_at_path else None,
            "context_length": _path_value(entry, context_length_path) if context_length_path else None,
            "capabilities": capabilities,
            "parent_model": _path_value(entry, parent_model_path) if parent_model_path else None,
        }
        for key, value in optional_fields.items():
            if value not in (None, "", []):
                normalized[key] = value
        items.append(normalized)
    return {"object": "list", "data": items}


def _extract_json_path(data: object, path: str) -> object | None:
    current = data
    if not path.strip():
        return current
    for segment in path.split("."):
        segment = segment.strip()
        if not segment:
            continue
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _normalize_models_payload(data: object, profile: EndpointProfile | None = None) -> dict:
    profile_format = (profile.models_format if profile else "").strip().lower()

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return {
            **data,
            "data": [_normalize_model_entry(entry) for entry in data.get("data", []) if isinstance(entry, dict)],
        }

    if profile_format == "ollama_tags" and isinstance(data, dict):
        return _normalize_ollama_tags_payload(data)

    if profile_format == "object_list" and isinstance(data, dict) and profile is not None:
        return _normalize_profile_object_list_payload(data, profile)

    if profile_format == "array" and isinstance(data, list):
        return {"object": "list", "data": [_normalize_model_entry(entry) for entry in data if isinstance(entry, dict)]}

    if isinstance(data, list):
        return {"object": "list", "data": [_normalize_model_entry(entry) for entry in data if isinstance(entry, dict)]}

    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return _normalize_ollama_tags_payload(data)

    raise TypeError(f"Unsupported model-list payload: {type(data).__name__}")

def _decode_json_response(response: httpx.Response, *, path: str) -> object:
    location = response.headers.get("location", "").strip()
    if 300 <= response.status_code < 400:
        detail = f"upstream {path} redirected"
        if location:
            detail = f"{detail} to {location}"
        raise RuntimeError(f"{detail}; check the API base URL and credentials")

    content_type = response.headers.get("content-type", "")
    if "json" not in content_type.lower():
        preview = response.text.strip()[:120]
        extra = f"; received {content_type or 'unknown content-type'}"
        if preview:
            extra = f"{extra}: {preview}"
        raise RuntimeError(f"upstream {path} did not return JSON{extra}")

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"upstream {path} returned invalid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenAIClient:
    def __init__(self, config: ProxyConfig) -> None:
        self._base = config.url
        self.endpoint_profile = resolve_endpoint_profile(config.url)
        # Always include auth headers — some providers require them on /models too.
        self._headers: dict[str, str] = (
            {"Authorization": f"Bearer {config.key}"} if config.key else {}
        )
        # Allow per-endpoint override of default httpx timeouts
        if self.endpoint_profile is None:
            timeout = _TIMEOUT
            logger.debug("no endpoint profile: using global timeouts %s", timeout)
        else:
            connect = self.endpoint_profile.timeout_connect if getattr(self.endpoint_profile, "timeout_connect", None) is not None else 10.0
            read = self.endpoint_profile.timeout_read if getattr(self.endpoint_profile, "timeout_read", None) is not None else 180.0
            write = self.endpoint_profile.timeout_write if getattr(self.endpoint_profile, "timeout_write", None) is not None else 30.0
            pool = self.endpoint_profile.timeout_pool if getattr(self.endpoint_profile, "timeout_pool", None) is not None else 10.0
            timeout = httpx.Timeout(connect=connect, read=read, write=write, pool=pool)
            logger.info(
                "endpoint profile: using %s for %s — timeouts(connect=%.0fs read=%.0fs write=%.0fs pool=%.0f)",
                self.endpoint_profile.id,
                self._base,
                connect,
                read,
                write,
                pool,
            )
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    def _url_for_path(self, path: str) -> str:
        return urljoin(f"{self._base.rstrip('/')}/", path)

    async def _request_json(self, method: str, path: str) -> object:
        headers = {**self._headers, "Accept": "application/json"}
        t0 = time.perf_counter()
        response = await self._client.request(method, self._url_for_path(path), headers=headers, follow_redirects=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        if 300 <= response.status_code < 400:
            _decode_json_response(response, path=path)
        response.raise_for_status()
        logger.debug("upstream %s %s ← %.0fms", method.upper(), path, latency_ms)
        return _decode_json_response(response, path=path)

    async def _resolve_models_variables(self) -> dict[str, str]:
        profile = self.endpoint_profile
        if profile is None or not profile.models_variables:
            return {}

        resolved: dict[str, str] = {}
        for name, config in profile.models_variables.items():
            method = str(config.get("method") or "GET").upper()
            path = str(config.get("path") or "").strip()
            json_path = str(config.get("json_path") or "").strip()
            strip_prefix = str(config.get("strip_prefix") or "")
            if not path:
                raise RuntimeError(f"endpoint profile variable {name!r} is missing a path")

            payload = await self._request_json(method, path)
            raw_value = _extract_json_path(payload, json_path)
            if raw_value is None:
                raise RuntimeError(f"upstream {path} did not provide {name!r} at {json_path!r}")

            value = str(raw_value).strip()
            if strip_prefix and value.startswith(strip_prefix):
                value = value[len(strip_prefix):]
            if not value:
                raise RuntimeError(f"upstream {path} returned an empty value for {name!r}")
            resolved[name] = value

        return resolved

    async def probe_ready(self) -> tuple[bool, str | None]:
        profile = self.endpoint_profile
        if profile is None or profile.health_mode == "internal-ready":
            return True, None
        if profile.health_mode != "http" or not profile.health_path:
            return True, None

        headers = {**self._headers, "Accept": "application/json"}
        try:
            response = await self._client.request(
                profile.health_method or "GET",
                self._url_for_path(profile.health_path),
                headers=headers,
            )
        except Exception as exc:
            return False, f"upstream health probe failed for {profile.health_path}: {exc}"

        if response.status_code >= 400:
            return False, f"upstream health probe returned {response.status_code} for {profile.health_path}"
        return True, None

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
        url = self._url_for_path(path)
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
        url = self._url_for_path(path)
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
        """GET the upstream model-list endpoint and normalize it to OpenAI format."""
        method = self.endpoint_profile.models_method if self.endpoint_profile else "GET"
        path = self.endpoint_profile.models_path if self.endpoint_profile else "models"
        if self.endpoint_profile and self.endpoint_profile.models_variables:
            path = path.format(**(await self._resolve_models_variables()))

        payload = await self._request_json(method, path)
        return _normalize_models_payload(_strip_vendor(payload), self.endpoint_profile)

    async def chat(self, body: dict) -> dict:
        """POST /v1/chat/completions (non-streaming)."""
        return await self._post_json("chat/completions", body)

    @asynccontextmanager
    async def stream_chat(self, body: dict) -> AsyncIterator:
        """POST /v1/chat/completions (streaming) as a context manager.

        Yields an async iterator of SSE lines as strings.
        Use as: ``async with client.stream_chat(body) as lines: ...``
        """
        response = _TracedStreamResponse(await self._open_stream("chat/completions", body))
        try:
            yield response.aiter_lines()
        finally:
            await response.aclose()

    async def open_stream_chat(self, body: dict) -> httpx.Response:
        """Begin a streaming POST /v1/chat/completions.

        Returns the raw ``httpx.Response`` with the stream open.
        The caller MUST call ``response.aclose()`` when done (use try/finally).
        """
        return _TracedStreamResponse(await self._open_stream("chat/completions", body))

    async def embeddings(self, body: dict) -> dict:
        """POST /v1/embeddings."""
        return await self._post_json("embeddings", body)
