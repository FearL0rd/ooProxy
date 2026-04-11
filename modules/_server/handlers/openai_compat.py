"""OpenAI-compatible endpoint handlers (/v1/...).

Ollama exposes these alongside its native /api/... endpoints.
VS Code Copilot Chat uses /v1/chat/completions rather than /api/chat.
These are pure pass-through — same format on both sides — so no translation needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("ooproxy")


_ROLE_FORMAT_ERRORS = (
    "System role not supported",
    "Conversation roles must alternate",
)

_TOOL_ERRORS = (
    "tool choice requires",
    "tool_choice",
    "tools",
    "function_call",
)

_STREAM_OPTIONS_ERRORS = (
    "stream_options",
    "extra inputs are not permitted",  # pydantic-style rejection of unknown fields
)

# Seconds to wait for the first streaming byte (response headers) before giving up
# and retrying the same request as non-streaming.  Models that support streaming
# respond within a few seconds; models that silently reject it never respond at all.
_TTFB_TIMEOUT = 30.0


def _normalize_messages(body: dict) -> dict:
    """Normalize messages for models with strict role constraints (e.g. Gemma).

    1. Merge system messages into the first user message as a prefix.
    2. Collapse consecutive same-role messages by joining their content.
    """
    messages = body.get("messages", [])

    # Step 1: drop system messages (models like Gemma don't support the system role,
    # and the VS Code system prompt is boilerplate that confuses the model when inlined)
    msgs = [dict(m) for m in messages if m.get("role") != "system"]

    # Step 2: collapse consecutive same-role messages
    normalized: list = []
    for msg in msgs:
        if normalized and normalized[-1]["role"] == msg["role"]:
            prev_content = normalized[-1].get("content") or ""
            curr_content = msg.get("content") or ""
            normalized[-1]["content"] = f"{prev_content}\n\n{curr_content}" if prev_content else curr_content
        else:
            normalized.append(msg)

    return {**body, "messages": normalized}


def _is_role_format_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(pattern in msg for pattern in _ROLE_FORMAT_ERRORS)


def _is_tool_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _TOOL_ERRORS)


def _is_stream_options_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _STREAM_OPTIONS_ERRORS)


def _strip_tools(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}


def _strip_stream_options(body: dict) -> dict:
    return {k: v for k, v in body.items() if k != "stream_options"}


def _synthetic_stream(model: str, text: str) -> StreamingResponse:
    """Return a fake streaming chat completion that renders as an assistant message in VS Code."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())

    def _chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def generate():
        yield _chunk({"role": "assistant", "content": ""}).encode()
        yield _chunk({"content": text}).encode()
        yield _chunk({}, finish_reason="stop").encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _synthetic_json(model: str, text: str) -> JSONResponse:
    """Return a fake non-streaming chat completion with the given text."""
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


def _upstream_error_response(exc: Exception, model: str, streaming: bool):
    """Convert an upstream error to a response.

    404 → synthetic assistant message (model listed but not available via API)
    Other errors → HTTP error response forwarding the upstream status code.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        logger.warning("v1 model not available (404) model=%s — listed but not accessible via API", model)
        msg = (
            f"**Model `{model}` is not available.**\n\n"
            f"It appears in the model list but the API returned 404 (Not Found). "
            f"This model may require a paid tier or special access on this provider.\n\n"
            f"Please select a different model."
        )
        return _synthetic_stream(model, msg) if streaming else _synthetic_json(model, msg)

    status = 502
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if 400 <= code < 500:
            status = code
    msg = str(exc) or f"{type(exc).__name__}"
    logger.error("v1 upstream error for %s: %s", model, msg)
    return JSONResponse({"error": {"message": msg, "type": "upstream_error"}}, status_code=status)


async def v1_chat_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /v1/chat/completions — OpenAI-format pass-through."""
    body = await request.json()
    client = request.app.state.client
    streaming = body.get("stream", False)
    model = body.get("model", "?")

    logger.info("v1/chat model=%s stream=%s msgs=%d tools=%s",
                model, streaming, len(body.get("messages", [])),
                body.get("tool_choice", "none"))
    for i, msg in enumerate(body.get("messages", [])):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        preview = (content[:200] + "…") if len(content) > 200 else content
        logger.debug("  msg[%d] role=%s: %s", i, role, preview)

    if streaming:
        # Open upstream eagerly so status is known before we commit to 200 OK.
        # Retry with progressively stripped bodies on known NVIDIA NIM errors.
        # Each transformation is applied at most once; retries continue until no
        # known error pattern matches or all transformations are exhausted.
        async def _open_stream(b: dict) -> httpx.Response:
            stripped_stream_options = False
            stripped_tools = False
            normalized = False
            current = b
            while True:
                try:
                    return await client.open_stream_chat(current)
                except httpx.HTTPStatusError as exc:
                    if not stripped_stream_options and _is_stream_options_error(exc):
                        logger.info("v1 retrying without stream_options for model=%s", model)
                        current = _strip_stream_options(current)
                        stripped_stream_options = True
                    elif not stripped_tools and _is_tool_error(exc):
                        logger.info("v1 retrying without tools for model=%s", model)
                        current = _strip_tools(current)
                        stripped_tools = True
                    elif not normalized and _is_role_format_error(exc):
                        logger.info("v1 retrying with normalized messages for model=%s", model)
                        current = _normalize_messages(current)
                        normalized = True
                    else:
                        raise

        try:
            upstream = await asyncio.wait_for(_open_stream(body), timeout=_TTFB_TIMEOUT)
        except asyncio.TimeoutError:
            # Server accepted the connection but never sent response headers — this
            # model does not support SSE streaming on this backend.  Re-issue as a
            # plain non-streaming request and wrap the reply in a synthetic stream.
            logger.warning(
                "v1 TTFB timeout (>%.0fs) model=%s — falling back to non-streaming",
                _TTFB_TIMEOUT, model,
            )
            fallback = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
            fallback["stream"] = False
            try:
                data = await client.chat(fallback)
            except Exception as exc:
                return _upstream_error_response(exc, model, streaming=True)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return _synthetic_stream(model, content)
        except Exception as exc:
            return _upstream_error_response(exc, model, streaming=True)

        async def generate():
            finish = None
            usage: dict = {}
            try:
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            chunk.pop("nvext", None)
                            for choice in chunk.get("choices", []):
                                if choice.get("finish_reason"):
                                    finish = choice["finish_reason"]
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                        except json.JSONDecodeError:
                            yield f"{line}\n\n".encode()
                    else:
                        yield f"{line}\n\n".encode()
            except Exception as exc:
                logger.error("v1 stream mid-error model=%s: %s", model, exc)
            finally:
                logger.info("v1 ← model=%s finish=%s prompt=%d compl=%d",
                            model, finish or "?",
                            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                await upstream.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream")

    stripped_stream_options = False
    stripped_tools = False
    normalized = False
    current = body
    data = None
    try:
        while data is None:
            try:
                data = await client.chat(current)
            except httpx.HTTPStatusError as exc:
                if not stripped_stream_options and _is_stream_options_error(exc):
                    logger.info("v1 retrying without stream_options for model=%s", model)
                    current = _strip_stream_options(current)
                    stripped_stream_options = True
                elif not stripped_tools and _is_tool_error(exc):
                    logger.info("v1 retrying without tools for model=%s", model)
                    current = _strip_tools(current)
                    stripped_tools = True
                elif not normalized and _is_role_format_error(exc):
                    logger.info("v1 retrying with normalized messages for model=%s", model)
                    current = _normalize_messages(current)
                    normalized = True
                else:
                    return _upstream_error_response(exc, model, streaming=False)
    except Exception as exc:
        return _upstream_error_response(exc, model, streaming=False)
    usage = data.get("usage") or {}
    finish = ((data.get("choices") or [{}])[0]).get("finish_reason", "?")
    logger.info("v1 ← model=%s finish=%s prompt=%d compl=%d",
                model, finish,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    return JSONResponse(data)


async def v1_models_handler(request: Request) -> JSONResponse:
    """GET /v1/models — pass through the remote model list."""
    client = request.app.state.client
    data = await client.get_models()
    return JSONResponse(data)


async def v1_embeddings_handler(request: Request) -> JSONResponse:
    """POST /v1/embeddings — pass through to remote."""
    body = await request.json()
    client = request.app.state.client
    data = await client.embeddings(body)
    return JSONResponse(data)
