"""OpenAI-compatible endpoint handlers (/v1/...).

Ollama exposes these alongside its native /api/... endpoints.
VS Code Copilot Chat uses /v1/chat/completions rather than /api/chat.
These are pure pass-through — same format on both sides — so no translation needed.
"""

from __future__ import annotations

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


def _strip_tools(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}


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
    logger.error("v1 upstream error for %s: %s", model, exc)
    return JSONResponse({"error": {"message": str(exc), "type": "upstream_error"}}, status_code=status)


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
        async def _open_stream(b: dict):
            try:
                return await client.open_stream_chat(b)
            except httpx.HTTPStatusError as exc:
                if _is_tool_error(exc):
                    logger.info("v1 retrying without tools for model=%s", model)
                    b = _strip_tools(b)
                    try:
                        return await client.open_stream_chat(b), b
                    except httpx.HTTPStatusError as exc2:
                        if _is_role_format_error(exc2):
                            logger.info("v1 retrying with normalized messages for model=%s", model)
                            return await client.open_stream_chat(_normalize_messages(b)), _normalize_messages(b)
                        raise
                elif _is_role_format_error(exc):
                    logger.info("v1 retrying with normalized messages for model=%s", model)
                    return await client.open_stream_chat(_normalize_messages(b)), _normalize_messages(b)
                raise

        try:
            result = await _open_stream(body)
            upstream = result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            return _upstream_error_response(exc, model, streaming=True)

        async def generate():
            try:
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            chunk.pop("nvext", None)
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                        except json.JSONDecodeError:
                            yield f"{line}\n\n".encode()
                    else:
                        yield f"{line}\n\n".encode()
            except Exception as exc:
                logger.error("v1 stream mid-error model=%s: %s", model, exc)
            finally:
                await upstream.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream")

    try:
        data = await client.chat(body)
    except httpx.HTTPStatusError as exc:
        if _is_tool_error(exc):
            logger.info("v1 retrying without tools for model=%s", model)
            body = _strip_tools(body)
            try:
                data = await client.chat(body)
            except httpx.HTTPStatusError as exc2:
                if _is_role_format_error(exc2):
                    logger.info("v1 retrying with normalized messages for model=%s", model)
                    data = await client.chat(_normalize_messages(body))
                else:
                    raise exc2
        elif _is_role_format_error(exc):
            logger.info("v1 retrying with normalized messages for model=%s", model)
            data = await client.chat(_normalize_messages(body))
        else:
            return _upstream_error_response(exc, model, streaming=False)
    except Exception as exc:
        return _upstream_error_response(exc, model, streaming=False)
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
