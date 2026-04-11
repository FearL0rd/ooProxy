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
from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("ooproxy")

from modules._server.translate.request import responses_to_openai_chat
from modules._server.translate.response import openai_chat_to_responses


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


def _responses_store(request: Request) -> dict[str, list[dict]]:
    store = getattr(request.app.state, "responses_store", None)
    if store is None:
        store = {}
        request.app.state.responses_store = store
    return store


def _remember_response(request: Request, response_id: str, messages: list[dict]) -> None:
    store = _responses_store(request)
    store[response_id] = [dict(message) for message in messages]
    while len(store) > 200:
        oldest = next(iter(store))
        del store[oldest]


def _response_event(event_type: str, payload: dict) -> bytes:
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body)}\n\n".encode()


def _response_error_event(message: str, *, code: str = "upstream_error", sequence_number: int = 1) -> bytes:
    return _response_event("error", {
        "code": code,
        "message": message,
        "param": None,
        "sequence_number": sequence_number,
    })


async def _chat_sse_to_responses_sse(
    upstream: httpx.Response,
    *,
    request_body: dict,
    response_id: str,
    previous_response_id: str | None,
    input_messages: list[dict],
    request: Request,
) -> AsyncIterator[bytes]:
    created_at = int(time.time())
    response_stub = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": None,
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "model": request_body["model"],
        "output": [],
        "parallel_tool_calls": bool(request_body.get("parallel_tool_calls", True)),
        "previous_response_id": previous_response_id,
        "reasoning": request_body.get("reasoning") or {"effort": None, "summary": None},
        "store": bool(request_body.get("store", True)),
        "temperature": request_body.get("temperature", 1),
        "text": request_body.get("text") or {"format": {"type": "text"}},
        "tool_choice": request_body.get("tool_choice", "auto"),
        "tools": request_body.get("tools") or [],
        "top_p": request_body.get("top_p", 1),
        "truncation": request_body.get("truncation", "disabled"),
        "usage": None,
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata") or {},
    }
    sequence_number = 1
    message_id = f"msg_{uuid.uuid4().hex}"
    tool_item_ids: dict[int, str] = {}
    accumulated_text = ""
    finish_reason = "stop"
    usage = None
    finalized_tool_calls: list[dict] = []
    yielded_message_start = False
    tool_buffers: dict[int, dict[str, str]] = {}

    yield _response_event("response.created", {"response": response_stub, "sequence_number": sequence_number})
    sequence_number += 1
    yield _response_event("response.in_progress", {"response": response_stub, "sequence_number": sequence_number})
    sequence_number += 1
    yield _response_event("response.output_item.added", {
        "output_index": 0,
        "item": {"id": message_id, "status": "in_progress", "type": "message", "role": "assistant", "content": []},
        "sequence_number": sequence_number,
    })
    sequence_number += 1
    yield _response_event("response.content_part.added", {
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
        "sequence_number": sequence_number,
    })
    sequence_number += 1
    yielded_message_start = True

    try:
        async for line in upstream.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage") and not chunk.get("choices"):
                usage = chunk["usage"]
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            content = delta.get("content")
            if isinstance(content, str) and content:
                accumulated_text += content
                yield _response_event("response.output_text.delta", {
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": content,
                    "sequence_number": sequence_number,
                })
                sequence_number += 1

            for tool_call in delta.get("tool_calls") or []:
                index = tool_call.get("index", 0)
                item_id = tool_item_ids.setdefault(index, f"fc_{uuid.uuid4().hex}")
                tool_buffers.setdefault(index, {"name": "", "arguments": ""})
                function = tool_call.get("function") or {}
                if function.get("name"):
                    tool_buffers[index]["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    tool_buffers[index]["arguments"] += function["arguments"]
                    yield _response_event("response.function_call_arguments.delta", {
                        "item_id": item_id,
                        "output_index": index + 1,
                        "delta": function["arguments"],
                        "sequence_number": sequence_number,
                    })
                    sequence_number += 1

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    except Exception as exc:
        logger.error("responses stream mid-error model=%s: %s", request_body.get("model", "?"), exc)
        yield _response_error_event(str(exc), sequence_number=sequence_number)
        sequence_number += 1
    finally:
        await upstream.aclose()

    assistant_message = {"role": "assistant", "content": accumulated_text}
    output_items = []
    if yielded_message_start:
        yield _response_event("response.output_text.done", {
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "text": accumulated_text,
            "sequence_number": sequence_number,
        })
        sequence_number += 1
        yield _response_event("response.content_part.done", {
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": accumulated_text, "annotations": []},
            "sequence_number": sequence_number,
        })
        sequence_number += 1
        message_item = {
            "id": message_id,
            "status": "completed",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": accumulated_text, "annotations": []}],
        }
        yield _response_event("response.output_item.done", {
            "output_index": 0,
            "item": message_item,
            "sequence_number": sequence_number,
        })
        sequence_number += 1
        output_items.append(message_item)

    if tool_buffers:
        tool_calls = []
        for index in sorted(tool_buffers):
            item_id = tool_item_ids[index]
            tool_item = {
                "id": item_id,
                "type": "function_call",
                "call_id": item_id,
                "name": tool_buffers[index]["name"] or "unknown_tool",
                "arguments": tool_buffers[index]["arguments"],
                "status": "completed",
            }
            yield _response_event("response.output_item.added", {
                "output_index": index + 1,
                "item": tool_item,
                "sequence_number": sequence_number,
            })
            sequence_number += 1
            yield _response_event("response.function_call_arguments.done", {
                "item_id": item_id,
                "output_index": index + 1,
                "name": tool_item["name"],
                "arguments": tool_item["arguments"],
                "sequence_number": sequence_number,
            })
            sequence_number += 1
            yield _response_event("response.output_item.done", {
                "output_index": index + 1,
                "item": tool_item,
                "sequence_number": sequence_number,
            })
            sequence_number += 1
            output_items.append(tool_item)
            tool_calls.append({
                "id": item_id,
                "type": "function",
                "function": {
                    "name": tool_item["name"],
                    "arguments": tool_item["arguments"],
                },
            })
        assistant_message["tool_calls"] = tool_calls
        finalized_tool_calls = tool_calls

    usage_payload = None
    if usage:
        usage_payload = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        }

    status = "completed" if finish_reason not in {"length", "content_filter"} else "incomplete"
    completed_response = {
        **response_stub,
        "completed_at": int(time.time()),
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if finish_reason == "length" else None,
        "output": output_items,
        "output_text": accumulated_text,
        "usage": usage_payload,
    }
    _remember_response(request, response_id, [*input_messages, assistant_message])
    yield _response_event(
        "response.completed" if status == "completed" else "response.incomplete",
        {"response": completed_response, "sequence_number": sequence_number},
    )


def _responses_error_response(exc: Exception) -> JSONResponse:
    status = 502
    message = str(exc) or type(exc).__name__
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if 400 <= code < 500:
            status = code
    return JSONResponse({"error": {"message": message, "type": "upstream_error"}}, status_code=status)


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


async def v1_responses_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /v1/responses — translate to /v1/chat/completions and adapt the result."""
    body = await request.json()
    client = request.app.state.client
    previous_response_id = body.get("previous_response_id")
    previous_messages = None
    if previous_response_id:
        previous_messages = _responses_store(request).get(previous_response_id)

    chat_body, input_messages = responses_to_openai_chat(body, previous_messages=previous_messages)
    model = body.get("model", "?")
    response_id = f"resp_{uuid.uuid4().hex}"

    logger.info("v1/responses model=%s stream=%s prev=%s input_items=%s",
                model, bool(body.get("stream")), previous_response_id or "-",
                type(body.get("input")).__name__)

    if body.get("stream"):
        try:
            upstream = await asyncio.wait_for(client.open_stream_chat(chat_body), timeout=_TTFB_TIMEOUT)
        except asyncio.TimeoutError:
            fallback = {k: v for k, v in chat_body.items() if k not in ("stream", "stream_options")}
            fallback["stream"] = False
            try:
                data = await client.chat(fallback)
            except Exception as exc:
                async def error_stream():
                    yield _response_error_event(str(exc))
                return StreamingResponse(error_stream(), media_type="text/event-stream")

            response_payload, assistant_messages = openai_chat_to_responses(
                data,
                body,
                response_id,
                previous_response_id=previous_response_id,
            )
            _remember_response(request, response_id, [*input_messages, *assistant_messages])

            async def synthetic_stream():
                seq = 1
                yield _response_event("response.created", {"response": {**response_payload, "status": "in_progress", "completed_at": None, "usage": None, "output": []}, "sequence_number": seq})
                seq += 1
                for item_index, item in enumerate(response_payload.get("output") or []):
                    yield _response_event("response.output_item.added", {"output_index": item_index, "item": item, "sequence_number": seq})
                    seq += 1
                    if item.get("type") == "message":
                        content = ((item.get("content") or [{}])[0]).get("text", "")
                        yield _response_event("response.content_part.added", {"item_id": item["id"], "output_index": item_index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}, "sequence_number": seq})
                        seq += 1
                        if content:
                            yield _response_event("response.output_text.delta", {"item_id": item["id"], "output_index": item_index, "content_index": 0, "delta": content, "sequence_number": seq})
                            seq += 1
                            yield _response_event("response.output_text.done", {"item_id": item["id"], "output_index": item_index, "content_index": 0, "text": content, "sequence_number": seq})
                            seq += 1
                            yield _response_event("response.content_part.done", {"item_id": item["id"], "output_index": item_index, "content_index": 0, "part": {"type": "output_text", "text": content, "annotations": []}, "sequence_number": seq})
                            seq += 1
                    elif item.get("type") == "function_call":
                        yield _response_event("response.function_call_arguments.done", {"item_id": item["id"], "output_index": item_index, "name": item.get("name", "unknown_tool"), "arguments": item.get("arguments", ""), "sequence_number": seq})
                        seq += 1
                    yield _response_event("response.output_item.done", {"output_index": item_index, "item": item, "sequence_number": seq})
                    seq += 1
                yield _response_event("response.completed", {"response": response_payload, "sequence_number": seq})

            return StreamingResponse(synthetic_stream(), media_type="text/event-stream")
        except Exception as exc:
            async def error_stream():
                yield _response_error_event(str(exc))
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        return StreamingResponse(
            _chat_sse_to_responses_sse(
                upstream,
                request_body=body,
                response_id=response_id,
                previous_response_id=previous_response_id,
                input_messages=input_messages,
                request=request,
            ),
            media_type="text/event-stream",
        )

    try:
        data = await client.chat(chat_body)
    except Exception as exc:
        return _responses_error_response(exc)

    response_payload, assistant_messages = openai_chat_to_responses(
        data,
        body,
        response_id,
        previous_response_id=previous_response_id,
    )
    _remember_response(request, response_id, [*input_messages, *assistant_messages])
    return JSONResponse(response_payload)
