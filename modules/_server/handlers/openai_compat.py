"""OpenAI-compatible endpoint handlers (/v1/...).

Ollama exposes these alongside its native /api/... endpoints.
VS Code Copilot Chat uses /v1/chat/completions rather than /api/chat.
These are pure pass-through — same format on both sides — so no translation needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("ooproxy")

from modules._server.upstream_errors import (
    assistant_error_text,
    iter_openai_error_stream,
    synthetic_anthropic_message,
    synthetic_openai_chat_completion,
    synthetic_responses_payload,
)
from modules._server.translate.request import responses_to_openai_chat
from modules._server.translate.request import anthropic_messages_to_openai_chat
from modules._server.translate.request import anthropic_direct_display_tool_reply
from modules._server.translate.request import direct_display_tool_reply
from modules._server.translate.request import sanitize_openai_chat_body
from modules._server.translate.response import openai_chat_to_anthropic_message, openai_chat_to_responses


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


def _should_disable_anthropic_tools(body: dict) -> bool:
    mode = os.environ.get("OOPROXY_ANTHROPIC_TOOL_PASSTHROUGH", "auto").strip().lower()
    if mode in {"1", "true", "on", "enabled", "allow", "passthrough"}:
        return False
    if mode in {"0", "false", "off", "disabled", "deny", "disable"}:
        return True

    if not body.get("tools"):
        return False

    model = str(body.get("model") or "").lower()
    if "claude" in model or "anthropic" in model:
        return False

    latest_user_text = _anthropic_latest_user_text(body)
    return _looks_like_trivial_greeting(latest_user_text)


def _anthropic_latest_user_text(body: dict) -> str:
    messages = body.get("messages") or []
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip().lower()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts).strip().lower()
    return ""


def _looks_like_trivial_greeting(text: str) -> bool:
    if not text:
        return False

    normalized = " ".join(text.split())
    greeting_phrases = {
        "hi",
        "hello",
        "hello there",
        "hey",
        "hey there",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "test",
        "testing",
    }
    if normalized in greeting_phrases:
        return True

    actionable_tokens = (
        "list",
        "show",
        "find",
        "read",
        "open",
        "run",
        "execute",
        "search",
        "grep",
        "edit",
        "write",
        "create",
        "delete",
        "remove",
        "rename",
        "move",
        "copy",
        "fix",
        "debug",
        "files",
        "folder",
        "directory",
        "cwd",
        "current folder",
        "current directory",
    )
    return not any(token in normalized for token in actionable_tokens)


def _apply_cached_flags(body: dict, flags: dict[str, bool]) -> dict:
    """Pre-apply known behavioral quirks from the cache so we skip trial-and-error."""
    current = body
    if flags.get("strip_stream_options"):
        current = _strip_stream_options(current)
    if flags.get("strip_tools"):
        current = _strip_tools(current)
    if flags.get("normalize_messages"):
        current = _normalize_messages(current)
    return current


def _effective_behavior_flags(
    *,
    behavior,
    base_url: str,
    model: str,
    endpoint_profile=None,
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    if endpoint_profile is not None:
        flags.update(getattr(endpoint_profile, "behavior_defaults", {}) or {})
    if behavior:
        flags.update(behavior.get_flags(base_url, model))
    return flags


def _profile_disables_streaming(endpoint_profile) -> bool:
    if endpoint_profile is None:
        return False
    mode = str(getattr(endpoint_profile, "chat_streaming", "sse") or "sse").strip().lower()
    return mode not in {"sse", "openai-sse", "supported", "trial"}


def _apply_profile_chat_defaults(body: dict, endpoint_profile=None) -> dict:
    if endpoint_profile is None:
        return body

    current = body
    tools_mode = str(getattr(endpoint_profile, "chat_tools", "trial") or "trial").strip().lower()
    if tools_mode in {"off", "disabled", "unsupported", "none", "never", "native"}:
        current = _strip_tools(current)

    system_prompt_mode = str(
        getattr(endpoint_profile, "chat_system_prompt", "supported") or "supported"
    ).strip().lower()
    if system_prompt_mode in {"unsupported", "normalize", "inline-user", "merge"}:
        current = _normalize_messages(current)

    if _profile_disables_streaming(endpoint_profile) and current.get("stream"):
        current = {k: v for k, v in current.items() if k not in ("stream", "stream_options")}
        current["stream"] = False

    return current


async def _open_stream_with_retries(
    client, body: dict, model: str, behavior=None, base_url: str = "", endpoint_profile=None
) -> tuple[httpx.Response, dict]:
    flags = _effective_behavior_flags(
        behavior=behavior,
        base_url=base_url,
        model=model,
        endpoint_profile=endpoint_profile,
    )
    current = _apply_cached_flags(_apply_profile_chat_defaults(body, endpoint_profile), flags)
    stripped_stream_options = flags.get("strip_stream_options", False)
    stripped_tools = flags.get("strip_tools", False)
    normalized = flags.get("normalize_messages", False)
    while True:
        try:
            return await client.open_stream_chat(current), current
        except httpx.HTTPStatusError as exc:
            if not stripped_stream_options and _is_stream_options_error(exc):
                logger.info("v1 retrying without stream_options for model=%s", model)
                current = _strip_stream_options(current)
                stripped_stream_options = True
                if behavior:
                    await behavior.record(base_url, model, "strip_stream_options")
            elif not stripped_tools and _is_tool_error(exc):
                logger.info("v1 retrying without tools for model=%s", model)
                current = _strip_tools(current)
                stripped_tools = True
                if behavior:
                    await behavior.record(base_url, model, "strip_tools")
            elif not normalized and _is_role_format_error(exc):
                logger.info("v1 retrying with normalized messages for model=%s", model)
                current = _normalize_messages(current)
                normalized = True
                if behavior:
                    await behavior.record(base_url, model, "normalize_messages")
            else:
                raise


async def _chat_with_retries(
    client, body: dict, model: str, behavior=None, base_url: str = "", endpoint_profile=None
) -> tuple[dict, dict]:
    flags = _effective_behavior_flags(
        behavior=behavior,
        base_url=base_url,
        model=model,
        endpoint_profile=endpoint_profile,
    )
    current = _apply_cached_flags(_apply_profile_chat_defaults(body, endpoint_profile), flags)
    stripped_stream_options = flags.get("strip_stream_options", False)
    stripped_tools = flags.get("strip_tools", False)
    normalized = flags.get("normalize_messages", False)
    while True:
        try:
            return await client.chat(current), current
        except httpx.HTTPStatusError as exc:
            if not stripped_stream_options and _is_stream_options_error(exc):
                logger.info("v1 retrying without stream_options for model=%s", model)
                current = _strip_stream_options(current)
                stripped_stream_options = True
                if behavior:
                    await behavior.record(base_url, model, "strip_stream_options")
            elif not stripped_tools and _is_tool_error(exc):
                logger.info("v1 retrying without tools for model=%s", model)
                current = _strip_tools(current)
                stripped_tools = True
                if behavior:
                    await behavior.record(base_url, model, "strip_tools")
            elif not normalized and _is_role_format_error(exc):
                logger.info("v1 retrying with normalized messages for model=%s", model)
                current = _normalize_messages(current)
                normalized = True
                if behavior:
                    await behavior.record(base_url, model, "normalize_messages")
            else:
                raise


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


async def _responses_error_stream(message: str) -> AsyncIterator[bytes]:
    yield _response_error_event(message)


async def _responses_failed_stream(
    request_body: dict,
    response_id: str,
    *,
    previous_response_id: str | None,
    message: str,
    code: str = "upstream_error",
) -> AsyncIterator[bytes]:
    created_at = int(time.time())
    failed_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": None,
        "status": "failed",
        "error": {"code": code, "message": message},
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
    yield _response_event("response.created", {"response": {**failed_response, "status": "in_progress", "error": None}, "sequence_number": 1})
    yield _response_event("response.in_progress", {"response": {**failed_response, "status": "in_progress", "error": None}, "sequence_number": 2})
    yield _response_error_event(message, code=code, sequence_number=3)
    yield _response_event("response.failed", {"response": failed_response, "sequence_number": 4})


async def _responses_synthetic_stream(response_payload: dict) -> AsyncIterator[bytes]:
    seq = 1
    yield _response_event(
        "response.created",
        {"response": {**response_payload, "status": "in_progress", "completed_at": None, "usage": None, "output": []}, "sequence_number": seq},
    )
    seq += 1
    for item_index, item in enumerate(response_payload.get("output") or []):
        yield _response_event("response.output_item.added", {"output_index": item_index, "item": item, "sequence_number": seq})
        seq += 1
        if item.get("type") == "message":
            content = ((item.get("content") or [{}])[0]).get("text", "")
            yield _response_event(
                "response.content_part.added",
                {
                    "item_id": item["id"],
                    "output_index": item_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                    "sequence_number": seq,
                },
            )
            seq += 1
            if content:
                yield _response_event(
                    "response.output_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": item_index,
                        "content_index": 0,
                        "delta": content,
                        "sequence_number": seq,
                    },
                )
                seq += 1
                yield _response_event(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": item_index,
                        "content_index": 0,
                        "text": content,
                        "sequence_number": seq,
                    },
                )
                seq += 1
                yield _response_event(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": item_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": content, "annotations": []},
                        "sequence_number": seq,
                    },
                )
                seq += 1
        elif item.get("type") == "function_call":
            yield _response_event(
                "response.function_call_arguments.done",
                {
                    "item_id": item["id"],
                    "output_index": item_index,
                    "name": item.get("name", "unknown_tool"),
                    "arguments": item.get("arguments", ""),
                    "sequence_number": seq,
                },
            )
            seq += 1
        yield _response_event("response.output_item.done", {"output_index": item_index, "item": item, "sequence_number": seq})
        seq += 1
    yield _response_event("response.completed", {"response": response_payload, "sequence_number": seq})


def _anthropic_event(event_type: str, payload: dict) -> bytes:
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body)}\n\n".encode()


async def _anthropic_synthetic_stream(message_payload: dict) -> AsyncIterator[bytes]:
    seq_usage = message_payload.get("usage") or {"input_tokens": 0, "output_tokens": 0}
    yield _anthropic_event("message_start", {"message": {**message_payload, "content": [], "stop_reason": None}})
    for index, block in enumerate(message_payload.get("content") or []):
        if block.get("type") == "text":
            yield _anthropic_event("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}})
            text = block.get("text") or ""
            if text:
                yield _anthropic_event("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": text}})
            yield _anthropic_event("content_block_stop", {"index": index})
        elif block.get("type") == "tool_use":
            yield _anthropic_event(
                "content_block_start",
                {
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": {},
                    },
                },
            )
            partial_json = json.dumps(block.get("input") or {}, ensure_ascii=False)
            yield _anthropic_event("content_block_delta", {"index": index, "delta": {"type": "input_json_delta", "partial_json": partial_json}})
            yield _anthropic_event("content_block_stop", {"index": index})

    yield _anthropic_event(
        "message_delta",
        {
            "delta": {"stop_reason": message_payload.get("stop_reason"), "stop_sequence": None},
            "usage": {"output_tokens": seq_usage.get("output_tokens", 0)},
        },
    )
    yield _anthropic_event("message_stop", {})


def _synthetic_stream(model: str, text: str) -> StreamingResponse:
    """Return a fake streaming chat completion that renders as an assistant message in VS Code."""
    return StreamingResponse(iter_openai_error_stream(model, text), media_type="text/event-stream")


def _synthetic_json(model: str, text: str) -> JSONResponse:
    """Return a fake non-streaming chat completion with the given text."""
    return JSONResponse(synthetic_openai_chat_completion(model, text))


def _upstream_error_response(exc: Exception, model: str, streaming: bool):
    """Convert an upstream error to a synthetic assistant reply."""
    msg = assistant_error_text(exc, model)
    logger.error("v1 upstream error for %s: %s", model, msg)
    return _synthetic_stream(model, msg) if streaming else _synthetic_json(model, msg)


async def v1_chat_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /v1/chat/completions — OpenAI-format pass-through."""
    raw_body = await request.json()
    direct_reply = direct_display_tool_reply(raw_body)
    body = sanitize_openai_chat_body(raw_body)
    client = request.app.state.client
    behavior = getattr(request.app.state, "behavior", None)
    base_url = getattr(request.app.state, "base_url", "")
    endpoint_profile = getattr(request.app.state, "endpoint_profile", None)
    streaming = body.get("stream", False)
    model = body.get("model", "?")

    if direct_reply is not None:
        return _synthetic_stream(model, direct_reply) if streaming else _synthetic_json(model, direct_reply)

    logger.info("v1/chat model=%s stream=%s msgs=%d tools=%s",
                model, streaming, len(body.get("messages", [])),
                body.get("tool_choice", "none"))
    for i, msg in enumerate(body.get("messages", [])):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        preview = (content[:200] + "…") if len(content) > 200 else content
        logger.debug("  msg[%d] role=%s: %s", i, role, preview)

    if streaming and _profile_disables_streaming(endpoint_profile):
        logger.info("v1/chat using synthetic stream due to endpoint profile for model=%s", model)
        fallback = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        fallback["stream"] = False
        try:
            data, _ = await _chat_with_retries(
                client,
                fallback,
                model,
                behavior=behavior,
                base_url=base_url,
                endpoint_profile=endpoint_profile,
            )
        except Exception as exc:
            return _upstream_error_response(exc, model, streaming=True)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _synthetic_stream(model, content)

    if streaming:
        try:
            upstream, _ = await asyncio.wait_for(
                _open_stream_with_retries(
                    client,
                    body,
                    model,
                    behavior=behavior,
                    base_url=base_url,
                    endpoint_profile=endpoint_profile,
                ),
                timeout=_TTFB_TIMEOUT,
            )
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
                data, _ = await _chat_with_retries(
                    client,
                    fallback,
                    model,
                    behavior=behavior,
                    base_url=base_url,
                    endpoint_profile=endpoint_profile,
                )
            except Exception as exc:
                return _upstream_error_response(exc, model, streaming=True)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return _synthetic_stream(model, content)
        except Exception as exc:
            return _upstream_error_response(exc, model, streaming=True)

        async def generate():
            finish = None
            usage: dict = {}
            sent_any = False
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
                            sent_any = True
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                        except json.JSONDecodeError:
                            sent_any = True
                            yield f"{line}\n\n".encode()
                    else:
                        sent_any = True
                        yield f"{line}\n\n".encode()
            except Exception as exc:
                logger.error("v1 stream mid-error model=%s: %s", model, exc)
                async for chunk in iter_openai_error_stream(
                    model,
                    assistant_error_text(exc, model),
                    include_role=not sent_any,
                ):
                    yield chunk
            finally:
                logger.info("v1 ← model=%s finish=%s prompt=%d compl=%d",
                            model, finish or "?",
                            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                await upstream.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream")

    current = body
    try:
        data, current = await _chat_with_retries(
            client,
            current,
            model,
            behavior=behavior,
            base_url=base_url,
            endpoint_profile=endpoint_profile,
        )
    except httpx.HTTPStatusError as exc:
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
    try:
        data = await client.get_models()
    except Exception as exc:
        return JSONResponse({"error": {"message": str(exc), "type": "upstream_error"}}, status_code=502)
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
    behavior = getattr(request.app.state, "behavior", None)
    base_url = getattr(request.app.state, "base_url", "")
    endpoint_profile = getattr(request.app.state, "endpoint_profile", None)
    previous_response_id = body.get("previous_response_id")
    previous_messages = None
    if previous_response_id:
        previous_messages = _responses_store(request).get(previous_response_id)

    chat_body, input_messages = responses_to_openai_chat(body, previous_messages=previous_messages)
    model = body.get("model", "?")
    response_id = f"resp_{uuid.uuid4().hex}"
    direct_reply = direct_display_tool_reply({"messages": input_messages, "tools": body.get("tools")})

    if direct_reply is not None:
        response_payload = synthetic_responses_payload(
            body,
            response_id,
            direct_reply,
            previous_response_id=previous_response_id,
        )
        _remember_response(request, response_id, [*input_messages, {"role": "assistant", "content": direct_reply}])
        if body.get("stream"):
            return StreamingResponse(_responses_synthetic_stream(response_payload), media_type="text/event-stream")
        return JSONResponse(response_payload)

    logger.info("v1/responses model=%s stream=%s prev=%s input_items=%s",
                model, bool(body.get("stream")), previous_response_id or "-",
                type(body.get("input")).__name__)

    async def _synthetic_from_chat(chat_request_body: dict) -> StreamingResponse | None:
        fallback = {k: v for k, v in chat_request_body.items() if k not in ("stream", "stream_options")}
        fallback["stream"] = False
        try:
            data, effective_nonstream_body = await _chat_with_retries(
                client,
                fallback,
                model,
                behavior=behavior,
                base_url=base_url,
                endpoint_profile=endpoint_profile,
            )
        except Exception as exc:
            response_payload = synthetic_responses_payload(
                body,
                response_id,
                assistant_error_text(exc, model),
                previous_response_id=previous_response_id,
            )
            return StreamingResponse(_responses_synthetic_stream(response_payload), media_type="text/event-stream")

        response_payload, assistant_messages = openai_chat_to_responses(
            data,
            body,
            response_id,
            previous_response_id=previous_response_id,
        )
        _remember_response(
            request,
            response_id,
            [*effective_nonstream_body.get("messages", input_messages), *assistant_messages],
        )
        return StreamingResponse(_responses_synthetic_stream(response_payload), media_type="text/event-stream")

    if body.get("stream"):
        logger.info("v1/responses using synthetic stream for model=%s", model)
        synthetic = await _synthetic_from_chat(chat_body)
        if synthetic is not None:
            return synthetic

    try:
        data, effective_chat_body = await _chat_with_retries(
            client,
            chat_body,
            model,
            behavior=behavior,
            base_url=base_url,
            endpoint_profile=endpoint_profile,
        )
    except Exception as exc:
        return JSONResponse(
            synthetic_responses_payload(
                body,
                response_id,
                assistant_error_text(exc, model),
                previous_response_id=previous_response_id,
            )
        )

    response_payload, assistant_messages = openai_chat_to_responses(
        data,
        body,
        response_id,
        previous_response_id=previous_response_id,
    )
    _remember_response(request, response_id, [*effective_chat_body.get("messages", input_messages), *assistant_messages])
    return JSONResponse(response_payload)


async def v1_messages_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /v1/messages — Anthropic Messages compatibility layer."""
    body = await request.json()
    client = request.app.state.client
    behavior = getattr(request.app.state, "behavior", None)
    base_url = getattr(request.app.state, "base_url", "")
    endpoint_profile = getattr(request.app.state, "endpoint_profile", None)
    model = body.get("model", "?")
    direct_reply = anthropic_direct_display_tool_reply(body)
    chat_body = anthropic_messages_to_openai_chat(body)

    if direct_reply is not None:
        anthropic_message = synthetic_anthropic_message(body, direct_reply)
        if body.get("stream"):
            return StreamingResponse(_anthropic_synthetic_stream(anthropic_message), media_type="text/event-stream")
        return JSONResponse(anthropic_message)

    if _should_disable_anthropic_tools(body):
        if "tools" in chat_body or "tool_choice" in chat_body:
            logger.info("v1/messages suppressing Anthropic tools for model=%s", model)
        chat_body = _strip_tools(chat_body)

    logger.info("v1/messages model=%s stream=%s msgs=%d tools=%s",
                model, bool(body.get("stream")), len(body.get("messages", [])),
                bool(body.get("tools")) and "tools" in chat_body)

    fallback = {k: v for k, v in chat_body.items() if k not in ("stream", "stream_options")}
    fallback["stream"] = False
    try:
        data, _ = await _chat_with_retries(
            client,
            fallback,
            model,
            behavior=behavior,
            base_url=base_url,
            endpoint_profile=endpoint_profile,
        )
    except Exception as exc:
        error_message = assistant_error_text(exc, model)
        anthropic_message = synthetic_anthropic_message(body, error_message)
        if body.get("stream"):
            return StreamingResponse(_anthropic_synthetic_stream(anthropic_message), media_type="text/event-stream")
        return JSONResponse(anthropic_message)

    anthropic_message = openai_chat_to_anthropic_message(data, body)
    if body.get("stream"):
        return StreamingResponse(_anthropic_synthetic_stream(anthropic_message), media_type="text/event-stream")
    return JSONResponse(anthropic_message)
