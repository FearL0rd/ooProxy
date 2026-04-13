"""Translate OpenAI SSE stream to Ollama NDJSON stream."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Union

logger = logging.getLogger("ooproxy")
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_tool_arguments(raw: str) -> dict | None:
    if raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _extract_reasoning_text(payload: dict) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _content_chunk(model: str, content: str) -> bytes:
    chunk = {
        "model": model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": False,
    }
    return (json.dumps(chunk) + "\n").encode()


def _finalize_tool_calls(tool_call_buffers: dict[int, dict[str, str | int]]) -> list[dict]:
    out: list[dict] = []
    for index in sorted(tool_call_buffers):
        buf = tool_call_buffers[index]
        name = str(buf.get("name") or "")
        arguments = _parse_tool_arguments(str(buf.get("arguments") or ""))
        if not name or arguments is None:
            continue
        out.append({
            "type": "function",
            "function": {
                "index": index,
                "name": name,
                "arguments": arguments,
            },
        })
    return out


async def sse_to_ndjson(sse_stream: AsyncIterator[Union[str, bytes]], model: str) -> AsyncIterator[bytes]:
    """Convert an OpenAI SSE stream to Ollama NDJSON byte chunks.

    Accepts either string lines (from httpx aiter_lines) or raw bytes.
    Yields each Ollama chunk as a JSON line followed by a newline byte.
    """
    eval_count = 0
    prompt_eval_count = 0
    usage_received = False
    finish_reason_seen: str | None = None
    tool_call_buffers: dict[int, dict[str, str | int]] = {}
    tool_calls_emitted = False
    reasoning_open = False

    async for raw in sse_stream:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace").strip()
        else:
            line = raw.strip()
        if not line:
            continue
        if not line.startswith("data: "):
            continue

        payload = line[6:]  # strip "data: "

        if payload == "[DONE]":
            if reasoning_open:
                yield _content_chunk(model, THINK_TAG_CLOSE)
                reasoning_open = False
            tool_calls = _finalize_tool_calls(tool_call_buffers) if not tool_calls_emitted else []
            if tool_calls:
                chunk = {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    },
                    "done": False,
                }
                yield (json.dumps(chunk) + "\n").encode()
                tool_calls_emitted = True
            if not usage_received:
                # Emit a done chunk with zero counts as fallback
                done_reason = finish_reason_seen or "stop"
                chunk = {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": done_reason,
                    "eval_count": 0,
                    "prompt_eval_count": 0,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": 0,
                }
                yield (json.dumps(chunk) + "\n").encode()
            return

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = data.get("choices", [])

        # Usage-only chunk (choices is empty list, usage is present)
        if not choices and data.get("usage"):
            usage = data["usage"]
            eval_count = usage.get("completion_tokens", 0)
            prompt_eval_count = usage.get("prompt_tokens", 0)
            usage_received = True
            if reasoning_open:
                yield _content_chunk(model, THINK_TAG_CLOSE)
                reasoning_open = False
            tool_calls = _finalize_tool_calls(tool_call_buffers) if not tool_calls_emitted else []
            if tool_calls:
                tool_chunk = {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    },
                    "done": False,
                }
                yield (json.dumps(tool_chunk) + "\n").encode()
                tool_calls_emitted = True
            chunk = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": finish_reason_seen or "stop",
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_duration": 0,
                "eval_duration": 0,
            }
            yield (json.dumps(chunk) + "\n").encode()
            continue

        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        reasoning = _extract_reasoning_text(delta)
        tool_calls = delta.get("tool_calls") or []
        finish_reason = choice.get("finish_reason")

        # Track finish_reason and warn on non-stop termination
        if finish_reason:
            finish_reason_seen = finish_reason
            if finish_reason not in ("stop", "tool_calls"):
                logger.warning(
                    "stream ended with finish_reason=%r for model=%s — response may be truncated",
                    finish_reason, model,
                )

        if tool_calls:
            if reasoning_open:
                yield _content_chunk(model, THINK_TAG_CLOSE)
                reasoning_open = False
            for tool_call in tool_calls:
                index = tool_call.get("index", 0)
                buf = tool_call_buffers.setdefault(index, {"index": index, "name": "", "arguments": ""})
                function = tool_call.get("function") or {}
                name = function.get("name")
                if name:
                    buf["name"] = name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    buf["arguments"] = f"{buf['arguments']}{arguments}"
                elif isinstance(arguments, dict):
                    buf["arguments"] = json.dumps(arguments, ensure_ascii=False)

        if finish_reason == "tool_calls":
            finalized_tool_calls = _finalize_tool_calls(tool_call_buffers) if not tool_calls_emitted else []
            if finalized_tool_calls:
                chunk = {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": finalized_tool_calls,
                    },
                    "done": False,
                }
                yield (json.dumps(chunk) + "\n").encode()
                tool_calls_emitted = True

        if reasoning:
            if not reasoning_open:
                yield _content_chunk(model, THINK_TAG_OPEN)
                reasoning_open = True
            yield _content_chunk(model, reasoning)

        if isinstance(content, str) and content and reasoning_open:
            yield _content_chunk(model, THINK_TAG_CLOSE)
            reasoning_open = False

        if finish_reason and content is None and reasoning_open:
            yield _content_chunk(model, THINK_TAG_CLOSE)
            reasoning_open = False

        # Skip finish_reason-only chunks (content is null/None)
        if finish_reason and content is None:
            continue

        # In-progress content chunk
        if isinstance(content, str) and content:
            yield _content_chunk(model, content)

    # Stream closed without [DONE] — emit a fallback done chunk so the client
    # gets a properly terminated NDJSON stream.
    if not usage_received:
        if reasoning_open:
            yield _content_chunk(model, THINK_TAG_CLOSE)
        done_reason = finish_reason_seen or "stop"
        logger.debug("stream ended without [DONE], emitting fallback done chunk (reason=%s)", done_reason)
        chunk = {
            "model": model,
            "created_at": _now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": done_reason,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_duration": 0,
            "eval_duration": 0,
        }
        yield (json.dumps(chunk) + "\n").encode()


async def sse_to_generate_ndjson(sse_stream: AsyncIterator[bytes], model: str) -> AsyncIterator[bytes]:
    """Like sse_to_ndjson but uses the /api/generate response shape (response field)."""
    async for raw_chunk in sse_to_ndjson(sse_stream, model):
        try:
            chunk = json.loads(raw_chunk.decode())
        except json.JSONDecodeError:
            yield raw_chunk
            continue

        if "message" in chunk:
            content = chunk["message"].get("content", "")
            del chunk["message"]
            chunk["response"] = content

        yield (json.dumps(chunk) + "\n").encode()
