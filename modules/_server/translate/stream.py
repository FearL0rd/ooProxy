"""Translate OpenAI SSE stream to Ollama NDJSON stream."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncIterator, Union


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


async def sse_to_ndjson(sse_stream: AsyncIterator[Union[str, bytes]], model: str) -> AsyncIterator[bytes]:
    """Convert an OpenAI SSE stream to Ollama NDJSON byte chunks.

    Accepts either string lines (from httpx aiter_lines) or raw bytes.
    Yields each Ollama chunk as a JSON line followed by a newline byte.
    """
    eval_count = 0
    prompt_eval_count = 0
    usage_received = False

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
            if not usage_received:
                # Emit a done chunk with zero counts as fallback
                chunk = {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
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
            chunk = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
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
        finish_reason = choice.get("finish_reason")

        # Skip finish_reason-only chunks (content is null/None)
        if finish_reason and content is None:
            continue

        # In-progress content chunk
        if content is not None:
            chunk = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": content},
                "done": False,
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
