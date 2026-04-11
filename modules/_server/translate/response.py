"""Translate OpenAI response bodies to Ollama format."""

from __future__ import annotations

import json
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_arguments(raw: object) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": raw}


def _tool_calls_to_ollama(tool_calls: list[dict]) -> list[dict]:
    out: list[dict] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        out.append({
            "type": tool_call.get("type", "function"),
            "function": {
                "index": index,
                "name": function.get("name", "unknown_tool"),
                "arguments": _parse_arguments(function.get("arguments")),
            },
        })
    return out


def openai_chat_to_ollama(data: dict, model: str) -> dict:
    """Convert a non-streaming OpenAI chat completion response to Ollama format."""
    choice = data["choices"][0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    ollama_message = {
        "role": message.get("role", "assistant"),
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        ollama_message["tool_calls"] = _tool_calls_to_ollama(message["tool_calls"])
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": ollama_message,
        "done": True,
        "done_reason": choice.get("finish_reason") or "stop",
        "eval_count": usage.get("completion_tokens", 0),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": 0,
    }


def openai_generate_to_ollama(data: dict, model: str) -> dict:
    """Convert a non-streaming OpenAI chat response to Ollama /api/generate format."""
    base = openai_chat_to_ollama(data, model)
    content = base["message"]["content"]
    del base["message"]
    base["response"] = content
    return base


def openai_embeddings_to_ollama(data: dict) -> dict:
    """Convert an OpenAI embeddings response to Ollama format."""
    return {
        "embedding": data["data"][0]["embedding"],
    }
