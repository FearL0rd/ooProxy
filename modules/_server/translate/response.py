"""Translate OpenAI response bodies to Ollama format."""

from __future__ import annotations

from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def openai_chat_to_ollama(data: dict, model: str) -> dict:
    """Convert a non-streaming OpenAI chat completion response to Ollama format."""
    choice = data["choices"][0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": {
            "role": message.get("role", "assistant"),
            "content": message.get("content") or "",
        },
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
