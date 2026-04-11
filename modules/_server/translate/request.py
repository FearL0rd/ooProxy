"""Translate Ollama request bodies to OpenAI format."""

from __future__ import annotations

import json

# NIM (and some other hosted endpoints) default to a very small max_tokens (e.g. 32)
# when the field is omitted, causing responses to be silently truncated.  Inject this
# default whenever the client does not specify num_predict / max_tokens explicitly.
_DEFAULT_MAX_TOKENS = 32768


def _normalize_native_messages(messages: list[dict]) -> list[dict]:
    """Convert Ollama-style tool messages into OpenAI-compatible messages."""
    out: list[dict] = []
    pending_tool_calls: list[dict] = []

    for message_index, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            converted_tool_calls: list[dict] = []
            for tool_index, tool_call in enumerate(message.get("tool_calls") or []):
                function = tool_call.get("function") or {}
                call_id = tool_call.get("id") or f"call_{message_index}_{tool_index}"
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                converted = {
                    "id": call_id,
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": function.get("name", "unknown_tool"),
                        "arguments": arguments,
                    },
                }
                converted_tool_calls.append(converted)
                pending_tool_calls.append({"id": call_id, "name": function.get("name", "unknown_tool")})

            normalized = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": converted_tool_calls,
            }
            out.append(normalized)
            continue

        if role == "tool":
            tool_name = message.get("tool_name") or message.get("name")
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                match_index = next((i for i, call in enumerate(pending_tool_calls) if call["name"] == tool_name), None)
                if match_index is None and pending_tool_calls:
                    match_index = 0
                if match_index is not None:
                    tool_call_id = pending_tool_calls.pop(match_index)["id"]

            normalized = {
                "role": "tool",
                "content": message.get("content") or "",
            }
            if tool_call_id:
                normalized["tool_call_id"] = tool_call_id
            out.append(normalized)
            continue

        out.append(dict(message))

    return out


def chat_to_openai(body: dict) -> dict:
    """Convert an Ollama /api/chat request body to OpenAI /v1/chat/completions."""
    out: dict = {
        "model": body["model"],
        "messages": _normalize_native_messages(body.get("messages", [])),
    }
    if "tools" in body:
        out["tools"] = body["tools"]
    if "tool_choice" in body:
        out["tool_choice"] = body["tool_choice"]
    if "stream" in body:
        out["stream"] = body["stream"]
        if body["stream"]:
            out["stream_options"] = {"include_usage": True}
    if "options" in body:
        opts = body["options"]
        if "temperature" in opts:
            out["temperature"] = opts["temperature"]
        if "top_p" in opts:
            out["top_p"] = opts["top_p"]
        if "num_predict" in opts:
            num = opts["num_predict"]
            if num and num > 0:
                out["max_tokens"] = num
            # num_predict == -1 means "unlimited" in Ollama — omit max_tokens
            # so the server uses its own ceiling rather than a small proxy default.
        if "stop" in opts:
            out["stop"] = opts["stop"]
    if "format" in body and body["format"] == "json":
        out["response_format"] = {"type": "json_object"}
    if "max_tokens" not in out:
        out["max_tokens"] = _DEFAULT_MAX_TOKENS
    return out


def generate_to_openai(body: dict) -> dict:
    """Convert an Ollama /api/generate request body to OpenAI /v1/chat/completions."""
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    messages.append({"role": "user", "content": body.get("prompt", "")})

    out: dict = {
        "model": body["model"],
        "messages": messages,
    }
    if "stream" in body:
        out["stream"] = body["stream"]
        if body["stream"]:
            out["stream_options"] = {"include_usage": True}
    if "options" in body:
        opts = body["options"]
        if "temperature" in opts:
            out["temperature"] = opts["temperature"]
        if "top_p" in opts:
            out["top_p"] = opts["top_p"]
        if "num_predict" in opts:
            num = opts["num_predict"]
            if num and num > 0:
                out["max_tokens"] = num
        if "stop" in opts:
            out["stop"] = opts["stop"]
    if "format" in body and body["format"] == "json":
        out["response_format"] = {"type": "json_object"}
    if "max_tokens" not in out:
        out["max_tokens"] = _DEFAULT_MAX_TOKENS
    return out


def embeddings_to_openai(body: dict) -> dict:
    """Convert an Ollama /api/embeddings request body to OpenAI /v1/embeddings."""
    return {
        "model": body["model"],
        "input": body.get("prompt", ""),
    }
