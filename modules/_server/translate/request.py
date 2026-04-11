"""Translate client request bodies to upstream OpenAI chat format."""

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
        opts = body.get("options") or {}
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
        opts = body.get("options") or {}
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


def _responses_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif item_type == "refusal":
                refusal = item.get("refusal")
                if isinstance(refusal, str):
                    parts.append(refusal)
        return "\n".join(part for part in parts if part)
    return ""


def _responses_item_to_messages(item: object) -> list[dict]:
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return []

    item_type = item.get("type")

    if item_type == "function_call_output":
        output = item.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        message = {
            "role": "tool",
            "content": output,
        }
        if item.get("call_id"):
            message["tool_call_id"] = item["call_id"]
        return [message]

    if item_type == "function_call":
        arguments = item.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": item.get("call_id") or item.get("id") or "call_0",
                "type": "function",
                "function": {
                    "name": item.get("name", "unknown_tool"),
                    "arguments": arguments,
                },
            }],
        }]

    if item_type == "message" or "role" in item:
        content = _responses_content_to_text(item.get("content"))
        return [{
            "role": item.get("role", "user"),
            "content": content,
        }]

    if item_type in {"input_text", "output_text", "text"}:
        text = item.get("text")
        if isinstance(text, str):
            return [{"role": item.get("role", "user"), "content": text}]

    return []


def _responses_input_to_messages(input_value: object) -> list[dict]:
    if input_value is None:
        return []
    if isinstance(input_value, (str, dict)):
        return _responses_item_to_messages(input_value)
    if isinstance(input_value, list):
        messages: list[dict] = []
        for item in input_value:
            messages.extend(_responses_item_to_messages(item))
        return messages
    return []


def responses_to_openai_chat(body: dict, previous_messages: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Convert an OpenAI /v1/responses request to /v1/chat/completions."""
    messages = [dict(message) for message in (previous_messages or [])]
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        if not messages or messages[0].get("role") != "system" or messages[0].get("content") != instructions:
            messages.insert(0, {"role": "system", "content": instructions})

    messages.extend(_responses_input_to_messages(body.get("input")))

    out: dict = {
        "model": body["model"],
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    if body.get("tools") is not None:
        out["tools"] = body["tools"]
    if body.get("tool_choice") is not None:
        out["tool_choice"] = body["tool_choice"]
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
    text_config = body.get("text") or {}
    response_format = text_config.get("format") if isinstance(text_config, dict) else None
    if isinstance(response_format, dict) and response_format.get("type") in {"json_object", "json_schema"}:
        out["response_format"] = response_format
    if "max_tokens" not in out:
        out["max_tokens"] = _DEFAULT_MAX_TOKENS
    return out, messages
