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


def _responses_tools_to_openai(tools: object) -> list[dict] | None:
    if not isinstance(tools, list):
        return None

    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue

        tool_type = tool.get("type")
        name = tool.get("name")
        parameters = tool.get("parameters")
        if tool_type == "function" and isinstance(name, str) and name:
            converted_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}, "required": []},
                },
            }
            if "strict" in tool:
                converted_tool["function"]["strict"] = bool(tool.get("strict"))
            converted.append(converted_tool)

    return converted or None


def _responses_tool_choice_to_openai(tool_choice: object) -> object:
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return "required"
        return tool_choice

    if not isinstance(tool_choice, dict):
        return tool_choice

    choice_type = tool_choice.get("type")
    if choice_type in {"auto", "none", "required"}:
        return choice_type
    if choice_type == "function" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if choice_type == "custom" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


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
    converted_tools = _responses_tools_to_openai(body.get("tools"))
    if converted_tools is not None:
        out["tools"] = converted_tools
    if body.get("tool_choice") is not None:
        out["tool_choice"] = _responses_tool_choice_to_openai(body["tool_choice"])
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


def _anthropic_text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part)


def anthropic_messages_to_openai_chat(body: dict) -> dict:
    """Convert Anthropic /v1/messages requests to OpenAI /v1/chat/completions."""
    messages: list[dict] = []

    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_text = _anthropic_text_from_content(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for item in body.get("messages") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": ""})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif block_type == "tool_use":
                arguments = block.get("input", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append({
                    "id": block.get("id") or f"toolu_{block_index}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", "unknown_tool"),
                        "arguments": arguments,
                    },
                })
            elif block_type == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = _anthropic_text_from_content(result_content)
                elif not isinstance(result_content, str):
                    result_content = json.dumps(result_content, ensure_ascii=False)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or f"toolu_{block_index}",
                    "content": result_content,
                })

        if role == "assistant":
            message = {"role": "assistant", "content": "\n".join(part for part in text_parts if part)}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        elif text_parts:
            messages.append({"role": role, "content": "\n".join(part for part in text_parts if part)})

        messages.extend(tool_results)

    out: dict = {
        "model": body["model"],
        "messages": messages,
        "stream": bool(body.get("stream", False)),
        "max_tokens": body.get("max_tokens") or _DEFAULT_MAX_TOKENS,
    }

    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]

    tools = body.get("tools")
    if isinstance(tools, list):
        converted_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            converted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}, "required": []},
                },
            })
        if converted_tools:
            out["tools"] = converted_tools

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            out["tool_choice"] = "auto"
        elif choice_type == "any":
            out["tool_choice"] = "required"
        elif choice_type == "tool" and tool_choice.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
        elif choice_type == "none":
            out["tool_choice"] = "none"

    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out
