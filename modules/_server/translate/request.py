"""Translate client request bodies to upstream OpenAI chat format."""

from __future__ import annotations

import json

# NIM (and some other hosted endpoints) default to a very small max_tokens (e.g. 32)
# when the field is omitted, causing responses to be silently truncated.  Inject this
# default whenever the client does not specify num_predict / max_tokens explicitly.
_DEFAULT_MAX_TOKENS = 32768


def _tool_failed(content: str) -> bool:
    if not content:
        return False
    try:
        payload = json.loads(content)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("return_code"), int):
        return payload["return_code"] != 0
    if payload.get("ok") is False:
        return True
    return bool(payload.get("error")) and payload.get("ok") is not True


def _tool_return_code_payload(content: str) -> str:
    return json.dumps({"return_code": 1 if _tool_failed(content) else 0}, ensure_ascii=False)


def _wrap_tool_output(content: str) -> str:
    return f"Exact tool output follows.\n~~~text\n{content}~~~"


def _tool_output_for_model(content: str, *, display_directly: bool) -> str:
    if display_directly:
        return _tool_return_code_payload(content)
    if "\n" in content or content.startswith((" ", "\t")) or content.endswith(("\n", " ", "\t")):
        return _wrap_tool_output(content)
    return content


def _direct_display_content_for_model(content: str) -> str:
    return content


def _sanitize_tool_schema(tool: dict) -> tuple[dict | None, str | None]:
    if not isinstance(tool, dict):
        return None, None

    sanitized = dict(tool)
    direct_name: str | None = None

    function = sanitized.get("function")
    if isinstance(function, dict):
        function_copy = dict(function)
        if function_copy.get("display_directly") and isinstance(function_copy.get("name"), str):
            direct_name = function_copy["name"]
        function_copy.pop("display_directly", None)
        sanitized["function"] = function_copy
        return sanitized, direct_name

    if sanitized.get("display_directly") and isinstance(sanitized.get("name"), str):
        direct_name = sanitized["name"]
    sanitized.pop("display_directly", None)
    return sanitized, direct_name


def _sanitize_tools(tools: object) -> tuple[list[dict] | None, set[str]]:
    if not isinstance(tools, list):
        return None, set()

    sanitized_tools: list[dict] = []
    direct_display_tools: set[str] = set()
    for tool in tools:
        sanitized, direct_name = _sanitize_tool_schema(tool)
        if sanitized is not None:
            sanitized_tools.append(sanitized)
        if direct_name:
            direct_display_tools.add(direct_name)
    return sanitized_tools, direct_display_tools


def _normalize_openai_messages(messages: list[dict], direct_display_tools: set[str]) -> list[dict]:
    out: list[dict] = []
    pending_tool_calls: list[dict] = []

    for message_index, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            tool_calls = [dict(tool_call) for tool_call in (message.get("tool_calls") or []) if isinstance(tool_call, dict)]
            for tool_index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") or {}
                call_id = tool_call.get("id") or f"call_{message_index}_{tool_index}"
                pending_tool_calls.append({"id": call_id, "name": function.get("name", "unknown_tool")})
            normalized = dict(message)
            normalized["tool_calls"] = tool_calls
            out.append(normalized)
            continue

        if role == "tool":
            normalized = dict(message)
            tool_call_id = normalized.get("tool_call_id")
            tool_name = normalized.get("tool_name") or normalized.get("name")
            if not tool_name and tool_call_id:
                matched = next((call for call in pending_tool_calls if call["id"] == tool_call_id), None)
                if matched is not None:
                    tool_name = matched["name"]
            if not tool_name:
                match_index = next((i for i, call in enumerate(pending_tool_calls) if call["name"]), None)
                if match_index is not None:
                    tool_name = pending_tool_calls[match_index]["name"]

            content = normalized.get("content") or ""
            string_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            is_direct_display = bool(tool_name and tool_name in direct_display_tools)
            include_direct_display_content = is_direct_display and any(
                isinstance(later_message, dict) and later_message.get("role") != "tool"
                for later_message in messages[message_index + 1:]
            )
            if include_direct_display_content:
                normalized["content"] = _direct_display_content_for_model(string_content)
            else:
                normalized["content"] = _tool_output_for_model(
                    string_content,
                    display_directly=is_direct_display,
                )
            normalized.pop("display_content", None)
            normalized.pop("display_directly", None)
            out.append(normalized)
            continue

        out.append(dict(message))

    return out


def _resolve_direct_display_tool_name(messages: list[dict], direct_display_tools: set[str], target_message: dict) -> str | None:
    pending_tool_calls: list[dict] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            for tool_index, tool_call in enumerate(message.get("tool_calls") or []):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                call_id = tool_call.get("id") or f"call_{message_index}_{tool_index}"
                pending_tool_calls.append({"id": call_id, "name": function.get("name", "unknown_tool")})
            continue
        if message is not target_message:
            continue
        tool_name = message.get("tool_name") or message.get("name")
        if isinstance(tool_name, str) and tool_name in direct_display_tools:
            return tool_name
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            matched = next((call for call in pending_tool_calls if call["id"] == tool_call_id and call["name"] in direct_display_tools), None)
            if matched is not None:
                return matched["name"]
        return None
    return None


def direct_display_tool_reply(body: dict) -> str | None:
    tools, direct_display_tools = _sanitize_tools(body.get("tools"))
    if not direct_display_tools:
        return None

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None

    last_message = messages[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "tool":
        return None

    tool_name = _resolve_direct_display_tool_name(messages, direct_display_tools, last_message)
    if not tool_name:
        return None

    content = last_message.get("display_content") if isinstance(last_message.get("display_content"), str) else last_message.get("content")
    return content if isinstance(content, str) else None


def sanitize_openai_chat_body(body: dict) -> dict:
    sanitized_tools, direct_display_tools = _sanitize_tools(body.get("tools"))
    out = dict(body)
    messages = body.get("messages") or []
    out["messages"] = _normalize_openai_messages(messages if isinstance(messages, list) else [], direct_display_tools)
    if sanitized_tools is not None:
        out["tools"] = sanitized_tools
    else:
        out.pop("tools", None)
    return out


def anthropic_direct_display_tool_reply(body: dict) -> str | None:
    _, direct_display_tools = _sanitize_tools(body.get("tools"))
    if not direct_display_tools:
        return None

    tool_use_names: dict[str, str] = {}
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
                tool_use_names[block["id"]] = block["name"]

    last_message = messages[-1]
    if not isinstance(last_message, dict):
        return None
    content = last_message.get("content")
    if not isinstance(content, list) or not content:
        return None
    last_block = content[-1]
    if not isinstance(last_block, dict) or last_block.get("type") != "tool_result":
        return None

    tool_name = tool_use_names.get(last_block.get("tool_use_id"))
    if tool_name not in direct_display_tools:
        return None

    result_content = last_block.get("content", "")
    if isinstance(result_content, list):
        return _anthropic_text_from_content(result_content)
    if isinstance(result_content, str):
        return result_content
    return json.dumps(result_content, ensure_ascii=False)


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
    sanitized_tools, direct_display_tools = _sanitize_tools(body.get("tools"))
    out: dict = {
        "model": body["model"],
        "messages": _normalize_openai_messages(_normalize_native_messages(body.get("messages", [])), direct_display_tools),
    }
    if sanitized_tools is not None:
        out["tools"] = sanitized_tools
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
    sanitized_tools, _ = _sanitize_tools(tools)
    if sanitized_tools is None:
        return None

    converted: list[dict] = []
    for tool in sanitized_tools:
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
    _, direct_display_tools = _sanitize_tools(body.get("tools"))
    sanitized_messages = _normalize_openai_messages(messages, direct_display_tools)

    out: dict = {
        "model": body["model"],
        "messages": sanitized_messages,
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
    _, direct_display_tools = _sanitize_tools(body.get("tools"))
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
                    "name": block.get("name"),
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
        "messages": _normalize_openai_messages(messages, direct_display_tools),
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
