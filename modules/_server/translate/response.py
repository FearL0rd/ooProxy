"""Translate OpenAI response bodies to client-compatible formats."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone


THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"


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


def _extract_reasoning_text(payload: dict) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


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
    reasoning = _extract_reasoning_text(message)
    content = message.get("content") or ""
    if reasoning:
        content = f"{THINK_TAG_OPEN}{reasoning}{THINK_TAG_CLOSE}{content}"
    ollama_message = {
        "role": message.get("role", "assistant"),
        "content": content,
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


def _response_output_text_part(text: str) -> dict:
    return {
        "type": "output_text",
        "text": text,
        "annotations": [],
    }


def _chat_message_to_responses_output(message: dict) -> tuple[list[dict], str]:
    output: list[dict] = []
    accumulated_text = []

    content = message.get("content") or ""
    if content:
        accumulated_text.append(content)
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": message.get("role", "assistant"),
            "content": [_response_output_text_part(content)],
        })

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        output.append({
            "id": tool_call.get("id") or f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:16]}",
            "name": function.get("name", "unknown_tool"),
            "arguments": function.get("arguments", ""),
            "status": "completed",
        })

    return output, "".join(accumulated_text)


def responses_usage_from_chat(data: dict) -> dict:
    usage = data.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": usage.get("completion_tokens", 0),
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.get("total_tokens", 0),
    }


def openai_chat_to_responses(
    data: dict,
    request_body: dict,
    response_id: str,
    *,
    previous_response_id: str | None = None,
    created_at: int | None = None,
) -> tuple[dict, list[dict]]:
    """Convert a non-streaming OpenAI chat completion to a Responses API payload."""
    created = created_at or int(time.time())
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {"role": "assistant", "content": ""}
    output, output_text = _chat_message_to_responses_output(message)
    finish_reason = choice.get("finish_reason") or "stop"
    status = "completed" if finish_reason not in {"length", "content_filter"} else "incomplete"
    incomplete_details = None
    if finish_reason == "length":
        incomplete_details = {"reason": "max_output_tokens"}

    response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "completed_at": int(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": request_body.get("instructions"),
        "metadata": request_body.get("metadata") or {},
        "model": request_body["model"],
        "output": output,
        "output_text": output_text,
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
        "usage": responses_usage_from_chat(data),
        "user": request_body.get("user"),
    }

    assistant_messages = [message]
    return response, assistant_messages


def anthropic_stop_reason(finish_reason: str | None) -> str | None:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason in {None, "stop"}:
        return "end_turn"
    return finish_reason


def openai_chat_to_anthropic_message(data: dict, request_body: dict) -> dict:
    """Convert a non-streaming OpenAI chat completion to an Anthropic message."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {"role": "assistant", "content": ""}
    usage = data.get("usage") or {}

    content_blocks: list[dict] = []
    text = message.get("content") or ""
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"_raw": arguments}
        else:
            parsed_arguments = arguments
        content_blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
            "name": function.get("name", "unknown_tool"),
            "input": parsed_arguments if isinstance(parsed_arguments, dict) else {"value": parsed_arguments},
        })

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": request_body["model"],
        "content": content_blocks,
        "stop_reason": anthropic_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
