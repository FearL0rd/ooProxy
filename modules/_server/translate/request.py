"""Translate Ollama request bodies to OpenAI format."""

from __future__ import annotations

# NIM (and some other hosted endpoints) default to a very small max_tokens (e.g. 32)
# when the field is omitted, causing responses to be silently truncated.  Inject this
# default whenever the client does not specify num_predict / max_tokens explicitly.
_DEFAULT_MAX_TOKENS = 32768


def chat_to_openai(body: dict) -> dict:
    """Convert an Ollama /api/chat request body to OpenAI /v1/chat/completions."""
    out: dict = {
        "model": body["model"],
        "messages": body.get("messages", []),
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
