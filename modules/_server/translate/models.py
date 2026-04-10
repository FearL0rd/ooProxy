"""Translate OpenAI model list to Ollama tags format."""

from __future__ import annotations

from datetime import datetime, timezone

_FAMILY_KEYWORDS = [
    "llama", "mistral", "gemma", "phi", "qwen", "falcon", "mpt",
    "starcoder", "codellama", "deepseek", "mixtral", "vicuna", "alpaca",
]

_EMBEDDING_KEYWORDS = ["embed", "bge", "e5-", "rerank", "retrieval", "minilm"]

# Models known to have limited (4K) context — everything else defaults to 128K
_SMALL_CONTEXT_KEYWORDS = ["llama2", "llama-2", "llama2-", "codellama-"]

# Models that are completion-capable but unlikely to support tool-calling
# (code-only, older LLMs, reward models)
_NO_TOOLS_KEYWORDS = [
    "llama2", "llama-2", "reward", "starcoder", "codellama",
    "fuyu", "vision", "vl-", "-vl", "image", "audio",
]


def _infer_family(model_id: str) -> str:
    lower = model_id.lower()
    for kw in _FAMILY_KEYWORDS:
        if kw in lower:
            return kw
    return "unknown"


def _infer_context_length(model_id: str) -> int:
    lower = model_id.lower()
    if any(kw in lower for kw in _SMALL_CONTEXT_KEYWORDS):
        return 4096
    # Check for explicit context hints in name
    if "128k" in lower:
        return 131072
    if "32k" in lower:
        return 32768
    if "8k" in lower:
        return 8192
    # Default: 128K for modern models
    return 131072


def _infer_capabilities(model_id: str) -> list[str]:
    lower = model_id.lower()
    if any(kw in lower for kw in _EMBEDDING_KEYWORDS):
        return ["embedding"]
    caps = ["completion"]
    if not any(kw in lower for kw in _NO_TOOLS_KEYWORDS):
        caps.append("tools")
    return caps


def _created_to_iso(created: int | None) -> str:
    if not created:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest_for(model_id: str) -> str:
    # Stable placeholder digest derived from the model name
    h = abs(hash(model_id)) % (16 ** 12)
    return f"sha256:{h:012x}{'0' * 52}"


def openai_models_to_ollama_tags(data: dict) -> dict:
    """Convert GET /v1/models response to Ollama /api/tags format."""
    models = []
    for entry in data.get("data", []):
        model_id = entry.get("id", "")
        family = _infer_family(model_id)
        models.append({
            "name": model_id,
            "model": model_id,
            "modified_at": _created_to_iso(entry.get("created")),
            "size": 0,
            "digest": _digest_for(model_id),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": family,
                "families": [family],
                "parameter_size": "",
                "quantization_level": "",
            },
        })
    return {"models": models}


def openai_model_to_ollama_show(model_id: str) -> dict:
    """Synthesize an Ollama /api/show response for a given model ID."""
    family = _infer_family(model_id)
    context_length = _infer_context_length(model_id)
    return {
        "modelfile": f"# Synthesized by ooProxy\nFROM {model_id}\n",
        "parameters": "",
        "template": "{{ .System }}\n{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "",
            "quantization_level": "",
        },
        "model_info": {
            "general.architecture": family,
            f"{family}.context_length": context_length,
        },
        "capabilities": _infer_capabilities(model_id),
    }
