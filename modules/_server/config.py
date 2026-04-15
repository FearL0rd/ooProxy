"""Proxy configuration dataclasses and loaders."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

from cli_contract import CommandError
from modules._server.key_store import ApiKeyStore, endpoint_from_url

_DEFAULT_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_PORT = 11434
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_CASCADE_PATH = Path.home() / ".ooProxy" / "cascade.json"
_DEFAULT_CASCADE_SYSTEM_PROMPT = """You are a task complexity evaluator. Your sole job is to assess whether a given prompt can be answered correctly and completely by a small, fast language model.

You must respond with ONLY a JSON object - no explanation, no preamble, no markdown fences.

Evaluate along these axes:
- Factual complexity: does this require deep or specialised knowledge?
- Reasoning depth: does this require multi-step logic, inference, or planning?
- Output quality bar: would a basic model's answer be good enough, or is nuance critical?
- Ambiguity: is the prompt clear, or does it require interpretation?

Scoring guide:
  1.0  Trivially simple - greetings, basic lookups, simple yes/no
  0.8  Straightforward - short factual questions, simple formatting tasks
  0.6  Moderate - requires some reasoning or domain knowledge
  0.4  Complex - multi-step reasoning, code generation, nuanced analysis
  0.2  Very complex - expert-level, long-form, high-stakes output
  0.0  Cannot be handled without a powerful model

Return exactly: {\"CONFIDENCE\": <number between 0 and 1>}"""
_DEFAULT_CASCADE_USER_PROMPT_TEMPLATE = """Evaluate the following prompt and return your confidence that a weak model can handle it well.

PROMPT TO EVALUATE:
\"\"\"
{USER_PROMPT}
\"\"\"

Respond ONLY with: {\"CONFIDENCE\": <0.0 to 1.0>}

If CONFIDENCE >= 0.70, a weak model will handle it.
If CONFIDENCE < 0.70, it will be escalated to a stronger model."""
_DEFAULT_CASCADE_RETRY_USER_PROMPT_TEMPLATE = """Return ONLY {\"CONFIDENCE\": <0.0 to 1.0>} for the following prompt.

PROMPT TO EVALUATE:
\"\"\"
{USER_PROMPT}
\"\"\""""


def render_cascade_decision_prompt(
    template: str,
    *,
    user_prompt: str,
    weak_model: str,
    strong_model: str,
    available_tools: str,
    tool_choice: str,
    request_json: str,
) -> str:
    rendered = str(template or "")
    replacements = {
        "USER_PROMPT": user_prompt,
        "WEAK_MODEL": weak_model,
        "STRONG_MODEL": strong_model,
        "AVAILABLE_TOOLS": available_tools,
        "TOOL_CHOICE": tool_choice,
        "REQUEST_JSON": request_json,
    }
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


@dataclass(frozen=True)
class CascadeDecisionConfig:
    threshold: float = 0.72
    max_tokens: int = 1024
    timeout_seconds: float = 5.0
    system_prompt: str = _DEFAULT_CASCADE_SYSTEM_PROMPT
    user_prompt_template: str = _DEFAULT_CASCADE_USER_PROMPT_TEMPLATE
    retry_user_prompt_template: str = _DEFAULT_CASCADE_RETRY_USER_PROMPT_TEMPLATE


@dataclass(frozen=True)
class CascadeRouteConfig:
    weak_model: str
    strong_model: str
    weak_url: str
    weak_key: str
    strong_url: str
    strong_key: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CascadeConfig:
    host: str
    port: int
    routes: tuple[CascadeRouteConfig, ...]
    decision: CascadeDecisionConfig = field(default_factory=CascadeDecisionConfig)


@dataclass
class ProxyConfig:
    url: str
    key: str
    port: int
    host: str = _DEFAULT_HOST
    cascade: CascadeConfig | None = None

    @classmethod
    def from_args(cls, args) -> "ProxyConfig":
        url = getattr(args, "url", None) or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_URL
        key = getattr(args, "key", None) or os.environ.get("OPENAI_API_KEY") or ""
        if not key:
            key = ApiKeyStore().get(endpoint_from_url(url)) or ""
        port = getattr(args, "port", None) or _DEFAULT_PORT
        host = getattr(args, "host", None) or _DEFAULT_HOST
        return cls(url=url.rstrip("/"), key=key, port=int(port), host=str(host))


def _normalize_config_url(value: object, *, field_name: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        raise CommandError(f"cascade route is missing {field_name}", exit_code=2)
    return text


def _resolve_config_secret(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    if value.startswith("env:"):
        return os.environ.get(value[4:], "")
    return value


def _route_key(url: str, explicit_key: object, shared_key: object, store: ApiKeyStore) -> str:
    resolved = _resolve_config_secret(explicit_key)
    if resolved:
        return resolved
    resolved = _resolve_config_secret(shared_key)
    if resolved:
        return resolved
    return store.get(endpoint_from_url(url)) or ""


def load_cascade_config(path: Path | None = None) -> ProxyConfig:
    config_path = _DEFAULT_CASCADE_PATH if path is None else Path(path).expanduser()
    if not config_path.exists():
        raise CommandError(
            f"Cascade config not found: {config_path}",
            show_usage=False,
            exit_code=2,
        )

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommandError(f"Invalid JSON in {config_path}: {exc}", exit_code=2) from exc

    if not isinstance(raw, dict):
        raise CommandError(f"Cascade config must contain a JSON object: {config_path}", exit_code=2)

    host = str(raw.get("host") or _DEFAULT_HOST).strip() or _DEFAULT_HOST
    port = int(raw.get("port") or _DEFAULT_PORT)

    decision_raw = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    decision = CascadeDecisionConfig(
        threshold=float(decision_raw.get("threshold", CascadeDecisionConfig.threshold)),
        max_tokens=int(decision_raw.get("max_tokens", CascadeDecisionConfig.max_tokens)),
        timeout_seconds=float(decision_raw.get("timeout_seconds", CascadeDecisionConfig.timeout_seconds)),
        system_prompt=str(decision_raw.get("system_prompt") or CascadeDecisionConfig.system_prompt),
        user_prompt_template=str(decision_raw.get("user_prompt_template") or CascadeDecisionConfig.user_prompt_template),
        retry_user_prompt_template=str(decision_raw.get("retry_user_prompt_template") or CascadeDecisionConfig.retry_user_prompt_template),
    )

    routes_raw = raw.get("routes")
    if not isinstance(routes_raw, list) or not routes_raw:
        raise CommandError("Cascade config must define a non-empty routes array", exit_code=2)

    store = ApiKeyStore()
    routes: list[CascadeRouteConfig] = []
    seen_models: set[str] = set()
    for index, entry in enumerate(routes_raw, start=1):
        if not isinstance(entry, dict):
            raise CommandError(f"Cascade route #{index} must be a JSON object", exit_code=2)
        weak_model = str(entry.get("weak_model") or "").strip()
        strong_model = str(entry.get("strong_model") or "").strip()
        if not weak_model or not strong_model:
            raise CommandError(
                f"Cascade route #{index} must define weak_model and strong_model",
                exit_code=2,
            )
        if weak_model in seen_models:
            raise CommandError(f"Cascade route weak_model is duplicated: {weak_model}", exit_code=2)
        seen_models.add(weak_model)

        shared_url = entry.get("url")
        weak_url = _normalize_config_url(entry.get("weak_url") or shared_url, field_name="weak_url")
        strong_url = _normalize_config_url(entry.get("strong_url") or shared_url, field_name="strong_url")
        shared_key = entry.get("key")
        weak_key = _route_key(weak_url, entry.get("weak_key"), shared_key, store)
        strong_key = _route_key(strong_url, entry.get("strong_key"), shared_key, store)
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}

        routes.append(
            CascadeRouteConfig(
                weak_model=weak_model,
                strong_model=strong_model,
                weak_url=weak_url,
                weak_key=weak_key,
                strong_url=strong_url,
                strong_key=strong_key,
                metadata=dict(metadata),
            )
        )

    cascade = CascadeConfig(host=host, port=port, routes=tuple(routes), decision=decision)
    return ProxyConfig(url="cascade://local", key="", port=port, host=host, cascade=cascade)
