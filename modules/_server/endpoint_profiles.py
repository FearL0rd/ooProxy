"""Static endpoint profiles for known upstream providers.

Profiles describe endpoint-specific behavior that should be preferred over
trial-and-error when the upstream host is already known.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("ooproxy")

_PROFILE_DIR = Path(__file__).resolve().parents[2] / "endpoints"


def _normalize_string_list(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(str(value).strip().lower() for value in values if str(value).strip())


def _normalize_port_list(values: Any) -> tuple[int, ...]:
    ports: list[int] = []
    if not isinstance(values, list):
        return ()
    for value in values:
        try:
            ports.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(ports)


def _default_port_for_scheme(scheme: str) -> int | None:
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    return None


def _host_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith(f".{suffix}")


@dataclass(frozen=True)
class EndpointProfile:
    id: str
    source_path: str
    schemes: tuple[str, ...] = field(default_factory=tuple)
    host_equals: tuple[str, ...] = field(default_factory=tuple)
    host_suffixes: tuple[str, ...] = field(default_factory=tuple)
    ports: tuple[int, ...] = field(default_factory=tuple)
    path_prefixes: tuple[str, ...] = field(default_factory=tuple)
    models_method: str = "GET"
    models_path: str = "models"
    models_format: str = "openai"
    models_variables: dict[str, dict[str, str]] = field(default_factory=dict)
    models_items_path: str = ""
    models_fields: dict[str, str] = field(default_factory=dict)
    models_owned_by: str | None = None
    models_capabilities: dict[str, Any] = field(default_factory=dict)
    chat_path: str = "chat/completions"
    chat_streaming: str = "sse"
    chat_tools: str = "trial"
    chat_system_prompt: str = "supported"
    # Seconds to wait for first streaming byte (TTFB). If None, use global default.
    ttfb_timeout: float | None = None
    health_mode: str = "internal-ready"
    health_path: str | None = None
    health_method: str = "GET"
    behavior_defaults: dict[str, bool] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def matches(self, base_url: str) -> bool:
        parsed = urlparse(base_url)
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port or _default_port_for_scheme(parsed.scheme)
        path = parsed.path or "/"

        if self.schemes and scheme not in self.schemes:
            return False
        if self.host_equals and host not in self.host_equals:
            return False
        if self.host_suffixes and not any(_host_matches(host, suffix) for suffix in self.host_suffixes):
            return False
        if self.ports and port not in self.ports:
            return False
        if self.path_prefixes and not any(path.startswith(prefix) for prefix in self.path_prefixes):
            return False
        return True


def _profile_from_json(path: Path, raw: dict[str, Any]) -> EndpointProfile:
    match = raw.get("match") if isinstance(raw.get("match"), dict) else {}
    models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
    chat = raw.get("chat") if isinstance(raw.get("chat"), dict) else {}
    health = raw.get("health") if isinstance(raw.get("health"), dict) else {}
    behavior = raw.get("behavior") if isinstance(raw.get("behavior"), dict) else {}
    # parse optional ttfb_timeout from chat section
    ttfb_timeout_val = None
    if "ttfb_timeout" in chat:
        try:
            ttfb_timeout_val = float(chat.get("ttfb_timeout"))
        except (TypeError, ValueError):
            ttfb_timeout_val = None

    return EndpointProfile(
        id=str(raw.get("id") or path.stem),
        source_path=str(path),
        schemes=_normalize_string_list(match.get("schemes")),
        host_equals=_normalize_string_list(match.get("host_equals")),
        host_suffixes=_normalize_string_list(match.get("host_suffixes")),
        ports=_normalize_port_list(match.get("ports")),
        path_prefixes=tuple(str(value).strip() for value in match.get("path_prefixes", []) if str(value).strip()),
        models_method=str(models.get("method") or "GET").upper(),
        models_path=str(models.get("path") or "models"),
        models_format=str(models.get("format") or "openai"),
        models_variables={
            str(name): {
                "method": str(config.get("method") or "GET").upper(),
                "path": str(config.get("path") or ""),
                "json_path": str(config.get("json_path") or ""),
                "strip_prefix": str(config.get("strip_prefix") or ""),
            }
            for name, config in (models.get("variables") or {}).items()
            if isinstance(name, str) and isinstance(config, dict)
        },
        models_items_path=str(models.get("items_path") or ""),
        models_fields={
            str(name): str(value)
            for name, value in (models.get("fields") or {}).items()
            if isinstance(name, str) and isinstance(value, (str, int, float))
        },
        models_owned_by=str(models.get("owned_by")) if models.get("owned_by") else None,
        models_capabilities={
            str(name): value
            for name, value in (models.get("capabilities") or {}).items()
            if isinstance(name, str)
        },
        chat_path=str(chat.get("path") or "chat/completions"),
        chat_streaming=str(chat.get("streaming") or "sse").strip().lower(),
        chat_tools=str(chat.get("tools") or "trial").strip().lower(),
        chat_system_prompt=str(chat.get("system_prompt") or "supported").strip().lower(),
        ttfb_timeout=ttfb_timeout_val,
        health_mode=str(health.get("mode") or "internal-ready"),
        health_path=str(health.get("path")) if health.get("path") else None,
        health_method=str(health.get("method") or "GET").upper(),
        behavior_defaults={
            str(key): bool(value)
            for key, value in behavior.items()
            if isinstance(value, bool)
        },
        raw=raw,
    )


@lru_cache(maxsize=1)
def load_endpoint_profiles() -> tuple[EndpointProfile, ...]:
    profiles: list[EndpointProfile] = []
    if not _PROFILE_DIR.exists():
        return ()
    for path in sorted(_PROFILE_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("profile must contain a JSON object")
            profiles.append(_profile_from_json(path, raw))
        except Exception as exc:
            logger.warning("endpoint profile: skipping %s: %s", path, exc)
    return tuple(profiles)


def resolve_endpoint_profile(base_url: str) -> EndpointProfile | None:
    for profile in load_endpoint_profiles():
        if profile.matches(base_url):
            return profile
    return None