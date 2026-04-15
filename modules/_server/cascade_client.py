"""Cascade routing client for weak/strong model pairs."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from modules._server.client import OpenAIClient
from modules._server.config import CascadeConfig, CascadeRouteConfig, ProxyConfig, render_cascade_decision_prompt

logger = logging.getLogger("ooproxy")

_DECISION_RETRY_MAX_TOKENS = 1024
_DECISION_LOG_PREVIEW_CHARS = 400
_ANSI_RESET = "\033[0m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"


def _has_tool_continuation(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            return True
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def _summarize_tools(tools: object) -> str:
    if not isinstance(tools, list) or not tools:
        return "none"
    items: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or "").strip()
        if description:
            items.append(f"{name}: {description}")
        else:
            items.append(name)
    return "; ".join(items) if items else "none"


def _text_from_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                for key in ("text", "content", "reasoning", "reasoning_content"):
                    nested = item.get(key)
                    if isinstance(nested, str) and nested.strip():
                        parts.append(nested.strip())
                        break
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _parse_json_object(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("decision payload was not a JSON object")


def _supports_color() -> bool:
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _format_confidence(value: float | None, threshold: float, *, state: str = "parsed") -> str:
    if value is None:
        label = "unparsed" if state == "failed" else state
        if not _supports_color():
            return label
        return f"{_ANSI_YELLOW}{label}{_ANSI_RESET}"
    text = f"{value:.2f}"
    if not _supports_color():
        return text
    color = _ANSI_GREEN if value >= threshold else _ANSI_RED
    return f"{color}{text}{_ANSI_RESET}"


class _RewrittenStream:
    def __init__(self, upstream, client_model: str) -> None:
        self._upstream = upstream
        self._client_model = client_model

    async def aiter_lines(self):
        async for line in self._upstream.aiter_lines():
            if not isinstance(line, str) or not line.startswith("data: ") or line == "data: [DONE]":
                yield line
                continue
            payload = line[6:]
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(chunk, dict):
                chunk["model"] = self._client_model
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}"
                continue
            yield line

    async def aclose(self) -> None:
        await self._upstream.aclose()

    def __getattr__(self, name: str):
        return getattr(self._upstream, name)


class CascadeClient:
    def __init__(self, config: ProxyConfig, client_factory=OpenAIClient) -> None:
        if config.cascade is None:
            raise ValueError("cascade configuration is required")
        self._config = config.cascade
        self._client_factory = client_factory
        self.endpoint_profile = None
        self._routes = {route.weak_model: route for route in self._config.routes}
        self._clients: dict[tuple[str, str], OpenAIClient] = {}
        self._weak_model_cache: dict[tuple[str, str], dict] = {}
        logger.info(
            "cascade initialized routes=%d weak_models=%s",
            len(self._config.routes),
            ", ".join(sorted(self._routes)),
        )

    def _client_for(self, url: str, key: str):
        cache_key = (url, key)
        client = self._clients.get(cache_key)
        if client is None:
            client_config = ProxyConfig(url=url, key=key, port=self._config.port, host=self._config.host)
            client = self._client_factory(client_config)
            self._clients[cache_key] = client
        return client

    def _route_for_model(self, model: str) -> CascadeRouteConfig:
        route = self._routes.get(str(model or "").strip())
        if route is None:
            raise RuntimeError(f"cascade model is not configured: {model}")
        logger.info(
            "cascade request model=%s weak=%s strong=%s",
            model,
            route.weak_model,
            route.strong_model,
        )
        return route

    def _body_with_model(self, body: dict, model: str) -> dict:
        updated = dict(body)
        updated["model"] = model
        return updated

    def _decision_body(self, body: dict, route: CascadeRouteConfig, *, compact: bool = False, max_tokens: int | None = None) -> dict:
        decision_input = {key: value for key, value in body.items() if key not in {"stream", "stream_options"}}
        tool_summary = _summarize_tools(decision_input.get("tools"))
        tool_choice_text = json.dumps(decision_input.get("tool_choice"), ensure_ascii=False, sort_keys=True) if decision_input.get("tool_choice") is not None else "none"
        request_json_text = json.dumps(decision_input, ensure_ascii=False, sort_keys=True)
        template = (
            self._config.decision.retry_user_prompt_template
            if compact
            else self._config.decision.user_prompt_template
        )
        user_prompt = render_cascade_decision_prompt(
            template,
            user_prompt=request_json_text,
            weak_model=route.weak_model,
            strong_model=route.strong_model,
            available_tools=tool_summary,
            tool_choice=tool_choice_text,
            request_json=request_json_text,
        )
        return {
            "model": route.weak_model,
            "messages": [
                {"role": "system", "content": self._config.decision.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens if max_tokens is not None else self._config.decision.max_tokens,
        }

    def _decision_text(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        for source in (
            message.get("content"),
            message.get("reasoning"),
            message.get("reasoning_content"),
            choice.get("reasoning"),
            choice.get("reasoning_content"),
        ):
            extracted = _text_from_value(source)
            if extracted:
                return extracted
        return ""

    def _parse_decision_from_payload(self, payload: dict) -> tuple[float, str]:
        decision_text = self._decision_text(payload)
        preview = decision_text[:_DECISION_LOG_PREVIEW_CHARS]
        logger.debug(
            "cascade raw decision weak=%s chars=%d truncated=%s text=%r",
            payload.get("model", "?"),
            len(decision_text),
            len(decision_text) > len(preview),
            preview,
        )
        return self._parse_decision(decision_text)

    def _parse_decision(self, raw_text: str) -> tuple[float, str]:
        candidate = raw_text.strip()
        if not candidate:
            raise ValueError("empty decision output")
        if "```" in candidate:
            parts = [part.strip() for part in candidate.split("```") if part.strip()]
            candidate = next((part for part in parts if "{" in part and "}" in part), candidate)
        if "{" in candidate and "}" in candidate:
            candidate = candidate[candidate.find("{"):candidate.rfind("}") + 1]
        payload = _parse_json_object(candidate)
        if "CONFIDENCE" not in payload:
            raise ValueError("decision JSON missing CONFIDENCE")
        confidence = float(payload.get("CONFIDENCE"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"decision confidence out of range: {confidence}")
        return confidence, "json"

    async def _choose_target(self, body: dict, route: CascadeRouteConfig) -> str:
        confidence_text = _format_confidence(None, self._config.decision.threshold, state="skipped")
        if _has_tool_continuation(body.get("messages")):
            logger.info(
                "cascade route selected target=weak model=%s confidence=%s reason=tool continuation",
                route.weak_model,
                confidence_text,
            )
            return "weak"

        weak_client = self._client_for(route.weak_url, route.weak_key)
        decision_attempts = (
            ("primary", self._decision_body(body, route)),
            (
                "retry",
                self._decision_body(
                    body,
                    route,
                    compact=True,
                    max_tokens=max(_DECISION_RETRY_MAX_TOKENS, self._config.decision.max_tokens),
                ),
            ),
        )
        for attempt_name, decision_body in decision_attempts:
            try:
                decision_payload = await asyncio.wait_for(
                    weak_client.chat(decision_body),
                    timeout=self._config.decision.timeout_seconds,
                )
                confidence, reason = self._parse_decision_from_payload(decision_payload)
                target = "weak" if confidence >= self._config.decision.threshold else "strong"
                confidence_text = _format_confidence(confidence, self._config.decision.threshold)
                logger.info(
                    "cascade decision weak=%s strong=%s confidence=%.2f threshold=%.2f target=%s reason=%s attempt=%s",
                    route.weak_model,
                    route.strong_model,
                    confidence,
                    self._config.decision.threshold,
                    target,
                    reason or "-",
                    attempt_name,
                )
                if target == "weak":
                    logger.info(
                        "cascade route selected target=weak model=%s confidence=%s",
                        route.weak_model,
                        confidence_text,
                    )
                    return "weak"
                logger.info(
                    "cascade route selected target=strong model=%s confidence=%s reason=%s",
                    route.strong_model,
                    confidence_text,
                    "confidence below threshold",
                )
                return "strong"
            except Exception as exc:
                logger.warning(
                    "cascade decision %s attempt failed for %s confidence=%s: %s",
                    attempt_name,
                    route.weak_model,
                    _format_confidence(None, self._config.decision.threshold, state="failed"),
                    exc,
                )
        logger.warning(
            "cascade route selected target=strong model=%s confidence=%s reason=decision failure or low confidence",
            route.strong_model,
            _format_confidence(None, self._config.decision.threshold, state="failed"),
        )
        return "strong"

    def _rewrite_model(self, data: dict, client_model: str) -> dict:
        rewritten = dict(data)
        rewritten["model"] = client_model
        return rewritten

    async def _chat_via_route(self, body: dict, route: CascadeRouteConfig, preferred: str, client_model: str) -> dict:
        weak_client = self._client_for(route.weak_url, route.weak_key)
        strong_client = self._client_for(route.strong_url, route.strong_key)
        weak_body = self._body_with_model(body, route.weak_model)
        strong_body = self._body_with_model(body, route.strong_model)

        if preferred == "weak":
            try:
                logger.info("cascade executing target=weak model=%s", route.weak_model)
                return self._rewrite_model(await weak_client.chat(weak_body), client_model)
            except Exception as exc:
                logger.warning("cascade weak call failed for %s; retrying strong: %s", client_model, exc)
        logger.info("cascade executing target=strong model=%s", route.strong_model)
        return self._rewrite_model(await strong_client.chat(strong_body), client_model)

    async def _open_stream_via_route(self, body: dict, route: CascadeRouteConfig, preferred: str, client_model: str):
        weak_client = self._client_for(route.weak_url, route.weak_key)
        strong_client = self._client_for(route.strong_url, route.strong_key)
        weak_body = self._body_with_model(body, route.weak_model)
        strong_body = self._body_with_model(body, route.strong_model)

        if preferred == "weak":
            try:
                logger.info("cascade opening stream target=weak model=%s", route.weak_model)
                return _RewrittenStream(await weak_client.open_stream_chat(weak_body), client_model)
            except Exception as exc:
                logger.warning("cascade weak stream failed for %s; retrying strong: %s", client_model, exc)
        logger.info("cascade opening stream target=strong model=%s", route.strong_model)
        return _RewrittenStream(await strong_client.open_stream_chat(strong_body), client_model)

    async def probe_ready(self) -> tuple[bool, str | None]:
        checked: set[tuple[str, str]] = set()
        for route in self._config.routes:
            for url, key in ((route.weak_url, route.weak_key), (route.strong_url, route.strong_key)):
                cache_key = (url, key)
                if cache_key in checked:
                    continue
                checked.add(cache_key)
                ready, reason = await self._client_for(url, key).probe_ready()
                if not ready:
                    return False, reason
        return True, None

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()

    async def get_models(self) -> dict:
        logger.info("cascade listing weak models count=%d", len(self._config.routes))
        upstream_models: dict[tuple[str, str], dict] = {}
        items: list[dict] = []
        for route in self._config.routes:
            weak_key = (route.weak_url, route.weak_key)
            if weak_key not in upstream_models:
                try:
                    upstream_models[weak_key] = await self._client_for(route.weak_url, route.weak_key).get_models()
                except Exception as exc:
                    logger.warning("cascade weak model metadata lookup failed for %s: %s", route.weak_model, exc)
                    upstream_models[weak_key] = {"data": []}
            candidates = upstream_models[weak_key].get("data", [])
            entry = next(
                (
                    candidate for candidate in candidates
                    if isinstance(candidate, dict) and str(candidate.get("id") or "").strip() == route.weak_model
                ),
                {},
            )
            model_entry = dict(entry)
            model_entry.update(route.metadata)
            model_entry["id"] = route.weak_model
            model_entry.setdefault("object", "model")
            model_entry.setdefault("owned_by", model_entry.get("owned_by") or "ooProxy")
            items.append(model_entry)
        return {"object": "list", "data": items}

    async def chat(self, body: dict) -> dict:
        client_model = str(body.get("model") or "").strip()
        route = self._route_for_model(client_model)
        preferred = await self._choose_target(body, route)
        return await self._chat_via_route(body, route, preferred, client_model)

    @asynccontextmanager
    async def stream_chat(self, body: dict) -> AsyncIterator:
        upstream = await self.open_stream_chat(body)
        try:
            yield upstream.aiter_lines()
        finally:
            await upstream.aclose()

    async def open_stream_chat(self, body: dict):
        client_model = str(body.get("model") or "").strip()
        route = self._route_for_model(client_model)
        preferred = await self._choose_target(body, route)
        return await self._open_stream_via_route(body, route, preferred, client_model)

    async def embeddings(self, body: dict) -> dict:
        client_model = str(body.get("model") or "").strip()
        route = self._route_for_model(client_model)
        weak_client = self._client_for(route.weak_url, route.weak_key)
        return await weak_client.embeddings(self._body_with_model(body, route.weak_model))