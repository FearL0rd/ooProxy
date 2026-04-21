"""Handlers for model-listing and model-info endpoints."""

from __future__ import annotations

from datetime import datetime, timezone


from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from modules._server.translate.models import openai_models_to_ollama_tags, openai_model_to_ollama_show
from modules._server.endpoint_profiles import resolve_endpoint_profile

logger = logging.getLogger("ooproxy")

def _models_error_response(exc: Exception) -> JSONResponse:
    return JSONResponse({"error": {"message": str(exc), "type": "upstream_error"}}, status_code=502)


async def tags_handler(request: Request) -> JSONResponse:
    """GET /api/tags — list available models in Ollama format."""
    client = request.app.state.client
    cache = getattr(request.app.state, "request_cache", None)
    base_url = getattr(request.app.state, "base_url", None)
    endpoint_id = None
    cache_ttl = 1800
    if base_url:
        profile = resolve_endpoint_profile(base_url)
        if profile:
            endpoint_id = profile.id
            # Look for cache_ttl in root or models section
            ttl_val = None
            if "cache_ttl" in profile.raw:
                ttl_val = profile.raw["cache_ttl"]
            elif "cache_ttl" in (profile.raw.get("models") or {}):
                ttl_val = profile.raw["models"]["cache_ttl"]
            try:
                if ttl_val is not None:
                    cache_ttl = int(ttl_val)
            except Exception:
                pass
    if not endpoint_id:
        endpoint_id = str(base_url or "default")
    cache_key = (endpoint_id, "model_list")
    if cache:
        cached = cache.get(endpoint_id, "model_list")
        if cached is not None:
            logger.info("[cache] Served model list for endpoint '%s' from cache", endpoint_id)
            return JSONResponse(openai_models_to_ollama_tags(cached))
    try:
        data = await client.get_models()
    except Exception as exc:
        return _models_error_response(exc)
    if cache:
        cache.set(endpoint_id, "model_list", data, cache_ttl)
    return JSONResponse(openai_models_to_ollama_tags(data))


async def ps_handler(request: Request) -> JSONResponse:
    """GET /api/ps — list 'running' models (synthesized from remote list)."""
    client = request.app.state.client
    cache = getattr(request.app.state, "request_cache", None)
    base_url = getattr(request.app.state, "base_url", None)
    endpoint_id = None
    cache_ttl = 1800
    if base_url:
        profile = resolve_endpoint_profile(base_url)
        if profile:
            endpoint_id = profile.id
            ttl_val = None
            if "cache_ttl" in profile.raw:
                ttl_val = profile.raw["cache_ttl"]
            elif "cache_ttl" in (profile.raw.get("models") or {}):
                ttl_val = profile.raw["models"]["cache_ttl"]
            try:
                if ttl_val is not None:
                    cache_ttl = int(ttl_val)
            except Exception:
                pass
    if not endpoint_id:
        endpoint_id = str(base_url or "default")
    cache_key = (endpoint_id, "model_list")
    if cache:
        cached = cache.get(endpoint_id, "model_list")
        if cached is not None:
            logger.info("[cache] Served model list for endpoint '%s' from cache", endpoint_id)
            tags = openai_models_to_ollama_tags(cached)
            for model in tags["models"]:
                model["size_vram"] = 0
                model["expires_at"] = "0001-01-01T00:00:00Z"
            return JSONResponse({"models": tags["models"]})
    try:
        data = await client.get_models()
    except Exception as exc:
        return _models_error_response(exc)
    if cache:
        cache.set(endpoint_id, "model_list", data, cache_ttl)
    tags = openai_models_to_ollama_tags(data)
    for model in tags["models"]:
        model["size_vram"] = 0
        model["expires_at"] = "0001-01-01T00:00:00Z"
    return JSONResponse({"models": tags["models"]})


async def show_handler(request: Request) -> JSONResponse:
    """POST /api/show — return synthesized model info."""
    body = await request.json()
    model_id = body.get("model") or body.get("name", "")
    client = request.app.state.client
    cache = getattr(request.app.state, "request_cache", None)
    base_url = getattr(request.app.state, "base_url", None)
    endpoint_id = None
    cache_ttl = 1800
    if base_url:
        profile = resolve_endpoint_profile(base_url)
        if profile:
            endpoint_id = profile.id
            ttl_val = None
            if "cache_ttl" in profile.raw:
                ttl_val = profile.raw["cache_ttl"]
            elif "cache_ttl" in (profile.raw.get("models") or {}):
                ttl_val = profile.raw["models"]["cache_ttl"]
            try:
                if ttl_val is not None:
                    cache_ttl = int(ttl_val)
            except Exception:
                pass
    if not endpoint_id:
        endpoint_id = str(base_url or "default")
    # Use the same cache key as other model list requests
    if cache:
        cached = cache.get(endpoint_id, "model_list")
        if cached is not None:
            logger.info("[cache] Served model list for endpoint '%s' from cache (show_handler)", endpoint_id)
            entry = next(
                (
                    candidate
                    for candidate in cached.get("data", [])
                    if isinstance(candidate, dict) and candidate.get("id") == model_id
                ),
                None,
            )
            return JSONResponse(openai_model_to_ollama_show(model_id, entry=entry))
    entry = None
    try:
        data = await client.get_models()
        if cache:
            cache.set(endpoint_id, "model_list", data, cache_ttl)
        entry = next(
            (
                candidate
                for candidate in data.get("data", [])
                if isinstance(candidate, dict) and candidate.get("id") == model_id
            ),
            None,
        )
    except Exception:
        entry = None
    return JSONResponse(openai_model_to_ollama_show(model_id, entry=entry))
