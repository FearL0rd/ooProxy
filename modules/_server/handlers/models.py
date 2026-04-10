"""Handlers for model-listing and model-info endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse

from modules._server.translate.models import openai_models_to_ollama_tags, openai_model_to_ollama_show


async def tags_handler(request: Request) -> JSONResponse:
    """GET /api/tags — list available models in Ollama format."""
    client = request.app.state.client
    data = await client.get_models()
    return JSONResponse(openai_models_to_ollama_tags(data))


async def ps_handler(request: Request) -> JSONResponse:
    """GET /api/ps — list 'running' models (synthesized from remote list)."""
    client = request.app.state.client
    data = await client.get_models()
    tags = openai_models_to_ollama_tags(data)
    # Add size_vram field expected by some clients
    for model in tags["models"]:
        model["size_vram"] = 0
        model["expires_at"] = "0001-01-01T00:00:00Z"
    return JSONResponse({"models": tags["models"]})


async def show_handler(request: Request) -> JSONResponse:
    """POST /api/show — return synthesized model info."""
    body = await request.json()
    model_id = body.get("model") or body.get("name", "")
    return JSONResponse(openai_model_to_ollama_show(model_id))
