"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

logger = logging.getLogger("ooproxy")

from modules._server.client import OpenAIClient
from modules._server.config import ProxyConfig
from modules._server.handlers.chat import chat_handler
from modules._server.handlers.embeddings import embeddings_handler
from modules._server.handlers.generate import generate_handler
from modules._server.handlers.models import ps_handler, show_handler, tags_handler
from modules._server.handlers.stubs import (
    blobs_handler,
    copy_handler,
    create_handler,
    delete_handler,
    pull_handler,
    push_handler,
)
from modules._server.handlers.openai_compat import v1_chat_handler, v1_embeddings_handler, v1_models_handler
from modules._server.handlers.version import version_handler


def create_app(config: ProxyConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = OpenAIClient(config)
        yield
        await app.state.client.aclose()

    app = FastAPI(title="ooProxy", lifespan=lifespan)

    _r = dict(response_model=None)

    # OpenAI-compatible layer (used by VS Code Copilot Chat)
    app.add_api_route("/v1/chat/completions", v1_chat_handler, methods=["POST"], **_r)
    app.add_api_route("/v1/models", v1_models_handler, methods=["GET"], **_r)
    app.add_api_route("/v1/embeddings", v1_embeddings_handler, methods=["POST"], **_r)

    # Tier 1 — VS Code Copilot core
    app.add_api_route("/api/version", version_handler, methods=["GET"], **_r)
    app.add_api_route("/api/tags", tags_handler, methods=["GET"], **_r)
    app.add_api_route("/api/chat", chat_handler, methods=["POST"], **_r)

    # Tier 2 — Open WebUI / extended support
    app.add_api_route("/api/generate", generate_handler, methods=["POST"], **_r)
    app.add_api_route("/api/embeddings", embeddings_handler, methods=["POST"], **_r)
    app.add_api_route("/api/ps", ps_handler, methods=["GET"], **_r)
    app.add_api_route("/api/show", show_handler, methods=["POST"], **_r)

    # Tier 3 — model management stubs
    app.add_api_route("/api/pull", pull_handler, methods=["POST"], **_r)
    app.add_api_route("/api/delete", delete_handler, methods=["DELETE"], **_r)
    app.add_api_route("/api/copy", copy_handler, methods=["POST"], **_r)
    app.add_api_route("/api/create", create_handler, methods=["POST"], **_r)
    app.add_api_route("/api/push", push_handler, methods=["POST"], **_r)
    app.add_api_route("/api/blobs/{digest:path}", blobs_handler, methods=["HEAD", "POST"], **_r)

    return app
