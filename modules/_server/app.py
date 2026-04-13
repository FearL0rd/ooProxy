"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from modules._server.behavior import BehaviorCache
from modules._server.client import OpenAIClient
from modules._server.config import ProxyConfig
from modules._server.handlers.chat import chat_handler
from modules._server.handlers.embeddings import embeddings_handler
from modules._server.handlers.generate import generate_handler
from modules._server.handlers.models import ps_handler, show_handler, tags_handler
from modules._server.handlers.openai_compat import (
    v1_chat_handler,
    v1_embeddings_handler,
    v1_messages_handler,
    v1_models_handler,
    v1_responses_handler,
)
from modules._server.handlers.stubs import (
    blobs_handler,
    copy_handler,
    create_handler,
    delete_handler,
    pull_handler,
    push_handler,
)
from modules._server.handlers.version import root_handler, version_handler

logger = logging.getLogger("ooproxy")


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class _RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with a short request-id, method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "[%s] %s %s ERROR %.0fms — %s",
                request_id, request.method, request.url.path, duration_ms, exc,
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[%s] %s %s %d %.0fms",
            request_id, request.method, request.url.path,
            response.status_code, duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def _v1_router() -> APIRouter:
    """OpenAI-compatible layer — used by VS Code Copilot Chat and direct clients."""
    r = APIRouter(prefix="/v1")
    _nr = dict(response_model=None)
    r.add_api_route("/chat/completions", v1_chat_handler,   methods=["POST"], **_nr)
    r.add_api_route("/messages",         v1_messages_handler, methods=["POST"], **_nr)
    r.add_api_route("/responses",        v1_responses_handler, methods=["POST"], **_nr)
    r.add_api_route("/models",           v1_models_handler,  methods=["GET"],  **_nr)
    r.add_api_route("/embeddings",       v1_embeddings_handler, methods=["POST"], **_nr)
    return r


def _api_router() -> APIRouter:
    """Ollama-native /api/* layer."""
    r = APIRouter(prefix="/api")
    _nr = dict(response_model=None)
    # Tier 1 — VS Code Copilot core
    r.add_api_route("/version",           version_handler,   methods=["GET"],          **_nr)
    r.add_api_route("/tags",              tags_handler,      methods=["GET"],          **_nr)
    r.add_api_route("/chat",              chat_handler,      methods=["POST"],         **_nr)
    # Tier 2 — Open WebUI / extended support
    r.add_api_route("/generate",          generate_handler,  methods=["POST"],         **_nr)
    r.add_api_route("/embeddings",        embeddings_handler, methods=["POST"],        **_nr)
    r.add_api_route("/ps",                ps_handler,        methods=["GET"],          **_nr)
    r.add_api_route("/show",              show_handler,      methods=["POST"],         **_nr)
    # Tier 3 — model management stubs (no-op responses)
    r.add_api_route("/pull",              pull_handler,      methods=["POST"],         **_nr)
    r.add_api_route("/delete",            delete_handler,    methods=["DELETE"],       **_nr)
    r.add_api_route("/copy",              copy_handler,      methods=["POST"],         **_nr)
    r.add_api_route("/create",            create_handler,    methods=["POST"],         **_nr)
    r.add_api_route("/push",              push_handler,      methods=["POST"],         **_nr)
    r.add_api_route("/blobs/{digest:path}", blobs_handler,   methods=["HEAD", "POST"], **_nr)
    return r


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

async def _healthz(request: Request) -> JSONResponse:
    """Liveness probe — returns 200 as long as the process is running."""
    return JSONResponse({"status": "ok"})


async def _readyz(request: Request) -> JSONResponse:
    """Readiness probe — returns 200 once the upstream client is initialized."""
    client = getattr(request.app.state, "client", None)
    if client is None:
        return JSONResponse({"status": "not ready", "reason": "client not initialized"}, status_code=503)
    return JSONResponse({"status": "ready"})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app(config: ProxyConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = OpenAIClient(config)
        app.state.base_url = config.url     # canonical upstream URL for behavior cache keys
        app.state.behavior = BehaviorCache()
        app.state.responses_store = {}      # explicit — avoids lazy init scattered in handlers
        yield
        await app.state.client.aclose()

    app = FastAPI(title="ooProxy", lifespan=lifespan)
    app.add_middleware(_RequestLoggingMiddleware)

    _nr = dict(response_model=None)
    app.add_api_route("/",       root_handler, methods=["GET", "HEAD"], **_nr)
    app.add_api_route("/healthz", _healthz,    methods=["GET"],         **_nr)
    app.add_api_route("/readyz",  _readyz,     methods=["GET"],         **_nr)

    app.include_router(_v1_router())
    app.include_router(_api_router())

    return app
