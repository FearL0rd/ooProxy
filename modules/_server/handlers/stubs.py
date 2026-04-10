"""Stub handlers for Ollama model-management endpoints with no remote equivalent."""

from __future__ import annotations

import json

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


async def pull_handler(request: Request) -> StreamingResponse:
    """POST /api/pull — fake a short pull progress stream."""
    body = await request.json()
    model = body.get("model") or body.get("name", "unknown")

    async def fake_pull():
        for status in [
            {"status": "pulling manifest"},
            {"status": f"pulling {model}"},
            {"status": "verifying sha256 digest"},
            {"status": "writing manifest"},
            {"status": "success"},
        ]:
            yield (json.dumps(status) + "\n").encode()

    return StreamingResponse(fake_pull(), media_type="application/x-ndjson")


async def delete_handler(request: Request) -> Response:
    """DELETE /api/delete — fake success."""
    return Response(status_code=200)


async def copy_handler(request: Request) -> Response:
    """POST /api/copy — fake success."""
    return Response(status_code=200)


async def create_handler(request: Request) -> Response:
    """POST /api/create — fake success."""
    return Response(status_code=200)


async def push_handler(request: Request) -> Response:
    """POST /api/push — fake success."""
    return Response(status_code=200)


async def blobs_handler(request: Request, digest: str) -> Response:
    """HEAD/POST /api/blobs/{digest} — fake success."""
    return Response(status_code=200)
