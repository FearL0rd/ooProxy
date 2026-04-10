"""Handler for GET /api/version."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

OLLAMA_VERSION = "0.6.5"


async def version_handler(request: Request) -> JSONResponse:
    return JSONResponse({"version": OLLAMA_VERSION})
