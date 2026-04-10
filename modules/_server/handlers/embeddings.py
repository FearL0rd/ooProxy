"""Handler for POST /api/embeddings."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from modules._server.translate.request import embeddings_to_openai
from modules._server.translate.response import openai_embeddings_to_ollama


async def embeddings_handler(request: Request) -> JSONResponse:
    """POST /api/embeddings — translate and proxy to remote embeddings."""
    body = await request.json()
    client = request.app.state.client
    openai_body = embeddings_to_openai(body)
    data = await client.embeddings(openai_body)
    return JSONResponse(openai_embeddings_to_ollama(data))
