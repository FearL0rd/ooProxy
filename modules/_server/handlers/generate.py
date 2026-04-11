"""Handler for POST /api/generate."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from modules._server.translate.request import generate_to_openai
from modules._server.translate.response import openai_generate_to_ollama
from modules._server.translate.stream import sse_to_generate_ndjson

logger = logging.getLogger("ooproxy")


async def generate_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /api/generate — wrap prompt as chat, proxy, return generate format."""
    body = await request.json()
    client = request.app.state.client
    openai_body = generate_to_openai(body)
    model = body.get("model", "")
    streaming = body.get("stream", False)

    logger.info("api/generate model=%s stream=%s", model, streaming)

    if streaming:
        async def generate():
            async with client.stream_chat(openai_body) as lines:
                async for chunk in sse_to_generate_ndjson(lines, model):
                    yield chunk

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    data = await client.chat(openai_body)
    usage = data.get("usage") or {}
    finish = ((data.get("choices") or [{}])[0]).get("finish_reason", "?")
    logger.info("api/generate ← model=%s finish=%s prompt=%d compl=%d",
                model, finish,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    return JSONResponse(openai_generate_to_ollama(data, model))
