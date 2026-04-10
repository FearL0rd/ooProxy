"""Handler for POST /api/chat."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from modules._server.translate.request import chat_to_openai
from modules._server.translate.response import openai_chat_to_ollama
from modules._server.translate.stream import sse_to_ndjson


async def chat_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /api/chat — translate and proxy to remote chat completions."""
    body = await request.json()
    client = request.app.state.client
    openai_body = chat_to_openai(body)
    model = body.get("model", "")
    streaming = body.get("stream", False)

    if streaming:
        async def generate():
            async with client.stream_chat(openai_body) as lines:
                async for chunk in sse_to_ndjson(lines, model):
                    yield chunk

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    data = await client.chat(openai_body)
    return JSONResponse(openai_chat_to_ollama(data, model))
