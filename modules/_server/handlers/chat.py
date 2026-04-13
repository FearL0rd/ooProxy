"""Handler for POST /api/chat."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from modules._server.translate.request import chat_to_openai, direct_display_tool_reply
from modules._server.translate.response import openai_chat_to_ollama
from modules._server.translate.stream import sse_to_ndjson
from modules._server.upstream_errors import (
    assistant_error_text,
    iter_ollama_chat_error_stream,
    synthetic_ollama_chat,
)

logger = logging.getLogger("ooproxy")


def _native_behavior_flags(request: Request, model: str) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    endpoint_profile = getattr(request.app.state, "endpoint_profile", None)
    if endpoint_profile is not None:
        flags.update(getattr(endpoint_profile, "behavior_defaults", {}) or {})
    behavior = getattr(request.app.state, "behavior", None)
    base_url = getattr(request.app.state, "base_url", "")
    if behavior and base_url and model:
        flags.update(behavior.get_flags(base_url, model))
    return flags


async def _record_native_behavior_flags(request: Request, model: str, observed_flags: set[str]) -> None:
    behavior = getattr(request.app.state, "behavior", None)
    base_url = getattr(request.app.state, "base_url", "")
    if not behavior or not base_url or not model:
        return
    for flag in sorted(observed_flags):
        await behavior.record(base_url, model, flag)


async def chat_handler(request: Request) -> StreamingResponse | JSONResponse:
    """POST /api/chat — translate and proxy to remote chat completions."""
    body = await request.json()
    client = request.app.state.client
    openai_body = chat_to_openai(body)
    model = body.get("model", "")
    streaming = body.get("stream", False)
    behavior_flags = _native_behavior_flags(request, model)
    direct_reply = direct_display_tool_reply(body)

    if direct_reply is not None:
        if streaming:
            return StreamingResponse(iter_ollama_chat_error_stream(model, direct_reply), media_type="application/x-ndjson")
        return JSONResponse(synthetic_ollama_chat(model, direct_reply))

    logger.info("api/chat model=%s stream=%s msgs=%d",
                model, streaming, len(body.get("messages", [])))

    if streaming:
        async def generate():
            observed_flags: set[str] = set()
            try:
                async with client.stream_chat(openai_body) as lines:
                    async for chunk in sse_to_ndjson(
                        lines,
                        model,
                        behavior_flags=behavior_flags,
                        observed_flags=observed_flags,
                    ):
                        yield chunk
                if observed_flags:
                    await _record_native_behavior_flags(request, model, observed_flags)
            except Exception as exc:
                logger.error("api/chat upstream error model=%s: %s", model, exc)
                async for chunk in iter_ollama_chat_error_stream(model, assistant_error_text(exc, model)):
                    yield chunk

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    try:
        data = await client.chat(openai_body)
    except Exception as exc:
        logger.error("api/chat upstream error model=%s: %s", model, exc)
        return JSONResponse(synthetic_ollama_chat(model, assistant_error_text(exc, model)))
    usage = data.get("usage") or {}
    finish = ((data.get("choices") or [{}])[0]).get("finish_reason", "?")
    logger.info("api/chat ← model=%s finish=%s prompt=%d compl=%d",
                model, finish,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    observed_flags: set[str] = set()
    translated = openai_chat_to_ollama(
        data,
        model,
        behavior_flags=behavior_flags,
        observed_flags=observed_flags,
    )
    if observed_flags:
        await _record_native_behavior_flags(request, model, observed_flags)
    return JSONResponse(translated)
