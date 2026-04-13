from __future__ import annotations

import unittest

from modules._server.client import OpenAIClient, _TracedStreamResponse
from modules._server.config import ProxyConfig


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


class ClientStreamTraceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = OpenAIClient(ProxyConfig(url="https://example.invalid/v1", key="", port=11434))

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_stream_chat_logs_raw_sse_lines_when_debug_enabled(self) -> None:
        response = _FakeStreamResponse([
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: [DONE]',
        ])

        async def fake_open_stream(path: str, body: dict):
            self.assertEqual(path, "chat/completions")
            self.assertEqual(body, {"model": "demo", "stream": True})
            return response

        self.client._open_stream = fake_open_stream  # type: ignore[method-assign]

        with self.assertLogs("ooproxy", level="DEBUG") as logs:
            async with self.client.stream_chat({"model": "demo", "stream": True}) as lines:
                collected = [line async for line in lines]

        self.assertEqual(
            collected,
            ['data: {"choices":[{"delta":{"content":"hello"}}]}', 'data: [DONE]'],
        )
        self.assertTrue(response.closed)
        self.assertTrue(any('upstream SSE ← data: {"choices":[{"delta":{"content":"hello"}}]}' in entry for entry in logs.output))
        self.assertTrue(any("upstream SSE ← data: [DONE]" in entry for entry in logs.output))

    async def test_open_stream_chat_returns_traced_response(self) -> None:
        response = _FakeStreamResponse(['data: {"choices":[]}'])

        async def fake_open_stream(path: str, body: dict):
            self.assertEqual(path, "chat/completions")
            self.assertEqual(body, {"model": "demo", "stream": True})
            return response

        self.client._open_stream = fake_open_stream  # type: ignore[method-assign]

        traced = await self.client.open_stream_chat({"model": "demo", "stream": True})

        self.assertIsInstance(traced, _TracedStreamResponse)

        with self.assertLogs("ooproxy", level="DEBUG") as logs:
            collected = [line async for line in traced.aiter_lines()]

        self.assertEqual(collected, ['data: {"choices":[]}'])
        self.assertTrue(any('upstream SSE ← data: {"choices":[]}' in entry for entry in logs.output))
        await traced.aclose()
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()