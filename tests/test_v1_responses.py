from __future__ import annotations

import httpx
import json
import unittest

from fastapi.testclient import TestClient

from modules._server.app import create_app
from modules._server.behavior import BehaviorCache
from modules._server.config import ProxyConfig


class _DummyStream:
    async def aiter_lines(self):
        yield 'data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        yield 'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}'
        yield 'data: [DONE]'

    async def aclose(self):
        return None


class _DummyClient:
    def __init__(self) -> None:
        self.last_chat_body: dict | None = None
        self.last_stream_body: dict | None = None

    async def chat(self, body: dict) -> dict:
        self.last_chat_body = body
        return {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    async def open_stream_chat(self, body: dict):
        self.last_stream_body = body
        return _DummyStream()

    async def get_models(self) -> dict:
        return {"data": []}

    async def embeddings(self, body: dict) -> dict:
        return {"data": []}

    async def aclose(self) -> None:
        return None


class _ToolCallingClient(_DummyClient):
    async def chat(self, body: dict) -> dict:
        self.last_chat_body = body
        return {
            "id": "chatcmpl_tool_test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_search_1",
                                "type": "function",
                                "function": {
                                    "name": "Search",
                                    "arguments": '{"pattern":"tools/*"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


class _FailingStreamClient(_DummyClient):
    async def chat(self, body: dict) -> dict:
        self.last_chat_body = body
        raise RuntimeError("simulated upstream stream failure")


class _RetryingStreamClient(_DummyClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def chat(self, body: dict) -> dict:
        self.calls.append(body)
        self.last_chat_body = body
        if "tools" in body:
            request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
            response = httpx.Response(400, request=request, text='{"error":"tool choice requires supported tools"}')
            raise httpx.HTTPStatusError("tool choice requires supported tools", request=request, response=response)
        return {
            "id": "chatcmpl_retry_test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }


class V1ResponsesCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(ProxyConfig(url="https://example.invalid/v1", key="", port=11434))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.dummy = _DummyClient()
        self.app.state.client = self.dummy
        self.app.state.behavior = BehaviorCache(path=None)  # in-memory only — no disk state between tests

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_non_streaming_response_shape(self) -> None:
        response = self.client.post("/v1/responses", json={"model": "demo", "input": "ping"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output_text"], "pong")
        self.assertEqual(self.dummy.last_chat_body["messages"][-1]["content"], "ping")

    def test_previous_response_id_replays_history(self) -> None:
        first = self.client.post("/v1/responses", json={"model": "demo", "input": "ping"}).json()

        response = self.client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "previous_response_id": first["id"],
                "input": "again",
            },
        )

        self.assertEqual(response.status_code, 200)
        history_messages = self.dummy.last_chat_body["messages"]
        self.assertEqual(
            [message["content"] for message in history_messages if message["role"] != "system"],
            ["ping", "pong", "again"],
        )

    def test_streaming_response_emits_response_events(self) -> None:
        with self.client.stream("POST", "/v1/responses", json={"model": "demo", "input": "stream me", "stream": True}) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: response.created", body)
        self.assertIn("event: response.output_text.delta", body)
        self.assertIn("event: response.completed", body)
        self.assertTrue('"delta": "pong"' in body or '"delta":"pong"' in body)
        self.assertFalse(self.dummy.last_chat_body.get("stream", False))
        self.assertNotIn("stream_options", self.dummy.last_chat_body)

    def test_streaming_error_returns_error_event(self) -> None:
        self.app.state.client = _FailingStreamClient()

        with self.client.stream("POST", "/v1/responses", json={"model": "demo", "input": "stream me", "stream": True}) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", body)
        self.assertIn("event: response.failed", body)
        self.assertIn("simulated upstream stream failure", body)

    def test_streaming_retries_without_stream_options(self) -> None:
        retrying_client = _RetryingStreamClient()
        self.app.state.client = retrying_client

        with self.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "demo",
                "input": "stream me",
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "Search", "parameters": {"type": "object", "properties": {}, "required": []}}}],
                "tool_choice": "auto",
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(retrying_client.calls), 2)
        self.assertIn("tools", retrying_client.calls[0])
        self.assertNotIn("tools", retrying_client.calls[1])
        self.assertIn("event: response.completed", body)
        self.assertIn("fallback worked", body)

    def test_responses_function_tools_are_converted_for_chat_completions(self) -> None:
        self.app.state.client = _ToolCallingClient()

        response = self.client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "show me the files in tools/",
                "tools": [
                    {
                        "type": "function",
                        "name": "Search",
                        "description": "Search files by glob or pattern.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pattern": {"type": "string"},
                            },
                            "required": ["pattern"],
                        },
                    }
                ],
                "tool_choice": {"type": "function", "name": "Search"},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["output"][0]["name"], "Search")
        self.assertIn("tools", self.app.state.client.last_chat_body)
        self.assertEqual(self.app.state.client.last_chat_body["tools"][0]["function"]["name"], "Search")
        self.assertEqual(
            self.app.state.client.last_chat_body["tool_choice"],
            {"type": "function", "function": {"name": "Search"}},
        )


if __name__ == "__main__":
    unittest.main()