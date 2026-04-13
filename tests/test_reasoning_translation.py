from __future__ import annotations

import json
import unittest

from modules._server.translate.response import openai_chat_to_ollama
from modules._server.translate.stream import sse_to_ndjson


class ReasoningTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_to_ndjson_wraps_reasoning_before_tool_calls(self) -> None:
        async def stream():
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "We"}, "finish_reason": None}]})
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": " need"}, "finish_reason": None}]})
            yield "data: " + json.dumps({
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read_file", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            })
            yield "data: " + json.dumps({
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": json.dumps({"path": "cleanup.sh"})},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            })
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}})
            yield 'data: [DONE]'

        chunks = []
        async for raw in sse_to_ndjson(stream(), "demo"):
            chunks.append(json.loads(raw.decode("utf-8")))

        contents = [chunk["message"]["content"] for chunk in chunks if "message" in chunk and "content" in chunk["message"]]
        self.assertEqual(contents[:4], ["<think>", "We", " need", "</think>"])

        tool_chunk = next(chunk for chunk in chunks if chunk.get("message", {}).get("tool_calls"))
        self.assertEqual(tool_chunk["message"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            tool_chunk["message"]["tool_calls"][0]["function"]["arguments"],
            {"path": "cleanup.sh"},
        )

        done_chunk = chunks[-1]
        self.assertTrue(done_chunk["done"])
        self.assertEqual(done_chunk["done_reason"], "tool_calls")

    async def test_sse_to_ndjson_closes_reasoning_before_normal_content(self) -> None:
        async def stream():
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "Think first"}, "finish_reason": None}]})
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "Answer"}, "finish_reason": None}]})
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2}})
            yield 'data: [DONE]'

        chunks = []
        async for raw in sse_to_ndjson(stream(), "demo"):
            chunks.append(json.loads(raw.decode("utf-8")))

        contents = [chunk["message"]["content"] for chunk in chunks if "message" in chunk and "content" in chunk["message"]]
        self.assertEqual(contents[:4], ["<think>", "Think first", "</think>", "Answer"])


class NonStreamingReasoningTranslationTests(unittest.TestCase):
    def test_openai_chat_to_ollama_preserves_reasoning_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Need to inspect file.",
                        "content": "Here is the file.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }

        translated = openai_chat_to_ollama(payload, "demo")

        self.assertEqual(
            translated["message"]["content"],
            "<think>Need to inspect file.</think>Here is the file.",
        )


if __name__ == "__main__":
    unittest.main()