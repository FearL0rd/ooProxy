from __future__ import annotations

import unittest

import httpx

from modules._server.client import OpenAIClient
from modules._server.config import ProxyConfig
from modules._server.endpoint_profiles import resolve_endpoint_profile


class EndpointProfileResolutionTests(unittest.TestCase):
    def test_resolves_openrouter_profile_from_api_base(self) -> None:
        profile = resolve_endpoint_profile("https://openrouter.ai/api/v1")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.id, "openrouter")
        self.assertEqual(profile.models_path, "models")
        self.assertEqual(profile.models_format, "openai")
        self.assertEqual(profile.chat_path, "chat/completions")
        self.assertEqual(profile.chat_streaming, "sse")
        self.assertEqual(profile.chat_tools, "supported")
        self.assertEqual(profile.chat_system_prompt, "supported")

    def test_resolves_together_profile_from_host(self) -> None:
        profile = resolve_endpoint_profile("https://api.together.xyz/v1")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.id, "together_ai")
        self.assertEqual(profile.models_path, "models")
        self.assertEqual(profile.models_format, "array")
        self.assertEqual(profile.chat_streaming, "sse")
        self.assertEqual(profile.chat_tools, "trial")
        self.assertEqual(profile.chat_system_prompt, "supported")

    def test_resolves_local_ollama_profile_from_host_and_port(self) -> None:
        profile = resolve_endpoint_profile("http://localhost:11434/v1")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.id, "local_ollama")
        self.assertEqual(profile.models_path, "/api/tags")
        self.assertEqual(profile.models_format, "ollama_tags")
        self.assertEqual(profile.chat_streaming, "ndjson")
        self.assertEqual(profile.chat_tools, "native")
        self.assertEqual(profile.health_method, "GET")


class EndpointProfileClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_models_normalizes_together_array_payload(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.together.xyz/v1/models")
            return httpx.Response(
                200,
                request=request,
                json=[{"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "created": 1}],
            )

        client = OpenAIClient(ProxyConfig(url="https://api.together.xyz/v1", key="", port=11434))
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            data = await client.get_models()
        finally:
            await client.aclose()

        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        self.assertEqual(data["data"][0]["object"], "model")

    async def test_get_models_normalizes_local_ollama_tags_payload(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "http://localhost:11434/api/tags")
            return httpx.Response(
                200,
                request=request,
                json={"models": [{"name": "llama3.2:latest"}]},
            )

        client = OpenAIClient(ProxyConfig(url="http://localhost:11434/v1", key="", port=11434))
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            data = await client.get_models()
        finally:
            await client.aclose()

        self.assertEqual(data, {
            "object": "list",
            "data": [
                {
                    "id": "llama3.2:latest",
                    "object": "model",
                    "created": None,
                    "owned_by": "ollama",
                }
            ],
        })

    async def test_probe_ready_uses_profile_health_endpoint(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url), "http://localhost:11434/api/tags")
            return httpx.Response(200, request=request, json={"models": []})

        client = OpenAIClient(ProxyConfig(url="http://localhost:11434/v1", key="", port=11434))
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            ready, reason = await client.probe_ready()
        finally:
            await client.aclose()

        self.assertTrue(ready)
        self.assertIsNone(reason)

    async def test_get_models_reports_redirect_to_signin_as_upstream_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.together.xyz/v1/models")
            return httpx.Response(
                307,
                request=request,
                headers={"location": "/signin?redirectUrl=%2Fmodels"},
            )

        client = OpenAIClient(ProxyConfig(url="https://api.together.xyz/v1", key="", port=11434))
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        try:
            with self.assertRaisesRegex(RuntimeError, "redirected to /signin"):
                await client.get_models()
        finally:
            await client.aclose()


if __name__ == "__main__":
    unittest.main()