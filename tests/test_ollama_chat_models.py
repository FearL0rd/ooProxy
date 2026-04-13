from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from tools import ollama_chat


class OllamaChatModelListingTests(unittest.TestCase):
    def test_get_available_models_sorts_native_ollama_results(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "models": [
                {"name": "zeta"},
                {"name": "alpha"},
                {"name": "Beta"},
            ]
        }

        with patch("tools.ollama_chat.requests.get", return_value=response):
            models = ollama_chat.get_available_models("http://localhost:11434")

        self.assertEqual(models, ["alpha", "Beta", "zeta"])

    def test_get_available_models_sorts_openai_fallback_results(self) -> None:
        native_response = Mock(status_code=404)
        fallback_response = Mock(status_code=200)
        fallback_response.json.return_value = {
            "data": [
                {"id": "zeta"},
                {"id": "alpha"},
                {"id": "Beta"},
            ]
        }

        with patch("tools.ollama_chat.requests.get", side_effect=[native_response, fallback_response]):
            models = ollama_chat.get_available_models("http://localhost:11434")

        self.assertEqual(models, ["alpha", "Beta", "zeta"])


if __name__ == "__main__":
    unittest.main()