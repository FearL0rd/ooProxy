from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ollama_list_models.py"
_SPEC = importlib.util.spec_from_file_location("ollama_list_models", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
ollama_list_models = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ollama_list_models)


class TestListOllamaModels(unittest.TestCase):
    def test_sorts_ollama_models_alphabetically(self) -> None:
        response = Mock()
        response.json.return_value = {
            "models": [
                {"name": "zeta"},
                {"name": "alpha"},
                {"name": "Beta"},
            ]
        }
        response.raise_for_status.return_value = None

        with patch.object(ollama_list_models.requests, "get", return_value=response):
            models = ollama_list_models.list_ollama_models("http://localhost:11434", use_openai=False)

        self.assertEqual([model["name"] for model in models], ["alpha", "Beta", "zeta"])

    def test_sorts_openai_models_alphabetically(self) -> None:
        response = Mock()
        response.json.return_value = {
            "data": [
                {"id": "zeta"},
                {"id": "alpha"},
                {"id": "Beta"},
            ]
        }
        response.raise_for_status.return_value = None

        with patch.object(ollama_list_models.requests, "get", return_value=response):
            models = ollama_list_models.list_ollama_models("http://localhost:11434", use_openai=True)

        self.assertEqual([model["name"] for model in models], ["alpha", "Beta", "zeta"])


if __name__ == "__main__":
    unittest.main()