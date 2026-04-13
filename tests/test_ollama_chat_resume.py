from __future__ import annotations

import sys
import unittest
from io import StringIO
from unittest.mock import patch

from tools import ollama_chat


class OllamaChatResumeHintTests(unittest.TestCase):
    def tearDown(self) -> None:
        ollama_chat.CURRENT_SESSION_ID = ""
        ollama_chat.LAUNCH_COMMAND_PREFIX = ""

    def test_resume_hint_uses_active_model_and_original_python_script_prefix(self) -> None:
        ollama_chat.CURRENT_SESSION_ID = "abc123"

        with patch.object(sys, "orig_argv", ["python", "tools/ollama_chat.py", "llama3.2", "-c"], create=True), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            ollama_chat._capture_launch_command_prefix()
            ollama_chat._print_resume_hint("openai/gpt-oss-120b")

        self.assertEqual(
            stdout.getvalue().strip(),
            "💡 To resume: python tools/ollama_chat.py openai/gpt-oss-120b -r abc123",
        )

    def test_resume_hint_falls_back_to_session_meta_model(self) -> None:
        ollama_chat.CURRENT_SESSION_ID = "abc123"

        with patch.object(sys, "orig_argv", ["python", "tools/ollama_chat.py", "llama3.2"], create=True), \
             patch("tools.ollama_chat._read_session_meta", return_value={"model": "qwen2.5-coder"}), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            ollama_chat._capture_launch_command_prefix()
            ollama_chat._print_resume_hint()

        self.assertEqual(
            stdout.getvalue().strip(),
            "💡 To resume: python tools/ollama_chat.py qwen2.5-coder -r abc123",
        )


if __name__ == "__main__":
    unittest.main()