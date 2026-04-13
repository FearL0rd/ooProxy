from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from tools import ollama_chat


class OllamaChatReplayTests(unittest.TestCase):
    def tearDown(self) -> None:
        ollama_chat.CONTINUE_SESSION = False
        ollama_chat.CONTEXT_FILE = ""
        ollama_chat.CURRENT_SESSION_ID = ""

    def test_help_text_mentions_redraw(self) -> None:
        with patch("tools.ollama_chat.render_markdown_to_terminal") as render_markdown, \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            ollama_chat._print_command_help()

        render_markdown.assert_called_once()
        output = render_markdown.call_args.args[0]
        self.assertIn("| Command | Action |", output)
        self.assertIn("/export <file>", output)
        self.assertIn("Export the whole session as raw Markdown", output)
        self.assertIn("/redraw", output)
        self.assertIn("Clear the screen and replay the saved conversation", output)
        self.assertEqual(stdout.getvalue(), "\n")

    def test_session_export_markdown_preserves_raw_markdown_content(self) -> None:
        messages = [
            {"role": "user", "content": "Summarize this repo."},
            {
                "role": "assistant",
                "content": "# Summary\n\n- item one\n- item two",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "arguments": json.dumps({"path": "."}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "list_directory",
                "content": '{"entries": ["README.md"]}',
            },
        ]

        ollama_chat.CURRENT_SESSION_ID = "session-123"

        with patch("tools.ollama_chat._read_session_meta", return_value={"model": "llama3.2", "host": "localhost", "port": "11434"}):
            exported = ollama_chat._session_export_markdown(messages)

        self.assertIn("# ooProxy Session Export", exported)
        self.assertIn("- Session ID: session-123", exported)
        self.assertIn("## 2. Assistant", exported)
        self.assertIn("# Summary\n\n- item one\n- item two", exported)
        self.assertIn("### Tool Calls", exported)
        self.assertIn("- `list_directory({\"path\": \".\"})`", exported)
        self.assertIn("## 3. Tool `list_directory`", exported)
        self.assertIn('{"entries": ["README.md"]}', exported)

    def test_export_command_writes_markdown_file(self) -> None:
        class FakePromptSession:
            def __init__(self, responses: list[str]) -> None:
                self._responses = iter(responses)

            def prompt(self, _prompt: str) -> str:
                return next(self._responses)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "# Hi"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = f"{tmpdir}/session.md"
            ollama_chat.HISTORY_FILE = f"{tmpdir}/history"
            ollama_chat.CURRENT_SESSION_ID = "session-abc"
            ollama_chat.CONTINUE_SESSION = False

            with patch("tools.ollama_chat.load_context", return_value=list(messages)), \
                 patch("tools.ollama_chat.save_context", return_value=len(messages)), \
                 patch("tools.ollama_chat.PromptSession", return_value=FakePromptSession([f"/export {export_path}", "/exit"])), \
                 patch("tools.ollama_chat._read_session_meta", return_value={"model": "llama3.2", "host": "localhost", "port": "11434"}), \
                 patch("sys.stdout", new_callable=StringIO) as stdout:
                ollama_chat.chat_with_ollama(
                    "llama3.2",
                    "http://localhost:11434",
                    use_openai=False,
                    enable_tools=True,
                    render_mode="markdown",
                    guardrails_mode="confirm-destructive",
                )

            output = stdout.getvalue()
            self.assertIn(f"💾 Session exported to: {export_path}", output)
            self.assertTrue(Path(export_path).exists())
            exported = Path(export_path).read_text(encoding="utf-8")
            self.assertIn("# ooProxy Session Export", exported)
            self.assertIn("## 1. User", exported)
            self.assertIn("hello", exported)
            self.assertIn("## 2. Assistant", exported)
            self.assertIn("# Hi", exported)

    def test_markdown_renderer_trims_rich_padding_lines(self) -> None:
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=False, width=100)

        ollama_chat.render_markdown_to_terminal(
            "\n".join([
                "| Name | Source | Mode | Description |",
                "| --- | --- | --- | --- |",
                "| alpha | builtin | read-only | Alpha tool |",
            ]),
            console=console,
        )

        output_lines = buffer.getvalue().splitlines()
        self.assertTrue(output_lines)
        self.assertEqual(output_lines[0].strip(), "Name   Source   Mode       Description")
        self.assertEqual(output_lines[-1].strip(), "alpha  builtin  read-only  Alpha tool")
        self.assertTrue(all(line.strip() for line in output_lines))

    def test_turn_separator_uses_horizontal_rule_renderer(self) -> None:
        with patch("tools.ollama_chat.render_horizontal_rule") as render_rule:
            ollama_chat._print_turn_separator()

        render_rule.assert_called_once_with()

    def test_horizontal_rule_uses_configured_character(self) -> None:
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=False, width=20)

        ollama_chat.render_horizontal_rule(console=console)

        output = buffer.getvalue().strip()
        self.assertTrue(output)
        self.assertEqual(set(output), {ollama_chat.TURN_SEPARATOR_CHAR})

    def test_load_context_replays_visible_conversation_and_hides_tool_results(self) -> None:
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "I will inspect the folder.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "arguments": json.dumps({"path": "."}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "list_directory",
                "tool_call_id": "call_1",
                "content": '{"entries": ["README.md"]}',
            },
            {"role": "assistant", "content": "I found README.md."},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            context_file = f"{tmpdir}/context.json"
            with open(context_file, "w", encoding="utf-8") as handle:
                json.dump(messages, handle)

            ollama_chat.CONTEXT_FILE = context_file
            ollama_chat.CONTINUE_SESSION = False

            with patch("sys.stdout", new_callable=StringIO) as stdout:
                loaded = ollama_chat.load_context()

        output = stdout.getvalue()
        self.assertEqual(loaded, messages)
        self.assertIn("📂 Loaded 3 previous messages.", output)
        self.assertIn(">>> list files", output)
        self.assertIn("I will inspect the folder.", output)
        self.assertIn("[tool] list_directory({\"path\": \".\"})", output)
        self.assertIn("I found README.md.", output)
        self.assertIn(ollama_chat.TURN_SEPARATOR_CHAR * 10, output)
        self.assertNotIn("============================================================", output)
        self.assertNotIn('{"entries": ["README.md"]}', output)
        self.assertTrue(ollama_chat.CONTINUE_SESSION)


if __name__ == "__main__":
    unittest.main()