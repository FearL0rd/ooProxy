from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tools import ollama_chat


def _tool_entry(name: str, *, command: str, description: str = "test tool") -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "command": command,
    }


class OllamaChatToolLoadingTests(unittest.TestCase):
    def tearDown(self) -> None:
        ollama_chat.TOOL_REGISTRY = ollama_chat._build_tool_registry()
        ollama_chat.TOOL_SCHEMAS = ollama_chat._build_tool_schemas()
        ollama_chat.EXTERNAL_TOOL_FILES = []
        ollama_chat.TOOL_LOAD_EVENTS = []
        ollama_chat.TOOL_LOAD_WARNINGS = []
        ollama_chat.TOOL_LOAD_SUMMARY_SHOWN = False

    def test_discovers_global_then_local_then_explicit_tool_files(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir, tempfile.TemporaryDirectory() as extra_dir:
            global_tools = Path(home_dir) / ".ooProxy" / "tools"
            local_tools = Path(cwd_dir) / ".ooProxy" / "tools"
            global_tools.mkdir(parents=True)
            local_tools.mkdir(parents=True)

            global_file = global_tools / "a.json"
            local_file = local_tools / "b.json"
            explicit_file = Path(extra_dir) / "c.json"

            global_file.write_text(json.dumps([_tool_entry("global_tool", command="printf global")]), encoding="utf-8")
            local_file.write_text(json.dumps([_tool_entry("local_tool", command="printf local")]), encoding="utf-8")
            explicit_file.write_text(json.dumps([_tool_entry("explicit_tool", command="printf explicit")]), encoding="utf-8")

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                discovered = ollama_chat.discover_tool_definition_files([str(explicit_file)])

            self.assertEqual(
                discovered,
                [str(global_file.resolve()), str(local_file.resolve()), str(explicit_file.resolve())],
            )

    def test_local_tool_overrides_global_and_builtin_tools(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            global_tools = Path(home_dir) / ".ooProxy" / "tools"
            local_tools = Path(cwd_dir) / ".ooProxy" / "tools"
            global_tools.mkdir(parents=True)
            local_tools.mkdir(parents=True)

            (global_tools / "global.json").write_text(
                json.dumps([
                    _tool_entry("list_directory", command="printf global-list", description="global override"),
                    _tool_entry("shared_tool", command="printf global-shared", description="global shared"),
                ]),
                encoding="utf-8",
            )
            (local_tools / "local.json").write_text(
                json.dumps([
                    _tool_entry("shared_tool", command="printf local-shared", description="local shared"),
                ]),
                encoding="utf-8",
            )

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                ollama_chat.configure_tool_registry([])

            self.assertEqual(
                ollama_chat.TOOL_REGISTRY["list_directory"]["description"],
                "global override",
            )
            self.assertEqual(
                ollama_chat.TOOL_REGISTRY["shared_tool"]["description"],
                "local shared",
            )
            self.assertEqual(
                ollama_chat.EXTERNAL_TOOL_FILES,
                [
                    str((global_tools / "global.json").resolve()),
                    str((local_tools / "local.json").resolve()),
                ],
            )

    def test_tool_load_summary_reports_overrides_once(self) -> None:
        ollama_chat.TOOL_LOAD_EVENTS = [
            {
                "name": "shared_tool",
                "source": "/tmp/local.json",
                "status": "override",
                "previous_source": "builtin",
            },
            {
                "name": "new_tool",
                "source": "/tmp/global.json",
                "status": "add",
                "previous_source": "",
            },
        ]
        ollama_chat.TOOL_LOAD_SUMMARY_SHOWN = False

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            ollama_chat._print_tool_load_summary()
            ollama_chat._print_tool_load_summary()

        output = stdout.getvalue()
        self.assertEqual(output.count("🧰 Added 2 tool definition(s):"), 1)
        self.assertIn("shared_tool [/tmp/local.json] overriding builtin", output)
        self.assertIn("new_tool [/tmp/global.json]", output)

    def test_tools_markdown_table_formats_inventory(self) -> None:
        ollama_chat.TOOL_REGISTRY = {
            "alpha": {
                "description": "Alpha tool",
                "source": "builtin",
                "read_only": True,
                "parameters": {"type": "object", "properties": {}, "required": []},
                "handler": lambda: "ok",
            },
            "beta": {
                "description": "Writes | updates files",
                "source": "/tmp/tools.json",
                "read_only": False,
                "parameters": {"type": "object", "properties": {}, "required": []},
                "handler": lambda: "ok",
            },
        }
        ollama_chat.TOOL_SCHEMAS = [
            {
                "type": "function",
                "function": {
                    "name": "alpha",
                    "description": "Alpha tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "beta",
                    "description": "Writes | updates files",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        table = ollama_chat._tools_markdown_table()

        self.assertIn("| Name | Source | Mode | Description |", table)
        self.assertIn("| alpha | builtin | read-only | Alpha tool |", table)
        self.assertIn("| beta | /tmp/tools.json | guarded | Writes \\| updates files |", table)

    def test_external_tool_receives_session_cwd_for_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            tool_file = Path(home_dir) / ".ooProxy" / "tools" / "tools.json"
            tool_file.parent.mkdir(parents=True)
            Path(cwd_dir, "sample.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            tool_file.write_text(
                json.dumps({
                    "tools": [
                        {
                            "name": "show_file_head",
                            "description": "Read a file relative to the chat cwd.",
                            "read_only": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "lines": {"type": "integer"},
                                },
                                "required": ["path"],
                            },
                            "argv": [
                                "python3",
                                "-c",
                                "import json,os,sys,pathlib; data=json.load(sys.stdin); base=pathlib.Path(os.environ['OLLAMA_TOOL_CWD']); path=pathlib.Path(data['path']); path = path if path.is_absolute() else (base / path); lines=int(data.get('lines',20)); print(''.join(path.read_text(encoding='utf-8').splitlines(True)[:lines]))",
                            ],
                            "cwd": ".",
                        }
                    ]
                }),
                encoding="utf-8",
            )

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                ollama_chat.configure_tool_registry([])
                result = ollama_chat.execute_tool_call("show_file_head", {"path": "sample.txt", "lines": 2}, "off")

            self.assertEqual(result, "alpha\nbeta")

    def test_single_tool_definition_object_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            tool_file = Path(home_dir) / ".ooProxy" / "tools" / "echo.json"
            tool_file.parent.mkdir(parents=True)
            tool_file.write_text(
                json.dumps({
                    "type": "function",
                    "function": {
                        "name": "echo_tool",
                        "description": "Echo arguments.",
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                        "required": ["value"],
                    },
                    "argv": [
                        "python3",
                        "-c",
                        "import json,sys; data=json.load(sys.stdin); print(data['value'])",
                    ],
                    "read_only": True,
                }),
                encoding="utf-8",
            )

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                ollama_chat.configure_tool_registry([])
                result = ollama_chat.execute_tool_call("echo_tool", {"value": "hello"}, "off")

            self.assertEqual(result, "hello")
            self.assertEqual(ollama_chat.TOOL_LOAD_WARNINGS, [])

    def test_external_tool_preserves_leading_spaces_in_multiline_output(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            tool_file = Path(home_dir) / ".ooProxy" / "tools" / "indent.json"
            tool_file.parent.mkdir(parents=True)
            tool_file.write_text(
                json.dumps({
                    "type": "function",
                    "function": {
                        "name": "indented_echo",
                        "description": "Emit indented multiline output.",
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                    "argv": [
                        "python3",
                        "-c",
                        "import sys; sys.stdout.write('  top\\n  next\\n')",
                    ],
                    "read_only": True,
                }),
                encoding="utf-8",
            )

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                ollama_chat.configure_tool_registry([])
                result = ollama_chat.execute_tool_call("indented_echo", {}, "off")

            self.assertEqual(result, "  top\n  next")

    def test_invalid_external_tool_file_is_skipped_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            valid_file = Path(home_dir) / ".ooProxy" / "tools" / "valid.json"
            invalid_file = Path(home_dir) / ".ooProxy" / "tools" / "invalid.json"
            valid_file.parent.mkdir(parents=True)
            valid_file.write_text(
                json.dumps([
                    _tool_entry("valid_tool", command="printf ok", description="valid tool"),
                ]),
                encoding="utf-8",
            )
            invalid_file.write_text(
                json.dumps({
                    "type": "function",
                    "function": {"name": "broken_tool"},
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }),
                encoding="utf-8",
            )

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                ollama_chat.configure_tool_registry([])

            self.assertIn("valid_tool", ollama_chat.TOOL_REGISTRY)
            self.assertTrue(ollama_chat.TOOL_LOAD_WARNINGS)
            self.assertIn(str(invalid_file.resolve()), ollama_chat.TOOL_LOAD_WARNINGS[0])

    def test_schema_only_tool_uses_sibling_python_companion(self) -> None:
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cwd_dir:
            tool_dir = Path(home_dir) / ".ooProxy" / "tools"
            tool_file = tool_dir / "tui_qr.json"
            companion_file = tool_dir / "show_qr_tui.py"
            tool_dir.mkdir(parents=True)
            tool_file.write_text(
                json.dumps({
                    "type": "function",
                    "function": {
                        "name": "show_qr_tui",
                        "description": "Render a QR code.",
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                        },
                        "required": ["text"],
                    },
                }),
                encoding="utf-8",
            )
            companion_file.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "import sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(f\"QR:{payload['text']}\")\n",
                encoding="utf-8",
            )

            with patch("tools.ollama_chat.Path.home", return_value=Path(home_dir)), \
                 patch("tools.ollama_chat.os.getcwd", return_value=cwd_dir):
                ollama_chat.configure_tool_registry([])
                result = ollama_chat.execute_tool_call("show_qr_tui", {"text": "hello"}, "off")

            self.assertEqual(result, "QR:hello")
            self.assertEqual(ollama_chat.TOOL_LOAD_WARNINGS, [])


if __name__ == "__main__":
    unittest.main()