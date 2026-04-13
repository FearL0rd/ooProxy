import argparse
import atexit
import hashlib
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# These three are rewritten by _init_session_paths() before anything else runs.
CONTEXT_FILE = ""
HISTORY_FILE = ""
LOCK_FILE = ""
CURRENT_SESSION_ID = ""
LAUNCH_COMMAND_PREFIX = ""

ATTACHMENT_BUFFER: List[str] = []
DEFAULT_TOOL_TIMEOUT = 120
MAX_TOOL_OUTPUT_CHARS = 16000
# Applied to OpenAI-compat requests (-o mode) where the proxy has no translation
# layer to inject a sensible default.  Native /api/chat requests are handled by
# the proxy's chat_to_openai() which already injects this value.
DEFAULT_MAX_TOKENS = 32768
EXTERNAL_TOOL_FILES: List[str] = []
DEFAULT_RENDER_MODE = "markdown"
DEFAULT_GUARDRAILS_MODE = "confirm-destructive"

# Optional rich-based Markdown renderer for prettier terminal output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    _RICH_CONSOLE = Console()
except Exception:
    _RICH_CONSOLE = None

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_SESSIONS_DIR = Path.home() / ".ooProxy" / "sessions"


def _cwd_hash() -> str:
    return hashlib.sha1(os.path.abspath(os.getcwd()).encode()).hexdigest()[:8]


def _new_session_id() -> str:
    return f"{_cwd_hash()}-{secrets.token_hex(3)}"


def _session_dir(session_id: str) -> Path:
    return _SESSIONS_DIR / session_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_session_meta(session_id: str) -> Dict[str, Any]:
    try:
        return json.loads((_session_dir(session_id) / "meta.json").read_text())
    except Exception:
        return {}


def _write_session_meta(session_id: str, **updates: Any) -> None:
    path = _session_dir(session_id) / "meta.json"
    meta = _read_session_meta(session_id)
    meta.update(updates)
    try:
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _session_locked(session_id: str) -> bool:
    """Return True if the session is held by a live ollama_chat process."""
    lock = _session_dir(session_id) / "lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip())
        return _pid_alive(pid) and _is_ollama_chat_process(pid)
    except Exception:
        return False


def _sessions_for_cwd(cwd: str) -> List[Dict[str, Any]]:
    """Return sessions whose meta.cwd matches *cwd*, sorted newest first."""
    if not _SESSIONS_DIR.exists():
        return []
    prefix = hashlib.sha1(os.path.abspath(cwd).encode()).hexdigest()[:8]
    results: List[Dict[str, Any]] = []
    for d in _SESSIONS_DIR.iterdir():
        sid = d.name
        if not sid.startswith(prefix):
            continue
        meta = _read_session_meta(sid)
        if os.path.abspath(meta.get("cwd", "")) != os.path.abspath(cwd):
            continue
        results.append({"id": sid, "meta": meta, "locked": _session_locked(sid)})
    results.sort(key=lambda s: s["meta"].get("last_used", ""), reverse=True)
    return results


def _create_session(session_id: str, model: str) -> None:
    _session_dir(session_id).mkdir(parents=True, exist_ok=True)
    _write_session_meta(
        session_id,
        cwd=os.path.abspath(os.getcwd()),
        model=model,
        created_at=_now_iso(),
        last_used=_now_iso(),
        message_count=0,
    )


def _prompt_session_selection(sessions: List[Dict[str, Any]]) -> str:
    """Print a pick-list and return the chosen session ID (or a new one)."""
    print("\nMultiple saved sessions for this folder:")
    for i, s in enumerate(sessions, 1):
        meta = s["meta"]
        ts = meta.get("last_used", "?")[:16].replace("T", " ")
        count = meta.get("message_count", "?")
        model = meta.get("model", "?")
        print(f"  [{i}] {s['id']}  {model:<30}  last used: {ts}  ({count} messages)")
    print()
    while True:
        try:
            answer = input(f"Resume [1-{len(sessions)}] or N for new session: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "n":
            return ""          # caller creates new session
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]["id"]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(sessions)}, or N.")


def _migrate_legacy_context(session_id: str) -> bool:
    """Copy .ollama_chat_context from CWD into the new session if it exists."""
    legacy = Path(".ollama_chat_context")
    if not legacy.exists():
        return False
    dest = _session_dir(session_id) / "context.json"
    try:
        dest.write_text(legacy.read_text())
        legacy_hist = Path(".ollama_chat_history")
        if legacy_hist.exists():
            hist_dest = _session_dir(session_id) / "history"
            hist_dest.write_text(legacy_hist.read_text())
        print(f"📦 Migrated existing context → session {session_id}")
        return True
    except Exception as exc:
        print(f"⚠️ Migration failed: {exc}")
        return False


def resolve_session(resume_id: Optional[str], new_session: bool, model: str) -> str:
    """Return the session ID to use; creates a new session directory if needed."""
    cwd = os.path.abspath(os.getcwd())

    if resume_id:
        if not _session_dir(resume_id).exists():
            print(f"❌ Session not found: {resume_id}")
            sys.exit(1)
        if _session_locked(resume_id):
            print(f"❌ Session {resume_id} is currently in use by another process.")
            sys.exit(1)
        return resume_id

    if new_session:
        sid = _new_session_id()
        _create_session(sid, model)
        return sid

    sessions = _sessions_for_cwd(cwd)
    unlocked = [s for s in sessions if not s["locked"]]

    if not unlocked:
        sid = _new_session_id()
        _create_session(sid, model)
        if not sessions:                 # very first session for this CWD
            _migrate_legacy_context(sid)
        return sid

    if len(unlocked) == 1:
        return unlocked[0]["id"]

    chosen = _prompt_session_selection(unlocked)
    if not chosen:
        sid = _new_session_id()
        _create_session(sid, model)
        return sid
    return chosen


def _init_session_paths(session_id: str) -> None:
    """Point the module-level path globals at the chosen session directory."""
    global CONTEXT_FILE, HISTORY_FILE, LOCK_FILE, CURRENT_SESSION_ID
    CURRENT_SESSION_ID = session_id
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    CONTEXT_FILE = str(d / "context.json")
    HISTORY_FILE = str(d / "history")
    LOCK_FILE = str(d / "lock")


# When True, we're resuming an existing conversation and should
# avoid printing the startup banner.
CONTINUE_SESSION = False


def _looks_like_chat_script(path: str) -> bool:
    return os.path.basename(path) == "ollama_chat.py"


def _capture_launch_command_prefix() -> None:
    global LAUNCH_COMMAND_PREFIX

    orig_argv = getattr(sys, "orig_argv", []) or []
    if len(orig_argv) >= 2 and _looks_like_chat_script(orig_argv[1]):
        LAUNCH_COMMAND_PREFIX = " ".join(shlex.quote(part) for part in orig_argv[:2])
        return

    argv0 = sys.argv[0] if sys.argv else ""
    if _looks_like_chat_script(argv0):
        LAUNCH_COMMAND_PREFIX = f"python {shlex.quote(argv0)}"
        return

    LAUNCH_COMMAND_PREFIX = f"python {shlex.quote(os.path.basename(__file__))}"


def _resume_model(active_model: str = "") -> str:
    if active_model:
        return active_model
    if CURRENT_SESSION_ID:
        return str(_read_session_meta(CURRENT_SESSION_ID).get("model", "")).strip()
    return ""


def render_markdown_to_terminal(text: str, console: Any = None) -> None:
    """Render Markdown `text` to the terminal using Rich if available,
    otherwise fall back to plain printing.
    """
    if console is None:
        console = _RICH_CONSOLE
    if console:
        try:
            console.print(Markdown(text))
            return
        except Exception:
            pass
    # Fallback
    print(text)


def _truncate_text(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n...[truncated {omitted} characters]"


def _json_result(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _tool_get_cwd() -> str:
    return _json_result({"cwd": os.getcwd()})


def _tool_list_directory(path: str = ".") -> str:
    target = os.path.abspath(path)
    if not os.path.exists(target):
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not os.path.isdir(target):
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = []
    for name in sorted(os.listdir(target)):
        full_path = os.path.join(target, name)
        entries.append({
            "name": name,
            "type": "dir" if os.path.isdir(full_path) else "file",
        })

    return _json_result({"path": target, "entries": entries})


def _tool_read_file(path: str) -> str:
    target = os.path.abspath(path)
    with open(target, "r", encoding="utf-8") as handle:
        content = handle.read()
    return _json_result({
        "path": target,
        "content": _truncate_text(content),
        "truncated": len(content) > MAX_TOOL_OUTPUT_CHARS,
    })


def _tool_write_file(path: str, content: str) -> str:
    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return _json_result({"path": target, "written_chars": len(content)})


def _tool_run_shell(command: str, cwd: Optional[str] = None, timeout: int = DEFAULT_TOOL_TIMEOUT) -> str:
    working_directory = os.path.abspath(cwd or os.getcwd())
    result = subprocess.run(
        command,
        shell=True,
        cwd=working_directory,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return _json_result({
        "cwd": working_directory,
        "command": command,
        "exit_code": result.returncode,
        "stdout": _truncate_text(result.stdout),
        "stderr": _truncate_text(result.stderr),
    })


ToolHandler = Callable[..., str]


def _build_tool_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "get_current_directory": {
            "description": "Return the current working directory for this chat session.",
            "read_only": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
            "handler": _tool_get_cwd,
        },
        "list_directory": {
            "description": "List files and folders in a directory.",
            "read_only": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to inspect. Defaults to the current working directory.",
                    },
                },
                "required": [],
            },
            "handler": _tool_list_directory,
        },
        "read_file": {
            "description": "Read a UTF-8 text file from disk.",
            "read_only": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                },
                "required": ["path"],
            },
            "handler": _tool_read_file,
        },
        "write_file": {
            "description": "Write UTF-8 text to a file, replacing existing contents.",
            "read_only": False,
            "confirm_on_overwrite": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
            "handler": _tool_write_file,
        },
        "run_shell": {
            "description": "Run a shell command in the current working directory and return stdout, stderr, and exit code.",
            "read_only": False,
            "shell_command": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory. Defaults to the current working directory.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Defaults to 120.",
                    },
                },
                "required": ["command"],
            },
            "handler": _tool_run_shell,
        },
    }


def _tool_display_source(spec: Dict[str, Any]) -> str:
    return spec.get("source", "builtin")


def _tool_command_handler_factory(name: str, command_spec: Dict[str, Any]) -> ToolHandler:
    command = command_spec.get("command")
    argv = command_spec.get("argv")
    cwd = command_spec.get("cwd")
    timeout = int(command_spec.get("timeout", DEFAULT_TOOL_TIMEOUT))
    if bool(command) == bool(argv):
        raise ValueError(f"External tool '{name}' must define exactly one of 'command' or 'argv'.")

    def _run_external_tool(**arguments: Any) -> str:
        input_payload = json.dumps(arguments, ensure_ascii=False)
        env = os.environ.copy()
        env["OLLAMA_TOOL_NAME"] = name
        env["OLLAMA_TOOL_ARGS"] = input_payload
        working_directory = os.path.abspath(cwd) if cwd else os.getcwd()

        result = subprocess.run(
            command if command else [str(part) for part in argv],
            shell=bool(command),
            cwd=working_directory,
            input=input_payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            raise RuntimeError(_json_result({
                "command": command if command else argv,
                "cwd": working_directory,
                "exit_code": result.returncode,
                "stdout": _truncate_text(stdout),
                "stderr": _truncate_text(stderr),
            }))

        if stdout:
            return _truncate_text(stdout)
        if stderr:
            return _truncate_text(stderr)
        return _json_result({
            "command": command if command else argv,
            "cwd": working_directory,
            "exit_code": result.returncode,
        })

    return _run_external_tool


def _normalize_external_tool_definition(raw_tool: Dict[str, Any], tool_file: str) -> Dict[str, Any]:
    function = raw_tool.get("function") if raw_tool.get("type") == "function" else raw_tool
    if not isinstance(function, dict):
        raise ValueError(f"Invalid tool definition in {tool_file}: expected an object.")

    name = function.get("name")
    if not name:
        raise ValueError(f"Invalid tool definition in {tool_file}: missing tool name.")

    parameters = function.get("parameters") or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    if not isinstance(parameters, dict):
        raise ValueError(f"Invalid tool definition for '{name}' in {tool_file}: parameters must be an object.")

    base_dir = os.path.dirname(tool_file)
    cwd = function.get("cwd")
    if isinstance(cwd, str) and cwd and not os.path.isabs(cwd):
        cwd = os.path.abspath(os.path.join(base_dir, cwd))

    spec = {
        "description": function.get("description", "External command-backed tool."),
        "parameters": parameters,
        "source": tool_file,
        "read_only": bool(function.get("read_only", False)),
        "destructive": bool(function.get("destructive", False)),
        "shell_command": bool(function.get("command")),
        "handler": _tool_command_handler_factory(name, {
            "command": function.get("command"),
            "argv": function.get("argv"),
            "cwd": cwd,
            "timeout": function.get("timeout", DEFAULT_TOOL_TIMEOUT),
        }),
    }
    return {"name": name, "spec": spec}


def _load_external_tools(tool_files: List[str]) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for tool_file in tool_files:
        target = os.path.abspath(tool_file)
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            raw_tools = payload.get("tools")
            if not isinstance(raw_tools, list):
                raise ValueError(f"Tool file {target} must contain a 'tools' array.")
        elif isinstance(payload, list):
            raw_tools = payload
        else:
            raise ValueError(f"Tool file {target} must be a JSON object or array.")

        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                raise ValueError(f"Tool file {target} contains a non-object tool entry.")
            normalized = _normalize_external_tool_definition(raw_tool, target)
            name = normalized["name"]
            if name in loaded:
                raise ValueError(f"Duplicate external tool name: {name}")
            loaded[name] = normalized["spec"]

    return loaded


TOOL_REGISTRY = _build_tool_registry()


def _build_tool_schemas() -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for name, spec in TOOL_REGISTRY.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        })
    return schemas


TOOL_SCHEMAS = _build_tool_schemas()


def configure_tool_registry(tool_files: Optional[List[str]] = None) -> None:
    global TOOL_REGISTRY, TOOL_SCHEMAS, EXTERNAL_TOOL_FILES

    registry = _build_tool_registry()
    resolved_files = [os.path.abspath(path) for path in (tool_files or [])]
    external_tools = _load_external_tools(resolved_files) if resolved_files else {}
    for name, spec in external_tools.items():
        if name in registry:
            raise ValueError(f"Tool name collision: {name}")
        registry[name] = spec

    TOOL_REGISTRY = registry
    TOOL_SCHEMAS = _build_tool_schemas()
    EXTERNAL_TOOL_FILES = resolved_files


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed
    raise ValueError(f"Unsupported tool arguments type: {type(raw).__name__}")


def _tool_error_message(error: Exception) -> str:
    return _json_result({
        "ok": False,
        "error": type(error).__name__,
        "message": str(error),
    })


def _tool_denied_message(reason: str) -> str:
    return _json_result({
        "ok": False,
        "error": "GuardrailDenied",
        "message": reason,
    })


def _command_looks_destructive(command: str) -> bool:
    lower = f" {command.lower()} "
    destructive_tokens = [
        " rm ",
        " rmdir ",
        " unlink ",
        " mv ",
        " dd ",
        " truncate ",
        " shred ",
        " chmod ",
        " chown ",
        " sed -i",
        " perl -pi",
        " git clean ",
        " git reset ",
        " git checkout ",
        " git restore ",
        " find ",
    ]
    if any(token in lower for token in destructive_tokens):
        return True
    if "find " in lower and " -delete" in lower:
        return True
    if ">" in command or ">>" in command:
        return True
    if " tee " in lower:
        return True
    return False


def _tool_guardrail_reason(name: str, arguments: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    if spec.get("confirm_on_overwrite"):
        path = arguments.get("path")
        if isinstance(path, str) and path:
            target = os.path.abspath(path)
            if os.path.exists(target):
                return f"Tool '{name}' would overwrite existing path: {target}"

    if spec.get("shell_command"):
        command = arguments.get("command")
        if isinstance(command, str) and _command_looks_destructive(command):
            return f"Tool '{name}' wants to run a potentially destructive shell command: {command}"

    if spec.get("destructive"):
        return f"Tool '{name}' is marked destructive."

    if spec.get("source") and not spec.get("read_only", False):
        return f"External tool '{name}' is not marked read-only."

    return None


def _confirm_tool_execution(name: str, arguments: Dict[str, Any], reason: str) -> bool:
    print(f"⚠️ Guardrail: {reason}")
    print(f"⚠️ Requested call: {_tool_summary({'name': name, 'arguments': arguments})}")
    try:
        answer = input("Allow? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def execute_tool_call(name: str, arguments: Dict[str, Any], guardrails_mode: str) -> str:
    spec = TOOL_REGISTRY.get(name)
    if not spec:
        raise KeyError(f"Unknown tool: {name}")

    reason = _tool_guardrail_reason(name, arguments, spec)
    if guardrails_mode == "read-only":
        if not spec.get("read_only", False) or reason:
            return _tool_denied_message(reason or f"Tool '{name}' is not allowed in read-only mode.")
    elif guardrails_mode == "confirm-destructive" and reason:
        if not _confirm_tool_execution(name, arguments, reason):
            return _tool_denied_message(f"User denied execution of tool '{name}'.")

    handler: ToolHandler = spec["handler"]
    return handler(**arguments)


def _tool_support_error(message: str) -> bool:
    lower = (message or "").lower()
    return any(token in lower for token in ("tool", "tool_choice", "tool_calls", "function_call"))


def _message_header(role: str) -> str:
    return {
        "user": ">>>",
        "assistant": "<<<",
        "tool": "[tool]",
    }.get(role, f"[{role}]")


def _message_display_text(message: Dict[str, Any]) -> str:
    role = message.get("role", "assistant")
    content = message.get("content") or ""
    if role == "tool":
        name = message.get("tool_name") or message.get("name") or message.get("tool_call_id") or "tool"
        return f"{name}\n{content}" if content else name

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        lines = []
        if content:
            lines.append(content)
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name", "unknown_tool")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                argument_text = arguments
            else:
                argument_text = json.dumps(arguments, ensure_ascii=False)
            lines.append(f"tool_call: {name}({argument_text})")
        return "\n".join(lines)

    return content


def _print_message(message: Dict[str, Any]) -> None:
    print(f"{_message_header(message.get('role', 'assistant'))}:")
    text = _message_display_text(message)
    if text:
        try:
            render_markdown_to_terminal(text)
        except Exception:
            print(text)
    print("\n")


def _normalize_openai_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        call_id = tool_call.get("id") or f"call_{index}"
        try:
            arguments = _parse_tool_arguments(function.get("arguments"))
        except Exception as exc:
            arguments = {"_raw": function.get("arguments"), "_parse_error": str(exc)}
        normalized.append({
            "id": call_id,
            "name": function.get("name", "unknown_tool"),
            "arguments": arguments,
        })
    return normalized


def _normalize_ollama_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        try:
            arguments = _parse_tool_arguments(function.get("arguments"))
        except Exception as exc:
            arguments = {"_raw": function.get("arguments"), "_parse_error": str(exc)}
        normalized.append({
            "id": tool_call.get("id") or f"call_{function.get('index', index)}",
            "name": function.get("name", "unknown_tool"),
            "arguments": arguments,
        })
    return normalized


def _assistant_message_from_response(data: Dict[str, Any], use_openai: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if use_openai:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        normalized_calls = _normalize_openai_tool_calls(message.get("tool_calls") or [])
        return _assistant_message_from_parts(content, normalized_calls, use_openai), normalized_calls

    message = data.get("message") or {}
    content = message.get("content") or ""
    normalized_calls = _normalize_ollama_tool_calls(message.get("tool_calls") or [])
    return _assistant_message_from_parts(content, normalized_calls, use_openai), normalized_calls


def _tool_result_message(name: str, result: str, call_id: str, use_openai: bool) -> Dict[str, Any]:
    if use_openai:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }
    return {
        "role": "tool",
        "tool_name": name,
        "content": result,
    }


def _tool_summary(call: Dict[str, Any]) -> str:
    return f"{call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})"


def _open_chat_stream(
    url: str,
    payload: Dict[str, Any],
    allow_tool_fallback: bool,
) -> Tuple[requests.Response, bool]:
    using_tools = "tools" in payload
    try:
        response = requests.post(url, json=payload, stream=True, timeout=180)
        response.raise_for_status()
        return response, using_tools
    except requests.exceptions.HTTPError as exc:
        response = exc.response
        if using_tools and allow_tool_fallback and response is not None and _tool_support_error(response.text):
            fallback_payload = {k: v for k, v in payload.items() if k not in ("tools", "tool_choice")}
            print("⚠️ Endpoint rejected tools for this request. Retrying without tool definitions.")
            retry = requests.post(url, json=fallback_payload, stream=True, timeout=180)
            retry.raise_for_status()
            return retry, False
        raise


def _update_openai_tool_buffers(buffers: Dict[int, Dict[str, Any]], tool_calls: List[Dict[str, Any]]) -> None:
    for tool_call in tool_calls:
        index = tool_call.get("index", 0)
        buffer = buffers.setdefault(index, {"id": tool_call.get("id") or f"call_{index}", "name": "", "arguments": ""})
        if tool_call.get("id"):
            buffer["id"] = tool_call["id"]
        function = tool_call.get("function") or {}
        if function.get("name"):
            buffer["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            buffer["arguments"] = f"{buffer['arguments']}{arguments}"
        elif isinstance(arguments, dict):
            buffer["arguments"] = json.dumps(arguments, ensure_ascii=False)


def _update_ollama_tool_buffers(buffers: Dict[int, Dict[str, Any]], tool_calls: List[Dict[str, Any]]) -> None:
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") or {}
        call_index = function.get("index", index)
        buffer = buffers.setdefault(call_index, {"id": tool_call.get("id") or f"call_{call_index}", "name": "", "arguments": {}})
        if tool_call.get("id"):
            buffer["id"] = tool_call["id"]
        if function.get("name"):
            buffer["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            buffer["arguments"] = arguments
        elif isinstance(arguments, str):
            try:
                buffer["arguments"] = _parse_tool_arguments(arguments)
            except Exception:
                buffer["arguments"] = {"_raw": arguments}


def _finalize_stream_tool_calls(buffers: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []
    for index in sorted(buffers):
        buffer = buffers[index]
        arguments = buffer.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_arguments = _parse_tool_arguments(arguments)
            except Exception as exc:
                parsed_arguments = {"_raw": arguments, "_parse_error": str(exc)}
        else:
            parsed_arguments = arguments
        finalized.append({
            "id": buffer.get("id") or f"call_{index}",
            "name": buffer.get("name") or "unknown_tool",
            "arguments": parsed_arguments or {},
        })
    return finalized


def _assistant_message_from_parts(content: str, normalized_calls: List[Dict[str, Any]], use_openai: bool) -> Dict[str, Any]:
    assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
    if not normalized_calls:
        return assistant_message

    if use_openai:
        assistant_message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for call in normalized_calls
        ]
        return assistant_message

    assistant_message["tool_calls"] = [
        {
            "type": "function",
            "function": {
                "index": index,
                "name": call["name"],
                "arguments": call["arguments"],
            },
        }
        for index, call in enumerate(normalized_calls)
    ]
    return assistant_message


def _stream_chat_response(
    url: str,
    payload: Dict[str, Any],
    use_openai: bool,
    allow_tool_fallback: bool,
    render_mode: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    response, tools_active_for_turn = _open_chat_stream(url, payload, allow_tool_fallback)
    content_parts: List[str] = []
    openai_tool_buffers: Dict[int, Dict[str, Any]] = {}
    ollama_tool_buffers: Dict[int, Dict[str, Any]] = {}
    printed_content = False

    try:
        for line in response.iter_lines():
            if not line:
                continue

            raw = line.decode("utf-8")
            if use_openai and raw.startswith("data: "):
                payload_text = raw[6:]
                if payload_text == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue

                if chunk.get("message"):
                    message = chunk.get("message") or {}
                    content = message.get("content") or ""
                    if content:
                        if render_mode == "stream":
                            print(content, end="", flush=True)
                            printed_content = True
                        content_parts.append(content)
                    _update_ollama_tool_buffers(ollama_tool_buffers, message.get("tool_calls") or [])
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if render_mode == "stream":
                        print(content, end="", flush=True)
                        printed_content = True
                    content_parts.append(content)
                _update_openai_tool_buffers(openai_tool_buffers, delta.get("tool_calls") or [])
                continue

            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            message = chunk.get("message") or {}
            content = message.get("content") or ""
            if content:
                if render_mode == "stream":
                    print(content, end="", flush=True)
                    printed_content = True
                content_parts.append(content)
            _update_ollama_tool_buffers(ollama_tool_buffers, message.get("tool_calls") or [])
    finally:
        response.close()

    if printed_content:
        print()

    normalized_calls = _finalize_stream_tool_calls(openai_tool_buffers if openai_tool_buffers else ollama_tool_buffers)
    content = "".join(content_parts)
    assistant_message = _assistant_message_from_parts(content, normalized_calls, use_openai)
    return assistant_message, normalized_calls, tools_active_for_turn


def _pid_alive(pid: int) -> bool:
    try:
        # signal 0 just tests for existence of process
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_ollama_chat_process(pid: int) -> bool:
    """Heuristic: return True if PID's command or cwd looks like this chat tool.

    Checks /proc/<pid>/cmdline for this script name or 'ollama_chat', and
    /proc/<pid>/cwd for matching working directory. Returns False if the
    checks fail or indicate a different program.
    """
    try:
        # Check cmdline
        cmdline_path = f"/proc/{pid}/cmdline"
        if os.path.exists(cmdline_path):
            with open(cmdline_path, "rb") as f:
                raw = f.read()
            # cmdline is null-separated
            parts = [p.decode(errors="ignore") for p in raw.split(b"\0") if p]
            script_name = os.path.basename(__file__)
            for p in parts:
                if script_name in p or "ollama_chat" in p.lower():
                    return True

        # Check cwd matches current working directory
        cwd_path = f"/proc/{pid}/cwd"
        if os.path.islink(cwd_path):
            try:
                other_cwd = os.readlink(cwd_path)
                if os.path.abspath(other_cwd) == os.path.abspath(os.getcwd()):
                    return True
            except Exception:
                pass
    except Exception:
        # If we cannot inspect the process, be conservative and assume it's not ours
        return False

    return False


def acquire_pidfile(lockfile: Optional[str] = None) -> None:
    """Create an exclusive pidfile. Exit if a live PID holds the lock."""
    if lockfile is None:
        lockfile = LOCK_FILE
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        print(f"🔐 Acquired lock: {lockfile} (PID {os.getpid()})")
        return
    except FileExistsError:
        # lock exists — check whether it's stale
        try:
            with open(lockfile, "r", encoding="utf-8") as f:
                content = f.read().strip()
                existing = int(content) if content else None
        except Exception:
            existing = None

        if existing and _pid_alive(existing):
            # If the running PID appears to be an ollama_chat process, respect it.
            if _is_ollama_chat_process(existing):
                print(f"⛔ Another session (PID {existing}) is using this folder. Exiting.")
                sys.exit(1)
            # Otherwise treat as stale and attempt to reclaim
            print(f"⚠️ Lock held by PID {existing} which is not an ollama_chat instance; reclaiming lock.")

        # stale lock: remove and retry once
        try:
            os.remove(lockfile)
        except Exception:
            print(f"⚠️ Could not remove stale lockfile: {lockfile}")
            sys.exit(1)

        try:
            fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            print(f"🔐 Acquired lock after removing stale lock: {lockfile} (PID {os.getpid()})")
            return
        except Exception as e:
            print(f"⛔ Failed to create lockfile: {e}")
            sys.exit(1)


def release_pidfile(lockfile: Optional[str] = None) -> None:
    if lockfile is None:
        lockfile = LOCK_FILE
    try:
        if os.path.exists(lockfile):
            try:
                with open(lockfile, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    existing = int(content) if content else None
            except Exception:
                existing = None

            # Only remove if this process owns the lock (best effort)
            if existing is None or existing == os.getpid():
                try:
                    os.remove(lockfile)
                    print(f"🗑️ Released lock: {lockfile}")
                except Exception:
                    pass
    except Exception:
        pass

def load_context() -> List[Dict]:
    if not os.path.exists(CONTEXT_FILE):
        return []
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if messages:
            print(f"📂 Loaded {len(messages)//2} previous messages.\n")
            print("📜 Previous conversation:")
            print("=" * 60)
            for msg in messages:
                _print_message(msg)
            print("=" * 60)
            # Mark that we are continuing a session so callers can
            # suppress redundant startup output (banner, help, etc.)
            global CONTINUE_SESSION
            CONTINUE_SESSION = True
        return messages
    except Exception as e:
        print(f"⚠️ Could not load context: {e}")
        return []

def save_context(messages: List[Dict]) -> int:
    """Save messages to context file. Returns number of messages saved, or -1 if removed."""
    try:
        # If the message list is empty, delete the context file instead of saving an empty one.
        if not messages:
            if os.path.exists(CONTEXT_FILE):
                os.remove(CONTEXT_FILE)
            if CURRENT_SESSION_ID:
                _write_session_meta(CURRENT_SESSION_ID, last_used=_now_iso(), message_count=0)
            return -1 # Indicates file was removed
        with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        if CURRENT_SESSION_ID:
            _write_session_meta(CURRENT_SESSION_ID, last_used=_now_iso(), message_count=len(messages))
        return len(messages)
    except Exception as e:
        print(f"⚠️ Could not save context: {e}")
        return 0

def read_file_content(filepath: str) -> str:
    """Read file and return content with filename header."""
    try:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(filepath)
        header = f"--- File: {filename} ---\n"
        return header + content
    except Exception as e:
        print(f"❌ Error reading file {filepath}: {e}")
        return None

def get_available_models(base_url: str) -> List[str]:
    """Fetch list of models from Ollama server."""
    try:
        # Native Ollama endpoint for listing models
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        
        # Fallback for OpenAI compatible endpoints (often at /v1/models)
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['id'] for m in data.get('data', [])]
            
    except Exception as e:
        print(f"⚠️ Could not fetch models: {e}")
    return []

def compact_context(model: str, messages: List[Dict], base_url: str, use_openai: bool) -> List[Dict]:
    if len(messages) < 4:
        print("Not enough conversation to compact.")
        return messages

    print("🗜️ Compacting conversation...")
    history_text = "\n".join(
        f"{msg.get('role', 'assistant').upper()}: {_message_display_text(msg)}"
        for msg in messages
    )
    summary_prompt = (
        "You are an expert summarizer. Condense the entire conversation history "
        "into a clear, concise summary (max 400 words) that retains all important "
        "details, decisions, and context. Write in neutral third-person style."
    )

    # Prepare payload based on API type
    if use_openai:
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Conversation:\n\n{history_text}\n\nProvide compact summary:"}
            ],
            "stream": True,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
    else:
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Conversation:\n\n{history_text}\n\nProvide compact summary:"}
            ],
            "stream": True,
        }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=180)
        response.raise_for_status()
        full_summary = ""
        print("Summary: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                try:
                    raw = line.decode('utf-8')
                    if use_openai:
                        if not raw.startswith('data: '):
                            continue
                        raw = raw[6:]
                        if raw == '[DONE]':
                            continue
                        chunk = json.loads(raw)
                        if chunk.get("choices") and chunk["choices"][0].get("delta"):
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_summary += content
                    else:
                        chunk = json.loads(raw)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            print(content, end="", flush=True)
                            full_summary += content
                except json.JSONDecodeError:
                    continue
        print("\n")
        return [{"role": "assistant", "content": full_summary.strip()}]
    except Exception as e:
        print(f"❌ Compact failed: {e}")
        return messages

def _print_resume_hint(active_model: str = "") -> None:
    if CURRENT_SESSION_ID:
        model = _resume_model(active_model) or "<model>"
        prefix = LAUNCH_COMMAND_PREFIX or f"python {shlex.quote(os.path.basename(__file__))}"
        print(f"💡 To resume: {prefix} {shlex.quote(model)} -r {shlex.quote(CURRENT_SESSION_ID)}")


def chat_with_ollama(model: str, base_url: str, use_openai: bool, enable_tools: bool, render_mode: str, guardrails_mode: str):
    global ATTACHMENT_BUFFER

    # Determine endpoint based on flag
    if use_openai:
        url = f"{base_url}/v1/chat/completions"
        print(f"🚀 Using OpenAI Compatible API at: {url}")
    else:
        url = f"{base_url}/api/chat"
        print(f"🚀 Using Native Ollama API at: {url}")

    messages: List[Dict] = load_context()

    # Key bindings: Enter submits, Alt-Enter / Shift-Enter / Ctrl-J insert newline
    kb = KeyBindings()

    @kb.add('enter')
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add('escape', 'enter')  # Alt-Enter
    def _newline_alt(event):
        event.current_buffer.insert_text('\n')

    @kb.add('c-j')  # Ctrl-J universal fallback
    def _newline_ctrl(event):
        event.current_buffer.insert_text('\n')

    session = PromptSession(
        history=FileHistory(HISTORY_FILE),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=True,
        key_bindings=kb,
        prompt_continuation=lambda width, line_number, wrap_count: '... ',
    )

    # Only show startup banner/help when NOT resuming an existing session
    if not CONTINUE_SESSION:
        print(f"🤖 Chat with **{model}** started  [session: {CURRENT_SESSION_ID}]")
        print("Available commands:")
        print(" /exit, /quit, /bye → Save and exit")
        print(" /reset → Clear all context")
        print(" /compact → Summarize and shorten history")
        print(" /model [name] → Switch model (no name lists available models)")
        print(" /file <filename> → Add file to next message")
        print(" /clearfiles → Clear attachment buffer")
        print(" /tools → List available local tools")
        print(" /sessions → List saved sessions for this folder")
        print(" /status → View session information")
        print(f" Render mode: {render_mode}")
        print(f" Guardrails: {guardrails_mode}")
        print(" Enter → submit | Alt-Enter / Shift-Enter / Ctrl-J → new line\n")

    while True:
        try:
            user_input = session.prompt(">>> ").strip()
            if not user_input:
                continue

            # Handle commands
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            # If input starts with '/' but doesn't match a known command,
            # treat it as a typo and do not forward to the model.
            known_cmds = {
                '/exit', '/quit', '/bye',
                '/reset', '/compact', '/model',
                '/file', '/clearfiles', '/status', '/sessions', '/tools', '/?', '/help',
                '/redraw'
            }
            if cmd.startswith('/') and cmd not in known_cmds:
                print(f"⚠️ Unknown command '{cmd}'. Messages starting with '/' are commands — not sent."
                      " If you meant to send a message starting with '/', escape or remove the leading '/'.")
                continue

            if cmd in ['/exit', '/quit', '/bye']:
                saved_count = save_context(messages)
                if saved_count == -1:
                    print("🗑️ Context file removed (no messages to save). Goodbye!")
                elif saved_count > 0:
                    print(f"💾 Context saved ({saved_count} messages). Goodbye!")
                    _print_resume_hint(model)
                else:
                    print("⚠️ Context could not be saved. Goodbye!")
                break

            elif cmd == '/reset':
                messages = []
                if os.path.exists(CONTEXT_FILE):
                    os.remove(CONTEXT_FILE)
                ATTACHMENT_BUFFER.clear()
                print("🗑️ Conversation fully reset.\n")
                continue

            elif cmd == '/compact':
                messages = compact_context(model, messages, base_url, use_openai)
                save_context(messages)
                continue

            elif cmd == '/model':
                # If no argument, list models
                if len(parts) < 2:
                    print(f"🔍 Fetching available models from {base_url}...")
                    models = get_available_models(base_url)
                    if models:
                        print(f"📋 Available models ({len(models)}):")
                        # Print in 4 columns
                        # Determine max width for columns
                        max_len = max(len(m) for m in models) + 4
                        cols = 2
                        for i in range(0, len(models), cols):
                            row = models[i:i+cols]
                            # Pad each item to align columns
                            print("  " + "".join(item.ljust(max_len) for item in row))
                        print(f"\nCurrent model: {model}")
                        print("Usage: /model <name>")
                    else:
                        print("⚠️ No models found or failed to retrieve list.")
                    continue
                
                # If argument provided, switch model
                new_model = parts[1].strip()
                model = new_model
                if CURRENT_SESSION_ID:
                    _write_session_meta(CURRENT_SESSION_ID, model=model)
                print(f"🔄 Model switched to: {model}")
                continue

            elif cmd == '/file':
                if len(parts) < 2:
                    print("Usage: /file <filename>")
                    continue
                filepath = parts[1].strip()
                content = read_file_content(filepath)
                if content:
                    ATTACHMENT_BUFFER.append(content)
                    filename = os.path.basename(filepath)
                    print(f"✅ Added to attachments: {filename} ({len(content)} characters)")
                continue

            elif cmd == '/tools':
                if not enable_tools:
                    print("🧰 Local tools are disabled for this session.")
                    continue
                print(f"🧰 Available local tools ({len(TOOL_SCHEMAS)}):")
                for schema in TOOL_SCHEMAS:
                    function = schema["function"]
                    tool_spec = TOOL_REGISTRY.get(function["name"], {})
                    tool_mode = "read-only" if tool_spec.get("read_only", False) else "guarded"
                    print(f" - {function['name']} [{_tool_display_source(tool_spec)}, {tool_mode}]: {function['description']}")
                continue

            elif cmd in ['/?', '/help']:
                # Show available commands/help
                print("Available commands:")
                print(" /exit, /quit, /bye → Save and exit")
                print(" /reset → Clear all context")
                print(" /compact → Summarize and shorten history")
                print(" /model [name] → Switch model (no name lists available models)")
                print(" /file <filename> → Add file to next message")
                print(" /clearfiles → Clear attachment buffer")
                print(" /tools → List available local tools")
                print(" /sessions → List saved sessions for this folder")
                print(" /status → View session information")
                print(" /? or /help → Show this help text")
                print(" Enter → submit | Alt-Enter / Shift-Enter / Ctrl-J → new line\n")
                continue

            elif cmd == '/clearfiles':
                ATTACHMENT_BUFFER.clear()
                print("🧹 Attachment buffer cleared.")
                continue

            elif cmd == '/redraw':
                # Save any recent messages, clear screen, and reload context as if resuming
                save_context(messages)
                # Clear terminal screen (ANSI escape code)
                print("\033c", end="")
                # Reset attachment buffer
                ATTACHMENT_BUFFER.clear()
                # Reload context (this will also print the previous conversation and set CONTINUE_SESSION)
                messages = load_context()
                continue

            elif cmd == '/sessions':
                all_sessions = _sessions_for_cwd(os.getcwd())
                if not all_sessions:
                    print("No saved sessions for this folder.")
                else:
                    print(f"\nSaved sessions for this folder ({len(all_sessions)}):")
                    for s in all_sessions:
                        meta = s["meta"]
                        ts = meta.get("last_used", "?")[:16].replace("T", " ")
                        count = meta.get("message_count", "?")
                        m = meta.get("model", "?")
                        active = " ← current" if s["id"] == CURRENT_SESSION_ID else ""
                        locked = " [in use]" if s["locked"] and s["id"] != CURRENT_SESSION_ID else ""
                        print(f"  {s['id']}  {m:<30}  {ts}  ({count} msgs){active}{locked}")
                    print(f"\nResume with: ollama_chat <model> -r <session-id>")
                print()
                continue

            elif cmd == '/status':
                print("\n--- 📊 Session Status ---")
                print(f"🆔 Session ID: {CURRENT_SESSION_ID}")
                print(f"🤖 Active Model: {model}")
                print(f"🌐 API Endpoint: {base_url}")
                print(f"🔌 API Mode: {'OpenAI Compatible' if use_openai else 'Native Ollama'}")
                print(f"📂 Session Dir: {_session_dir(CURRENT_SESSION_ID)}")

                # Calculate context size
                ctx_len = len(messages)
                ctx_chars = sum(len(m.get('content', '')) for m in messages)
                print(f"💬 Context Size: {ctx_len} messages ({ctx_chars:,} chars)")

                # Check file status
                file_exists = "Yes" if os.path.exists(CONTEXT_FILE) else "No"
                print(f"💾 File Saved: {file_exists}")

                # Attachment buffer status
                print(f"📎 Attachments: {len(ATTACHMENT_BUFFER)} file(s) pending")
                print(f"🧰 Local Tools: {'Enabled' if enable_tools else 'Disabled'} ({len(TOOL_SCHEMAS) if enable_tools else 0} available)")
                print(f"🎨 Render Mode: {render_mode}")
                print(f"🛡️ Guardrails: {guardrails_mode}")
                if EXTERNAL_TOOL_FILES:
                    print(f"📦 External Tool Files: {len(EXTERNAL_TOOL_FILES)} loaded")
                print("-------------------------\n")
                continue

            # Normal message - attach files if any
            full_user_message = user_input
            if ATTACHMENT_BUFFER:
                full_user_message += "\n\n" + "\n\n".join(ATTACHMENT_BUFFER)
                print(f"📎 Sending with {len(ATTACHMENT_BUFFER)} attached file(s)")
                ATTACHMENT_BUFFER.clear()

            messages.append({"role": "user", "content": full_user_message})

            print("<<< ", end="", flush=True)

            request_uses_tools = enable_tools
            while True:
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                }
                if use_openai:
                    # Proxy passthrough doesn't inject max_tokens; set it here
                    # to avoid NIM's tiny default (32 tokens).
                    payload["max_tokens"] = DEFAULT_MAX_TOKENS
                if request_uses_tools:
                    payload["tools"] = TOOL_SCHEMAS
                    if use_openai:
                        payload["tool_choice"] = "auto"

                assistant_message, normalized_calls, tools_active_for_turn = _stream_chat_response(
                    url,
                    payload,
                    use_openai,
                    allow_tool_fallback=True,
                    render_mode=render_mode,
                )
                messages.append(assistant_message)

                content = assistant_message.get("content") or ""
                if content and render_mode == "markdown":
                    print()
                    try:
                        render_markdown_to_terminal(content)
                    except Exception:
                        print(content)

                if not normalized_calls:
                    if not content:
                        print()
                    break

                print()
                for call in normalized_calls:
                    print(f"[tool] { _tool_summary(call) }")
                    try:
                        result = execute_tool_call(call["name"], call["arguments"], guardrails_mode)
                    except Exception as exc:
                        result = _tool_error_message(exc)
                    messages.append(_tool_result_message(call["name"], result, call["id"], use_openai))

                request_uses_tools = tools_active_for_turn

        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to server at {base_url}. Is it running?")
            _print_resume_hint(model)
            break
        except KeyboardInterrupt:
            saved_count = save_context(messages)
            if saved_count == -1:
                print("\n\n👋 Chat ended. Context file removed (no messages to save).")
            elif saved_count > 0:
                print(f"\n\n👋 Chat ended. Context saved ({saved_count} messages).")
                _print_resume_hint(model)
            else:
                print("\n\n👋 Chat ended. Context could not be saved.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    _capture_launch_command_prefix()
    parser = argparse.ArgumentParser(description="Chat with Ollama models via CLI.")
    parser.add_argument("model", help="The model name to use (e.g., llama3.2)")
    parser.add_argument("-o", "--openai", action="store_true", help="Use OpenAI compatible API endpoint")
    parser.add_argument("-H", "--host", default="localhost", help="Hostname or IP address of the Ollama server (default: localhost)")
    parser.add_argument("-P", "--port", default="11434", help="Port of the Ollama server (default: 11434)")
    parser.add_argument("-r", "--resume", metavar="SESSION_ID", default=None,
                        help="Resume a specific session by ID (see /sessions inside chat)")
    parser.add_argument("--new", action="store_true",
                        help="Always start a new session, ignoring any existing saved sessions")
    parser.add_argument("-c", "--clean", action="store_true",
                        help="Start a new session with an empty context (alias for --new)")
    parser.add_argument("-t", "--tools", action="append", default=[], help="Path to a JSON file defining additional command-backed tools. Can be passed multiple times.")
    parser.add_argument("--no-tools", action="store_true", help="Disable local tool definitions for this session")
    parser.add_argument("--render-mode", choices=["markdown", "stream"], default=DEFAULT_RENDER_MODE, help="How assistant replies are shown: buffered markdown view or live raw stream.")
    parser.add_argument("--guardrails", choices=["confirm-destructive", "read-only", "off"], default=DEFAULT_GUARDRAILS_MODE, help="How destructive tool calls are handled.")
    args = parser.parse_args()

    configure_tool_registry(args.tools)

    # Construct base URL
    base_url = f"http://{args.host}:{args.port}"

    # Resolve which session to use and point the path globals at it.
    new_session = args.new or args.clean
    session_id = resolve_session(args.resume, new_session, args.model)
    _init_session_paths(session_id)

    # Make signals trigger clean exit so atexit handlers run
    def _handle_signal(signum, frame):
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            pass

    # Acquire per-session pidfile to prevent two processes on the same session.
    acquire_pidfile()
    atexit.register(release_pidfile)

    chat_with_ollama(
        args.model,
        base_url,
        args.openai,
        enable_tools=not args.no_tools,
        render_mode=args.render_mode,
        guardrails_mode=args.guardrails,
    )

if __name__ == "__main__":
    main()
