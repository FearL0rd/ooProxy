import requests
import json
import os
import argparse
import atexit
import sys
import signal
from typing import List, Dict
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

CONTEXT_FILE = ".ollama_chat_context"
HISTORY_FILE = ".ollama_chat_history"
LOCK_FILE = ".ollama_chat_lock"
ATTACHMENT_BUFFER: List[Dict] = []

# Optional rich-based Markdown renderer for prettier terminal output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    _RICH_CONSOLE = Console()
except Exception:
    _RICH_CONSOLE = None

# When True, we're resuming an existing conversation and should
# avoid printing the startup banner.
CONTINUE_SESSION = False


def render_markdown_to_terminal(text: str, console: Console = None) -> None:
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


def acquire_pidfile(lockfile: str = LOCK_FILE) -> None:
    """Create an exclusive pidfile. Exit if a live PID holds the lock."""
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


def release_pidfile(lockfile: str = LOCK_FILE) -> None:
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
                role = ">>>" if msg["role"] == "user" else "<<<"
                # Print role header then render the message content using Rich
                print(f"{role}:")
                try:
                    render_markdown_to_terminal(msg.get('content', ''))
                except Exception:
                    # Fallback to plain printing if rendering fails
                    print(msg.get('content', ''))
                print("\n")
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
            return -1 # Indicates file was removed
        with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
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
    history_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)
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
            "stream": True
        }
    else:
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Conversation:\n\n{history_text}\n\nProvide compact summary:"}
            ],
            "stream": True
        }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=180)
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

def chat_with_ollama(model: str, base_url: str, use_openai: bool):
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
        print(f"🤖 Chat with **{model}** started")
        print("Available commands:")
        print(" /exit, /quit, /bye → Save and exit")
        print(" /reset → Clear all context")
        print(" /compact → Summarize and shorten history")
        print(" /model [name] → Switch model (no name lists available models)")
        print(" /file <filename> → Add file to next message")
        print(" /clearfiles → Clear attachment buffer")
        print(" /status → View session information")
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
                '/file', '/clearfiles', '/status', '/?', '/help',
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

            elif cmd in ['/?', '/help']:
                # Show available commands/help
                print("Available commands:")
                print(" /exit, /quit, /bye → Save and exit")
                print(" /reset → Clear all context")
                print(" /compact → Summarize and shorten history")
                print(" /model [name] → Switch model (no name lists available models)")
                print(" /file <filename> → Add file to next message")
                print(" /clearfiles → Clear attachment buffer")
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

            elif cmd == '/status':
                print("\n--- 📊 Session Status ---")
                print(f"🤖 Active Model: {model}")
                print(f"🌐 API Endpoint: {base_url}")
                print(f"🔌 API Mode: {'OpenAI Compatible' if use_openai else 'Native Ollama'}")
                print(f"📂 Context File: {CONTEXT_FILE}")

                # Calculate context size
                ctx_len = len(messages)
                ctx_chars = sum(len(m.get('content', '')) for m in messages)
                print(f"💬 Context Size: {ctx_len} messages ({ctx_chars:,} chars)")

                # Check file status
                file_exists = "Yes" if os.path.exists(CONTEXT_FILE) else "No"
                print(f"💾 File Saved: {file_exists}")

                # Attachment buffer status
                print(f"📎 Attachments: {len(ATTACHMENT_BUFFER)} file(s) pending")
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

            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }

            response = requests.post(url, json=payload, stream=True, timeout=180)
            response.raise_for_status()

            full_response = ""
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
                            if chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                            # Fallback: server responded in native Ollama format despite -o
                            elif "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                full_response += content
                        else:
                            chunk = json.loads(raw)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                full_response += content
                    except json.JSONDecodeError:
                        continue

            print()
            if full_response:
                messages.append({"role": "assistant", "content": full_response})
                # Render nicely if Rich is available
                try:
                    render_markdown_to_terminal(full_response)
                except Exception:
                    print(full_response)

        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to server at {base_url}. Is it running?")
            break
        except KeyboardInterrupt:
            saved_count = save_context(messages)
            if saved_count == -1:
                print("\n\n👋 Chat ended. Context file removed (no messages to save).")
            elif saved_count > 0:
                print(f"\n\n👋 Chat ended. Context saved ({saved_count} messages).")
            else:
                print("\n\n👋 Chat ended. Context could not be saved.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Chat with Ollama models via CLI.")
    parser.add_argument("model", help="The model name to use (e.g., llama3.2)")
    parser.add_argument("-o", "--openai", action="store_true", help="Use OpenAI compatible API endpoint")
    parser.add_argument("-H", "--host", default="localhost", help="Hostname or IP address of the Ollama server (default: localhost)")
    parser.add_argument("-P", "--port", default="11434", help="Port of the Ollama server (default: 11434)")
    parser.add_argument("-c", "--clean", action="store_true", help="Start with empty context (overwrites old context on exit)")
    args = parser.parse_args()

    # Construct base URL
    base_url = f"http://{args.host}:{args.port}"

    # Acquire pidfile to prevent concurrent sessions in same folder
    acquire_pidfile()

    # Ensure pidfile is removed on exit
    atexit.register(release_pidfile)

    # Make signals trigger clean exit so atexit handlers run
    def _handle_signal(signum, frame):
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            # some signals may not be available on all platforms
            pass

    # If requested, start with a clean context (delete existing context file).
    if getattr(args, "clean", False):
        try:
            if os.path.exists(CONTEXT_FILE):
                os.remove(CONTEXT_FILE)
                print(f"🧹 Starting with a clean context (removed {CONTEXT_FILE}).")
            else:
                print("🧹 Starting with a clean context (no existing context file).")
        except Exception as e:
            print(f"⚠️ Could not remove context file: {e}")
        # Also remove history file when clean flag is used
        try:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
                print(f"🧹 Removed history file: {HISTORY_FILE}")
            else:
                print("🧹 No history file to remove.")
        except Exception as e:
            print(f"⚠️ Could not remove history file: {e}")

    chat_with_ollama(args.model, base_url, args.openai)

if __name__ == "__main__":
    main()
