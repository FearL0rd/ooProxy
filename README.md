# ooProxy

A lightweight proxy that impersonates an [Ollama](https://ollama.com/) server locally while forwarding requests to any remote OpenAI-compatible API backend (NVIDIA NIM, OpenAI, Groq, Together AI, etc.).

This lets you use tools that are hardcoded to talk to Ollama — such as **VS Code Copilot Chat** — with cloud-hosted models, without modifying the client.

---

## How it works

```
VS Code Copilot Chat
  (Ollama @ localhost:11434)
         │
         ▼
      ooProxy
  translates Ollama ↔ OpenAI format
         │
         ▼
  Remote OpenAI-compatible API
  (NVIDIA NIM, OpenAI, Groq, …)
```

ooProxy listens on `localhost:11434` and exposes the full Ollama API surface (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/v1/chat/completions`, etc.). It translates requests to OpenAI format, forwards them to the configured remote backend, and translates the responses back — including streaming.

---

## Requirements

- Python 3.11+
- Dependencies:

```
pip install -r requirements.txt
```

---

## Setup

```bash
git clone https://github.com/youruser/ooProxy.git
cd ooProxy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Start the proxy

```bash
python ooproxy.py --serve \
    --url https://integrate.api.nvidia.com/v1 \
    --key nvapi-YOUR_KEY_HERE
```

Or using environment variables:

```bash
export OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export OPENAI_API_KEY=nvapi-YOUR_KEY_HERE
python ooproxy.py --serve
```

The proxy starts on `http://127.0.0.1:11434` by default.

#### Options

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--url URL` | `OPENAI_BASE_URL` | NVIDIA NIM | Remote API base URL |
| `--key KEY` | `OPENAI_API_KEY` | _(none)_ | API key for the remote backend |
| `--port PORT` | — | `11434` | Local port to listen on |

---

### List available models

```bash
python ooproxy.py --list \
    --url https://integrate.api.nvidia.com/v1 \
    --key nvapi-YOUR_KEY_HERE
```

Output:

```
[system] meta/llama-3.3-70b-instruct
[system] meta/llama-3.1-8b-instruct
[system] google/gemma-3-12b-it
[system] mistralai/mistral-7b-instruct-v0.3
...
```

JSON output:

```bash
python ooproxy.py --list --json
```

### ollama_chat tool files

`tools/ollama_chat.py` supports built-in tools plus additional external tools loaded from JSON files with `-t/--tools`.

Example:

```bash
python tools/ollama_chat.py llama3.1 -t ./toolset.json
```

Example tool file:

```json
{
  "tools": [
    {
      "name": "git_status",
      "description": "Return git status for the current repository.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "command": "git status --short"
    },
    {
      "type": "function",
      "function": {
        "name": "echo_args",
        "description": "Echo stdin JSON arguments via Python.",
        "parameters": {
          "type": "object",
          "properties": {
            "value": {"type": "string"}
          },
          "required": ["value"]
        },
        "argv": ["python3", "scripts/echo_tool.py"],
        "cwd": "."
      }
    }
  ]
}
```

Command-backed tools receive the tool arguments on stdin as a JSON object and in the `OLLAMA_TOOL_ARGS` environment variable.

Built-in guardrails are enabled by default in `tools/ollama_chat.py` with `--guardrails confirm-destructive`.

- Read-only built-in tools run automatically.
- `write_file` asks for confirmation before overwriting an existing path.
- `run_shell` asks for confirmation on commands that look destructive.
- External tools are treated as guarded unless they explicitly set `"read_only": true`.

Available guardrail modes:

```bash
python tools/ollama_chat.py openai/gpt-oss-120b --guardrails confirm-destructive
python tools/ollama_chat.py openai/gpt-oss-120b --guardrails read-only
python tools/ollama_chat.py openai/gpt-oss-120b --guardrails off
```

### ollama_chat regression test

There is also an end-to-end regression script for the interactive tool-calling flow:

```bash
python tools/test_ollama_chat_e2e.py --model openai/gpt-oss-120b
```

By default it tests both the native `/api/chat` path and the OpenAI-compatible `/v1/chat/completions` path against an already running proxy.

To let the script start and stop its own proxy instance, pass a command such as:

```bash
python tools/test_ollama_chat_e2e.py \
  --model openai/gpt-oss-120b \
  --proxy-command './start.sh'
```

---

## Supported backends

Any OpenAI-compatible API works. Just change `--url`:

```bash
# OpenAI
python ooproxy.py -s --url https://api.openai.com/v1 --key sk-...

# Groq
python ooproxy.py -s --url https://api.groq.com/openai/v1 --key gsk_...

# Together AI
python ooproxy.py -s --url https://api.together.xyz/v1 --key ...

# Fireworks AI
python ooproxy.py -s --url https://api.fireworks.ai/inference/v1 --key fw_...

# NVIDIA NIM (default)
python ooproxy.py -s --url https://integrate.api.nvidia.com/v1 --key nvapi-...
```

---

## VS Code Copilot Chat setup

1. Install the [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) extension.
2. Open VS Code settings and configure Ollama as the provider:
   - Set the Ollama endpoint to `http://localhost:11434`
3. Start ooProxy pointing at your preferred backend.
4. Open Copilot Chat — the model list will be populated from the remote API.
5. Select a model and chat.

> **Tip:** Use instruction-tuned models (names ending in `-instruct` or `-it`). Base models like `gemma-7b` are not chat assistants and will not follow instructions correctly.

> **Tip:** When asking Copilot to modify a specific file, explicitly add it to context with `#file:yourfile.py` in the chat input.

### VS Code context quirks

When using official GitHub Copilot (with GitHub's own backend), VS Code automatically injects the active editor file, selected code, and workspace information into every request. **This automatic injection does not happen with custom Ollama backends.**

As a result, when using ooProxy you need to add context explicitly:

| What you want to share | How to add it |
|---|---|
| A specific file | `#file:path/to/file.py` in the chat input |
| Selected code | Highlight it, then right-click → *Copilot* → *Explain* or use `#selection` |
| The entire workspace | `#codebase` (may be slow on large projects) |

**Example:**

Instead of:
```
Describe the render_text function
```

Write:
```
#file:modules/list.py Describe the render_text function
```

Without the `#file:` reference the model has no visibility into your code and will either guess, hallucinate a file path, or ask you to provide the content manually.

---

## Resilience features

ooProxy automatically handles backend-specific quirks without requiring any configuration:

- **Tool stripping** — Some backends reject `tool_choice: "auto"`. ooProxy retries the request without tools if this error is received, so basic chat always works.
- **System role normalisation** — Some models (e.g. Gemma) do not support the `system` role. ooProxy retries with system messages removed if the backend rejects them.
- **Message alternation** — Some models require strict `user/assistant/user/assistant` alternation. ooProxy collapses consecutive same-role messages on retry.
- **Vendor field stripping** — Backend-specific response fields (e.g. NVIDIA's `nvext`) are stripped before the response is returned to the client.

All of these are triggered only on actual error responses — for backends that behave correctly, requests pass straight through with no transformation.

---

## Project structure

```
ooproxy.py              # CLI host — discovers and dispatches to modules/
cli_contract.py         # Module protocol (ModuleSpec, ResultEnvelope, …)
requirements.txt
modules/
  serve.py              # -s/--serve  start the proxy server
  list.py               # -l/--list   list remote models
  _server/              # Internal server package (not exposed as CLI modules)
    app.py              # FastAPI application factory
    client.py           # Async HTTP client for the remote backend
    config.py           # ProxyConfig (URL, key, port)
    handlers/           # Route handlers for each Ollama endpoint
    translate/          # Ollama ↔ OpenAI request/response translation
```

---

## License

MIT
