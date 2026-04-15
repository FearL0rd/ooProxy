# ooProxy

A lightweight proxy that impersonates an [Ollama](https://ollama.com/) server locally while forwarding requests to any remote OpenAI-compatible API backend (NVIDIA NIM, OpenAI, Groq, Together AI, OpenRouter, local Ollama, etc.).

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

ooProxy listens on `localhost:11434` and exposes the Ollama-native endpoints used by VS Code and Open WebUI (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/embeddings`, `/api/ps`) plus OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses`) and an Anthropic-compatible bridge at `/v1/messages`. It translates requests to OpenAI format where needed, forwards them to the configured remote backend, and translates the responses back — including streaming.

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
git clone https://github.com/MockbaTheBorg/ooProxy.git
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

Or store the remote key once and let ooProxy resolve it from the remote URL host:

```bash
python tools/ooproxy_keys.py --host integrate.api.nvidia.com --key nvapi-YOUR_KEY_HERE
python ooproxy.py --serve --url https://integrate.api.nvidia.com/v1
```

Or using environment variables:

```bash
export OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
export OPENAI_API_KEY=nvapi-YOUR_KEY_HERE
python ooproxy.py --serve
```

The proxy starts on `http://127.0.0.1:11434` by default.

Health and readiness endpoints are also available:

- `GET /healthz` — process liveness
- `GET /readyz` — upstream readiness based on the active endpoint profile
- `GET /api/status` — Ollama-style readiness endpoint used by some clients

#### Options

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--url URL` | `OPENAI_BASE_URL` | _(none for `--serve`)_ | Remote API base URL; if omitted for `--serve`, ooProxy offers stored profile URLs from `~/.ooProxy/keys.json` |
| `--key KEY` | `OPENAI_API_KEY` | _(none)_ | API key for the remote backend |
| `-H, --host HOST` | — | `127.0.0.1` | Local address to bind |
| `--port PORT` | — | `11434` | Local port to listen on |

Global flags available to all CLI commands:

- `-j, --json` — emit JSON envelopes from CLI commands
- `-v, --verbose` — show more detail in CLI output and server logs
- `-d, --debug` — enable debug logging (implies `--verbose`)

If `--key` and `OPENAI_API_KEY` are both omitted, ooProxy looks up a stored key in `~/.ooProxy/keys.json` using the host portion of the remote URL.

When `--serve` is started without `--url`, ooProxy scans the shipped `endpoints/*.json` profiles and offers the profiled URLs whose host or host:port already has a stored key. If no profiled stored endpoint is available, `--serve` exits and asks for `--url` explicitly.

---

### List available models

```bash
python ooproxy.py --list \
    --url https://integrate.api.nvidia.com/v1 \
    --key nvapi-YOUR_KEY_HERE
```

Stored profiled endpoints are also used by `--list`, so this works after the `ooproxy_keys.py` step above:

```bash
python ooproxy.py --list
```

### Manage stored API keys

```bash
python tools/ooproxy_keys.py --host integrate.api.nvidia.com --key nvapi-YOUR_KEY_HERE
python tools/ooproxy_keys.py --host integrate.api.nvidia.com
python tools/ooproxy_keys.py
python tools/ooproxy_keys.py --host integrate.api.nvidia.com --delete
```

Keys are stored in `~/.ooProxy/keys.json`. The value is only weakly obfuscated using the endpoint string itself, so this avoids casual shoulder-surfing rather than providing strong cryptographic protection.

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

### Global and project-local `.ooProxy` folders

ooProxy uses `~/.ooProxy/` as its main global state directory. This is where shared data lives across all projects, including:

- `~/.ooProxy/keys.json` for stored API keys
- `~/.ooProxy/behavior.json` for learned per-model backend quirks
- `~/.ooProxy/sessions/` for resumable `ooproxy_chat.py` sessions
- `~/.ooProxy/tools/*.json` for your default tool definitions

You can also create a repo-local `.ooProxy/` folder inside an individual project when you want project-specific tools or related helper files. In practice, the important path today is `./.ooProxy/tools/*.json`.

Tool definitions are discovered in this order:

- `~/.ooProxy/tools/*.json`
- `./.ooProxy/tools/*.json`
- Any files passed explicitly with `-t/--tools`

Later definitions win, so a project-local tool overrides a global tool, and an explicit `-t` file overrides both. This makes `~/.ooProxy/` the shared default and `./.ooProxy/` the per-project override layer.

### ooproxy_chat CLI

`tools/ooproxy_chat.py` supports built-in tools plus additional external tools loaded from JSON files with `-t/--tools`.

For the external tool-file schema and the process for researching or implementing a new tool definition, see `examples/tools.md`.

It also keeps resumable per-folder sessions under `~/.ooProxy/sessions/`.

Example:

```bash
python tools/ooproxy_chat.py llama3.1 -t ./toolset.json
```

Resume or force a fresh session:

```bash
python tools/ooproxy_chat.py llama3.1 --resume SESSION_ID
python tools/ooproxy_chat.py llama3.1 --new
python tools/ooproxy_chat.py llama3.1 --clean
```

By default, tool definition files are discovered in this order:

- `~/.ooProxy/tools/*.json`
- `./.ooProxy/tools/*.json`
- Any files passed explicitly with `-t/--tools`

Later definitions override earlier ones, so a repo-local tool can override a global tool or a built-in one.

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

External tool processes also receive `OLLAMA_TOOL_CWD`, so relative paths can resolve against the active chat working directory.

Built-in guardrails are enabled by default in `tools/ooproxy_chat.py` with `--guardrails confirm-destructive`.

- Read-only built-in tools run automatically.
- `write_file` asks for confirmation before overwriting an existing path.
- `run_shell` asks for confirmation on commands that look destructive.
- External tools are treated as guarded unless they explicitly set `"read_only": true`.

Available guardrail modes:

```bash
python tools/ooproxy_chat.py openai/gpt-oss-120b --guardrails confirm-destructive
python tools/ooproxy_chat.py openai/gpt-oss-120b --guardrails read-only
python tools/ooproxy_chat.py openai/gpt-oss-120b --guardrails off
```

Other useful options:

- `-o, --openai` — talk to the proxy through `/v1/chat/completions` instead of `/api/chat`
- `--no-tools` — disable all local tool definitions for the session
- `--render-mode markdown|stream|hybrid` — control how assistant output is rendered in the terminal
- `-H, --host` / `-P, --port` — point the tool at a different ooProxy or Ollama endpoint

### ooproxy_chat regression test

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

# NVIDIA NIM
python ooproxy.py -s --url https://integrate.api.nvidia.com/v1 --key nvapi-...

# Or pick from stored endpoint profiles after saving a key
python tools/ooproxy_keys.py --host integrate.api.nvidia.com --key nvapi-...
python ooproxy.py -s

# Local Ollama used as the upstream backend
python ooproxy.py -s --url http://localhost:11434/v1
```

ooProxy also ships static endpoint profiles in `endpoints/*.json` for known providers, currently including NVIDIA NIM, OpenRouter, Together AI, Fireworks AI, and local Ollama. For the endpoint profile schema and the process for researching or implementing a new provider profile, see `endpoints/endpoints.md`.

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
- **Learned model behavior cache** — ooProxy records per-endpoint/model quirks in `~/.ooProxy/behavior.json`, including request-side retry flags and response-shape quirks such as embedded textual tool calls.
- **Vendor field stripping** — Backend-specific response fields (e.g. NVIDIA's `nvext`) are stripped before the response is returned to the client.
- **Assistant-shaped upstream errors** — Upstream API errors are converted into normal assistant replies or stream events on the Ollama, OpenAI Responses, and Anthropic-compatible surfaces so clients still receive a valid protocol response.
- **Reasoning-to-Ollama translation** — OpenAI-style `reasoning_content` is translated into Ollama-visible `<think>...</think>` blocks, including streaming.

Request retry rules are learned from actual upstream errors; response-shape quirks are learned when a model returns a successful but malformed-compatible response. For backends that behave correctly, requests pass straight through with no transformation.

## Route coverage

Implemented HTTP routes include:

- Native and health routes: `/`, `/healthz`, `/readyz`, `/api/status`, `/api/version`
- Ollama-compatible routes: `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/embeddings`, `/api/ps`
- OpenAI-compatible routes: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses`
- Anthropic-compatible route: `/v1/messages`
- Ollama model-management stubs: `/api/pull`, `/api/delete`, `/api/copy`, `/api/create`, `/api/push`, `/api/blobs/{digest}`

The model-management endpoints return no-op success responses so Ollama-oriented clients can stay happy even when the remote backend has no equivalent model lifecycle API.

---

## Project structure

```text
.gitignore
LICENSE
ooproxy.py
cli_contract.py
README.md
requirements.txt
endpoints/
  endpoints.md          # Endpoint profile format and notes
  fireworks_ai.json
  local_ollama.json
  nvidia_nim.json
  openrouter.json
  together_ai.json
examples/
  my_tools.json
  tools.md
  tui_qr.json
  tui_qr.py
modules/
  __init__.py
  list.py               # -l/--list
  serve.py              # -s/--serve
  _server/
    __init__.py
    app.py
    behavior.py
    client.py
    config.py
    endpoint_profiles.py
    key_store.py
    upstream_errors.py
    handlers/
      __init__.py
      chat.py
      embeddings.py
      generate.py
      models.py
      openai_compat.py
      stubs.py
      version.py
    translate/
      __init__.py
      models.py
      request.py
      response.py
      stream.py
tools/
  ooproxy_chat.py        # Interactive chat CLI with tool loading and sessions
  ooproxy_keys.py        # Manage ~/.ooProxy/keys.json
  ooproxy_list_models.py # Query model lists from an Ollama-compatible endpoint
```

---

## License

MIT
