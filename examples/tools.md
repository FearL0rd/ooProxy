# Tool Definition Files

This folder can contain example JSON files for external tools loaded by `tools/ollama_chat.py`.

Tool files are data-driven. They describe command-backed tools that the chat CLI exposes to the model at runtime.

This file is the authoritative reference for researching, implementing, and maintaining external tool definition files.

## Purpose

A tool definition file answers two questions:

1. What tool schemas should be shown to the model?
2. How should ooProxy execute each tool call locally?

## Supported File Shapes

The loader accepts three top-level JSON shapes:

### 1. Object with `tools` array

```json
{
  "tools": [
    {
      "name": "git_status",
      "description": "Return git status for the current repository.",
      "read_only": true,
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "command": "git status --short"
    }
  ]
}
```

### 2. Single tool object

```json
{
  "name": "git_status",
  "description": "Return git status for the current repository.",
  "read_only": true,
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  },
  "command": "git status --short"
}
```

### 3. Top-level array of tool objects

```json
[
  {
    "name": "git_status",
    "description": "Return git status for the current repository.",
    "read_only": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    },
    "command": "git status --short"
  }
]
```

## Tool Object Shapes

Each tool entry can be written in either of these equivalent forms.

### Flat object form

```json
{
  "name": "git_status",
  "description": "Return git status for the current repository.",
  "read_only": true,
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  },
  "command": "git status --short"
}
```

### OpenAI-style function wrapper form

```json
{
  "type": "function",
  "function": {
    "name": "git_status",
    "description": "Return git status for the current repository.",
    "read_only": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    },
    "command": "git status --short"
  }
}
```

If a wrapped entry also places keys such as `description`, `parameters`, or `read_only` at the outer level, ooProxy copies them into `function` when missing there.

## Tool Schema

All supported keys for an external tool definition are shown below.

```json
{
  "name": "tool_name",
  "description": "What the tool does.",
  "read_only": true,
  "destructive": false,
  "display_directly": false,
  "parameters": {
    "type": "object",
    "properties": {
      "value": {
        "type": "string",
        "description": "Example argument."
      }
    },
    "required": ["value"]
  },
  "command": "echo {value}",
  "argv": ["python3", "tool.py", "--value", "{value}"],
  "cwd": ".",
  "timeout": 30
}
```

## Per-Tool Keys

### `name`

- Type: string
- Required: yes
- Purpose: tool name exposed to the model

This should be stable, ASCII-safe, and unique after all tool files are merged.

### `description`

- Type: string
- Required: no
- Default: `External command-backed tool.`
- Purpose: natural-language description shown to the model

Keep this precise. The model relies heavily on it when deciding whether to call the tool.

### `parameters`

- Type: object
- Required: no
- Default:

```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

- Purpose: JSON Schema-like parameter definition shown to the model

The loader requires `parameters` to be an object if provided.

In practice, use an object schema with:

- `type: "object"`
- `properties`
- `required`
- optional per-property metadata such as `description`, `default`, `enum`, and similar schema fields the model can use for guidance

ooProxy does not perform full JSON Schema validation locally. The schema is mainly for the model.

### `read_only`

- Type: boolean
- Required: no
- Default: `false`
- Purpose: marks the tool as safe to run without confirmation in permissive guardrail modes

If omitted, external tools are treated as guarded by default.

### `destructive`

- Type: boolean
- Required: no
- Default: `false`
- Purpose: explicitly marks the tool as destructive so guardrails always treat it cautiously

Use this for tools that can modify files, mutate state, delete data, or invoke privileged actions.

### `display_directly`

- Type: boolean
- Required: no
- Default: `false`
- Purpose: tells the chat flow to surface tool output directly to the user instead of only feeding it back into the model

Use this for utilities whose output is itself the desired final response, such as QR-code rendering or formatted previews.

### `command`

- Type: string
- Required: conditionally
- Purpose: shell command to run with placeholder expansion

If `command` is used:

- ooProxy executes it with `shell=True`
- placeholders like `{path}` or `{text}` are replaced from the tool arguments
- substituted values are shell-quoted before insertion

This is convenient, but it should be reserved for simple shell-oriented tools.

### `argv`

- Type: array
- Required: conditionally
- Purpose: argument vector to execute without a shell

If `argv` is used:

- ooProxy executes it with `shell=False`
- each element is expanded independently using tool arguments
- expansion is inserted as plain text, not shell-quoted, because there is no shell layer

Prefer `argv` for most tools. It is more predictable and avoids shell parsing issues.

### `cwd`

- Type: string
- Required: no
- Default: current chat working directory at runtime
- Purpose: working directory for the external process

If `cwd` is relative, ooProxy resolves it relative to the directory containing the tool JSON file.

### `timeout`

- Type: integer
- Required: no
- Default: `120`
- Purpose: subprocess timeout in seconds

Use a shorter timeout for quick utilities and a longer one only when the tool genuinely needs it.

## Execution Rules

### Exactly one of `command` or `argv`

External tools must define exactly one of these execution styles:

- `command`
- `argv`

If both are present, the tool is invalid.

If neither is present, ooProxy checks for a companion script named `<tool_name>.py` in the same folder as the JSON file.

If that companion script exists:

- ooProxy uses `[sys.executable, <tool_name>.py]` as `argv`
- `cwd` defaults to the JSON file's directory if not otherwise set

This is useful when you want a simple one-file Python tool next to its JSON definition.

### Placeholder expansion

Tool arguments can be referenced with placeholders such as `{path}` or `{text}` inside `command` or `argv` entries.

Behavior:

- placeholders must match argument names like `text`, `path`, or `count`
- missing arguments leave the placeholder unchanged
- non-string values are JSON-stringified before substitution
- shell commands use shell quoting during substitution
- `argv` entries use plain substitution without shell quoting

Example `command` tool:

```json
{
  "name": "show_head",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    },
    "required": ["path"]
  },
  "command": "head -n 20 {path}"
}
```

Example `argv` tool:

```json
{
  "name": "show_qr_tui",
  "argv": ["python3", "tui_qr.py", "--text", "{text}"]
}
```

## Process I/O And Environment

External tools receive the tool arguments in two ways:

- on stdin as a JSON object
- in the `OLLAMA_TOOL_ARGS` environment variable as the same JSON string

External tools also receive:

- `OLLAMA_TOOL_NAME`: the tool name
- `OLLAMA_TOOL_CWD`: the active chat working directory, which may be different from the subprocess `cwd`

This means a tool can:

- read structured arguments from stdin
- resolve relative user paths against `OLLAMA_TOOL_CWD`
- still run from a controlled subprocess `cwd` if needed

## Output And Error Handling

On success:

- stdout is returned if non-empty
- otherwise stderr is returned if non-empty
- otherwise ooProxy returns a JSON object containing the executed command, working directory, and exit code

On failure:

- the subprocess exit code becomes a tool error
- ooProxy returns a JSON error payload including command, cwd, exit code, stdout, and stderr

Tool output is truncated to keep chat context manageable.

## Guardrails

External tools interact with the chat CLI guardrails as follows:

- tools not marked `read_only: true` are treated as guarded
- tools marked `destructive: true` always trigger guardrail scrutiny
- shell-command tools may trigger confirmation if the requested command looks destructive
- in `read-only` guardrail mode, non-read-only external tools are denied

If you want a tool to run automatically in normal guarded sessions, mark it `read_only: true` only when that claim is actually correct.

## Discovery And Override Order

By default, `tools/ollama_chat.py` loads tool files in this order:

1. `~/.ooProxy/tools/*.json`
2. `./.ooProxy/tools/*.json`
3. Files passed explicitly with `-t/--tools`

Later definitions override earlier ones when tool names collide.

## Research Workflow

To create a good tool definition file, collect the following information first.

### 1. What the tool should do

You need:

- the exact user-visible action the tool should perform
- whether the output is intended for the model, the user, or both
- whether the tool is read-only or state-changing

This determines `name`, `description`, `read_only`, `destructive`, and `display_directly`.

### 2. What arguments the tool needs

You need:

- required inputs
- optional inputs
- argument types
- useful descriptions and defaults

This determines `parameters`.

### 3. How the tool should execute

You need:

- whether it is best expressed as a shell command or an explicit argv list
- whether a companion Python script would be clearer than embedding logic inline
- whether placeholder substitution is sufficient or stdin JSON should be parsed directly

This determines `command`, `argv`, and whether a companion script is appropriate.

### 4. Working directory behavior

You need:

- whether the subprocess should run in the repo root, the tool file directory, or the active chat cwd
- whether user-provided relative paths should resolve against `OLLAMA_TOOL_CWD`

This determines `cwd` and how the tool script should interpret paths.

### 5. Runtime limits and safety

You need:

- expected runtime
- whether the tool can block or hang
- whether it can overwrite files, mutate state, or run shell commands with side effects

This determines `timeout`, `read_only`, and `destructive`.

## Recommended Authoring Workflow

1. Start from a small JSON file with a single tool.
2. Prefer `argv` over `command` unless shell behavior is the point.
3. Keep the parameter schema minimal but precise.
4. Mark the tool `read_only` only if it truly has no side effects.
5. If output should be shown directly to the user, set `display_directly: true`.
6. If the tool logic grows, move it into a companion Python script.
7. Test the tool from `tools/ollama_chat.py` with a real prompt and verify argument passing, cwd handling, and guardrails.

## Implementation Rules

When adding or changing external tool files, keep these constraints in mind:

- Prefer the default flat object shape unless the wrapped `type: "function"` shape adds clarity.
- Prefer `argv` over `command` for reliability.
- Keep tool definitions data-driven; do not require ooProxy runtime changes for a one-off tool.
- Use companion scripts for non-trivial logic instead of long inline one-liners.
- Treat `read_only` as a safety contract, not a convenience switch.
- Keep descriptions and parameter docs concrete so the model can choose the tool correctly.

## Minimal Example

```json
{
  "tools": [
    {
      "name": "git_status",
      "description": "Return git status for the current repository.",
      "read_only": true,
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "command": "git status --short"
    }
  ]
}
```

## Companion Script Example

JSON file:

```json
{
  "name": "echo_args",
  "description": "Echo the provided arguments.",
  "read_only": true,
  "parameters": {
    "type": "object",
    "properties": {
      "value": {
        "type": "string"
      }
    },
    "required": ["value"]
  }
}
```

Companion script in the same folder named `echo_args.py`:

```python
import json
import sys

data = json.load(sys.stdin)
print(json.dumps({"received": data}, indent=2))
```

ooProxy will automatically execute that script as the tool backend.