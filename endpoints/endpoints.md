# Endpoint Descriptor Files

This folder contains static endpoint profiles used by ooProxy to recognize known upstream providers and skip trial-and-error.

Each file is a JSON object. A profile is optional for any OpenAI-compatible backend, but adding one is useful when the provider has a stable base URL and known quirks.

Profiles are data-driven. When a provider needs variables such as `account_id` to build the model-list URL, those variables are declared in the profile as part of the `models` section.

This file is the authoritative reference for researching, implementing, and maintaining endpoint profile files. The main README should only point here rather than duplicating this guidance.

## Purpose

An endpoint descriptor answers two questions:

1. How does ooProxy recognize that the configured upstream URL belongs to this provider?
2. What request and response behavior should ooProxy assume before it starts probing and retrying?

## Full Schema

All supported top-level items are shown below. Any omitted item falls back to the runtime defaults documented in this file.

```json
{
  "id": "provider_name",
  "match": {
    "schemes": ["https"],
    "host_equals": ["api.example.com"],
    "host_suffixes": ["example.com"],
    "ports": [443],
    "path_prefixes": ["/v1"]
  },
  "models": {
    "method": "GET",
    "path": "models",
    "format": "openai",
    "variables": {
      "account_id": {
        "method": "GET",
        "path": "/v1/accounts",
        "json_path": "accounts.0.name",
        "strip_prefix": "accounts/"
      }
    },
    "items_path": "models",
    "owned_by": "provider-name",
    "fields": {
      "id": "name",
      "created": "createTime",
      "modified_at": "updateTime",
      "context_length": "contextLength",
      "parent_model": "importedFrom"
    },
    "capabilities": {
      "embedding_when": {
        "path": "kind",
        "equals": "EMBEDDING_MODEL"
      },
      "completion_when_any_present": ["conversationConfig", "contextLength"],
      "tools_when_truthy": "supportsTools"
    }
  },
  "health": {
    "mode": "internal-ready",
    "method": "GET",
    "path": "/health"
  },
  "chat": {
    "path": "chat/completions",
    "streaming": "sse",
    "tools": "trial",
    "system_prompt": "supported",
    "ttfb_timeout": 30
  },
  "behavior": {
    "strip_stream_options": true,
    "strip_tools": true,
    "strip_tool_choice_auto": true,
    "normalize_messages": true,
    "embedded_tool_call_text": true,
    "embedded_tool_call_stop_finish": true
  }
}
```

## Top-Level Items

### `id`

- Type: string
- Required: no, but it should always be present
- Purpose: human-readable profile identifier used in logs and tests
- Default if omitted: filename stem

Use a stable snake_case name such as `openrouter`, `together_ai`, or `fireworks_ai`.

### `match`

- Type: object
- Required: no
- Purpose: describes how ooProxy matches a configured base URL to this profile

All populated match constraints are combined with logical AND. Inside each list, any listed value may match.

#### `match.schemes`

- Type: array of strings
- Example: `["https"]`
- Match rule: the URL scheme must equal one of these values, case-insensitive
- Default: no scheme constraint

Use this when the canonical endpoint is restricted to `http` or `https` and the profile should not match the other transport.

#### `match.host_equals`

- Type: array of strings
- Example: `["api.fireworks.ai"]`
- Match rule: the URL hostname must equal one of these values, case-insensitive
- Default: no host equality constraint

Use this when the provider has one or more exact canonical hostnames.

#### `match.host_suffixes`

- Type: array of strings
- Example: `["together.xyz"]`
- Match rule: the URL hostname must equal the suffix or end with `.` plus the suffix
- Default: no host suffix constraint

Use this when the provider uses multiple subdomains under one parent domain.

#### `match.ports`

- Type: array of integers
- Example: `[11434]`
- Match rule: the parsed URL port must equal one of these values; if the URL omits a port, ooProxy uses `80` for `http` and `443` for `https`
- Default: no port constraint

This is mostly useful for local or self-hosted endpoints.

#### `match.path_prefixes`

- Type: array of strings
- Example: `["/inference/v1", "/v1"]`
- Match rule: the URL path must start with one of these prefixes
- Default: no path constraint

Use this when a provider exposes a distinctive API root beneath the host.

### `models`

- Type: object
- Required: no
- Purpose: tells ooProxy how to fetch and normalize the upstream model list

#### `models.method`

- Type: string
- Default: `GET`
- Purpose: HTTP method used for the model-list request

This is endpoint-specific, so the profile controls the exact method used to retrieve the model list.

#### `models.path`

- Type: string
- Default: `models`
- Purpose: request path joined onto the configured base URL after replacing any declared variables such as `{account_id}`

Examples:

- `models` for OpenAI-style `/v1/models`
- `/api/tags` for Ollama-style model listing
- `/v1/accounts/{account_id}/models` for a provider that requires account discovery before listing models

Absolute paths are allowed. Because ooProxy uses URL joining, a path beginning with `/` replaces the path portion of the configured base URL while keeping the same scheme and host.

#### `models.variables`

- Type: object
- Default: `{}`
- Purpose: declares named variables that must be resolved before `models.path` can be requested

Each variable entry supports these items:

- `method`: HTTP method used to fetch the discovery payload. Default: `GET`
- `path`: request path used to fetch the discovery payload
- `json_path`: dot-separated path into the returned JSON payload. List indexes are written as numeric segments such as `accounts.0.name`
- `strip_prefix`: optional string removed from the start of the extracted value before substitution

Example:

```json
{
  "models": {
    "method": "GET",
    "path": "/v1/accounts/{account_id}/models",
    "format": "object_list",
    "variables": {
      "account_id": {
        "method": "GET",
        "path": "/v1/accounts",
        "json_path": "accounts.0.name",
        "strip_prefix": "accounts/"
      }
    }
  }
}
```

#### `models.items_path`

- Type: string
- Default: empty
- Purpose: dot-separated path to the list of model entries when `models.format` is a profile-driven object list

#### `models.fields`

- Type: object
- Default: `{}`
- Purpose: maps normalized output fields to dot-separated paths inside each upstream model entry

Supported field keys in the current code:

- `id`
- `created`
- `modified_at`
- `context_length`
- `parent_model`

#### `models.owned_by`

- Type: string
- Default: none
- Purpose: optional constant value copied into each normalized model entry as `owned_by`

#### `models.capabilities`

- Type: object
- Default: `{}`
- Purpose: optional rules for inferring normalized model capabilities from provider-specific fields

Supported capability rules in the current code:

- `embedding_when`: object with `path` and `equals`
- `completion_when_any_present`: array of entry paths; if any is present, `completion` is added
- `tools_when_truthy`: entry path; if truthy, `tools` is added and `completion` is also ensured
- `default_embedding`: boolean fallback for embedding-only catalogs

#### `models.format`

- Type: string
- Default: `openai`
- Purpose: tells ooProxy how to normalize the returned payload

Supported values in the current code:

- `openai`: upstream returns an object with a `data` array, already in OpenAI shape
- `array`: upstream returns a bare JSON array of model objects
- `ollama_tags`: upstream returns an Ollama `/api/tags` payload with a top-level `models` array
- `object_list`: upstream returns an object containing a list of model entries at `models.items_path`, and the profile maps each entry into the normalized OpenAI list shape using `models.fields`, `models.owned_by`, and optional `models.capabilities`

### `health`

- Type: object
- Required: no
- Purpose: controls how `/readyz` decides whether the upstream is ready

#### `health.mode`

- Type: string
- Default: `internal-ready`

Supported values in the current code:

- `internal-ready`: ooProxy reports ready without probing a provider-specific endpoint
- `http`: ooProxy performs an HTTP request to the configured health path

#### `health.method`

- Type: string
- Default: `GET`
- Purpose: HTTP method used for the readiness probe when `health.mode` is `http`

#### `health.path`

- Type: string
- Default: none
- Purpose: path joined onto the configured base URL for readiness probing

If `health.mode` is `http`, this should usually be present.

### `chat`

- Type: object
- Required: no
- Purpose: describes how ooProxy should talk to the upstream chat-completions endpoint

#### `chat.path`

- Type: string
- Default: `chat/completions`
- Purpose: request path joined onto the configured base URL for chat requests

Examples:

- `chat/completions` for OpenAI-compatible providers
- `/api/chat` for an Ollama upstream

#### `chat.streaming`

- Type: string
- Default: `sse`
- Purpose: tells ooProxy whether upstream streaming should be treated as supported

Values treated as streaming-capable by the current code:

- `sse`
- `openai-sse`
- `supported`
- `trial`

Any other value is treated as streaming-disabled, and ooProxy forces non-streaming requests.

Practical guidance:

- Use `sse` for standard OpenAI-style server-sent events
- Use `trial` if documentation is unclear and you want behavior learning to remain active
- Use a non-supported value such as `disabled` if the provider does not support streaming on this endpoint

#### `chat.tools`

- Type: string
- Default: `trial`
- Purpose: tells ooProxy whether to send `tools` and `tool_choice`

Values that cause ooProxy to strip tools immediately:

- `off`
- `disabled`
- `unsupported`
- `none`
- `never`
- `native`

Values that leave tools intact:

- `supported`
- `trial`

Practical guidance:

- Use `supported` when function calling is documented and reliable
- Use `trial` when support is uncertain or model-dependent
- Use `unsupported` when the provider rejects `tools` or `tool_choice`

#### `chat.system_prompt`

- Type: string
- Default: `supported`
- Purpose: tells ooProxy whether system-role messages can be passed through unchanged

Values that cause ooProxy to normalize messages first:

- `unsupported`
- `normalize`
- `inline-user`
- `merge`

Values that leave messages unchanged:

- `supported`
- `trial`

Normalization currently means:

- Drop `system` messages
- Collapse adjacent messages with the same role

Use a normalization value when the provider or model rejects system-role messages or needs simplified role formatting.

#### `chat.ttfb_timeout`

- Type: number (seconds)
- Required: no
- Default: 30 (seconds)
- Purpose: maximum seconds to wait for the first streaming byte (response headers) when issuing a streaming (`stream=true`) request. If the upstream accepts the connection but does not send response headers within this interval, ooProxy will fall back to a non-streaming request and synthesize a streamed response.

Practical guidance:

- Set this to a lower value for backends that are known to respond quickly over SSE.
- Set this to a higher value for backends that may take longer to start streaming but do eventually send events.
- Omit the field to use the global default (30s).

### `behavior`

- Type: object of boolean values
- Required: no
- Default: `{}`
- Purpose: preloads known request or response quirks so ooProxy does not have to discover them by retrying

Supported behavior keys in the current code:

#### `behavior.strip_stream_options`

- If `true`, remove `stream_options` from OpenAI-compatible requests before sending them upstream
- Use when the provider supports streaming but rejects the OpenAI `stream_options` field

#### `behavior.strip_tools`

- If `true`, remove `tools` and `tool_choice` before sending requests upstream
- Use when the provider or endpoint rejects tool-calling fields entirely

#### `behavior.strip_tool_choice_auto`

- If `true`, keep `tools` but remove `tool_choice` when it is set to `auto`
- Use when tool calling works but the provider rejects the automatic tool-choice variant

#### `behavior.normalize_messages`

- If `true`, normalize messages before sending requests upstream
- This uses the same transform as `chat.system_prompt=normalize`-style values

#### `behavior.embedded_tool_call_text`

- If `true`, suppress assistant text content when a tool call is present and the provider emits both together
- Use when the provider includes stray natural-language text alongside structured tool calls and that text should not surface in the translated Ollama response

#### `behavior.embedded_tool_call_stop_finish`

- If `true`, reinterpret a `finish_reason` of `stop` as `tool_calls` when a tool call is present
- Use when the provider emits tool calls but marks the finish reason incorrectly for ooProxy's downstream translation

## Runtime Defaults

If a descriptor omits a field, ooProxy uses these defaults:

```json
{
  "models": {
    "method": "GET",
    "path": "models",
    "format": "openai",
    "variables": {},
    "items_path": "",
    "fields": {},
    "capabilities": {}
  },
  "chat": {
    "path": "chat/completions",
    "streaming": "sse",
    "tools": "trial",
    "system_prompt": "supported"
  },
  "health": {
    "mode": "internal-ready",
    "method": "GET"
  },
  "behavior": {}
}
```

For matching fields, omission means no constraint.

## Information To Gather Before Writing A Profile

To create a correct descriptor, collect the following facts from provider docs or by direct API calls.

### 1. Canonical base URL

You need:

- the documented base URL users should pass to `--url`
- whether the provider expects `/v1`, `/api/v1`, `/inference/v1`, or another prefix
- whether multiple equivalent base URLs should map to the same profile

This determines `match.host_equals`, `match.host_suffixes`, and `match.path_prefixes`.

### 2. Whether scheme, host, and port are distinctive

You need:

- the expected scheme, if the provider is only valid over `http` or `https`
- exact hostnames, if the provider uses fixed public hosts
- suffix matching, if the provider serves multiple subdomains
- explicit port requirements, if the endpoint is local or self-hosted

This determines the rest of `match`.

### 3. Model-list endpoint and payload shape

You need:

- the path used to list models
- the HTTP method used for the list request
- whether model listing lives on the same API root as chat inference or on a separate management API path
- whether listing models requires a discovery step such as resolving an account id first
- the exact variables needed to build the final list URL and how to extract them from discovery responses
- whether authentication is required on that path
- whether the response is OpenAI-style, a bare array, or a provider-specific object such as Ollama tags

This determines `models.method`, `models.path`, `models.variables`, and `models.format`.

### 4. Chat endpoint path

You need:

- the path for chat completions or the equivalent chat API
- whether it is genuinely OpenAI-compatible or requires a provider-specific path

This determines `chat.path`.

### 5. Streaming behavior

You need:

- whether the chat endpoint supports `stream=true`
- whether the streamed response is standard SSE
- whether streaming only works on some models or is globally unsupported

This determines `chat.streaming` and whether any extra behavior flags are needed.

### 6. Tool-calling support

You need:

- whether `tools` are accepted at all
- whether only function tools are supported
- whether `tool_choice: auto` is accepted
- whether support varies by model and should remain `trial`

This determines `chat.tools` and possibly `behavior.strip_tool_choice_auto`.

### 7. System-message handling

You need:

- whether `system` role messages are accepted
- whether they are ignored, rejected, or must be merged into user content
- whether role formatting is stricter than OpenAI's message schema

This determines `chat.system_prompt` and possibly `behavior.normalize_messages`.

### 8. Readiness probe behavior

You need:

- whether ooProxy can treat the provider as ready immediately
- whether there is a cheap authenticated endpoint suitable for readiness checks
- which HTTP method and path should be used

This determines `health.mode`, `health.method`, and `health.path`.

### 9. Known translation quirks

You need:

- whether the provider returns tool calls and assistant text in the same message chunk
- whether tool-call responses end with `finish_reason=stop` instead of `tool_calls`
- whether `stream_options` causes request failures
- whether any other retry-discovered quirk is stable enough to encode statically

This determines `behavior`.

## Recommended Research Workflow

1. Start with the provider docs and record the canonical base URL.
2. Check the model-list endpoint and save one real response sample.
3. Check chat completions with and without `stream=true`.
4. Check tool calling with a minimal single-function example.
5. Check a request containing a `system` message.
6. Confirm whether scheme matching is needed and whether the endpoint requires a non-default port.
7. Add only the fields that are known and stable; rely on defaults for the rest.
8. Add or update a targeted test in `tests/test_endpoint_profiles.py`.

## Implementation Rules

When adding or changing a profile, keep these constraints in mind:

- The default assumption is OpenAI-standard behavior.
- Provider-specific differences must be encoded in profile data, not in provider-named runtime code paths.
- If a provider needs extra discovery state such as `account_id`, declare it under `models.variables`.
- If a provider returns a non-standard model-list object, use the generic profile mapping fields before considering runtime changes.
- If a behavior cannot be expressed by the current profile schema, extend the schema generically and document the new field here.

## Minimal Example

For a normal OpenAI-compatible provider with a unique host and no quirks:

```json
{
  "id": "example_provider",
  "match": {
    "schemes": ["https"],
    "host_equals": ["api.example.com"],
    "ports": [443],
    "path_prefixes": ["/v1"]
  },
  "chat": {
    "tools": "supported"
  }
}
```

## When Not To Add A Profile

Do not add a static profile if:

- the provider is only reachable through user-specific custom domains
- the behavior is too model-specific and is better learned dynamically
- the endpoint is nominally OpenAI-compatible and the defaults already work well

In those cases, ooProxy can still operate using dynamic behavior discovery.