"""CLI module: list models available on the remote OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from cli_contract import ModuleSpec, OptionSpec, ResultEnvelope, command_result
from modules._server.client import OpenAIClient
from modules._server.config import ProxyConfig
from modules._server.endpoint_selection import resolve_profile_url

SPEC = ModuleSpec(
    name="list",
    action_flags=("-l", "--list"),
    help="List models available on the remote server",
    options=(
        OptionSpec(
            flags=("--url",),
            dest="url",
            help="Remote OpenAI-compatible base URL, including port if non-standard (env: OPENAI_BASE_URL)",
            metavar="URL",
        ),
        OptionSpec(
            flags=("--key",),
            dest="key",
            help="Remote API key (env: OPENAI_API_KEY)",
            metavar="KEY",
        ),
    ),
    usage_examples=(
        "ooproxy.py -l --url https://integrate.api.nvidia.com/v1 --key nvapi-xxx",
        "ooproxy.py -l  # select from keyed endpoint profiles",
    ),
)


async def _fetch_models(config: ProxyConfig) -> list[dict]:
    client = OpenAIClient(config)
    try:
        data = await client.get_models()
        return data.get("data", [])
    finally:
        await client.aclose()


def run(args) -> ResultEnvelope:
    args.url = resolve_profile_url(args)
    config = ProxyConfig.from_args(args)
    models = asyncio.run(_fetch_models(config))
    result = command_result("list", config.url, data=models)
    # Attach verbosity flags so render_text can use them
    result._args = args
    return result


def render_text(result: ResultEnvelope) -> str:
    if not result.data:
        return "(no models returned)"
    args = getattr(result, "_args", None)
    debug = getattr(args, "debug", False)
    verbose = getattr(args, "verbose", False) or debug

    lines = []
    for m in result.data:
        if debug:
            lines.append(json.dumps(m, indent=2))
            continue
        owner = m.get("owned_by", "")
        model_id = m.get("id", "")
        prefix = f"[{owner}] " if owner else ""
        if verbose:
            created = m.get("created")
            date_str = ""
            if created:
                dt = datetime.fromtimestamp(created, tz=timezone.utc)
                if dt.year >= 2020:  # suppress obviously synthetic placeholder dates
                    date_str = f"  (added {dt.strftime('%Y-%m-%d')})"
            lines.append(f"{prefix}{model_id}{date_str}")
        else:
            lines.append(f"{prefix}{model_id}")
    return "\n".join(lines)
