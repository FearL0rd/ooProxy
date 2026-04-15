"""CLI module: start ooProxy in cascade mode from ~/.ooProxy/cascade.json."""

from __future__ import annotations

import uvicorn

from cli_contract import ModuleSpec, ResultEnvelope, command_result
from modules._server.app import create_app
from modules._server.config import load_cascade_config
from modules.serve import _configure_logging

SPEC = ModuleSpec(
    name="cascade",
    action_flags=("-c", "--cascade"),
    help="Start the proxy using ~/.ooProxy/cascade.json",
    options=(),
    usage_examples=(
        "ooproxy.py --cascade",
    ),
)


def run(args) -> ResultEnvelope:
    debug = getattr(args, "debug", False)
    verbose = getattr(args, "verbose", False) or debug
    uv_log_level = _configure_logging(debug, verbose)
    config = load_cascade_config()
    app = create_app(config)
    try:
        uvicorn.run(app, host=config.host, port=config.port, log_level=uv_log_level)
    except KeyboardInterrupt:
        pass
    return command_result("cascade", None, data=None)


def render_text(result: ResultEnvelope) -> str:
    return ""