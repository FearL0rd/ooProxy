"""CLI module: start the Ollama-compatible proxy server."""

from __future__ import annotations

import logging
import uvicorn

from cli_contract import CommandError, ModuleSpec, OptionSpec, ResultEnvelope, command_result
from modules._server.app import create_app
from modules._server.config import ProxyConfig

SPEC = ModuleSpec(
    name="serve",
    action_flags=("-s", "--serve"),
    help="Start the Ollama-compatible proxy server",
    options=(
        OptionSpec(
            flags=("--url",),
            dest="url",
            help="Remote OpenAI-compatible base URL (env: OPENAI_BASE_URL)",
            metavar="URL",
        ),
        OptionSpec(
            flags=("--key",),
            dest="key",
            help="Remote API key (env: OPENAI_API_KEY)",
            metavar="KEY",
        ),
        OptionSpec(
            flags=("--port",),
            dest="port",
            help="Local port to listen on (default: 11434)",
            metavar="PORT",
            default=11434,
        ),
    ),
    usage_examples=(
        "ooproxy.py -s --url https://integrate.api.nvidia.com/v1 --key nvapi-xxx",
        "OPENAI_BASE_URL=https://... OPENAI_API_KEY=sk-... ooproxy.py -s",
    ),
)


def _configure_logging(debug: bool, verbose: bool) -> str:
    """Set up logging levels and return the uvicorn log_level string."""
    fmt = "%(levelname)s %(name)s: %(message)s"
    if debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, force=True)
        return "debug"
    if verbose:
        logging.basicConfig(level=logging.INFO, format=fmt, force=True)
        # Keep httpcore quiet even in verbose mode — it's very noisy
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        return "info"
    # Default: show ooproxy INFO, suppress third-party chatter
    logging.basicConfig(level=logging.WARNING, format=fmt, force=True)
    logging.getLogger("ooproxy").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return "info"


def run(args) -> ResultEnvelope:
    debug = getattr(args, "debug", False)
    verbose = getattr(args, "verbose", False) or debug
    uv_log_level = _configure_logging(debug, verbose)
    config = ProxyConfig.from_args(args)
    app = create_app(config)
    try:
        uvicorn.run(app, host="127.0.0.1", port=config.port, log_level=uv_log_level)
    except KeyboardInterrupt:
        pass
    return command_result("serve", None, data=None)


def render_text(result: ResultEnvelope) -> str:
    return ""
