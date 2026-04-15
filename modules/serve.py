"""CLI module: start the Ollama-compatible proxy server."""

from __future__ import annotations

import logging

import uvicorn

from cli_contract import ModuleSpec, OptionSpec, ResultEnvelope, command_result
from modules._server.app import create_app
from modules._server.config import ProxyConfig
from modules._server.endpoint_selection import resolve_profile_url

SPEC = ModuleSpec(
    name="serve",
    action_flags=("-s", "--serve"),
    help="Start the Ollama-compatible proxy server",
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
        OptionSpec(
            flags=("-H", "--host"),
            dest="host",
            help="Local IP address to listen on (default: 127.0.0.1)",
            metavar="HOST",
            default="127.0.0.1",
        ),
        OptionSpec(
            flags=("-P", "--port"),
            dest="port",
            help="Local port to listen on (default: 11434)",
            metavar="PORT",
            default=11434,
        ),
    ),
    usage_examples=(
        "ooproxy.py -s --url https://integrate.api.nvidia.com/v1 --key nvapi-xxx",
        "ooproxy.py -s --url http://myserver:8080/v1 --key sk-xxx",
        "ooproxy.py -s  # select from keyed endpoint profiles",
        "ooproxy.py -s --host 0.0.0.0 --port 11434",
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
    args.url = resolve_profile_url(args)
    config = ProxyConfig.from_args(args)
    app = create_app(config)
    host = getattr(args, "host", "127.0.0.1")
    try:
        uvicorn.run(app, host=host, port=config.port, log_level=uv_log_level)
    except KeyboardInterrupt:
        pass
    return command_result("serve", None, data=None)


def render_text(result: ResultEnvelope) -> str:
    return ""
