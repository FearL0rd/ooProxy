"""CLI module: start the Ollama-compatible proxy server."""

from __future__ import annotations

import logging
import sys

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


class _ColorFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }

    def __init__(self, fmt: str, *, use_color: bool) -> None:
        super().__init__(fmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self._use_color:
            color = self._COLORS.get(record.levelno)
            if color:
                record.levelname = f"{color}{original_levelname}{self._RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _configure_root_logging(level: int) -> None:
    handler = logging.StreamHandler()
    use_color = bool(getattr(handler.stream, "isatty", lambda: False)())
    handler.setFormatter(_ColorFormatter("%(levelname)s %(name)s: %(message)s", use_color=use_color))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def _configure_logging(debug: bool, verbose: bool) -> str:
    """Set up logging levels and return the uvicorn log_level string."""
    if debug:
        _configure_root_logging(logging.DEBUG)
        return "debug"
    if verbose:
        _configure_root_logging(logging.INFO)
        # Keep httpcore quiet even in verbose mode — it's very noisy
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        return "info"
    # Default: show ooproxy INFO, suppress third-party chatter
    _configure_root_logging(logging.WARNING)
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
