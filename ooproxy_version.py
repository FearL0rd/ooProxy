"""Project version constants for ooProxy."""

from __future__ import annotations

OO_PROXY_VERSION = "1.0.3"
OO_PROXY_VERSION_TAG = f"v{OO_PROXY_VERSION}"

# The compatibility version of the upstream product we emulate. Use the
# original product-specific name so callers explicitly reference the emulated
# upstream product version.
OLLAMA_COMPAT_VERSION = "0.6.5"


def cli_version(program_name: str) -> str:
    return f"{program_name} {OO_PROXY_VERSION_TAG}"
