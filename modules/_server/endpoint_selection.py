"""Resolve command-line upstream URLs from stored endpoint profiles."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from cli_contract import CommandError
from modules._server.endpoint_profiles import load_endpoint_profiles
from modules._server.key_store import ApiKeyStore


@dataclass(frozen=True)
class EndpointChoice:
    url: str
    profile_id: str


def _default_port_for_scheme(scheme: str) -> int | None:
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    return None


def _format_base_url(scheme: str, host: str, port: int | None, path_prefix: str) -> str:
    default_port = _default_port_for_scheme(scheme)
    netloc = host if port in (None, default_port) else f"{host}:{port}"
    path = str(path_prefix or "/").strip() or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    path = path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _has_stored_key(stored_endpoints: set[str], host: str, port: int | None, default_port: int | None) -> bool:
    if port is None:
        return host in stored_endpoints
    endpoint_with_port = f"{host}:{port}"
    if default_port is not None and port == default_port:
        return host in stored_endpoints or endpoint_with_port in stored_endpoints
    return endpoint_with_port in stored_endpoints


def available_endpoint_choices() -> list[EndpointChoice]:
    stored_endpoints = set(ApiKeyStore().hosts())
    if not stored_endpoints:
        return []

    choices: list[EndpointChoice] = []
    seen_urls: set[str] = set()
    for profile in load_endpoint_profiles():
        if not profile.host_equals:
            continue
        path_prefix = profile.path_prefixes[0] if profile.path_prefixes else "/v1"
        schemes = profile.schemes or ("https",)
        for scheme in schemes:
            default_port = _default_port_for_scheme(scheme)
            ports = profile.ports or ((default_port,) if default_port is not None else (None,))
            for host in profile.host_equals:
                for port in ports:
                    if not _has_stored_key(stored_endpoints, host, port, default_port):
                        continue
                    url = _format_base_url(scheme, host, port, path_prefix)
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    choices.append(EndpointChoice(url=url, profile_id=profile.id))
    return choices


def format_endpoint_choices(choices: list[EndpointChoice]) -> str:
    lines = ["No --url provided. Available stored endpoint profiles:"]
    for index, choice in enumerate(choices, start=1):
        lines.append(f"{index}. {choice.url} [{choice.profile_id}]")
    return "\n".join(lines)


def resolve_profile_url(args) -> str:
    explicit_url = str(getattr(args, "url", "") or "").strip()
    if explicit_url:
        return explicit_url

    choices = available_endpoint_choices()
    if not choices:
        raise CommandError(
            "No stored endpoint profiles matched keys in ~/.ooProxy/keys.json. Provide --url URL.",
            show_usage=True,
            exit_code=2,
        )

    if len(choices) == 1:
        print(format_endpoint_choices(choices))
        return choices[0].url

    if not sys.stdin.isatty():
        raise CommandError(
            f"{format_endpoint_choices(choices)}\nRe-run with --url URL.",
            show_usage=True,
            exit_code=2,
        )

    print(format_endpoint_choices(choices))
    while True:
        try:
            raw = input("Endpoint number: ").strip()
        except EOFError as exc:
            raise CommandError("No endpoint selected. Re-run with --url URL.", show_usage=True, exit_code=2) from exc
        if not raw:
            raise CommandError("No endpoint selected. Re-run with --url URL.", show_usage=True, exit_code=2)
        try:
            selected_index = int(raw)
        except ValueError:
            print(f"Enter a number from 1 to {len(choices)}.", file=sys.stderr)
            continue
        if 1 <= selected_index <= len(choices):
            return choices[selected_index - 1].url
        print(f"Enter a number from 1 to {len(choices)}.", file=sys.stderr)