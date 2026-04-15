"""Manage weakly obfuscated API keys stored in ~/.ooProxy/keys.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules._server.key_store import ApiKeyStore, normalize_endpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage API keys stored in ~/.ooProxy/keys.json.")
    parser.add_argument("-H", "--host", help="Endpoint host or host:port used as the key-store index")
    parser.add_argument("--key", help="API key to store for the endpoint")
    parser.add_argument("--delete", action="store_true", help="Delete the stored key for the endpoint")
    parser.add_argument("-j", "--json", dest="json_output", action="store_true", help="Return JSON output")
    return parser


def _emit(payload: object, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, list):
        if payload:
            print("\n".join(payload))
        return
    if isinstance(payload, str) and payload:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.key and not args.host:
        parser.error("--key requires --host")
    if args.delete and not args.host:
        parser.error("--delete requires --host")
    if args.delete and args.key:
        parser.error("cannot combine --delete and --key")

    store = ApiKeyStore()

    if args.host and args.key:
        endpoint = store.set(args.host, args.key)
        _emit({"host": endpoint, "status": "stored"}, json_output=args.json_output)
        if not args.json_output:
            print(f"stored key for {endpoint}")
        return 0

    if args.delete:
        endpoint = normalize_endpoint(args.host)
        removed = store.delete(endpoint)
        if args.json_output:
            _emit({"host": endpoint, "status": "deleted" if removed else "missing"}, json_output=True)
        elif removed:
            print(f"deleted key for {endpoint}")
        else:
            print(f"no stored key for {endpoint}", file=sys.stderr)
        return 0 if removed else 1

    if args.host:
        endpoint = normalize_endpoint(args.host)
        key = store.get(endpoint)
        if key is None:
            print(f"no stored key for {endpoint}", file=sys.stderr)
            return 1
        if args.json_output:
            _emit({"host": endpoint, "key": key}, json_output=True)
        else:
            print(key)
        return 0

    hosts = store.hosts()
    if args.json_output:
        _emit({"hosts": hosts}, json_output=True)
    else:
        _emit(hosts, json_output=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())