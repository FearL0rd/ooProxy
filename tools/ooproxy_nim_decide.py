"""Send a cascade decision probe directly to NVIDIA NIM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules._server.config import CascadeDecisionConfig, _DEFAULT_URL, load_cascade_config, render_cascade_decision_prompt
from modules._server.key_store import ApiKeyStore, endpoint_from_url
from ooproxy_version import cli_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a cascade decision request directly to NVIDIA NIM and print the raw response."
    )
    parser.add_argument("--version", action="version", version=cli_version("ooproxy_nim_decide"))
    parser.add_argument("model", help="Model name to test, for example openai/gpt-oss-20b")
    parser.add_argument("prompt", help="User prompt to evaluate for weak-vs-strong routing")
    parser.add_argument(
        "--url",
        default="",
        help="Base URL for the OpenAI-compatible endpoint. Defaults to the URL from ~/.ooProxy/cascade.json or NVIDIA NIM.",
    )
    parser.add_argument(
        "--key",
        default="",
        help="API key to use. Defaults to the key from ~/.ooProxy/cascade.json or ~/.ooProxy/keys.json.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override. By default the request omits max_tokens entirely.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout override in seconds. Defaults to the cascade config decision.timeout_seconds value.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold to print alongside the parsed confidence. Defaults to the cascade config threshold.",
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        metavar="NAME[:DESCRIPTION]",
        help="Available tool to include in the decision prompt. Can be repeated.",
    )
    parser.add_argument(
        "--tool-choice",
        default="none",
        help="Tool choice summary string to include in the decision prompt. Default: none",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Override the decision system prompt.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print a compact JSON summary instead of formatted text.",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Print only the parsed confidence value.",
    )
    return parser


def _tool_summary(tool_specs: list[str]) -> str:
    if not tool_specs:
        return "none"
    items: list[str] = []
    for spec in tool_specs:
        name, separator, description = spec.partition(":")
        name = name.strip()
        description = description.strip()
        if not name:
            continue
        items.append(f"{name}: {description}" if separator and description else name)
    return "; ".join(items) if items else "none"


def _resolve_defaults(args) -> tuple[str, str, CascadeDecisionConfig]:
    try:
        config = load_cascade_config()
        route = config.cascade.routes[0]
        url = args.url or route.weak_url
        key = args.key or route.weak_key
        decision = config.cascade.decision
        return url, key, decision
    except Exception:
        url = args.url or _DEFAULT_URL
        key = args.key or ApiKeyStore().get(endpoint_from_url(url)) or ""
        return url, key, CascadeDecisionConfig()


def _decision_messages(
    *,
    model: str,
    prompt: str,
    decision: CascadeDecisionConfig,
    tool_summary: str,
    tool_choice: str,
) -> list[dict[str, str]]:
    request_json = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}, ensure_ascii=False, sort_keys=True)
    user_prompt = render_cascade_decision_prompt(
        decision.user_prompt_template,
        user_prompt=prompt,
        weak_model=model,
        strong_model=f"stronger-than-{model}",
        available_tools=tool_summary,
        tool_choice=tool_choice,
        request_json=request_json,
    )
    return [
        {"role": "system", "content": decision.system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    for key in ("content", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("reasoning", "reasoning_content"):
        value = choice.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_confidence(payload: dict) -> float | None:
    text = _extract_text(payload)
    if not text:
        return None
    candidate = text
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{"):candidate.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(parsed, dict) or "CONFIDENCE" not in parsed:
        return None
    try:
        confidence = float(parsed["CONFIDENCE"])
    except (TypeError, ValueError):
        return None
    return confidence if 0.0 <= confidence <= 1.0 else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parse = parser.parse_intermixed_args if hasattr(parser, "parse_intermixed_args") else parser.parse_args
    args = parse(argv)

    url, key, decision = _resolve_defaults(args)
    if not key:
        parser.error("No API key found. Pass --key or store one in ~/.ooProxy/keys.json")

    tool_summary = _tool_summary(args.tool)
    tool_choice = args.tool_choice
    if args.system_prompt:
        decision = CascadeDecisionConfig(
            threshold=decision.threshold,
            max_tokens=decision.max_tokens,
            timeout_seconds=decision.timeout_seconds,
            system_prompt=args.system_prompt,
            user_prompt_template=decision.user_prompt_template,
            retry_user_prompt_template=decision.retry_user_prompt_template,
        )
    threshold = decision.threshold if args.threshold is None else float(args.threshold)
    timeout_seconds = decision.timeout_seconds if args.timeout is None else float(args.timeout)

    request_body = {
        "model": args.model,
        "messages": _decision_messages(
            model=args.model,
            prompt=args.prompt,
            decision=decision,
            tool_summary=tool_summary,
            tool_choice=tool_choice,
        ),
        "stream": False,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if args.max_tokens is not None:
        request_body["max_tokens"] = int(args.max_tokens)

    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=timeout_seconds, write=30.0, pool=10.0)) as client:
        response = client.post(
            f"{url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=request_body,
        )
    payload = response.json()
    confidence = _extract_confidence(payload)

    if args.short:
        print("unparsed" if confidence is None else f"{confidence:.2f}")
        return 0

    if args.json_output:
        print(
            json.dumps(
                {
                    "url": url,
                    "model": args.model,
                    "status": response.status_code,
                    "threshold": threshold,
                    "parsed_confidence": confidence,
                    "response": payload,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    print(f"URL: {url}")
    print(f"Model: {args.model}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Status: {response.status_code}")
    print(f"Max tokens: {request_body.get('max_tokens', 'omitted')}")
    print(f"Parsed confidence: {'unparsed' if confidence is None else f'{confidence:.2f}'}")
    print("Decision request:")
    print(json.dumps(request_body, indent=2, ensure_ascii=False))
    print("Response:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())