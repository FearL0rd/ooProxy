"""Persistent cache of per-model behavioral quirks discovered via trial-and-error.

When ooProxy learns that a specific model at a specific endpoint needs special
handling (e.g. strip stream_options, strip tools, normalize messages), it saves
that knowledge to ~/.ooProxy/behavior.json so future sessions skip the retries.

Schema of behavior.json:
    {
      "<base_url>|<model>": {
        "strip_stream_options": true,
        "strip_tools": true,
        "normalize_messages": true
      },
      ...
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("ooproxy")

_CACHE_PATH = Path.home() / ".ooProxy" / "behavior.json"

# All boolean flags that can be recorded for a model.
KNOWN_FLAGS = frozenset({"strip_stream_options", "strip_tools", "normalize_messages"})


class BehaviorCache:
    """In-memory + on-disk cache of model quirks.

    Thread-safe for concurrent async callers via asyncio.Lock.

    Pass ``path=None`` to disable file I/O entirely (useful in tests).
    """

    def __init__(self, path: Path | None = _CACHE_PATH) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._data: dict[str, dict[str, bool]] = {}
        if path is not None:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def key(self, base_url: str, model: str) -> str:
        return f"{base_url.rstrip('/')}|{model}"

    def get_flags(self, base_url: str, model: str) -> dict[str, bool]:
        """Return the cached quirk flags for a model (all False if unknown)."""
        return dict(self._data.get(self.key(base_url, model), {}))

    async def record(self, base_url: str, model: str, flag: str, *, value: bool = True) -> None:
        """Record a newly discovered quirk and persist immediately."""
        if flag not in KNOWN_FLAGS:
            raise ValueError(f"Unknown behavior flag: {flag!r}")
        k = self.key(base_url, model)
        async with self._lock:
            entry = self._data.setdefault(k, {})
            if entry.get(flag) == value:
                return  # already known, nothing to do
            entry[flag] = value
            logger.info(
                "behavior: discovered %s=%s for %s — saving to %s",
                flag, value, k, self._path,
            )
            self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self._data = {
                    k: {f: bool(v) for f, v in flags.items() if f in KNOWN_FLAGS}
                    for k, flags in data.items()
                    if isinstance(flags, dict)
                }
                logger.debug("behavior: loaded %d entries from %s", len(self._data), self._path)
        except Exception as exc:
            logger.warning("behavior: could not load %s: %s", self._path, exc)

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("behavior: could not save %s: %s", self._path, exc)
