"""
General-purpose in-memory cache for ooProxy server, supporting per-endpoint, per-request-type caching with TTL.
"""

import time
import threading
from typing import Any, Optional

class RequestCache:
    def __init__(self):
        self._lock = threading.Lock()
        # Structure: {(endpoint_id, request_type): (data, expires_at)}
        self._cache: dict[tuple[str, str], tuple[Any, float]] = {}

    def get(self, endpoint_id: str, request_type: str) -> Optional[Any]:
        key = (endpoint_id, request_type)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                data, expires_at = entry
                if time.time() < expires_at:
                    return data
                else:
                    del self._cache[key]
        return None

    def set(self, endpoint_id: str, request_type: str, data: Any, ttl: int):
        key = (endpoint_id, request_type)
        expires_at = time.time() + ttl
        with self._lock:
            self._cache[key] = (data, expires_at)

    def clear(self):
        with self._lock:
            self._cache.clear()
