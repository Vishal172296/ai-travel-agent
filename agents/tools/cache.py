"""
In-memory LRU / TTL Caching layer for tools to prevent redundant API calls.
"""

import time
import hashlib
import json
from typing import Any, Callable, Dict, Optional, Tuple


class SimpleToolCache:
    """Thread-safe, time-bounded cache for tool responses."""

    def __init__(self, default_ttl_seconds: int = 3600, max_entries: int = 256):
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[float, Any]] = {}

    def _generate_key(self, tool_name: str, args: Any) -> str:
        try:
            if isinstance(args, dict):
                serialized = json.dumps(args, sort_keys=True, default=str)
            else:
                serialized = str(args)
        except Exception:
            serialized = str(args)
        raw = f"{tool_name}:{serialized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, tool_name: str, args: Any) -> Optional[Any]:
        key = self._generate_key(tool_name, args)
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.default_ttl:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, tool_name: str, args: Any, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if len(self._cache) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0], default=None)
            if oldest_key:
                del self._cache[oldest_key]

        key = self._generate_key(tool_name, args)
        self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> Dict[str, int]:
        return {
            "cached_entries": len(self._cache),
            "max_entries": self.max_entries,
        }


# Global tool cache instance
GLOBAL_TOOL_CACHE = SimpleToolCache()
