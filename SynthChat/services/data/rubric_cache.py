"""Rubric cache - handles ONLY in-memory caching.

Single Responsibility: Manage cached rubrics in memory.
"""

from typing import Dict, Optional, List


class RubricCache:
    """
    In-memory cache for loaded rubrics.

    Responsibility: ONLY memory management (SRP).
    Does NOT handle loading, discovery, or querying.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize rubric cache.

        Args:
            max_size: Maximum number of rubrics to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, Dict] = {}
        self._access_count: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Dict]:
        """
        Get rubric from cache.

        Args:
            key: Rubric key

        Returns:
            Rubric dict or None if not cached
        """
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        return None

    def put(self, key: str, rubric: Dict) -> None:
        """
        Add rubric to cache.

        Args:
            key: Rubric key
            rubric: Rubric dict
        """
        # Evict if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_least_used()

        self._cache[key] = rubric
        self._access_count[key] = 1

    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            key: If provided, clear only this rubric. Otherwise clear all.
        """
        if key:
            self._cache.pop(key, None)
            self._access_count.pop(key, None)
        else:
            self._cache.clear()
            self._access_count.clear()

    def contains(self, key: str) -> bool:
        """
        Check if rubric is cached.

        Args:
            key: Rubric key

        Returns:
            True if cached
        """
        return key in self._cache

    def keys(self) -> List[str]:
        """
        Get list of cached rubric keys.

        Returns:
            List of keys
        """
        return list(self._cache.keys())

    def size(self) -> int:
        """
        Get number of cached rubrics.

        Returns:
            Cache size
        """
        return len(self._cache)

    def _evict_least_used(self) -> None:
        """Evict the least recently used rubric from cache."""
        if not self._cache:
            return

        # Find least used
        least_used_key = min(self._access_count.items(), key=lambda x: x[1])[0]

        # Evict
        del self._cache[least_used_key]
        del self._access_count[least_used_key]

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "cached_keys": list(self._cache.keys()),
            "access_counts": dict(self._access_count),
            "total_accesses": sum(self._access_count.values())
        }
