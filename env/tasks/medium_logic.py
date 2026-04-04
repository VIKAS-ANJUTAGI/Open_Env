"""Medium task: deterministic logic flaw requiring semantic reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MediumLogicTask:
    """Logic bug scenario where incorrect condition creates regression risk."""

    name: str = "medium"

    def to_spec(self) -> dict[str, Any]:
        """Return task specification used by state loading and graders."""

        return {
            "max_steps": 10,
            "pr_title": "Review cache expiration fix before merge",
            "pr_description": "The patch claims to fix stale cache reads, but QA still sees edge-case failures at ttl boundaries.",
            "diff_by_file": {
                "src/cache.py": "@@ -10,9 +10,10 @@\n-    if entry and entry[1] > now:\n+    if entry and entry[1] >= now:\n         return entry[0]\n",
            },
            "files_changed": ["src/cache.py", "src/service.py", "tests/test_cache.py"],
            "repo_snapshot": {
                "src/cache.py": "def get_cached(store, key, now):\n    entry = store.get(key)\n    if entry and entry[1] > now:\n        return entry[0]\n    return None\n",
                "src/service.py": "def update_user(store, key, value, ttl, now):\n    store[key] = (value, now + ttl)\n",
                "tests/test_cache.py": "from src.cache import get_cached\n\n\ndef test_expired_item():\n    store = {'k': ('v', 10)}\n    assert get_cached(store, 'k', 10) is None\n",
            },
            "dependency_graph": {"src/cache.py": ["src/service.py", "tests/test_cache.py"]},
            "ground_truth": {
                "relevant_files": ["src/cache.py", "tests/test_cache.py"],
                "critical_files": ["src/cache.py"],
                "expected_categories": ["logic", "regression"],
                "expected_decision": "REQUEST_CHANGES",
                "expected_patch_files": ["src/cache.py"],
            },
        }
