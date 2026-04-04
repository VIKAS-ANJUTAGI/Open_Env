"""Easy task: deterministic surface-level correctness bug."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EasyBugTask:
    """Single-file bug scenario with a deterministic expected fix path."""

    name: str = "easy"

    def to_spec(self) -> dict[str, Any]:
        """Return task specification used by state loading and graders."""

        return {
            "max_steps": 8,
            "pr_title": "Fix divide-by-zero guard in ratio utility",
            "pr_description": "Quick hotfix PR. Tests are flaky around edge inputs; please double-check behavior before approval.",
            "diff_by_file": {
                "src/metrics.py": "@@ -1,7 +1,9 @@\n-def ratio(total, count):\n-    return total / count\n+def ratio(total, count):\n+    if count == 0:\n+        return 0\n+    return total / count\n",
            },
            "files_changed": ["src/metrics.py", "tests/test_metrics.py"],
            "repo_snapshot": {
                "src/metrics.py": "def ratio(total, count):\n    return total / count\n",
                "tests/test_metrics.py": "from src.metrics import ratio\n\n\ndef test_zero_count():\n    assert ratio(5, 0) == 0\n",
            },
            "dependency_graph": {"src/metrics.py": ["tests/test_metrics.py"]},
            "ground_truth": {
                "relevant_files": ["src/metrics.py", "tests/test_metrics.py"],
                "critical_files": ["src/metrics.py"],
                "expected_categories": ["correctness"],
                "expected_decision": "REQUEST_CHANGES",
                "expected_patch_files": ["src/metrics.py"],
            },
        }
