"""Hard task: deterministic cross-file security and regression bug."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardCrossFileTask:
    """Cross-file scenario where root cause sits away from failure symptom."""

    name: str = "hard"

    def to_spec(self) -> dict[str, Any]:
        """Return task specification used by state loading and graders."""

        return {
            "max_steps": 12,
            "pr_title": "Audit auth path for token leakage and validation correctness",
            "pr_description": "A refactor touched auth and logging modules. Incident report mentions intermittent token exposure in logs.",
            "diff_by_file": {
                "src/auth.py": "@@ -3,8 +3,9 @@\n-    return token == expected\n+    return token.strip() == expected.strip()\n",
                "src/logger.py": "@@ -1,4 +1,4 @@\n-def log_login(user, token):\n-    print(f'user={user} token={token}')\n+def log_login(user, token):\n+    print(f'user={user} token=<redacted>')\n",
            },
            "files_changed": ["src/auth.py", "src/logger.py", "src/api.py", "tests/test_auth.py"],
            "repo_snapshot": {
                "src/auth.py": "def is_valid_token(token, expected):\n    return token == expected\n",
                "src/logger.py": "def log_login(user, token):\n    print(f'user={user} token={token}')\n",
                "src/api.py": "from src.auth import is_valid_token\nfrom src.logger import log_login\n\n\ndef login(user, token, expected):\n    log_login(user, token)\n    return is_valid_token(token, expected)\n",
                "tests/test_auth.py": "from src.api import login\n\n\ndef test_login_hides_token(capsys):\n    login('u', 'secret', 'secret')\n    assert 'secret' not in capsys.readouterr().out\n",
            },
            "dependency_graph": {
                "src/auth.py": ["src/api.py", "tests/test_auth.py"],
                "src/logger.py": ["src/api.py", "tests/test_auth.py"],
            },
            "ground_truth": {
                "relevant_files": ["src/auth.py", "src/logger.py", "src/api.py", "tests/test_auth.py"],
                "critical_files": ["src/auth.py", "src/logger.py"],
                "expected_categories": ["security", "performance"],
                "expected_decision": "REQUEST_CHANGES",
                "expected_patch_files": ["src/auth.py", "src/logger.py"],
            },
        }
