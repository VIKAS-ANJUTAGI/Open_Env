"""Deterministic grader for easy surface-bug task."""

from __future__ import annotations

from typing import Any

from .base_grader import BaseGrader


class BugGrader(BaseGrader):
    """Score easy task with weighted correctness, issue, and fix signals."""

    def evaluate(self, state: dict[str, Any]) -> float:
        score, _ = self.evaluate_with_breakdown(state)
        return score

    def evaluate_with_breakdown(self, state: dict[str, Any]) -> tuple[float, dict[str, float]]:
        critical = self._critical_files(state)
        expected_categories = self._expected_categories(state)
        expected_patch_files = self._expected_patch_files(state)

        opened = self._opened_files(state)
        commented_categories = self._commented_categories(state)
        patched = self._patched_files(state)

        file_score = 0.4 if critical & opened else 0.0

        has_correct_issue = bool(commented_categories & expected_categories)
        has_weak_issue = len(self._comments_on_files(state, self._relevant_files(state))) > 0
        if has_correct_issue:
            issue_score = 0.3
        elif has_weak_issue:
            issue_score = 0.15
        else:
            issue_score = 0.0

        fix_overlap = len(patched & expected_patch_files)
        if expected_patch_files:
            fix_score = 0.3 * (fix_overlap / len(expected_patch_files))
        else:
            fix_score = 0.0

        base_score = file_score + issue_score + fix_score
        final_score = self._apply_common_penalties(base_score, state)
        penalty_total = max(0.0, base_score - final_score)
        breakdown = {
            "file_score": file_score,
            "issue_score": issue_score,
            "fix_score": fix_score,
            "penalty_total": penalty_total,
            "final_score": final_score,
        }
        return final_score, breakdown
