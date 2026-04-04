"""Deterministic grader for hard cross-file review task."""

from __future__ import annotations

from typing import Any

from .base_grader import BaseGrader


class CrossFileGrader(BaseGrader):
    """Score hard task with exploration breadth, root-cause quality, and fix validity."""

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

        if critical:
            exploration_score = 0.3 * (len(opened & critical) / len(critical))
        else:
            exploration_score = 0.0

        root_categories_found = len(commented_categories & expected_categories)
        if expected_categories:
            root_cause_score = 0.3 * (root_categories_found / len(expected_categories))
        else:
            root_cause_score = 0.0

        fix_overlap = len(patched & expected_patch_files)
        if expected_patch_files:
            fix_score = 0.4 * (fix_overlap / len(expected_patch_files))
        else:
            fix_score = 0.0

        base_score = exploration_score + root_cause_score + fix_score
        final_score = self._apply_common_penalties(base_score, state)
        penalty_total = max(0.0, base_score - final_score)
        breakdown = {
            "exploration_score": exploration_score,
            "root_cause_score": root_cause_score,
            "fix_score": fix_score,
            "penalty_total": penalty_total,
            "final_score": final_score,
        }
        return final_score, breakdown
