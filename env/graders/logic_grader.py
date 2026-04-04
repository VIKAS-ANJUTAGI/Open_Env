"""Deterministic grader for medium logic bug task."""

from __future__ import annotations

from typing import Any

from .base_grader import BaseGrader


class LogicGrader(BaseGrader):
    """Score medium task with logic flaw, reasoning quality, and fix validity."""

    def evaluate(self, state: dict[str, Any]) -> float:
        score, _ = self.evaluate_with_breakdown(state)
        return score

    def evaluate_with_breakdown(self, state: dict[str, Any]) -> tuple[float, dict[str, float]]:
        expected_categories = self._expected_categories(state)
        relevant = self._relevant_files(state)
        critical = self._critical_files(state)
        expected_patch_files = self._expected_patch_files(state)

        opened = self._opened_files(state)
        comments = state.get("comments", [])
        commented_categories = self._commented_categories(state)
        patched = self._patched_files(state)

        correct_logic = bool(commented_categories & expected_categories)
        logic_score = 0.4 if correct_logic else 0.0

        relevant_comments = [c for c in comments if c.get("file_path") in relevant]
        strong_reasoning = bool(relevant_comments) and len(opened & relevant) >= 2
        weak_reasoning = bool(relevant_comments)
        if strong_reasoning:
            reasoning_score = 0.3
        elif weak_reasoning:
            reasoning_score = 0.15
        else:
            reasoning_score = 0.0

        fix_overlap = len(patched & expected_patch_files)
        if expected_patch_files:
            fix_score = 0.3 * (fix_overlap / len(expected_patch_files))
        else:
            fix_score = 0.0

        if critical and not (opened & critical):
            fix_score = min(fix_score, 0.15)

        base_score = logic_score + reasoning_score + fix_score
        final_score = self._apply_common_penalties(base_score, state)
        penalty_total = max(0.0, base_score - final_score)
        breakdown = {
            "logic_score": logic_score,
            "reasoning_score": reasoning_score,
            "fix_score": fix_score,
            "penalty_total": penalty_total,
            "final_score": final_score,
        }
        return final_score, breakdown
