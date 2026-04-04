"""Base deterministic grader utilities for code-review-openenv."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGrader(ABC):
    """Shared scorer utilities and anti-exploit penalties."""

    @abstractmethod
    def evaluate(self, state: dict[str, Any]) -> float:
        """Evaluate task-specific performance and return score in [0, 1]."""

    @abstractmethod
    def evaluate_with_breakdown(self, state: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Evaluate and return both final score and deterministic sub-score breakdown."""

    def _relevant_files(self, state: dict[str, Any]) -> set[str]:
        return set(state.get("ground_truth", {}).get("relevant_files", []))

    def _critical_files(self, state: dict[str, Any]) -> set[str]:
        return set(state.get("ground_truth", {}).get("critical_files", []))

    def _expected_categories(self, state: dict[str, Any]) -> set[str]:
        return set(state.get("ground_truth", {}).get("expected_categories", []))

    def _expected_patch_files(self, state: dict[str, Any]) -> set[str]:
        return set(state.get("ground_truth", {}).get("expected_patch_files", []))

    def _opened_files(self, state: dict[str, Any]) -> set[str]:
        return set(state.get("opened_files", []))

    def _commented_categories(self, state: dict[str, Any]) -> set[str]:
        return {c.get("category") for c in state.get("comments", []) if c.get("category")}

    def _comments_on_files(self, state: dict[str, Any], targets: set[str]) -> list[dict[str, Any]]:
        return [c for c in state.get("comments", []) if c.get("file_path") in targets]

    def _patched_files(self, state: dict[str, Any]) -> set[str]:
        files: set[str] = set()
        for patch_entry in state.get("suggested_patches", []):
            for target in patch_entry.get("targets", []):
                files.add(target)
        return files

    def _penalty_false_positives(self, state: dict[str, Any]) -> float:
        """Penalty when comments claim categories not expected for this task."""

        expected = self._expected_categories(state)
        commented = self._commented_categories(state)
        false_positive_count = len([cat for cat in commented if cat not in expected])
        return 0.2 if false_positive_count > 0 else 0.0

    def _penalty_irrelevant_comments(self, state: dict[str, Any]) -> float:
        """Penalty when comments target irrelevant files."""

        relevant = self._relevant_files(state)
        irrelevant_count = len([c for c in state.get("comments", []) if c.get("file_path") not in relevant])
        return min(0.2, irrelevant_count * 0.1)

    def _penalty_excessive_actions(self, state: dict[str, Any]) -> float:
        """Penalty when trajectory overuses action budget."""

        steps = int(state.get("step_count", 0))
        max_steps = max(int(state.get("max_steps", 1)), 1)
        threshold = int(max_steps * 0.8)
        if steps <= threshold:
            return 0.0
        over_ratio = (steps - threshold) / max(max_steps - threshold, 1)
        return min(0.2, 0.2 * over_ratio)

    def _decision_score(self, state: dict[str, Any]) -> float:
        history = state.get("history", [])
        expected_decision = state.get("ground_truth", {}).get("expected_decision")
        for entry in reversed(history):
            if "\"action\":\"FINISH\"" in entry:
                if expected_decision == "REQUEST_CHANGES":
                    return 1.0
                return 0.0
            if "finish:REQUEST_CHANGES" in entry and expected_decision == "REQUEST_CHANGES":
                return 1.0
        return 0.0

    def _clamp(self, score: float) -> float:
        return max(0.0, min(1.0, score))

    def _apply_common_penalties(self, score: float, state: dict[str, Any]) -> float:
        penalty = 0.0
        penalty += self._penalty_false_positives(state)
        penalty += self._penalty_irrelevant_comments(state)
        penalty += self._penalty_excessive_actions(state)
        return self._clamp(score - penalty)
