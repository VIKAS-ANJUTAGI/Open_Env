"""Deterministic two-layer reward engine for code-review-openenv."""

from __future__ import annotations

from typing import Any

from .models import CodeReviewAction, CodeReviewReward


def _map_grader_score_to_reward(grader_score: float) -> float:
    """Map [0, 1] grader score into normalized reward [-1, 1]."""

    return (2.0 * grader_score) - 1.0


def _compute_step_reward(state: dict[str, Any], action: CodeReviewAction, action_result: dict[str, Any]) -> tuple[float, list[str]]:
    """Compute dense per-step reward aligned with benchmark policy."""

    score = 0.0
    reasons: list[str] = []

    if action.action_type == "READ_FILE":
        if action_result.get("read_relevant"):
            score += 0.05
            reasons.append("read relevant file")
        else:
            score -= 0.02
            reasons.append("read irrelevant file")

    elif action.action_type == "COMMENT":
        if action_result.get("spam_comment"):
            score -= 0.1
            reasons.append("duplicate or spam comment")
        elif action_result.get("meaningful_comment"):
            score += 0.1
            reasons.append("strong structured comment")
        else:
            score += 0.05
            reasons.append("weak comment")

    elif action.action_type == "SUGGEST_FIX":
        if action_result.get("patch_valid"):
            score += 0.15
            reasons.append("valid patch")
        else:
            score -= 0.15
            reasons.append("invalid patch")

    if action_result.get("is_repeated_action"):
        score -= 0.05
        reasons.append("repeated action")

    if action_result.get("invalid_action"):
        score -= 0.1
        reasons.append("invalid action")

    step_count = int(state.get("step_count", 0))
    max_steps = max(int(state.get("max_steps", 1)), 1)
    step_decay = 0.02 * (step_count / max_steps)
    score -= step_decay
    reasons.append(f"step decay={step_decay:.3f}")

    return score, reasons


def compute_reward(state: dict[str, Any], action: CodeReviewAction, action_result: dict[str, Any]) -> CodeReviewReward:
    """Compute step-level and final blended reward.

    Final reward formula:
        (0.3 * step_rewards_sum) + (0.7 * (2 * grader_score - 1))
    """

    step_score, reasons = _compute_step_reward(state, action, action_result)

    if action_result.get("finalized"):
        grader_score = float(action_result.get("grader_score", 0.0))
        mapped = _map_grader_score_to_reward(grader_score)
        step_rewards_sum = float(action_result.get("step_rewards_sum", 0.0))
        final_score = (0.3 * step_rewards_sum) + (0.7 * mapped)
        final_score = max(-1.0, min(1.0, final_score))
        reasons.append(f"grader_score={grader_score:.3f}")
        reasons.append(f"step_rewards_sum={step_rewards_sum:.3f}")
        reasons.append("blended final reward")
        return CodeReviewReward(score=final_score, reason="; ".join(reasons))

    step_score = max(-1.0, min(1.0, step_score))
    return CodeReviewReward(score=step_score, reason="; ".join(reasons))
