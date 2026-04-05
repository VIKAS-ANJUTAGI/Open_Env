"""Core execution engine for deterministic code-review environment."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .graders.bug_grader import BugGrader
from .graders.cross_file_grader import CrossFileGrader
from .graders.logic_grader import LogicGrader
from .models import CodeReviewAction, CodeReviewObservation
from .reward import compute_reward
from .state_manager import StateManager


class CodeReviewEnv:
    """Production-grade OpenEnv loop for context-aware code review tasks."""

    def __init__(self, task_id: str = "easy", max_steps: int = 10) -> None:
        self.state_manager = StateManager(max_steps=max_steps)
        self.task_id = task_id
        self._step_rewards_sum = 0.0
        self._graders = {
            "easy": BugGrader(),
            "medium": LogicGrader(),
            "hard": CrossFileGrader(),
        }

    def _build_observation(self) -> CodeReviewObservation:
        """Construct agent-facing observation without leaking full repository."""

        state = self.state_manager.current_state()
        return CodeReviewObservation(
            pr_title=state["pr_title"],
            pr_description=state["pr_description"],
            pr_diff=state["pr_diff"],
            diff_by_file=state["diff_by_file"],
            files_changed=state["files_changed"],
            file_contents=state["file_contents"],
            current_file=state["current_file"],
            history=state["history"],
            step_count=state["step_count"],
            max_steps=state["max_steps"],
            remaining_steps=state["remaining_steps"],
            last_action=state["last_action"],
            last_action_status=state["last_action_status"],
            last_error=state["last_error"],
        )

    def reset(self, task_id: str | None = None) -> CodeReviewObservation:
        """Reset episode state and load requested deterministic task."""

        selected_task = task_id or self.task_id
        self.state_manager.reset(selected_task)
        self._step_rewards_sum = 0.0
        return self._build_observation()

    def _evaluate_final_score(self) -> tuple[float, dict[str, float]]:
        """Run deterministic task grader and return score with score breakdown."""

        active_task = self.state_manager.current_state().get("active_task", "easy")
        grader = self._graders.get(active_task, self._graders["easy"])
        return grader.evaluate_with_breakdown(self.state_manager.current_state())

    def _validate_action(self, action: CodeReviewAction | dict[str, Any]) -> tuple[CodeReviewAction | None, str | None]:
        """Validate action via strict Pydantic schema."""

        try:
            if isinstance(action, CodeReviewAction):
                return action, None
            return CodeReviewAction.model_validate(action), None
        except ValidationError as exc:
            return None, str(exc)

    def step(self, action: CodeReviewAction | dict[str, Any]) -> tuple[CodeReviewObservation, float, bool, dict[str, Any]]:
        """Execute one deterministic transition with strict action semantics."""

        if self.state_manager.episode_done:
            observation = self._build_observation()
            return observation, 0.0, True, {"error": "episode already completed"}

        validated_action, validation_error = self._validate_action(action)
        action_result: dict[str, Any] = {
            "invalid_action": False,
            "is_repeated_action": False,
            "read_relevant": False,
            "meaningful_comment": False,
            "spam_comment": False,
            "patch_valid": False,
            "finalized": False,
            "grader_score": 0.0,
        }

        if validation_error is not None or validated_action is None:
            fallback_action = CodeReviewAction(action_type="FINISH", finish_decision="REQUEST_CHANGES")
            action_result["invalid_action"] = True
            self.state_manager.set_last_action({"action_type": "INVALID"})
            self.state_manager.set_action_result("error", validation_error)
            self.state_manager.increment_step()

            step_reward = compute_reward(self.state_manager.current_state(), fallback_action, action_result)
            self._step_rewards_sum += step_reward.score

            if self.state_manager.remaining_steps() <= 0:
                self.state_manager.episode_done = True
                action_result["finalized"] = True
                grader_score, score_breakdown = self._evaluate_final_score()
                action_result["grader_score"] = grader_score
                action_result["grader_breakdown"] = score_breakdown
                action_result["step_rewards_sum"] = self._step_rewards_sum
                reward = compute_reward(self.state_manager.current_state(), fallback_action, action_result)
            else:
                reward = step_reward

            self.state_manager.append_history_entry(
                {
                    "step": self.state_manager.step_count,
                    "action": "INVALID",
                    "status": "error",
                    "error": validation_error,
                    "reward": reward.score,
                }
            )
            observation = self._build_observation()
            return observation, reward.score, self.state_manager.episode_done, {"error": validation_error}

        action_data = validated_action.model_dump(exclude_none=True)
        action_result["is_repeated_action"] = self.state_manager.set_last_action(action_data)
        self.state_manager.set_action_result("ok", None)
        info: dict[str, Any] = {"action_type": validated_action.action_type}

        if validated_action.action_type == "READ_FILE":
            success, error, reopened, relevant = self.state_manager.open_file(validated_action.file_path or "")
            action_result["read_relevant"] = relevant
            info["reopened"] = reopened
            if not success:
                self.state_manager.set_action_result("error", error)
                info["error"] = error

        elif validated_action.action_type == "COMMENT":
            meaningful, spam, relevant = self.state_manager.add_comment(
                file_path=validated_action.file_path or "",
                line=validated_action.line or 1,
                comment=validated_action.comment or "",
                severity=validated_action.severity,
                category=validated_action.category,
            )
            action_result["meaningful_comment"] = meaningful
            action_result["spam_comment"] = spam
            info["comment_relevant"] = relevant

        elif validated_action.action_type == "SUGGEST_FIX":
            success, error, target_files = self.state_manager.apply_suggested_patch(
                patch=validated_action.suggested_patch or "",
                file_path=validated_action.file_path,
            )
            action_result["patch_valid"] = success
            info["patch_targets"] = target_files
            if not success:
                self.state_manager.set_action_result("error", error)
                info["error"] = error

        elif validated_action.action_type == "FINISH":
            grader_score, score_breakdown = self._evaluate_final_score()
            action_result["finalized"] = True
            action_result["grader_score"] = grader_score
            action_result["grader_breakdown"] = score_breakdown
            action_result["step_rewards_sum"] = self._step_rewards_sum
            self.state_manager.episode_done = True
            info["grader_score"] = grader_score
            info["grader_breakdown"] = score_breakdown

        self.state_manager.increment_step()

        if not self.state_manager.episode_done and self.state_manager.remaining_steps() <= 0:
            step_reward = compute_reward(self.state_manager.current_state(), validated_action, action_result)
            self._step_rewards_sum += step_reward.score

            grader_score, score_breakdown = self._evaluate_final_score()
            action_result["finalized"] = True
            action_result["grader_score"] = grader_score
            action_result["grader_breakdown"] = score_breakdown
            action_result["step_rewards_sum"] = self._step_rewards_sum
            self.state_manager.episode_done = True
            info["auto_finalized"] = True
            info["grader_score"] = grader_score
            info["grader_breakdown"] = score_breakdown

        reward = compute_reward(self.state_manager.current_state(), validated_action, action_result)
        if not action_result.get("finalized"):
            self._step_rewards_sum += reward.score

        self.state_manager.append_history_entry(
            {
                "step": self.state_manager.step_count,
                "action": validated_action.action_type,
                "status": self.state_manager.last_action_status,
                "error": self.state_manager.last_error,
                "result": action_result,
                "reward": reward.score,
            }
        )

        observation = self._build_observation()
        return observation, reward.score, self.state_manager.episode_done, info

    def state(self) -> dict[str, Any]:
        """Return full internal state for deterministic debugging and grading."""

        return self.state_manager.current_state()

    def close(self) -> None:
        """Compatibility no-op for evaluators that expect close()."""

        return None


class OpenEnvBenchmark(CodeReviewEnv):
    """Backward-compatible alias for existing validation commands."""

    pass
