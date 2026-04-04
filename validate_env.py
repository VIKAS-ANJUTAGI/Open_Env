from __future__ import annotations

from typing import Any

from env.core import CodeReviewEnv
from env.models import CodeReviewAction


def run_test_1_normal_flow() -> tuple[bool, dict[str, Any]]:
    """TEST 1: reset -> READ_FILE -> COMMENT -> FINISH."""

    details: dict[str, Any] = {}
    try:
        env = CodeReviewEnv(task_id="easy")
        obs = env.reset()

        target_file = obs.files_changed[0] if obs.files_changed else "src/metrics.py"

        _, r1, d1, _ = env.step(CodeReviewAction(action_type="READ_FILE", file_path=target_file))
        _, r2, d2, _ = env.step(
            CodeReviewAction(
                action_type="COMMENT",
                file_path=target_file,
                line=1,
                comment="Potential correctness issue around edge-case handling.",
                severity="bug",
                category="correctness",
            )
        )
        _, r3, d3, _ = env.step(
            CodeReviewAction(
                action_type="FINISH",
                finish_decision="REQUEST_CHANGES",
                finish_summary="Issue found and review completed.",
            )
        )

        rewards = [r1, r2, r3]
        is_float_rewards = all(isinstance(r, float) for r in rewards)
        not_always_zero = any(abs(r) > 1e-12 for r in rewards)
        passed = (not d1) and (not d2) and d3 and is_float_rewards and not_always_zero

        details["rewards"] = rewards
        details["done_sequence"] = [d1, d2, d3]
        return passed, details
    except Exception as exc:  # pragma: no cover
        details["error"] = str(exc)
        return False, details


def run_test_2_invalid_action() -> tuple[bool, dict[str, Any]]:
    """TEST 2: invalid READ_FILE without file_path should not crash and should penalize."""

    details: dict[str, Any] = {}
    try:
        env = CodeReviewEnv(task_id="easy")
        env.reset()

        obs, reward, done, info = env.step({"action_type": "READ_FILE"})
        passed = (
            obs is not None
            and isinstance(reward, float)
            and reward < 0.0
            and isinstance(done, bool)
            and isinstance(info, dict)
        )

        details["reward"] = reward
        details["done"] = done
        details["info"] = info
        return passed, details
    except Exception as exc:  # pragma: no cover
        details["error"] = str(exc)
        return False, details


def _run_normal_flow_rewards_and_obs() -> tuple[list[float], dict[str, Any]]:
    """Helper for determinism check."""

    env = CodeReviewEnv(task_id="easy")
    obs = env.reset()
    target_file = obs.files_changed[0] if obs.files_changed else "src/metrics.py"

    rewards: list[float] = []

    obs, r, _, _ = env.step(CodeReviewAction(action_type="READ_FILE", file_path=target_file))
    rewards.append(r)
    obs, r, _, _ = env.step(
        CodeReviewAction(
            action_type="COMMENT",
            file_path=target_file,
            line=1,
            comment="Potential correctness issue around edge-case handling.",
            severity="bug",
            category="correctness",
        )
    )
    rewards.append(r)
    obs, r, _, _ = env.step(
        CodeReviewAction(
            action_type="FINISH",
            finish_decision="REQUEST_CHANGES",
            finish_summary="Issue found and review completed.",
        )
    )
    rewards.append(r)

    return rewards, obs.model_dump()


def run_test_3_determinism() -> tuple[bool, dict[str, Any]]:
    """TEST 3: identical sequence should produce identical rewards and observations."""

    details: dict[str, Any] = {}
    try:
        rewards_1, obs_1 = _run_normal_flow_rewards_and_obs()
        rewards_2, obs_2 = _run_normal_flow_rewards_and_obs()

        passed = (rewards_1 == rewards_2) and (obs_1 == obs_2)
        details["rewards_run1"] = rewards_1
        details["rewards_run2"] = rewards_2
        return passed, details
    except Exception as exc:  # pragma: no cover
        details["error"] = str(exc)
        return False, details


def run_test_4_patch_validation() -> tuple[bool, dict[str, Any]]:
    """TEST 4: valid patch positive, invalid patch negative."""

    details: dict[str, Any] = {}
    try:
        # Valid patch case
        env_valid = CodeReviewEnv(task_id="easy")
        env_valid.reset()
        _, valid_reward, _, _ = env_valid.step(
            CodeReviewAction(
                action_type="SUGGEST_FIX",
                file_path="src/metrics.py",
                suggested_patch=(
                    "@@ -1,2 +1,4 @@\n"
                    " def ratio(total, count):\n"
                    "-    return total / count\n"
                    "+    if count == 0:\n"
                    "+        return 0\n"
                    "+    return total / count\n"
                ),
            )
        )

        # Invalid patch case (old hunk does not match snapshot)
        env_invalid = CodeReviewEnv(task_id="easy")
        env_invalid.reset()
        _, invalid_reward, _, _ = env_invalid.step(
            CodeReviewAction(
                action_type="SUGGEST_FIX",
                file_path="src/metrics.py",
                suggested_patch=(
                    "@@ -1,2 +1,2 @@\n"
                    " def ratio(total, count):\n"
                    "-    return total * count\n"
                    "+    return total / count\n"
                ),
            )
        )

        passed = valid_reward > 0.0 and invalid_reward < 0.0
        details["valid_patch_reward"] = valid_reward
        details["invalid_patch_reward"] = invalid_reward
        return passed, details
    except Exception as exc:  # pragma: no cover
        details["error"] = str(exc)
        return False, details


def run_test_5_step_limit() -> tuple[bool, dict[str, Any]]:
    """TEST 5: reaching max_steps without FINISH should auto-complete and grade."""

    details: dict[str, Any] = {}
    try:
        env = CodeReviewEnv(task_id="easy")
        obs = env.reset()
        target_file = obs.files_changed[0] if obs.files_changed else "src/metrics.py"

        final_info: dict[str, Any] = {}
        final_done = False

        for _ in range(obs.max_steps + 2):
            _, _, done, info = env.step(CodeReviewAction(action_type="READ_FILE", file_path=target_file))
            final_info = info
            final_done = done
            if done:
                break

        passed = final_done and ("grader_score" in final_info)
        details["final_info"] = final_info
        return passed, details
    except Exception as exc:  # pragma: no cover
        details["error"] = str(exc)
        return False, details


def run_test_6_hard_task_check() -> tuple[bool, dict[str, Any]]:
    """TEST 6: hard task should score low if only one file is explored."""

    details: dict[str, Any] = {}
    try:
        # One-file exploration path
        env_one = CodeReviewEnv(task_id="hard")
        env_one.reset()
        env_one.step(CodeReviewAction(action_type="READ_FILE", file_path="src/auth.py"))
        _, _, _, info_one = env_one.step(
            CodeReviewAction(
                action_type="FINISH",
                finish_decision="REQUEST_CHANGES",
                finish_summary="Minimal review completed.",
            )
        )
        score_one = float(info_one.get("grader_score", 0.0))

        # Multi-file exploration path
        env_multi = CodeReviewEnv(task_id="hard")
        env_multi.reset()
        env_multi.step(CodeReviewAction(action_type="READ_FILE", file_path="src/auth.py"))
        env_multi.step(CodeReviewAction(action_type="READ_FILE", file_path="src/logger.py"))
        env_multi.step(
            CodeReviewAction(
                action_type="COMMENT",
                file_path="src/logger.py",
                line=2,
                comment="Token leakage in logs.",
                severity="security",
                category="security",
            )
        )
        _, _, _, info_multi = env_multi.step(
            CodeReviewAction(
                action_type="FINISH",
                finish_decision="REQUEST_CHANGES",
                finish_summary="Cross-file risk identified.",
            )
        )
        score_multi = float(info_multi.get("grader_score", 0.0))

        passed = (score_one < score_multi) and (score_one <= 0.3)
        details["one_file_score"] = score_one
        details["multi_file_score"] = score_multi
        return passed, details
    except Exception as exc:  # pragma: no cover
        details["error"] = str(exc)
        return False, details


def main() -> None:
    tests = [
        run_test_1_normal_flow,
        run_test_2_invalid_action,
        run_test_3_determinism,
        run_test_4_patch_validation,
        run_test_5_step_limit,
        run_test_6_hard_task_check,
    ]

    print("[START TESTS]")

    results: list[bool] = []
    for index, test_fn in enumerate(tests, start=1):
        passed, details = test_fn()
        results.append(passed)
        if passed:
            print(f"[TEST {index} PASS]")
        else:
            print(f"[TEST {index} FAIL]")
            print(f"[TEST {index} DETAILS] {details}")

    print("[END TESTS]")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
