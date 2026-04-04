from __future__ import annotations

from env.core import CodeReviewEnv
from env.models import CodeReviewAction


def _run_normal_flow() -> tuple[list[float], dict]:
    env = CodeReviewEnv(task_id="easy")
    obs = env.reset()
    target_file = obs.files_changed[0] if obs.files_changed else "src/metrics.py"

    rewards: list[float] = []

    obs, r1, d1, _ = env.step(CodeReviewAction(action_type="READ_FILE", file_path=target_file))
    assert not d1
    rewards.append(r1)

    obs, r2, d2, _ = env.step(
        CodeReviewAction(
            action_type="COMMENT",
            file_path=target_file,
            line=1,
            comment="Potential correctness issue around edge-case handling.",
            severity="bug",
            category="correctness",
        )
    )
    assert not d2
    rewards.append(r2)

    obs, r3, d3, _ = env.step(
        CodeReviewAction(
            action_type="FINISH",
            finish_decision="REQUEST_CHANGES",
            finish_summary="Issue found and review completed.",
        )
    )
    assert d3
    rewards.append(r3)

    return rewards, obs.model_dump()


def test_1_normal_flow() -> None:
    rewards, _ = _run_normal_flow()
    assert all(isinstance(value, float) for value in rewards)
    assert any(abs(value) > 1e-12 for value in rewards)


def test_2_invalid_action_handling() -> None:
    env = CodeReviewEnv(task_id="easy")
    env.reset()

    obs, reward, done, info = env.step({"action_type": "READ_FILE"})

    assert obs is not None
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert isinstance(reward, float)
    assert reward < 0.0


def test_3_determinism() -> None:
    rewards_1, obs_1 = _run_normal_flow()
    rewards_2, obs_2 = _run_normal_flow()

    assert rewards_1 == rewards_2
    assert obs_1 == obs_2


def test_4_patch_validation() -> None:
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

    assert valid_reward > 0.0
    assert invalid_reward < 0.0


def test_5_step_limit_auto_finalize() -> None:
    env = CodeReviewEnv(task_id="easy")
    obs = env.reset()
    target_file = obs.files_changed[0] if obs.files_changed else "src/metrics.py"

    final_done = False
    final_info: dict = {}

    for _ in range(obs.max_steps + 2):
        _, _, done, info = env.step(CodeReviewAction(action_type="READ_FILE", file_path=target_file))
        final_done = done
        final_info = info
        if done:
            break

    assert final_done is True
    assert "grader_score" in final_info


def test_6_hard_task_multi_file_signal() -> None:
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

    assert score_one < score_multi
    assert score_one <= 0.3
