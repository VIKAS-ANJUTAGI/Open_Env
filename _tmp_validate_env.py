from env.core import CodeReviewEnv
from env.models import CodeReviewAction


def run(task: str) -> tuple[float, bool, float]:
    env = CodeReviewEnv(task_id=task)
    env.reset()

    if task == "easy":
        actions = [
            CodeReviewAction(action_type="READ_FILE", file_path="src/metrics.py"),
            CodeReviewAction(
                action_type="COMMENT",
                file_path="src/metrics.py",
                line=1,
                comment="Division by zero risk.",
                severity="bug",
                category="correctness",
            ),
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
            ),
            CodeReviewAction(
                action_type="FINISH",
                finish_decision="REQUEST_CHANGES",
                finish_summary="Bug confirmed and fix suggested.",
            ),
        ]
    elif task == "medium":
        actions = [
            CodeReviewAction(action_type="READ_FILE", file_path="src/cache.py"),
            CodeReviewAction(
                action_type="COMMENT",
                file_path="src/cache.py",
                line=3,
                comment="Boundary condition appears wrong.",
                severity="bug",
                category="logic",
            ),
            CodeReviewAction(
                action_type="SUGGEST_FIX",
                file_path="src/cache.py",
                suggested_patch=(
                    "@@ -1,5 +1,5 @@\n"
                    " def get_cached(store, key, now):\n"
                    "     entry = store.get(key)\n"
                    "-    if entry and entry[1] > now:\n"
                    "+    if entry and entry[1] >= now:\n"
                    "         return entry[0]\n"
                    "     return None\n"
                ),
            ),
            CodeReviewAction(
                action_type="FINISH",
                finish_decision="REQUEST_CHANGES",
                finish_summary="Logic flaw identified.",
            ),
        ]
    else:
        actions = [
            CodeReviewAction(action_type="READ_FILE", file_path="src/auth.py"),
            CodeReviewAction(action_type="READ_FILE", file_path="src/logger.py"),
            CodeReviewAction(
                action_type="COMMENT",
                file_path="src/logger.py",
                line=2,
                comment="Token is logged in plain text.",
                severity="security",
                category="security",
            ),
            CodeReviewAction(
                action_type="SUGGEST_FIX",
                file_path="src/logger.py",
                suggested_patch=(
                    "@@ -1,2 +1,2 @@\n"
                    " def log_login(user, token):\n"
                    "-    print(f'user={user} token={token}')\n"
                    "+    print(f'user={user} token=<redacted>')\n"
                ),
            ),
            CodeReviewAction(
                action_type="FINISH",
                finish_decision="REQUEST_CHANGES",
                finish_summary="Cross-file security issue found.",
            ),
        ]

    last = (None, 0.0, False, {})
    for action in actions:
        last = env.step(action)
        if last[2]:
            break

    _, final_reward, done, info = last
    return round(float(final_reward), 6), bool(done), round(float(info.get("grader_score", 0.0)), 6)


if __name__ == "__main__":
    for task_name in ["easy", "medium", "hard"]:
        first = run(task_name)
        second = run(task_name)
        print(task_name, "run1=", first, "run2=", second, "deterministic=", first == second)
