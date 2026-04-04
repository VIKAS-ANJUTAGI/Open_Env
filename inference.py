"""Baseline inference entry point for code-review-openenv.

The script emits structured logs and reads model configuration from environment
variables so it can be reproduced in a containerized evaluation setting.
"""

from __future__ import annotations

import json
import os
from typing import Any

from env.core import CodeReviewEnv
from env.models import CodeReviewAction

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency may be installed later
    OpenAI = None  # type: ignore[assignment]


def _log(tag: str, payload: dict[str, Any]) -> None:
    """Emit a structured stdout log line."""

    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}")


def main() -> None:
    """Run the baseline inference loop against the environment."""

    api_base_url = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")
    hf_token = os.getenv("HF_TOKEN", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", hf_token)

    _log(
        "START",
        {
            "api_base_url": api_base_url,
            "model_name": model_name,
            "has_hf_token": bool(hf_token),
            "has_openai_api_key": bool(openai_api_key),
        },
    )

    if OpenAI is not None and openai_api_key:
        client = OpenAI(api_key=openai_api_key, base_url=api_base_url or None)
    else:
        client = None

    env = CodeReviewEnv()
    observation = env.reset()
    total_score = 0.0

    scripted_actions = [
        CodeReviewAction(action_type="COMMENT", file_path="README.md", line=1, comment="Review scaffold in place.", severity="suggestion", category="documentation"),
        CodeReviewAction(action_type="FINISH", finish_decision="REQUEST_CHANGES", finish_summary="Baseline placeholder review completed."),
    ]

    for step_index, action in enumerate(scripted_actions, start=1):
        if client is not None:
            _ = client

        observation, reward, done, info = env.step(action)
        total_score += reward
        _log(
            "STEP",
            {
                "step": step_index,
                "action_type": action.action_type,
                "reward": reward,
                "done": done,
                "info": info,
                "remaining_steps": observation.remaining_steps,
            },
        )
        if done:
            break

    _log(
        "END",
        {
            "total_score": total_score,
            "final_step_count": observation.step_count,
            "episode_done": env.state()["episode_done"],
        },
    )


if __name__ == "__main__":
    main()
