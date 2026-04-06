"""FastAPI wrapper exposing the OpenEnv code review environment over HTTP."""

from __future__ import annotations

from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException

from env.core import CodeReviewEnv
from env.models import CodeReviewAction


app = FastAPI(title="OpenEnv CodeReview Benchmark")
_env_lock = Lock()
_env = CodeReviewEnv()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "benchmark": "code-review-openenv"}


@app.post("/reset")
def reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    task_id = "easy"
    if isinstance(payload, dict):
        task_id = str(payload.get("task_id") or payload.get("task") or task_id)

    with _env_lock:
        observation = _env.reset(task_id)
        return {"observation": observation.model_dump()}


@app.post("/step")
def step(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    action_payload = payload.get("action", payload)

    with _env_lock:
        try:
            action = CodeReviewAction.model_validate(action_payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        observation, reward, done, info = _env.step(action)
        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }


@app.get("/state")
def state() -> dict[str, Any]:
    with _env_lock:
        return _env.state()
