"""Minimal FastAPI server exposing the OpenEnv environment."""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException

from env.core import CodeReviewEnv
from env.models import CodeReviewAction


asgi_app = FastAPI(title="OpenEnv CodeReview Benchmark")
_env_lock = Lock()
_env = CodeReviewEnv()


class _AppEntrypoint:
    """Support both ASGI calls and console-script invocation."""

    def __init__(self, wrapped_app: FastAPI) -> None:
        self._wrapped_app = wrapped_app

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            import uvicorn

            host = os.getenv("HOST", "0.0.0.0")
            port = int(os.getenv("PORT", "7860"))
            uvicorn.run("server.app:asgi_app", host=host, port=port)
            return None
        if len(args) == 1 and not kwargs:
            scope = args[0]

            async def _asgi2_adapter(receive: Any, send: Any) -> Any:
                return await self._wrapped_app(scope, receive, send)

            return _asgi2_adapter
        return self._wrapped_app(*args, **kwargs)


app = _AppEntrypoint(asgi_app)


@asgi_app.post("/reset")
def reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    task_id = "easy"
    if isinstance(payload, dict):
        task_id = str(payload.get("task_id") or payload.get("task") or task_id)

    with _env_lock:
        try:
            observation = _env.reset(task_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"observation": observation.model_dump()}


@asgi_app.post("/step")
def step(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    action_payload = payload.get("action", payload)

    with _env_lock:
        try:
            action = CodeReviewAction.model_validate(action_payload)
            observation, reward, done, info = _env.step(action)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }


@asgi_app.get("/state")
def state() -> dict[str, Any]:
    with _env_lock:
        try:
            return _env.state()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
