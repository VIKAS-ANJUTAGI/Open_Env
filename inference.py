"""Deterministic OpenEnv inference runner with strict stdout logging."""

from __future__ import annotations

import json
import os
from typing import Any

from env.core import CodeReviewEnv
from env.models import CodeReviewAction, CodeReviewObservation

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


TASKS = ("easy", "medium", "hard")
BENCHMARK_NAME = "code-review-openenv"
MAX_HISTORY_ITEMS = 6
MAX_OBSERVATION_CHARS = 2000

_ALLOWED_SEVERITIES = {"nit", "suggestion", "bug", "security", "performance"}


def _compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _truncate_text(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    return text if len(text) <= limit else text[:limit]


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _build_prompt(observation: CodeReviewObservation) -> str:
    progress = {
        "step_count": observation.step_count,
        "remaining_steps": observation.remaining_steps,
        "opened_files": list(observation.file_contents.keys()),
        "last_history": observation.history[-2:],
    }
    payload = {
        "pr_title": observation.pr_title,
        "pr_description": observation.pr_description,
        "diff_by_file": observation.diff_by_file,
        "current_file": observation.current_file,
        "file_contents": observation.file_contents,
        "history": observation.history[-MAX_HISTORY_ITEMS:],
        "progress": progress,
    }
    return _truncate_text(
        "Return exactly one JSON object. Choose ONE action type and include only valid fields for that type. "
        "Allowed action_type: READ_FILE, COMMENT, SUGGEST_FIX, FINISH. "
        "COMMENT requires file_path,line,comment and severity in [nit,suggestion,bug,security,performance]. "
        "SUGGEST_FIX requires suggested_patch. FINISH requires finish_decision in [APPROVE,REQUEST_CHANGES].\n"
        f"OBSERVATION={_compact_json(payload)}"
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
                continue
            for attr in ("text", "content"):
                value = getattr(item, attr, None)
                if isinstance(value, str):
                    parts.append(value)
                    break
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def _extract_from_responses_output(output_items: Any) -> str:
    if not isinstance(output_items, list):
        return ""
    chunks: list[str] = []
    for item in output_items:
        content = getattr(item, "content", None)
        text = _content_to_text(content)
        if text:
            chunks.append(text)
            continue
        if isinstance(item, dict):
            text = _content_to_text(item.get("content"))
            if text:
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk)


def _extract_model_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if choices:
            message = choices[0].message
            text = _content_to_text(getattr(message, "content", ""))
            if text:
                return text
    except Exception:
        pass

    try:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text
    except Exception:
        pass

    try:
        output = getattr(response, "output", None)
        text = _extract_from_responses_output(output)
        if text:
            return text
    except Exception:
        pass

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            text = _content_to_text(message.get("content", "")) if isinstance(message, dict) else ""
            if text:
                return text
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        text = _extract_from_responses_output(response.get("output"))
        if text:
            return text

    return ""


def _parse_action_json(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_file_path(candidate: Any, observation: CodeReviewObservation) -> str | None:
    if not isinstance(candidate, str) or not candidate.strip():
        return None

    requested = candidate.strip()
    if requested in observation.files_changed:
        return requested

    lowered = requested.lower()
    for path in observation.files_changed:
        if path.lower() == lowered or path.lower().endswith("/" + lowered) or path.lower().endswith("\\" + lowered):
            return path
    return None


def _normalize_severity(value: Any) -> str:
    if not isinstance(value, str):
        return "suggestion"
    normalized = value.strip().lower()
    if normalized in _ALLOWED_SEVERITIES:
        return normalized
    aliases = {
        "info": "suggestion",
        "warning": "bug",
        "warn": "bug",
        "medium": "bug",
        "high": "security",
        "critical": "security",
        "low": "nit",
    }
    return aliases.get(normalized, "suggestion")


def _infer_category(comment: str) -> str:
    lowered = comment.lower()
    if any(token in lowered for token in ("token", "auth", "secret", "leak", "security")):
        return "security"
    if any(token in lowered for token in ("slow", "latency", "performance", "overhead")):
        return "performance"
    if any(token in lowered for token in ("logic", "condition", "boundary", "ttl", "stale", "regression")):
        return "logic"
    return "correctness"


def _fallback_action(observation: CodeReviewObservation) -> dict[str, Any]:
    unread = [path for path in observation.files_changed if path not in observation.file_contents]
    if unread and observation.remaining_steps > 1:
        return {"action_type": "READ_FILE", "file_path": unread[0]}

    if observation.current_file and observation.remaining_steps > 1:
        return {
            "action_type": "COMMENT",
            "file_path": observation.current_file,
            "line": 1,
            "comment": "Potential correctness issue requires review.",
            "severity": "suggestion",
            "category": "correctness",
        }

    return {
        "action_type": "FINISH",
        "finish_decision": "REQUEST_CHANGES",
        "finish_summary": "Fallback after invalid or unavailable model output.",
    }


def _coerce_model_action(action_data: dict[str, Any], observation: CodeReviewObservation) -> dict[str, Any]:
    action_type = str(action_data.get("action_type", "")).strip().upper()
    if action_type not in {"READ_FILE", "COMMENT", "SUGGEST_FIX", "FINISH"}:
        return _fallback_action(observation)

    if action_type == "READ_FILE":
        file_path = _normalize_file_path(action_data.get("file_path"), observation)
        if not file_path:
            unread = [path for path in observation.files_changed if path not in observation.file_contents]
            file_path = unread[0] if unread else (observation.files_changed[0] if observation.files_changed else None)
        if not file_path:
            return _fallback_action(observation)
        return {"action_type": "READ_FILE", "file_path": file_path}

    if action_type == "COMMENT":
        file_path = _normalize_file_path(action_data.get("file_path"), observation)
        if not file_path:
            file_path = observation.current_file or (observation.files_changed[0] if observation.files_changed else None)
        if not file_path:
            return _fallback_action(observation)

        line = action_data.get("line")
        if not isinstance(line, int) or line < 1:
            line = 1

        comment = action_data.get("comment")
        if not isinstance(comment, str) or not comment.strip():
            comment = "Potential issue found in this file."

        return {
            "action_type": "COMMENT",
            "file_path": file_path,
            "line": line,
            "comment": comment.strip(),
            "severity": _normalize_severity(action_data.get("severity")),
            "category": _infer_category(comment),
        }

    if action_type == "SUGGEST_FIX":
     patch = action_data.get("suggested_patch") or action_data.get("diff")

    if not isinstance(patch, str) or not patch.strip():
        return _fallback_action(observation)

    file_path = _normalize_file_path(action_data.get("file_path"), observation)

    if not file_path:
        file_path = observation.current_file or (
            observation.files_changed[0] if observation.files_changed else None
        )

    if not file_path:
        return _fallback_action(observation)

    return {
        "action_type": "SUGGEST_FIX",
        "file_path": file_path,
        "diff": patch.strip(),   # ✅ IMPORTANT
    }

    decision_raw = str(action_data.get("finish_decision", "REQUEST_CHANGES")).strip().upper()
    decision = decision_raw if decision_raw in {"APPROVE", "REQUEST_CHANGES"} else "REQUEST_CHANGES"
    summary = action_data.get("finish_summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "Review completed."
    return {"action_type": "FINISH", "finish_decision": decision, "finish_summary": summary.strip()}


def _call_model(client: Any, model_name: str, observation: CodeReviewObservation) -> dict[str, Any]:
    if client is None or not model_name:
        return _fallback_action(observation)

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return only strict JSON."},
                {"role": "user", "content": _build_prompt(observation)},
            ],
        )
        parsed = _parse_action_json(_extract_model_text(response))
        if parsed is None:
            return _fallback_action(observation)
        return _coerce_model_action(parsed, observation)
    except Exception:
        return _fallback_action(observation)


def _normalize_action(action_data: dict[str, Any], observation: CodeReviewObservation) -> CodeReviewAction:
    try:
        return CodeReviewAction.model_validate(action_data)
    except Exception:
        return CodeReviewAction.model_validate(_fallback_action(observation))


def _serialize_action(action: CodeReviewAction) -> str:
    return _compact_json(action.model_dump(exclude_none=True))


def _log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={model_name}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={_format_number(reward)} done={_format_bool(done)} error={error_text}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(_format_number(value) for value in rewards)
    print(
        f"[END] success={_format_bool(success)} steps={steps} score={_format_number(score)} rewards={rewards_text}",
        flush=True,
    )


def _build_client(api_key: str, api_base_url: str) -> Any:
    if OpenAI is None or not api_key:
        return None
    try:
        return OpenAI(api_key=api_key, base_url=api_base_url or None)
    except Exception:
        return None


def _run_task(env: CodeReviewEnv, client: Any, model_name: str, task_name: str) -> tuple[float, int, bool, list[float]]:
    observation = env.reset(task_name)
    _log_start(task_name, model_name)

    rewards: list[float] = []
    final_score = 0.0
    steps_taken = 0
    success = False
    previous_signature: tuple[str, str] | None = None
    repeated_count = 0

    try:
        while steps_taken < observation.max_steps:
            action_data = _call_model(client, model_name, observation)
            action = _normalize_action(action_data, observation)
            signature = (action.action_type, action.file_path or "")
            if signature == previous_signature:
                repeated_count += 1
            else:
                repeated_count = 0
            previous_signature = signature

            if repeated_count >= 2:
                action = CodeReviewAction(
                    action_type="FINISH",
                    finish_decision="REQUEST_CHANGES",
                    finish_summary="Loop guard forced finish.",
                )

            try:
                observation, reward, done, info = env.step(action)
                error = info.get("error") if isinstance(info, dict) else None
                final_score = float(info.get("grader_score", final_score)) if isinstance(info, dict) else final_score
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)
                final_score = 0.0

            steps_taken += 1
            rewards.append(float(reward))
            _log_step(steps_taken, _serialize_action(action), float(reward), bool(done), error)

            if done:
                break

        success = final_score >= 0.5
        return final_score, steps_taken, success, rewards
    finally:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception:
            pass


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    hf_token = os.getenv("HF_TOKEN", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = openai_api_key or hf_token

    client = _build_client(api_key=api_key, api_base_url=api_base_url)
    env = CodeReviewEnv()

    scores: list[float] = []
    for task_name in TASKS:
        final_score, steps_taken, success, rewards = _run_task(env, client, model_name, task_name)
        _log_end(success, steps_taken, final_score, rewards)
        scores.append(final_score)


if __name__ == "__main__":
    main()
