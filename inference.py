"""Production-grade deterministic inference runner for code-review-openenv."""

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


TASKS = ["easy", "medium", "hard"]
MAX_HISTORY_ITEMS = 8
MAX_TEXT_CHARS = 8000


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _build_prompt(observation: CodeReviewObservation) -> str:
    history = observation.history[-MAX_HISTORY_ITEMS:]
    payload = {
        "pr_title": observation.pr_title,
        "pr_description": observation.pr_description,
        "diff_by_file": observation.diff_by_file,
        "current_file": observation.current_file,
        "file_contents": observation.file_contents,
        "history": history,
        "remaining_steps": observation.remaining_steps,
    }
    context = _truncate_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), MAX_TEXT_CHARS)
    return (
        "You are a deterministic code review agent. "
        "Respond with exactly one JSON object and no markdown. "
        "Allowed action_type values: READ_FILE, COMMENT, SUGGEST_FIX, FINISH. "
        "Output schema keys: action_type, file_path, line, comment, severity, suggested_patch, finish_decision, finish_summary. "
        "Choose one valid next action based on the observation context. "
        "If uncertain, return FINISH with finish_decision REQUEST_CHANGES.\n"
        f"OBSERVATION={context}"
    )


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                nested_text = item.get("content")
                if isinstance(nested_text, str):
                    parts.append(nested_text)
                    continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
                continue
            nested_attr = getattr(item, "content", None)
            if isinstance(nested_attr, str):
                parts.append(nested_attr)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        nested_text = content.get("content")
        if isinstance(nested_text, str):
            return nested_text
    return ""


def _extract_from_responses_output(output_items: Any) -> str:
    if not isinstance(output_items, list):
        return ""
    chunks: list[str] = []
    for item in output_items:
        item_content = getattr(item, "content", None)
        text = _extract_content_text(item_content)
        if text:
            chunks.append(text)
            continue

        if isinstance(item, dict):
            text = _extract_content_text(item.get("content"))
            if text:
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk)


def _extract_model_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if choices:
            message = choices[0].message
            content = getattr(message, "content", "")
            text = _extract_content_text(content)
            if text:
                return text
    except Exception:
        pass

    try:
        output = getattr(response, "output_text", None)
        if isinstance(output, str):
            return output
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
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                if isinstance(message, dict):
                    text = _extract_content_text(message.get("content", ""))
                    if text:
                        return text
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text
        text = _extract_from_responses_output(response.get("output"))
        if text:
            return text

    return ""


def _safe_json_dict(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    brace_positions = [idx for idx, char in enumerate(text) if char == "{"]
    for start in brace_positions:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _fallback_action(observation: CodeReviewObservation) -> dict[str, Any]:
    unread = [path for path in observation.files_changed if path not in observation.file_contents]
    if unread and observation.remaining_steps > 1:
        return {"action_type": "READ_FILE", "file_path": unread[0]}
    if observation.current_file and observation.remaining_steps > 1:
        return {
            "action_type": "COMMENT",
            "file_path": observation.current_file,
            "line": 1,
            "comment": "Potential issue needs verification.",
            "severity": "suggestion",
            "category": "correctness",
        }
    return {
        "action_type": "FINISH",
        "finish_decision": "REQUEST_CHANGES",
        "finish_summary": "Review completed with deterministic fallback.",
    }


def _next_action(client: Any, model_name: str, observation: CodeReviewObservation) -> dict[str, Any]:
    if client is None or not model_name:
        return _fallback_action(observation)

    prompt = _build_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = _extract_model_text(response)
        parsed = _safe_json_dict(raw)
        if parsed is None:
            return {
                "action_type": "FINISH",
                "finish_decision": "REQUEST_CHANGES",
                "finish_summary": "Model output invalid JSON.",
            }
        return parsed
    except Exception:
        return {
            "action_type": "FINISH",
            "finish_decision": "REQUEST_CHANGES",
            "finish_summary": "Model call failed.",
        }


def _normalized_action(action_data: dict[str, Any]) -> CodeReviewAction:
    try:
        return CodeReviewAction.model_validate(action_data)
    except Exception:
        return CodeReviewAction(
            action_type="FINISH",
            finish_decision="REQUEST_CHANGES",
            finish_summary="Action validation fallback.",
        )


def _print_start(task_name: str) -> None:
    print("[START]")
    print(f"task_name={task_name}")


def _print_step(step: int, action_type: str, reward: float, done: bool) -> None:
    print("[STEP]")
    print(f"step={step}")
    print(f"action={action_type}")
    print(f"reward={reward}")
    print(f"done={_bool_text(done)}")


def _print_end(task_name: str, final_reward: float) -> None:
    print("[END]")
    print(f"task_name={task_name}")
    print(f"final_reward={final_reward}")


def _run_task(env: CodeReviewEnv, client: Any, model_name: str, task_name: str) -> float:
    observation = env.reset(task_name)
    _print_start(task_name)

    total_reward = 0.0
    step = 0
    previous_signature: tuple[str, str] | None = None
    repeated_count = 0

    while step < observation.max_steps:
        step += 1
        action_data = _next_action(client, model_name, observation)
        action = _normalized_action(action_data)

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

        observation, reward, done, _ = env.step(action)
        total_reward += reward
        _print_step(step=step, action_type=action.action_type, reward=reward, done=done)

        if done:
            break

    _print_end(task_name, total_reward)
    return total_reward


def _build_client(api_key: str, api_base_url: str) -> Any:
    if OpenAI is None or not api_key:
        return None
    try:
        return OpenAI(api_key=api_key, base_url=api_base_url or None)
    except Exception:
        return None


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")
    hf_token = os.getenv("HF_TOKEN", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = openai_api_key or hf_token

    client = _build_client(api_key=api_key, api_base_url=api_base_url)
    env = CodeReviewEnv()

    scores: list[float] = []
    for task_name in TASKS:
        task_score = _run_task(env=env, client=client, model_name=model_name, task_name=task_name)
        scores.append(task_score)

    average_score = sum(scores) / len(scores) if scores else 0.0
    print(f"average_score={average_score}")


if __name__ == "__main__":
    main()
