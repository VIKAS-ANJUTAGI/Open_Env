"""Deterministic state management for code-review-openenv.

The StateManager keeps full internal repository snapshots and hidden ground
truth while exposing only opened-file content to the agent.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .tasks.easy_bug import EasyBugTask
from .tasks.hard_cross_file import HardCrossFileTask
from .tasks.medium_logic import MediumLogicTask


def _task_catalog() -> dict[str, dict[str, Any]]:
    """Return benchmark tasks sourced directly from task definitions."""

    return {
        "easy": EasyBugTask().to_spec(),
        "medium": MediumLogicTask().to_spec(),
        "hard": HardCrossFileTask().to_spec(),
    }


class StateManager:
    """Manage full internal episode state for deterministic code-review evaluation."""

    def __init__(self, max_steps: int = 10) -> None:
        self._catalog = _task_catalog()
        self._default_max_steps = max_steps
        self.reset_runtime_only()

    def reset_runtime_only(self) -> None:
        """Reset transient state without loading a task."""

        self.active_task = ""
        self.max_steps = self._default_max_steps
        self.step_count = 0
        self.episode_done = False

        self.pr_title = ""
        self.pr_description = ""
        self.pr_diff = ""
        self.diff_by_file: dict[str, str] = {}
        self.files_changed: list[str] = []

        self.repo_snapshot: dict[str, str] = {}
        self.working_snapshot: dict[str, str] = {}
        self.dependency_graph: dict[str, list[str]] = {}
        self.ground_truth: dict[str, Any] = {}

        self.opened_files: list[str] = []
        self.opened_counts: dict[str, int] = {}
        self.current_file: str | None = None

        self.comments: list[dict[str, Any]] = []
        self.comment_counts: dict[tuple[str, int, str], int] = {}
        self.suggested_patches: list[dict[str, Any]] = []
        self.last_action_signature: str | None = None

        self.history: list[str] = []
        self.last_action: str | None = None
        self.last_action_status: str = "ok"
        self.last_error: str | None = None

    def reset(self, task_id: str = "easy") -> None:
        """Reset environment state and load a deterministic task fixture."""

        task = self._catalog.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.reset_runtime_only()
        self.active_task = task_id
        self.max_steps = int(task["max_steps"])
        self.pr_title = task["pr_title"]
        self.pr_description = task["pr_description"]
        self.diff_by_file = copy.deepcopy(task["diff_by_file"])
        self.files_changed = list(task["files_changed"])
        self.repo_snapshot = copy.deepcopy(task["repo_snapshot"])
        self.working_snapshot = copy.deepcopy(task["repo_snapshot"])
        self.dependency_graph = copy.deepcopy(task["dependency_graph"])
        self.ground_truth = copy.deepcopy(task["ground_truth"])
        self.pr_diff = "\n".join(f"## {path}\n{diff}" for path, diff in self.diff_by_file.items())
        self.history.append(f"reset:{task_id}")

    def remaining_steps(self) -> int:
        """Return remaining step budget."""

        return max(self.max_steps - self.step_count, 0)

    def _serialize_action(self, action_data: dict[str, Any]) -> str:
        """Return a stable action signature used for repetition detection."""

        return json.dumps(action_data, sort_keys=True, separators=(",", ":"))

    def set_last_action(self, action_data: dict[str, Any]) -> bool:
        """Store last action and return whether action is repeated."""

        self.last_action = self._serialize_action(action_data)
        is_repeated = self.last_action_signature == self.last_action
        self.last_action_signature = self.last_action
        return is_repeated

    def set_action_result(self, status: str, error: str | None = None) -> None:
        """Store status and optional error for the latest action."""

        self.last_action_status = status
        self.last_error = error

    def open_file(self, file_path: str) -> tuple[bool, str | None, bool, bool]:
        """Open a file from the internal snapshot without exposing full repository."""

        if file_path not in self.working_snapshot:
            return False, f"file does not exist in repository snapshot: {file_path}", False, False

        was_opened_before = file_path in self.opened_counts
        self.opened_counts[file_path] = self.opened_counts.get(file_path, 0) + 1
        if not was_opened_before:
            self.opened_files.append(file_path)
        self.current_file = file_path
        is_relevant = file_path in set(self.ground_truth.get("relevant_files", []))
        self.history.append(f"read:{file_path}:reopen={was_opened_before}")
        return True, None, was_opened_before, is_relevant

    def add_comment(
        self,
        file_path: str,
        line: int,
        comment: str,
        severity: str | None,
        category: str | None,
    ) -> tuple[bool, bool, bool]:
        """Store a structured comment and return quality markers.

        Returns:
            meaningful: whether comment aligns with task signals
            spam: whether this comment is repeated spam
            relevant: whether file is relevant to ground truth
        """

        normalized = comment.strip().lower()
        key = (file_path, line, normalized)
        self.comment_counts[key] = self.comment_counts.get(key, 0) + 1
        spam = self.comment_counts[key] > 1

        relevant = file_path in set(self.ground_truth.get("relevant_files", []))
        expected_categories = set(self.ground_truth.get("expected_categories", []))
        meaningful = bool(normalized) and relevant and (category in expected_categories or category is None)

        self.comments.append(
            {
                "file_path": file_path,
                "line": line,
                "comment": comment,
                "severity": severity,
                "category": category,
                "spam": spam,
                "meaningful": meaningful,
            }
        )
        self.history.append(f"comment:{file_path}:{line}:spam={spam}:meaningful={meaningful}")
        return meaningful, spam, relevant

    def _extract_target_files_from_patch(self, patch: str) -> list[str]:
        """Extract target file paths from unified diff headers."""

        targets: list[str] = []
        lines = patch.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ "):
                plus_path = lines[index + 1][4:].strip()
                normalized = plus_path
                if normalized.startswith("b/"):
                    normalized = normalized[2:]
                targets.append(normalized)
        return targets

    def _apply_patch_to_content(self, content: str, patch: str) -> tuple[bool, str, str | None]:
        """Apply a simplified unified diff hunk set to file content deterministically."""

        lines = patch.splitlines()
        hunks: list[tuple[list[str], list[str]]] = []
        current_old: list[str] | None = None
        current_new: list[str] | None = None

        for line in lines:
            if line.startswith("@@"):
                if current_old is not None and current_new is not None:
                    hunks.append((current_old, current_new))
                current_old = []
                current_new = []
                continue
            if current_old is None or current_new is None:
                continue
            if line.startswith("+") and not line.startswith("+++"):
                current_new.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current_old.append(line[1:])
            elif line.startswith(" "):
                current_old.append(line[1:])
                current_new.append(line[1:])

        if current_old is not None and current_new is not None:
            hunks.append((current_old, current_new))

        if not hunks:
            return False, content, "patch contains no hunks"

        updated = content
        for old_block, new_block in hunks:
            old_text = "\n".join(old_block)
            new_text = "\n".join(new_block)
            if old_text and old_text not in updated:
                return False, content, "hunk does not apply cleanly to current snapshot"
            if old_text:
                updated = updated.replace(old_text, new_text, 1)
            else:
                updated = f"{new_text}\n{updated}" if new_text else updated

        if content.endswith("\n") and not updated.endswith("\n"):
            updated += "\n"
        return True, updated, None

    def apply_suggested_patch(self, patch: str, file_path: str | None) -> tuple[bool, str | None, list[str]]:
        """Validate and apply patch to a copy of repository snapshot.

        Returns:
            success
            error
            target_files
        """

        target_files = self._extract_target_files_from_patch(patch)
        if not target_files and file_path:
            target_files = [file_path]

        if not target_files:
            return False, "unable to determine target file for patch", []

        next_snapshot = copy.deepcopy(self.working_snapshot)
        for target in target_files:
            if target not in next_snapshot:
                return False, f"patch target missing from repository snapshot: {target}", target_files

            success, new_content, error = self._apply_patch_to_content(next_snapshot[target], patch)
            if not success:
                return False, error, target_files
            next_snapshot[target] = new_content

        self.working_snapshot = next_snapshot
        self.suggested_patches.append({"file_path": file_path, "patch": patch, "targets": list(target_files)})
        self.history.append(f"patch:{','.join(target_files)}")
        return True, None, target_files

    def finalize_grade(self, finish_decision: str) -> float:
        """Compute deterministic final grader score in [0, 1]."""

        score = 0.0
        expected_decision = self.ground_truth.get("expected_decision")
        expected_categories = set(self.ground_truth.get("expected_categories", []))
        expected_patch_files = set(self.ground_truth.get("expected_patch_files", []))

        if finish_decision == expected_decision:
            score += 0.25

        commented_categories = {c["category"] for c in self.comments if c.get("category")}
        matched_categories = len(commented_categories & expected_categories)
        if expected_categories:
            score += 0.25 * (matched_categories / len(expected_categories))

        patched_files = {
            target
            for patch_entry in self.suggested_patches
            for target in patch_entry.get("targets", [])
        }
        if expected_patch_files:
            score += 0.3 * (len(patched_files & expected_patch_files) / len(expected_patch_files))

        relevant_files = set(self.ground_truth.get("relevant_files", []))
        opened_relevant = len(set(self.opened_files) & relevant_files)
        if relevant_files:
            score += 0.2 * (opened_relevant / len(relevant_files))

        return max(0.0, min(1.0, score))

    def append_history_entry(self, entry: dict[str, Any]) -> None:
        """Append a full trace entry as JSON for deterministic debug output."""

        self.history.append(json.dumps(entry, sort_keys=True, separators=(",", ":")))

    def increment_step(self) -> None:
        """Advance step counter by one."""

        self.step_count += 1

    def get_observation_history(self) -> list[str]:
        """Return full trace history visible to the agent."""

        return list(self.history)

    def get_visible_file_contents(self) -> dict[str, str]:
        """Return opened file contents only, never full repository snapshot."""

        return {path: self.working_snapshot[path] for path in self.opened_files if path in self.working_snapshot}

    def current_state(self) -> dict[str, Any]:
        """Return full internal state for rewarding, grading, and debugging."""

        return {
            "active_task": self.active_task,
            "pr_title": self.pr_title,
            "pr_description": self.pr_description,
            "pr_diff": self.pr_diff,
            "diff_by_file": copy.deepcopy(self.diff_by_file),
            "files_changed": list(self.files_changed),
            "file_contents": self.get_visible_file_contents(),
            "current_file": self.current_file,
            "history": self.get_observation_history(),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "remaining_steps": self.remaining_steps(),
            "last_action": self.last_action,
            "last_action_status": self.last_action_status,
            "last_error": self.last_error,
            "episode_done": self.episode_done,
            "ground_truth": copy.deepcopy(self.ground_truth),
            "repo_snapshot": copy.deepcopy(self.repo_snapshot),
            "working_snapshot": copy.deepcopy(self.working_snapshot),
            "opened_files": list(self.opened_files),
            "opened_counts": dict(self.opened_counts),
            "comments": copy.deepcopy(self.comments),
            "suggested_patches": copy.deepcopy(self.suggested_patches),
            "dependency_graph": copy.deepcopy(self.dependency_graph),
        }
