"""Strict protocol models for code-review-openenv.

These models define the benchmark interaction contract between an agent and the
environment. They intentionally enforce structured actions, bounded reward
values, and deterministic validation semantics.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


_UNIFIED_DIFF_FILE_HEADER = re.compile(r"^---\s+.+\n\+\+\+\s+.+", re.MULTILINE)


class CodeReviewObservation(BaseModel):
    """Agent-facing observation for the code-review benchmark.

    The observation deliberately exposes only opened-file contents and review
    context needed for controlled multi-step reasoning. The full repository
    state remains internal to the environment.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pr_title: str = Field(default="", description="Short review title or issue summary.")
    pr_description: str = Field(default="", description="Review description or task objective.")
    pr_diff: str = Field(default="", description="Full diff for the current review context.")
    diff_by_file: dict[str, str] = Field(default_factory=dict, description="Per-file diff snippets keyed by file path.")
    files_changed: list[str] = Field(default_factory=list, description="Ordered list of files changed in the review.")
    file_contents: dict[str, str] = Field(
        default_factory=dict,
        description="Contents of opened files only; never the full repository dump.",
    )
    current_file: Optional[str] = Field(default=None, description="Currently focused file path, if any.")
    history: list[str] = Field(default_factory=list, description="Compact textual history of prior agent actions.")
    step_count: int = Field(default=0, ge=0, description="Number of steps already executed in the episode.")
    max_steps: int = Field(default=1, ge=1, description="Maximum number of steps allowed in the episode.")
    remaining_steps: int = Field(default=1, ge=0, description="Remaining steps before termination.")
    last_action: Optional[str] = Field(default=None, description="Serialized representation of the last action taken.")
    last_action_status: Literal["ok", "error"] = Field(default="ok", description="Outcome of the last action.")
    last_error: Optional[str] = Field(default=None, description="Last validation or execution error, if any.")

    @field_validator("files_changed")
    @classmethod
    def _validate_files_changed(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("files_changed must not contain duplicate file paths")
        return value

    @field_validator("file_contents")
    @classmethod
    def _validate_file_contents(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) != len(set(value.keys())):
            raise ValueError("file_contents must not contain duplicate keys")
        return value

    @field_validator("remaining_steps")
    @classmethod
    def _validate_remaining_steps(cls, value: int, info: ValidationInfo) -> int:
        step_count = info.data.get("step_count")
        max_steps = info.data.get("max_steps")
        if isinstance(step_count, int) and isinstance(max_steps, int):
            expected_remaining = max(max_steps - step_count, 0)
            if value != expected_remaining:
                raise ValueError(
                    f"remaining_steps must equal max_steps - step_count ({expected_remaining})"
                )
        return value

    @model_validator(mode="after")
    def _validate_observation(self) -> "CodeReviewObservation":
        if self.step_count > self.max_steps:
            raise ValueError("step_count cannot exceed max_steps")
        if self.current_file is not None and self.current_file not in self.file_contents:
            raise ValueError("current_file must reference one of the opened files in file_contents")
        if self.last_action_status == "error" and not self.last_error:
            raise ValueError("last_error must be provided when last_action_status is 'error'")
        if self.last_action_status == "ok" and self.last_error is not None:
            raise ValueError("last_error must be omitted when last_action_status is 'ok'")
        return self


class CodeReviewAction(BaseModel):
    """Structured agent action for the code-review benchmark.

    Free-text-only actions are disallowed. Each action type has explicit
    constraints so the environment can validate and grade actions deterministically.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: Literal["READ_FILE", "COMMENT", "SUGGEST_FIX", "FINISH"] = Field(
        description="Primary action capability requested by the agent."
    )
    file_path: Optional[str] = Field(default=None, description="Target file path for file-scoped actions.")
    line: Optional[int] = Field(default=None, ge=1, description="1-based source line for COMMENT actions.")
    comment: Optional[str] = Field(default=None, description="Structured review comment text.")
    severity: Optional[Literal["nit", "suggestion", "bug", "security", "performance"]] = Field(
        default=None,
        description="Severity classification for COMMENT actions.",
    )
    category: Optional[str] = Field(default=None, description="Stable issue category for grader mapping.")
    suggested_patch: Optional[str] = Field(default=None, description="Unified diff proposing a concrete fix.")
    finish_decision: Optional[Literal["APPROVE", "REQUEST_CHANGES"]] = Field(
        default=None,
        description="Final review decision for FINISH actions.",
    )
    finish_summary: Optional[str] = Field(default=None, description="Brief explanation for the final decision.")

    @field_validator("file_path")
    @classmethod
    def _normalize_file_path(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value:
            raise ValueError("file_path cannot be empty when provided")
        return value

    @field_validator("comment")
    @classmethod
    def _normalize_comment(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("comment cannot be empty when provided")
        return value

    @field_validator("suggested_patch")
    @classmethod
    def _validate_suggested_patch(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("suggested_patch cannot be empty when provided")
        return value

    @field_validator("finish_summary")
    @classmethod
    def _normalize_finish_summary(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("finish_summary cannot be empty when provided")
        return value

    @model_validator(mode="after")
    def _validate_action_combinations(self) -> "CodeReviewAction":
        action_type = self.action_type

        if action_type == "READ_FILE":
            if self.file_path is None:
                raise ValueError("file_path must be provided for READ_FILE actions")
            forbidden = [self.comment, self.suggested_patch, self.finish_decision, self.finish_summary, self.line, self.severity, self.category]
            if any(value is not None for value in forbidden):
                raise ValueError("READ_FILE actions cannot include comment, fix, or finish fields")

        elif action_type == "COMMENT":
            if self.file_path is None:
                raise ValueError("file_path must be provided for COMMENT actions")
            if self.line is None:
                raise ValueError("line must be provided for COMMENT actions")
            if self.comment is None:
                raise ValueError("comment must be provided for COMMENT actions")
            if self.suggested_patch is not None or self.finish_decision is not None or self.finish_summary is not None:
                raise ValueError("COMMENT actions cannot include suggested_patch or finish fields")

        elif action_type == "SUGGEST_FIX":
            if self.suggested_patch is None:
                raise ValueError("suggested_patch must be provided for SUGGEST_FIX actions")
            has_file_path = self.file_path is not None
            has_file_headers = bool(_UNIFIED_DIFF_FILE_HEADER.search(self.suggested_patch))
            if not has_file_path and not has_file_headers:
                raise ValueError("SUGGEST_FIX actions require file_path or a unified diff with file headers")
            if self.comment is not None or self.line is not None or self.severity is not None or self.category is not None:
                raise ValueError("SUGGEST_FIX actions cannot include comment fields")
            if self.finish_decision is not None or self.finish_summary is not None:
                raise ValueError("SUGGEST_FIX actions cannot include finish fields")

        elif action_type == "FINISH":
            if self.finish_decision is None:
                raise ValueError("finish_decision must be provided for FINISH actions")
            if self.file_path is not None or self.comment is not None or self.suggested_patch is not None or self.line is not None or self.severity is not None or self.category is not None:
                raise ValueError("FINISH actions cannot include file, comment, or fix fields")

        return self


class CodeReviewReward(BaseModel):
    """Normalized reward emitted by the code-review environment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Normalized reward score in [-1.0, 1.0].")
    reason: str = Field(default="", description="Human-readable explanation for the reward value.")

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason cannot be empty")
        return value
