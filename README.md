---
title: OpenEnv Code Review Benchmark
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: inference.py
pinned: false
---

# OpenEnv CodeReview Benchmark

**A realistic, multi-step environment for evaluating AI code review agents under production-like constraints.**

## Motivation

Code review is not a single-step classification task. Strong reviewers must inspect diffs, open the right files, track dependencies, validate hypotheses, and propose actionable fixes. Most benchmark setups flatten this process into one-shot judgments, which fails to measure real review behavior.

This project closes that gap by modeling code review as an iterative decision process with structured actions, partial observability, deterministic grading, and exploit-resistant rewards. The result is a benchmark that measures not just whether an agent can guess an answer, but whether it can review code like an engineer.

## Key Features

- Multi-step reasoning environment with explicit review actions
- Cross-file dependency awareness and multi-hop bug localization
- Patch validation engine that checks whether unified diffs apply correctly
- Deterministic task graders with structured component-level scoring
- Dense trajectory rewards plus grader-aligned final reward
- Anti-exploit scoring penalties for spam, false positives, and inefficient behavior
- Precision-over-recall evaluation design suitable for high-stakes code review

## Environment Overview

Each episode simulates a pull request review lifecycle:

1. Agent receives PR context (title, description, diffs, visible files, history).
2. Agent chooses a structured action (`READ_FILE`, `COMMENT`, `SUGGEST_FIX`, `FINISH`).
3. Environment updates state deterministically and returns reward + feedback.
4. Deterministic grader evaluates review quality when the agent finishes (or on auto-finalization at step limit).

This mirrors real engineering workflows where reviewers progressively gather evidence before deciding.

## Action Space

The action schema is strictly typed and validated:

- `READ_FILE`: open a file for additional context
- `COMMENT`: submit a structured finding (`file_path`, `line`, `severity`, `category`, `comment`)
- `SUGGEST_FIX`: propose a patch as unified diff; patch must apply to current snapshot
- `FINISH`: end the review with decision (`APPROVE` or `REQUEST_CHANGES`) and optional summary

## Observation Space

Agent observations include:

- PR metadata: title and description
- File-level diffs: `diff_by_file`
- Controlled file visibility: only opened files are exposed in `file_contents`
- Interaction trace: compact history of prior actions
- Runtime signals: step counters, remaining budget, last action status, last error

This design enforces realistic information-gathering rather than full-repo leakage.

## Tasks

| Task | Difficulty | Description | Skills Tested |
|---|---|---|---|
| Easy Bug | Easy | Surface-level correctness defect in a utility path | Basic issue detection, targeted file reading, minimal fix proposal |
| Medium Logic | Medium | Semantic boundary-condition logic error with regression implications | Logical reasoning, contextual validation, consistent review judgment |
| Hard Cross-File | Hard | Dependency-driven failure spanning multiple files and side effects | Multi-hop reasoning across files to identify root cause and propose safe corrective action |

## Reward System

The reward is two-layer by design:

- **Dense step rewards** guide behavior during exploration
	- positive signal for relevant reads, meaningful comments, valid patches
	- negative signal for irrelevant reads, spam, invalid actions, repeated patterns
- **Final aligned reward** ties trajectory to objective quality via grader score

Final reward formula:

$$
	ext{final} = 0.3 \times \text{step\_rewards} + 0.7 \times (2 \times \text{score} - 1)
$$

Why this works:

- step rewards provide stable optimization signal during long-horizon interaction
- final grader alignment preserves benchmark correctness and prevents reward hacking

## Evaluation and Grading

Grading is deterministic, componentized, and auditable.

- deterministic task-specific graders produce `score \in [0,1]`
- structured comment checks verify category relevance and file relevance
- patch scoring verifies correctness through deterministic patch application
- penalties reduce inflated scores from false positives, irrelevant comments, and excessive action usage
- score breakdowns expose sub-scores and penalties for traceability

## Baseline Performance

Use this section to publish your official model-backed baseline:

- Easy: X.XX
- Medium: X.XX
- Hard: X.XX
- Average: X.XX

Note: if no API credentials are configured, the runner uses safe fallback behavior and should not be treated as official baseline quality.

## Setup and Usage

```bash
pip install -r requirements.txt
python inference.py
```

Recommended validation before submission:

```bash
python validate_env.py
python -m pytest tests/test_env_validation.py -v
```

Required environment variables for model-backed inference:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

## Docker Support

The project is fully containerized and intended for reproducible evaluation.

```bash
docker build -t code-review-openenv .
docker run --rm code-review-openenv
```

## Design Principles

- **Precision over recall**: reward high-confidence, evidence-backed review signals
- **Exploit resistance**: penalize low-quality shortcuts and spam interactions
- **Deterministic evaluation**: identical trajectories produce identical outcomes
- **Real-world fidelity**: model authentic engineering review workflow, not synthetic games

## Why This Matters

Industry-grade AI reviewers must do more than label bugs. They need to reason over evolving context, prioritize relevant evidence, and submit valid fixes with minimal noise. This benchmark provides a rigorous testbed for building and evaluating such systems.

It supports research and engineering teams who need reproducible, behavior-sensitive evaluation for code review agents deployed in real software pipelines.

## Conclusion

OpenEnv CodeReview Benchmark is a deterministic, high-fidelity environment for measuring the capabilities that matter in practical code review: structured reasoning, cross-file diagnosis, and reliable fix generation. It is designed to be both scientifically evaluable and operationally useful.
