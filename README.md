# code-review-openenv

## Description

`code-review-openenv` is a deterministic OpenEnv benchmark for real-world code
review. It simulates tasks humans actually do: reviewing diffs, finding logic
bugs, spotting cross-file regressions, and proposing concrete fixes.

## Environment Interface

The environment exposes typed Pydantic models for:

- Observation: `CodeReviewObservation`
- Action: `CodeReviewAction`
- Reward: `CodeReviewReward`

The main API is:

- `reset(task_id)` returns the initial observation.
- `step(action)` returns `(observation, reward, done, info)`.
- `state()` returns the current internal state.

## Tasks

Three deterministic tasks are included:

- `easy`: a single-file divide-by-zero bug in `src/metrics.py`
- `medium`: a cache-boundary logic bug in `src/cache.py`
- `hard`: a cross-file auth/logging security issue in `src/auth.py` and `src/logger.py`

Each task has a programmatic grader that returns a normalized score in the range
`0.0` to `1.0`.

## Reward

The environment uses dense step rewards plus a final blended score:

- partial progress is rewarded during the episode
- repeated actions, invalid actions, and weak signals are penalized
- the final reward blends the step trajectory with the grader score

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Environment variables used by the inference runner:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

## Usage

Run the validator:

```powershell
python validate_env.py
python -m pytest tests/test_env_validation.py -v
```

Run the baseline inference script:

```powershell
python inference.py
```

## Baseline

The baseline runner is deterministic when the same model endpoint is used.
Scores are reproducible across runs when the environment variables and model
endpoint remain unchanged.

Current local fallback run with no API credentials configured:

- easy: 0.00
- medium: 0.00
- hard: 0.00

Set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` or `OPENAI_API_KEY` to obtain
real model-backed baseline scores.
