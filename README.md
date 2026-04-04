# code-review-openenv

## Description

`code-review-openenv` is a clean-architecture scaffold for an OpenEnv-style
code review environment. It is intentionally minimal and contains only the
project structure, placeholder models, and stub environment components.

## Structure Overview

- `env/` contains the environment core, shared models, reward helpers, and state management.
- `env/tasks/` contains placeholder task definitions for easy, medium, and hard scenarios.
- `env/graders/` contains placeholder grader classes for scoring task states.
- `data/` contains repository and task data folders reserved for future use.
- Root files provide the entry point, container configuration, package dependencies, and environment metadata.
