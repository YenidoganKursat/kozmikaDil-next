# Repository Hardening Guide

## Scope

This document defines the repository baseline for:

- maintainability (clean structure, clear ownership, reproducible commands),
- correctness (core test gates and cross-language numeric verification),
- portability (CPU/GPU/OS reporting and smoke gates),
- release confidence (CI/CD jobs aligned with local reproducible commands).

## Baseline Principles

1. Deterministic commands first.
2. Test and benchmark scripts must expose explicit CLI knobs.
3. CI logic must call repository scripts, not duplicate hidden shell logic.
4. Generated outputs and local artifacts must never pollute version control.
5. Strict precision/correctness policy remains mandatory.

## Required Local Validation

Use:

```bash
bash scripts/ci/run_all_local_ci.sh --mode full
```

This runs:

1. CMake configure/build with tests enabled.
2. Core CTest suite.
3. Stability replay for critical test groups.
4. Cross-language primitive correctness gate.
5. Phase10 portability report generation.

For fast iteration:

```bash
bash scripts/ci/run_all_local_ci.sh --mode quick
```

## CI/CD Model

Primary workflow:

- `.github/workflows/full-validation.yml`

Jobs:

1. `linux-full-validation`: full correctness and portability path.
2. `host-portability-smoke`: script-level portability smoke on Linux/macOS/Windows.

Existing workflows remain valid; this guide adds a consolidated, reproducible entrypoint so local and CI runs stay aligned.

## Repository Hygiene

Ignored artifacts include build directories, caches, profiling artifacts, coverage/runtime logs, and benchmark outputs under `bench/results`.

Before major refactors:

1. Run local quick validation.
2. Keep generated outputs outside tracked source.
3. Keep script and docs updates in the same change-set as behavior changes.

