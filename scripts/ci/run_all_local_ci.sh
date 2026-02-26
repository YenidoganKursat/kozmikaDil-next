#!/usr/bin/env bash
set -euo pipefail

# Unified local CI runner.
# This script is intentionally explicit so that developers and CI use the same
# validation sequence with predictable parameters.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="quick"
JOBS="${JOBS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-quick}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-4}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "--mode must be 'quick' or 'full'" >&2
  exit 2
fi

echo "[ci] root: $ROOT_DIR"
echo "[ci] mode: $MODE jobs=$JOBS"

cd "$ROOT_DIR"

BUILD_DIR="build"
rm -rf "$BUILD_DIR"

echo "[ci] configure"
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPARK_BUILD_TESTS=ON

echo "[ci] build"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "[ci] ctest core"
ctest --test-dir "$BUILD_DIR" \
  --parallel "$JOBS" \
  --output-on-failure \
  --timeout 1800 \
  -E "sparkc_phase5_crosslang_primitives"

if [[ "$MODE" == "full" ]]; then
  echo "[ci] ctest stability replay"
  ctest --test-dir "$BUILD_DIR" \
    --output-on-failure \
    --timeout 1800 \
    --repeat until-fail:2 \
    -R "sparkc_typecheck_tests|sparkc_codegen_tests|sparkc_phase(5|6|7|8|9|10)_tests"

  echo "[ci] cross-language primitive correctness"
  python3 ./tests/phase5/primitives/crosslang_native_primitives_tests.py \
    --int-loops 25000 \
    --int-extreme-random 64 \
    --float-loops 500 \
    --float-python-loops 500 \
    --float-extreme-random 32 \
    --single-op-loops 120000 \
    --single-op-runs 2

  echo "[ci] phase10 portability reports"
  python3 ./scripts/phase10/platform_matrix.py \
    --preset market \
    --include-experimental \
    --include-embedded \
    --include-gpu-experimental \
    --include-gpu-planning \
    --json-out bench/results/phase10_platform_matrix_local.json

  python3 ./scripts/phase10/gpu_smoke_matrix.py \
    --backends all \
    --include-planning \
    --json-out bench/results/phase10_gpu_smoke_local.json
fi

echo "[ci] PASS"
