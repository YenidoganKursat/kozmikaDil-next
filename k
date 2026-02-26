#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a CANDIDATES=(
  "${SCRIPT_DIR}/build/compiler/sparkc"
  "${SCRIPT_DIR}/build_local_full/compiler/sparkc"
  "${SCRIPT_DIR}/build_local/compiler/sparkc"
)

SELECTED=""
for candidate in "${CANDIDATES[@]}"; do
  if [[ -x "${candidate}" ]]; then
    if [[ -z "${SELECTED}" || "${candidate}" -nt "${SELECTED}" ]]; then
      SELECTED="${candidate}"
    fi
  fi
done

if [[ -z "${SELECTED}" ]]; then
  echo "No sparkc binary found. Build first (e.g. cmake --build build -j)." >&2
  exit 1
fi

exec "${SELECTED}" "$@"
