#!/usr/bin/env bash
# Run one shard of the full pytest suite exactly as .github/workflows/ci.yml does.
#
#   scripts/ci_pytest_shard.sh <core|modal-inference-a-l|modal-inference-m-z> [extra pytest args]
#
# The three shards together cover every test file: the Modal inference provider
# tests dominate wall time and are split alphabetically into two shards; "core"
# is everything else. Extra arguments are passed to pytest (e.g. --basetemp).
set -euo pipefail
shopt -s nullglob

shard="${1:?usage: ci_pytest_shard.sh <core|modal-inference-a-l|modal-inference-m-z> [pytest args]}"
shift

inference_glob='tests/execution/providers/test_modal_inference_*.py'
inference=(tests/execution/providers/test_modal_inference_*.py)
first=()
second=()
for test_file in "${inference[@]}"; do
  name="${test_file#tests/execution/providers/test_modal_inference_}"
  if [[ "$name" < "m" ]]; then first+=("$test_file"); else second+=("$test_file"); fi
done
if (( ${#first[@]} == 0 || ${#second[@]} == 0 )); then
  echo "Modal inference shard selection is empty" >&2
  exit 1
fi

case "$shard" in
  core) selection=(--ignore-glob="$inference_glob" tests) ;;
  modal-inference-a-l) selection=("${first[@]}") ;;
  modal-inference-m-z) selection=("${second[@]}") ;;
  *) echo "Unknown shard: $shard" >&2; exit 2 ;;
esac

exec python -B -m pytest -p no:cacheprovider -q -o addopts="" \
  --tb=short -rfE --continue-on-collection-errors --durations=20 \
  --junitxml="pytest-$shard.xml" "$@" "${selection[@]}"
