#!/usr/bin/env bash
set -euo pipefail

APP_PORT="${SPACE_APP_PORT:-${APP_PORT:-7860}}"
BASE_MODEL="${BASE_MODEL:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LORA_LOCAL_DIR="${LORA_LOCAL_DIR:-/data/adapters/current}"
LORA_MODULE_NAME="${LORA_MODULE_NAME:-finetuned}"
LORA_MAX_RANK="${LORA_MAX_RANK:-64}"
BUCKET_SYNC_PYTHON="${BUCKET_SYNC_PYTHON:-/opt/bucket-sync-venv/bin/python}"

export HF_HOME="${HF_HOME:-/data/.huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/.cache/pip}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export USE_TORCH="${USE_TORCH:-1}"
export USE_TF="${USE_TF:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$(dirname "$LORA_LOCAL_DIR")"

if [[ -z "$BASE_MODEL" ]]; then
  echo "BASE_MODEL is required" >&2
  exit 2
fi

if [[ -n "${ADAPTER_BUCKET_URI:-}" ]]; then
  echo "[space] Syncing adapter from ${ADAPTER_BUCKET_URI} to ${LORA_LOCAL_DIR}"
  "${BUCKET_SYNC_PYTHON}" /home/user/app/sync_bucket_prefix.py \
    --source "${ADAPTER_BUCKET_URI}" \
    --dest "${LORA_LOCAL_DIR}"
fi

EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  while IFS= read -r arg; do
    [[ -n "$arg" ]] && EXTRA_ARGS+=("$arg")
  done < <(python3 - <<'PY'
import os
import shlex

for token in shlex.split(os.environ.get("VLLM_EXTRA_ARGS", "")):
    print(token)
PY
)
fi

CMD=(
  python3 -m vllm.entrypoints.openai.api_server
  --model "$BASE_MODEL"
  --host 0.0.0.0
  --port "$APP_PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enforce-eager
)

if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  CMD+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
fi

if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "${LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_JSON="$(
    python3 - <<'PY'
import json
import os
import shlex

raw = os.environ.get("LIMIT_MM_PER_PROMPT", "").strip()
if not raw:
    raise SystemExit(1)
if raw.startswith("{"):
    print(raw)
else:
    payload = {}
    for token in shlex.split(raw):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        payload[key] = int(value)
    print(json.dumps(payload))
PY
  )"
  CMD+=(--limit-mm-per-prompt "$LIMIT_MM_JSON")
fi

if [[ -d "$LORA_LOCAL_DIR" && -f "$LORA_LOCAL_DIR/adapter_config.json" ]]; then
  CMD+=(--enable-lora --max-lora-rank "$LORA_MAX_RANK" --lora-modules "${LORA_MODULE_NAME}=${LORA_LOCAL_DIR}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[space] Starting: ${CMD[*]}"
exec "${CMD[@]}"
