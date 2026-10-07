#!/usr/bin/env bash
# Local proof of scripts/chat_model.py on the reviewed Modal inference image.
#
# Runs the one-prompt chat command inside the exact vLLM image digest that the
# Modal inference runtime is built from (`base_registry_reference` in
# tuner/execution/providers/modal/inference-runtime.lock.json), on a local GPU
# through Docker. One container, one attempt, one prompt. A local success is not
# provider shutdown proof and does not qualify the Modal path.
#
# Usage:  scripts/chat_model_local_docker.sh <attempt-name>
#
# The attempt name becomes /out/<attempt-name> inside the results volume. The
# launcher refuses if that directory already exists and never deletes anything;
# pick a new name per attempt (the chat command's one-shot claim is preserved).
#
# Environment overrides (all optional):
#   DOCKER                          docker client binary        (default: docker)
#   SYNAPTIC_ENGINE_MOUNT           engine checkout as the docker client sees it
#                                   (default: this script's parent directory)
#   SYNAPTIC_CHAT_CONFIGURATION     config path relative to the engine root
#                                   (default: examples/model_chat/smoke.json)
#   SYNAPTIC_CHAT_RESULTS_VOLUME    named volume mounted at /out
#                                   (default: synaptic-local-chat-results)
#   SYNAPTIC_CHAT_HF_CACHE_VOLUME   named volume for the Hub cache
#                                   (default: synaptic-local-hf-cache)
#   SYNAPTIC_CHAT_GPUS              value for --gpus              (default: all)
#   SYNAPTIC_CHAT_SHM_SIZE          value for --shm-size          (default: 2g)
#
# WSL2 with Docker Desktop: the in-WSL docker client may be unusable, so call
# the Windows client and hand it the WSL path of the engine in UNC form:
#   DOCKER='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
#   SYNAPTIC_ENGINE_MOUNT='\\wsl.localhost\<distro>\home\<user>\...\engine'
# Results and the Hub cache live in named volumes rather than bind mounts: the
# chat command requires its output directory to be owned by the running euid
# with mode 0700, which a Windows-drive bind mount cannot satisfy, while
# `mkdir -m 700` inside a named volume can. The engine mount is read-only; the
# command inserts its parent onto sys.path and `-B` suppresses bytecode writes.
# The chat command prints closed status codes only, and its process lease
# discards the serving child's stdout/stderr, so vLLM's own logs do not appear
# in the container output; see the README section on reading failures.
set -euo pipefail

usage() { echo "usage: $0 <attempt-name>" >&2; exit 2; }
[ "$#" -eq 1 ] || usage
attempt="$1"
case "$attempt" in
  ''|*[!A-Za-z0-9._-]*|.|..) echo "error: attempt name must match [A-Za-z0-9._-]+ and not be . or .." >&2; exit 2 ;;
esac

engine_root="$(cd "$(dirname "$0")/.." && pwd)"
docker_bin="${DOCKER:-docker}"
engine_mount="${SYNAPTIC_ENGINE_MOUNT:-$engine_root}"
configuration="${SYNAPTIC_CHAT_CONFIGURATION:-examples/model_chat/smoke.json}"
results_volume="${SYNAPTIC_CHAT_RESULTS_VOLUME:-synaptic-local-chat-results}"
hf_cache_volume="${SYNAPTIC_CHAT_HF_CACHE_VOLUME:-synaptic-local-hf-cache}"
gpus="${SYNAPTIC_CHAT_GPUS:-all}"
shm_size="${SYNAPTIC_CHAT_SHM_SIZE:-2g}"
lock_file="$engine_root/tuner/execution/providers/modal/inference-runtime.lock.json"

case "$configuration" in
  /*|*..*) echo "error: SYNAPTIC_CHAT_CONFIGURATION must be relative to the engine root without '..'" >&2; exit 2 ;;
esac
[ -f "$engine_root/$configuration" ] || { echo "error: configuration not found: $engine_root/$configuration" >&2; exit 2; }
[ -f "$engine_root/scripts/chat_model.py" ] || { echo "error: scripts/chat_model.py not found under $engine_root" >&2; exit 2; }

# The image digest is read from the reviewed lock at runtime so the local proof
# always uses the same image the Modal path is reviewed against.
image="$(python3 - "$lock_file" <<'PY'
import json, sys
try:
    with open(sys.argv[1], "rb") as handle:
        value = json.load(handle)["base_registry_reference"]
except Exception as error:
    print(f"error: cannot read base_registry_reference from {sys.argv[1]}: {error}", file=sys.stderr)
    raise SystemExit(1)
if not isinstance(value, str) or "@sha256:" not in value:
    print(f"error: base_registry_reference is not a digest reference: {value!r}", file=sys.stderr)
    raise SystemExit(1)
print(value)
PY
)" || { echo "error: inference runtime lock unreadable; refusing to guess an image" >&2; exit 1; }

# Everything attempt-specific reaches the container through -e, never through
# string interpolation into the container script.
container_script='
set -euo pipefail
out="/out/${SYNAPTIC_CHAT_ATTEMPT}"
if [ -e "$out" ]; then
  echo "refused: ${out} already exists; choose a new attempt name (nothing was deleted)" >&2
  exit 3
fi
mkdir -m 700 "$out"
python3 -B /engine/scripts/chat_model.py --configuration "/engine/${SYNAPTIC_CHAT_CONFIGURATION}" --check
status=0
python3 -B /engine/scripts/chat_model.py --configuration "/engine/${SYNAPTIC_CHAT_CONFIGURATION}" --output-directory "$out" || status=$?
if [ -f "${out}/chat-result.jsonl" ]; then
  echo "--- ${out}/chat-result.jsonl"
  cat "${out}/chat-result.jsonl"
else
  echo "--- ${out}/chat-result.jsonl was not written" >&2
fi
exit "$status"
'

command=(
  "$docker_bin" run --rm --gpus "$gpus" --shm-size "$shm_size"
  -v "${engine_mount}:/engine:ro"
  -v "${results_volume}:/out"
  -v "${hf_cache_volume}:/root/.cache/huggingface"
  -e "SYNAPTIC_CHAT_ATTEMPT=${attempt}"
  -e "SYNAPTIC_CHAT_CONFIGURATION=${configuration}"
  -w /root --entrypoint /bin/bash "$image" -lc "$container_script"
)

echo "engine root:   $engine_root"
echo "image (lock):  $image"
printf 'docker command:'; printf ' %q' "${command[@]}"; printf '\n'
exec "${command[@]}"
