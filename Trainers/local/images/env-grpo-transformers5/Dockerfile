ARG BASE_IMAGE=unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39
FROM ${BASE_IMAGE}

ENV TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRL_EXPERIMENTAL_SILENCE=1 \
    TERM=xterm

RUN python3 -m pip uninstall -y vllm vllm-ascend vllm_ascend || true && \
    python3 -m pip install --no-cache-dir --upgrade \
      "huggingface_hub>=1.0,<2" \
      "transformers==5.5.0" \
      "trl==0.29.0" \
      weave \
      asciimatics && \
    python3 - <<'PY'
import inspect
import json
import huggingface_hub
import transformers
import trl
from trl import GRPOTrainer

print(json.dumps({
    "huggingface_hub": huggingface_hub.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "has_rollout_func": "rollout_func" in inspect.signature(GRPOTrainer.__init__).parameters,
}))
PY
