"""Read-only BackendClient for complete, host-verified native MLX responses.

This adapter loads no model and authenticates no execution. The host validates
the original selection, custody and manifest before supplying ordered records.
It exposes the existing evaluator protocol, not a permissive native launcher.
"""
from copy import deepcopy
import hashlib
import json

from Evaluator.comparison_basis import GENERATION_KEYS
from Evaluator.protocols import BackendError, BackendResponse


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


class NativeMLXRetainedClient:
    def __init__(self, records, *, generation, profile):
        self._records = deepcopy(tuple(records))
        self._generation = deepcopy(generation)
        self._profile = deepcopy(profile)
        self.records_consumed = 0

    def comparison_configuration(self):
        g, p = self._generation, self._profile
        generation = {name: None for name in GENERATION_KEYS}
        generation.update(temperature=0, max_tokens=g["max_completion_tokens"], seed=g["seed"],
            chat_template_kwargs=g["chat_template_kwargs"])
        # These finite controls are outside the ordinary generation-key set.
        # Bind them in the producer revision rather than silently comparing
        # different stop/context/tokenizer policies as an identical backend.
        policy = {"mode": g["mode"], "stop_token_ids": g["stop_token_ids"], "use_fast": g["use_fast"],
            "max_prompt_tokens": g["max_prompt_tokens"], "maximum_context_tokens": p["maximum_context_tokens"],
            "source_digest": p["source_digest"], "runtime_digest": p["runtime_digest"]}
        return {"backend": "native_mlx_retained", "revision": "native-greedy/v1/" +
            hashlib.sha256(_canonical(policy)).hexdigest(), "generation": generation}

    def chat(self, messages):
        if self.records_consumed >= len(self._records):
            raise BackendError("native_retained_records_exhausted")
        row = self._records[self.records_consumed]
        if row["messages_sha256"] != hashlib.sha256(_canonical(list(messages))).hexdigest():
            raise BackendError("native_retained_prompt_mismatch")
        self.records_consumed += 1
        return BackendResponse(message={"role": "assistant", "content": row["text"]},
            raw={"finish_reason": row["finish_reason"], "prompt_tokens": len(row["prompt_tokens"]),
                 "completion_tokens": len(row["completion_tokens"])}, latency_s=row["latency_seconds"])
