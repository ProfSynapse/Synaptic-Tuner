"""Internal config-driven ordinary Jobs selection, distinct from protected smoke."""

from dataclasses import dataclass, replace, field
from types import MappingProxyType
import hashlib
import json
import re

from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.runtime_profiles import RuntimeProfile
from tuner.training.contracts import ResourceSpec


EXACT_ROLES = frozenset({"final_model", "tokenizer", "training_lineage",
                         "training_metrics", "workload_record"})
MAX_ARTIFACT_BYTES = 192 * 1024 * 1024
MAX_ARTIFACT_TOTAL_BYTES = 256 * 1024 * 1024
# Current baseline native prepared-input verifier bound; broader input parity
# requires the independently reviewed runtime/verification bound update.
MAX_INPUT_BYTES = 2 * 1024 * 1024


class HFJobsError(ValueError):
    """Closed non-secret local/provider boundary diagnostic."""

    def __init__(self, code):
        if code not in {"input_invalid", "binding_mismatch", "stage_unavailable",
                        "submit_indeterminate", "read_unavailable", "artifact_invalid",
                        "deadline_exceeded", "capability_unavailable"}:
            raise ValueError("unknown HF diagnostic")
        self.code = code
        super().__init__(code)


def document_bytes(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class HFJobsSelection:
    """Explicit host config; no bucket, account, image or hardware defaults.

    Runtime facts are retained as immutable original inventory bytes. The
    host API credential and remote job credential are separate named refs.
    No credential values enter this value or its digest.
    """

    provider: ProviderRef
    scope: ExecutionScopeV1
    runtime_profile: RuntimeProfile
    inventory_bytes: bytes = field(repr=False)
    bucket_id: str
    source_prefix: str
    run_prefix: str
    flavor: str
    timeout_seconds: int
    host_token: SecretRef
    job_token: SecretRef
    quote_digest: str
    resources: ResourceSpec

    def __post_init__(self):
        if (type(self.provider) is not ProviderRef
                or self.provider.provider_id != "hf_jobs"
                or type(self.resources) is not ResourceSpec
                or type(self.scope) is not ExecutionScopeV1
                or type(self.runtime_profile) is not RuntimeProfile
                or type(self.inventory_bytes) is not bytes
                or type(self.host_token) is not SecretRef
                or type(self.job_token) is not SecretRef):
            raise HFJobsError("input_invalid")
        profile = replace(self.runtime_profile,
            model_revisions=MappingProxyType({name: tuple(values) for name, values in self.runtime_profile.model_revisions.items()}),
            runtime_facts=MappingProxyType(dict(self.runtime_profile.runtime_facts)))
        object.__setattr__(self, "runtime_profile", profile)
        if (not self.inventory_bytes or len(self.inventory_bytes) > 1024 * 1024
                or hashlib.sha256(self.inventory_bytes).hexdigest()
                != profile.inventory_sha256.removeprefix("sha256:")):
            raise HFJobsError("binding_mismatch")
        if (type(self.timeout_seconds) is not int
                or not 1 <= self.timeout_seconds <= 86400
                or self.resources.timeout_seconds != self.timeout_seconds
                or type(self.quote_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", self.quote_digest) is None):
            raise HFJobsError("input_invalid")
        if (type(self.bucket_id) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",
                                self.bucket_id) is None
                or type(self.flavor) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.flavor) is None):
            raise HFJobsError("input_invalid")
        for prefix in (self.source_prefix, self.run_prefix):
            if (type(prefix) is not str or not 1 <= len(prefix) <= 256
                    or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", part) is None
                           or part in {".", ".."} for part in prefix.split("/"))):
                raise HFJobsError("input_invalid")
        if (self.source_prefix == self.run_prefix
                or self.source_prefix.startswith(self.run_prefix + "/")
                or self.run_prefix.startswith(self.source_prefix + "/")):
            raise HFJobsError("input_invalid")

    def projection(self):
        p = self.runtime_profile
        return {"schema_version": "synaptic-hf-jobs-selection/v1",
                "provider": self.provider.to_dict(), "scope": self.scope.to_dict(),
                "runtime_profile": p.name, "runtime_profile_sha256": p.profile_sha256,
                "compatibility": {"models": {name: list(values) for name, values in p.model_revisions.items()},
                                  "methods": list(p.methods)},
                "image": p.image, "inventory_sha256": hashlib.sha256(self.inventory_bytes).hexdigest(),
                "bucket_id": self.bucket_id, "source_prefix": self.source_prefix,
                "run_prefix": self.run_prefix, "flavor": self.flavor,
                "timeout_seconds": self.timeout_seconds,
                "resources": {"accelerator": self.resources.accelerator, "accelerator_count": self.resources.accelerator_count, "timeout_seconds": self.resources.timeout_seconds},
                "host_token": self.host_token.to_dict(), "job_token": self.job_token.to_dict(),
                "job_secret_key": "HF_TOKEN", "model_preparation_token": False,
                "quote_digest": self.quote_digest}

    @property
    def digest(self):
        return hashlib.sha256(document_bytes(self.projection())).hexdigest()
