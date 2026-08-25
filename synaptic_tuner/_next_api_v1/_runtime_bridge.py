"""Bytes-only bridge from core canonical SFT workloads to runtime contracts."""
from __future__ import annotations

from typing import Any, Mapping, Protocol

from Trainers.sft.v1_runtime.contracts import (
    BoundWorkloadV1, ClosedWorkloadDecoderV1, FixtureRuntimeBindingEvidenceV1,
)

from ._workloads import CanonicalWorkload, decode_sft_workload


def _fixture_evidence(workload: CanonicalWorkload) -> FixtureRuntimeBindingEvidenceV1:
    d = workload.document
    if d["provenance"]["class"] != "fixture_verified":
        raise ValueError("fixture runtime bridge requires fixture provenance")
    return FixtureRuntimeBindingEvidenceV1(
        workload_digest=workload.digest,
        engine_commit=d["code"]["engine_commit"],
        source_digest=d["code"]["source_digest"],
        dependency_lock_digest=d["runtime"]["dependency_lock_digest"],
        model_revision=d["model"]["revision"],
        tokenizer_revision=d["model"]["tokenizer_revision"],
        dataset_revision=d["dataset"]["revision"],
        target_profile_id=d["adapter"]["target_profile_id"],
        target_profile_digest=d["adapter"]["target_profile_digest"],
        target_modules=tuple(d["adapter"]["target_modules"]),
        fixture_manifest_digest=d["provenance"]["fixture_manifest_digest"],
    )

class _BoundFixtureWorkload:
    __slots__ = ("_workload", "_evidence")

    def __init__(self, workload: CanonicalWorkload, evidence: FixtureRuntimeBindingEvidenceV1) -> None:
        self._workload = workload
        self._evidence = evidence

    @property
    def canonical_bytes(self) -> bytes:
        return self._workload.canonical_bytes

    @property
    def digest(self) -> str:
        return self._workload.digest

    @property
    def document(self) -> Mapping[str, Any]:
        return self._workload.document

    @property
    def runtime_binding_evidence(self) -> FixtureRuntimeBindingEvidenceV1:
        return self._evidence


class FixtureCoreSFTWorkloadDecoderAdapter:
    __slots__ = ("_expected",)

    def __init__(self, expected: FixtureRuntimeBindingEvidenceV1) -> None:
        if not isinstance(expected, FixtureRuntimeBindingEvidenceV1):
            raise TypeError("expected must be FixtureRuntimeBindingEvidenceV1")
        self._expected = expected

    def decode_and_bind_sft_v1(self, canonical_bytes: bytes) -> BoundWorkloadV1:
        if not isinstance(canonical_bytes, bytes):
            raise TypeError("runtime bridge accepts canonical bytes only")
        workload = decode_sft_workload(canonical_bytes)
        actual = _fixture_evidence(workload)
        if actual != self._expected:
            raise ValueError("runtime binding evidence mismatch")
        return _BoundFixtureWorkload(workload, actual)


class ProductionCoreSFTWorkloadDecoderAdapter(ClosedWorkloadDecoderV1, Protocol):
    """Interface only; production resolver/provider phase supplies implementation."""


__all__ = [
    "FixtureCoreSFTWorkloadDecoderAdapter", "ProductionCoreSFTWorkloadDecoderAdapter",
    "FixtureRuntimeBindingEvidenceV1",
]
