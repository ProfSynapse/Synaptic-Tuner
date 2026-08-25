from __future__ import annotations

import copy
import hashlib
import json

import pytest

from synaptic_tuner._next_api_v1._compiled_plan import (
    CompilerReceipt,
    FixtureCompiledTrainingPlan,
    ProductionCompilationUnavailable,
    ProductionCompiledTrainingPlan,
)
from synaptic_tuner._next_api_v1._runtime_bridge import (
    FixtureCoreSFTWorkloadDecoderAdapter,
    FixtureRuntimeBindingEvidenceV1,
)
from synaptic_tuner._next_api_v1._workloads import (
    MAX_WORKLOAD_BYTES,
    WORKLOAD_DIGEST_DOMAIN,
    RemoteDatasetBinding,
    canonical_sft_workload,
    decode_sft_workload,
    fixture_verified_for_tests,
)
from synaptic_tuner._next_api_v1.methods.sft import (
    ARTIFACT_ROLES,
    FixtureSFTInputs,
    SFTSpec,
    canonical_decimal,
    compile_fixture_sft,
    compile_live_sft,
)
from Trainers.sft.v1_runtime.contracts import (
    ARTIFACT_ROLES_V1,
    FixtureDecoderV1,
    ProductionDecoderV1,
    ProductionRuntimeBindingEvidenceV1,
    ProductionRuntimeCompositionUnavailable,
)
from Trainers.sft.v1_runtime.decoding import decode_verified_document_v1


H40 = "1" * 40
H64 = "2" * 64


def fixture_inputs() -> FixtureSFTInputs:
    provenance = fixture_verified_for_tests(
        fixture_manifest_digest="3" * 64,
        target_profile_id="smollm2-lora-v1",
        target_profile_digest="4" * 64,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    return FixtureSFTInputs(
        provenance=provenance,
        model_repository="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_revision=H40,
        tokenizer_revision="5" * 40,
        dataset=RemoteDatasetBinding(
            repository="claudesidian/synaptic-sft",
            revision="6" * 40,
            split="train",
            file_selector="data/train.jsonl",
        ),
        engine_commit="7" * 40,
        source_digest=H64,
        dependency_lock_digest="8" * 64,
    )


def compiled() -> FixtureCompiledTrainingPlan:
    return compile_fixture_sft(
        SFTSpec(
            max_steps=5,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate="0.0002",
            max_seq_length=2048,
            seed=3407,
            warmup_ratio="0.02",
            weight_decay="0",
            adapter_rank=64,
            adapter_alpha=128,
        ),
        fixture_inputs(),
    )


def test_offline_successor_is_deterministic_and_domain_digested() -> None:
    first = compiled()
    second = compiled()
    assert first.workload.canonical_bytes == second.workload.canonical_bytes
    assert first.fingerprint == second.fingerprint
    assert first.workload.digest == hashlib.sha256(
        WORKLOAD_DIGEST_DOMAIN + first.workload.canonical_bytes
    ).hexdigest()
    assert first.workload.canonical_bytes == json.dumps(
        first.workload.document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_fixture_bridge_returns_runtime_nominal_evidence() -> None:
    plan = compiled()
    document = plan.workload.document
    expected = FixtureRuntimeBindingEvidenceV1(
        workload_digest=plan.workload.digest,
        engine_commit=document["code"]["engine_commit"],
        source_digest=document["code"]["source_digest"],
        dependency_lock_digest=document["runtime"]["dependency_lock_digest"],
        model_revision=document["model"]["revision"],
        tokenizer_revision=document["model"]["tokenizer_revision"],
        dataset_revision=document["dataset"]["revision"],
        target_profile_id=document["adapter"]["target_profile_id"],
        target_profile_digest=document["adapter"]["target_profile_digest"],
        target_modules=tuple(document["adapter"]["target_modules"]),
        fixture_manifest_digest=document["provenance"]["fixture_manifest_digest"],
    )
    decoder = FixtureDecoderV1(FixtureCoreSFTWorkloadDecoderAdapter(expected))
    bound, decoded = decode_verified_document_v1(
        plan.workload.canonical_bytes, decoder, FixtureRuntimeBindingEvidenceV1
    )
    assert type(bound.runtime_binding_evidence) is FixtureRuntimeBindingEvidenceV1
    assert bound.canonical_bytes == plan.workload.canonical_bytes
    assert decoded == plan.workload.document


def test_fixture_evidence_mismatch_is_closed() -> None:
    plan = compiled()
    document = plan.workload.document
    wrong = FixtureRuntimeBindingEvidenceV1(
        plan.workload.digest, document["code"]["engine_commit"], "9" * 64,
        document["runtime"]["dependency_lock_digest"], document["model"]["revision"],
        document["model"]["tokenizer_revision"], document["dataset"]["revision"],
        document["adapter"]["target_profile_id"], document["adapter"]["target_profile_digest"],
        tuple(document["adapter"]["target_modules"]),
        document["provenance"]["fixture_manifest_digest"],
    )
    with pytest.raises(ValueError, match="binding evidence mismatch"):
        FixtureCoreSFTWorkloadDecoderAdapter(wrong).decode_and_bind_sft_v1(
            plan.workload.canonical_bytes
        )


def test_production_composition_is_unavailable_offline() -> None:
    with pytest.raises(ProductionCompilationUnavailable):
        ProductionCompiledTrainingPlan()
    with pytest.raises(ProductionCompilationUnavailable):
        compile_live_sft(SFTSpec(max_steps=1), object())
    with pytest.raises(ProductionRuntimeCompositionUnavailable):
        ProductionRuntimeBindingEvidenceV1()
    with pytest.raises(ProductionRuntimeCompositionUnavailable):
        ProductionDecoderV1(object())


def test_artifact_contract_is_exactly_shared() -> None:
    plan = compiled()
    assert ARTIFACT_ROLES == ARTIFACT_ROLES_V1
    assert tuple(plan.workload.document["artifacts"]["required_roles"]) == ARTIFACT_ROLES_V1


@pytest.mark.parametrize("value", [1, "1", "1.0", "1e0", 1.0])
def test_semantically_equal_decimals_have_one_canonical_value(value: object) -> None:
    assert canonical_decimal(value, "value", positive=True) == "1"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "-1"])
def test_invalid_decimal_inputs_are_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_decimal(value, "value")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xef\xbb\xbf{}",
        b'{"x":1}\n',
        b'{"x":NaN}',
        b'{"x":1,"x":2}',
        b"[]",
        b"{\xff}",
    ],
)
def test_closed_decoder_rejects_noncanonical_or_invalid_bytes(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_sft_workload(payload)


def test_decoder_rejects_oversize_before_json_acceptance() -> None:
    with pytest.raises(ValueError, match="1..65536"):
        decode_sft_workload(b" " * (MAX_WORKLOAD_BYTES + 1))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("entrypoint",), "arbitrary"),
        (("objective", "packing"), True),
        (("model", "trust_remote_code"), True),
        (("dataset", "file_selector"), "../private.jsonl"),
        (("adapter", "target_modules"), []),
        (("optimization", "dtype"), "fp16"),
        (("logging", "external_reporting"), "wandb"),
        (("artifacts", "required_roles"), ["final_adapter"]),
    ],
)
def test_fixed_and_security_fields_fail_closed(path: tuple[str, ...], value: object) -> None:
    document = copy.deepcopy(dict(compiled().workload.document))
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        canonical_sft_workload(document)


def test_unknown_and_missing_fields_fail_closed() -> None:
    document = copy.deepcopy(dict(compiled().workload.document))
    document["argv"] = ["--unsafe"]
    with pytest.raises(ValueError):
        canonical_sft_workload(document)
    document.pop("argv")
    document.pop("runtime")
    with pytest.raises(ValueError):
        canonical_sft_workload(document)


def test_document_mutation_cannot_change_authoritative_workload() -> None:
    workload = compiled().workload
    original = workload.canonical_bytes
    view = workload.document
    view["model"]["repository"] = "evil/repo"
    view["adapter"]["target_modules"].append("evil")
    assert workload.canonical_bytes == original
    assert workload.document["model"]["repository"] != "evil/repo"
    assert "evil" not in workload.document["adapter"]["target_modules"]


def test_compiler_authority_types_cannot_be_forged() -> None:
    plan = compiled()
    with pytest.raises(TypeError):
        CompilerReceipt(b"{}", b"{}", plan.workload, b"{}", ())
    with pytest.raises(TypeError):
        FixtureCompiledTrainingPlan(plan.workload, plan.receipt)

