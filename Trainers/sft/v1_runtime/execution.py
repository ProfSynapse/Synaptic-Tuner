"""Fixed dispatcher for the isolated SFT v1 runtime."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .contracts import (
    ARTIFACT_ROLES_V1, ENTRYPOINT_V1,
    FixtureArtifactWriterV1, FixtureDecoderV1, FixtureHubAccessV1,
    ProductionArtifactWriterV1, ProductionDecoderV1, ProductionHubAccessV1,
    RuntimeContractError,
)
from .decoding import decode_verified_document_v1
from .loading import (
    apply_adapter_v1, load_model_v1, load_remote_dataset_v1, load_tokenizer_v1,
)
from .preprocessing import materialize_conversations_v1


@dataclass(frozen=True, slots=True)
class SFTResultV1:
    workload_digest: str
    artifact_slot_ref: str
    artifact_roles: tuple[str, ...]
    metrics: tuple[tuple[str, int | float | str | bool | None], ...]
    entrypoint: str = ENTRYPOINT_V1


def _metric_scalars(values: Mapping[str, Any]) -> dict[str, int | float | str | bool | None]:
    result: dict[str, int | float | str | bool | None] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise RuntimeContractError("trainer metric key is invalid")
        if value is None or isinstance(value, (str, bool, int)):
            result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
        else:
            raise RuntimeContractError("trainer metric is not a finite scalar")
    return result


def _run_fixture_sft_v1(
    canonical_workload_bytes: bytes,
    *,
    decoder: FixtureDecoderV1,
    artifacts: FixtureArtifactWriterV1,
    hub_access: FixtureHubAccessV1,
) -> SFTResultV1:
    bound, document = decode_verified_document_v1(canonical_workload_bytes, decoder)
    if artifacts.workload_digest != bound.digest:
        raise RuntimeContractError("artifact slot is not bound to workload digest")
    if not isinstance(artifacts.artifact_slot_ref, str) or not artifacts.artifact_slot_ref:
        raise RuntimeContractError("artifact slot reference is invalid")
    artifacts.assert_link_safe()

    model, model_identity = load_model_v1(document["model"], document["objective"], hub_access)
    tokenizer, tokenizer_identity = load_tokenizer_v1(document["model"], hub_access)
    rows, dataset_identity = load_remote_dataset_v1(document["dataset"], hub_access)
    prepared = materialize_conversations_v1(
        rows,
        tokenizer=tokenizer,
        max_seq_length=document["objective"]["max_seq_length"],
        loss_scope=document["objective"]["loss_scope"],
        masking_contract=document["objective"]["masking_contract"],
    )
    model, target_proof = apply_adapter_v1(model, document["adapter"], document["optimization"])

    artifacts.write_bytes("workload_record", bound.canonical_bytes)

    def _train(workspace: Path):
        if not isinstance(workspace, Path) or not workspace.is_absolute():
            raise RuntimeContractError("artifact capability returned an invalid workspace")
        from datasets import Dataset
        from transformers import Trainer, default_data_collator
        from trl import SFTConfig

        dataset = Dataset.from_list([{
            "input_ids": list(row.input_ids),
            "attention_mask": list(row.attention_mask),
            "labels": list(row.labels),
        } for row in prepared])
        optimization = document["optimization"]
        duration = optimization["duration"]
        checkpoints = document["checkpoints"]
        args = SFTConfig(
            output_dir=str(workspace),
            per_device_train_batch_size=optimization["batch_size"],
            gradient_accumulation_steps=optimization["gradient_accumulation_steps"],
            learning_rate=float(Decimal(optimization["learning_rate"])),
            num_train_epochs=float(Decimal(duration.get("epochs", "1"))),
            max_steps=duration.get("max_steps", -1),
            max_seq_length=document["objective"]["max_seq_length"],
            packing=False,
            gradient_checkpointing=True,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            max_grad_norm=float(Decimal(optimization["max_grad_norm"])),
            warmup_ratio=float(Decimal(optimization["warmup_ratio"])),
            weight_decay=float(Decimal(optimization["weight_decay"])),
            logging_steps=document["logging"]["structured_metrics_steps"],
            save_strategy="no" if checkpoints["strategy"] == "none" else "steps",
            save_steps=checkpoints["steps"] or 500,
            save_total_limit=checkpoints["limit"] or None,
            bf16=True,
            fp16=False,
            report_to="none",
            seed=optimization["seed"],
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )
        return trainer.train()

    train_result = artifacts.in_training_workspace(_train)
    metrics = _metric_scalars(dict(train_result.metrics))
    artifacts.write_json("training_lineage", {
        "entrypoint": ENTRYPOINT_V1,
        "workload_digest": bound.digest,
        "runtime_binding_evidence": asdict(bound.runtime_binding_evidence),
        "model_identity": model_identity,
        "tokenizer_identity": tokenizer_identity,
        "dataset_identity": dataset_identity,
        "target_module_proof": target_proof,
    })
    artifacts.write_json("training_metrics", metrics)
    artifacts.write_directory(
        "final_adapter",
        lambda destination: model.save_pretrained(destination, safe_serialization=True),
    )
    artifacts.write_directory("tokenizer", tokenizer.save_pretrained)
    roles = artifacts.finish(ARTIFACT_ROLES_V1)
    if roles != ARTIFACT_ROLES_V1:
        raise RuntimeContractError("artifact writer did not prove exact-one cardinality")
    return SFTResultV1(
        workload_digest=bound.digest,
        artifact_slot_ref=artifacts.artifact_slot_ref,
        artifact_roles=roles,
        metrics=tuple(sorted(metrics.items())),
    )


def dispatch_sft_v1(
    canonical_workload_bytes: bytes,
    *,
    decoder: ProductionDecoderV1,
    artifacts: ProductionArtifactWriterV1,
    hub_access: ProductionHubAccessV1,
) -> SFTResultV1:
    """Fail closed until trusted production composition is implemented."""
    del canonical_workload_bytes, decoder, artifacts, hub_access
    from .contracts import ProductionRuntimeCompositionUnavailable

    raise ProductionRuntimeCompositionUnavailable(
        "production SFT runtime composition is not installed"
    )

def dispatch_fixture_sft_v1(
    canonical_workload_bytes: bytes,
    *,
    decoder: FixtureDecoderV1,
    artifacts: FixtureArtifactWriterV1,
    hub_access: FixtureHubAccessV1,
) -> SFTResultV1:
    """Offline fixture-only dispatcher; fixture ports cannot enter production."""
    if type(decoder) is not FixtureDecoderV1:
        raise TypeError("fixture dispatch requires FixtureDecoderV1")
    if type(artifacts) is not FixtureArtifactWriterV1:
        raise TypeError("fixture dispatch requires FixtureArtifactWriterV1")
    if type(hub_access) is not FixtureHubAccessV1:
        raise TypeError("fixture dispatch requires FixtureHubAccessV1")
    return _run_fixture_sft_v1(
        canonical_workload_bytes,
        decoder=decoder,
        artifacts=artifacts,
        hub_access=hub_access,
    )