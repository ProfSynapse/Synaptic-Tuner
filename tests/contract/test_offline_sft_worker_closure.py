from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tuner.runtime.offline_sft_worker import (
    OFFLINE_SFT_CLOSURE_REF,
    OFFLINE_SFT_CLOSURE_SCHEMA,
    OFFLINE_SFT_ENTRYPOINT,
    OFFLINE_SFT_TRAINER_ENTRYPOINT,
    OWNED_MODULE_PREFIXES,
    closure_digest,
)


_ROOT = Path(__file__).parents[2]
_MANIFEST = (
    _ROOT / "tuner" / "runtime" / "manifests" / "offline-sft-worker-v1.json"
)
_MEMBERS = (
    "Trainers/sft/configs/config.yaml",
    "Trainers/sft/configs/config_loader.py",
    "Trainers/sft/runtime_v1.py",
    "Trainers/sft/src/data_loader.py",
    "Trainers/sft/src/model_loader.py",
    "Trainers/sft/src/preprocessing.py",
    "Trainers/sft/src/training_callbacks.py",
    "Trainers/sft/train_sft.py",
    "Trainers/shared/__init__.py",
    "Trainers/shared/callbacks/__init__.py",
    "Trainers/shared/callbacks/base.py",
    "Trainers/shared/callbacks/checkpoints.py",
    "Trainers/shared/callbacks/health_checks.py",
    "Trainers/shared/callbacks/log_suppression.py",
    "Trainers/shared/callbacks/lr_schedules.py",
    "schemas/synaptic-execution-source-v1.schema.json",
    "schemas/synaptic-sft-workload-v1.schema.json",
    "shared/cloud_artifacts.py",
    "shared/env_bootstrap.py",
    "shared/experiment_tracking/__init__.py",
    "shared/experiment_tracking/experiment.py",
    "shared/experiment_tracking/experiment_spec.py",
    "shared/experiment_tracking/lineage_enrichment.py",
    "shared/experiment_tracking/schema.py",
    "shared/sft_preprocessing.py",
    "shared/training_capacity.py",
    "shared/training_utils.py",
    "shared/utilities/__init__.py",
    "shared/utilities/env.py",
    "shared/utilities/paths.py",
    "shared/utilities/unique_ids.py",
    "shared/utilities/yaml_loader.py",
    "synaptic_tuner/__init__.py",
    "synaptic_tuner/_version.py",
    "synaptic_tuner/api/__init__.py",
    "synaptic_tuner/api/v1/__init__.py",
    "synaptic_tuner/api/v1/_contract.py",
    "synaptic_tuner/api/v1/execution.py",
    "synaptic_tuner/api/v1/sources.py",
    "synaptic_tuner/api/v1/training.py",
    "tuner/__init__.py",
    "tuner/cloud/__init__.py",
    "tuner/cloud/hardware_planner.py",
    "tuner/execution/__init__.py",
    "tuner/execution/contracts.py",
    "tuner/execution/evidence.py",
    "tuner/execution/lifecycle.py",
    "tuner/execution/registry.py",
    "tuner/execution/service.py",
    "tuner/project/__init__.py",
    "tuner/project/config_layers.py",
    "tuner/project/context.py",
    "tuner/project/errors.py",
    "tuner/project/execution_source.py",
    "tuner/project/git_verification.py",
    "tuner/project/manifest.py",
    "tuner/project/path_refs.py",
    "tuner/project/secrets.py",
    "tuner/project/source_bundle.py",
    "tuner/runtime/__init__.py",
    "tuner/runtime/artifacts.py",
    "tuner/runtime/offline_sft_worker.py",
    "tuner/training/__init__.py",
    "tuner/training/methods/__init__.py",
    "tuner/training/methods/sft.py",
    "tuner/training/recipes.py",
)


def test_manifest_is_the_exact_authoritative_offline_sft_closure() -> None:
    payload = _MANIFEST.read_bytes()
    document = json.loads(payload.decode("utf-8"))
    members = document["members"]

    assert document["schema_version"] == OFFLINE_SFT_CLOSURE_SCHEMA
    assert document["closure_ref"] == OFFLINE_SFT_CLOSURE_REF
    assert document["entrypoint"] == OFFLINE_SFT_ENTRYPOINT
    assert document["trainer_entrypoint"] == OFFLINE_SFT_TRAINER_ENTRYPOINT
    assert document["owned_module_prefixes"] == list(OWNED_MODULE_PREFIXES)
    assert document["optional_features"] == []
    assert tuple(member["path"] for member in members) == _MEMBERS
    assert document["member_count"] == len(_MEMBERS) == 66
    assert document["payload_bytes"] == sum(
        member["size_bytes"] for member in members
    )
    assert document["closure_digest"] == closure_digest(document)
    assert payload == json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"

    for member in members:
        path = _ROOT.joinpath(*member["path"].split("/"))
        content = path.read_bytes()
        assert member["git_mode"] == "100644"
        assert member["size_bytes"] == len(content)
        assert member["sha256"] == hashlib.sha256(content).hexdigest()


def test_manifest_is_declared_as_tuner_runtime_package_data() -> None:
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"tuner.runtime" = ["manifests/offline-sft-worker-v1.json"]' in pyproject
