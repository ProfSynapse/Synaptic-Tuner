"""Capture is explicit, bounded and digest-bound without changing old workloads."""
from copy import deepcopy
from pathlib import Path
import json

import jsonschema
import pytest

from tests.training.test_modal_recipe import _load
from tests.training.test_packaged_execution_material import packaged_fixture
from tuner.training.contracts import CanonicalDocument
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload, packaged_configuration_digest,
    validate_observation_capture_config,
)


CAPTURE = {
    "schema_version": "synaptic-training-observation-capture/v1",
    "every_n_steps": 5, "max_records": 10000,
}


def test_absence_keeps_exact_legacy_workload_bytes():
    _, components = packaged_fixture()
    config = components.resolved_config
    first = compile_packaged_sft_workload(resolved_config=config)
    raw = config.to_dict()
    assert "observation_capture" not in raw
    # Rebuild the pre-extension document explicitly: no null or default field.
    legacy = {key: raw[key] for key in (
        "schema_version", "method", "execution", "model", "dataset", "sft",
    )}
    assert first == compile_packaged_sft_workload(
        resolved_config=CanonicalDocument.from_mapping(legacy),
    )


def test_capture_is_bound_by_configuration_and_workload_digests():
    _, components = packaged_fixture()
    baseline = components.resolved_config
    configs = []
    for capture in (CAPTURE, {**CAPTURE, "every_n_steps": 6}, {**CAPTURE, "max_records": 11}):
        value = baseline.to_dict()
        value["observation_capture"] = capture
        configs.append(CanonicalDocument.from_mapping(value))
    configs.append(baseline)
    assert len({packaged_configuration_digest(config) for config in configs}) == 4
    assert len({compile_packaged_sft_workload(resolved_config=config).fingerprint
                for config in configs}) == 4
    workload = compile_packaged_sft_workload(resolved_config=configs[0])
    assert workload.document["configuration"]["document"]["observation_capture"] == CAPTURE
    schema = json.loads((Path(__file__).resolve().parents[2] /
        "schemas/synaptic-packaged-sft-workload-v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(workload.document, schema)


@pytest.mark.parametrize("invalid", [
    None, {}, {**CAPTURE, "schema_version": "unknown"}, {**CAPTURE, "enabled": True},
    {**CAPTURE, "max_records": 10001},
    *({**CAPTURE, key: value} for key in ("every_n_steps", "max_records")
      for value in (0, -1, True, False, 1.0, "1", 1000001, None)),
])
def test_invalid_capture_rejected_by_recipe_and_compiler(monkeypatch, invalid):
    _, components = packaged_fixture()
    document = components.resolved_config.to_dict()
    document["observation_capture"] = invalid
    with pytest.raises(ValueError):
        compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(document))
    with pytest.raises(ValueError):
        _load(monkeypatch, lambda config: config.update(observation_capture=invalid))


def test_recipe_projects_capture_without_adding_it_to_training_hyperparameters(monkeypatch):
    recipe = _load(monkeypatch, lambda config: config.update(observation_capture=deepcopy(CAPTURE)))
    from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest, recipe.dataset_digest,
        "a" * 64, 123, "syntunia-sft-row/v2",
    )
    config = recipe.packaged_config(identity).to_dict()
    assert config["observation_capture"] == CAPTURE
    assert "observation_capture" not in config["sft"]
    clone = validate_observation_capture_config(CAPTURE)
    clone["max_records"] = 1
    assert CAPTURE["max_records"] == 10000
