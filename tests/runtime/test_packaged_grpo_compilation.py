"""The packaged wrapper cannot widen the exact public GRPO declaration lane."""
import copy
import json
import sys

import pytest

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.training.grpo_compilation import compile_grpo_input
from tuner.training.packaged_grpo_compilation import (
    _core, _declared_input, compile_packaged_grpo_workload, parse_packaged_grpo_workload,
    packaged_grpo_configuration_digest,
)
from tuner.training.recipes import CompiledWorkload, canonical_json_bytes
from tests.contract.test_grpo_training_input_v1 import document, prepared, bound
from tests.runtime.test_packaged_runtime_releases import _release


def _fixture(raw=b'{"prompt":"say OK"}\n'):
    declaration = bound(TrainingInputV1.from_dict(document()), raw)
    core = compile_grpo_input(declaration, prepared(raw))
    release = _release(worker_entrypoint="tuner.runtime.packaged_training_worker:main",
        compatible_methods=("grpo",), compatible_models=(("model/example", "a" * 40),),
        compatible_dataset_formats=("synaptic-grpo-prompts/v1",),
        workload_schema="synaptic-packaged-grpo-workload/v1",
        artifact_contract_schema="synaptic-grpo-artifacts/v1", python_executable=sys.executable)
    context = {"schema_version": "synaptic-packaged-execution-context/v1", "runtime_release": release.to_dict()}
    return core, context


def test_wrapper_preserves_core_bytes_and_commits_child_limit():
    core, context = _fixture()
    workload = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    assert canonical_json_bytes(workload.document["core_workload"]) == core.canonical_bytes
    assert _declared_input(core.document).method.value == "grpo"
    assert parse_packaged_grpo_workload(workload.canonical_bytes) == workload
    second = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=61)
    assert second.fingerprint != workload.fingerprint
    assert packaged_grpo_configuration_digest(second) != packaged_grpo_configuration_digest(workload)
    context["runtime_release"]["compatibility"]["methods"][0] = "sft"
    assert workload.document["context"]["runtime_release"]["compatibility"]["methods"] == ["grpo"]


@pytest.mark.parametrize("bad", [True, 0, -1, 86401, 0.5, float("nan")])
def test_invalid_child_limit_refuses(bad):
    core, context = _fixture()
    with pytest.raises((TypeError, ValueError)):
        compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=bad)


@pytest.mark.parametrize("mutation", [
    lambda d: d["config"]["training"].update(optim="sgd"),
    lambda d: d["config"]["model"].update(dtype="bfloat16"),
    lambda d: d["config"]["lora"].update(bias="all"),
    lambda d: d["config"]["lora"].update(use_gradient_checkpointing=False),
    lambda d: d["config"]["training"].update(max_steps=100001),
    lambda d: d["config"]["training"].update(temperature=11),
    lambda d: d["reward_documents"]["exact"].update(extra="unadmitted"),
    lambda d: d["reward_documents"]["exact"]["validation"][0].update(match="x" * 4097),
])
def test_hash_consistent_core_cannot_bypass_public_compiler(mutation):
    import hashlib
    core, _ = _fixture()
    changed = copy.deepcopy(core.document)
    mutation(changed)
    changed["config_sha256"] = hashlib.sha256(canonical_json_bytes(changed["config"])).hexdigest()
    changed["rewards_sha256"] = hashlib.sha256(canonical_json_bytes({
        "documents": changed["reward_documents"], "schema": changed["reward_schema"]})).hexdigest()
    with pytest.raises((TypeError, ValueError, KeyError)):
        _core(changed)


@pytest.mark.parametrize("raw", ['{"prompt":"say OK"}\n'.encode("utf-16"),
    b'\xef\xbb\xbf{"prompt":"say OK"}\n'])
def test_hash_consistent_non_utf8_or_bom_snapshot_rejects(raw):
    from tuner.runtime.packaged_grpo_execution import _admit_dataset
    core, context = _fixture(raw)
    compiled = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    with pytest.raises(ValueError):
        _admit_dataset(compiled, raw)


def test_unknown_packaged_fields_and_altered_role_contract_reject():
    core, context = _fixture()
    workload = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    for change in (lambda d: d.update(extra=True),
                   lambda d: d["artifacts"]["requirements"].pop()):
        value = json.loads(workload.canonical_bytes)
        change(value)
        with pytest.raises(ValueError):
            parse_packaged_grpo_workload(canonical_json_bytes(value))
    bad = CompiledWorkload("grpo", core.schema_version, core.entrypoint, core.canonical_bytes)
    assert bad == core
