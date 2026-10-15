"""Actual isolated child/native path with explicit tiny CPU library boundary."""
from dataclasses import replace
import json
import os
import time

import pytest

if os.environ.get("SYNAPTIC_TEST_GRPO_CPU_EVIDENCE") != "1":
    pytest.skip("requires the explicit pinned CPU GRPO evidence lane", allow_module_level=True)

from tuner.runtime import packaged_grpo_execution as runtime
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.packaged_grpo_compilation import _core, compile_packaged_grpo_workload, packaged_grpo_configuration_digest
from tests.runtime.test_packaged_grpo_child import _BOOTSTRAP, _ports, _prepare
from tests.runtime.test_packaged_grpo_native import material

POLICY = {"max_trainable_bytes": 1024, "max_trainable_tensors": 4, "max_reward_values": 8}


def admitted_policy(tmp_path, policy):
    old, baseline = material(tmp_path)
    compiled = compile_packaged_grpo_workload(core_workload=_core(baseline.document["core_workload"]),
        context=baseline.document["context"], child_max_seconds=30, qualification_evidence=policy)
    identity = compiled.document["core_workload"]["prepared_input"]
    binding = PackagedExecutionBindingV1.build(run_ref=old.execution.run_ref,
        runtime_release=old.release, provider_runtime_binding=old.provider_binding,
        prepared_input_ref=identity["ref"], prepared_input_revision=identity["revision"],
        prepared_input_content_digest=identity["content_digest"], prepared_input_size_bytes=identity["size_bytes"],
        prepared_input_format=identity["format"], workload_digest=compiled.fingerprint,
        configuration_digest=packaged_grpo_configuration_digest(compiled),
        artifact_policy_digest=old.execution.artifact_policy_digest)
    return replace(old, workload_bytes=compiled.canonical_bytes, execution=binding)


def bootstrap(update):
    code = _BOOTSTRAP.replace("grpo_child_fixture.install(source)",
        f"grpo_child_fixture.install(source,cpu_evidence=True,update={update!r})")
    witness = '''
from Trainers.grpo import runtime_v1 as measured
original_measurement_init=measured._QualificationMeasurements.__init__
exact_payload=json.loads(Path(grp_args[1]).read_bytes())
def check_original_deadline(self,policy,deadline):
    assert deadline == exact_payload['deadline']
    original_measurement_init(self,policy,deadline)
    Path(exact_payload['paths']['tracking'],'measurement-deadline.json').write_text(json.dumps({'deadline':deadline}))
measured._QualificationMeasurements.__init__=check_original_deadline
'''
    return code.replace("result=child.main()", witness + "\nresult=child.main()")


@pytest.mark.parametrize("update", [False, True])
def test_enabled_policy_real_child_native_completion(tmp_path, monkeypatch, update):
    admitted = admitted_policy(tmp_path, POLICY)
    commands = _ports(monkeypatch, bootstrap(update))
    try:
        result = runtime.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
        assert len(commands) == 1 and len(result.artifacts) == 5
        lineage = json.loads((admitted.paths.artifacts / "training_lineage.json").read_bytes())
        evidence = lineage["trainer_lineage"]["qualification_evidence"]
        assert evidence["parameters"]["changed"] is update
        assert evidence["measurement_result"] == ("changed_with_grouped_variation" if update else "inconclusive")
        assert evidence["rewards"]["value_count"] == 6
        payload = json.loads((admitted.paths.state / "packaged-grpo-child.json").read_bytes())
        witness = json.loads((admitted.paths.tracking / "measurement-deadline.json").read_bytes())
        assert witness["deadline"] == payload["deadline"]
        assert witness["deadline"] <= time.monotonic() + 30
        assert result.terminal_path.is_file()
        with pytest.raises(runtime.PackagedGRPOExecutionError):
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=lambda *a: pytest.fail("no replay"))
        assert len(commands) == 1
    finally:
        admitted.close()


def test_enabled_policy_bound_rejects_child_without_native_completion(tmp_path, monkeypatch):
    admitted = admitted_policy(tmp_path, {**POLICY, "max_trainable_bytes": 1})
    commands = _ports(monkeypatch, bootstrap(True))
    try:
        with pytest.raises(runtime.PackagedGRPOExecutionError) as caught:
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
        assert caught.value.stage == "TRAINER_CHILD_EXEC_VALUE_MEASUREMENT_BEGIN"
        assert len(commands) == 1
        assert not (admitted.paths.state / "packaged-terminal.json").exists()
        assert not list(admitted.paths.artifacts.iterdir())
        assert not (admitted.paths.state / "grpo-output/training_lineage.json").exists()
        assert admitted._used
        with pytest.raises(runtime.PackagedGRPOExecutionError):
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=lambda *a: pytest.fail("no replay"))
        assert len(commands) == 1
    finally:
        admitted.close()
