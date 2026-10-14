"""Optional evidence bindings and real native archives over controlled CPU records."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from Trainers.grpo import runtime_v1 as core_runtime
from tuner.runtime import packaged_grpo_execution as execution
from tuner.runtime import packaged_sft_execution as shared
from tuner.runtime.packaged_grpo_verification import assemble_grpo_artifacts, verify_packaged_grpo_artifacts
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import CanonicalDocument
from tuner.training.packaged_boundary import _selected_packaged_projection
from tuner.training.packaged_grpo_compilation import (
    _core, compile_packaged_grpo_workload, packaged_grpo_configuration_digest,
    parse_packaged_grpo_workload,
)
from tuner.training.recipes import canonical_json_bytes
from tests.runtime.test_packaged_grpo_compilation import _fixture
from tests.runtime.test_packaged_grpo_native import native as native, _rewrite_role

POLICY = {"max_trainable_bytes": 1024, "max_trainable_tensors": 4, "max_reward_values": 8}


def test_absent_policy_preserves_exact_old_wrapper_and_configuration():
    core, context = _fixture()
    plain = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    expected = {
        "schema_version": "synaptic-packaged-grpo-workload/v1", "method": "grpo",
        "entrypoint": "tuner.runtime.packaged_training_worker:main", "core_workload": core.document,
        "context": context, "limits": {"child_max_seconds": 60},
        "artifacts": {"schema_version": "synaptic-grpo-artifacts/v1", "requirements": [
            {"role": role, "minimum": 1, "maximum": 1} for role in (
                "workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer")]},
    }
    assert plain.canonical_bytes == canonical_json_bytes(expected)
    explicit_none = compile_packaged_grpo_workload(core_workload=core, context=context,
        child_max_seconds=60, qualification_evidence=None)
    assert explicit_none == plain
    assert packaged_grpo_configuration_digest(explicit_none) == packaged_grpo_configuration_digest(plain)


def test_policy_is_detached_committed_and_forwarded_through_existing_boundary():
    core, context = _fixture()
    policy = dict(POLICY)
    enabled = compile_packaged_grpo_workload(core_workload=core, context=context,
        child_max_seconds=60, qualification_evidence=policy)
    policy["max_reward_values"] = 1
    assert enabled.document["limits"]["qualification_evidence"] == POLICY
    assert parse_packaged_grpo_workload(enabled.canonical_bytes) == enabled
    config = CanonicalDocument.from_mapping({
        "schema_version": "synaptic-packaged-grpo-config/v1", "method": "grpo",
        "execution": {"mode": "packaged_runtime"}, "core_workload": core.document,
        "limits": enabled.document["limits"],
    })
    actual, digest, _ = _selected_packaged_projection(config, CanonicalDocument.from_mapping(context))
    assert actual == enabled and digest == packaged_grpo_configuration_digest(enabled)
    plain = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    assert enabled.fingerprint != plain.fingerprint
    assert digest != packaged_grpo_configuration_digest(plain)


@pytest.mark.parametrize("bad", [None, False, {}, {**POLICY, "max_trainable_bytes": True},
    {**POLICY, "max_trainable_tensors": 4097}, {**POLICY, "extra": 1}])
def test_present_bad_policy_cannot_fall_back_to_absence(bad):
    core, context = _fixture()
    plain = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    value = plain.document
    value["limits"]["qualification_evidence"] = bad
    with pytest.raises(ValueError):
        parse_packaged_grpo_workload(canonical_json_bytes(value))


def measured_record(core):
    """Real tiny gradients/reward values, not pretrained model/GPU evidence."""
    if os.environ.get("SYNAPTIC_TEST_GRPO_CPU_EVIDENCE") != "1":
        pytest.skip("requires the explicit pinned CPU GRPO evidence lane")
    import torch
    assert torch.__version__ == "2.9.0+cpu" and torch.version.cuda is None
    from Trainers.grpo.src.rewards import build_combined_reward_function
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0, 2.0])))
    m = core_runtime._QualificationMeasurements(POLICY, time.monotonic() + 30)
    m.begin(model, SimpleNamespace(world_size=1, scale_rewards="group"))
    reward, _ = build_combined_reward_function(core["config"]["rewards"], Path("."),
        documents=core["reward_documents"], schema_config=core["reward_schema"], raw_only=True)
    actual_reward = m.reward_wrapper(reward)
    optimizer = torch.optim.SGD(model.parameters(), lr=core["config"]["training"]["learning_rate"])
    logs = []
    for step in range(1, core["config"]["training"]["max_steps"] + 1):
        values = actual_reward(["OK", "bad"], prompts=["same", "same"])
        optimizer.zero_grad()
        model.weight.square().sum().backward()
        optimizer.step()
        std = torch.tensor(values).std(correction=1).item()
        log = {"step": step, "reward_std": std, "frac_reward_zero_std": float(std <= 1e-8)}
        logs.append(log)
        m.log(SimpleNamespace(global_step=step), log)
    m.end(model)
    return m.record(core), logs


@pytest.fixture
def enabled_native(native):
    admitted, old, custody, output = native
    compiled = compile_packaged_grpo_workload(core_workload=_core(old.document["core_workload"]),
        context=old.document["context"], child_max_seconds=60, qualification_evidence=POLICY)
    identity = compiled.document["core_workload"]["prepared_input"]
    binding = PackagedExecutionBindingV1.build(run_ref=admitted.execution.run_ref,
        runtime_release=admitted.release, provider_runtime_binding=admitted.provider_binding,
        prepared_input_ref=identity["ref"], prepared_input_revision=identity["revision"],
        prepared_input_content_digest=identity["content_digest"], prepared_input_size_bytes=identity["size_bytes"],
        prepared_input_format=identity["format"], workload_digest=compiled.fingerprint,
        configuration_digest=packaged_grpo_configuration_digest(compiled),
        artifact_policy_digest=admitted.execution.artifact_policy_digest)
    selected = replace(admitted, execution=binding, workload_bytes=compiled.canonical_bytes)
    _, path, child = execution._invocation_spec(selected, custody.private_copy.manifest, 100, 101, time.monotonic() + 60)
    path.write_bytes(child)
    record, logs = measured_record(compiled.document["core_workload"])
    for filename in ("training_lineage.json", "training_metrics.json"):
        path = output / filename
        value = json.loads(path.read_bytes())
        value["qualification_evidence"] = record
        if filename == "training_metrics.json":
            value["log_history"] = logs
        path.write_bytes(shared._canonical(value))
    result = assemble_grpo_artifacts(selected, compiled, custody.private_copy.manifest)
    inventory = result.inventory_path.read_bytes()
    terminal = shared._canonical({"schema_version": execution.TERMINAL_SCHEMA, **shared._bindings(selected),
        "status": "completed", "trainer_exit_code": 0, "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "artifact_roles": list(execution.ROLES)})
    return selected, compiled, inventory, terminal


def test_real_native_evidence_roundtrip_and_schema(enabled_native):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    admitted, compiled, inventory, terminal = enabled_native
    assert verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal)
    lineage = json.loads((admitted.paths.artifacts / "training_lineage.json").read_bytes())
    assert lineage["trainer_lineage"]["qualification_evidence"]["measurement_result"] == "changed_with_grouped_variation"
    root = Path(execution.__file__).resolve().parents[2] / "schemas"
    registry = Registry()
    for name in ("synaptic-packaged-grpo-workload-v1.schema.json", "synaptic-grpo-workload-v1.schema.json",
                 "synaptic-grpo-config-v1.schema.json", "synaptic-packaged-training-runtime-release-v1.schema.json",
                 "synaptic-packaged-training-runtime-release-v2.schema.json"):
        value = json.loads(root.joinpath(name).read_bytes())
        registry = registry.with_resource("https://schemas.synapticlabs.ai/" + name, Resource.from_contents(value))
    Draft202012Validator(json.loads(root.joinpath("synaptic-packaged-grpo-workload-v1.schema.json").read_bytes()), registry=registry).validate(compiled.document)
    Draft202012Validator(json.loads(root.joinpath("synaptic-packaged-grpo-training-lineage-v1.schema.json").read_bytes())).validate(lineage)


@pytest.mark.parametrize("change", ["missing", "changed", "fraction", "policy"])
def test_hash_consistent_native_evidence_changes_refuse(enabled_native, change):
    admitted, _, inventory, _ = enabled_native
    def mutate(raw):
        value = json.loads(raw)
        parent = value["trainer_lineage"] if "trainer_lineage" in value else value
        evidence = parent["qualification_evidence"]
        if change == "missing": del parent["qualification_evidence"]
        elif change == "changed": evidence["parameters"]["changed"] = False
        elif change == "fraction": evidence["rewards"]["grouped_observations"][0]["frac_reward_zero_std"] = 1
        else: evidence["policy"]["max_reward_values"] += 1
        return shared._canonical(value)
    inventory, _ = _rewrite_role(admitted, inventory, "training_lineage", mutate)
    inventory, terminal = _rewrite_role(admitted, inventory, "training_metrics", mutate)
    with pytest.raises(ValueError):
        verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal)


def test_evidence_requires_explicit_admitted_policy(native):
    admitted, compiled, custody, _ = native
    record, _ = measured_record(compiled.document["core_workload"])
    result = assemble_grpo_artifacts(admitted, compiled, custody.private_copy.manifest)
    inventory = result.inventory_path.read_bytes()
    def add(raw):
        value = json.loads(raw)
        parent = value["trainer_lineage"] if "trainer_lineage" in value else value
        parent["qualification_evidence"] = record
        return shared._canonical(value)
    inventory, _ = _rewrite_role(admitted, inventory, "training_lineage", add)
    inventory, terminal = _rewrite_role(admitted, inventory, "training_metrics", add)
    with pytest.raises(ValueError):
        verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal)
