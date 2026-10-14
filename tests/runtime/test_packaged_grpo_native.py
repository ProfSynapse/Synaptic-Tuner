"""Real native archive/byte checks over explicit CPU fixture outputs, not ML proof."""
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from tuner.runtime import packaged_sft_execution as shared
from tuner.runtime import packaged_grpo_execution as runtime
from tuner.runtime.packaged_grpo_verification import assemble_grpo_artifacts, verify_packaged_grpo_artifacts
from tuner.runtime.releases import PackagedExecutionBindingV1, parse_packaged_runtime_release
from tuner.training.contracts import ArtifactPolicy
from tuner.training.packaged_compilation import packaged_artifact_policy_digest
from tuner.training.packaged_grpo_compilation import (
    ROLES, compile_packaged_grpo_workload, packaged_grpo_configuration_digest,
)
from tests.runtime.test_packaged_grpo_compilation import _fixture
from tests.runtime.test_packaged_runtime_releases import _provider
from tests.runtime.test_artifact_verification import _safetensors


def material(tmp_path):
    raw = b'{"prompt":"say OK","declared":"retained"}\n'
    core, context = _fixture(raw)
    compiled = compile_packaged_grpo_workload(core_workload=core, context=context, child_max_seconds=60)
    release = parse_packaged_runtime_release(context["runtime_release"])
    provider = _provider(release)
    policy = ArtifactPolicy(tuple(sorted(ROLES)), False)
    prepared = core.document["prepared_input"]
    execution = PackagedExecutionBindingV1.build(run_ref="cpu-grpo-fixture", runtime_release=release,
        provider_runtime_binding=provider, prepared_input_ref=prepared["ref"], prepared_input_revision=prepared["revision"],
        prepared_input_content_digest=prepared["content_digest"], prepared_input_size_bytes=prepared["size_bytes"],
        prepared_input_format=prepared["format"], workload_digest=compiled.fingerprint,
        configuration_digest=packaged_grpo_configuration_digest(compiled),
        artifact_policy_digest=packaged_artifact_policy_digest(policy))
    roots = {name: tmp_path / name for name in ("artifacts", "state", "tracking", "cache", "tmp")}
    for root in roots.values():
        root.mkdir(mode=0o700)
    dataset = tmp_path / "input.jsonl"
    dataset.write_bytes(raw)
    paths = shared.PackagedSFTPaths(prepared_input=dataset, **roots)
    directories = shared._hold_paths(paths)
    admitted = runtime.AdmittedPackagedGRPO(release, provider, execution, compiled.canonical_bytes,
        policy, paths, (), directories, runtime._TOKEN, shared._physical_commitment(paths, ()))
    return admitted, compiled


@pytest.fixture
def native(tmp_path, monkeypatch):
    admitted, compiled = material(tmp_path)
    paths, execution, directories, release = admitted.paths, admitted.execution, admitted._directories, admitted.release
    core = SimpleNamespace(document=compiled.document["core_workload"])
    from Trainers.sft import runtime_v1 as utils
    def prepare(model, cache):
        snapshot = utils._model_snapshot_path(model, cache)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b"{}")
        return snapshot
    monkeypatch.setattr(shared, "_inspect_parent_release", lambda *args, **kwargs: None)
    custody = shared._prepared_execution_inputs(paths=paths, execution=execution, directories=directories,
        release=release, model=runtime._model(compiled), model_preparer=prepare, method="grpo")
    invocation, transport, child = runtime._invocation_spec(admitted, custody.private_copy.manifest, 100, 101,
                                                          time.monotonic() + 60)
    transport.write_bytes(child)
    output = paths.state / "grpo-output"
    output.mkdir(mode=0o700)
    final = output / "final_model"
    token = output / "tokenizer"
    final.mkdir(mode=0o700)
    token.mkdir(mode=0o700)
    model_ref = core.document["model"]["model_id"]
    (final / "adapter_config.json").write_bytes(shared._canonical({"peft_type": "LORA", "base_model_name_or_path": model_ref}))
    (final / "adapter_model.safetensors").write_bytes(_safetensors())
    for root in (final, token):
        (root / "tokenizer_config.json").write_bytes(shared._canonical({"tokenizer_class": "ExampleTokenizer"}))
        (root / "tokenizer.json").write_bytes(shared._canonical({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0}}}))
    core_document = core.document
    lineage = {"method": "grpo", "model": core_document["model"],
        "input_commitments": {key: core_document[key] for key in ("config_sha256", "dataset_sha256", "rewards_sha256")},
        "reward_plan": [{"type": "rubric", "name": "exact", "weight": 1.0, "source": "exact"}],
        "actual_steps": 3}
    metrics = {"global_step": 3, "log_history": [{"reward": 0.5}], "train_metrics": {}}
    for name, document in (("workload_record", core_document), ("training_lineage", lineage), ("training_metrics", metrics)):
        (output / (name + ".json")).write_bytes(shared._canonical(document))
    try:
        yield admitted, compiled, custody, output
    finally:
        custody.close()
        admitted.close()


def _assemble(native):
    admitted, compiled, custody, _ = native
    result = assemble_grpo_artifacts(admitted, compiled, custody.private_copy.manifest)
    inventory = result.inventory_path.read_bytes()
    terminal = {"schema_version": runtime.TERMINAL_SCHEMA, **shared._bindings(admitted),
        "status": "completed", "trainer_exit_code": 0, "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "artifact_roles": list(ROLES)}
    return result, inventory, shared._canonical(terminal)


def _rewrite_role(admitted, inventory, role, change):
    value = json.loads(inventory)
    entry = next(item for item in value["artifacts"] if item["role"] == role)
    path = admitted.paths.artifacts / entry["path"]
    content = change(path.read_bytes())
    path.write_bytes(content)
    entry.update(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    updated = shared._canonical(value)
    (admitted.paths.state / "runtime-v1-inventory.json").write_bytes(updated)
    terminal = {"schema_version": runtime.TERMINAL_SCHEMA, **shared._bindings(admitted),
        "status": "completed", "trainer_exit_code": 0, "inventory_sha256": hashlib.sha256(updated).hexdigest(),
        "artifact_roles": list(ROLES)}
    return updated, shared._canonical(terminal)


def test_actual_five_role_archive_verification(native):
    admitted, _, _, _ = native
    _, inventory, terminal = _assemble(native)
    assert verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal)


@pytest.mark.parametrize("mutation", ("model_snapshot", "trainer_model", "step", "extra"))
def test_hash_consistent_changed_native_lineage_refuses(native, mutation):
    admitted, _, _, _ = native
    _, inventory, _ = _assemble(native)
    def change(raw):
        value = json.loads(raw)
        if mutation == "model_snapshot":
            value["model_snapshot"]["inode"] += 1
        elif mutation == "trainer_model":
            value["trainer_lineage"]["model"]["revision"] = "b" * 40
        elif mutation == "step":
            value["trainer_lineage"]["actual_steps"] = True
        else:
            value["extra"] = True
        return shared._canonical(value)
    inventory, terminal = _rewrite_role(admitted, inventory, "training_lineage", change)
    with pytest.raises(ValueError):
        verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal)


def test_terminal_false_exit_and_extra_artifact_are_not_completion(native):
    admitted, _, _, _ = native
    _, inventory, terminal = _assemble(native)
    value = json.loads(terminal)
    value["trainer_exit_code"] = False
    with pytest.raises(ValueError):
        verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=shared._canonical(value))
    (admitted.paths.artifacts / "unexpected").write_bytes(b"extra")
    with pytest.raises(ValueError):
        verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal)


def test_closed_admission_never_prepares_or_spawns(native, monkeypatch):
    admitted, _, _, _ = native
    admitted.close()
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawn"))
    with pytest.raises(runtime.PackagedGRPOExecutionError):
        runtime.execute_admitted_packaged_grpo(admitted, model_preparer=lambda *args: pytest.fail("prepare"))


def test_emitted_native_documents_match_installed_schema_resources(native):
    from pathlib import Path
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    # Anchor to the actual production package, including editable installs.
    schema_root = Path(runtime.__file__).resolve().parents[2] / "schemas"
    names = (
        "synaptic-packaged-grpo-workload-v1.schema.json",
        "synaptic-grpo-workload-v1.schema.json",
        "synaptic-grpo-config-v1.schema.json",
        "synaptic-packaged-training-runtime-release-v1.schema.json",
        "synaptic-packaged-training-runtime-release-v2.schema.json",
        "synaptic-packaged-grpo-training-lineage-v1.schema.json",
        "synaptic-packaged-grpo-terminal-v1.schema.json",
    )
    documents = {name: json.loads(schema_root.joinpath(name).read_bytes()) for name in names}
    registry = Registry()
    for name, document in documents.items():
        registry = registry.with_resource(
            "https://schemas.synapticlabs.ai/" + name, Resource.from_contents(document)
        )
    result, inventory, terminal = _assemble(native)
    values = {
        "synaptic-packaged-grpo-workload-v1.schema.json": native[1].document,
        "synaptic-packaged-grpo-training-lineage-v1.schema.json": json.loads(
            (native[0].paths.artifacts / "training_lineage.json").read_bytes()
        ),
        "synaptic-packaged-grpo-terminal-v1.schema.json": json.loads(terminal),
    }
    for name, value in values.items():
        validator = Draft202012Validator(documents[name], registry=registry)
        validator.check_schema(documents[name])
        validator.validate(value)


@pytest.mark.parametrize("mutation", ("environment", "workload"))
def test_hash_consistent_child_payload_mutation_is_not_execution_evidence(native, mutation):
    admitted, _, _, _ = native
    _, inventory, _ = _assemble(native)
    path = admitted.paths.state / "packaged-grpo-child.json"
    payload = json.loads(path.read_bytes())
    if mutation == "environment":
        payload["environment"].append(["UNBOUND", "value"])
    else:
        payload["workload"]["limits"]["child_max_seconds"] += 1
    raw = shared._canonical(payload)
    path.write_bytes(raw)
    def change(content):
        lineage = json.loads(content)
        lineage["child_transport_sha256"] = hashlib.sha256(raw).hexdigest()
        return shared._canonical(lineage)
    inventory, terminal = _rewrite_role(admitted, inventory, "training_lineage", change)
    with pytest.raises(ValueError):
        verify_packaged_grpo_artifacts(
            admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal
        )
