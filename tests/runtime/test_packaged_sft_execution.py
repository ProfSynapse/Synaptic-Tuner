from dataclasses import replace
import hashlib
import json
import os
import secrets
import shutil
import stat
import traceback
from pathlib import Path, PurePosixPath

import pytest

from Trainers.sft import runtime_v1 as core
from tuner.runtime import packaged_sft_execution as seam
from tuner.runtime.packaged_worker_closure import load_packaged_worker_closure
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import CanonicalDocument
from tuner.training.packaged_compilation import compile_packaged_sft_workload, packaged_artifact_policy_digest
from tests.runtime.test_packaged_runtime_releases import _release, _provider
from tests.training.test_packaged_execution_material import packaged_fixture, rebound
from tests.runtime.test_artifact_verification import _safetensors


@pytest.fixture
def material(tmp_path, monkeypatch):
    _, components = packaged_fixture()
    raw = b'{"text":"training input"}\n'
    config = components.resolved_config.to_dict()
    config["model"]["tokenizer_revision"] = config["model"]["revision"]
    config["dataset"].update(content_digest=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    config["sft"]["lora_dropout"] = 0.125
    compiled = compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(config))
    release = _release(worker_entrypoint="tuner.runtime.packaged_training_worker:main",
        worker_closure_digest=load_packaged_worker_closure().digest,
        workload_schema=compiled.schema_version,
        compatible_models=((config["model"]["ref"], config["model"]["revision"]),))
    provider = _provider(release)
    identity = config["dataset"]
    binding = PackagedExecutionBindingV1.build(run_ref="run:test", runtime_release=release,
        provider_runtime_binding=provider, prepared_input_ref=identity["ref"],
        prepared_input_revision=identity["revision"], prepared_input_content_digest=identity["content_digest"],
        prepared_input_size_bytes=identity["size_bytes"], prepared_input_format=identity["format"],
        workload_digest=compiled.fingerprint, configuration_digest=compiled.document["configuration"]["digest"],
        artifact_policy_digest=packaged_artifact_policy_digest(components.artifact_policy))
    paths = seam.PackagedSFTPaths(tmp_path / "data.jsonl", *(tmp_path / name for name in ("artifacts", "state", "tracking", "cache", "tmp")))
    paths.prepared_input.write_bytes(raw)
    for name in ("artifacts", "state", "tracking", "cache", "tmp"):
        getattr(paths, name).mkdir()
    monkeypatch.setattr(seam, "_inspect_release", lambda release: Path("/installed/Trainers/sft/train_sft.py"))
    return dict(runtime_release=release, provider_binding=provider, execution_binding=binding,
        workload_bytes=compiled.canonical_bytes, artifact_policy=components.artifact_policy, paths=paths)


def test_admission_is_read_only_and_path_free(material):
    admitted = seam.admit_packaged_sft(**material)
    try:
        assert str(admitted.paths.state).encode() not in admitted.workload_bytes
        assert list(admitted.paths.state.iterdir()) == []
        assert list(admitted.paths.cache.iterdir()) == []
    finally:
        admitted.close()


@pytest.mark.parametrize("field", ["workload_digest", "configuration_digest", "prepared_input", "artifact_policy_digest", "runtime_release_digest", "provider_runtime_binding_digest"])
def test_cross_binding_rejected_before_preparation(material, field):
    changes = {field: "0" * 64}
    if field == "prepared_input":
        changes[field] = {**material["execution_binding"].to_dict()[field], "content_digest": "0" * 64}
    material["execution_binding"] = rebound(material["execution_binding"], **changes)
    with pytest.raises(seam.PackagedSFTExecutionError, match="ADMISSION"):
        seam.admit_packaged_sft(**material)
    assert list(material["paths"].state.iterdir()) == []


@pytest.mark.parametrize("change", ["noncanonical", "extra", "input", "overlap", "hardlink", "credentials"])
def test_invalid_inputs_fail_closed(material, change):
    if change == "noncanonical":
        material["workload_bytes"] += b"\n"
    elif change == "extra":
        value = seam._document(material["workload_bytes"])
        value["physical_path"] = "/private"
        material["workload_bytes"] = seam._canonical(value)
    elif change == "input":
        material["paths"].prepared_input.write_bytes(b"changed")
    elif change == "overlap":
        material["paths"] = replace(material["paths"], artifacts=material["paths"].state)
    elif change == "hardlink":
        os.link(material["paths"].prepared_input, material["paths"].prepared_input.with_suffix(".link"))
    else:
        material["environment"] = (("HF_TOKEN", "secret"),)
    with pytest.raises(seam.PackagedSFTExecutionError, match="ADMISSION"):
        seam.admit_packaged_sft(**material)


def prepare(model, cache):
    snapshot = core._model_snapshot_path(model, cache)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return snapshot


def test_input_changed_after_admission_never_prepares(material):
    admitted = seam.admit_packaged_sft(**material)
    admitted.paths.prepared_input.write_bytes(b"changed")
    calls = []
    with pytest.raises(seam.PackagedSFTExecutionError, match="REVALIDATION"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=lambda *args: calls.append(args))
    assert not calls


def test_preparer_input_mutation_never_launches(material):
    admitted = seam.admit_packaged_sft(**material)
    def corrupt(model, cache):
        result = prepare(model, cache)
        admitted.paths.prepared_input.write_bytes(b"changed")
        return result
    with pytest.raises(seam.PackagedSFTExecutionError, match="REVALIDATION"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=corrupt)


class FakeRunner:
    def run(self, invocation):
        assert invocation.argv[:4] == ("/usr/local/bin/python", "-I", "-m", "tuner.runtime.packaged_sft_child")
        env = dict(invocation.environment)
        assert "HF_TOKEN" not in env and "PYTHONPATH" not in env
        assert "LD_LIBRARY_PATH" not in env
        assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
        assert "SYNAPTIC_ENGINE_ROOT" not in env
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert "--require-memory-efficient-loss" not in invocation.argv
        model = invocation.final_model_dir
        model.mkdir(parents=True)
        (model / "adapter_config.json").write_bytes(seam._canonical({"peft_type": "LORA", "base_model_name_or_path": "example/model"}))
        (model / "adapter_model.safetensors").write_bytes(_safetensors())
        (model / "tokenizer_config.json").write_bytes(seam._canonical({"tokenizer_class": "TestTokenizer"}))
        (model / "tokenizer.json").write_bytes(seam._canonical({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0}}}))
        lineage = {"synaptic_runtime_projection": invocation.expected_projection}
        return core.TrainerEvidence(0, model, model, lineage, invocation.expected_projection, {"loss": 0.5})


@pytest.fixture
def seal(monkeypatch, tmp_path):
    if os.name != "posix":
        # Windows CPU test double only; production still requires Linux seals.
        def sealed(raw, *, content_digest):
            path = tmp_path / "sealed-test-only"
            path.write_bytes(raw)
            fd = os.open(path, os.O_RDONLY)
            return fd, PurePosixPath(f"/proc/self/fd/{fd}")
        monkeypatch.setattr(core, "_sealed_prepared_dataset", sealed)
        # Production copying requires Linux dir_fd/O_NOFOLLOW. This Windows
        # fixture exercises the same retained-copy validation after a CPU copy.
        def private_copy(snapshot, directories, inventory, paths, model):
            root = paths.state / ("packaged-model-" + secrets.token_hex(16))
            root.mkdir(mode=0o700)
            copied = core._model_snapshot_path(model, root)
            members = []
            for name, size, digest in inventory:
                target = copied / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((snapshot / name).read_bytes())
                target.chmod(stat.S_IREAD)
                info = target.stat()
                members.append(dict(path=name, size_bytes=size, sha256=digest, device=info.st_dev, inode=info.st_ino))
            info = root.stat()
            manifest = dict(root=str(root), device=info.st_dev, inode=info.st_ino, members=members)
            return seam._retain_private_snapshot(manifest, paths, model)
        monkeypatch.setattr(seam, "_copy_private_snapshot", private_copy)


def test_fake_execution_emits_five_bound_roles_and_terminal(material, seal):
    admitted = seam.admit_packaged_sft(**material)
    result = seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    assert tuple(item["role"] for item in result.artifacts) == seam._ROLES
    lineage = seam._document((admitted.paths.artifacts / "training_lineage.json").read_bytes())
    assert lineage["schema_version"] == seam.LINEAGE_SCHEMA
    assert "execution_source" not in lineage
    assert lineage["trainer_projection"]["lora"]["dropout"] == 0.125
    assert seam.verify_packaged_sft_artifacts(admitted=admitted, inventory_bytes=result.inventory_path.read_bytes(), terminal_bytes=result.terminal_path.read_bytes())
    with pytest.raises(seam.PackagedSFTExecutionError, match="REVALIDATION"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare)


def test_terminal_substitution_rejected(material, seal):
    admitted = seam.admit_packaged_sft(**material)
    result = seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    terminal = seam._document(result.terminal_path.read_bytes())
    terminal["execution_binding_digest"] = "0" * 64
    with pytest.raises(ValueError):
        seam.verify_packaged_sft_artifacts(admitted=admitted, inventory_bytes=result.inventory_path.read_bytes(), terminal_bytes=seam._canonical(terminal))


def test_raw_exception_details_are_suppressed(material):
    admitted = seam.admit_packaged_sft(**material)
    def fail(*args):
        raise RuntimeError("secret-token-and-private-path")
    with pytest.raises(seam.PackagedSFTExecutionError) as caught:
        seam.execute_admitted_packaged_sft(admitted, model_preparer=fail)
    assert caught.value.__suppress_context__
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("target", ["projection", "argv", "environment", "extra_field", "extra_role"])
def test_self_consistent_artifact_forgery_rejected(material, seal, target):
    admitted = seam.admit_packaged_sft(**material)
    result = seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    inventory = seam._document(result.inventory_path.read_bytes())
    terminal = seam._document(result.terminal_path.read_bytes())
    lineage_path = admitted.paths.artifacts / "training_lineage.json"
    lineage = seam._document(lineage_path.read_bytes())
    if target == "projection":
        lineage["trainer_projection"]["training"]["learning_rate"] = 0.75
        lineage["trainer_lineage"]["synaptic_runtime_projection"] = lineage["trainer_projection"]
    elif target == "argv":
        lineage["execution_evidence"]["argv"][0] = "/unowned/python"
    elif target == "environment":
        lineage["execution_evidence"]["environment"]["HF_TOKEN"] = "credential"
    elif target == "extra_field":
        lineage["execution_source"] = {"git": "fabricated"}
    else:
        inventory["artifacts"].append(dict(inventory["artifacts"][0]))
    lineage["execution_evidence_sha256"] = seam._digest(seam._canonical(lineage["execution_evidence"]))
    raw = seam._canonical(lineage)
    lineage_path.write_bytes(raw)
    inventory["artifacts"][1].update(sha256=seam._digest(raw), size=len(raw))
    inventory_raw = seam._canonical(inventory)
    terminal["inventory_sha256"] = seam._digest(inventory_raw)
    with pytest.raises(ValueError):
        seam.verify_packaged_sft_artifacts(admitted=admitted, inventory_bytes=inventory_raw, terminal_bytes=seam._canonical(terminal))


def test_packaged_evidence_schemas(material, seal):
    import jsonschema
    admitted = seam.admit_packaged_sft(**material)
    result = seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    root = Path(__file__).parents[2]
    for filename, payload in (
        ("synaptic-packaged-sft-terminal-v1.schema.json", result.terminal_path.read_bytes()),
        ("synaptic-packaged-sft-training-lineage-v1.schema.json", (admitted.paths.artifacts / "training_lineage.json").read_bytes()),
    ):
        jsonschema.validate(json.loads(payload), json.loads((root / "schemas" / filename).read_text()))


@pytest.mark.parametrize("section", ["trainer_projection", "execution_evidence", "model_snapshot"])
def test_lineage_schema_closes_authoritative_nested_fields(material, seal, section):
    import jsonschema
    admitted = seam.admit_packaged_sft(**material)
    seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    lineage = seam._document((admitted.paths.artifacts / "training_lineage.json").read_bytes())
    lineage[section]["unreviewed"] = True
    schema = json.loads((Path(__file__).parents[2] / "schemas/synaptic-packaged-sft-training-lineage-v1.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(lineage, schema)


def test_physical_binding_substitution_rejected(material):
    admitted = seam.admit_packaged_sft(**material)
    altered = replace(admitted, environment=(("PATH", "/unowned"),))
    calls = []
    with pytest.raises(seam.PackagedSFTExecutionError, match="REVALIDATION"):
        seam.execute_admitted_packaged_sft(altered, model_preparer=lambda *args: calls.append(args))
    assert not calls


def test_snapshot_hardlink_rejected(material):
    admitted = seam.admit_packaged_sft(**material)
    def hardlink(model, cache):
        snapshot = prepare(model, cache)
        os.link(snapshot / "config.json", snapshot / "linked.json")
        return snapshot
    with pytest.raises(seam.PackagedSFTExecutionError, match="PREPARATION"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=hardlink)


def test_nonzero_trainer_does_not_emit_success(material, seal):
    admitted = seam.admit_packaged_sft(**material)
    class Failed:
        def run(self, invocation):
            return core.TrainerEvidence(1, invocation.final_model_dir, invocation.tokenizer_dir, {}, {}, {})
    with pytest.raises(seam.PackagedSFTExecutionError, match="TRAINER"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=Failed())
    assert not list(admitted.paths.artifacts.iterdir())
    assert not (admitted.paths.state / "packaged-terminal.json").exists()


def test_invalid_artifact_set_never_publishes_completed_terminal(material, seal):
    admitted = seam.admit_packaged_sft(**material)
    class ExtraArtifact(FakeRunner):
        def run(self, invocation):
            result = super().run(invocation)
            (admitted.paths.artifacts / "unexpected.txt").write_text("unexpected")
            return result
    with pytest.raises(seam.PackagedSFTExecutionError, match="ARTIFACT"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=ExtraArtifact())
    assert not (admitted.paths.state / "packaged-terminal.json").exists()


def test_snapshot_changed_during_invocation_never_launches(material, seal, monkeypatch):
    admitted = seam.admit_packaged_sft(**material)
    original = seam._invocation
    calls = []
    def mutate(admitted, workload, snapshot, dataset_bytes, model_snapshot):
        result = original(admitted, workload, snapshot, dataset_bytes, model_snapshot)
        (snapshot / "config.json").chmod(stat.S_IWRITE | stat.S_IREAD)
        (snapshot / "config.json").write_text("changed")
        return result
    class Runner:
        def run(self, invocation):
            calls.append(invocation)
    monkeypatch.setattr(seam, "_invocation", mutate)
    with pytest.raises(seam.PackagedSFTExecutionError, match="REVALIDATION"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=Runner())
    assert not calls


def test_exact_python_version_rejected_before_inventory_inspection(monkeypatch):
    import tuner.runtime.packaged_training_worker as worker
    calls = []
    monkeypatch.setattr(seam, "stable_read", lambda *args, **kwargs: b"{}")
    monkeypatch.setattr(worker, "inspect_installed_runtime", lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        seam._inspect_release(_release(python_version="0.0.1"))
    assert not calls


@pytest.mark.parametrize("name", ["PATH", "LD_LIBRARY_PATH"])
def test_caller_loader_search_paths_are_rejected(material, name):
    material["environment"] = ((name, "/hostile"),)
    with pytest.raises(seam.PackagedSFTExecutionError, match="ADMISSION"):
        seam.admit_packaged_sft(**material)


@pytest.mark.parametrize("boundary", ["admission", "preparer", "runner", "cleanup"])
@pytest.mark.parametrize("exception", [SystemExit, KeyboardInterrupt, GeneratorExit])
def test_baseexception_diagnostics_never_disclose_text(material, seal, monkeypatch, boundary, exception):
    def fail(*args):
        raise exception("PRIVATE_SENTINEL_DO_NOT_DISCLOSE")
    if boundary == "admission":
        monkeypatch.setattr(seam, "_inspect_release", fail)
        operation = lambda: seam.admit_packaged_sft(**material)
    else:
        admitted = seam.admit_packaged_sft(**material)
        preparer, runner = prepare, FakeRunner()
        if boundary == "preparer":
            preparer = fail
        elif boundary == "runner":
            class RaisingRunner:
                run = staticmethod(fail)
            runner = RaisingRunner()
        else:
            original = seam.AdmittedPackagedSFT.close
            def cleanup(value):
                original(value)
                fail()
            monkeypatch.setattr(seam.AdmittedPackagedSFT, "close", cleanup)
        operation = lambda: seam.execute_admitted_packaged_sft(admitted, model_preparer=preparer, runner=runner)
    with pytest.raises(seam.PackagedSFTExecutionError) as caught:
        operation()
    rendered = "".join(traceback.format_exception(caught.value))
    assert "PRIVATE_SENTINEL_DO_NOT_DISCLOSE" not in rendered
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("replace_source", [False, True])
def test_same_size_source_mutation_after_launch_cannot_change_private_consumption(material, seal, replace_source):
    admitted = seam.admit_packaged_sft(**material)
    model = seam._document(admitted.workload_bytes)["configuration"]["document"]["model"]
    source = core._model_snapshot_path(model, admitted.paths.cache) / "config.json"
    class MutateSharedCache(FakeRunner):
        def run(self, invocation):
            private = Path(invocation.argv[invocation.argv.index("--model-snapshot") + 1]) / "config.json"
            assert private != source
            if replace_source:
                source.unlink()
            source.write_bytes(b"[]")
            assert private.read_bytes() == b"{}"
            assert private.stat().st_ino != source.stat().st_ino
            return super().run(invocation)
    result = seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=MutateSharedCache())
    assert result.terminal_path.exists()


def test_same_size_private_copy_mutation_rejects_results(material, seal):
    admitted = seam.admit_packaged_sft(**material)
    class TamperPrivateCopy(FakeRunner):
        def run(self, invocation):
            private = Path(invocation.argv[invocation.argv.index("--model-snapshot") + 1]) / "config.json"
            private.chmod(stat.S_IREAD | stat.S_IWRITE)
            private.write_bytes(b"[]")
            private.chmod(stat.S_IREAD)
            return super().run(invocation)
    with pytest.raises(seam.PackagedSFTExecutionError, match="EVIDENCE"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=TamperPrivateCopy())
    assert not list(admitted.paths.artifacts.iterdir())
    assert not (admitted.paths.state / "packaged-terminal.json").exists()


def test_child_retains_private_file_identity_through_consumption(material, seal):
    from tuner.runtime.packaged_sft_child import _run_private_trainer
    admitted = seam.admit_packaged_sft(**material)
    model = seam._document(admitted.workload_bytes)["configuration"]["document"]["model"]
    snapshot = prepare(model, admitted.paths.cache)
    directories, inventory = seam._snapshot_inventory(snapshot, admitted.paths.cache)
    private = seam._copy_private_snapshot(snapshot, directories, inventory, admitted.paths, model)
    path = private.snapshot / "config.json"
    source = "from pathlib import Path\np = Path(" + repr(str(path)) + ")\np.chmod(0o600)\np.write_bytes(b'[]')\np.chmod(0o400)\n"
    try:
        with pytest.raises(ValueError, match="PACKAGED_MODEL_COPY_REJECTED"):
            _run_private_trainer(compile(source, "private-copy-test", "exec"), Path("/installed/trainer.py"), private.manifest, admitted.paths, model)
    finally:
        private.close()
        seam._close_resources(directories)
        admitted.close()


@pytest.mark.skipif(os.name != "posix", reason="Requires actual Linux descriptor-relative private copy")
def test_linux_private_copy_keeps_exact_readonly_single_link_handles(material):
    admitted = seam.admit_packaged_sft(**material)
    model = seam._document(admitted.workload_bytes)["configuration"]["document"]["model"]
    snapshot = prepare(model, admitted.paths.cache)
    directories, inventory = seam._snapshot_inventory(snapshot, admitted.paths.cache)
    private = seam._copy_private_snapshot(snapshot, directories, inventory, admitted.paths, model)
    try:
        assert stat.S_IMODE(Path(private.manifest["root"]).stat().st_mode) == 0o700
        assert all(os.fstat(member.fd).st_nlink == 1 and stat.S_IMODE(os.fstat(member.fd).st_mode) == 0o400 for member in private.files)
        (snapshot / "config.json").write_bytes(b"[]")
        private.check()
        assert (private.snapshot / "config.json").read_bytes() == b"{}"
    finally:
        private.close()
        seam._close_resources(directories)
        admitted.close()


def _replace_artifact_claim(admitted, role):
    from tests.runtime.test_artifact_verification import _tar
    inventory_path = admitted.paths.state / "runtime-v1-inventory.json"
    inventory = seam._document(inventory_path.read_bytes())
    entry = next(item for item in inventory["artifacts"] if item["role"] == role)
    if role == "training_metrics":
        raw = b'{"loss":0.9}'
    else:
        raw = _tar({"adapter_config.json": seam._canonical({"peft_type": "LORA", "base_model_name_or_path": "example/model"}),
                    "adapter_model.safetensors": _safetensors(payload=b"\x02\x00\x00\x00")})
    (admitted.paths.artifacts / entry["path"]).write_bytes(raw)
    entry.update(size=len(raw), sha256=seam._digest(raw))
    inventory_path.write_bytes(seam._canonical(inventory))


@pytest.mark.parametrize("role", ["training_metrics", "final_model"])
@pytest.mark.parametrize("when", ["before_retention", "during_terminal_write"])
def test_synchronized_artifact_inventory_forgery_cannot_publish_terminal(material, seal, monkeypatch, role, when):
    admitted = seam.admit_packaged_sft(**material)
    if when == "before_retention":
        original = seam._retain_result_artifacts
        def mutate(value, result):
            _replace_artifact_claim(value, role)
            return original(value, result)
        monkeypatch.setattr(seam, "_retain_result_artifacts", mutate)
    else:
        original = seam._write_terminal_bytes
        def mutate(descriptor, payload):
            original(descriptor, payload)
            _replace_artifact_claim(admitted, role)
        monkeypatch.setattr(seam, "_write_terminal_bytes", mutate)
    with pytest.raises(seam.PackagedSFTExecutionError, match="ARTIFACT"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    assert not (admitted.paths.state / "packaged-terminal.json").exists()


@pytest.mark.parametrize("when", ["before_terminal_write", "during_terminal_write"])
def test_extra_artifact_member_cannot_publish_terminal(material, seal, monkeypatch, when):
    admitted = seam.admit_packaged_sft(**material)
    if when == "before_terminal_write":
        original = seam._write_terminal_exclusive
        def mutate(path, payload, parent, check):
            (admitted.paths.artifacts / "unexpected.bin").write_bytes(b"unexpected")
            return original(path, payload, parent, check)
        monkeypatch.setattr(seam, "_write_terminal_exclusive", mutate)
    else:
        original = seam._write_terminal_bytes
        def mutate(descriptor, payload):
            original(descriptor, payload)
            (admitted.paths.artifacts / "unexpected.bin").write_bytes(b"unexpected")
        monkeypatch.setattr(seam, "_write_terminal_bytes", mutate)
    with pytest.raises(seam.PackagedSFTExecutionError, match="ARTIFACT"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=prepare, runner=FakeRunner())
    assert not (admitted.paths.state / "packaged-terminal.json").exists()
    assert (admitted.paths.artifacts / "unexpected.bin").read_bytes() == b"unexpected"


@pytest.mark.skipif(os.name != "posix", reason="Windows retained directory handles prevent replacement")
def test_root_replacement_rejected_before_preparation(material):
    admitted = seam.admit_packaged_sft(**material)
    admitted.paths.state.rename(admitted.paths.state.with_name("old-state"))
    admitted.paths.state.mkdir()
    calls = []
    with pytest.raises(seam.PackagedSFTExecutionError, match="REVALIDATION"):
        seam.execute_admitted_packaged_sft(admitted, model_preparer=lambda *args: calls.append(args))
    assert not calls
