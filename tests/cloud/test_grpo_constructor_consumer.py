"""Provider-free checks for the private constructor consumer's review gates."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from examples.training_qualification import grpo_constructor as consumer
from tuner.execution.foundation_v2.canonical import canonical_bytes


def _configuration(tmp_path):
    fields = {"schema_version": consumer._SCHEMA,
        "engine_root": str(tmp_path / "engine"), "host_root": str(tmp_path / "host"),
        "project_root": str(tmp_path / "project"), "recipe_path": str(tmp_path / "recipe.yaml"),
        "plan_ready_path": str(tmp_path / "plan.json"), "dataset_path": str(tmp_path / "rows.jsonl"),
        "profiles_root": str(tmp_path / "profiles"), "private_root": str(tmp_path / "private"),
        "retained_root": str(tmp_path / "retained"),
        "retained_run": {"run_id": "modal-historical", "project_ref": "standalone-training"},
        "modal_profile": "reviewed-profile", "modal_environment": "synaptic-smoke-v1",
        "hf_token_file": str(tmp_path / "hf-token"),
        "model_ref": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_revision": "a91318be21aeaf0879874faa161dcb40c68847e9",
        "cpu_attempt_ref": "cpu-qual-" + "a" * 24,
        "diagnostic_attempt_ref": "grpo-diag-" + "b" * 24}
    path = tmp_path / "config.json"
    consumer._write(path, fields)
    return path


def test_private_config_and_canonical_records_reject_mutations(tmp_path):
    config = _configuration(tmp_path)
    parsed, digest = consumer._config(config)
    assert parsed["engine_root"] == tmp_path / "engine"
    assert len(digest) == 64
    record = tmp_path / "record.json"
    selected = consumer._write(record, {"schema_version": "candidate", "value": 1})
    assert consumer._document(record)[1] == selected
    with pytest.raises(FileExistsError):
        consumer._write(record, {"schema_version": "candidate", "value": 2})
    record.chmod(0o644)
    with pytest.raises(ValueError):
        consumer._document(record)


def test_explicit_token_file_is_private_and_exact(tmp_path, monkeypatch):
    config = _configuration(tmp_path)
    parsed, _ = consumer._config(config)
    token = parsed["hf_token_file"]
    monkeypatch.setenv("HF_TOKEN", "hf_ambient_must_not_be_used")
    consumer._write(token, {"wrong": "format"})
    with pytest.raises(ValueError):
        consumer._hf_token(parsed)
    token.unlink()
    token.write_bytes(b"hf_exactbytes")
    token.chmod(0o600)
    assert consumer._hf_token(parsed) == "hf_exactbytes"
    token.write_bytes(b"hf_exactbytes\n")
    with pytest.raises(ValueError):
        consumer._hf_token(parsed)


def test_build_review_digest_precedes_launcher_or_provider(monkeypatch, tmp_path):
    config = _configuration(tmp_path)
    candidate = {"schema_version": "build", "source_commit": "a" * 40}
    monkeypatch.setattr(consumer, "_plan", lambda *args: (candidate, object(), object()))
    monkeypatch.setattr(consumer, "validate_local_launcher", lambda: pytest.fail("launcher reached"))
    monkeypatch.setattr(consumer, "_client", lambda config: pytest.fail("client reached"))
    first = consumer.run("plan", config)
    assert first == hashlib.sha256(canonical_bytes(candidate)).hexdigest()
    with pytest.raises(ValueError):
        consumer.run("prepare", config, reviewed="0" * 64)
    assert consumer._document(tmp_path / "private" / "build-admission.json")[1] == first


def test_execution_review_digest_precedes_metadata_or_client(monkeypatch, tmp_path):
    config = _configuration(tmp_path)
    candidate = {"schema_version": "build", "source_commit": "a" * 40}
    monkeypatch.setattr(consumer, "_plan", lambda *args: (candidate, object(), object()))
    consumer.run("plan", config)
    root = tmp_path / "private"
    capture = consumer._write(root / "runtime-capture.json", {"schema_version": "capture"})
    model = consumer._write(root / "model-manifest.json", {"schema_version": "model"})
    consumer._write(root / "execution-admission.json", {"schema_version": "execution"})
    monkeypatch.setattr(consumer, "validate_model_inventory", lambda value: None)
    monkeypatch.setattr(consumer, "_client", lambda config: pytest.fail("client reached"))
    monkeypatch.setattr(consumer, "_admission", lambda *args: pytest.fail("metadata reached"))
    with pytest.raises(ValueError):
        consumer.run("execute-once", config, reviewed="0" * 64,
                     reviewed_capture=capture, reviewed_model=model)


def test_help_is_a_success_without_opening_config(capsys):
    assert consumer.main(["--help"]) == 0
    assert "UNAVAILABLE" not in capsys.readouterr().out


def test_prepare_composes_normal_cpu_qualification_with_explicit_token(monkeypatch, tmp_path):
    config = _configuration(tmp_path)
    token = tmp_path / "hf-token"
    token.write_bytes(b"hf_explicit_fixture")
    token.chmod(0o600)
    candidate = {"schema_version": "build", "source_commit": "a" * 40}
    plan = object()
    monkeypatch.setattr(consumer, "_plan", lambda *args: (candidate, plan, object()))
    reviewed = consumer.run("plan", config)
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"fixed-wheel")
    wheel_sha = hashlib.sha256(b"fixed-wheel").hexdigest()
    closed = []
    monkeypatch.setattr(consumer, "validate_local_launcher", lambda: None)
    monkeypatch.setattr(consumer, "prepare_modal_standalone_local", lambda **kwargs:
        SimpleNamespace(wheel=wheel, close=lambda: closed.append(True)))
    monkeypatch.setattr(consumer, "_client", lambda config: (SimpleNamespace(__version__="1.5.4"), object()))
    def qualify(**kwargs):
        assert kwargs["plan"] is plan
        assert kwargs["hf_token"] == "hf_explicit_fixture"
        assert kwargs["effect_id"] == "cpu-qual-" + "a" * 24
        assert kwargs["resources"].accelerator == "L40S"
        release = SimpleNamespace(package_digest=wheel_sha, to_dict=lambda: {"release": "fixed"})
        runtime = SimpleNamespace(release=release,
            deployment_facts=SimpleNamespace(to_dict=lambda: {"deployment": "fixed"}),
            facts=SimpleNamespace(to_dict=lambda: {"facts": "fixed"}),
            quote=SimpleNamespace(to_dict=lambda: {"quote": "fixed"}),
            volume_names_by_id=(("vo-new", "new-volume"),), qualification_key=b"q" * 32)
        qualification = SimpleNamespace(output_sha256="1" * 64,
            provider_call_id="fc-fixture", receipt=SimpleNamespace(to_dict=lambda: {"receipt": "fixed"}))
        return None, None, None, runtime, qualification
    monkeypatch.setattr(consumer, "_qualified_runtime", qualify)
    captured_sha = consumer.run("prepare", config, reviewed=reviewed)
    capture, digest = consumer._document(tmp_path / "private" / "runtime-capture.json")
    assert digest == captured_sha
    assert capture["wheel_sha256"] == wheel_sha
    assert capture["cpu_qualification"]["training_executed"] is False
    assert closed == [True]


def test_admit_then_execute_rederives_exact_reviewed_record(monkeypatch, tmp_path):
    from tuner.training.modal_host_storage import ModalHostStorageV1
    config = _configuration(tmp_path)
    candidate = {"schema_version": "build", "source_commit": "a" * 40}
    monkeypatch.setattr(consumer, "_plan", lambda *args: (candidate, object(), object()))
    build_sha = consumer.run("plan", config)
    root = tmp_path / "private"
    wheel = root / "source-wheel.whl"
    wheel.write_bytes(b"fixed-wheel")
    wheel.chmod(0o600)
    wheel_sha = hashlib.sha256(b"fixed-wheel").hexdigest()
    capture = {"build_admission_sha256": build_sha, "wheel_sha256": wheel_sha}
    capture_sha = consumer._write(root / "runtime-capture.json", capture)
    model_sha = consumer._write(root / "model-manifest.json", {"members": []})
    monkeypatch.setattr(consumer, "validate_model_inventory", lambda model: None)
    sdk, client = object(), object()
    runtime = SimpleNamespace(deployment_facts=SimpleNamespace(client_binding=object()))
    snapshot = object()
    monkeypatch.setattr(consumer, "_client", lambda config: (sdk, client))
    monkeypatch.setattr(consumer, "_read_capture", lambda *args, **kwargs:
        (capture, runtime, snapshot))
    admission = SimpleNamespace(attempt_ref="grpo-diag-" + "b" * 24,
        source_commit="a" * 40, host_source_commit="c" * 40,
        wheel_sha256=wheel_sha, runtime_release_digest="d" * 64,
        deployment_facts_digest="e" * 64, image_id="im-fixed",
        snapshot_deployment_facts_digest="f" * 64,
        model_cache_volume_id="vo-old", model_cache_volume_name="old-cache",
        quote_digest="1" * 64, dispatch_bytes=b"exact-dispatch")
    calls = []
    def admit(*args, **kwargs):
        calls.append("admit")
        return admission
    monkeypatch.setattr(consumer, "_admission", admit)
    reviewed = consumer.run("admit", config, reviewed_capture=capture_sha,
                            reviewed_model=model_sha)
    with ModalHostStorageV1(root / "journal.sqlite3", "grpo-constructor-diagnostic"):
        pass
    def execute(**kwargs):
        assert kwargs["sdk"] is sdk and kwargs["client"] is client
        assert kwargs["admission"] is admission
        assert kwargs["snapshot_runtime"] is snapshot
        calls.append("execute")
        return {"body": {"primary": "INDETERMINATE", "cleanup": "TERMINAL_CONFIRMED",
                         "failure_phase": "OUTPUT", "result": None}, "tag_hex": "00"}
    monkeypatch.setattr(consumer, "execute_modal_grpo_constructor_once", execute)
    outcome = consumer.run("execute-once", config, reviewed=reviewed,
                           reviewed_capture=capture_sha, reviewed_model=model_sha)
    assert outcome["primary"] == "INDETERMINATE"
    assert calls == ["admit", "admit", "execute"]


def test_readback_restart_uses_only_reviewed_retained_material(monkeypatch, tmp_path):
    from tuner.training.modal_grpo_constructor_diagnostic import _catalog, _receipt
    from tuner.training.modal_host_storage import ModalHostStorageV1

    config_path = _configuration(tmp_path)
    config, config_sha = consumer._config(config_path)
    root = consumer.private_directory(config["private_root"], create=True)
    build = {"schema_version": "synaptic-private-grpo-constructor-build-admission/v1",
        "config_sha256": config_sha, "diagnostic_attempt_ref": config["diagnostic_attempt_ref"],
        "cpu_attempt_ref": config["cpu_attempt_ref"], "model_ref": config["model_ref"],
        "model_revision": config["model_revision"], "environment_ref": config["modal_environment"],
        "account_ref": "synaptic-labs", "workspace_ref": "synaptic-labs",
        "engine_commit": "a" * 40, "host_commit": "b" * 40,
        "old_deployment_facts_digest": "c" * 64,
        "old_model_volume_id": "vo-old", "old_model_volume_name": "old-cache"}
    build_sha = consumer._write(root / "build-admission.json", build)
    model_sha = consumer._write(root / "model-manifest.json", {"members": []})
    wheel_sha = "d" * 64
    capture = {"schema_version": "synaptic-private-grpo-constructor-runtime-capture/v1",
        "build_admission_sha256": build_sha, "wheel_sha256": wheel_sha,
        "cpu_qualification": {"training_executed": False}, "quote": {"quote_digest": "e" * 64},
        "runtime": {"retained": True}}
    capture_sha = consumer._write(root / "runtime-capture.json", capture)
    key = b"k" * 32
    release = SimpleNamespace(manifest_digest="f" * 64, package_digest=wheel_sha)
    deployment = SimpleNamespace(facts_digest="1" * 64, image_id="im-new")
    facts = SimpleNamespace(account_ref="synaptic-labs", workspace_ref="synaptic-labs",
                            environment_ref="synaptic-smoke-v1")
    monkeypatch.setattr(consumer, "_quote", lambda value: object())
    monkeypatch.setattr(consumer, "_runtime_components", lambda value:
        (release, deployment, facts, (), key))
    monkeypatch.setattr(consumer, "validate_model_inventory", lambda value: None)
    admission = SimpleNamespace(attempt_ref=config["diagnostic_attempt_ref"],
        dispatch_bytes=b"exact-dispatch", source_commit=build["engine_commit"],
        host_source_commit=build["host_commit"], wheel_sha256=wheel_sha,
        runtime_release_digest=release.manifest_digest,
        deployment_facts_digest=deployment.facts_digest, image_id=deployment.image_id,
        snapshot_deployment_facts_digest=build["old_deployment_facts_digest"],
        model_cache_volume_id="vo-old", model_cache_volume_name="old-cache",
        quote_digest=capture["quote"]["quote_digest"])
    record = consumer._admission_record(config, admission, capture_sha, model_sha)
    reviewed = consumer._write(root / "execution-admission.json", record)
    runtime = SimpleNamespace(release=release, deployment_facts=deployment,
                              qualification_key=key)
    receipt = _receipt(runtime, admission, sandbox_id="sb-fixed", primary="REJECTED",
                       cleanup="TERMINAL_CONFIRMED", result=None, terminal=1,
                       failure_phase="OUTPUT")
    with ModalHostStorageV1(root / "journal.sqlite3", "grpo-constructor-diagnostic") as storage:
        _catalog(storage, "modal-grpo-constructor-results").publish_if_absent(
            config["diagnostic_attempt_ref"], receipt)
    (tmp_path / "recipe.yaml").write_text("changed recipe")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "dirty").write_text("advanced source")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    for forbidden in ("_plan", "_client", "_admission", "execute_modal_grpo_constructor_once"):
        monkeypatch.setattr(consumer, forbidden,
            lambda *args, **kwargs: pytest.fail("readback reached live authority"))
    result = consumer.run("readback", config_path, reviewed=reviewed,
                          reviewed_capture=capture_sha, reviewed_model=model_sha)
    assert result["primary"] == "REJECTED"
    assert result["cleanup"] == "TERMINAL_CONFIRMED"
    with pytest.raises(ValueError):
        consumer.run("readback", config_path, reviewed="0" * 64,
                     reviewed_capture=capture_sha, reviewed_model=model_sha)
