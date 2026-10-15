"""Host CLI over real retained Modal/native APIs; scripted SDK/ML, no live effects.

The CPU fixture uses GRPO's real native five-role path to test this method-neutral
publication join. The separately retained live SFT run supplies qualification.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synaptic-tuner"))
spec = importlib.util.spec_from_file_location(
    "modal_publication_host", ROOT / "publish_artifacts.py"
)
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.training.test_modal_grpo_standalone_runner import _joined
from tuner.training import modal_retained_reader as reader
from tuner.training import modal_standalone_runner as runner


def config(tmp_path, root, run):
    raw = (root / "readback-inputs.json").read_bytes()
    return {
        "schema_version": "synaptic-local-publication-host/v1",
        "consumer": {
            "path": str(root / "readback-inputs.json"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
        "run": run.to_dict(),
        "modal": {"root": str(root), "profile": "fixture", "environment": "main"},
        "root": str(tmp_path / "publication"),
        "destination_ref": "local/native",
        "authority_ref": "publication-host",
        "key_ref": "publication-key",
        "secret": {"provider": "env", "name": "PUBLICATION_KEY"},
        "maximum_artifact_bytes": 4 << 20,
        "maximum_total_bytes": 8 << 20,
        "seconds": 60,
    }


def completed(tmp_path, monkeypatch, request):
    plan, context, events, _, cpu, trainer, _, _ = _joined(
        tmp_path, monkeypatch, request, remove_source=True
    )
    result = runner.run_modal_standalone_job(
        plan=plan, context=context, modal_profile="fixture", modal_environment="main"
    )
    root = tmp_path / "state/synaptic-training" / result.run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=result.run)
    facts = reader.ModalPackagedRuntimeFactsV1.from_dict(
        retained.inputs["runtime_facts"]
    )
    calls = []
    monkeypatch.setattr(reader, "validate_local_launcher", lambda: None)

    def scope(**kwargs):
        calls.append("scope")
        return runner.open_modal_host_scope(**kwargs)

    monkeypatch.setattr(reader, "open_modal_host_scope", scope)
    monkeypatch.setattr(
        reader,
        "observe_modal_host_scope",
        lambda **kwargs: ("workspace", "workspace", "main", "train-config-host"),
    )
    monkeypatch.setattr(
        reader._CurrentPackagedDeploymentReader, "observe", lambda *a, **kw: facts
    )
    monkeypatch.setenv("PUBLICATION_KEY", "fixture-publication-" + "k" * 40)
    assert len(cpu) == len(trainer) == 1
    return root, result, events, calls


def test_modal_native_cli_publish_then_fresh_lookup_no_streams(
    tmp_path, monkeypatch, request, capsys
):
    root, training, events, calls = completed(tmp_path, monkeypatch, request)
    publication = config(tmp_path, root, training.run)
    path = tmp_path / "publication.json"
    path.write_text(json.dumps(publication))
    consumer.main(["publish", "--config", str(path)])
    first = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert first["verified"] and len(first["publication"]["artifacts"]) == 5
    before = {
        p.relative_to(tmp_path / "publication"): p.read_bytes()
        for directory in ("destination", "spool")
        for p in (tmp_path / "publication" / directory).rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("lookup reopened native streams or training operation")

    monkeypatch.setattr(RunsAPI, "artifacts", forbidden)
    monkeypatch.setattr(RunsAPI, "outcome", forbidden)
    consumer.main(["lookup", "--config", str(path)])
    assert json.loads(capsys.readouterr().out) == first
    after = {
        p.relative_to(tmp_path / "publication"): p.read_bytes()
        for directory in ("destination", "spool")
        for p in (tmp_path / "publication" / directory).rglob("*")
        if p.is_file()
    }
    assert before == after and calls == ["scope", "scope"]
    assert events == ["local", "bootstrap", "cpu", "training"]
    with pytest.raises(FileExistsError):
        consumer.main(["publish", "--config", str(path)])


@pytest.mark.parametrize("mutation", ["commitment", "run", "authority_key"])
def test_modal_original_binding_refuses_before_publication(
    tmp_path, monkeypatch, request, mutation
):
    root, training, _, calls = completed(tmp_path, monkeypatch, request)
    publication = config(tmp_path, root, training.run)
    if mutation == "commitment":
        publication["consumer"]["sha256"] = "0" * 64
    elif mutation == "run":
        publication["run"] = TrainingRunRef(
            "different", "standalone-training"
        ).to_dict()
    else:
        signing = root / "authority-keys.json"
        keys = json.loads(signing.read_bytes())

        # Use the real original private-key file shape without disclosing values.
        def replace(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "SYNAPTIC_TRAIN_AUTHORITY_KEY":
                        value[key] = "0" * 64
                        return True
                    if replace(item):
                        return True
            return False

        assert replace(keys)
        signing.write_text(json.dumps(keys, sort_keys=True, separators=(",", ":")))
    path = tmp_path / "publication.json"
    path.write_text(json.dumps(publication))
    with pytest.raises((ValueError, reader.ModalRetainedReadUnavailable)):
        consumer.main(["publish", "--config", str(path)])
    assert not (tmp_path / "publication").exists()
    assert not calls


@pytest.mark.parametrize("mutation", ["extra", "root", "profile", "path"])
def test_modal_config_rejects_before_sdk(tmp_path, mutation):
    value = {
        "schema_version": "synaptic-local-publication-host/v1",
        "consumer": {
            "path": "/original/readback-inputs.json",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "run": {"run_id": "original", "project_ref": "project"},
        "modal": {"root": "/original", "profile": "fixture", "environment": "main"},
        "root": str(tmp_path / "publication"),
        "destination_ref": "local/native",
        "authority_ref": "authority",
        "key_ref": "key",
        "secret": {"provider": "env", "name": "PUBLICATION_KEY"},
        "maximum_artifact_bytes": 1024,
        "maximum_total_bytes": 4096,
        "seconds": 60,
    }
    if mutation == "extra":
        value["modal"]["extra"] = True
    elif mutation == "root":
        value["modal"]["root"] = "relative"
    elif mutation == "profile":
        value["modal"]["profile"] = ""
    else:
        value["consumer"]["path"] = "/foreign/readback-inputs.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        consumer.load(path)


def test_cli_refusal_uses_fixed_diagnostic_without_secret_text(
    tmp_path, monkeypatch, capsys
):
    import runpy

    private = "PRIVATE_SECRET_SENTINEL"
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"private": private}))
    monkeypatch.setenv("PUBLICATION_KEY", private)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(ROOT / "publish_artifacts.py"), "publish", "--config", str(config)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / "publish_artifacts.py"), run_name="__main__")
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and output.err == "local_publication_unavailable\n"
    assert private not in output.out + output.err
