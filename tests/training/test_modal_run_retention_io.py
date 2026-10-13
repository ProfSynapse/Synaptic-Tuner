"""Private retention custody failures cannot silently mint a readable bundle."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
import tuner.training.modal_run_retention as retention


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "run"
    root.mkdir(mode=0o700)
    run = TrainingRunRef("modal-test", "project")
    scope = {key: "test" for key in ("account_ref", "workspace_ref", "environment_ref", "client_ref", "sdk_version")}
    inputs = {
        "prepared_request": {"project_ref": "project"},
        "components": {
            "execution_source": {"run_ref": run.run_id},
            "execution_context": retention.retained_bytes(b'{"context":1}'),
            "resolved_config": retention.retained_bytes(b'{"rate":0.125}'),
            "runtime": {}, "resources": {}, "artifact_policy": {},
        },
        "workload": retention.retained_bytes(b'{"learning_rate":0.125}'),
        "runtime_release": {}, "runtime_facts": {"scope": scope},
        "deployment_facts": {"scope": scope}, "client_binding": scope,
        "volume_names": [], "modal_profile": "test", "modal_environment": "test",
        "profile_digest": "a" * 64, "quote_digest": "b" * 64,
        "maximum_cost_minor_units": 100, "attempt_journal": str(tmp_path / "journal.sqlite3"),
        "workflow_file": retention.WORKFLOW_FILE, "engine_root": str(tmp_path),
        "cpu_receipt": {}, "training_plan": {}, "training_preflight": {},
    }
    keys = {"SYNAPTIC_TRAIN_AUTHORITY_KEY": "c" * 64, "SYNAPTIC_TRAIN_READER_KEY": "d" * 64}
    return dict(root=root, run=run, signing_keys=keys, qualification_key=b"k" * 32, inputs=inputs)


def test_roundtrip_exact_keys_and_fractional_canonical_bytes_are_private(bundle):
    retention.retain_modal_run(**bundle)
    loaded = retention.load_retained_modal_run(bundle["root"], expected_run=bundle["run"])
    assert dict(loaded.signing_keys) == bundle["signing_keys"]
    assert loaded.qualification_key == bundle["qualification_key"]
    assert dict(loaded.inputs) == bundle["inputs"]
    assert retention._decode_bytes(loaded.inputs["workload"]) == b'{"learning_rate":0.125}'
    text = repr(loaded)
    assert "c" * 64 not in text and "d" * 64 not in text and "kkkk" not in text
    stored_inputs = (bundle["root"] / retention.INPUTS_FILE).read_text()
    assert "c" * 64 not in stored_inputs and "d" * 64 not in stored_inputs
    for name in (retention.KEYS_FILE, retention.INPUTS_FILE):
        assert (bundle["root"] / name).stat().st_mode & 0o777 == 0o600


def test_exclusive_collision_preserves_existing_bundle(bundle):
    retention.retain_modal_run(**bundle)
    previous = {name: (bundle["root"] / name).read_bytes() for name in (retention.KEYS_FILE, retention.INPUTS_FILE)}
    with pytest.raises(retention.ModalRunRetentionUnavailable):
        retention.retain_modal_run(**bundle)
    assert previous == {name: (bundle["root"] / name).read_bytes() for name in previous}


@pytest.mark.parametrize("fault", ["wrong_run", "mutated_inputs", "key_mode", "root_mode", "symlink", "missing_keys"])
def test_corrupt_or_unbound_custody_refuses_load(bundle, fault):
    retention.retain_modal_run(**bundle)
    root, expected = bundle["root"], bundle["run"]
    if fault == "wrong_run":
        expected = TrainingRunRef("other-run", "project")
    elif fault == "mutated_inputs":
        path = root / retention.INPUTS_FILE
        document = json.loads(path.read_bytes())
        document["inputs"]["maximum_cost_minor_units"] = 200
        path.write_bytes(retention._encode(document, retention._MAX_INPUTS))
    elif fault == "key_mode":
        (root / retention.KEYS_FILE).chmod(0o644)
    elif fault == "root_mode":
        root.chmod(0o755)
    elif fault == "symlink":
        original = root / retention.INPUTS_FILE
        original.rename(root / "original.json")
        original.symlink_to(root / "original.json")
    else:
        (root / retention.KEYS_FILE).unlink()
    with pytest.raises(retention.ModalRunRetentionUnavailable):
        retention.load_retained_modal_run(root, expected_run=expected)


def test_fsync_failure_retains_partial_attempt_without_secret_exception_context(bundle, monkeypatch):
    monkeypatch.setattr(retention.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("private-secret-message")))
    with pytest.raises(retention.ModalRunRetentionUnavailable) as caught:
        retention.retain_modal_run(**bundle)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert str(caught.value) == "modal_run_retention_unavailable"
    assert (bundle["root"] / retention.KEYS_FILE).exists()
    assert not (bundle["root"] / retention.INPUTS_FILE).exists()


@pytest.mark.parametrize("attack", ["root_replacement", "file_permission"])
def test_read_mutation_is_rejected_through_same_owned_descriptor(bundle, monkeypatch, attack):
    retention.retain_modal_run(**bundle)
    original = retention.os.read
    root = bundle["root"]
    attacked = False
    def mutate_during_read(descriptor, maximum):
        nonlocal attacked
        raw = original(descriptor, maximum)
        if not attacked:
            attacked = True
            if attack == "root_replacement":
                root.rename(root.with_name("retained-old"))
                root.mkdir(mode=0o700)
            else:
                (root / retention.KEYS_FILE).chmod(0o644)
        return raw
    monkeypatch.setattr(retention.os, "read", mutate_during_read)
    with pytest.raises(retention.ModalRunRetentionUnavailable):
        retention.load_retained_modal_run(root, expected_run=bundle["run"])


def test_invalid_key_inventory_rejects_before_writing(bundle):
    bundle["signing_keys"]["HF_TOKEN"] = "forbidden"
    with pytest.raises(retention.ModalRunRetentionUnavailable):
        retention.retain_modal_run(**bundle)
    assert not list(bundle["root"].iterdir())

@pytest.mark.parametrize("name", [retention.KEYS_FILE, retention.INPUTS_FILE])
def test_fifo_leaf_refuses_in_bounded_isolated_process(bundle, name):
    retention.retain_modal_run(**bundle)
    leaf = bundle["root"] / name
    leaf.unlink()
    os.mkfifo(leaf, 0o600)
    # A timeout kills the isolated child if a regression blocks in open().
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.training.modal_run_retention import load_retained_modal_run, ModalRunRetentionUnavailable
try:
    load_retained_modal_run(Path(sys.argv[2]), expected_run=TrainingRunRef("modal-test", "project"))
except ModalRunRetentionUnavailable:
    print("rejected")
else:
    raise AssertionError("FIFO was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(retention.__file__).resolve().parents[2]), str(bundle["root"])],
        capture_output=True, text=True, timeout=5, check=True,
    )
    assert result.stdout.strip() == "rejected"
