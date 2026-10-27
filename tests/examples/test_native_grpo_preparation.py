"""Real host preparation/publication controls; no model/runtime/provider effects."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from examples.training_qualification.native_grpo_preparation import prepare
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.training.grpo_preparation import GRPOPreparedInputFormatVerifierV1, snapshot_grpo_publication

EXAMPLE = Path(prepare.__file__).parent


@pytest.fixture
def case():
    previous = os.umask(0o077)
    try:
        # Shared proof-cache and /tmp ancestors can be writable. Never chmod
        # those roots or weaken NativeTree; test only under a fresh home root.
        with tempfile.TemporaryDirectory(prefix="native-grpo-prep-test-", dir=Path.home()) as name:
            root = Path(name)
            config = json.loads((EXAMPLE / "configuration.example.json").read_bytes())
            (root / "seed.jsonl").write_bytes((EXAMPLE / "seed.jsonl").read_bytes())
            configuration = root / "configuration.json"
            configuration.write_bytes(canonical(config))
            yield root, configuration, config
    finally:
        os.umask(previous)


def _write(configuration, config):
    configuration.write_bytes(canonical(config))


@pytest.mark.parametrize("mask", [0o022, 0o002])
def test_actual_service_publication_identity_canonical_request_and_private_modes(case, mask):
    root, configuration, config = case
    original = (root / "seed.jsonl").read_bytes()
    previous = os.umask(mask)
    try:
        status, inspected = prepare.run("inspect", configuration)
        assert status == 0 and inspected["status"] == "INPUT_BYTES_VERIFIED_ONLY"
        assert not (root / config["output_root"]).exists()
        status, summary = prepare.run("prepare", configuration)
    finally:
        os.umask(previous)
    assert status == 0 and summary["status"] == "PREPARED_INPUT_ONLY" and summary["rows"] == 2
    assert len(canonical(summary)) <= 8192 and summary["automatic_retry"] is False
    output, publication = Path(summary["output_root"]), Path(summary["publication"])
    for path in (output, output / "source", output / "publications"):
        assert path.stat().st_mode & 0o777 == 0o700
    for name in ("configuration.json", "source/seed.jsonl", "identity.json", "training-input.json",
                 "prepared-request.json", "result.json"):
        assert (output / name).stat().st_mode & 0o777 == 0o600
    assert (output / "source/seed.jsonl").stat().st_nlink == 1
    assert (output / "source/seed.jsonl").read_bytes() == original
    manifest, published = snapshot_grpo_publication(publication)
    assert published == original and manifest["row_count"] == 2
    identity = PreparedTrainingInputIdentity.from_dict(strict_json((output / "identity.json").read_bytes()))
    observed = GRPOPreparedInputFormatVerifierV1().verify(publication)
    assert observed.snapshot == original and observed.semantic_digest == identity.revision
    assert identity.content_digest == hashlib.sha256(original).hexdigest()
    assert identity.size_bytes == 118 and identity.ref == summary["identity_ref"]
    actual_raw = (output / "training-input.json").read_bytes()
    actual = TrainingInputV1.from_json(actual_raw.decode())
    expected = TrainingInputV1.from_dict(config["training_input"]).to_dict()
    expected["dataset"]["ref"] = identity.ref
    assert actual.to_dict() == expected and actual.canonical_json().encode() == actual_raw
    assert identity.ref != config["training_input"]["dataset"]["ref"]
    assert strict_json((output / "prepared-request.json").read_bytes())["identity"] == identity.to_dict()


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("size_bytes", 119)])
def test_source_commitment_mismatch_rejects_before_claim(case, field, value):
    root, configuration, config = case
    config["source"][field] = value
    _write(configuration, config)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] == "source" and result["automatic_retry"] is False
    assert not (root / config["output_root"]).exists()


def test_existing_private_output_collision_never_overwrites_or_adopts(case):
    root, configuration, config = case
    status, summary = prepare.run("prepare", configuration)
    assert status == 0
    output = root / config["output_root"]
    with NativeTree(output, private=True) as tree:
        before = tree.inventory()
        for action in ("inspect", "prepare"):
            status, refused = prepare.run(action, configuration)
            assert status == 2 and refused["phase"] == "output"
        assert tree.inventory() == before
        assert tree.read("result.json") == canonical(summary)
        assert "failure.json" not in tree.members()


def test_invalid_rows_preserve_owned_source_and_failure_without_replay(case):
    root, configuration, config = case
    raw = b'{"prompt":"PRIVATE-MARKER","prompt":"duplicate"}\n'
    (root / "seed.jsonl").write_bytes(raw)
    config["source"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    _write(configuration, config)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] == "preparation"
    assert "PRIVATE-MARKER" not in str(result)
    output = root / config["output_root"]
    assert (output / "source/seed.jsonl").read_bytes() == raw
    assert strict_json((output / "failure.json").read_bytes()) == result
    assert not (output / "identity.json").exists()
    with NativeTree(output, private=True) as tree:
        before = tree.inventory()
        assert prepare.run("prepare", configuration)[1]["phase"] == "output"
        assert tree.inventory() == before


def test_failed_verification_keeps_actual_committed_publication(case, monkeypatch):
    root, configuration, config = case
    def refuse(*args, **kwargs):
        raise ValueError("PRIVATE-EXCEPTION-MARKER")
    monkeypatch.setattr(GRPOPreparedInputFormatVerifierV1, "validate_training_input", refuse)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] == "preparation"
    output = root / config["output_root"]
    publications = list((output / "publications").glob("dataset-*"))
    assert len(publications) == 1
    _, raw = snapshot_grpo_publication(publications[0])
    assert raw == (root / "seed.jsonl").read_bytes()
    assert (output / "failure.json").exists() and not (output / "identity.json").exists()
    assert "PRIVATE-EXCEPTION-MARKER" not in str(result)


def test_failed_normalizer_write_retains_actual_partial_stage(case, monkeypatch):
    from tuner.dataset_prep import publication as storage
    root, configuration, config = case
    original = storage._write_stage_member
    def fail_after_dataset(stage, name, raw):
        original(stage, name, raw)
        if name == "dataset.jsonl":
            raise OSError("PRIVATE-STAGE-MARKER")
    monkeypatch.setattr(storage, "_write_stage_member", fail_after_dataset)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] == "preparation"
    output = root / config["output_root"]
    with NativeTree(output / "publications", private=True) as tree:
        staged = [name for name in tree.members() if name.endswith("/dataset.jsonl")]
        assert len(staged) == 1
        assert tree.read(staged[0]) == (root / "seed.jsonl").read_bytes()
        assert not any(name.endswith("/manifest.json") for name in tree.members())
    assert (output / "failure.json").exists() and "PRIVATE-STAGE-MARKER" not in str(result)


def test_prompt_column_mismatch_refuses_before_private_output(case):
    root, configuration, config = case
    config["source"]["prompt_column"] = "question"
    _write(configuration, config)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] == "configuration"
    assert not (root / config["output_root"]).exists()


@pytest.mark.parametrize("member,phase", [("configuration.json", "configuration"), ("seed.jsonl", "source")])
def test_group_writable_input_copies_are_refused_without_permission_repair(case, member, phase):
    root, configuration, config = case
    (root / member).chmod(0o660)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] == phase
    assert (root / member).stat().st_mode & 0o777 == 0o660
    assert not (root / config["output_root"]).exists()


@pytest.mark.parametrize("kind", ["symlink_seed", "hardlink_seed", "symlink_output", "writable_parent"])
def test_unsafe_or_occupied_paths_refuse_without_repair(case, kind):
    root, configuration, config = case
    if kind in ("symlink_seed", "hardlink_seed"):
        (root / "seed.jsonl").rename(root / "original.jsonl")
        if kind == "symlink_seed":
            (root / "seed.jsonl").symlink_to("original.jsonl")
        else:
            os.link(root / "original.jsonl", root / "seed.jsonl")
    elif kind == "symlink_output":
        (root / config["output_root"]).symlink_to("unrelated-target")
    else:
        # Only the fixture-owned directory is writable, never a shared parent.
        (root / "writable").mkdir(mode=0o770)
        (root / "writable").chmod(0o770)
        config["output_root"] = "writable/new-output"
        _write(configuration, config)
    status, result = prepare.run("prepare", configuration)
    assert status == 2 and result["phase"] in ("source", "output")
    assert not (root / "unrelated-target").exists()
    if kind == "writable_parent":
        assert (root / "writable").stat().st_mode & 0o777 == 0o770
        assert not (root / "writable/new-output").exists()


def test_extra_row_fields_are_preserved_without_becoming_the_prompt(case):
    root, configuration, config = case
    raw = canonical({"question": "Choose a color", "answer": "PRIVATE-METADATA"}) + b"\n"
    (root / "seed.jsonl").write_bytes(raw)
    config["source"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), prompt_column="question")
    config["training_input"]["hyperparameters"]["prompt_column"] = "question"
    _write(configuration, config)
    status, summary = prepare.run("prepare", configuration)
    assert status == 0 and "PRIVATE-METADATA" not in str(summary)
    manifest, actual = snapshot_grpo_publication(Path(summary["publication"]))
    assert actual == raw and manifest["prompt_column"] == "question"
    request = strict_json((Path(summary["output_root"]) / "training-input.json").read_bytes())
    assert request["hyperparameters"]["prompt_column"] == "question"


def test_example_proposal_matches_declared_mechanics_without_runtime_authority():
    config = strict_json((EXAMPLE / "configuration.example.json").read_bytes())
    training = TrainingInputV1.from_dict(config["training_input"])
    hp = training.hyperparameters
    proposal = strict_json((EXAMPLE / "qualification-controls.example.json").read_bytes())
    assert (hp.batch_size, hp.num_generations, hp.max_steps) == (4, 4, 3)
    assert (hp.max_prompt_length, hp.max_completion_length, hp.max_seq_length, hp.temperature) == (64, 8, 128, 1.)
    assert hp.chat_template_kwargs == {"enable_thinking": False} and hp.lora_rank == 4
    assert proposal["lora_layers"] == 2 and proposal["optimizer"]["schedule"] == "constant"
    assert hp.learning_rate == proposal["optimizer"]["learning_rate"] == .0001
    assert proposal["status"] == "ROOT_REVIEW_REQUIRED_NOT_EXECUTION_AUTHORITY"
    assert not {"runtime_digest", "source_digest", "admission", "profile_ref"} & set(proposal)


def test_actual_cli_has_closed_output_and_no_ml_or_network_effects(case):
    root, configuration, _ = case
    code = '''
import importlib.abc, runpy, socket, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'torch', 'transformers', 'modal', 'huggingface_hub'}:
            raise AssertionError('unexpected model/provider import')
sys.meta_path.insert(0, Guard())
def forbidden(*a, **k):
    raise AssertionError('unexpected network')
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = forbidden
script, configuration = sys.argv[1:]
sys.argv = [script, 'prepare', configuration]
runpy.run_path(script, run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(EXAMPLE / "prepare.py"), str(configuration)],
        cwd=root, capture_output=True, timeout=15)
    assert result.returncode == 0 and result.stderr == b"" and len(result.stdout) <= 8192
    assert json.loads(result.stdout)["status"] == "PREPARED_INPUT_ONLY"


def test_actual_cli_failure_has_no_exception_or_source_text(case):
    root, configuration, config = case
    raw = b'{"prompt":"PRIVATE-MARKER","prompt":"duplicate"}\n'
    (root / "seed.jsonl").write_bytes(raw)
    config["source"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    _write(configuration, config)
    result = subprocess.run([sys.executable, "-B", str(EXAMPLE / "prepare.py"), "prepare", str(configuration)],
        cwd=root, capture_output=True, timeout=15)
    assert result.returncode == 2 and result.stderr == b"" and len(result.stdout) <= 8192
    assert b"PRIVATE-MARKER" not in result.stdout
    assert json.loads(result.stdout) == {"schema": prepare.RESULT_SCHEMA, "status": "REFUSED",
        "phase": "preparation", "automatic_retry": False}
