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


@pytest.fixture
def tokenizer_case(case, monkeypatch):
    from dataclasses import asdict, replace
    from types import ModuleType, SimpleNamespace
    from tests.runtime.test_native_mlx_grpo import inputs
    from tests.runtime.test_native_mlx_grpo_tokenization import MappingDefaultTokenizer
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    from tuner.training.native_mlx_grpo_compilation import compile_native_grpo

    root, configuration, _ = case
    status, summary = prepare.run("prepare", configuration)
    assert status == 0
    prepared = Path(summary["output_root"])
    training = TrainingInputV1.from_json((prepared / "training-input.json").read_text())
    source = PreparedTrainingInputIdentity.from_dict(strict_json((prepared / "identity.json").read_bytes()))
    profile = replace(inputs()[2], model_ref=training.model.ref, model_revision=training.model.revision,
        tokenizer_revision=training.model.tokenizer_revision)
    projection = compile_native_grpo(training, source, profile)
    (root / "workload.json").write_bytes(projection.workload.canonical_bytes)
    token_root = root / "tokenizer"
    token_root.mkdir(mode=0o700)
    (token_root / "tokenizer.json").write_bytes(b'{"fixture":true}')
    (token_root / "tokenizer_config.json").write_bytes(b'{"fixture":true}')
    with NativeTree(token_root, private=True) as tree:
        snapshot = NativeModelSnapshot(profile.model_ref, profile.tokenizer_revision, str(token_root), tree.inventory())
    (root / "tokenizer-snapshot.json").write_bytes(canonical(asdict(snapshot)))
    config = {"schema": prepare.TOKENIZER_SCHEMA, "workload_file": "workload.json",
        "prepared_publication": summary["publication"], "tokenizer_snapshot_file": "tokenizer-snapshot.json",
        "transformers_version": "5.20.2", "tokenizer_budget": {"per_file": 1024, "aggregate": 4096},
        "maximum_token_evidence_bytes": 65536, "output_root": "tokenizer-profile"}
    path = root / "tokenizer-profile-config.json"
    path.write_bytes(canonical(config))
    tokenizer, loads = MappingDefaultTokenizer(), []
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        # Third-party output must not escape the standalone host summary.
        print("PRIVATE-TOKENIZER-MARKER")
        print("PRIVATE-TOKENIZER-MARKER", file=sys.stderr)
        return tokenizer
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", module)
    original = prepare.importlib.metadata.version
    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda name:
        "5.20.2" if name == "transformers" else original(name))
    return SimpleNamespace(root=root, configuration=path, config=config, projection=projection,
        snapshot=snapshot, tokenizer=tokenizer, loads=loads)


def test_tokenizer_profile_exact_shared_ids_private_evidence_and_collision(tokenizer_case, capsys):
    t = tokenizer_case
    status, result = prepare.run("profile-tokenizer", t.configuration)
    assert status == 0 and result["status"] == "TOKENIZER_ONLY_COMPLETE"
    assert result["rows"] == 2 and result["total_prompt_tokens"] == 6
    assert t.loads == [((t.snapshot.root,), {"local_files_only": True, "trust_remote_code": False,
        "use_fast": t.projection.tokenization.use_fast})]
    assert capsys.readouterr() == ("", "")
    output = t.root / t.config["output_root"]
    raw = (output / "token-evidence.json").read_bytes()
    evidence = strict_json(raw)
    assert evidence["prompt_tokens"] == [[41, 42, 43], [41, 42, 43]]
    assert evidence["eos_token_id"] == 99 and evidence["return_dict"] is False
    assert evidence["add_generation_prompt"] is True
    assert evidence["tokenization"] == t.projection.tokenization.to_dict()
    assert evidence["source"] == t.projection.source.to_dict()
    assert result["token_evidence_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o700
    with NativeTree(output, private=True) as tree:
        assert set(tree.members()) == {"configuration.json", "token-evidence.json", "result.json"}
        before = tree.inventory()
        assert prepare.run("profile-tokenizer", t.configuration)[1]["phase"] == "output"
        assert tree.inventory() == before
    assert len(t.loads) == 1
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())


@pytest.mark.parametrize("fault,phase", [("workload", "input_admission"), ("source", "input_admission"),
    ("snapshot", "input_admission"), ("version", "tokenizer_load"), ("tokens", "input_tokenization"),
    ("during_load", "custody"), ("evidence", "publication")])
def test_tokenizer_profile_refuses_changed_inputs_or_bounds_and_preserves_failures(tokenizer_case, monkeypatch, fault, phase):
    t = tokenizer_case
    if fault == "workload":
        value = strict_json((t.root / "workload.json").read_bytes())
        value["controls"]["beta"] = .9
        (t.root / "workload.json").write_bytes(canonical(value))
    elif fault == "source":
        publication = Path(t.config["prepared_publication"])
        (publication / "dataset.jsonl").chmod(0o600)  # fixture-owned publication only
        (publication / "dataset.jsonl").write_bytes(b'{"prompt":"changed"}\n')
    elif fault == "snapshot":
        (Path(t.snapshot.root) / "tokenizer.json").write_bytes(b'changed')
    elif fault == "version":
        t.config["transformers_version"] = "0.0.1"
    elif fault == "tokens":
        monkeypatch.setattr(t.tokenizer, "apply_chat_template", lambda *a, **k: {"input_ids": [1]})
    elif fault == "during_load":
        auto = sys.modules["transformers"].AutoTokenizer
        load = auto.from_pretrained
        def load_then_tamper(*args, **kwargs):
            value = load(*args, **kwargs)
            (Path(t.snapshot.root) / "tokenizer.json").write_bytes(b"changed during load")
            return value
        monkeypatch.setattr(auto, "from_pretrained", load_then_tamper)
    else:
        t.config["maximum_token_evidence_bytes"] = 1
    t.configuration.write_bytes(canonical(t.config))
    status, result = prepare.run("profile-tokenizer", t.configuration)
    assert status == 2 and result["phase"] == phase and result["automatic_retry"] is False
    output = t.root / t.config["output_root"]
    if phase == "input_admission":
        assert not output.exists() and not t.loads
    else:
        assert (output / "configuration.json").exists()
        assert strict_json((output / "failure.json").read_bytes()) == result
        assert not (output / "result.json").exists() and not (output / "token-evidence.json").exists()
        with NativeTree(output, private=True) as tree:
            before = tree.inventory()
            # A tampered source is refused even before the occupied output is
            # reached. Neither admission refusal nor collision repairs it.
            expected = "input_admission" if fault == "during_load" else "output"
            assert prepare.run("profile-tokenizer", t.configuration)[1]["phase"] == expected
            assert tree.inventory() == before


def test_tokenizer_profile_darwin_path_without_changing_linux_preparation(tokenizer_case, monkeypatch):
    from types import SimpleNamespace
    from tuner.training import grpo_preparation

    def linux_only(*args, **kwargs):
        raise AssertionError("Linux-only publication custody must not run on Darwin")

    monkeypatch.setattr(grpo_preparation, "_private", linux_only)
    monkeypatch.setattr(prepare, "sys", SimpleNamespace(platform="darwin"))
    assert prepare.run("profile-tokenizer", tokenizer_case.configuration)[0] == 0
    assert prepare.run("prepare", tokenizer_case.configuration)[1]["phase"] == "configuration"


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_tokenizer_profile_accepts_sealed_original_or_native_copy(tokenizer_case, mode):
    publication = Path(tokenizer_case.config["prepared_publication"])
    for path in publication.iterdir():
        path.chmod(mode)
    assert prepare.run("profile-tokenizer", tokenizer_case.configuration)[0] == 0


@pytest.mark.parametrize("fault", ["extra", "writable", "hardlink", "symlink"])
def test_tokenizer_profile_rejects_publication_custody_before_load(tokenizer_case, fault):
    t = tokenizer_case
    publication = Path(t.config["prepared_publication"])
    member = publication / "dataset.jsonl"
    if fault == "extra":
        (publication / "extra.json").write_bytes(b"{}")
    elif fault == "writable":
        member.chmod(0o640)
    elif fault == "hardlink":
        os.link(member, t.root / "hardlink.jsonl")
    else:
        member.rename(t.root / "original.jsonl")
        member.symlink_to(t.root / "original.jsonl")
    status, result = prepare.run("profile-tokenizer", t.configuration)
    assert status == 2 and result["phase"] == "input_admission"
    assert not t.loads and not (t.root / t.config["output_root"]).exists()


def test_tokenizer_profile_rejects_identical_publication_replacement_after_load(tokenizer_case, monkeypatch):
    t = tokenizer_case
    member = Path(t.config["prepared_publication"]) / "dataset.jsonl"
    original = sys.modules["transformers"].AutoTokenizer.from_pretrained

    def replacing_load(*args, **kwargs):
        raw = member.read_bytes()
        # Retain the original inode to exclude inode reuse from the test.
        member.rename(t.root / "original.jsonl")
        member.write_bytes(raw)
        member.chmod(0o600)
        return original(*args, **kwargs)

    monkeypatch.setattr(sys.modules["transformers"].AutoTokenizer, "from_pretrained", replacing_load)
    status, result = prepare.run("profile-tokenizer", t.configuration)
    assert status == 2 and result["phase"] == "custody"
    output = t.root / t.config["output_root"]
    assert (output / "failure.json").exists()
    assert not (output / "token-evidence.json").exists()


def test_actual_tokenizer_profile_cli_with_stubs_blocks_model_imports_and_network(tokenizer_case):
    t = tokenizer_case
    code = '''
import importlib.abc, importlib.metadata, runpy, socket, sys, types
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'torch', 'modal', 'huggingface_hub'}:
            raise AssertionError('unexpected ML/provider import')
sys.meta_path.insert(0, Guard())
def forbidden(*a, **k):
    raise AssertionError('unexpected network')
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = forbidden
module = types.ModuleType('transformers')
class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, *, tokenize=True, return_dict=True, add_generation_prompt=False, **kw):
        assert tokenize is True and add_generation_prompt is True and kw == {'enable_thinking':False}
        return {'input_ids':[41,42,43], 'attention_mask':[1,1,1]} if return_dict else [41,42,43]
module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
sys.modules['transformers'] = module
version = importlib.metadata.version
importlib.metadata.version = lambda name: '5.20.2' if name == 'transformers' else version(name)
script, configuration = sys.argv[1:]
sys.argv = [script, 'profile-tokenizer', configuration]
runpy.run_path(script, run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(EXAMPLE / "prepare.py"), str(t.configuration)],
        cwd=t.root, capture_output=True, timeout=15)
    assert result.returncode == 0 and result.stderr == b"" and len(result.stdout) <= 8192
    assert json.loads(result.stdout)["status"] == "TOKENIZER_ONLY_COMPLETE"
