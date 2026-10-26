from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import pytest

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_mlx_child import _PinnedFinder, _capture_document, run_request
from tuner.runtime.native_mlx_material import canonical
from tuner.training.native_mlx_compilation import NativeMLXProfile, compile_native_sft, parse_native_workload


def inputs():
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + "a" * 64, "a" * 64, "b" * 64, 100, "syntunia-sft-row/v1",
    )
    training = TrainingInputV1.from_dict({
        "schema_version": "synaptic-training-input/v1", "method": "sft",
        "model": {"ref": "configured/model", "revision": "c" * 40, "tokenizer_revision": "c" * 40},
        "dataset": {"ref": identity.ref},
        "hyperparameters": {
            "schema_version": "synaptic-sft-hyperparameters/v1",
            "batch_size": 1, "gradient_accumulation_steps": 1, "learning_rate": .0001,
            "duration": {"max_steps": 3, "num_epochs": None}, "max_seq_length": 128, "seed": 17,
            "save_steps": 3, "save_total_limit": 1, "lora_rank": 4, "lora_alpha": 32,
            "lora_dropout": 0, "lora_target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
            "use_dora": False, "use_rslora": False, "init_lora_weights": True,
            "split_dataset": False, "dataset_format": "raw_text",
            "completion_only_loss": False, "assistant_only_loss": False,
            "use_preassigned_splits": True, "prompt_render": "full_conversation",
            "packing": False, "require_memory_efficient_loss": False,
        },
        "artifacts": {"required_kinds": ["final_model", "training_lineage"], "retain_checkpoints": True},
    })
    profile = NativeMLXProfile(
        "candidate", "configured/model", "c" * 40, "c" * 40, "d" * 64, "e" * 64,
        2, "alpha_over_rank", .9, .999, 1e-8, 0, True, "constant", True, 1, 3, 3, 128,
    )
    return training, identity, profile


def test_public_input_compiles_exact_native_settings_and_reparses():
    projection = compile_native_sft(*inputs())
    assert projection.lora.scale == 8
    assert projection.controls.iterations == 3
    assert projection.optimizer.learning_rate == .0001
    assert projection.tokenization.assistant_only_loss is False
    assert parse_native_workload(projection.workload.canonical_bytes) == projection
    mutated = projection.workload.document
    mutated["controls"]["iterations"] = 4
    with pytest.raises(ValueError, match="compilation"):
        parse_native_workload(canonical(mutated))


@pytest.mark.parametrize("changes", [
    {"gradient_accumulation_steps": 2}, {"use_dora": True}, {"use_rslora": True},
    {"init_lora_weights": False}, {"packing": True}, {"split_dataset": True},
    {"require_memory_efficient_loss": True}, {"use_preassigned_splits": False},
    {"duration": {"max_steps": 4, "num_epochs": None}},
    {"duration": {"max_steps": None, "num_epochs": 1}},
    {"save_steps": 1},
])
def test_unsupported_public_controls_are_rejected_not_silently_ignored(changes):
    training, identity, profile = inputs()
    document = training.to_dict()
    document["hyperparameters"].update(changes)
    with pytest.raises(ValueError):
        compile_native_sft(TrainingInputV1.from_dict(document), identity, profile)


def test_profile_model_and_source_identity_do_not_float():
    training, identity, profile = inputs()
    with pytest.raises(ValueError, match="model profile"):
        compile_native_sft(training, identity, replace(profile, model_revision="f" * 40))
    with pytest.raises(ValueError, match="publication"):
        compile_native_sft(training, replace(identity, ref="prepared://sha256/" + "f" * 64,
                                            revision="f" * 64), profile)
    with pytest.raises(ValueError, match="alpha_over_rank"):
        replace(profile, lora_scale_policy="guess")


def test_public_completion_controls_project_to_shared_mask_operation():
    training, identity, profile = inputs()
    document = training.to_dict()
    document["hyperparameters"].update({
        "dataset_format": "messages", "completion_only_loss": True,
        "assistant_only_loss": False, "prompt_render": "prompt_completion",
        "chat_template_kwargs": {"enable_thinking": False},
    })
    projection = compile_native_sft(
        TrainingInputV1.from_dict(document), replace(identity, format="syntunia-sft-row/v2"), profile,
    )
    assert projection.tokenization.assistant_only_loss is True
    assert projection.tokenization.chat_template_json == b'{"enable_thinking":false}'


@pytest.fixture
def private_root():
    mask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-child-test-", dir=Path.home()) as path:
            yield Path(path)
    finally:
        os.umask(mask)


def test_import_finder_rejects_ambient_sibling_and_compiles_only_pinned_source(private_root):
    source = private_root / "owned_fixture.py"
    source.write_text("result = 17\n")
    finder = _PinnedFinder({str(source): hashlib.sha256(source.read_bytes()).hexdigest()})
    spec = finder.find_spec("owned_fixture", [str(private_root)])
    module = ModuleType("owned_fixture")
    spec.loader.exec_module(module)
    assert module.result == 17
    (private_root / "ambient_fixture.py").write_text("result = 99\n")
    with pytest.raises(ImportError, match="outside capture"):
        finder.find_spec("ambient_fixture", [str(private_root)])
    source.write_text("result = 23\n")
    with pytest.raises(ImportError, match="source bytes"):
        spec.loader.exec_module(module)


def test_worker_recompiles_and_joins_workload_before_invoking_runtime(private_root, monkeypatch):
    from tuner.runtime import native_mlx_input, native_mlx_composition
    projection = compile_native_sft(*inputs())
    output = private_root / "output"
    output.mkdir(mode=0o700)
    for name, content in (("capture.json", b"capture"), ("lock", b"lock"), ("dataset.jsonl", b"dataset")):
        (private_root / name).write_bytes(content)
    from dataclasses import asdict
    from tuner.runtime.native_files import NativeTree
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    model_root = private_root / "model"
    model_root.mkdir(mode=0o700)
    (model_root / "config.json").write_bytes(b"{}")
    with NativeTree(model_root, private=True) as model_tree:
        snapshot = asdict(NativeModelSnapshot(
            "configured/model", "c" * 40, str(model_root), model_tree.inventory(),
        ))
        snapshot["files"] = list(snapshot["files"])
    request = {
        "schema": "synaptic-native-child-request/v1", "workload": projection.workload.document,
        "runtime_capture": str(private_root / "capture.json"),
        "dependency_lock": str(private_root / "lock"), "dataset": str(private_root / "dataset.jsonl"),
        "model_snapshot": snapshot, "tokenizer_snapshot": snapshot, "output": str(output),
        "tokenizer_members": ["config.json"],
        "artifact_budget": {"per_file": 1024 * 1024, "aggregate": 4 * 1024 * 1024},
    }
    observed = {}

    def prepare(**kwargs):
        observed.update(kwargs)
        return "prepared-fixture"

    def execute(**kwargs):
        assert kwargs["bound"] == "prepared-fixture"
        assert kwargs["lora"] == projection.lora and kwargs["optimizer"] == projection.optimizer
        return b"verified-fixture-result"

    monkeypatch.setattr(native_mlx_input, "prepare_native_input", prepare)
    monkeypatch.setattr(native_mlx_composition, "execute_native_sft", execute)
    from tuner.runtime import native_mlx_publication, native_mlx_child
    monkeypatch.setattr(native_mlx_publication, "tokenizer_publication_bytes", lambda *a, **k: b"fixture")
    monkeypatch.setattr(native_mlx_child, "_load_published_tokenizer", lambda *a, **k: None)
    assert run_request(request, request_root=private_root) == b"verified-fixture-result"
    assert observed["workload_digest"] == projection.workload.fingerprint
    assert observed["runtime_digest"] == projection.profile.runtime_digest
    assert (output / "workload.json").read_bytes() == projection.workload.canonical_bytes


def test_child_requires_isolation_and_has_closed_diagnostics(private_root):
    from tuner.runtime import native_mlx_child
    path = Path(native_mlx_child.__file__)
    result = subprocess.run(
        [sys.executable, str(path), "--request", str(private_root / "missing")],
        capture_output=True, timeout=5,
    )
    assert result.returncode == 120 and result.stdout == result.stderr == b""


def test_child_bootstrap_accepts_only_canonical_v1_or_v2_capture():
    for version in ("v1", "v2"):
        value = {"schema": f"synaptic-native-mlx-capture/{version}",
                 "status": "OBSERVED_NOT_ADMITTED", "config": {}}
        if version == "v2":
            value.update(dynamic_system_libraries=["observed"], builtin_modules={},
                         dynamic_os_build={})
            value["config"].update(dynamic_system_libraries=["selected"],
                                   required_builtin_modules=["_ctypes"],
                                   expected_os_build={})
        raw = canonical(value)
        assert _capture_document(raw)["schema"].endswith(version)
        with pytest.raises(ValueError, match="capture schema"):
            _capture_document(raw + b"\n")
        if version == "v2":
            del value["config"]["expected_os_build"]
            with pytest.raises(ValueError, match="dynamic capture fields"):
                _capture_document(canonical(value))
    with pytest.raises(ValueError, match="capture schema"):
        _capture_document(canonical({"schema": "synaptic-native-mlx-capture/v3"}))


def test_child_loads_only_complete_published_tokenizer_bytes(private_root, monkeypatch):
    from tuner.runtime.native_mlx_child import _load_published_tokenizer
    from tuner.runtime.native_mlx_publication import _archive, NativeArtifactBudget
    loaded = []
    transformers = ModuleType("transformers")
    def from_pretrained(root, **kwargs):
        loaded.append(((Path(root) / "tokenizer.json").read_bytes(), kwargs))
        return object()
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    payload = _archive({"tokenizer.json": b"complete", "native_tokenizer.json": b"{}"},
                       NativeArtifactBudget(1024, 16384))
    _load_published_tokenizer(payload, request_root=private_root,
                              expected_members=("tokenizer.json",), use_fast=True)
    assert loaded == [(b"complete", {"local_files_only": True,
                                      "trust_remote_code": False, "use_fast": True})]
    with pytest.raises(ValueError, match="archive inventory"):
        _load_published_tokenizer(payload, request_root=private_root,
                                  expected_members=("tokenizer.json", "missing.json"), use_fast=True)
    assert not list(private_root.glob("tokenizer-publication-*"))
