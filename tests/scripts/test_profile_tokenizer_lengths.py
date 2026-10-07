import copy
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


LOCK_PATH = Path("tuner/execution/providers/modal/modal-runtime-v1.lock.json").resolve()
LOCK_SHA = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
EXAMPLE_CONFIG = Path(".skills/fine-tuning/configs/qwen35_4b_token_profile.yaml").resolve()


def _load_module():
    path = Path(".skills/fine-tuning/scripts/profile_tokenizer_lengths.py").resolve()
    spec = importlib.util.spec_from_file_location("tokenizer_length_profiler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self, name_or_path):
        self.name_or_path = name_or_path

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        count = 2 + sum(len(str(message["content"]).split()) for message in messages)
        return list(range(count + int(add_generation_prompt)))


def _install_fake_transformers(monkeypatch, callback=None):
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs, {key: os.environ.get(key) for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN")}))
        if callback:
            callback(Path(path))
        return FakeTokenizer(path)

    fake = types.ModuleType("transformers")
    fake.__version__ = "4.57.1"
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return calls


def _snapshot(root: Path):
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text('{"version":"1.0","model":{}}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"FakeTokenizer"}', encoding="utf-8")
    return root


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _limits():
    return {
        "max_input_bytes": 100_000,
        "max_rows": 1_000,
        "max_line_bytes": 10_000,
        "max_record_bytes": 9_999,
        "max_tokens_per_record": 1_000,
        "max_components": 10,
        "max_component_label_chars": 30,
        "max_snapshot_files": 10,
        "max_snapshot_file_bytes": 100_000,
        "max_snapshot_total_bytes": 200_000,
    }


def _config(snapshot: Path, dataset: Path, revision="a" * 40, mode="text"):
    input_config = {"jsonl_path": str(dataset), "mode": mode}
    if mode == "text":
        input_config["text_field"] = "secret"
    elif mode == "messages":
        input_config.update(messages_field="messages", use_chat_template=True, add_generation_prompt=False)
    else:
        input_config["components"] = {"completion": "answer", "prompt": "instruction"}
    return {
        "model": {"ref": "acme/test-model", "model_revision": revision, "tokenizer_revision": revision, "local_tokenizer_path": str(snapshot)},
        "runtime": {"provider": "modal", "lock_path": str(LOCK_PATH), "expected_lock_sha256": LOCK_SHA},
        "input": input_config,
        "budgets": {"max_sequence_tokens": 8, "completion_reserve_tokens": 2},
        "limits": _limits(),
    }


def _profile(tmp_path, monkeypatch, rows=None, mode="text"):
    module = _load_module()
    calls = _install_fake_transformers(monkeypatch)
    dataset = _jsonl(tmp_path / "data.jsonl", rows or [{"secret": "one two"}])
    result = module.profile(_config(_snapshot(tmp_path / "snapshot"), dataset, mode=mode))
    return module, result, calls






class ScaffoldTokenizer:
    chat_template = "different scaffold"
    eos_token_id = 999
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        assert kwargs.get("enable_thinking") is False or not kwargs
        text = " ".join(m["content"] for m in messages) + (" anchor disabled" if add_generation_prompt else " full closing scaffold extra")
        return self.encode(text) if tokenize else text


def test_exact_real_materializer_budget_privacy_and_verification(tmp_path, monkeypatch):
    from shared.sft_preprocessing import materialize_sft_example
    module = _load_module()
    row = {"schema_version": "syntunia-sft-row/v2", "format": "messages", "split": "train", "messages": [{"role": "user", "content": "private prompt"}, {"role": "assistant", "content": "private target"}]}
    config = _config(_snapshot(tmp_path / "snapshot"), _jsonl(tmp_path / "rows.jsonl", [row]), mode="messages")
    config["input"].update(prompt_render="prompt_completion", chat_template_kwargs={"enable_thinking": False, "private_setting": "private literal"})
    tokenizer = ScaffoldTokenizer()
    _install_fake_transformers(monkeypatch)
    def load(path, **kwargs):
        tokenizer.name_or_path = path
        return tokenizer
    monkeypatch.setattr(sys.modules["transformers"].AutoTokenizer, "from_pretrained", load)
    config["budgets"] = {"max_sequence_tokens": 2, "completion_reserve_tokens": 0}
    result = module.profile(config)
    actual = materialize_sft_example(tokenizer=tokenizer, record=row, max_seq_length=1000, assistant_only_loss=True, prompt_render="prompt_completion", chat_template_kwargs=config["input"]["chat_template_kwargs"])
    assert result["total"]["max"] == len(actual.input_ids) == 7
    assert actual.input_ids[-1] == tokenizer.eos_token_id
    assert result["total"]["max"] != len(tokenizer.apply_chat_template(row["messages"]))
    assert result["budget"]["over_limit_count"] == 1
    assert result["components"]["assistant"]["max"] == 2
    assert all(secret not in json.dumps(result) for secret in ("private prompt", "private target", "private literal", str(tmp_path)))
    published = module.publish_profile(result, tmp_path / "exact")
    module.verify_profile_directory(tmp_path / "exact.token-profile", published["profile_id"])
    tampered = copy.deepcopy(result)
    tampered["config_provenance"]["rendering"]["helper_source_sha256"] = "0" * 64
    with pytest.raises(module.ProfilerError, match="PROFILE_ARTIFACT_INVALID"):
        module._validate_profile_shape(tampered)
    tampered = copy.deepcopy(result)
    tampered["config_provenance"]["rendering"]["chat_template_kwargs_sha256"] = "0" * 64
    assert module._compute_profile_id(tampered) != result["profile_semantic_id"]
    normalized = module.validate_config(config)
    with pytest.raises(module.ProfilerError, match="TOKEN_LIMIT_EXCEEDED"):
        module._row_counts(tokenizer, row, normalized["input"], dict(normalized["limits"], max_tokens_per_record=5))
    row["split"] = "invalid"
    with pytest.raises(module.ProfilerError, match="CHAT_TEMPLATE_RENDER_FAILED"):
        module._row_counts(tokenizer, row, normalized["input"], normalized["limits"])
    row["split"] = "train"
    tokenizer.eos_token_id = None
    with pytest.raises(module.ProfilerError, match="CHAT_TEMPLATE_RENDER_FAILED"):
        module._row_counts(tokenizer, row, normalized["input"], normalized["limits"])
    monkeypatch.setattr(module, "SFT_HELPER_SHA256", "0" * 64)
    with pytest.raises(module.ProfilerError, match="SFT_HELPER_SOURCE_MISMATCH"):
        module._sft_materializer()


@pytest.mark.parametrize("kwargs", [{"tokenize": False}, {"truncation": True}, {"bad-key": True}, {"x": float("nan")}, {"x": "a" * 4097}, {"x": object()}, {"x": [[[[[False]]]]]}])
def test_exact_bounded_kwargs(kwargs):
    module = _load_module()
    with pytest.raises(module.ProfilerError, match="INVALID_CHAT_TEMPLATE_KWARGS"):
        module._template_kwargs(kwargs)


def test_legacy_profile_keeps_identity_shape(tmp_path, monkeypatch):
    module, result, _ = _profile(tmp_path, monkeypatch)
    assert set(result["config_provenance"]) == {"semantic_config_sha256", "component_label_max_chars"}
    config = _config(tmp_path / "snapshot", tmp_path / "data.jsonl", mode="messages")
    assert module.validate_config(config)["input"] == config["input"]


def test_exact_render_rejects_alternate_messages_field(tmp_path):
    module = _load_module()
    config = _config(tmp_path / "snapshot", tmp_path / "rows.jsonl", mode="messages")
    config["input"].update(prompt_render="prompt_completion", messages_field="alternate_messages")
    with pytest.raises(module.ProfilerError, match="INVALID_INPUT_CONFIG"):
        module.validate_config(config)
    config["input"].pop("prompt_render")
    assert module.validate_config(config)["input"]["messages_field"] == "alternate_messages"


def _named_config(tmp_path, inventory_change=None):
    directory = tmp_path / "profiles"
    directory.mkdir()
    versions = {"tokenizers": "0.23.2", "torch": "2.0", "transformers": "5.17.0", "trl": "1.0", "unsloth": "1.0", "unsloth-zoo": "1.0"}
    image = "example/runtime@sha256:" + "b" * 64
    inventory = {"schema_version": "syntunia-python-distribution-inventory/v1", "image": image, "distributions": [{"name": name, "version": version} for name, version in sorted(versions.items())], "runtime": {"python_implementation": "CPython", "python_version": ".".join(map(str, sys.version_info[:3])), "transformers_version": versions["transformers"], "torch_version": versions["torch"], "trl_version": versions["trl"], "unsloth_version": versions["unsloth"], "unsloth_zoo_version": versions["unsloth-zoo"]}}
    if inventory_change:
        inventory_change(inventory)
    raw = json.dumps(inventory).encode()
    (directory / "inventory.json").write_bytes(raw)
    inventory_sha = hashlib.sha256(raw).hexdigest()
    profile = {"schema_version": "syntunia-runtime-profile/v1", "name": "test-runtime", "compatibility": {"models": [{"name": "acme/test-model", "revisions": ["a" * 40]}], "methods": ["sft"]}, "runtime": {"image": image, "inventory": {"path": "inventory.json", "sha256": "sha256:" + inventory_sha}}}
    raw = json.dumps(profile).encode()
    (directory / "test-runtime.yaml").write_bytes(raw)
    config = _config(_snapshot(tmp_path / "snapshot"), _jsonl(tmp_path / "rows.jsonl", [{"secret": "private text"}]))
    config["runtime"] = {"kind": "named_profile", "name": "test-runtime", "profiles_dir": str(directory), "expected_profile_sha256": hashlib.sha256(raw).hexdigest(), "expected_inventory_sha256": inventory_sha}
    return config


def _named_transformers(monkeypatch):
    calls = _install_fake_transformers(monkeypatch)
    sys.modules["transformers"].__version__ = "5.17.0"
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.__version__ = "0.23.2"
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    return calls


def test_named_profile_versions_binding_privacy_and_tamper(tmp_path, monkeypatch):
    module = _load_module()
    config = _named_config(tmp_path)
    calls = _named_transformers(monkeypatch)
    result = module.profile(config)
    assert len(calls) == 1
    assert result["runtime"]["tokenizers_version"] == "0.23.2"
    assert result["runtime"]["admitted_model"] == {"ref": "acme/test-model", "model_revision": "a" * 40, "method": "sft"}
    assert "private text" not in json.dumps(result) and str(tmp_path) not in json.dumps(result)
    module.publish_profile(result, tmp_path / "named")
    module.verify_profile_directory(tmp_path / "named.token-profile", result["profile_semantic_id"])
    for key, value in (("name", "another"), ("profile_sha256", "0" * 64), ("inventory_sha256", "0" * 64), ("admitted_model", {"ref": "acme/other", "model_revision": "a" * 40, "method": "sft"})):
        changed = copy.deepcopy(result)
        changed["runtime"][key] = value
        with pytest.raises(module.ProfilerError, match="PROFILE_ARTIFACT_INVALID"):
            module._validate_profile_shape(changed)
    moved = copy.deepcopy(config)
    moved["runtime"]["profiles_dir"] = "another-private-path"
    assert module._semantic_config(module.validate_config(moved), result["runtime"]) == module._semantic_config(module.validate_config(config), result["runtime"])


@pytest.mark.parametrize("package,version,code", [("transformers", "5.16.0", "TRANSFORMERS_VERSION_MISMATCH"), ("tokenizers", "0.22.2", "TOKENIZERS_VERSION_MISMATCH")])
def test_named_package_drift_fails_before_load(tmp_path, monkeypatch, package, version, code):
    module = _load_module()
    config = _named_config(tmp_path)
    calls = _named_transformers(monkeypatch)
    sys.modules[package].__version__ = version
    with pytest.raises(module.ProfilerError, match=code):
        module.profile(config)
    assert not calls


@pytest.mark.parametrize("mutation", ["profile_digest", "inventory_digest", "model", "revision", "prefixed_digest", "extra_key"])
def test_named_admission_rejects_wrong_binding(tmp_path, mutation):
    module = _load_module()
    config = _named_config(tmp_path)
    if mutation == "profile_digest":
        config["runtime"]["expected_profile_sha256"] = "0" * 64
    elif mutation == "inventory_digest":
        config["runtime"]["expected_inventory_sha256"] = "0" * 64
    elif mutation == "model":
        config["model"]["ref"] = "acme/other"
    elif mutation == "revision":
        config["model"]["model_revision"] = "b" * 40
    elif mutation == "prefixed_digest":
        config["runtime"]["expected_profile_sha256"] = "sha256:" + "0" * 64
    else:
        config["runtime"]["lock_path"] = "private-path"
    with pytest.raises(module.ProfilerError):
        module._load_named_runtime(module.validate_config(config))


@pytest.mark.parametrize("mutation", ["missing_tokenizers", "duplicate_distribution", "wrong_runtime_fact", "wrong_image", "python_version"])
def test_named_inventory_failures(tmp_path, monkeypatch, mutation):
    module = _load_module()
    def change(inventory):
        if mutation == "missing_tokenizers":
            inventory["distributions"] = [row for row in inventory["distributions"] if row["name"] != "tokenizers"]
        elif mutation == "duplicate_distribution":
            inventory["distributions"].append(inventory["distributions"][0])
        elif mutation == "wrong_runtime_fact":
            inventory["runtime"]["transformers_version"] = "0.0"
        elif mutation == "wrong_image":
            inventory["image"] = "other/runtime@sha256:" + "c" * 64
        else:
            inventory["runtime"]["python_version"] = "0.0.0"
    config = _named_config(tmp_path, change)
    calls = _named_transformers(monkeypatch)
    with pytest.raises(module.ProfilerError):
        module.profile(config)
    assert not calls


def test_named_authenticated_inventory_reread_rejects_race(tmp_path, monkeypatch):
    module = _load_module()
    config = _named_config(tmp_path)
    real_read = module._stable_read_file
    def changed_read(path, maximum, code):
        if path.name == "inventory.json":
            return b"{}", None
        return real_read(path, maximum, code)
    monkeypatch.setattr(module, "_stable_read_file", changed_read)
    with pytest.raises(module.ProfilerError, match="RUNTIME_PROFILE_DIGEST_MISMATCH"):
        module._load_named_runtime(config)


def test_checked_in_qwen_example_matches_named_profiler_contract():
    module = _load_module()
    config = module.validate_config(module.load_config(EXAMPLE_CONFIG))

    assert config["model"]["ref"] == "Qwen/Qwen3.5-4B"
    assert config["input"] == {
        "jsonl_path": "/replace/with/private/prepared-dataset.jsonl",
        "mode": "messages",
        "messages_field": "messages",
        "use_chat_template": True,
        "add_generation_prompt": False,
        "prompt_render": "prompt_completion",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    from tuner.runtime_profiles import load_runtime_profile
    actual = load_runtime_profile("qwen35-sft-v1", Path("Trainers/runtime_profiles")).resolve(model=config["model"]["ref"], model_revision=config["model"]["model_revision"], method="sft")
    assert config["runtime"] == {"kind": "named_profile", "name": actual.name, "profiles_dir": "Trainers/runtime_profiles", "expected_profile_sha256": actual.profile_sha256.removeprefix("sha256:"), "expected_inventory_sha256": actual.inventory_sha256.removeprefix("sha256:")}
    assert config["budgets"] == {"max_sequence_tokens": 32768, "completion_reserve_tokens": 0}
    evidence = module._load_named_runtime(config)
    assert evidence["transformers_version"] == "5.17.0"
    assert evidence["tokenizers_version"] == "0.23.2"
    assert evidence["image"] == actual.image


def test_profile_binds_modal_lock_capsule_offline_and_exact_histograms(tmp_path, monkeypatch):
    rows = [{"secret": "x " * count} for count in range(1, 101)]
    module, result, calls = _profile(tmp_path, monkeypatch, rows)
    assert result["runtime"]["runtime_lock_sha256"] == LOCK_SHA
    assert result["runtime"]["transformers_version"] == "4.57.1"
    assert result["runtime"]["tokenizer_stack_claim"].endswith("tokenizers_version_unavailable")
    assert result["total"] == {"n": 100, "min": 1, "p50": 50, "p90": 90, "p95": 95, "p99": 99, "max": 100}
    assert len(result["tokenizer"]["load_closure"]["sha256"]) == 64
    inventory = result["tokenizer"]["load_closure"]["files"]
    assert inventory == sorted(inventory, key=lambda item: item["path"])
    assert {item["path"] for item in inventory} == {"tokenizer.json", "tokenizer_config.json"}
    assert all(set(item) == {"path", "bytes", "sha256"} and len(item["sha256"]) == 64 for item in inventory)
    path, kwargs, environment = calls[0]
    assert kwargs == {"local_files_only": True, "trust_remote_code": False}
    assert environment == {"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    assert ("tokenizer-profile-capsule-" in path) if os.name == "nt" else path.startswith("/proc/self/fd/")
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized
    assert "x x x" not in serialized


def test_semantic_identity_is_path_independent_and_config_validation_is_pure(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    first_data = _jsonl(tmp_path / "first" / "data.jsonl", [{"secret": "same bytes"}])
    second_data = _jsonl(tmp_path / "second" / "data.jsonl", [{"secret": "same bytes"}])
    first = _config(_snapshot(tmp_path / "first" / "snapshot"), first_data)
    second = _config(_snapshot(tmp_path / "second" / "snapshot"), second_data)
    before = copy.deepcopy(first)
    first_result, second_result = module.profile(first), module.profile(second)
    assert first == before
    assert first_result["config_provenance"] == second_result["config_provenance"]
    assert first_result["tokenizer"]["load_closure"] == second_result["tokenizer"]["load_closure"]
    assert first_result["profile_semantic_id"] == second_result["profile_semantic_id"]
    assert first_result["operational_provenance"] != second_result["operational_provenance"]


@pytest.mark.parametrize("unsafe", ["C:\\model", "/tmp/model", "../model", "https://host/model", "org/model/extra", "org\\model", "org/.."])
def test_safe_public_model_ref_and_legacy_config_are_rejected(tmp_path, unsafe):
    module = _load_module()
    config = _config(tmp_path / "snapshot", tmp_path / "data")
    config["model"]["ref"] = unsafe
    with pytest.raises(module.ProfilerError, match="INVALID_MODEL_REF"):
        module.validate_config(config)
    legacy = _config(tmp_path / "snapshot", tmp_path / "data")
    legacy["model"]["name"] = legacy["model"].pop("ref")
    with pytest.raises(module.ProfilerError, match="INVALID_MODEL_CONFIG"):
        module.validate_config(legacy)
    legacy_runtime = _config(tmp_path / "snapshot", tmp_path / "data")
    legacy_runtime["runtime"] = {"transformers_version": "5.5.0"}
    with pytest.raises(module.ProfilerError, match="INVALID_RUNTIME_CONFIG"):
        module.validate_config(legacy_runtime)


def test_runtime_lock_digest_schema_and_installed_version_fail_closed(tmp_path, monkeypatch):
    module = _load_module()
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    snapshot = _snapshot(tmp_path / "snapshot")
    config = _config(snapshot, dataset)
    config["runtime"]["expected_lock_sha256"] = "0" * 64
    with pytest.raises(module.ProfilerError, match="RUNTIME_LOCK_DIGEST_MISMATCH"):
        module.profile(config)

    invalid_lock = tmp_path / "lock.json"
    invalid_lock.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    config = _config(snapshot, dataset)
    config["runtime"].update(lock_path=str(invalid_lock), expected_lock_sha256=hashlib.sha256(invalid_lock.read_bytes()).hexdigest())
    with pytest.raises(module.ProfilerError, match="RUNTIME_LOCK_SCHEMA_MISMATCH"):
        module.profile(config)

    fake = types.ModuleType("transformers")
    fake.__version__ = "5.5.0"
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: pytest.fail("load must not run"))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(module.ProfilerError, match="TRANSFORMERS_VERSION_MISMATCH"):
        module.profile(_config(snapshot, dataset))


def test_snapshot_contract_rejects_extra_directory_special_and_case_collision(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])

    extra = _snapshot(tmp_path / "extra")
    (extra / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(module.ProfilerError, match="SNAPSHOT_INVENTORY_INVALID"):
        module.profile(_config(extra, dataset))

    directory = _snapshot(tmp_path / "directory")
    (directory / "vocab.txt").mkdir()
    with pytest.raises(module.ProfilerError, match="SNAPSHOT_MEMBER_UNSTABLE"):
        module.profile(_config(directory, dataset))

    collision = _snapshot(tmp_path / "collision")
    try:
        (collision / "TOKENIZER.JSON").write_text("{}", encoding="utf-8")
    except OSError:
        pytest.skip("case-insensitive filesystem cannot construct collision")
    if len(list(collision.iterdir())) == 3:
        with pytest.raises(module.ProfilerError, match="SNAPSHOT_CASE_COLLISION"):
            module.profile(_config(collision, dataset))


def test_snapshot_and_input_links_are_rejected_when_supported(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    real_snapshot = _snapshot(tmp_path / "real-snapshot")
    snapshot_link = tmp_path / "snapshot-link"
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    try:
        snapshot_link.symlink_to(real_snapshot, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(module.ProfilerError, match="SNAPSHOT_INVALID"):
        module.profile(_config(snapshot_link, dataset))
    input_link = tmp_path / "input-link.jsonl"
    input_link.symlink_to(dataset)
    with pytest.raises(module.ProfilerError, match="INPUT_READ_FAILED"):
        module.profile(_config(real_snapshot, input_link))


def test_capsule_mutation_is_detected_after_tokenizer_load(tmp_path, monkeypatch):
    module = _load_module()
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    blocked = []

    def mutate(capsule):
        try:
            (capsule / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")
        except OSError:
            blocked.append(True)

    _install_fake_transformers(monkeypatch, mutate)
    try:
        module.profile(_config(_snapshot(tmp_path / "snapshot"), dataset))
    except module.ProfilerError as error:
        assert error.code == "CAPSULE_MUTATED" and not blocked
    else:
        assert blocked


def test_capsule_swap_and_restore_cannot_change_loaded_bytes(tmp_path, monkeypatch):
    module = _load_module()
    snapshot = _snapshot(tmp_path / "snapshot")
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    original_digest = hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    observed = {}

    def swap_restore(loader_path):
        if os.name == "nt":
            physical = loader_path
        else:
            physical = Path(os.readlink(loader_path))
        backup = physical.with_name(physical.name + "-held")
        try:
            physical.rename(backup)
        except OSError:
            observed["swap_blocked"] = True
            observed["loaded_sha256"] = hashlib.sha256((loader_path / "tokenizer.json").read_bytes()).hexdigest()
            return
        physical.mkdir()
        (physical / "tokenizer.json").write_text('{"attacker":true}', encoding="utf-8")
        (physical / "tokenizer_config.json").write_text('{"attacker":true}', encoding="utf-8")
        try:
            observed["loaded_sha256"] = hashlib.sha256((loader_path / "tokenizer.json").read_bytes()).hexdigest()
        finally:
            for member in physical.iterdir():
                member.unlink()
            physical.rmdir()
            backup.rename(physical)

    _install_fake_transformers(monkeypatch, swap_restore)
    if os.name == "nt":
        result = module.profile(_config(snapshot, dataset))
        inventory_digest = next(item["sha256"] for item in result["tokenizer"]["load_closure"]["files"] if item["path"] == "tokenizer.json")
        assert observed["loaded_sha256"] == original_digest == inventory_digest
    else:
        with pytest.raises(module.ProfilerError, match="CAPSULE_MUTATED"):
            module.profile(_config(snapshot, dataset))
        assert observed["loaded_sha256"] == original_digest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc retained-root contract only")
def test_linux_capsule_member_swap_and_restore_fails_closed(tmp_path, monkeypatch):
    module = _load_module()
    snapshot = _snapshot(tmp_path / "snapshot")
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    observed = {}

    def swap_member_restore(loader_path):
        physical = Path(os.readlink(loader_path))
        member = physical / "tokenizer.json"
        backup = physical / "tokenizer.json.held"
        os.chmod(physical, 0o700)
        try:
            member.rename(backup)
            member.write_text('{"attacker":true}', encoding="utf-8")
            try:
                observed["loaded_sha256"] = hashlib.sha256((loader_path / "tokenizer.json").read_bytes()).hexdigest()
            finally:
                member.unlink()
                backup.rename(member)
        finally:
            os.chmod(physical, 0o500)

    _install_fake_transformers(monkeypatch, swap_member_restore)
    with pytest.raises(module.ProfilerError, match="CAPSULE_MUTATED"):
        module.profile(_config(snapshot, dataset))
    assert observed["loaded_sha256"] != hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()


def test_source_member_content_swap_and_restore_fails_or_is_blocked(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    snapshot = _snapshot(tmp_path / "snapshot")
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    original_read = module._stable_read_member
    attempted = blocked = False

    def mutate_after_first_read(guard, name, max_bytes, code):
        nonlocal attempted, blocked
        value = original_read(guard, name, max_bytes, code)
        if guard.path == snapshot and not attempted:
            attempted = True
            member = snapshot / name
            original = member.read_bytes()
            try:
                member.write_bytes(b"attacker-content")
                member.write_bytes(original)
            except OSError:
                blocked = True
        return value

    monkeypatch.setattr(module, "_stable_read_member", mutate_after_first_read)
    try:
        module.profile(_config(snapshot, dataset))
    except module.ProfilerError as error:
        assert error.code == "SNAPSHOT_MEMBER_UNSTABLE" and not blocked
    else:
        assert blocked


def test_message_roles_template_digest_and_zero_buckets(tmp_path, monkeypatch):
    rows = [{"messages": [{"role": "system", "content": "one"}, {"role": "user", "content": "two three"}]}]
    module, result, _ = _profile(tmp_path, monkeypatch, rows, mode="messages")
    assert list(result["components"]) == ["assistant", "developer", "system", "tool", "user"]
    assert result["components"]["assistant"] == {"n": 1, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    assert result["tokenizer"]["chat_template_sha256"] == hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest()
    unknown = _config(tmp_path / "snapshot", tmp_path / "data.jsonl", mode="messages")
    _jsonl(Path(unknown["input"]["jsonl_path"]), [{"messages": [{"role": "private-name", "content": "secret"}]}])
    with pytest.raises(module.ProfilerError, match="INVALID_MESSAGES_ROW"):
        module.profile(unknown)


@pytest.mark.parametrize(
    ("limit", "value", "code"),
    [
        ("max_input_bytes", 5, "INPUT_READ_FAILED"),
        ("max_rows", 1, "INPUT_LIMIT_EXCEEDED"),
        ("max_line_bytes", 10, "INPUT_LIMIT_EXCEEDED"),
        ("max_record_bytes", 10, "INPUT_LIMIT_EXCEEDED"),
        ("max_tokens_per_record", 1, "TOKEN_LIMIT_EXCEEDED"),
    ],
)
def test_streaming_input_bounds(tmp_path, monkeypatch, limit, value, code):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "one two three"}, {"secret": "four five"}])
    config = _config(_snapshot(tmp_path / "snapshot"), dataset)
    config["limits"][limit] = value
    if limit == "max_input_bytes":
        config["limits"]["max_line_bytes"] = min(config["limits"]["max_line_bytes"], value)
        config["limits"]["max_record_bytes"] = min(config["limits"]["max_record_bytes"], value)
    if limit == "max_line_bytes":
        config["limits"]["max_record_bytes"] = value
    with pytest.raises(module.ProfilerError, match=code):
        module.profile(config)


def test_streaming_path_never_uses_path_read_bytes(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "stream me"}])
    snapshot = _snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("Path.read_bytes is forbidden"))
    result = module.profile(_config(snapshot, dataset))
    assert result["input_provenance"]["row_count"] == 1


def test_input_digest_commits_blank_lines_and_snapshot_root_replacement_fails(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    dataset = tmp_path / "data.jsonl"
    raw = b'\n{"secret":"one"}\n\n'
    dataset.write_bytes(raw)
    snapshot = _snapshot(tmp_path / "snapshot")
    result = module.profile(_config(snapshot, dataset))
    assert result["input_provenance"] == {"jsonl_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "row_count": 1}

    original_read = module._stable_read_member
    replaced = False
    blocked = False

    def replace_root_after_first_member(guard, name, max_bytes, code):
        nonlocal replaced, blocked
        value = original_read(guard, name, max_bytes, code)
        if guard.path == snapshot and not replaced:
            replaced = True
            old = tmp_path / "old-snapshot"
            try:
                snapshot.rename(old)
            except OSError:
                blocked = True
            else:
                _snapshot(snapshot)
        return value

    monkeypatch.setattr(module, "_stable_read_member", replace_root_after_first_member)
    try:
        module.profile(_config(snapshot, dataset))
    except module.ProfilerError as error:
        assert error.code == "SNAPSHOT_UNSTABLE" and not blocked
    else:
        assert blocked


def test_all_declared_limits_and_component_bounds_are_enforced(tmp_path):
    module = _load_module()
    base = _config(tmp_path / "snapshot", tmp_path / "data")
    base["limits"]["max_snapshot_files"] = 1
    normalized = module.validate_config(base)
    assert normalized["limits"]["max_snapshot_files"] == 1
    for key in module.HARD_LIMITS:
        invalid = _config(tmp_path / "snapshot", tmp_path / "data")
        invalid["limits"][key] = True
        with pytest.raises(module.ProfilerError, match="INVALID_LIMIT_CONFIG"):
            module.validate_config(invalid)
    components = _config(tmp_path / "snapshot", tmp_path / "data", mode="components")
    components["limits"]["max_components"] = 1
    with pytest.raises(module.ProfilerError, match="INVALID_COMPONENT_CONFIG"):
        module.validate_config(components)


@pytest.mark.parametrize("label", ["../secret", "private label", "http:path", "slash/name", "back\\name", "_hidden", "name-dot"])
def test_component_labels_are_safe_public_identifiers(tmp_path, label):
    module = _load_module()
    config = _config(tmp_path / "snapshot", tmp_path / "data", mode="components")
    config["input"]["components"] = {label: "field"}
    with pytest.raises(module.ProfilerError, match="INVALID_COMPONENT_CONFIG"):
        module.validate_config(config)
    valid = _config(tmp_path / "snapshot", tmp_path / "data", mode="components")
    valid["input"]["components"] = {"prompt_1": "field"}
    assert module.validate_config(valid)["input"]["components"] == {"prompt_1": "field"}


@pytest.mark.parametrize("label", ["../secret", "private label", "name-dot", "A" * 31])
def test_profile_artifact_verification_rejects_unsafe_component_labels(tmp_path, monkeypatch, label):
    module, result, _ = _profile(tmp_path, monkeypatch, [{"instruction": "one", "answer": "two"}], mode="components")
    distribution = next(iter(result["components"].values()))
    result["components"] = {label: distribution}
    with pytest.raises(module.ProfilerError, match="PROFILE_ARTIFACT_INVALID"):
        module._validate_profile_shape(result)


@pytest.mark.parametrize(
    "duplicate",
    [
        "model:\n  ref: acme/test\n  ref: acme/other\n",
        "runtime:\n  provider: modal\n  provider: other\n",
        "input:\n  mode: text\n  mode: messages\n",
        "budgets:\n  max_sequence_tokens: 8\n  max_sequence_tokens: 9\n",
        "limits:\n  max_rows: 1\n  max_rows: 2\n",
    ],
)
def test_yaml_duplicate_security_keys_are_rejected(tmp_path, duplicate):
    module = _load_module()
    config = tmp_path / "duplicate.yaml"
    config.write_text(duplicate, encoding="utf-8")
    with pytest.raises(module.ProfilerError, match="CONFIG_READ_FAILED"):
        module.load_config(config)
    components = _config(tmp_path / "snapshot", tmp_path / "data", mode="components")
    components["limits"]["max_component_label_chars"] = 3
    with pytest.raises(module.ProfilerError, match="INVALID_COMPONENT_CONFIG"):
        module.validate_config(components)


def test_publication_verify_reconcile_collision_and_tamper(tmp_path, monkeypatch):
    module, result, _ = _profile(tmp_path, monkeypatch)
    published = module.publish_profile(result, tmp_path / "run.alpha")
    profile_dir = tmp_path / "run.alpha.token-profile"
    assert profile_dir.is_dir()
    assert published["state"] == ("VERIFIED_DURABILITY_UNPROVEN" if os.name == "nt" else "PUBLISHED_DURABLE")
    verified = module.verify_profile_directory(profile_dir, result["profile_semantic_id"])
    assert verified["profile_id"] == result["profile_semantic_id"]
    reconciled = module.reconcile_profile(profile_dir, result["profile_semantic_id"])
    assert reconciled["profile_id"] == result["profile_semantic_id"]
    with pytest.raises(module.ProfilerError, match="PROFILE_ID_MISMATCH"):
        module.verify_profile_directory(profile_dir, "0" * 64)
    original = {path.name: path.read_bytes() for path in profile_dir.iterdir()}
    with pytest.raises(module.ProfilerError, match="OUTPUT_COLLISION"):
        module.publish_profile(result, tmp_path / "run.alpha")
    assert {path.name: path.read_bytes() for path in profile_dir.iterdir()} == original
    (profile_dir / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(module.ProfilerError, match="PROFILE_ARTIFACT_INVALID"):
        module.verify_profile_directory(profile_dir, result["profile_semantic_id"])


def test_prepublication_failure_cleans_stage_and_postcommit_failures_are_typed(tmp_path, monkeypatch):
    module, result, _ = _profile(tmp_path, monkeypatch)
    real_rename = module._rename_directory_no_replace
    monkeypatch.setattr(module, "_rename_directory_no_replace", lambda *_: (_ for _ in ()).throw(OSError("secret path")))
    with pytest.raises(module.ProfilerError, match="OUTPUT_WRITE_FAILED"):
        module.publish_profile(result, tmp_path / "before")
    assert not (tmp_path / "before.token-profile").exists()
    assert not list(tmp_path.glob(".before.token-profile.stage-*"))
    monkeypatch.setattr(module, "_rename_directory_no_replace", real_rename)

    monkeypatch.setattr(module, "_parent_durability_supported", lambda: True)
    calls = []

    def fsync_parent(path):
        calls.append(path)
        if path == tmp_path and calls.count(tmp_path) == 1:
            raise OSError("private parent")

    monkeypatch.setattr(module, "_fsync_directory", fsync_parent)
    with pytest.raises(module.PublicationUncertain) as parent_uncertain:
        module.publish_profile(result, tmp_path / "parent")
    assert parent_uncertain.value.phase == "parent_durability"
    assert (tmp_path / "parent.token-profile").is_dir()

    module, result, _ = _profile(tmp_path / "second", monkeypatch)
    real_verify = module.verify_profile_directory
    verify_calls = []

    def fail_final(path, expected=None):
        verify_calls.append(path)
        if len(verify_calls) == 2:
            raise module.ProfilerError("PROFILE_ARTIFACT_INVALID")
        return real_verify(path, expected)

    monkeypatch.setattr(module, "verify_profile_directory", fail_final)
    with pytest.raises(module.PublicationUncertain) as final_uncertain:
        module.publish_profile(result, tmp_path / "final")
    assert final_uncertain.value.phase == "final_verification"
    assert (tmp_path / "final.token-profile").is_dir()


def test_reconcile_retries_parent_durability_before_verification(tmp_path, monkeypatch):
    module, result, _ = _profile(tmp_path, monkeypatch)
    profile_dir = tmp_path / "ordered.token-profile"
    module.publish_profile(result, tmp_path / "ordered")
    order = []
    real_verify = module.verify_profile_directory
    monkeypatch.setattr(module, "_parent_durability_supported", lambda: True)
    monkeypatch.setattr(module, "_fsync_directory", lambda path: order.append("durability"))

    def verify(path, expected=None):
        order.append("verify")
        return real_verify(path, expected)

    monkeypatch.setattr(module, "verify_profile_directory", verify)
    first = module.reconcile_profile(profile_dir, result["profile_semantic_id"])
    second = module.reconcile_profile(profile_dir, result["profile_semantic_id"])
    assert first == second
    assert order == ["durability", "verify", "durability", "verify"]


def test_cli_create_verify_and_errors_are_single_sanitized_json(tmp_path, monkeypatch, capsys):
    module, result, _ = _profile(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "load_config", lambda _: {})
    monkeypatch.setattr(module, "profile", lambda _: result)
    code = module.main(["create", "--config", "private-secret.yaml", "--output-prefix", str(tmp_path / "cli")])
    captured = capsys.readouterr()
    assert captured.err == ""
    response = json.loads(captured.out)
    assert code == (3 if os.name == "nt" else 0)
    assert response["profile_id"] == result["profile_semantic_id"]
    assert str(tmp_path) not in captured.out and "private-secret" not in captured.out

    code = module.main(["verify", "--profile-dir", str(tmp_path / "cli.token-profile"), "--expected-profile-id", result["profile_semantic_id"]])
    response = json.loads(capsys.readouterr().out)
    assert code == 0 and response["status"] == "verified"

    monkeypatch.setattr(module, "publish_profile", lambda *_: (_ for _ in ()).throw(RuntimeError("secret corpus and path")))
    assert module.main(["create", "--config", "secret", "--output-prefix", "secret-output"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"status": "error", "error": {"code": "PROFILE_FAILED"}}
    assert "secret corpus" not in captured.out


def test_boolean_budget_unknown_keys_duplicate_json_and_empty_input_fail_closed(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_transformers(monkeypatch)
    dataset = _jsonl(tmp_path / "data.jsonl", [{"secret": "text"}])
    snapshot = _snapshot(tmp_path / "snapshot")
    invalid = _config(snapshot, dataset)
    invalid["budgets"]["max_sequence_tokens"] = True
    with pytest.raises(module.ProfilerError, match="INVALID_BUDGET_CONFIG"):
        module.profile(invalid)
    unknown = _config(snapshot, dataset)
    unknown["surprise"] = True
    with pytest.raises(module.ProfilerError, match="INVALID_CONFIG"):
        module.validate_config(unknown)
    dataset.write_text('{"secret":"one","secret":"two"}\n', encoding="utf-8")
    with pytest.raises(module.ProfilerError, match="INVALID_JSONL_ROW"):
        module.profile(_config(snapshot, dataset))
    dataset.write_text("\n", encoding="utf-8")
    with pytest.raises(module.ProfilerError, match="EMPTY_INPUT"):
        module.profile(_config(snapshot, dataset))
