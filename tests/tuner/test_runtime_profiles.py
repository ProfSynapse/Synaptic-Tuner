from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import tuner.runtime_profiles as runtime_profiles
from tuner.runtime_profiles import RuntimeProfileError, load_runtime_profile


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "Trainers" / "runtime_profiles"


def test_qwen35_sft_v1_binds_captured_complete_inventory() -> None:
    profile = load_runtime_profile("qwen35-sft-v1", PROFILES)

    assert profile.image == (
        "unsloth/unsloth@sha256:"
        "1644d635bc7c5b57ed64cabbab1ae00647dfb583c2135fdfb6a461aeeba52739"
    )
    assert profile.model_revisions == {
        "Qwen/Qwen3.5-4B": ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",)
    }
    assert profile.methods == ("sft",)
    assert profile.packaged_build_profile == "qwen35_4b_packaged_sft_3360351c"
    assert profile.distribution_count == 327
    assert profile.runtime_facts == {
        "architecture": "x86_64",
        "cuda_build": "12.8",
        "libc": "glibc 2.39",
        "os": "Linux",
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "torch_version": "2.11.0+cu128",
        "transformers_version": "5.17.0",
        "trl_version": "0.24.0",
        "unsloth_version": "2026.9.7",
        "unsloth_zoo_version": "2026.9.6",
    }


def test_runtime_profile_rejects_unlisted_model_and_method() -> None:
    profile = load_runtime_profile("qwen35-sft-v1", PROFILES)

    with pytest.raises(RuntimeProfileError, match="does not support model"):
        profile.resolve(
            model="Qwen/Qwen3.5-2B", model_revision="revision", method="sft"
        )
    with pytest.raises(RuntimeProfileError, match="does not support model revision"):
        profile.resolve(
            model="Qwen/Qwen3.5-4B", model_revision="", method="sft"
        )
    with pytest.raises(RuntimeProfileError, match="does not support model revision"):
        profile.resolve(
            model="Qwen/Qwen3.5-4B", model_revision="arbitrary", method="sft"
        )
    with pytest.raises(RuntimeProfileError, match="does not support method"):
        profile.resolve(
            model="Qwen/Qwen3.5-4B",
            model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            method="kto",
        )


def test_runtime_profile_rejects_inventory_digest_mismatch(tmp_path: Path) -> None:
    source_profile = yaml.safe_load(
        (PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8")
    )
    source_inventory = json.loads(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_text(encoding="utf-8")
    )
    source_inventory["runtime"]["python_version"] = "0.0.0"
    inventory_bytes = (
        json.dumps(source_inventory, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (tmp_path / "qwen35-sft-v1.inventory.json").write_bytes(inventory_bytes)
    # Deliberately retain the checked-in digest while mutating inventory bytes.
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(source_profile, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(RuntimeProfileError, match="inventory digest mismatch"):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


def test_runtime_profile_accepts_exact_recomputed_inventory_digest(
    tmp_path: Path,
) -> None:
    source_profile = yaml.safe_load(
        (PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8")
    )
    inventory_bytes = (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    (tmp_path / "qwen35-sft-v1.inventory.json").write_bytes(inventory_bytes)
    source_profile["runtime"]["inventory"]["sha256"] = (
        "sha256:" + hashlib.sha256(inventory_bytes).hexdigest()
    )
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(source_profile, sort_keys=False), encoding="utf-8"
    )

    assert load_runtime_profile("qwen35-sft-v1", tmp_path).distribution_count == 327


def test_runtime_profile_schema_v1_rejects_unknown_methods(tmp_path: Path) -> None:
    profile = yaml.safe_load(
        (PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8")
    )
    profile["compatibility"]["methods"] = ["kto"]
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "qwen35-sft-v1.inventory.json").write_bytes(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    )

    with pytest.raises(RuntimeProfileError, match="unsupported methods: kto"):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


@pytest.mark.parametrize("name", ["../other", "other/profile", "C:\\other", ".hidden", "name.yaml"])
def test_runtime_profile_rejects_unsafe_build_profile_name(tmp_path: Path, name: str) -> None:
    profile = yaml.safe_load((PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8"))
    profile["runtime"]["packaged_build_profile"] = name
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(RuntimeProfileError, match="build profile name is invalid"):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


def test_runtime_profile_without_modal_build_remains_valid_but_cannot_plan_modal(
    tmp_path: Path,
) -> None:
    profile = yaml.safe_load((PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8"))
    del profile["runtime"]["packaged_build_profile"]
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "qwen35-sft-v1.inventory.json").write_bytes(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    )
    admitted = load_runtime_profile("qwen35-sft-v1", tmp_path)
    with pytest.raises(RuntimeProfileError, match="no packaged Modal build profile"):
        admitted.modal_build_profile_path(tmp_path.parent / "image_profiles")


def _write_mutated_inventory(tmp_path: Path, mutate) -> None:
    profile = yaml.safe_load(
        (PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_text(encoding="utf-8")
    )
    mutate(inventory)
    inventory_bytes = (
        json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (tmp_path / "qwen35-sft-v1.inventory.json").write_bytes(inventory_bytes)
    profile["runtime"]["inventory"]["sha256"] = (
        "sha256:" + hashlib.sha256(inventory_bytes).hexdigest()
    )
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )


def test_runtime_profile_rejects_normalized_distribution_collision(
    tmp_path: Path,
) -> None:
    def mutate(inventory):
        inventory["distributions"].append(
            {"name": "unsloth-zoo", "version": "2026.9.6"}
        )
        inventory["distributions"].sort(
            key=lambda item: (
                runtime_profiles.canonicalize_name(item["name"]),
                item["version"],
            )
        )

    _write_mutated_inventory(tmp_path, mutate)
    with pytest.raises(RuntimeProfileError, match="colliding normalized"):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


def test_runtime_profile_rejects_runtime_fact_distribution_mismatch(
    tmp_path: Path,
) -> None:
    _write_mutated_inventory(
        tmp_path,
        lambda inventory: inventory["runtime"].__setitem__(
            "transformers_version", "0.0.0"
        ),
    )
    with pytest.raises(RuntimeProfileError, match="transformers_version"):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


def test_stable_file_read_rejects_replacement_between_stat_and_open(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "profile.yaml"
    target.write_bytes(b"original")
    real_open = runtime_profiles.os.open
    replaced = False

    def replace_then_open(path, flags):
        nonlocal replaced
        if not replaced:
            replaced = True
            target.write_bytes(b"replacement-content")
        return real_open(path, flags)

    monkeypatch.setattr(runtime_profiles.os, "open", replace_then_open)
    with pytest.raises(RuntimeProfileError, match="identity changed before open"):
        runtime_profiles._read_stable_regular_file(
            target, maximum=1024, label="Runtime profile"
        )


def _profile_with_methods(tmp_path: Path, methods: list[object]) -> None:
    profile = yaml.safe_load((PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8"))
    profile["compatibility"]["methods"] = methods
    (tmp_path / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "qwen35-sft-v1.inventory.json").write_bytes(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    )


@pytest.mark.parametrize("methods", [["grpo"], ["sft", "grpo"]])
def test_runtime_profile_accepts_declared_grpo_method(tmp_path: Path, methods) -> None:
    _profile_with_methods(tmp_path, methods)
    profile = load_runtime_profile("qwen35-sft-v1", tmp_path)
    assert profile.methods == tuple(methods)
    assert profile.resolve(
        model="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        method="grpo",
    ) is profile


def test_grpo_only_profile_does_not_admit_sft(tmp_path: Path) -> None:
    _profile_with_methods(tmp_path, ["grpo"])
    profile = load_runtime_profile("qwen35-sft-v1", tmp_path)
    with pytest.raises(RuntimeProfileError, match="does not support method"):
        profile.resolve(
            model="Qwen/Qwen3.5-4B",
            model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            method="sft",
        )


@pytest.mark.parametrize("methods", [
    ["sft", "kto"], ["GRPO"], ["sft", "sft"], [], ["grpo", 1],
])
def test_runtime_profile_rejects_unknown_duplicate_or_malformed_methods(
    tmp_path: Path, methods,
) -> None:
    _profile_with_methods(tmp_path, methods)
    with pytest.raises(RuntimeProfileError):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


GRPO_PROFILES = {"qwen35-env-grpo-v1"}


def test_checked_in_profiles_declare_one_method_each() -> None:
    for path in sorted(PROFILES.glob("*.yaml")):
        profile = load_runtime_profile(path.stem, PROFILES)
        assert profile.methods == (("grpo",) if path.stem in GRPO_PROFILES else ("sft",))


def _inventory_versions(path: Path) -> dict[str, str]:
    from packaging.utils import canonicalize_name

    return {
        canonicalize_name(item["name"]): item["version"]
        for item in json.loads(path.read_bytes())["distributions"]
    }


def test_env_grpo_profile_binds_base_and_captured_stack_inventory() -> None:
    from tuner.cloud.derived_training_image import load_profile

    grpo = load_runtime_profile("qwen35-env-grpo-v1", PROFILES)
    sft = load_runtime_profile("qwen35-sft-v1", PROFILES)
    assert grpo.image == sft.image
    assert grpo.model_revisions == sft.model_revisions
    assert grpo.packaged_build_profile == "qwen35_4b_packaged_env_grpo"
    assert grpo.runtime_facts == {
        **sft.runtime_facts,
        "trl_version": "1.13.0",
        "unsloth_version": "2026.10.2",
        "unsloth_zoo_version": "2026.10.2",
    }
    # Same distribution set as the base; only the bootstrap-replaced pins move.
    grpo_versions = _inventory_versions(grpo.inventory_path)
    sft_versions = _inventory_versions(sft.inventory_path)
    assert set(grpo_versions) == set(sft_versions)
    changed = {name for name in sft_versions if grpo_versions[name] != sft_versions[name]}
    assert changed == {"datasets", "trl", "unsloth", "unsloth-zoo"}
    assert {name: grpo_versions[name] for name in changed} == {
        "datasets": "4.8.5", "trl": "1.13.0", "unsloth": "2026.10.2", "unsloth-zoo": "2026.10.2",
    }

    build = load_profile(grpo.modal_build_profile_path(ROOT / "Trainers" / "image_profiles"))
    assert build.base_image.removeprefix("docker.io/") == grpo.image
    assert build.packages == ()
    pins = {item["distribution"]: item["version"] for item in build.packaged_runtime["bootstrap"]}
    # Every bootstrap wheel that replaces a base distribution is what the
    # inventory records; the rest are additive packaged-runtime closure.
    assert {name: pins[name] for name in pins if name in sft_versions} == {
        name: grpo_versions[name] for name in changed
    }
    capabilities = build.packaged_runtime["capabilities"]
    assert capabilities["compatibility"] == {
        "methods": ["grpo"],
        "models": [{"ref": "Qwen/Qwen3.5-4B",
                    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}],
        "dataset_formats": ["syntunia-env-rollout-row/v1"],
    }
    assert capabilities["contracts"] == {
        "workload_schema": "synaptic-packaged-env-grpo-workload/v1",
        "prepared_input_schema": "synaptic-prepared-training-input/v1",
        "artifact_contract_schema": "synaptic-env-grpo-artifacts/v1",
    }


def test_env_grpo_profile_rejects_sft_resolution() -> None:
    profile = load_runtime_profile("qwen35-env-grpo-v1", PROFILES)
    profile.resolve(
        model="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        method="grpo",
    )
    with pytest.raises(RuntimeProfileError, match="does not support method 'sft'"):
        profile.resolve(
            model="Qwen/Qwen3.5-4B",
            model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            method="sft",
        )
