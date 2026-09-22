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


def test_runtime_profile_schema_v1_rejects_non_sft_methods(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeProfileError, match="supports only method sft"):
        load_runtime_profile("qwen35-sft-v1", tmp_path)


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
