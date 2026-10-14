"""Real captured configuration/preparation; CPU-only synthetic Git build intent."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest
import yaml
import tuner.runtime_profiles as profile_module
from tuner.runtime_profiles import load_runtime_profile
from tuner.cloud.derived_training_image import load_profile
from tuner.training.modal_recipe import load_modal_grpo_recipe, plan_modal_recipe
from tuner.training.grpo_compilation import admit_grpo_snapshot, compile_grpo_input
from tuner.training.grpo_preparation import (
    GRPONormalizerConfigV1,
    GRPOTrainingInputNormalizerV1,
    GRPOPreparedInputFormatVerifierV1,
)
from tuner.training.input_preparation import (
    TrainingInputPreparationServiceV1,
    PosixPrivatePreparedRootAuthorityV1,
)
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1,
    TrainingPreparationConfigV1,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(profile_module.__file__).resolve().parent.parent
NAME = "smollm2-135m-grpo-candidate-v1"
BUILD_NAME = "smollm2_135m_packaged_grpo_candidate_v1"
REVISION = "a91318be21aeaf0879874faa161dcb40c68847e9"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
INVENTORY_SHA = "46f0feddd1ba6d5e155dfb227d92679f9afd65634b98d80260cf0517b4b102cd"
ANCHOR_SHA = "30d5f1b1b92f6826c2b624ca88daa32d0e8437595adfd93f689ae813d7fad863"


def build_path():
    # Explicit consumer-owned input in installed tests; no production fallback.
    return Path(
        os.environ.get(
            "SYNAPTIC_TEST_GRPO_BUILD_PROFILE",
            str(ROOT / "Trainers/image_profiles" / BUILD_NAME / "profile.yaml"),
        )
    )


def example_path():
    return ROOT / "examples/grpo_captured_candidate"


def prepared_project(tmp_path):
    project = tmp_path / "project"
    source = project / "private-grpo-source"
    output = project / ".tracking/datasets"
    for directory in (project, project / ".tracking", source, output):
        directory.mkdir(mode=0o700)
    shutil.copyfile(example_path() / "rows.jsonl", source / "rows.jsonl")
    (source / "rows.jsonl").chmod(0o600)
    raw_recipe = (example_path() / "recipe.yaml").read_bytes()
    (project / "recipe.yaml").write_bytes(raw_recipe)
    declaration = TrainingInputV1.from_dict(
        yaml.safe_load(raw_recipe)["training_input"]
    )
    config = GRPONormalizerConfigV1("rows.jsonl", "prompt")
    verifier = GRPOPreparedInputFormatVerifierV1()
    service = TrainingInputPreparationServiceV1(
        prepared_root=output,
        normalizers={
            config.normalizer_ref: GRPOTrainingInputNormalizerV1(
                allowed_source_roots=(source,)
            )
        },
        format_verifiers={verifier.format: verifier},
        root_authority=PosixPrivatePreparedRootAuthorityV1(),
    )
    result = service.prepare(
        LocalTrainingInputPathV1(source),
        TrainingPreparationConfigV1(
            "candidate-request",
            "candidate-project",
            declaration.canonical_json(),
            config,
        ),
    )
    return project, source, result, declaration


def test_actual_full_inventory_and_exact_candidate_pins():
    profile = load_runtime_profile(NAME, ASSETS / "Trainers/runtime_profiles")
    assert profile.distribution_count == 396
    assert profile.methods == ("grpo",)
    assert (
        profile.resolve(model=MODEL, model_revision=REVISION, method="grpo") == profile
    )
    assert profile.inventory_sha256 == "sha256:" + INVENTORY_SHA
    raw = profile.inventory_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == INVENTORY_SHA
    inventory = json.loads(raw)
    assert (
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        == raw
    )
    distributions = {row["name"]: row["version"] for row in inventory["distributions"]}
    assert len(distributions) == len(inventory["distributions"]) == 396
    assert distributions["torch"] == "2.9.0+cu128"
    assert distributions["trl"] == "0.24.0"
    assert distributions["unsloth"] == distributions["unsloth-zoo"] == "2026.1.2"
    assert distributions["transformers"] == "4.57.1"
    assert "modal" not in distributions and "synaptic-tuner" not in distributions
    assert profile.runtime_facts["python_version"] == "3.11.14"
    assert profile.runtime_facts["cuda_build"] == "12.8"
    with pytest.raises(ValueError):
        profile.resolve(model=MODEL, model_revision=REVISION, method="sft")
    with pytest.raises(ValueError):
        profile.resolve(model=MODEL, model_revision="b" * 40, method="grpo")


def test_source_owned_real_wheel_anchor_and_grpo_contracts():
    build = load_profile(build_path())
    assert build.packages == ()
    packaged = build.packaged_runtime
    assert packaged["wheel"]["sha256"] == ANCHOR_SHA
    assert packaged["python"] == {
        "implementation": "cpython",
        "version": "3.11.14",
        "executable": "/opt/conda/bin/python3",
        "executable_digest": "06950bbe281eade6c8c0a9c6c70e6a9a338e6fb63b641dc3a71b1854e74fc859",
        "purelib": "/opt/conda/lib/python3.11/site-packages",
        "platlib": "/opt/conda/lib/python3.11/site-packages",
    }
    assert packaged["capabilities"]["compatibility"] == {
        "methods": ["grpo"],
        "models": [{"ref": MODEL, "revision": REVISION}],
        "dataset_formats": ["synaptic-grpo-prompts/v1"],
    }
    assert (
        packaged["capabilities"]["contracts"]["artifact_contract_schema"]
        == "synaptic-grpo-artifacts/v1"
    )
    assert len(packaged["bootstrap"]) == 9
    assert (
        next(row for row in packaged["bootstrap"] if row["distribution"] == "modal")[
            "version"
        ]
        == "1.5.4"
    )


def test_actual_service_preserves_bytes_and_retained_identity(tmp_path):
    project, source, result, declaration = prepared_project(tmp_path)
    metadata = json.loads((example_path() / "preparation.json").read_bytes())
    identity = result.prepared.identity
    assert identity.ref == declaration.dataset.ref
    assert identity.revision == metadata["expected_semantic_digest"]
    assert identity.content_digest == metadata["source_sha256"]
    assert identity.size_bytes == metadata["source_size_bytes"] == 379
    shutil.rmtree(source)
    lease = result.retained_source.open_lease()
    try:
        raw = lease.take_stream().read()
    finally:
        lease.close()
    assert raw == (example_path() / "rows.jsonl").read_bytes()
    config, rows, _, _ = admit_grpo_snapshot(declaration, identity, raw)
    assert len(rows) == metadata["expected_row_count"] == 4
    assert all(row["prompt"][0]["role"] == "user" for row in rows)
    assert config["training"]["max_steps"] == 3
    assert config["training"]["fp16"] is True and config["training"]["bf16"] is False
    core = compile_grpo_input(declaration, identity)
    assert core.document["dataset_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        core.document["model"]["revision"]
        == core.document["model"]["tokenizer_revision"]
        == REVISION
    )


def test_actual_named_recipe_preplan_without_provider(tmp_path, monkeypatch):
    project, _, result, _ = prepared_project(tmp_path)
    # The installed wheel intentionally has no host image-profile declaration.
    # Copy only actual installed declarative resources + caller build YAML.
    host_assets = tmp_path / "host-assets"
    profiles = host_assets / "runtime_profiles"
    profiles.mkdir(parents=True)
    for suffix in (".yaml", ".inventory.json"):
        shutil.copyfile(
            ASSETS / "Trainers/runtime_profiles" / (NAME + suffix),
            profiles / (NAME + suffix),
        )
    destination = host_assets / "image_profiles" / BUILD_NAME
    destination.mkdir(parents=True)
    shutil.copyfile(build_path(), destination / "profile.yaml")
    from tuner.execution.providers.modal import runtime_build

    calls = []
    installed_root = Path(runtime_build.__file__).resolve().parents[4]

    def fixture_git(argv, **kwargs):
        # Synthetic CPU planning identity only; no deployment/source admission.
        assert argv == ["git", "-C", str(installed_root), "rev-parse", "HEAD"]
        assert kwargs == {"capture_output": True, "timeout": 30, "check": False}
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, ("d" * 40 + "\n").encode(), b"")

    monkeypatch.setattr(runtime_build, "subprocess", SimpleNamespace(run=fixture_git))
    plan = plan_modal_recipe(
        project / "recipe.yaml", project_root=project, profiles_root=profiles
    )
    assert len(calls) == 1
    assert plan.prepared_identity == result.prepared.identity
    assert plan.recipe.runtime_profile == NAME
    assert plan.recipe.child_max_seconds == 300
    assert dict(plan.recipe.qualification_evidence) == {
        "max_trainable_bytes": 16777216,
        "max_trainable_tensors": 256,
        "max_reward_values": 4096,
    }
    assert "runtime_release" not in plan.to_dict()


@pytest.mark.parametrize("change", ["model", "method", "input"])
def test_candidate_recipe_mismatches_reject_before_build(tmp_path, change):
    project, _, _, _ = prepared_project(tmp_path)
    path = project / "recipe.yaml"
    config = yaml.safe_load(path.read_bytes())
    if change == "model":
        config["training_input"]["model"]["revision"] = "b" * 40
    elif change == "method":
        config["method"] = "sft"
    else:
        config["dataset"]["expected_digest"] = "f" * 64
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_modal_grpo_recipe(path, profiles_root=ASSETS / "Trainers/runtime_profiles")


def test_actual_replacement_wheel_bound_not_static_anchor(tmp_path):
    from tuner.execution.providers.modal.runtime_build import (
        prepare_capture_build_inputs,
    )
    from tuner.execution.providers.modal import runtime_build

    profile = load_profile(build_path())
    packaged = deepcopy(profile.packaged_runtime)
    # This control isolates actual replacement derivation, not bootstrap/build qualification.
    packaged["bootstrap"] = []
    fixture_profile = replace(profile, packaged_runtime=packaged)
    inspector = (
        Path(runtime_build.__file__).resolve().parents[4]
        / "tuner/runtime/modal_build_inspector.py"
    )
    wheel = tmp_path / "synaptic_tuner-1.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "tuner/runtime/modal_build_inspector.py", inspector.read_bytes()
        )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert digest != ANCHOR_SHA
    staged = tmp_path / "staged"
    staged.mkdir()
    captured, _, _, _, recipe, _ = prepare_capture_build_inputs(
        fixture_profile, staged, wheel, digest
    )
    assert profile.packaged_runtime["wheel"]["sha256"] == ANCHOR_SHA
    assert captured["wheel"]["sha256"] == digest
    assert recipe["wheels"] == [{"filename": wheel.name, "sha256": digest}]


def test_installed_wheel_contains_runtime_assets_not_host_build_profile():
    wheel = os.environ.get("SYNAPTIC_TEST_GRPO_CANDIDATE_WHEEL")
    if wheel is None:
        pytest.skip(
            "root-owned fresh wheel supplied in dedicated installed candidate lane"
        )
    with zipfile.ZipFile(wheel) as archive:
        for suffix in (".yaml", ".inventory.json"):
            name = "Trainers/runtime_profiles/" + NAME + suffix
            assert archive.read(name) == (ASSETS / name).read_bytes()
        assert not any(
            name.startswith("Trainers/image_profiles/" + BUILD_NAME + "/")
            for name in archive.namelist()
        )
