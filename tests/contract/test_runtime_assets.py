"""Executable boundary for Synaptic Tuner runtime assets and wheel claims.

This test deliberately distinguishes an editable/source distribution from a
self-contained wheel. Merely adding ``pyproject.toml`` or successfully building
a wheel does not assert that checkout-relative runtime assets are available.
"""

from __future__ import annotations

import os
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PUBLICATION_ENV = "SYNAPTIC_REQUIRE_SELF_CONTAINED_WHEEL"
WHEEL_ENV = "SYNAPTIC_WHEEL_UNDER_TEST"
RUNTIME_BUNDLE_PREFIX = "synaptic_tuner/runtime/"
BOOTSTRAP_CAPSULE_MEMBERS = (
    "tuner/cloud/bootstrap_core.py",
    "tuner/cloud/bootstrap_capsule.py",
)
BOOTSTRAP_CAPSULE_SCHEMA = "schemas/synaptic-bootstrap-capsule-v1.schema.json"


@dataclass(frozen=True)
class AssetFamily:
    """A deliberate family in the supported runtime boundary."""

    name: str
    classification: str
    patterns: tuple[str, ...]


PACKAGE_RESOURCE_FAMILIES = (
    AssetFamily("project schemas", "package-resource", ("schemas/synaptic-*.schema.json",)),
    AssetFamily("training method catalog", "package-resource", ("Trainers/methods.yaml",)),
    AssetFamily(
        "trainer defaults and registries",
        "package-resource",
        (
            "Trainers/sft/configs/**/*.yaml",
            "Trainers/kto/configs/**/*.yaml",
            "Trainers/dpo/configs/**/*.yaml",
            "Trainers/grpo/configs/**/*.yaml",
            "Trainers/embedding/configs/**/*.yaml",
            "Trainers/ace_step/configs/**/*.yaml",
        ),
    ),
    AssetFamily("job recipe catalog", "package-resource", ("Trainers/recipes/*.yaml",)),
    AssetFamily(
        "cloud defaults and experiments",
        "package-resource",
        ("Trainers/cloud/cloud_config.yaml", "Trainers/cloud/experiments/*.yaml"),
    ),
    AssetFamily(
        "evaluator configuration and recipes",
        "package-resource",
        ("Evaluator/config/**/*.yaml", "Evaluator/recipes/*.yaml"),
    ),
    AssetFamily(
        "SynthChat core configuration",
        "package-resource",
        (
            "SynthChat/config/defaults.yaml",
            "SynthChat/config/label_mappings.yaml",
            "SynthChat/config/privacy_profiles.yaml",
            "SynthChat/config/settings.yaml",
            "SynthChat/config/tool_call_formats.yaml",
            "SynthChat/config/validation.yaml",
            "SynthChat/config/workspace_formats.yaml",
        ),
    ),
    AssetFamily(
        "SynthChat rubric and scenario catalogs",
        "package-resource",
        (
            "SynthChat/rubrics/*.yaml",
            "SynthChat/rubrics/*.example",
            "SynthChat/scenarios/*.yaml",
        ),
    ),
    AssetFamily(
        "MechInterp templates",
        "package-resource",
        (
            "MechInterp/configs/templates/*.yaml",
            "MechInterp/configs/templates/*.json",
            "MechInterp/configs/templates/*.jsonl",
        ),
    ),
    AssetFamily(
        "engine workflow presets",
        "package-resource",
        (
            "configs/flywheel/*.yaml",
            "configs/prompt_optimization/*.yaml",
            "configs/lora_surgery.yaml",
            "configs/transcript_import/default.yaml",
        ),
    ),
    AssetFamily(
        "tool schema catalogs",
        "package-resource",
        ("cli-first-tool-schemas.json", "Tools/tool_schemas.json"),
    ),
)


ENGINE_CHECKOUT_FAMILIES = (
    AssetFamily(
        "trainer entry points",
        "engine-checkout",
        (
            "Trainers/sft/train_sft.py",
            "Trainers/kto/train_kto.py",
            "Trainers/dpo/train_dpo.py",
            "Trainers/grpo/train_grpo.py",
            "Trainers/grpo/train_env_grpo.py",
            "Trainers/embedding/train_embedding.py",
            "Trainers/ace_step/train_ace_step.py",
            "Trainers/mlx_sft_mac/train_sft.py",
        ),
    ),
    AssetFamily(
        "provider Python entry points",
        "engine-checkout",
        (
            "Trainers/cloud/train_modal.py",
            "Trainers/cloud/runpod_sync.py",
            "Evaluator/cloud_hf_job.py",
            "Evaluator/cloud_hf_job_vllm.py",
            "MechInterp/cloud/modal_runner.py",
        ),
    ),
    AssetFamily(
        "path-executed CLI helpers",
        "engine-checkout",
        ("Tools/compare_runs.py", "Tools/convert_to_webllm.py", "scripts/cloud_gguf_convert.py"),
    ),
)


PROVIDER_RESOURCE_FAMILIES = (
    AssetFamily(
        "MechInterp Docker build context",
        "provider-container",
        (
            "docker/mechinterp-runner/Dockerfile",
            "docker/mechinterp-runner/entrypoint.sh",
            "docker/mechinterp-runner/print_provenance.py",
        ),
    ),
    AssetFamily(
        "warm vLLM Space template",
        "provider-container",
        (
            "Trainers/cloud/spaces/vllm_warm/Dockerfile.tmpl",
            "Trainers/cloud/spaces/vllm_warm/entrypoint.sh",
            "Trainers/cloud/spaces/vllm_warm/sync_bucket_prefix.py",
        ),
    ),
    AssetFamily(
        "ACE-STEP image definition",
        "provider-container",
        ("Trainers/ace_step/Dockerfile", "Trainers/ace_step/requirements.txt"),
    ),
    AssetFamily(
        "runtime dependency overlays",
        "provider-container",
        (
            "Evaluator/requirements.txt",
            "Trainers/sft/requirements.txt",
            "Trainers/kto/requirements.txt",
            "Trainers/dpo/requirements.txt",
            "Trainers/grpo/requirements.txt",
            "Trainers/embedding/requirements.txt",
            "Trainers/mlx_sft_mac/requirements.txt",
            "requirements-cloud.txt",
            "requirements-flywheel.txt",
        ),
    ),
)


SELF_CONTAINED_WHEEL_FAMILIES = (
    *PACKAGE_RESOURCE_FAMILIES,
    *ENGINE_CHECKOUT_FAMILIES,
    *PROVIDER_RESOURCE_FAMILIES,
)

FORBIDDEN_WHEEL_PREFIXES = (
    ".tracking/",
    ".synaptic/",
    "Datasets/",
    "Evaluator/results/",
    "Evaluator/interactions/",
    "SynthChat/output/",
    "personal_finetune/",
    "tmp/",
)


def _expand(pattern: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.glob(pattern)
            if path.is_file()
        )
    )


def _expanded_assets(families: tuple[AssetFamily, ...]) -> tuple[str, ...]:
    assets: set[str] = set()
    for family in families:
        for pattern in family.patterns:
            matches = _expand(pattern)
            assert matches, f"runtime asset pattern has no source members: {family.name}: {pattern}"
            assets.update(matches)
    return tuple(sorted(assets))


def _runtime_asset_metadata() -> dict[str, object]:
    if not PYPROJECT.is_file():
        return {}
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    synaptic = tool.get("synaptic-tuner", {})
    if not isinstance(synaptic, dict):
        return {}
    metadata = synaptic.get("runtime-assets", {})
    return metadata if isinstance(metadata, dict) else {}


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _wheel_members(wheel_path: Path) -> set[str]:
    assert wheel_path.is_file(), f"{WHEEL_ENV} does not identify a file: {wheel_path}"
    assert wheel_path.suffix == ".whl", f"{WHEEL_ENV} must identify a .whl file: {wheel_path}"
    with zipfile.ZipFile(wheel_path) as archive:
        return {name.replace("\\", "/").lstrip("./") for name in archive.namelist()}


def _asset_is_in_wheel(asset: str, members: set[str]) -> bool:
    return asset in members or f"{RUNTIME_BUNDLE_PREFIX}{asset}" in members


def test_required_runtime_asset_families_are_present_in_source() -> None:
    assets = _expanded_assets(SELF_CONTAINED_WHEEL_FAMILIES)

    assert "schemas/synaptic-project-v1.schema.json" in assets
    assert "Trainers/methods.yaml" in assets
    assert "Evaluator/config/eval_run.yaml" in assets
    assert "SynthChat/config/validation.yaml" in assets
    assert "MechInterp/configs/templates/pipeline.yaml" in assets
    assert "docker/mechinterp-runner/print_provenance.py" in assets


def test_bootstrap_capsule_assets_are_explicit_and_run_agnostic() -> None:
    for member in (*BOOTSTRAP_CAPSULE_MEMBERS, BOOTSTRAP_CAPSULE_SCHEMA):
        assert (REPO_ROOT / member).is_file(), f"missing J0 bootstrap asset: {member}"

    from tuner.cloud.bootstrap_capsule import CAPSULE_MODULE_PATHS

    assert CAPSULE_MODULE_PATHS == BOOTSTRAP_CAPSULE_MEMBERS
    assert all("source_lock" not in member and "policy" not in member for member in CAPSULE_MODULE_PATHS)


def test_bootstrap_capsule_does_not_enable_self_contained_wheel_publication() -> None:
    metadata = _runtime_asset_metadata()
    assert metadata.get("self-contained-wheel", False) is False


def test_runtime_asset_manifest_excludes_outputs_state_and_private_data() -> None:
    assets = _expanded_assets(SELF_CONTAINED_WHEEL_FAMILIES)
    violations = [
        asset
        for asset in assets
        if any(asset.startswith(prefix) for prefix in FORBIDDEN_WHEEL_PREFIXES)
    ]

    assert not violations, f"runtime asset inventory crosses the host/output boundary: {violations}"


def test_self_contained_wheel_publication_gate() -> None:
    """Fail closed when publication is requested or self-containment is claimed."""

    metadata = _runtime_asset_metadata()
    claim = metadata.get("self-contained-wheel", False)
    assert isinstance(claim, bool), (
        "[tool.synaptic-tuner.runtime-assets] self-contained-wheel must be a TOML boolean"
    )

    publication_required = _truthy_environment(PUBLICATION_ENV)
    wheel_value = os.environ.get(WHEEL_ENV, "").strip()

    if not claim:
        assert not publication_required, (
            "self-contained wheel publication is blocked: pyproject.toml must explicitly set "
            "[tool.synaptic-tuner.runtime-assets] self-contained-wheel = true only after the "
            "runtime boundary is implemented"
        )
        return

    assert wheel_value, (
        "a self-contained wheel claim requires evidence: set "
        f"{WHEEL_ENV} to the built wheel and run this contract"
    )
    members = _wheel_members(Path(wheel_value).expanduser().resolve())
    required_assets = _expanded_assets(SELF_CONTAINED_WHEEL_FAMILIES)
    missing = [asset for asset in required_assets if not _asset_is_in_wheel(asset, members)]

    assert not missing, (
        "wheel claims self-contained support but omits runtime assets; missing "
        f"{len(missing)} member(s): {missing[:25]}"
    )
