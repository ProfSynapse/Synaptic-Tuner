import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tuner" / "backends" / "training" / "cloud"))

from modal_job_plan import (  # noqa: E402
    ConfigurationError,
    RepoSource,
    build_modal_sft_plan,
)


COMMIT = "a" * 40
IMAGE = "registry.example/research/train@sha256:" + "b" * 64


def _inputs(tmp_path: Path, *, configured_dataset: str = "/vol/inputs/smoke/data.jsonl"):
    repo = tmp_path / "repo"
    (repo / "Trainers" / "cloud").mkdir(parents=True)
    (repo / "Trainers" / "cloud" / "train_modal.py").write_text("# wrapper\n")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations": []}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"dataset": {"local_file": configured_dataset}}),
        encoding="utf-8",
    )
    return repo, config, dataset


def _plan(tmp_path: Path, **overrides):
    repo, config, dataset = _inputs(tmp_path)
    kwargs = dict(
        repo_root=repo,
        config_path=config,
        dataset_path=dataset,
        input_volume_name="private-smoke-inputs",
        input_prefix="smoke",
        runtime_image=IMAGE,
        pip_packages=["peft==0.18.1", "transformers==5.5.0"],
        gpu="A10G",
        timeout_hours=2,
        source=RepoSource(
            url="https://github.com/example/research.git", branch="feature/smoke", commit=COMMIT
        ),
    )
    kwargs.update(overrides)
    return build_modal_sft_plan(**kwargs)


def test_plan_hash_binds_inputs_and_emits_detached_exact_command(tmp_path):
    plan = _plan(tmp_path)

    assert plan["inspection_only"] is True
    assert len(plan["inputs"]["config"]["sha256"]) == 64
    assert len(plan["inputs"]["dataset"]["sha256"]) == 64
    assert plan["inputs"]["dataset"]["mounted_path"] == "/vol/inputs/smoke/data.jsonl"
    assert plan["runtime"]["gpu"] == "A10G"
    assert plan["runtime"]["timeout_hours"] == 2
    assert plan["artifacts"]["canonical_root"] == "/vol/artifacts/outputs/runs/modal/sft"
    assert plan["artifacts"]["publish_final_model"] is False

    launch = plan["launch"]
    assert launch["argv"][:3] == ["modal", "run", "--detach"]
    assert launch["argv"][launch["argv"].index("--repo-commit") + 1] == COMMIT
    assert launch["argv"][3].endswith("train_modal.py::run_training")
    assert launch["environment"]["MODAL_GPU"] == "A10G"
    assert launch["environment"]["MODAL_TIMEOUT_SECONDS"] == "7200"
    assert launch["verification"]["require_running_or_completed_task"] is True
    assert json.loads(launch["environment"]["MODAL_TRAINING_PIP_PACKAGES_JSON"]) == [
        "peft==0.18.1",
        "transformers==5.5.0",
    ]


def test_plan_rejects_config_that_points_anywhere_except_staged_dataset(tmp_path):
    repo, config, dataset = _inputs(tmp_path, configured_dataset="/tmp/data.jsonl")
    with pytest.raises(ConfigurationError, match="dataset.local_file must name the staged"):
        build_modal_sft_plan(
            repo_root=repo,
            config_path=config,
            dataset_path=dataset,
            input_volume_name="inputs",
            input_prefix="smoke",
            runtime_image=IMAGE,
            pip_packages=[],
            gpu="A10G",
            timeout_hours=1,
            source=RepoSource(url="https://example/repo.git", branch="main", commit=COMMIT),
        )


@pytest.mark.parametrize("prefix", ["", "/absolute", "../escape", "a/../b"])
def test_plan_rejects_unsafe_input_prefix(tmp_path, prefix):
    with pytest.raises(ConfigurationError, match="input_prefix"):
        _plan(tmp_path, input_prefix=prefix)


def test_plan_requires_immutable_runtime_image(tmp_path):
    with pytest.raises(ConfigurationError, match="immutable"):
        _plan(tmp_path, runtime_image="unsloth/unsloth:latest")


def test_plan_rejects_unpinned_pip_overlay(tmp_path):
    with pytest.raises(ConfigurationError, match="pip overlays"):
        _plan(tmp_path, pip_packages=["peft>=0.18"])


def test_plan_rejects_credential_bearing_repo_url(tmp_path):
    with pytest.raises(ConfigurationError, match="Credential-bearing"):
        _plan(
            tmp_path,
            source=RepoSource(
                url="https://token@example.invalid/repo.git",
                branch="main",
                commit=COMMIT,
            ),
        )


def test_plan_fails_closed_on_missing_private_dataset(tmp_path):
    repo, config, dataset = _inputs(tmp_path)
    dataset.unlink()
    with pytest.raises((ConfigurationError, FileNotFoundError)):
        build_modal_sft_plan(
            repo_root=repo,
            config_path=config,
            dataset_path=dataset,
            input_volume_name="inputs",
            input_prefix="smoke",
            runtime_image=IMAGE,
            pip_packages=[],
            gpu="A10G",
            timeout_hours=1,
            source=RepoSource(url="https://example/repo.git", branch="main", commit=COMMIT),
        )
