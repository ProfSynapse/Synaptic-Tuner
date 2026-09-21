from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import qualify_derived_training_image as cli


PROFILE = (
    Path(__file__).resolve().parents[2]
    / "Trainers"
    / "image_profiles"
    / "qwen35_4b_32k_prompt_completion.yaml"
)


def test_plan_is_provider_free(capsys) -> None:
    assert cli.main(["plan", "--config", str(PROFILE)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY"
    assert result["profile"] == "qwen35-4b-32k-prompt-completion"


def test_build_without_execute_only_prints_plan(tmp_path: Path, capsys) -> None:
    missing_docker = tmp_path / "docker-does-not-exist"
    assert cli.main(
        [
            "build",
            "--config",
            str(PROFILE),
            "--docker",
            str(missing_docker),
            "--docker-config",
            str(tmp_path / "config"),
            "--tag",
            "registry.example/syntunia/qwen35:candidate",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY"
    assert not (tmp_path / "receipt.json").exists()


def test_help_works_under_isolated_python() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(cli.REPO_ROOT / "scripts" / Path(cli.__file__).name), "--help"],
        cwd=cli.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
