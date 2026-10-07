from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import shutil

from scripts import capture_hf_training_image_lock as capture


def test_cli_requires_explicit_image_docker_config_and_output() -> None:
    parser = capture.build_parser()
    args = parser.parse_args(
        [
            "--image", "docker.io/unsloth/unsloth@sha256:" + "a" * 64,
            "--docker", "docker.exe", "--docker-config", "empty", "--output", "image-lock.candidate.json",
        ]
    )
    assert args.output == Path("image-lock.candidate.json")


def test_cli_reports_candidate_only_without_exposing_capture_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        capture, "capture_candidate",
        lambda **kwargs: {"schema_version": "synaptic-hf-training-runtime-lock-candidate/v1"},
    )
    assert capture.main(
        ["--image", "docker.io/unsloth/unsloth@sha256:" + "a" * 64, "--docker", "docker",
         "--docker-config", "empty", "--output", "image-lock.candidate.json"]
    ) == 0
    output = capsys.readouterr().out
    assert "CANDIDATE_ONLY" in output
    assert "packages" not in output and "image" not in output


def test_cli_uses_exit_125_and_closed_reason_without_raw_error(monkeypatch, capsys) -> None:
    from tuner.cloud.hf_training_image_lock import TrainingImageLockError

    secret = "registry-secret-must-not-appear"
    def fail(**_kwargs):
        try:
            raise RuntimeError(secret)
        except RuntimeError as exc:
            raise TrainingImageLockError("EVIDENCE_INVALID") from exc

    monkeypatch.setattr(capture, "capture_candidate", fail)
    assert capture.main(
        ["--image", "docker.io/unsloth/unsloth@sha256:" + "a" * 64, "--docker", "docker",
         "--docker-config", "empty", "--output", "image-lock.candidate.json"]
    ) == 125
    error = capsys.readouterr().err
    assert "EVIDENCE_INVALID" in error and secret not in error


def test_cli_maps_unexpected_parser_resource_failure_without_traceback(monkeypatch, capsys) -> None:
    secret = "provider-controlled-secret"
    monkeypatch.setattr(
        capture, "capture_candidate",
        lambda **_kwargs: (_ for _ in ()).throw(RecursionError(secret)),
    )
    assert capture.main(
        ["--image", "docker.io/unsloth/unsloth@sha256:" + "a" * 64, "--docker", "docker",
         "--docker-config", "empty", "--output", "image-lock.candidate.json"]
    ) == 125
    error = capsys.readouterr().err
    assert error.strip().endswith("COMMAND_FAILED")
    assert secret not in error and "Traceback" not in error


def test_script_runs_isolated_from_any_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(capture.REPO_ROOT / "scripts" / "capture_hf_training_image_lock.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0 and "usage:" in completed.stdout.lower()


def test_copied_script_fails_root_authentication_with_exit_125(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    copied = scripts / "capture_hf_training_image_lock.py"
    shutil.copyfile(capture.REPO_ROOT / "scripts" / copied.name, copied)
    completed = subprocess.run(
        [sys.executable, "-I", str(copied), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 125
    assert completed.stderr.strip().endswith("SCRIPT_IDENTITY_INVALID")
    assert "Traceback" not in completed.stderr
