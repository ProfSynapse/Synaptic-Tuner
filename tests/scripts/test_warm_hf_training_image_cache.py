from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from scripts import warm_hf_training_image_cache as warm


def _argv() -> list[str]:
    return [
        "--image", "docker.io/unsloth/unsloth@sha256:" + "a" * 64,
        "--docker", "docker.exe", "--docker-config", "empty",
    ]


def test_cli_requires_only_exact_image_docker_and_empty_config() -> None:
    args = warm.build_parser().parse_args(_argv())
    assert args.docker == Path("docker.exe")
    assert not hasattr(args, "output")


def test_cli_reports_status_without_receipt_or_identity_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(warm, "warm_image_cache", lambda **kwargs: {"status": "CACHE_WARMED"})
    assert warm.main(_argv()) == 0
    output = capsys.readouterr().out
    assert output.strip() == '{"status": "CACHE_WARMED"}'
    assert "digest" not in output and "path" not in output and "receipt" not in output


def test_cli_uses_closed_failure_without_traceback_or_raw_error(monkeypatch, capsys) -> None:
    from tuner.cloud.hf_training_image_lock import TrainingImageLockError

    monkeypatch.setattr(
        warm, "warm_image_cache",
        lambda **kwargs: (_ for _ in ()).throw(TrainingImageLockError("OPERATION_TIMEOUT")),
    )
    assert warm.main(_argv()) == 125
    error = capsys.readouterr().err
    assert error.strip().endswith("OPERATION_TIMEOUT") and "Traceback" not in error


def test_script_runs_isolated_from_any_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(warm.REPO_ROOT / "scripts" / "warm_hf_training_image_cache.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0 and "usage:" in completed.stdout.lower()


def test_copied_script_fails_root_authentication(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    copied = scripts / "warm_hf_training_image_cache.py"
    shutil.copyfile(warm.REPO_ROOT / "scripts" / copied.name, copied)
    completed = subprocess.run(
        [sys.executable, "-I", str(copied), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 125
    assert completed.stderr.strip().endswith("SCRIPT_IDENTITY_INVALID")
    assert "Traceback" not in completed.stderr
