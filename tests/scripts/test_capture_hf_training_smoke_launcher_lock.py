from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import capture_hf_training_smoke_launcher_lock as capture


WHEELS = {
    "huggingface-hub": ("1.27.0", "huggingface_hub-1.27.0-py3-none-any.whl"),
    "jsonschema": ("4.23.0", "jsonschema-4.23.0-py3-none-any.whl"),
    "packaging": ("24.1", "packaging-24.1-py3-none-any.whl"),
    "python-dotenv": ("1.0.1", "python_dotenv-1.0.1-py3-none-any.whl"),
    "pyyaml": ("6.0.2", "PyYAML-6.0.2-cp312-cp312-win_amd64.whl"),
}


class HermeticRunner:
    def __init__(self) -> None:
        self.calls: list[capture.CommandSpec] = []
        self.wheel_bytes = {
            name: f"synthetic-wheel:{name}:{version}\n".encode("ascii")
            for name, (version, _filename) in WHEELS.items()
        }

    def __call__(self, spec: capture.CommandSpec) -> capture.CommandResult:
        self.calls.append(spec)
        argv = list(spec.argv)
        if "-c" in argv:
            if "distributions" in argv[-1]:
                return capture.CommandResult(
                    capture._canonical_json(
                        {
                            "target": capture.TARGET,
                            "distributions": dict(
                                sorted({**{name: value[0] for name, value in WHEELS.items()}, "pip": "24.2"}.items())
                            ),
                        }
                    )
                )
            return capture.CommandResult(capture._canonical_json(capture.TARGET))
        if "--report" in argv:
            report_path = Path(argv[argv.index("--report") + 1])
            install = []
            for name, (version, filename) in WHEELS.items():
                digest = hashlib.sha256(self.wheel_bytes[name]).hexdigest()
                install.append(
                    {
                        "download_info": {
                            "url": f"https://files.example.invalid/{filename}",
                            "archive_info": {"hashes": {"sha256": digest}},
                        },
                        "is_direct": False,
                        "is_yanked": False,
                        "metadata": {"name": name, "version": version},
                        "requested": True,
                    }
                )
            report_path.write_bytes(
                capture._canonical_json(
                    {"version": "1", "pip_version": "24.2", "install": install, "environment": {}}
                )
            )
        if "download" in argv:
            wheelhouse = Path(argv[argv.index("--dest") + 1])
            for name, (_version, filename) in WHEELS.items():
                (wheelhouse / filename).write_bytes(self.wheel_bytes[name])
        return capture.CommandResult()


def _direct(path: Path) -> Path:
    path.write_text("\n".join(capture.setup.EXPECTED_DIRECT) + "\n", encoding="utf-8")
    return path


def _capture(tmp_path: Path, suffix: str = "one") -> tuple[Path, HermeticRunner]:
    runner = HermeticRunner()
    repository = tmp_path / "repository"
    repository.mkdir(exist_ok=True)
    output = capture.capture_candidate(
        python=tmp_path / "python.exe",
        direct=_direct(tmp_path / f"direct-{suffix}.txt"),
        workspace=tmp_path / f"workspace-{suffix}",
        output=tmp_path / f"candidate-{suffix}",
        repo_root=repository,
        runner=runner,
    )
    return output, runner


def test_candidate_capture_is_deterministic_closed_and_never_promotes(tmp_path: Path) -> None:
    canonical_lock = capture.REPO_ROOT / "requirements-hf-training-smoke.lock"
    canonical_allowlist = capture.REPO_ROOT / "requirements-hf-training-smoke-installed.json"
    before = (canonical_lock.read_bytes(), canonical_allowlist.read_bytes())
    first, _ = _capture(tmp_path, "one")
    second, _ = _capture(tmp_path, "two")

    names = {
        "requirements-hf-training-smoke.lock.candidate",
        "requirements-hf-training-smoke-installed.candidate.json",
        "hf-training-smoke-launcher-lock.candidate.json",
        "wheelhouse",
    }
    assert {path.name for path in first.iterdir()} == names
    for filename in names - {"wheelhouse"}:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert sorted(path.name for path in (first / "wheelhouse").iterdir()) == sorted(
        path.name for path in (second / "wheelhouse").iterdir()
    )
    evidence = json.loads((first / "hf-training-smoke-launcher-lock.candidate.json").read_bytes())
    assert set(evidence) == {
        "schema_version", "status", "target", "direct_requirements", "lock_sha256",
        "allowlist_sha256", "report_sha256", "wheelhouse", "verification",
    }
    assert evidence["status"] == "CANDIDATE" and evidence["target"] == capture.TARGET
    assert (canonical_lock.read_bytes(), canonical_allowlist.read_bytes()) == before


def test_pipeline_uses_report_target_hashed_wheelhouse_offline_install_check_and_inspect(tmp_path: Path) -> None:
    _output, runner = _capture(tmp_path)
    calls = [list(spec.argv) for spec in runner.calls]
    report = next(call for call in calls if "--report" in call)
    download = next(call for call in calls if "download" in call)
    install = next(call for call in calls if "--no-index" in call)
    assert ["--platform", "win_amd64"] == report[report.index("--platform"):report.index("--platform") + 2]
    assert "--abi" in report and report[report.index("--abi") + 1] == "cp312"
    assert "--only-binary=:all:" in report and "--only-binary=:all:" in download
    assert "--find-links" in install and "--require-hashes" in install and "--no-deps" in install
    assert any(call[-3:] == ["-m", "pip", "check"] for call in calls)
    assert sum("-c" in call for call in calls) == 3


def test_capture_rejects_nonexternal_existing_output_and_workspace(tmp_path: Path) -> None:
    direct = _direct(tmp_path / "direct.txt")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(capture.CandidateCaptureError, match="OUTPUT_INVALID"):
        capture.capture_candidate(
            python=tmp_path / "python.exe", direct=direct,
            workspace=tmp_path / "workspace", output=existing, runner=HermeticRunner(),
        )


def test_capture_sanitizes_runner_failures_without_raw_output(tmp_path: Path) -> None:
    secret = "never-echo-this-subprocess-output"

    def hostile_runner(_spec):
        raise RuntimeError(secret)

    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(capture.CandidateCaptureError) as caught:
        capture.capture_candidate(
            python=tmp_path / "python.exe", direct=_direct(tmp_path / "direct.txt"),
            workspace=tmp_path / "workspace", output=tmp_path / "candidate",
            repo_root=repository, runner=hostile_runner,
        )
    assert str(caught.value) == "COMMAND_FAILED" and secret not in str(caught.value)


def test_capture_rejects_report_with_ml_distribution(tmp_path: Path) -> None:
    runner = HermeticRunner()
    original = runner.__call__

    def hostile(spec: capture.CommandSpec) -> capture.CommandResult:
        result = original(spec)
        if "--report" in spec.argv:
            path = Path(spec.argv[spec.argv.index("--report") + 1])
            report = json.loads(path.read_bytes())
            report["install"][0]["metadata"] = {"name": "torch", "version": "2.9.0"}
            path.write_bytes(capture._canonical_json(report))
        return result

    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(capture.CandidateCaptureError, match="REPORT_INVALID"):
        capture.capture_candidate(
            python=tmp_path / "python.exe", direct=_direct(tmp_path / "direct.txt"),
            workspace=tmp_path / "workspace", output=tmp_path / "candidate",
            repo_root=repository, runner=hostile,
        )


@pytest.mark.parametrize(
    "script",
    ["setup_hf_training_smoke_launcher.py", "capture_hf_training_smoke_launcher_lock.py"],
)
def test_scripts_run_isolated_from_any_working_directory(tmp_path: Path, script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(capture.REPO_ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower()
