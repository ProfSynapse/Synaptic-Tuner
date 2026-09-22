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


def test_local_cpu_cli_defaults_to_plan_and_verify_is_read_only(monkeypatch, tmp_path, capsys):
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {"status": "PLAN_ONLY"}
    monkeypatch.setattr(cli, "qualify_local_runtime", capture)
    flags = []
    for name in ("qualification-config", "release", "config", "build-receipt", "candidate", "image-verification", "final-runtime-capture", "output"):
        flags.extend(("--" + name, str(tmp_path / (name + ".json"))))
    assert cli.main(["release", "qualify-local", *flags, "--docker", str(tmp_path / "docker"), "--docker-config", str(tmp_path / "config")]) == 0
    assert calls[0]["execute"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
    monkeypatch.setattr(cli, "verify_local_runtime", capture)
    assert cli.main(["release", "verify-local", *flags, "--evidence", str(tmp_path / "evidence.json")]) == 0
    assert "docker" not in calls[1]


def test_release_capture_without_execute_is_plan_only(tmp_path: Path, capsys) -> None:
    output = tmp_path / "final-capture.json"
    assert cli.main([
        "release", "capture", "--config", str(PROFILE),
        "--build-receipt", str(tmp_path / "missing-build.json"),
        "--candidate", str(tmp_path / "missing-candidate.json"),
        "--image-verification", str(tmp_path / "missing-verification.json"),
        "--docker", str(tmp_path / "docker-does-not-exist"),
        "--docker-config", str(tmp_path / "missing-config"), "--output", str(output),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY"
    assert not output.exists()


def test_packaged_closure_maintenance_preserves_inventory(tmp_path: Path, monkeypatch, capsys) -> None:
    import shutil
    runtime = tmp_path / "tuner" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "manifests").mkdir()
    source = cli.REPO_ROOT / "tuner" / "runtime"
    names = ("packaged_sft_child.py", "packaged_sft_execution.py", "packaged_training_worker.py", "packaged_worker_closure.py", "releases.py")
    for name in names:
        shutil.copyfile(source / name, runtime / name)
    target = runtime / "manifests" / "packaged-training-worker-v1.json"
    shutil.copyfile(source / "manifests" / target.name, target)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli.main(["release", "closure"]) == 0
    before = target.read_bytes()
    (runtime / "releases.py").write_bytes(b"# reviewed test fixture change\n")
    assert cli.main(["release", "closure"]) == 3
    assert target.read_bytes() == before
    # Bind the mandatory installed-resource recheck to the test checkout.
    import tuner.runtime.packaged_worker_closure as closure
    monkeypatch.setattr(closure.importlib.resources, "files", lambda _: runtime)
    assert cli.main(["release", "closure", "--write"]) == 0
    assert cli.main(["release", "closure"]) == 0
    refreshed = json.loads(target.read_bytes())
    assert tuple(item["path"] for item in refreshed["members"]) == names
    refreshed["members"].pop()
    target.write_bytes(closure._canonical(refreshed) + b"\n")
    assert cli.main(["release", "closure", "--write"]) == 125
