from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import setup_hf_jp_launcher as setup


def _requirements_text() -> str:
    return "# launcher only\n" + "\n".join(setup.EXPECTED_REQUIREMENTS) + "\n"


def test_requirements_are_exactly_pinned(tmp_path: Path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text(_requirements_text(), encoding="utf-8")
    setup.validate_requirements(valid)
    valid.write_text("huggingface_hub>=1.27.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        setup.validate_requirements(valid)


@pytest.mark.parametrize(
    "lines",
    [
        setup.EXPECTED_REQUIREMENTS[:-1],
        setup.EXPECTED_REQUIREMENTS + ("torch==2.9.0",),
        setup.EXPECTED_REQUIREMENTS + (setup.EXPECTED_REQUIREMENTS[0],),
        ("huggingface_hub>=1.27.0",) + setup.EXPECTED_REQUIREMENTS[1:],
    ],
)
def test_requirements_reject_missing_extra_duplicate_and_ranged_dependencies(
    tmp_path, lines
) -> None:
    path = tmp_path / "requirements.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        setup.validate_requirements(path)


def test_setup_uses_explicit_python_new_venv_and_pinned_requirements(tmp_path, monkeypatch) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(_requirements_text(), encoding="utf-8")
    python = tmp_path / "python3.12"
    venv = tmp_path / "launcher"
    calls = []
    verified = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:3] == ["-c", "import json,sys;print(json.dumps(list(sys.version_info[:2])))"]:
            return SimpleNamespace(stdout="[3, 12]\n")
        if argv[1:3] == ["-m", "venv"]:
            (venv / ("Scripts" if setup.sys.platform == "win32" else "bin")).mkdir(parents=True)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    monkeypatch.setattr(
        setup,
        "verify_launcher",
        lambda **kwargs: verified.append(kwargs),
    )
    launcher = setup.setup_launcher(
        python=python,
        venv=venv,
        requirements=requirements,
        repo_root=tmp_path,
    )
    assert launcher.name in {"python", "python.exe"}
    assert calls[1][0] == [str(python), "-m", "venv", str(venv)]
    assert "--requirement" in calls[2][0]
    assert all(kwargs["check"] for _argv, kwargs in calls)
    assert verified == [{"launcher_python": launcher, "repo_root": tmp_path}]


def test_existing_target_is_never_repaired_or_overwritten(tmp_path, monkeypatch) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(_requirements_text(), encoding="utf-8")
    venv = tmp_path / "existing"
    venv.mkdir()
    monkeypatch.setattr(setup, "require_python_312", lambda python: None)
    with pytest.raises(FileExistsError, match="refusing"):
        setup.setup_launcher(
            python=tmp_path / "python",
            venv=venv,
            requirements=requirements,
            repo_root=tmp_path,
        )


def test_python_version_must_be_exactly_312(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="[3, 13]\n"),
    )
    with pytest.raises(ValueError, match="3.12 exactly"):
        setup.require_python_312(tmp_path / "python")


def test_post_install_smoke_uses_exact_worktree_and_sanitized_environment(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(setup.__file__).resolve().parents[1]
    launcher = tmp_path / "launcher-python"
    calls = []
    monkeypatch.setenv("HF_TOKEN", "ambient-must-not-cross")
    monkeypatch.setenv("HF_API_KEY", "alias-must-not-cross")

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        command = argv[-2] if argv[-1] == "--help" else ""
        return SimpleNamespace(stdout=f"protected help {command}\n")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    setup.verify_launcher(launcher_python=launcher, repo_root=repo_root)
    assert len(calls) == 3
    assert calls[0][0][1:3] == ["-I", "-c"]
    assert calls[1][0][-2:] == ["hf-source", "--help"]
    assert calls[2][0][-2:] == ["hf-smoke", "--help"]
    for _argv, kwargs in calls:
        assert kwargs["cwd"] == repo_root.resolve()
        assert kwargs["check"] is True
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == str(repo_root.resolve())
        assert "HF_TOKEN" not in kwargs["env"]
        assert "HF_API_KEY" not in kwargs["env"]


def test_post_install_smoke_requires_exact_repo_worktree(tmp_path) -> None:
    with pytest.raises(ValueError, match="exact repository worktree"):
        setup.verify_launcher(
            launcher_python=tmp_path / "python",
            repo_root=tmp_path,
        )
