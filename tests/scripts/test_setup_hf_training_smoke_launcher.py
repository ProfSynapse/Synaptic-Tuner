from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import setup_hf_training_smoke_launcher as setup


def _write_direct(path: Path, lines=setup.EXPECTED_DIRECT) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lock(path: Path) -> dict[str, str]:
    versions = {
        "certifi": "2026.1.1",
        "huggingface-hub": "1.27.0",
        "jsonschema": "4.23.0",
        "packaging": "24.1",
        "python-dotenv": "1.0.1",
        "pyyaml": "6.0.2",
    }
    lines = [f"{name}=={versions[name]} --hash=sha256:{'a' * 64}" for name in sorted(versions)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return versions


def _write_allowlist(path: Path, lock: Path, versions: dict[str, str]) -> dict[str, str]:
    expected = dict(sorted({**versions, "pip": "24.2"}.items()))
    document = {
        "schema_version": "synaptic-hf-training-launcher-installed/v1",
        "python": setup.EXPECTED_PYTHON,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "distributions": expected,
    }
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return expected


@pytest.mark.parametrize(
    "lines",
    [
        setup.EXPECTED_DIRECT[:-1],
        setup.EXPECTED_DIRECT + ("torch==2.9.0",),
        ("huggingface_hub>=1.27.0",) + setup.EXPECTED_DIRECT[1:],
        ("HUGGINGFACE_HUB==1.27.0",) + setup.EXPECTED_DIRECT[1:],
        ("-e git+https://example.invalid/x",) + setup.EXPECTED_DIRECT[1:],
    ],
)
def test_direct_requirements_reject_drift_and_injection(tmp_path: Path, lines) -> None:
    direct = tmp_path / "direct.txt"
    _write_direct(direct, lines)
    with pytest.raises(setup.LauncherContractError, match="exact ordered"):
        setup.validate_direct_requirements(direct)


def test_checked_in_lock_and_allowlist_are_exact_reviewed_contract() -> None:
    root = Path(setup.__file__).resolve().parents[1]
    lock = root / "requirements-hf-training-smoke.lock"
    attributes = (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "requirements-hf-training-smoke.lock text eol=lf" in attributes
    assert b"\r" not in lock.read_bytes()
    locked = setup.validate_hashed_lock(lock)
    installed = setup.validate_installed_allowlist(
        root / "requirements-hf-training-smoke-installed.json",
        lock_path=lock,
    )
    assert installed == dict(sorted({**locked, "pip": "24.2"}.items()))
    assert installed["huggingface-hub"] == "1.27.0"


@pytest.mark.parametrize(
    "bad_line",
    [
        "torch==2.9.0 --hash=sha256:" + "a" * 64,
        "httpx>=0.28 --hash=sha256:" + "a" * 64,
        "httpx==0.28.1;python_version>'3' --hash=sha256:" + "a" * 64,
        "httpx @ https://example.invalid/x.whl --hash=sha256:" + "a" * 64,
        "-e git+https://example.invalid/x",
        "../local.whl --hash=sha256:" + "a" * 64,
    ],
)
def test_hashed_lock_rejects_extra_ranged_marked_url_vcs_local_and_ml(tmp_path: Path, bad_line: str) -> None:
    lock = tmp_path / "lock.txt"
    versions = _write_lock(lock)
    lock.write_text(lock.read_text() + bad_line + "\n", encoding="utf-8")
    with pytest.raises(setup.LauncherContractError):
        setup.validate_hashed_lock(lock)


def test_allowlist_binds_lock_python_exact_installed_set_and_no_ml(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    versions = _write_lock(lock)
    allowlist = tmp_path / "installed.json"
    expected = _write_allowlist(allowlist, lock, versions)
    assert setup.validate_installed_allowlist(allowlist, lock_path=lock) == expected
    document = json.loads(allowlist.read_text())
    document["distributions"]["torch"] = "2.9.0"
    allowlist.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    with pytest.raises(setup.LauncherContractError):
        setup.validate_installed_allowlist(allowlist, lock_path=lock)


def test_python_must_be_cpython_3127(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout='{"implementation":"CPython","version":[3,12,6]}'))
    with pytest.raises(setup.LauncherContractError, match="3.12.7"):
        setup.require_exact_python(tmp_path / "python")


def test_post_install_verification_runs_pip_dependency_check_before_import_audit(monkeypatch, tmp_path: Path) -> None:
    root = Path(setup.__file__).resolve().parents[1]
    calls = []
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda argv, **kwargs: calls.append((argv, kwargs)) or SimpleNamespace(stdout=""),
    )
    setup.verify_installed(
        launcher_python=tmp_path / "python", repo_root=root,
        expected={"pip": "24.2"}, environment={"PATH": "safe"},
    )
    assert calls[0][0][-3:] == ["-m", "pip", "check"]
    assert calls[1][0][1:3] == ["-I", "-c"]


def test_injection_environment_fails_closed(monkeypatch) -> None:
    for key in setup.INJECTION_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHONPATH", "hostile")
    with pytest.raises(setup.LauncherContractError, match="injection"):
        setup.sanitized_environment()


def test_setup_uses_fresh_venv_require_hashes_no_deps_and_verifies(tmp_path: Path, monkeypatch) -> None:
    direct, lock, allowlist = (tmp_path / name for name in ("direct.txt", "lock.txt", "installed.json"))
    _write_direct(direct)
    versions = _write_lock(lock)
    expected = _write_allowlist(allowlist, lock, versions)
    python, venv = tmp_path / "python", tmp_path / "venv"
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    calls = []
    verified = []
    monkeypatch.setattr(setup, "require_exact_python", lambda value: None)
    monkeypatch.setattr(setup, "sanitized_environment", lambda: {"PATH": "safe"})
    monkeypatch.setattr(setup, "verify_installed", lambda **kwargs: verified.append(kwargs))

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[2:4] == ["-m", "venv"]:
            (venv / ("Scripts" if setup.os.name == "nt" else "bin")).mkdir(parents=True)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    launcher = setup.setup_launcher(
        python=python, venv=venv, direct=direct, lock=lock,
        allowlist=allowlist, wheelhouse=wheelhouse, repo_root=tmp_path,
    )
    install = calls[1][0]
    assert "--require-hashes" in install and "--no-deps" in install
    assert "--no-index" in install and install[install.index("--find-links") + 1] == str(wheelhouse)
    assert verified[0]["expected"] == expected
    assert launcher.name in {"python", "python.exe"}
