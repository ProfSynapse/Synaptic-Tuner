from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "scripts" / "prepare_modal_inference_python.py"
SPEC = importlib.util.spec_from_file_location("prepare_modal_inference_python", PATH)
assert SPEC is not None and SPEC.loader is not None
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    (base / "vllm-0.17.1.dist-info").mkdir(parents=True)
    return base


class Builder:
    def __init__(self, **kwargs):
        assert kwargs == {
            "system_site_packages": False,
            "clear": False,
            "symlinks": False,
            "with_pip": False,
        }

    def create(self, destination):
        (destination / "bin").mkdir()
        (destination / "bin" / "python").write_bytes(b"python")
        (destination / "lib" / "python3.12" / "site-packages").mkdir(parents=True)


def _runner(destination: Path, base: Path):
    def run(argv, **kwargs):
        assert argv[:3] == [str(destination / "bin" / "python"), "-I", "-c"]
        assert kwargs["env"] == {"PATH": "/usr/bin:/bin"}
        payload = {
            "base_count": 1,
            "base_prefix": "/base-python",
            "executable": str((destination / "bin" / "python").resolve()),
            "os_site": False,
            "prefix": str(destination.resolve()),
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload).encode())

    return run


def test_prepare_uses_isolated_copied_venv_and_exact_pth(tmp_path, monkeypatch):
    destination = tmp_path / "venv"
    base = _base(tmp_path)
    monkeypatch.setattr(
        preparation.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(preparation.sys, "version_info", (3, 12))
    preparation._prepare(
        destination=destination,
        base_site=base,
        pth_bytes=(str(base) + "\n").encode(),
        builder_factory=Builder,
        runner=_runner(destination, base),
    )
    assert (
        destination / "lib/python3.12/site-packages" / preparation._PTH_NAME
    ).read_bytes() == (str(base) + "\n").encode()


def test_existing_destination_is_rejected_before_builder(tmp_path, monkeypatch):
    destination = tmp_path / "venv"
    destination.mkdir()
    base = _base(tmp_path)
    monkeypatch.setattr(
        preparation.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(preparation.sys, "version_info", (3, 12))
    with pytest.raises(preparation._PreparationFailure, match="DESTINATION_EXISTS"):
        preparation._prepare(
            destination=destination,
            base_site=base,
            pth_bytes=b"unused\n",
            builder_factory=lambda **kwargs: pytest.fail("builder called"),
        )


def test_base_requires_one_vllm_metadata_directory(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(
        preparation._PreparationFailure, match="BASE_ML_METADATA_INVALID"
    ):
        preparation._base_site(base)


def test_base_rejects_vllm_metadata_symlink(tmp_path):
    base = tmp_path / "base"
    target = tmp_path / "metadata"
    base.mkdir()
    target.mkdir()
    (base / "vllm-0.17.1.dist-info").symlink_to(target, target_is_directory=True)
    with pytest.raises(
        preparation._PreparationFailure, match="BASE_ML_METADATA_INVALID"
    ):
        preparation._base_site(base)


def test_preflight_mismatch_is_closed(tmp_path, monkeypatch):
    destination = tmp_path / "venv"
    base = _base(tmp_path)
    monkeypatch.setattr(
        preparation.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(preparation.sys, "version_info", (3, 12))
    runner = lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"private")
    with pytest.raises(
        preparation._PreparationFailure, match="PREPARATION_FAILED"
    ) as caught:
        preparation._prepare(
            destination=destination,
            base_site=base,
            pth_bytes=(str(base) + "\n").encode(),
            builder_factory=Builder,
            runner=runner,
        )
    assert "private" not in str(caught.value)


def test_main_emits_only_closed_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        preparation,
        "_prepare",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private-token")),
    )
    assert preparation.main() == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": "PREPARATION_FAILED",
        "schema_version": preparation._SCHEMA,
        "status": "FAILED",
    }
    assert "private-token" not in captured.err


def test_real_throwaway_venv_preflight(tmp_path):
    if (
        preparation.platform.python_implementation() != "CPython"
        or preparation.sys.version_info[:2] != (3, 12)
    ):
        pytest.skip("requires CPython 3.12")
    destination = tmp_path / "venv"
    base = _base(tmp_path).resolve()
    preparation._prepare(
        destination=destination,
        base_site=base,
        pth_bytes=(str(base) + "\n").encode(),
    )
