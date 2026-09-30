"""The host builder fetches only reviewed wheel bytes and then stays offline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tuner.execution.providers.modal import modal_wheel_builder as builder


def _item(raw: bytes) -> dict[str, object]:
    return {
        "filename": "pip-26.2-py3-none-any.whl",
        "url": "https://files.pythonhosted.org/packages/fixed/pip-26.2-py3-none-any.whl",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_locked_build_closure_is_exact_and_supported():
    wheels = builder._locked_wheels()
    assert [(w["name"], w["version"]) for w in wheels] == [
        ("pip", "26.2"), ("setuptools", "84.0.0"),
        ("wheel", "0.48.0"), ("packaging", "26.3"),
    ]
    assert sum(int(w["size"]) for w in wheels) < 3 * 1024 * 1024


@pytest.mark.skipif(os.name != "posix", reason="retained POSIX file identity")
def test_wheel_cache_reuses_only_verified_bytes(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    raw = b"verified-wheel"
    item = _item(raw)
    calls = []
    monkeypatch.setattr(builder, "_download_exact", lambda url, size: calls.append(url) or raw)
    assert builder._wheel_bytes(item, cache) == raw
    assert len(calls) == 1
    monkeypatch.setattr(builder, "_download_exact", lambda *_: pytest.fail("cache miss"))
    assert builder._wheel_bytes(item, cache) == raw
    wheel = cache / str(item["filename"])
    wheel.write_bytes(b"tampered-wheel")
    with pytest.raises(ValueError, match="cached Modal wheel"):
        builder._wheel_bytes(item, cache)


@pytest.mark.skipif(os.name != "posix", reason="retained POSIX file identity")
def test_download_hash_mismatch_never_populates_cache(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    item = _item(b"good")
    monkeypatch.setattr(builder, "_download_exact", lambda *_: b"evil")
    with pytest.raises(ValueError, match="digest differs"):
        builder._wheel_bytes(item, cache)
    assert list(cache.iterdir()) == []


@pytest.mark.skipif(os.name != "posix" or sys.version_info[:2] != (3, 11),
                    reason="locked Linux builder Python")
def test_bootstrap_and_build_commands_have_no_resolver(tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    cache = tmp_path / "cache"
    seen = []
    wheels = builder._locked_wheels()
    monkeypatch.setattr(builder, "_wheel_bytes", lambda *_: b"wheel-fixture")

    def run(command, **kwargs):
        seen.append((command, kwargs))
        if command[-2:] == ["-c", command[-1]]:
            payload = {str(w["name"]): str(w["version"]) for w in wheels}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload).encode(), b"")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(builder.subprocess, "run", run)
    path = builder.create_offline_wheel_builder(scratch, cache)
    assert path == scratch / "builder" / "bin" / "python"
    install = seen[1][0]
    assert {"--no-index", "--no-deps", "--require-hashes"}.issubset(install)
    assert "PYTHONPATH" in seen[1][1]["env"]
    assert "PYTHONPATH" not in seen[2][1]["env"]


def test_lock_rejects_unlisted_dependency(monkeypatch):
    lock = json.loads(builder.builder_lock_bytes())
    lock["wheels"].append(dict(lock["wheels"][0]))
    monkeypatch.setattr(builder, "builder_lock_bytes", lambda: json.dumps(lock).encode())
    with pytest.raises(ValueError, match="builder lock is invalid"):
        builder._locked_wheels()
