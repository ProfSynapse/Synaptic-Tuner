"""Cold preparation and rejection before Modal authority/effects."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.training import modal_local_preparation as local


def _profile(raw=b"pinned-wheel"):
    item = {"distribution": "fixture", "version": "1.0", "filename": "fixture-1.0-py3-none-any.whl",
            "sha256": hashlib.sha256(raw).hexdigest()}
    return SimpleNamespace(packaged_runtime={"bootstrap": [item]}), item


@pytest.mark.parametrize("mask", [0o022, 0o002])
def test_cold_cache_fetches_exact_pins_and_warm_cache_uses_no_network(tmp_path, monkeypatch, mask):
    profile, item = _profile()
    calls = []
    monkeypatch.setattr(local, "_public_artifact", lambda item, **kw: calls.append(item) or b"pinned-wheel")
    prior = os.umask(mask)
    try:
        root = local.private_directory(tmp_path / "private" / "nested")
        result = local.bootstrap_paths(profile, root, prepare=True)
    finally:
        os.umask(prior)
    path = result[item["filename"]]
    assert path.read_bytes() == b"pinned-wheel" and calls == [item]
    assert path.stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700
    assert path.parent.stat().st_mode & 0o777 == 0o700
    monkeypatch.setattr(local, "_public_artifact", lambda *a, **kw: pytest.fail("warm cache downloaded"))
    assert local.bootstrap_paths(profile, root, prepare=True) == result


@pytest.mark.parametrize("fault", ["bytes", "symlink", "mode", "directory-mode"])
def test_corrupt_existing_cache_rejects_without_overwrite_or_download(tmp_path, monkeypatch, fault):
    profile, item = _profile()
    root = local.private_directory(tmp_path / "private")
    directory = local.private_directory(root / "modal-bootstrap" / item["sha256"])
    path = directory / item["filename"]
    path.write_bytes(b"pinned-wheel")
    path.chmod(0o600)
    if fault == "bytes":
        path.write_bytes(b"corrupt")
    elif fault == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "missing")
    elif fault == "mode":
        path.chmod(0o644)
    else:
        directory.chmod(0o775)
    monkeypatch.setattr(local, "_public_artifact", lambda *a, **kw: pytest.fail("corruption downloaded"))
    with pytest.raises(local.ModalLocalPreparationUnavailable):
        local.bootstrap_paths(profile, root, prepare=True)


def test_read_only_plan_does_not_create_or_fetch_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "missing"))
    monkeypatch.setattr(local, "_public_artifact", lambda *a, **kw: pytest.fail("plan downloaded"))
    assert local.local_readiness(Path("unused"))["status"] == "unprepared"
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("fault", ["mode", "symlink"])
def test_private_root_invalid_rejects(tmp_path, fault):
    root = tmp_path / "private"
    if fault == "mode":
        root.mkdir(mode=0o775)
    else:
        root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(local.ModalLocalPreparationUnavailable):
        local.private_directory(root)


@pytest.mark.parametrize("python,modal,code", [("3.12.3", "1.5.4", "LAUNCHER_PYTHON"),
                                               ("3.11.14", "1.6.0", "LAUNCHER_MODAL")])
def test_launcher_mismatch_stops_before_profile_cache_or_source_build(tmp_path, monkeypatch, python, modal, code):
    monkeypatch.setattr(local.platform, "python_version", lambda: python)
    monkeypatch.setattr(local, "version", lambda name: modal)
    monkeypatch.setattr(local, "load_profile", lambda *a: pytest.fail("profile accessed"))
    with pytest.raises(local.ModalLocalPreparationUnavailable) as caught:
        local.prepare_local_inputs(Path("unused"), root=tmp_path / "private", expected_intent="a" * 64)
    assert caught.value.code == code and not (tmp_path / "private").exists()


def _prepared(tmp_path, monkeypatch):
    import tempfile
    import tuner.training.modal_metrics_qualification as admission
    root = local.private_directory(tmp_path / "private")
    temporary = tempfile.TemporaryDirectory(dir=root)
    staged = Path(temporary.name)
    wheel = staged / "synaptic_tuner-1.1.0-py3-none-any.whl"
    wheel.write_bytes(b"engine")
    digest = hashlib.sha256(b"engine").hexdigest()
    inputs, requirements = staged / "build-inputs.json", staged / "requirements.txt"
    inputs.write_bytes(b"inputs")
    requirements.write_bytes(b"requirements")
    packaged = {"wheel": {"filename": wheel.name, "sha256": digest}, "bootstrap": []}
    layout = (packaged, inputs, requirements, "command", {"wheels": [packaged["wheel"]]}, {"kind": "modal_build"})
    intent = {"intent_digest": "a" * 64, "engine_source_commit": "b" * 40}
    monkeypatch.setattr(local, "plan_modal_build_material", lambda path: intent)
    monkeypatch.setattr(admission, "current_clean_commit", lambda root: "b" * 40)
    return local.PreparedModalLocalInputs(Path("profile"), intent, temporary, layout, token=local._ATTESTED)


@pytest.mark.parametrize("fault", ["bytes", "replace", "file-mode", "root-mode", "staged-mode", "layout", "dirty-source"])
def test_retained_inputs_reject_mutation_before_authority(tmp_path, monkeypatch, fault):
    prepared = _prepared(tmp_path, monkeypatch)
    try:
        prepared.revalidate(Path("profile"), "a" * 64)
        if fault == "bytes":
            prepared.wheel.write_bytes(b"modified")
        elif fault == "replace":
            raw = prepared.wheel.read_bytes()
            prepared.wheel.unlink()
            prepared.wheel.write_bytes(raw)
            prepared.wheel.chmod(0o600)
        elif fault == "file-mode":
            prepared.wheel.chmod(0o644)
        elif fault == "root-mode":
            prepared.staged.parent.chmod(0o775)
        elif fault == "staged-mode":
            prepared.staged.chmod(0o775)
        elif fault == "layout":
            prepared.layout[5]["kind"] = "different"
        else:
            import tuner.training.modal_metrics_qualification as admission
            monkeypatch.setattr(admission, "current_clean_commit", lambda root: (_ for _ in ()).throw(ValueError()))
        with pytest.raises(local.ModalLocalPreparationUnavailable) as caught:
            prepared.revalidate(Path("profile"), "a" * 64)
        assert caught.value.code == "LOCAL_INPUTS_CHANGED"
    finally:
        prepared.close()


def test_metadata_hash_and_download_hash_are_independently_checked(monkeypatch):
    profile, item = _profile()
    url = "https://files.pythonhosted.org/packages/" + item["filename"]
    row = {"filename": item["filename"], "url": url, "size": 12,
           "packagetype": "bdist_wheel", "digests": {"sha256": item["sha256"]}}
    document = {"info": {"name": "fixture", "version": "1.0"}, "urls": [row]}
    monkeypatch.setattr(local, "_metadata_bytes", lambda url: json.dumps(document).encode())
    monkeypatch.setattr(local, "_bounded", lambda operation, **kw: operation())
    monkeypatch.setattr(local, "_download_exact", lambda *a, **kw: b"corrupt")
    with pytest.raises(local.ModalLocalPreparationUnavailable) as caught:
        local._public_artifact(item, deadline=1)
    assert caught.value.code == "BOOTSTRAP_HASH_INVALID"
    row["digests"]["sha256"] = "0" * 64
    monkeypatch.setattr(local, "_download_exact", lambda *a, **kw: pytest.fail("bad metadata downloaded"))
    with pytest.raises(local.ModalLocalPreparationUnavailable) as caught:
        local._public_artifact(item, deadline=1)
    assert caught.value.code == "BOOTSTRAP_METADATA_INVALID"
