"""Private native model preparation contracts; no SDK network or ML imports."""

from hashlib import sha1
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest

from examples.training_qualification import native_model_preparation as preparation


def _git(raw):
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


@pytest.fixture
def candidate():
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-model-prep-", dir=Path.home()) as name:
            root = Path(name)
            files = {"config.json": b"{}", "model.safetensors": b"weights",
                     "tokenizer.json": b"{}"}
            config = {
                "schema": "synaptic-native-model-acquisition/v1",
                "repository": "reviewed/model", "revision": "a" * 40,
                "workspace_root": str(root), "maximum_bytes": 512,
                "maximum_files": 3, "timeout_seconds": 5,
                "members": [{"path": path, "size": len(raw), "kind": "git",
                             "upstream_digest": _git(raw)} for path, raw in sorted(files.items())],
                "tokenizer_members": ["tokenizer.json"],
            }
            (root / "config.json").write_bytes(preparation._canonical(config))
            yield config, files, root
    finally:
        os.umask(previous)


def _sdk(config, files, *, add_extra=False, redirect=None):
    calls = []

    class Api:
        def __init__(self, *, endpoint, token):
            assert endpoint == "https://huggingface.co" and token is False

        def model_info(self, repository, *, revision, files_metadata, token, timeout):
            calls.append("metadata")
            assert repository == config["repository"] and revision == config["revision"]
            assert files_metadata and token is False and timeout == 30
            siblings = [SimpleNamespace(rfilename=name, size=len(raw), lfs=None,
                                        blob_id=_git(raw)) for name, raw in sorted(files.items())]
            if add_extra:
                siblings.append(SimpleNamespace(rfilename="extra.txt", size=1, lfs=None,
                                                blob_id=_git(b"x")))
            return SimpleNamespace(sha=revision, siblings=siblings)

    def download(**options):
        calls.append("download")
        assert options["repo_id"] == config["repository"]
        assert options["revision"] == config["revision"] and options["token"] is False
        assert options["max_workers"] == 1 and options["etag_timeout"] == 10
        assert options["allow_patterns"] == sorted(files)
        root = Path(options["local_dir"])
        for name, raw in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if redirect == name:
                target.symlink_to(root / "config.json")
            else:
                target.write_bytes(raw)
        return str(root)

    return Api, download, calls


def test_model_plan_is_local_and_fake_sdk_publishes_only_exact_bytes(candidate, capsys):
    config, files, root = candidate
    assert preparation.main(["inspect", str(root / "config.json")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "REVIEW_REQUIRED_NO_NETWORK"
    api, download, calls = _sdk(config, files)
    receipt = preparation._acquire(config, api_type=api, download=download)
    assert calls == ["metadata", "download"]
    assert receipt["status"] == "OBSERVED_NOT_ADMITTED"
    model = json.loads((root / "model-snapshot.json").read_bytes())
    tokenizer = json.loads((root / "tokenizer-snapshot.json").read_bytes())
    assert [item["path"] for item in model["files"]] == sorted(files)
    assert [item["path"] for item in tokenizer["files"]] == ["tokenizer.json"]
    assert (root / "model-snapshot/model.safetensors").read_bytes() == b"weights"
    assert (root / "tokenizer-snapshot/tokenizer.json").read_bytes() == b"{}"
    with pytest.raises(ValueError, match="prior acquisition claim"):
        preparation._acquire(config, api_type=api, download=download)
    assert calls == ["metadata", "download"]


def test_nested_model_publication_stays_private_under_ambient_group_umask(candidate):
    config, files, root = candidate
    files["weights/model.safetensors"] = files.pop("model.safetensors")
    config["members"] = [
        {"path": name, "size": len(raw), "kind": "git", "upstream_digest": _git(raw)}
        for name, raw in sorted(files.items())
    ]
    api, download, calls = _sdk(config, files)
    previous = os.umask(0o002)
    try:
        preparation._acquire(config, api_type=api, download=download)
    finally:
        os.umask(previous)
    assert calls == ["metadata", "download"]
    assert (root / "model-snapshot/weights/model.safetensors").read_bytes() == b"weights"
    assert (root / "model-snapshot/weights").stat().st_mode & 0o077 == 0


def test_model_metadata_substitution_rejects_before_download(candidate):
    config, files, root = candidate
    config["maximum_files"] = 4
    api, download, calls = _sdk(config, files, add_extra=True)
    with pytest.raises(ValueError, match="reviewed complete selection"):
        preparation._acquire(config, api_type=api, download=download)
    assert calls == ["metadata"]
    assert (root / "scratch").is_dir() and not (root / "model-snapshot").exists()


def test_metadata_candidate_is_review_only_and_create_once(candidate):
    config, files, root = candidate
    config["members"] = []
    config["tokenizer_members"] = []
    api, download, calls = _sdk(config, files)
    result = preparation._metadata_candidate(config, api_type=api)
    assert result["status"] == "REVIEW_REQUIRED_NO_AUTHORITY"
    assert result["member_count"] == 3 and calls == ["metadata"]
    stored = json.loads((root / "metadata-candidate.json").read_bytes())
    assert stored["members"] == [
        {"path": name, "size": len(raw), "kind": "git", "upstream_digest": _git(raw)}
        for name, raw in sorted(files.items())
    ]
    with pytest.raises(ValueError, match="already exists"):
        preparation._metadata_candidate(config, api_type=api)
    assert calls == ["metadata"] and not (root / "model-snapshot").exists()


def test_model_sdk_symlink_rejects_before_publication(candidate):
    config, files, root = candidate
    api, download, calls = _sdk(config, files, redirect="tokenizer.json")
    with pytest.raises(OSError):
        preparation._acquire(config, api_type=api, download=download)
    assert calls == ["metadata", "download"]
    assert not (root / "model-snapshot").exists()


@pytest.mark.parametrize("change", [
    lambda c: c.update(maximum_bytes=512 * 1024 * 1024 + 1),
    lambda c: c.update(tokenizer_members=["missing.json"]),
    lambda c: c["members"][0].update(path="../config.json"),
    lambda c: c["members"][1].update(size=1024),
])
def test_model_unreviewed_or_overbudget_selection_rejects_before_claim(candidate, change):
    config, files, root = candidate
    change(config)
    api, download, calls = _sdk(config, files)
    with pytest.raises(ValueError):
        preparation._acquire(config, api_type=api, download=download)
    assert calls == [] and not (root / "scratch").exists()
