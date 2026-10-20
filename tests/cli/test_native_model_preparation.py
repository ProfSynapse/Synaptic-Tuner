"""Private native model preparation contracts; no SDK network or ML imports."""

from contextlib import contextmanager
from hashlib import sha1, sha256
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
from urllib.request import Request

import pytest

from examples.training_qualification import native_model_preparation as preparation
from examples.training_qualification.bounded_command import run_bounded_command


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
        cache = Path(options["cache_dir"])
        assert cache.is_dir() and cache.stat().st_mode & 0o077 == 0
        for name, raw in files.items():
            target = root / name
            assert target.parent.is_dir()
            assert target.parent.stat().st_mode & 0o077 == 0
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


@pytest.mark.parametrize("ambient", [0o022, 0o002])
def test_nested_model_publication_stays_private_under_ambient_group_umask(candidate, ambient):
    config, files, root = candidate
    files["weights/model.safetensors"] = files.pop("model.safetensors")
    config["members"] = [
        {"path": name, "size": len(raw), "kind": "git", "upstream_digest": _git(raw)}
        for name, raw in sorted(files.items())
    ]
    api, download, calls = _sdk(config, files)
    previous = os.umask(ambient)
    try:
        preparation._acquire(config, api_type=api, download=download)
    finally:
        os.umask(previous)
    assert calls == ["metadata", "download"]
    assert (root / "model-snapshot/weights/model.safetensors").read_bytes() == b"weights"
    assert (root / "model-snapshot/weights").stat().st_mode & 0o077 == 0


class _Response(io.BytesIO):
    status = 200

    def __init__(self, body, url, *, headers=None):
        super().__init__(body)
        self.url = url
        self.headers = headers or {}

    def geturl(self):
        return self.url


def test_metadata_body_limit_rejects_before_json_parse(candidate):
    config, _, _ = candidate
    seen = []
    oversized = preparation._canonical({
        "id": config["repository"], "sha": config["revision"], "siblings": [],
        "cardData": "x" * preparation._MAX_METADATA_RESPONSE,
    })
    assert len(oversized) > preparation._MAX_METADATA_RESPONSE

    def open_url(request, timeout):
        seen.append((request.full_url, timeout))
        return _Response(oversized, request.full_url)

    with pytest.raises(ValueError, match="response exceeds bound"):
        preparation._bounded_metadata(config, opener=open_url)
    assert len(seen) == 1 and seen[0][0].endswith("?blobs=true")


def test_bounded_hub_metadata_is_parsed_after_complete_stream(candidate):
    config, files, _ = candidate
    body = preparation._canonical({
        "id": config["repository"], "sha": config["revision"],
        "siblings": [{"rfilename": name, "size": len(raw), "blobId": _git(raw)}
                     for name, raw in sorted(files.items())],
    })

    def open_url(request, timeout):
        assert request.get_header("Accept-encoding") == "identity"
        return _Response(body, request.full_url, headers={"Content-Length": str(len(body))})

    info = preparation._bounded_metadata(config, opener=open_url)
    assert info.sha == config["revision"]
    assert [sibling.rfilename for sibling in info.siblings] == sorted(files)


def test_model_transfer_stops_before_overflow_write_and_retains_claim(candidate):
    config, files, workspace = candidate
    member = next(item for item in config["members"] if item["path"] == "model.safetensors")
    repository = workspace / "repository"
    repository.mkdir(mode=0o700)

    def open_url(request, timeout):
        return _Response(files["model.safetensors"] + b"excess", request.full_url)

    with preparation.NativeTree(repository, private=True) as root:
        with pytest.raises(ValueError, match="exceeds admitted bytes"):
            preparation._fetch_member(config, member, root, remaining=512, opener=open_url)
    assert (repository / "model.safetensors").read_bytes() == b""
    assert not (workspace / "model-snapshot").exists()


def test_observed_exact_cdn_is_allowed_for_bounded_member_bytes(candidate):
    config, files, workspace = candidate
    member = next(item for item in config["members"] if item["path"] == "model.safetensors")
    repository = workspace / "repository"
    repository.mkdir(mode=0o700)
    cdn = "https://us.aws.cdn.hf.co/opaque?signature=PRIVATE"

    def open_url(request, timeout):
        assert request.full_url.startswith("https://huggingface.co/")
        assert timeout == 20
        return _Response(files["model.safetensors"], cdn,
                         headers={"Content-Length": str(member["size"])})

    with preparation.NativeTree(repository, private=True) as root:
        assert preparation._fetch_member(config, member, root, remaining=512,
                                         opener=open_url) == member["size"]
    assert (repository / "model.safetensors").read_bytes() == files["model.safetensors"]
    initial = Request("https://huggingface.co/reviewed/model/resolve/" + "a" * 40 + "/model.safetensors")
    handler = preparation._RestrictedRedirect(preparation._REDIRECT_HOSTS, 2)
    assert handler.redirect_request(initial, None, 302, "", {}, cdn).full_url == cdn
    assert preparation._origin_class(cdn) == "official_cdn_aws_us"
    preparation._url_host(cdn, preparation._REDIRECT_HOSTS)
    for rejected in ("https://us.aws.cdn.hf.co.evil.test/file",
                     "http://us.aws.cdn.hf.co/file",
                     "https://@us.aws.cdn.hf.co/file",
                     "https://:@us.aws.cdn.hf.co/file",
                     "https://user@us.aws.cdn.hf.co/file",
                     "https://user:pass@us.aws.cdn.hf.co/file"):
        with pytest.raises(ValueError, match="HTTPS destination"):
            preparation._url_host(rejected, preparation._REDIRECT_HOSTS)
        assert preparation._origin_class(rejected) != "official_cdn_aws_us"
    with pytest.raises(ValueError, match="HTTPS destination"):
        handler.redirect_request(initial, None, 302, "", {}, "https://unreviewed.example/file")


def _probe_config(candidate):
    source, _, workspace = candidate
    source_path = workspace / "config.json"
    record = workspace / "transport-probe"
    record.mkdir(mode=0o700)
    return {"schema": "synaptic-native-model-transfer-probe/v1",
            "source_config": str(source_path),
            "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
            "member_path": "model.safetensors", "record_root": str(record),
            "timeout_seconds": 120, "maximum_redirects": 3,
            "get_body_bytes": 1, "range_requested": False}


def test_transfer_probe_caps_body_and_preserves_consumed_acquisition(candidate):
    config = _probe_config(candidate)
    root = Path(config["record_root"])
    old = Path(candidate[0]["workspace_root"])
    (old / "scratch").mkdir(mode=0o700)
    (old / "scratch" / "original").write_bytes(b"retained")
    calls = []

    class Limited(_Response):
        def read(self, size=-1):
            assert size == 1
            return super().read(size)

    def open_url(request, timeout):
        calls.append((request.get_method(), request.get_header("Range"), timeout))
        return Limited(b"large body", request.full_url, headers={"Content-Length": "335450584"})

    result = preparation._probe_transfer(config, opener=open_url)
    assert calls == [("HEAD", None, 20), ("GET", None, 20)]
    assert result["range_requested"] is False
    assert result["observations"][1]["hops"][0]["body_bytes_read"] == 1
    assert (old / "scratch" / "original").read_bytes() == b"retained"
    assert (root / "probe-result.json").read_bytes() == preparation._canonical(result)
    with pytest.raises(ValueError, match="consumed"):
        preparation._probe_transfer(config, opener=open_url)
    assert len(calls) == 2


def test_transfer_probe_unknown_redirect_stops_before_target_and_never_leaks_url(candidate):
    config = _probe_config(candidate)
    calls = []

    class Redirect(io.BytesIO):
        status = 302

        def __init__(self, url):
            super().__init__(b"")
            self.url = url
            self.headers = {"Location": "https://unknown.example/private?token=SECRET"}

        def geturl(self):
            return self.url

    def redirect(request, timeout):
        calls.append(request.full_url)
        return Redirect(request.full_url)

    result = preparation._probe_transfer(config, opener=redirect)
    assert len(calls) == 2
    assert all(item["outcome"] == "redirect_destination_rejected"
               for item in result["observations"])
    assert all(item["hops"][0]["target_origin_class"] == "unrecognized_origin"
               for item in result["observations"])
    assert b"SECRET" not in (Path(config["record_root"]) / "probe-result.json").read_bytes()


def test_transfer_probe_binds_source_bytes_and_retains_claim_on_transport_error(candidate):
    config = _probe_config(candidate)
    source_path = Path(config["source_config"])
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source bytes"):
        preparation._probe_transfer(config)
    assert not (Path(config["record_root"]) / "probe-claim.json").exists()
    config["source_sha256"] = sha256(source_path.read_bytes()).hexdigest()

    def fail(request, timeout):
        raise OSError("PRIVATE_URL_TOKEN")

    result = preparation._probe_transfer(config, opener=fail)
    assert [item["outcome"] for item in result["observations"]] == ["transport_error"] * 2
    assert b"PRIVATE_URL_TOKEN" not in (Path(config["record_root"]) / "probe-result.json").read_bytes()


def test_interrupted_transfer_retains_scratch_and_rejects_replay(candidate):
    config, files, root = candidate
    api, _, calls = _sdk(config, files)

    def interrupted(**options):
        calls.append("download")
        (Path(options["local_dir"]) / "config.json").write_bytes(b"{")
        raise OSError("untrusted network detail")

    with pytest.raises(OSError):
        preparation._acquire(config, api_type=api, download=interrupted)
    assert calls == ["metadata", "download"]
    assert (root / "scratch/repository/config.json").read_bytes() == b"{"
    assert not (root / "model-snapshot").exists()
    with pytest.raises(ValueError, match="prior acquisition claim"):
        preparation._acquire(config, api_type=api, download=interrupted)
    assert calls == ["metadata", "download"]


def test_real_bounded_transfer_overflow_retains_claim_without_publication(candidate, monkeypatch):
    config, files, root = candidate
    api, _, calls = _sdk(config, files)

    @contextmanager
    def overflow(request_url, **_options):
        with _Response(b"{}x", request_url) as response:
            yield response

    monkeypatch.setattr(preparation, "_response", overflow)
    with pytest.raises(ValueError, match="exceeds admitted bytes"):
        preparation._acquire(config, api_type=api)
    assert calls == ["metadata"]
    assert (root / "scratch/repository/config.json").read_bytes() == b""
    assert not (root / "model-snapshot").exists()
    with pytest.raises(ValueError, match="prior acquisition claim"):
        preparation._acquire(config, api_type=api)
    assert calls == ["metadata"]


def test_real_child_newline_is_accepted_by_parent_readback(candidate, monkeypatch, capsys):
    config, files, root = candidate
    config["members"] = []
    config["tokenizer_members"] = []
    (root / "config.json").write_bytes(preparation._canonical(config))
    api, _, _ = _sdk(config, files)
    result = preparation._metadata_candidate(config, api_type=api)

    def real_child(_argv, **_kwargs):
        body = preparation._canonical(result).decode()
        return run_bounded_command([sys.executable, "-c", "print(" + repr(body) + ")"],
                                   environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                                   maximum=4096, timeout=5)

    monkeypatch.setattr(preparation, "run_bounded_command", real_child)
    assert preparation.main(["metadata-candidate", str(root / "config.json")]) == 0
    assert json.loads(capsys.readouterr().out) == result


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
