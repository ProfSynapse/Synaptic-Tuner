#!/usr/bin/env python3
"""Private, bounded model snapshot preparation candidate; never grants training authority.

``inspect`` validates an independently reviewed inventory without network access.
``acquire`` is a later, separately authorized effect. It retains a consumed
workspace on any uncertain failure and never retries automatically.
``probe-transfer`` observes only a separately admitted HEAD and one-byte GET
chain in a fresh private record root; it never repairs an acquisition.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import ssl
import stat
import sys
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounded_command import run_bounded_command
from tuner.runtime.native_files import NativeTree, relative_member


_SCHEMA = "synaptic-native-model-acquisition/v1"
_ENDPOINT = "https://huggingface.co"
_MAX_BYTES = 512 * 1024 * 1024
_MAX_FILES = 128
_MAX_METADATA_RESPONSE = 2 * 1024 * 1024
_CHUNK = 64 * 1024
# The exact additional CDN origin was observed in a separately admitted
# one-byte probe and is listed in Hugging Face's model-download documentation.
_REDIRECT_HOSTS = frozenset({"huggingface.co", "cdn-lfs.huggingface.co",
                             "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co"})
_OFFICIAL_ORIGIN_CLASSES = {
    "huggingface.co": "hub",
    "cdn-lfs.huggingface.co": "legacy_lfs",
    "cas-bridge.xethub.hf.co": "xet_bridge",
    "cas-server.xethub.hf.co": "official_xet_cas_us",
    "cas-server.xethub-eu.hf.co": "official_xet_cas_eu",
    "transfer.xethub.hf.co": "official_xet_transfer_us",
    "transfer.xethub-eu.hf.co": "official_xet_transfer_eu",
    "us.aws.cdn.hf.co": "official_cdn_aws_us",
    "us.gcp.cdn.hf.co": "official_cdn_gcp_us",
    "cdn-lfs-us-1.hf.co": "official_lfs_us",
    "cdn-lfs-eu-1.hf.co": "official_lfs_eu",
}
_MEMBER_FIELDS = {"path", "size", "kind", "upstream_digest"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError("duplicate model preparation field")
        result[name] = value
    return result


def _read_config(path):
    path = Path(path)
    if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
        raise ValueError("model preparation config path differs")
    with NativeTree(path.parent, private=True) as parent:
        raw = parent.read(path.name, maximum=1 << 20)
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    return value


def _member(value):
    if type(value) is not dict or set(value) != _MEMBER_FIELDS:
        raise ValueError("model member fields differ")
    name = value["path"]
    relative_member(name)
    if (any(part in {".git", ".cache"} for part in name.split("/"))
            or unicodedata.normalize("NFC", name) != name
            or any(character in name for character in "*?[]")):
        raise ValueError("model member path is unsupported")
    size, kind, digest = value["size"], value["kind"], value["upstream_digest"]
    if (type(size) is not int or not 0 <= size <= _MAX_BYTES
            or kind not in ("git", "lfs") or type(kind) is not str
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{40}" if kind == "git" else r"[0-9a-f]{64}", digest) is None):
        raise ValueError("model member metadata is invalid")
    return value


def _validate_common(value):
    fields = {"schema", "repository", "revision", "workspace_root",
              "maximum_bytes", "maximum_files", "timeout_seconds", "members",
              "tokenizer_members"}
    if type(value) is not dict or set(value) != fields or value["schema"] != _SCHEMA:
        raise ValueError("model preparation configuration differs")
    repository, revision = value["repository"], value["revision"]
    if (type(repository) is not str
            or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
            or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ValueError("model repository pin is invalid")
    workspace = Path(value["workspace_root"])
    if (type(value["workspace_root"]) is not str or not workspace.is_absolute()
            or str(workspace) != value["workspace_root"]
            or str(workspace) != os.path.normpath(str(workspace))):
        raise ValueError("model private workspace is invalid")
    if (type(value["maximum_bytes"]) is not int or not 1 <= value["maximum_bytes"] <= _MAX_BYTES
            or type(value["maximum_files"]) is not int or not 1 <= value["maximum_files"] <= _MAX_FILES
            or type(value["timeout_seconds"]) is not int or not 1 <= value["timeout_seconds"] <= 300):
        raise ValueError("model acquisition bounds differ")
    with NativeTree(workspace, private=True):
        pass
    return workspace


def _validate(value):
    workspace = _validate_common(value)
    raw_members = value["members"]
    if (type(raw_members) is not list or not 1 <= len(raw_members) <= value["maximum_files"]):
        raise ValueError("model member count differs")
    members = tuple(_member(item) for item in raw_members)
    names = tuple(item["path"] for item in members)
    if (names != tuple(sorted(set(names)))
            or len({name.casefold() for name in names}) != len(names)
            or sum(item["size"] for item in members) > value["maximum_bytes"]
            or "config.json" not in names or not any(name.endswith(".safetensors") for name in names)):
        raise ValueError("model complete selection or bound differs")
    selected = value["tokenizer_members"]
    if (type(selected) is not list or not selected
            or selected != sorted(set(selected)) or not set(selected) <= set(names)):
        raise ValueError("model tokenizer selection differs")
    with NativeTree(workspace, private=True) as root:
        for name in ("scratch", "model-snapshot", "tokenizer-snapshot",
                     "model-snapshot.json", "tokenizer-snapshot.json", "acquisition-result.json"):
            try:
                os.stat(name, dir_fd=root.fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError("model workspace has a prior acquisition claim")
    return members


def _probe_selection(value, *, require_fresh):
    """Bind a diagnostic to reviewed acquisition bytes, not its consumed workspace."""
    if (type(value) is not dict or set(value) != {
            "schema", "source_config", "source_sha256", "member_path", "record_root",
            "timeout_seconds", "maximum_redirects", "get_body_bytes", "range_requested",
    } or value["schema"] != "synaptic-native-model-transfer-probe/v1"
            or type(value["source_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]) is None
            or value["timeout_seconds"] != 120 or type(value["timeout_seconds"]) is not int
            or value["maximum_redirects"] != 3 or type(value["maximum_redirects"]) is not int
            or value["get_body_bytes"] != 1 or type(value["get_body_bytes"]) is not int
            or value["range_requested"] is not False):
        raise ValueError("model transfer probe selection differs")
    source_path = value["source_config"]
    if (type(source_path) is not str or not Path(source_path).is_absolute()
            or str(Path(source_path)) != os.path.normpath(source_path)):
        raise ValueError("model transfer source path differs")
    with NativeTree(Path(source_path).parent, private=True) as parent:
        source_raw = parent.read(Path(source_path).name, maximum=1 << 20)
    if hashlib.sha256(source_raw).hexdigest() != value["source_sha256"]:
        raise ValueError("model transfer source bytes differ")
    source = json.loads(source_raw, object_pairs_hook=_pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    _validate_common(source)
    member_path = value["member_path"]
    if type(member_path) is not str:
        raise ValueError("model transfer member differs")
    relative_member(member_path)
    selected = [_member(item) for item in source["members"] if type(item) is dict
                and item.get("path") == member_path]
    if len(selected) != 1:
        raise ValueError("model transfer member differs from source")
    record_root = value["record_root"]
    if (type(record_root) is not str or not Path(record_root).is_absolute()
            or str(Path(record_root)) != os.path.normpath(record_root)):
        raise ValueError("model transfer record root differs")
    with NativeTree(Path(record_root), private=True) as root:
        if require_fresh and root.members():
            raise ValueError("model transfer probe slot is consumed")
    return source, selected[0]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _origin_class(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or parsed.port not in (None, 443) or parsed.fragment):
            return "invalid_origin"
        return _OFFICIAL_ORIGIN_CLASSES.get(parsed.hostname, "unrecognized_origin")
    except ValueError:
        return "invalid_origin"


def _probe_chain(url, method, *, expected_size, opener=None):
    """Observe one method chain; never follow a URL outside acquisition policy."""
    if opener is None:
        opener = build_opener(ProxyHandler({}), _NoRedirect()).open
    hops = []
    for redirect_count in range(4):
        if _origin_class(url) == "invalid_origin" or urlsplit(url).hostname not in _REDIRECT_HOSTS:
            return {"method": method, "outcome": "destination_rejected", "hops": hops}
        request = Request(url, method=method,
                          headers={"Accept-Encoding": "identity",
                                   "User-Agent": "synaptic-native-model-transfer-probe/1"})
        try:
            response = opener(request, timeout=20)
        except HTTPError as error:
            response = error
        except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError):
            return {"method": method, "outcome": "transport_error", "hops": hops}
        with response:
            status = response.status
            entry = {"origin_class": _origin_class(url),
                     "status": status if type(status) is int and 100 <= status <= 599 else 0}
            headers = response.headers
            if status in (301, 302, 303, 307, 308):
                location = headers.get("Location")
                target = urljoin(url, location) if location else ""
                entry["target_origin_class"] = _origin_class(target)
                entry["target_allowed"] = (entry["target_origin_class"] != "invalid_origin"
                                           and urlsplit(target).hostname in _REDIRECT_HOSTS)
                hops.append(entry)
                if not entry["target_allowed"]:
                    return {"method": method, "outcome": "redirect_destination_rejected",
                            "hops": hops}
                if redirect_count == 3:
                    return {"method": method, "outcome": "redirect_limit", "hops": hops}
                url = target
                continue
            declared = headers.get("Content-Length")
            entry["content_length"] = (int(declared) if declared is not None
                                       and declared.isdecimal() and len(declared) <= 12 else None)
            entry["declared_size_matches"] = entry["content_length"] == expected_size
            entry["encoding_identity"] = headers.get("Content-Encoding", "identity") == "identity"
            if method == "GET" and status == 200:
                try:
                    entry["body_bytes_read"] = len(response.read(1))
                except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError):
                    hops.append(entry)
                    return {"method": method, "outcome": "body_read_error", "hops": hops}
            hops.append(entry)
            return {"method": method, "outcome": "response_observed", "hops": hops}
    raise AssertionError("unreachable redirect bound")


def _probe_transfer(config, *, opener=None):
    source, member = _probe_selection(config, require_fresh=True)
    root_path = Path(config["record_root"])
    selection = hashlib.sha256(_canonical(config)).hexdigest()
    claim = {"schema": "synaptic-native-model-transfer-probe-claim/v1",
             "selection_sha256": selection, "status": "CONSUMED"}
    with NativeTree(root_path, private=True) as root:
        root.write_exclusive("probe-claim.json", _canonical(claim))
    url = (f"{_ENDPOINT}/{source['repository']}/resolve/{source['revision']}/"
           f"{quote(member['path'], safe='/')}")
    observations = [_probe_chain(url, method, expected_size=member["size"], opener=opener)
                    for method in ("HEAD", "GET")]
    result = {"schema": "synaptic-native-model-transfer-probe-result/v1",
              "status": "OBSERVED_NOT_ADMITTED", "selection_sha256": selection,
              "source_sha256": config["source_sha256"], "member_path": member["path"],
              "member_declared_bytes": member["size"], "range_requested": False,
              "observations": observations}
    raw = _canonical(result)
    if len(raw) > 4096:
        raise ValueError("model transfer probe result exceeds bound")
    with NativeTree(root_path, private=True) as root:
        root.write_exclusive("probe-result.json", raw)
    return result


def _upstream(member, raw):
    if len(raw) != member["size"]:
        raise ValueError("model member byte count differs")
    if member["kind"] == "lfs":
        digest = hashlib.sha256(raw).hexdigest()
    else:
        digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if digest != member["upstream_digest"]:
        raise ValueError("model member upstream digest differs")
    return hashlib.sha256(raw).hexdigest()


def _url_host(url, allowed):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in allowed
            or parsed.port not in (None, 443) or parsed.username or parsed.password
            or parsed.fragment):
        raise ValueError("model HTTPS destination differs")


class _RestrictedRedirect(HTTPRedirectHandler):
    def __init__(self, hosts, maximum):
        self.hosts, self.maximum, self.count = hosts, maximum, 0

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        self.count += 1
        if self.count > self.maximum:
            raise ValueError("model redirect count differs")
        _url_host(newurl, self.hosts)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


@contextmanager
def _response(url, *, hosts, redirects, opener=None):
    _url_host(url, hosts)
    request = Request(url, headers={"Accept-Encoding": "identity",
                                    "User-Agent": "synaptic-native-model-preparation/1"})
    if opener is None:
        opener = build_opener(ProxyHandler({}), _RestrictedRedirect(hosts, redirects)).open
    with opener(request, timeout=20) as response:
        _url_host(response.geturl(), hosts)
        if (response.status != 200 or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("model HTTP response differs")
        yield response


def _bounded_metadata(config, *, opener=None):
    url = (f"{_ENDPOINT}/api/models/{config['repository']}/revision/"
           f"{config['revision']}?blobs=true")
    with _response(url, hosts=frozenset({"huggingface.co"}), redirects=0,
                   opener=opener) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and (not declared.isdecimal() or int(declared) > _MAX_METADATA_RESPONSE):
            raise ValueError("model metadata response exceeds bound")
        chunks, count = [], 0
        while True:
            chunk = response.read(min(_CHUNK, _MAX_METADATA_RESPONSE + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > _MAX_METADATA_RESPONSE:
                raise ValueError("model metadata response exceeds bound")
            chunks.append(chunk)
    raw = b"".join(chunks)
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    if type(value) is not dict:
        raise ValueError("model metadata object differs")
    from huggingface_hub.hf_api import ModelInfo
    return ModelInfo(**value)


def _observe_metadata(config, *, api_type):
    if api_type is None:
        info = _bounded_metadata(config)
    else:
        info = api_type(endpoint=_ENDPOINT, token=False).model_info(
            config["repository"], revision=config["revision"], files_metadata=True,
            token=False, timeout=30,
        )
    if info.sha != config["revision"] or type(info.siblings) not in (list, tuple):
        raise ValueError("model metadata revision differs")
    observed = []
    for sibling in info.siblings:
        kind = "lfs" if sibling.lfs is not None else "git"
        digest = sibling.lfs.sha256 if kind == "lfs" else sibling.blob_id
        observed.append(_member({"path": sibling.rfilename, "size": sibling.size,
                                 "kind": kind, "upstream_digest": digest}))
        if len(observed) > config["maximum_files"]:
            raise ValueError("model metadata count exceeds bound")
    names = tuple(item["path"] for item in observed)
    if (len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names)
            or sum(item["size"] for item in observed) > config["maximum_bytes"]):
        raise ValueError("model metadata exceeds exact selection bounds")
    return tuple(sorted(observed, key=lambda item: item["path"]))


def _metadata(members, config, *, api_type):
    observed = _observe_metadata(config, api_type=api_type)
    if observed != members:
        raise ValueError("model metadata differs from reviewed complete selection")


def _metadata_candidate(config, *, api_type=None):
    workspace = _validate_common(config)
    if config["members"] != [] or config["tokenizer_members"] != []:
        raise ValueError("metadata candidate must not contain reviewed members")
    with NativeTree(workspace, private=True) as root:
        try:
            os.stat("metadata-candidate.json", dir_fd=root.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("metadata candidate already exists")
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        observed = _observe_metadata(config, api_type=api_type)
    names = tuple(item["path"] for item in observed)
    if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError("model metadata is incomplete")
    candidate = {"schema": "synaptic-native-model-metadata-candidate/v1",
                 "status": "REVIEW_REQUIRED_NO_AUTHORITY", "repository": config["repository"],
                 "revision": config["revision"], "members": list(observed),
                 "declared_bytes": sum(item["size"] for item in observed),
                 "member_count": len(observed)}
    with NativeTree(workspace, private=True) as root:
        root.write_exclusive("metadata-candidate.json", _canonical(candidate))
    return {"status": candidate["status"], "repository": config["repository"],
            "revision": config["revision"], "member_count": candidate["member_count"],
            "declared_bytes": candidate["declared_bytes"],
            "metadata_candidate_sha256": hashlib.sha256(_canonical(candidate)).hexdigest()}


def _publish_member(root, name, raw):
    parts = relative_member(name)
    with ExitStack() as stack:
        parent = root.fd
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("model destination parent is not private")
            parent = fd
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    root.check()


@contextmanager
def _member_parent(root, name):
    parts = relative_member(name)
    with ExitStack() as stack:
        parent = root.fd
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("model SDK destination parent is not private")
            parent = fd
        root.check()
        yield parent, parts[-1]
        root.check()


def _fetch_member(config, member, root, *, remaining, opener=None):
    """One no-retry, no-cache HTTPS transfer; stop before any excess byte is written."""
    if remaining < member["size"]:
        raise ValueError("model aggregate transfer exceeds bound")
    name = member["path"]
    expected_url = (f"{_ENDPOINT}/{config['repository']}/resolve/{config['revision']}/"
                    f"{quote(name, safe='/')}")
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(config["repository"], name, revision=config["revision"], endpoint=_ENDPOINT)
    if url != expected_url:
        raise ValueError("model Hub resolve URL differs")
    with _member_parent(root, name) as (parent, leaf):
        fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        size = 0
        digest = hashlib.sha256() if member["kind"] == "lfs" else hashlib.sha1()
        if member["kind"] == "git":
            digest.update(f"blob {member['size']}\0".encode())
        with os.fdopen(fd, "wb") as output:
            with _response(url, hosts=_REDIRECT_HOSTS, redirects=2, opener=opener) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None and (not declared.isdecimal() or int(declared) != member["size"]):
                    raise ValueError("model transfer declared size differs")
                while True:
                    chunk = response.read(min(_CHUNK, member["size"] + 1 - size))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > member["size"] or size > remaining:
                        raise ValueError("model transfer exceeds admitted bytes")
                    output.write(chunk)
                    digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
    if size != member["size"] or digest.hexdigest() != member["upstream_digest"]:
        raise ValueError("model transfer identity differs")
    return size


def _acquire(config, *, api_type=None, download=None):
    previous = os.umask(0o077)
    try:
        return _acquire_private(config, api_type=api_type, download=download)
    finally:
        os.umask(previous)


def _acquire_private(config, *, api_type=None, download=None):
    members = _validate(config)
    workspace = Path(config["workspace_root"])
    with NativeTree(workspace, private=True) as root:
        os.mkdir("scratch", mode=0o700, dir_fd=root.fd)
        root.check()
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        _metadata(members, config, api_type=api_type)
    scratch = workspace / "scratch"
    with NativeTree(scratch, private=True) as held:
        os.mkdir("repository", mode=0o700, dir_fd=held.fd)
        os.mkdir("sdk-cache", mode=0o700, dir_fd=held.fd)
        held.check()
    repository = scratch / "repository"
    with NativeTree(repository, private=True) as source, NativeTree(scratch / "sdk-cache", private=True) as cache:
        # Every selected nested destination exists at 0700 before a transport can write.
        for member in members:
            with _member_parent(source, member["path"]):
                pass
        cache.check()
        if download is None:
            transferred = 0
            with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
                for member in members:
                    transferred += _fetch_member(config, member, source,
                                                 remaining=config["maximum_bytes"] - transferred)
        else:
            # Injection is used by CPU publication fixtures only. The public CLI
            # always uses the bounded transfer above.
            with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
                result = download(
                    repo_id=config["repository"], revision=config["revision"], token=False,
                    endpoint=_ENDPOINT, local_dir=repository, cache_dir=scratch / "sdk-cache",
                    allow_patterns=[item["path"] for item in members], max_workers=1,
                    etag_timeout=10,
                )
            if Path(result) != repository:
                raise ValueError("model SDK destination differs")
        source.check()
        cache.check()
    bytes_by_name = {}
    content_hashes = {}
    with NativeTree(repository, private=True) as source:
        for member in members:
            raw = source.read(member["path"], maximum=member["size"])
            content_hashes[member["path"]] = _upstream(member, raw)
            bytes_by_name[member["path"]] = raw
        source.check()
    with NativeTree(workspace, private=True) as root:
        os.mkdir("model-snapshot", mode=0o700, dir_fd=root.fd)
        os.mkdir("tokenizer-snapshot", mode=0o700, dir_fd=root.fd)
        root.check()
    manifests = []
    for role, names in (("model", tuple(item["path"] for item in members)),
                        ("tokenizer", tuple(config["tokenizer_members"]))):
        target = workspace / f"{role}-snapshot"
        with NativeTree(target, private=True) as published:
            for name in names:
                _publish_member(published, name, bytes_by_name[name])
            inventory = published.inventory(maximum_total=config["maximum_bytes"])
            if (tuple(item.path for item in inventory) != names
                    or any(item.sha256 != content_hashes[item.path] for item in inventory)):
                raise ValueError("model publication readback differs")
        manifest = {"repository": config["repository"], "revision": config["revision"],
                    "root": str(target),
                    "files": [{"path": item.path, "size": item.size, "sha256": item.sha256}
                              for item in inventory]}
        manifests.append((role, manifest))
    receipt = {"schema": "synaptic-native-model-acquisition-observation/v1",
               "status": "OBSERVED_NOT_ADMITTED", "repository": config["repository"],
               "revision": config["revision"], "selection_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
               "model_manifest_sha256": hashlib.sha256(_canonical(manifests[0][1])).hexdigest(),
               "tokenizer_manifest_sha256": hashlib.sha256(_canonical(manifests[1][1])).hexdigest()}
    with NativeTree(workspace, private=True) as root:
        for role, manifest in manifests:
            root.write_exclusive(f"{role}-snapshot.json", _canonical(manifest))
        root.write_exclusive("acquisition-result.json", _canonical(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "metadata-candidate", "acquire",
                                          "probe-transfer", "_probe_worker",
                                          "_metadata_worker", "_worker"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = _read_config(args.config)
        if args.action in {"probe-transfer", "_probe_worker"}:
            _probe_selection(config, require_fresh=True)
        elif args.action in {"metadata-candidate", "_metadata_worker"}:
            _validate_common(config)
        else:
            _validate(config)
        if args.action == "inspect":
            result = {"status": "REVIEW_REQUIRED_NO_NETWORK",
                      "selection_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
                      "member_count": len(config["members"]),
                      "declared_bytes": sum(item["size"] for item in config["members"])}
        elif args.action == "_probe_worker":
            result = _probe_transfer(config)
        elif args.action == "_metadata_worker":
            result = _metadata_candidate(config)
        elif args.action == "_worker":
            result = _acquire(config)
        else:
            workspace = (config["record_root"] if args.action == "probe-transfer"
                         else config["workspace_root"])
            environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C",
                           "HF_HOME": workspace,
                           "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                           "XDG_CACHE_HOME": workspace, "TMPDIR": workspace}
            status, output = run_bounded_command(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                 ("_probe_worker" if args.action == "probe-transfer" else
                  "_metadata_worker" if args.action == "metadata-candidate" else "_worker"),
                 str(args.config)], environment=environment,
                maximum=4096, timeout=config["timeout_seconds"],
            )
            if status != 0:
                raise ValueError("model acquisition worker uncertain")
            if not output.endswith(b"\n") or output.endswith(b"\n\n"):
                raise ValueError("model acquisition worker framing differs")
            worker_result = json.loads(output[:-1], object_pairs_hook=_pairs)
            if _canonical(worker_result) + b"\n" != output:
                raise ValueError("model acquisition worker result differs")
            if args.action == "probe-transfer":
                with NativeTree(Path(config["record_root"]), private=True) as root:
                    result = json.loads(root.read("probe-result.json", maximum=4096),
                                        object_pairs_hook=_pairs)
                if (result != worker_result or result.get("status") != "OBSERVED_NOT_ADMITTED"
                        or result.get("selection_sha256") != hashlib.sha256(_canonical(config)).hexdigest()):
                    raise ValueError("model transfer probe result differs")
            elif args.action == "metadata-candidate":
                with NativeTree(Path(config["workspace_root"]), private=True) as root:
                    candidate = json.loads(root.read("metadata-candidate.json", maximum=1 << 20))
                result = {"status": candidate["status"], "repository": config["repository"],
                          "revision": config["revision"], "member_count": candidate["member_count"],
                          "declared_bytes": candidate["declared_bytes"],
                          "metadata_candidate_sha256": hashlib.sha256(_canonical(candidate)).hexdigest()}
                if result != worker_result or result["status"] != "REVIEW_REQUIRED_NO_AUTHORITY":
                    raise ValueError("model metadata candidate differs")
            else:
                with NativeTree(Path(config["workspace_root"]), private=True) as root:
                    result = json.loads(root.read("acquisition-result.json", maximum=1 << 20))
                if (result != worker_result or result.get("status") != "OBSERVED_NOT_ADMITTED"
                        or result.get("selection_sha256") != hashlib.sha256(_canonical(config)).hexdigest()):
                    raise ValueError("model acquisition result differs")
        print(_canonical(result).decode())
        return 0
    except Exception:
        print('{"diagnostic_code":"native_model_preparation_refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
