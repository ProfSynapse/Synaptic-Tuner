"""Hugging Face read-only transport for the verified bootstrap capsule.

This module transports bytes only.  It deliberately knows nothing about Git,
source-lock semantics, credentials, or repository reconstruction.  Those
responsibilities remain in the J0 capsule and ``bootstrap_core``.
"""

from __future__ import annotations

import inspect
import json
import posixpath
import re
import shlex
import hashlib
import os
import stat
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from tuner.core.exceptions import CloudProviderError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_MOUNT_ROOT = "/workspace/synaptic-bootstrap-input"
_CHECKOUT_ROOT = "/workspace/source"
_BOOTSTRAP_RESULT = ".synaptic-bootstrap-result.json"


@dataclass(frozen=True)
class HFVerifiedVolumeSpec:
    """Pre-provisioned, digest-bound regular members mounted read-only.

    The provider workflow must place the capsule and the two per-run JSON
    documents in ``source``/``path`` before launch.  This class performs no
    upload or synchronization and never accepts either JSON document in an
    environment variable. ``local_root`` is an operator-provisioned mirror of
    the exact mounted directory, not the launcher's staging directory.
    """

    source: str
    capsule_path: str
    capsule_manifest_sha256: str
    source_lock_path: str
    source_lock_sha256: str
    checkout_policy_path: str
    checkout_policy_sha256: str
    local_root: Path
    path: str | None = None
    mount_path: str = _MOUNT_ROOT

    def __post_init__(self) -> None:
        if not _BUCKET_RE.fullmatch(self.source):
            raise CloudProviderError("HF bootstrap volume requires a sanitized namespaced bucket id.")
        if self.mount_path != _MOUNT_ROOT:
            raise CloudProviderError("HF bootstrap volume mount path is fixed by the verified transport contract.")
        for label, value in (
            ("capsule path", self.capsule_path),
            ("source-lock path", self.source_lock_path),
            ("checkout-policy path", self.checkout_policy_path),
        ):
            _validate_relative_member(value, label=label)
        if len({self.capsule_path, self.source_lock_path, self.checkout_policy_path}) != 3:
            raise CloudProviderError("HF bootstrap volume members must be distinct.")
        if self.path is not None:
            _validate_relative_member(self.path, label="volume subpath")
        for label, value in (
            ("capsule manifest", self.capsule_manifest_sha256),
            ("source lock", self.source_lock_sha256),
            ("checkout policy", self.checkout_policy_sha256),
        ):
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise CloudProviderError(f"HF bootstrap {label} digest must be lowercase SHA-256.")
        if not isinstance(self.local_root, Path) or not self.local_root.is_absolute():
            raise CloudProviderError("HF bootstrap local proof root must be an absolute path.")

    def mounted(self, member: str) -> str:
        return posixpath.join(self.mount_path, member)


@dataclass(frozen=True)
class HFVerifiedVolume:
    """A locally proven volume plus the closed inputs needed to revalidate it.

    Construction of this public value is never authorization.  The submission
    boundary independently reloads the descriptor and bundle, revalidates the
    evidence binding, and checks the provider wire object before failing at the
    currently-unimplemented exact-run approval gate.
    """

    spec: HFVerifiedVolumeSpec
    provider_volume: Any
    descriptor_sha256: str | None = None
    provisioning_evidence_sha256: str | None = None
    descriptor_uri: str | None = None
    source_lock_uri: str | None = None
    transport_root: Path | None = None
    provisioning_evidence: Mapping[str, object] | None = None
    verification_context: Any = None


def _validate_relative_member(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CloudProviderError(f"HF bootstrap {label} must be a canonical relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CloudProviderError(f"HF bootstrap {label} must be a contained relative path.")
    if path.as_posix() != value:
        raise CloudProviderError(f"HF bootstrap {label} must be canonically encoded.")


def _explicit_keyword(signature: inspect.Signature, name: str) -> bool:
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _read_provisioned_regular(path: Path) -> bytes:
    current = Path(path)
    if not current.is_absolute():
        raise CloudProviderError("HF bootstrap proof path must be absolute.")
    chain: list[Path] = []
    cursor = current
    while cursor != cursor.parent:
        chain.append(cursor)
        cursor = cursor.parent
    for component in reversed(chain):
        try:
            info = component.lstat()
        except OSError as exc:
            raise CloudProviderError("HF bootstrap pre-provisioned member is absent.") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CloudProviderError("HF bootstrap pre-provisioned members cannot traverse links or reparse points.")
    try:
        info = current.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
            raise CloudProviderError("HF bootstrap pre-provisioned member must be a bounded regular file.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(current, flags)
        try:
            chunks: list[bytes] = []
            remaining = 4 * 1024 * 1024 + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            opened = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except CloudProviderError:
        raise
    except OSError as exc:
        raise CloudProviderError("HF bootstrap pre-provisioned member could not be authenticated.") from exc
    if len(content) > 4 * 1024 * 1024 or not stat.S_ISREG(opened.st_mode):
        raise CloudProviderError("HF bootstrap pre-provisioned member exceeds its bound.")
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
    ):
        raise CloudProviderError("HF bootstrap pre-provisioned member changed during authentication.")
    return content


def prove_preprovisioned_members(spec: HFVerifiedVolumeSpec) -> None:
    """Authenticate the local mirror of every required mounted member."""

    members = (
        (Path(spec.capsule_path) / "synaptic-bootstrap-capsule.json", spec.capsule_manifest_sha256),
        (Path(spec.source_lock_path), spec.source_lock_sha256),
        (Path(spec.checkout_policy_path), spec.checkout_policy_sha256),
    )
    for relative, expected in members:
        candidate = spec.local_root.joinpath(*relative.parts)
        try:
            candidate.relative_to(spec.local_root)
        except ValueError as exc:
            raise CloudProviderError("HF bootstrap member escapes its local proof root.") from exc
        content = _read_provisioned_regular(candidate)
        if hashlib.sha256(content).hexdigest() != expected:
            raise CloudProviderError("HF bootstrap pre-provisioned member digest mismatch.")


def prove_read_only_volume(huggingface_hub: Any, spec: HFVerifiedVolumeSpec) -> HFVerifiedVolume:
    """Prove installed-client Volume semantics without invoking ``run_job``."""

    prove_preprovisioned_members(spec)
    volume_type = getattr(huggingface_hub, "Volume", None)
    run_job = getattr(huggingface_hub, "run_job", None)
    if not callable(volume_type) or not callable(run_job):
        raise CloudProviderError("Installed huggingface_hub lacks verified Jobs volume support.")
    try:
        run_signature = inspect.signature(run_job)
    except (TypeError, ValueError) as exc:
        raise CloudProviderError("Installed huggingface_hub Jobs signature cannot be verified.") from exc
    if not _explicit_keyword(run_signature, "volumes"):
        raise CloudProviderError("Installed huggingface_hub run_job does not explicitly support volumes.")
    kwargs: dict[str, object] = {
        "type": "bucket",
        "source": spec.source,
        "mount_path": spec.mount_path,
        "read_only": True,
    }
    if spec.path is not None:
        kwargs["path"] = spec.path
    try:
        volume = volume_type(**kwargs)
        to_dict = getattr(volume, "to_dict", None)
        wire = to_dict() if callable(to_dict) else None
    except Exception as exc:
        raise CloudProviderError("Installed huggingface_hub Volume contract could not be constructed.") from exc
    validate_read_only_volume_object(volume, spec)
    return HFVerifiedVolume(spec=spec, provider_volume=volume)


def validate_read_only_volume_object(volume: Any, spec: HFVerifiedVolumeSpec) -> None:
    """Require the exact provider serialization promised by ``spec``."""

    try:
        to_dict = getattr(volume, "to_dict", None)
        wire = to_dict() if callable(to_dict) else None
    except Exception as exc:
        raise CloudProviderError("Installed huggingface_hub Volume contract could not be inspected.") from exc
    expected: dict[str, object] = {
        "type": "bucket",
        "source": spec.source,
        "mountPath": spec.mount_path,
        "readOnly": True,
    }
    if spec.path is not None:
        expected["path"] = spec.path
    if (
        not isinstance(wire, Mapping)
        or dict(wire) != expected
        or type(wire.get("readOnly")) is not bool
    ):
        raise CloudProviderError("Installed huggingface_hub Volume serialization semantics have drifted.")
    if getattr(volume, "read_only", None) is not True:
        raise CloudProviderError("Installed huggingface_hub cannot prove an explicit read-only mount.")


# Trusted launcher code.  This duplicates only the bounded authentication
# needed before importing J0.  The verified J0 module then performs complete
# capsule verification/private copying and invokes the sole bootstrap core.
_INLINE_VERIFIER = r'''import hashlib,importlib.util,json,os,stat,sys,tempfile,shutil
root,expected,lock,lockhash,policy,policyhash,destination=sys.argv[1:8]
manifest_path=os.path.join(root,"synaptic-bootstrap-capsule.json")
def read_regular(path,limit):
 st=os.lstat(path)
 if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_size>limit: raise RuntimeError("bootstrap member is not a bounded regular file")
 flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)
 fd=os.open(path,flags)
 try:
  data=os.read(fd,limit+1); end=os.read(fd,1); opened=os.fstat(fd)
 finally: os.close(fd)
 if end or len(data)>limit or not stat.S_ISREG(opened.st_mode): raise RuntimeError("bootstrap member exceeds its bound")
 if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=(opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns): raise RuntimeError("bootstrap member changed")
 return data
raw=read_regular(manifest_path,65536)
if hashlib.sha256(raw).hexdigest()!=expected: raise RuntimeError("bootstrap manifest digest mismatch")
doc=json.loads(raw.decode("ascii"))
if set(doc)!={"schema_version","engine_commit","files","limits"} or doc.get("schema_version")!="synaptic-bootstrap-capsule/v1": raise RuntimeError("bootstrap manifest shape mismatch")
entries=doc.get("files")
if not isinstance(entries,list) or len(entries)!=2: raise RuntimeError("bootstrap manifest file table mismatch")
entry=next((item for item in entries if isinstance(item,dict) and item.get("path")=="tuner/cloud/bootstrap_capsule.py"),None)
if not entry or set(entry)!={"path","size","sha256","mode"} or type(entry.get("size")) is not int or entry["size"]>2097152: raise RuntimeError("bootstrap verifier entry mismatch")
source=os.path.join(root,"tuner","cloud","bootstrap_capsule.py")
code=read_regular(source,2097152)
if len(code)!=entry["size"] or hashlib.sha256(code).hexdigest()!=entry["sha256"]: raise RuntimeError("bootstrap verifier integrity failure")
scratch=tempfile.mkdtemp(prefix="synaptic-hf-loader-")
try:
 os.chmod(scratch,0o700); target=os.path.join(scratch,"bootstrap_capsule.py")
 with open(target,"xb") as handle: handle.write(code)
 os.chmod(target,0o600)
 module_name="synaptic_verified_capsule"
 spec=importlib.util.spec_from_file_location(module_name,target)
 if spec is None or spec.loader is None or not callable(getattr(spec.loader,"exec_module",None)): raise RuntimeError("bootstrap verifier import contract is unavailable")
 module=importlib.util.module_from_spec(spec)
 if sys.modules.setdefault(module_name,module) is not module: raise RuntimeError("bootstrap verifier module name is already registered")
 try: spec.loader.exec_module(module)
 finally:
  if sys.modules.get(module_name) is module: del sys.modules[module_name]
 result=module.invoke_verified_capsule(root,expected,source_lock_path=lock,source_lock_sha256=lockhash,checkout_policy_path=policy,checkout_policy_sha256=policyhash,destination=destination,scratch_parent=tempfile.gettempdir(),environment=os.environ)
 if result.returncode:
  if result.stderr: sys.stderr.write(result.stderr)
  raise SystemExit(result.returncode)
 raw=result.stdout.encode("ascii")
 if len(raw)>4096: raise RuntimeError("bootstrap result exceeds its bound")
 doc=json.loads(raw.decode("ascii"))
 if set(doc)!={"schema_version","project_root","engine_root","project_commit","engine_commit"} or doc.get("schema_version")!="synaptic-bootstrap-result/v1": raise RuntimeError("bootstrap result shape mismatch")
 if (json.dumps(doc,sort_keys=True,separators=(",",":"))+"\n").encode("ascii")!=raw: raise RuntimeError("bootstrap result is not canonical")
 for name in ("project_commit","engine_commit"):
  value=doc.get(name)
  if not isinstance(value,str) or len(value) not in (40,64) or any(char not in "0123456789abcdef" for char in value): raise RuntimeError("bootstrap result commit identity is invalid")
 base=os.path.realpath(destination)
 for name in ("project_root","engine_root"):
  value=doc.get(name)
  if not isinstance(value,str) or not os.path.isabs(value) or os.path.commonpath((base,os.path.realpath(value)))!=base: raise RuntimeError("bootstrap result root escapes checkout")
 with open(os.path.join(destination,".synaptic-bootstrap-result.json"),"xb") as handle: handle.write(raw)
finally: shutil.rmtree(scratch)
'''


def build_verified_bootstrap_step(
    spec: HFVerifiedVolumeSpec, *, destination: str = _CHECKOUT_ROOT,
) -> str:
    """Return one deterministic shell step that verifies J0 before import."""

    if not destination.startswith("/") or ".." in PurePosixPath(destination).parts:
        raise CloudProviderError("HF bootstrap checkout destination must be an absolute contained path.")
    arguments = [
        "$(command -v python3 || command -v python)",
        "-I",
        "-c",
        _INLINE_VERIFIER,
        spec.mounted(spec.capsule_path),
        spec.capsule_manifest_sha256,
        spec.mounted(spec.source_lock_path),
        spec.source_lock_sha256,
        spec.mounted(spec.checkout_policy_path),
        spec.checkout_policy_sha256,
        destination,
    ]
    return " ".join(arguments[:1] + [shlex.quote(value) for value in arguments[1:]])


def transport_metadata(spec: HFVerifiedVolumeSpec) -> dict[str, object]:
    """Return a sanitized, JSON-safe description for dry-run evidence."""

    return {
        "profile": "hf_read_only_volume",
        "source": spec.source,
        "path": spec.path,
        "mount_path": spec.mount_path,
        "read_only": True,
        "capsule_path": spec.capsule_path,
        "capsule_manifest_sha256": spec.capsule_manifest_sha256,
        "source_lock_path": spec.source_lock_path,
        "source_lock_sha256": spec.source_lock_sha256,
        "checkout_policy_path": spec.checkout_policy_path,
        "checkout_policy_sha256": spec.checkout_policy_sha256,
    }


def _load_bootstrap_result(path: Path) -> dict[str, str]:
    raw = _read_provisioned_regular(path)
    if len(raw) > 4096:
        raise CloudProviderError("HF bootstrap result exceeds its bound.")
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudProviderError("HF bootstrap result is invalid.") from exc
    keys = {"schema_version", "project_root", "engine_root", "project_commit", "engine_commit"}
    if not isinstance(document, dict) or set(document) != keys or document.get("schema_version") != "synaptic-bootstrap-result/v1":
        raise CloudProviderError("HF bootstrap result wire shape has drifted.")
    canonical = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    if canonical != raw or any(not isinstance(document[key], str) or not document[key] for key in keys):
        raise CloudProviderError("HF bootstrap result is not canonical.")
    if not _COMMIT_RE.fullmatch(document["project_commit"]) or not _COMMIT_RE.fullmatch(document["engine_commit"]):
        raise CloudProviderError("HF bootstrap result commit identity is invalid.")
    if not Path(document["project_root"]).is_absolute() or not Path(document["engine_root"]).is_absolute():
        raise CloudProviderError("HF bootstrap result roots must be absolute.")
    return document


def _remove_source_write_bits(root: Path) -> None:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise CloudProviderError("HF authenticated source root is unavailable.") from exc
    if not stat.S_ISDIR(root_info.st_mode) or _is_link_or_reparse(root_info):
        raise CloudProviderError("HF authenticated source root must be a real directory.")
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        safe_directories: list[str] = []
        for name in directory_names:
            candidate = current / name
            info = candidate.lstat()
            if _is_link_or_reparse(info):
                continue
            safe_directories.append(name)
        directory_names[:] = safe_directories
        os.chmod(current, current.lstat().st_mode & ~0o222)
        for name in file_names:
            candidate = current / name
            info = candidate.lstat()
            if _is_link_or_reparse(info):
                continue
            if not stat.S_ISREG(info.st_mode):
                raise CloudProviderError("HF authenticated source contains an unsupported member.")
            os.chmod(candidate, info.st_mode & ~0o222)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    """Return whether an lstat result is a link-like filesystem object."""

    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _validate_physical_root(root: Path, *, checkout_root: Path, label: str) -> Path:
    """Validate a physical source root without traversing a link or junction."""

    if not root.is_absolute() or not checkout_root.is_absolute():
        raise CloudProviderError("HF authenticated physical roots must be absolute.")
    if any(part in {".", ".."} for part in root.parts):
        raise CloudProviderError("HF authenticated physical roots must be canonical.")
    try:
        relative = root.relative_to(checkout_root)
    except ValueError as exc:
        raise CloudProviderError(f"HF authenticated {label} root escapes the checkout.") from exc

    current = checkout_root
    paths = [checkout_root]
    for part in relative.parts:
        current = current / part
        paths.append(current)
    for component in paths:
        try:
            info = component.lstat()
        except OSError as exc:
            raise CloudProviderError(f"HF authenticated {label} root is unavailable.") from exc
        if _is_link_or_reparse(info):
            raise CloudProviderError(
                f"HF authenticated {label} root cannot traverse links or reparse points."
            )
        if not stat.S_ISDIR(info.st_mode):
            raise CloudProviderError(f"HF authenticated {label} root must be a real directory.")

    checkout_resolved = checkout_root.resolve(strict=True)
    resolved = root.resolve(strict=True)
    try:
        resolved.relative_to(checkout_resolved)
    except ValueError as exc:
        raise CloudProviderError(f"HF authenticated {label} root escapes the checkout.") from exc
    return resolved


def _validate_mode_topology(*, mode: str, project: Path, engine: Path) -> None:
    if mode == "standalone":
        valid = project == engine
    elif mode == "superproject":
        valid = project != engine and project in engine.parents
    elif mode == "dual_clone":
        valid = project != engine and project not in engine.parents and engine not in project.parents
    else:
        raise CloudProviderError("HF expected projection mode is invalid.")
    if not valid:
        raise CloudProviderError("HF bootstrap result does not match the locked mode topology.")


def project_runtime_layout(
    bootstrap_result_path: Path,
    *,
    expected_project_root: Path,
    expected_engine_root: Path,
    expected_project_commit: str,
    expected_engine_commit: str,
    expected_mode: str,
    logical_project_root: Path,
    logical_engine_root: Path,
) -> None:
    """Project authenticated physical roots onto canonical logical aliases."""

    result = _load_bootstrap_result(bootstrap_result_path)
    if not _COMMIT_RE.fullmatch(expected_project_commit) or not _COMMIT_RE.fullmatch(expected_engine_commit):
        raise CloudProviderError("HF expected projection commit identity is invalid.")
    project = Path(result["project_root"])
    engine = Path(result["engine_root"])
    if project != expected_project_root or engine != expected_engine_root:
        raise CloudProviderError("HF bootstrap result does not match the locked topology.")
    if (
        result["project_commit"] != expected_project_commit
        or result["engine_commit"] != expected_engine_commit
    ):
        raise CloudProviderError("HF bootstrap result does not match the locked commits.")
    checkout_root = bootstrap_result_path.parent
    project_resolved = _validate_physical_root(project, checkout_root=checkout_root, label="project")
    engine_resolved = _validate_physical_root(engine, checkout_root=checkout_root, label="engine")
    _validate_mode_topology(mode=expected_mode, project=project_resolved, engine=engine_resolved)
    for logical in (logical_project_root, logical_engine_root):
        if not logical.is_absolute() or os.path.lexists(logical):
            raise CloudProviderError("HF logical source target must be an absent absolute path.")
        if not logical.parent.is_dir():
            raise CloudProviderError("HF logical source target parent is unavailable.")
    created: list[Path] = []
    try:
        for logical, physical in (
            (logical_project_root, project_resolved),
            (logical_engine_root, engine_resolved),
        ):
            relative = os.path.relpath(physical, start=logical.parent)
            os.symlink(relative, logical, target_is_directory=True)
            created.append(logical)
            if logical.resolve(strict=True) != physical:
                raise CloudProviderError("HF logical source alias does not resolve to its authenticated root.")
        _remove_source_write_bits(project_resolved)
        if engine_resolved != project_resolved and project_resolved not in engine_resolved.parents:
            _remove_source_write_bits(engine_resolved)
        if (
            _validate_physical_root(project, checkout_root=checkout_root, label="project") != project_resolved
            or _validate_physical_root(engine, checkout_root=checkout_root, label="engine") != engine_resolved
            or not stat.S_ISLNK(logical_project_root.lstat().st_mode)
            or not stat.S_ISLNK(logical_engine_root.lstat().st_mode)
            or logical_project_root.resolve(strict=True) != project_resolved
            or logical_engine_root.resolve(strict=True) != engine_resolved
        ):
            raise CloudProviderError("HF logical source aliases changed during source freezing.")
    except Exception as exc:
        for logical in reversed(created):
            try:
                logical.unlink()
            except OSError:
                pass
        if isinstance(exc, CloudProviderError):
            raise
        raise CloudProviderError("HF logical source projection failed closed.") from exc


def build_runtime_projection_step(
    *, expected_project_root: str, expected_engine_root: str,
    expected_project_commit: str, expected_engine_commit: str, expected_mode: str,
) -> str:
    arguments = [
        f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(expected_engine_root)} $(command -v python3 || command -v python)",
        "-m", "tuner.cloud.hf_volume_transport", "_project-layout",
        "--bootstrap-result", f"{_CHECKOUT_ROOT}/{_BOOTSTRAP_RESULT}",
        "--expected-project-root", expected_project_root,
        "--expected-engine-root", expected_engine_root,
        "--expected-project-commit", expected_project_commit,
        "--expected-engine-commit", expected_engine_commit,
        "--expected-mode", expected_mode,
        "--logical-project-root", "/workspace/project",
        "--logical-engine-root", "/workspace/engine",
    ]
    return " ".join(arguments[:1] + [shlex.quote(value) for value in arguments[1:]])


def _resolve_identity_uri(uri: str, *, project_root: Path, engine_root: Path) -> Path:
    if uri.startswith("project://"):
        root, relative = project_root, uri.removeprefix("project://")
    elif uri.startswith("engine://"):
        root, relative = engine_root, uri.removeprefix("engine://")
    else:
        raise CloudProviderError("Remote source identity uses an unsupported URI scheme.")
    _validate_relative_member(relative, label="identity URI")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CloudProviderError("Remote source identity escapes its reconstructed root.") from exc
    return candidate


def verify_reconstructed_identities(
    source_lock_path: Path, *, project_root: Path, engine_root: Path,
) -> None:
    """Reverify manifest/config/plugin/input hashes after canonical checkout.

    This consumes identities from the sole SourceLock; it does not interpret
    repository URLs, Git metadata, credentials, or checkout policy.
    """

    raw = _read_provisioned_regular(source_lock_path)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudProviderError("Authenticated source lock is not valid JSON.") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "synaptic-source-lock/v1":
        raise CloudProviderError("Authenticated source lock schema is unsupported.")
    identities: list[tuple[str, str]] = []
    project = document.get("project")
    configuration = document.get("configuration")
    if not isinstance(project, dict) or not isinstance(configuration, dict):
        raise CloudProviderError("Source lock identity sections are incomplete.")
    identities.append((str(project.get("manifest_uri", "")), str(project.get("manifest_sha256", ""))))
    documents = configuration.get("documents")
    if not isinstance(documents, list) or not documents:
        raise CloudProviderError("Source lock requires at least one configuration document identity.")
    for item in documents:
        if not isinstance(item, dict):
            raise CloudProviderError("Source lock configuration identity is malformed.")
        identities.append((str(item.get("uri", "")), str(item.get("sha256", ""))))
    for section in ("plugins", "inputs"):
        entries = document.get(section)
        if not isinstance(entries, list):
            raise CloudProviderError(f"Source lock {section} identities are malformed.")
        for item in entries:
            if not isinstance(item, dict):
                raise CloudProviderError(f"Source lock {section} identity is malformed.")
            uri = item.get("source") if section == "plugins" else item.get("uri")
            identities.append((str(uri or ""), str(item.get("sha256", ""))))
    for uri, expected in identities:
        if not _SHA256_RE.fullmatch(expected):
            raise CloudProviderError("Source lock contains an invalid identity digest.")
        content = _read_provisioned_regular(
            _resolve_identity_uri(uri, project_root=project_root, engine_root=engine_root)
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise CloudProviderError("Reconstructed source identity digest mismatch.")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synaptic-hf-volume-transport")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("_verify-identities")
    verify.add_argument("--source-lock", required=True)
    verify.add_argument("--project-root", required=True)
    verify.add_argument("--engine-root", required=True)
    project = sub.add_parser("_project-layout")
    project.add_argument("--bootstrap-result", required=True)
    project.add_argument("--expected-project-root", required=True)
    project.add_argument("--expected-engine-root", required=True)
    project.add_argument("--expected-project-commit", required=True)
    project.add_argument("--expected-engine-commit", required=True)
    project.add_argument("--expected-mode", required=True)
    project.add_argument("--logical-project-root", required=True)
    project.add_argument("--logical-engine-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "_verify-identities":
            verify_reconstructed_identities(
                Path(args.source_lock), project_root=Path(args.project_root), engine_root=Path(args.engine_root),
            )
        else:
            project_runtime_layout(
                Path(args.bootstrap_result),
                expected_project_root=Path(args.expected_project_root),
                expected_engine_root=Path(args.expected_engine_root),
                expected_project_commit=args.expected_project_commit,
                expected_engine_commit=args.expected_engine_commit,
                expected_mode=args.expected_mode,
                logical_project_root=Path(args.logical_project_root),
                logical_engine_root=Path(args.logical_engine_root),
            )
    except CloudProviderError as exc:
        sys.stderr.write(f"source identity verification failed: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "HFVerifiedVolume", "HFVerifiedVolumeSpec", "build_verified_bootstrap_step",
    "prove_preprovisioned_members", "prove_read_only_volume", "transport_metadata",
    "verify_reconstructed_identities", "build_runtime_projection_step", "project_runtime_layout",
    "validate_read_only_volume_object",
]
