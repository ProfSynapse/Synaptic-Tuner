"""Fixed, provider-free HF bootstrap verification workload.

This module runs only after J0 has authenticated and reconstructed the locked
sources and the HF transport has projected them onto the canonical logical
layout.  It deliberately imports only the Python standard library and exposes
no command, configuration, training, or publication extension point.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


WORKLOAD_KIND = "bootstrap_verification"
WORKLOAD_SCHEMA = "synaptic-hf-bootstrap-smoke-workload/v1"
RESULT_SCHEMA = "synaptic-hf-bootstrap-smoke-result/v1"
MAX_BOOTSTRAP_RESULT_BYTES = 4096
MAX_SMOKE_RESULT_BYTES = 4096
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_WRITABLE_NAMES = ("artifacts", "state", "tracking", "cache", "tmp")

# This is data, not a configurable template.  Any workload change necessarily
# changes WORKLOAD_SHA256 and therefore invalidates an exact-run approval.
_WORKLOAD_DOCUMENT: dict[str, object] = {
    "schema_version": WORKLOAD_SCHEMA,
    "kind": WORKLOAD_KIND,
    "runtime": {"image": "python:3.12"},
    "hardware": {"flavor": "cpu-basic"},
    "limits": {
        "provider_timeout_seconds": 600,
        "cancel_after_seconds": 720,
        "outer_observation_seconds": 900,
        "projected_compute_usd": "0.01",
        "hard_total_usd": "2.00",
    },
    "network": {"ports": [], "ssh": False},
    "retries": 0,
    "effects": {"training": False, "publication": False},
}


class BootstrapSmokeError(RuntimeError):
    """Fail-closed, path-free error from the fixed smoke workload."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


WORKLOAD: Mapping[str, object] = _freeze(_WORKLOAD_DOCUMENT)  # type: ignore[assignment]
_WORKLOAD_BYTES = _canonical_json(_WORKLOAD_DOCUMENT)
WORKLOAD_SHA256 = hashlib.sha256(_WORKLOAD_BYTES).hexdigest()


def canonical_workload_bytes() -> bytes:
    """Return the immutable bytes bound into an exact-run approval."""

    return _WORKLOAD_BYTES


def workload_sha256() -> str:
    """Return the canonical fixed-workload digest."""

    if hashlib.sha256(_WORKLOAD_BYTES).hexdigest() != WORKLOAD_SHA256:
        raise BootstrapSmokeError("Bootstrap smoke workload identity changed.")
    return WORKLOAD_SHA256


def _read_regular(path: Path, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise BootstrapSmokeError("Bootstrap smoke input must be a regular file.")
        if before.st_size > maximum:
            raise BootstrapSmokeError("Bootstrap smoke input exceeds its bound.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            opened = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except BootstrapSmokeError:
        raise
    except OSError as exc:
        raise BootstrapSmokeError("Bootstrap smoke input is unavailable.") from exc
    content = b"".join(chunks)
    if len(content) > maximum or not stat.S_ISREG(opened.st_mode):
        raise BootstrapSmokeError("Bootstrap smoke input exceeds its bound.")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ):
        raise BootstrapSmokeError("Bootstrap smoke input changed during verification.")
    return content


def _load_bootstrap_result(path: Path) -> dict[str, str]:
    raw = _read_regular(path, maximum=MAX_BOOTSTRAP_RESULT_BYTES)

    class DuplicateKey(ValueError):
        pass

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKey
            result[key] = value
        return result

    try:
        document = json.loads(raw.decode("ascii"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateKey):
        raise BootstrapSmokeError("Bootstrap result is invalid.") from None
    keys = {"schema_version", "project_root", "engine_root", "project_commit", "engine_commit"}
    if (
        not isinstance(document, dict)
        or set(document) != keys
        or document.get("schema_version") != "synaptic-bootstrap-result/v1"
        or _canonical_json(document) != raw
    ):
        raise BootstrapSmokeError("Bootstrap result does not match its canonical contract.")
    for name in ("project_root", "engine_root"):
        value = document.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise BootstrapSmokeError("Bootstrap result contains an invalid source root.")
    for name in ("project_commit", "engine_commit"):
        value = document.get(name)
        if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
            raise BootstrapSmokeError("Bootstrap result contains an invalid commit identity.")
    return {key: str(value) for key, value in document.items()}


def _validate_physical_root(root: Path, checkout_root: Path) -> Path:
    if not root.is_absolute() or not checkout_root.is_absolute():
        raise BootstrapSmokeError("Bootstrap source roots must be absolute.")
    try:
        relative = root.relative_to(checkout_root)
    except ValueError:
        raise BootstrapSmokeError("Bootstrap source root escapes its checkout.") from None
    current = checkout_root
    components = [current]
    for part in relative.parts:
        current = current / part
        components.append(current)
    for current in components:
        try:
            info = current.lstat()
        except OSError as exc:
            raise BootstrapSmokeError("Bootstrap source root is unavailable.") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise BootstrapSmokeError("Bootstrap source root is not a physical directory.")
    try:
        resolved_checkout = checkout_root.resolve(strict=True)
        resolved = root.resolve(strict=True)
        resolved.relative_to(resolved_checkout)
    except (OSError, ValueError):
        raise BootstrapSmokeError("Bootstrap source root escapes its checkout.") from None
    return resolved


def _is_reparse(info: os.stat_result) -> bool:
    return bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _validate_logical_root(logical: Path, physical: Path) -> None:
    if not logical.is_absolute():
        raise BootstrapSmokeError("Bootstrap logical source root must be absolute.")
    try:
        info = logical.lstat()
        resolved = logical.resolve(strict=True)
    except OSError as exc:
        raise BootstrapSmokeError("Bootstrap logical source root is unavailable.") from exc
    # HF Jobs runs Linux.  Windows cannot reliably create unprivileged symlinks,
    # so local contract tests may use a physical alias; the POSIX runtime may not.
    if os.name != "nt" and not stat.S_ISLNK(info.st_mode):
        raise BootstrapSmokeError("Bootstrap logical source root is not a verified alias.")
    if resolved != physical:
        raise BootstrapSmokeError("Bootstrap logical source root changed identity.")


def _verify_git_commit(root: Path, expected: str) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR"}
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BootstrapSmokeError("Bootstrap commit could not be verified.") from exc
    actual = completed.stdout.strip().lower()
    if completed.returncode or not _COMMIT_RE.fullmatch(actual) or actual != expected:
        raise BootstrapSmokeError("Bootstrap commit identity does not match.")


def _verify_source_read_only(root: Path) -> None:
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        try:
            current_info = current.lstat()
        except OSError as exc:
            raise BootstrapSmokeError("Bootstrap source permissions could not be verified.") from exc
        if current_info.st_mode & 0o222:
            raise BootstrapSmokeError("Bootstrap source root remains writable.")
        kept: list[str] = []
        for name in directory_names:
            candidate = current / name
            try:
                info = candidate.lstat()
            except OSError as exc:
                raise BootstrapSmokeError(
                    "Bootstrap source permissions could not be verified."
                ) from exc
            if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
                raise BootstrapSmokeError(
                    "Bootstrap source cannot contain links or reparse points."
                )
            if not stat.S_ISDIR(info.st_mode):
                raise BootstrapSmokeError("Bootstrap source contains an unsupported member.")
            kept.append(name)
        directory_names[:] = kept
        for name in file_names:
            try:
                info = (current / name).lstat()
            except OSError as exc:
                raise BootstrapSmokeError(
                    "Bootstrap source permissions could not be verified."
                ) from exc
            if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
                raise BootstrapSmokeError(
                    "Bootstrap source cannot contain links or reparse points."
                )
            if not stat.S_ISREG(info.st_mode):
                raise BootstrapSmokeError("Bootstrap source contains an unsupported member.")
            if info.st_mode & 0o222:
                raise BootstrapSmokeError("Bootstrap source root remains writable.")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _verify_writable_roots(
    workspace_root: Path, writable_roots: Mapping[str, Path], source_roots: tuple[Path, ...]
) -> None:
    if set(writable_roots) != set(_WRITABLE_NAMES):
        raise BootstrapSmokeError("Bootstrap writable-root inventory is not canonical.")
    workspace = workspace_root.resolve(strict=True)
    resolved_writable: list[Path] = []
    for name in _WRITABLE_NAMES:
        root = writable_roots[name]
        if not root.is_absolute() or root.parent != workspace_root or root.name != name:
            raise BootstrapSmokeError("Bootstrap writable root is outside its canonical layout.")
        try:
            info = root.lstat()
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise BootstrapSmokeError("Bootstrap writable root is unavailable.") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise BootstrapSmokeError("Bootstrap writable root must be a physical directory.")
        if not _contains(workspace, resolved):
            raise BootstrapSmokeError("Bootstrap writable root escapes its workspace.")
        if any(_contains(source, resolved) or _contains(resolved, source) for source in source_roots):
            raise BootstrapSmokeError("Bootstrap writable root overlaps source.")
        if any(_contains(other, resolved) or _contains(resolved, other) for other in resolved_writable):
            raise BootstrapSmokeError("Bootstrap writable roots overlap.")
        try:
            descriptor, probe = tempfile.mkstemp(prefix=".synaptic-smoke-", dir=root)
            os.close(descriptor)
            Path(probe).unlink()
        except OSError as exc:
            raise BootstrapSmokeError("Bootstrap writable root is not writable.") from exc
        resolved_writable.append(resolved)


def _result_document(*, project_commit: str, engine_commit: str) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "success": True,
        "workload": {"kind": WORKLOAD_KIND, "sha256": workload_sha256()},
        "sources": {
            "project": {"identity": "project://", "commit": project_commit, "read_only": True},
            "engine": {"identity": "engine://", "commit": engine_commit, "read_only": True},
        },
        "writable_roots": list(_WRITABLE_NAMES),
        "checks": {
            "bootstrap_result": True,
            "commit_identities": True,
            "logical_roots": True,
            "source_permissions": True,
            "writable_containment": True,
        },
    }
    result = dict(base)
    result["result_sha256"] = hashlib.sha256(_canonical_json(base)).hexdigest()
    return result


def canonical_result_bytes(result: Mapping[str, object]) -> bytes:
    """Serialize one successful fixed-smoke result within its wire bound."""

    document = dict(result)
    expected_keys = {
        "schema_version", "success", "workload", "sources", "writable_roots", "checks",
        "result_sha256",
    }
    if set(document) != expected_keys or document.get("schema_version") != RESULT_SCHEMA:
        raise BootstrapSmokeError("Bootstrap smoke result shape is invalid.")
    if document.get("success") is not True:
        raise BootstrapSmokeError("Bootstrap smoke result is not successful.")
    workload = document.get("workload")
    if not isinstance(workload, Mapping) or dict(workload) != {
        "kind": WORKLOAD_KIND,
        "sha256": workload_sha256(),
    }:
        raise BootstrapSmokeError("Bootstrap smoke result workload binding is invalid.")
    sources = document.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"project", "engine"}:
        raise BootstrapSmokeError("Bootstrap smoke result source binding is invalid.")
    for name, identity in (("project", "project://"), ("engine", "engine://")):
        source = sources.get(name)
        if (
            not isinstance(source, Mapping)
            or set(source) != {"identity", "commit", "read_only"}
            or source.get("identity") != identity
            or source.get("read_only") is not True
            or not isinstance(source.get("commit"), str)
            or not _COMMIT_RE.fullmatch(str(source.get("commit")))
        ):
            raise BootstrapSmokeError("Bootstrap smoke result source binding is invalid.")
    if document.get("writable_roots") != list(_WRITABLE_NAMES):
        raise BootstrapSmokeError("Bootstrap smoke result writable-root binding is invalid.")
    expected_checks = {
        "bootstrap_result": True,
        "commit_identities": True,
        "logical_roots": True,
        "source_permissions": True,
        "writable_containment": True,
    }
    checks = document.get("checks")
    if not isinstance(checks, Mapping) or dict(checks) != expected_checks:
        raise BootstrapSmokeError("Bootstrap smoke result checks are invalid.")
    digest = document.pop("result_sha256")
    if not isinstance(digest, str) or digest != hashlib.sha256(_canonical_json(document)).hexdigest():
        raise BootstrapSmokeError("Bootstrap smoke result digest is invalid.")
    document["result_sha256"] = digest
    raw = _canonical_json(document)
    if len(raw) > MAX_SMOKE_RESULT_BYTES:
        raise BootstrapSmokeError("Bootstrap smoke result exceeds its bound.")
    return raw


def run_bootstrap_smoke(
    *,
    bootstrap_result_path: Path = Path("/workspace/source/.synaptic-bootstrap-result.json"),
    workspace_root: Path = Path("/workspace"),
    project_root: Path = Path("/workspace/project"),
    engine_root: Path = Path("/workspace/engine"),
    writable_roots: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Run the sole fixed bootstrap-verification workload."""

    if writable_roots is None:
        writable_roots = {name: workspace_root / name for name in _WRITABLE_NAMES}
    bootstrap = _load_bootstrap_result(bootstrap_result_path)
    checkout_root = bootstrap_result_path.parent
    physical_project = _validate_physical_root(Path(bootstrap["project_root"]), checkout_root)
    physical_engine = _validate_physical_root(Path(bootstrap["engine_root"]), checkout_root)
    _validate_logical_root(project_root, physical_project)
    _validate_logical_root(engine_root, physical_engine)
    _verify_git_commit(project_root, bootstrap["project_commit"])
    _verify_git_commit(engine_root, bootstrap["engine_commit"])
    unique_sources = tuple(dict.fromkeys((physical_project, physical_engine)))
    for source in unique_sources:
        _verify_source_read_only(source)
    _verify_writable_roots(workspace_root, writable_roots, unique_sources)
    return _result_document(
        project_commit=bootstrap["project_commit"], engine_commit=bootstrap["engine_commit"]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """No-argument remote entrypoint for the immutable workload."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        sys.stderr.write("bootstrap smoke failed: arguments are not supported\n")
        return 2
    try:
        result = run_bootstrap_smoke()
        sys.stdout.buffer.write(canonical_result_bytes(result))
    except BootstrapSmokeError as exc:
        sys.stderr.write(f"bootstrap smoke failed: {exc}\n")
        return 2
    except Exception:
        # Unexpected platform/runtime failures must not leak paths, environment
        # values, command output, or provider context into remote logs.
        sys.stderr.write("bootstrap smoke failed: internal verification error\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapSmokeError",
    "RESULT_SCHEMA",
    "WORKLOAD",
    "WORKLOAD_KIND",
    "WORKLOAD_SCHEMA",
    "WORKLOAD_SHA256",
    "canonical_result_bytes",
    "canonical_workload_bytes",
    "main",
    "run_bootstrap_smoke",
    "workload_sha256",
]
