"""Deterministic builder and bounded verifier for the bootstrap capsule.

The capsule is tied to one committed engine revision and contains code only.
Per-run source locks and checkout policies are transported as separate,
launcher-hash-bound JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping, Sequence


CAPSULE_SCHEMA = "synaptic-bootstrap-capsule/v1"
CAPSULE_MANIFEST = "synaptic-bootstrap-capsule.json"
CAPSULE_MODULE_PATHS = (
    "tuner/cloud/bootstrap_core.py",
    "tuner/cloud/bootstrap_capsule.py",
)
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CAPSULE_FILE_BYTES = 2 * 1024 * 1024
MAX_CAPSULE_BYTES = 4 * 1024 * 1024
MAX_EXTERNAL_INPUT_BYTES = 4 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MANIFEST_KEYS = {"schema_version", "engine_commit", "files", "limits"}
_LIMIT_KEYS = {"max_file_bytes", "max_total_bytes"}
_MEMBER_KEYS = {"path", "size", "sha256", "mode"}


class CapsuleError(RuntimeError):
    """A deterministic, non-secret-bearing capsule failure."""


def _is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _assert_safe_path_components(
    value: str | os.PathLike[str], *, require_leaf: bool = False,
) -> Path:
    """Inspect lexical components without erasing link/reparse evidence."""

    path = Path(os.path.abspath(os.fspath(value)))
    parts = path.parts
    if not parts:
        raise CapsuleError("Capsule path is invalid")
    current = Path(parts[0])
    for index in range(len(parts)):
        if index:
            current = current / parts[index]
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise CapsuleError("Capsule path could not be inspected safely") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise CapsuleError("Capsule paths cannot contain links or reparse points")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise CapsuleError("Capsule path contains a non-directory component")
    if require_leaf and not path.exists():
        raise CapsuleError("Capsule path does not exist")
    return path


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str,
) -> None:
    if set(value) != expected:
        raise CapsuleError(f"{label} does not match the canonical wire shape")


def _require_exact_int(
    value: object, *, label: str, minimum: int | None = None,
    maximum: int | None = None, allowed: set[int] | None = None,
) -> int:
    if type(value) is not int:
        raise CapsuleError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise CapsuleError(f"{label} is outside the allowed range")
    if maximum is not None and value > maximum:
        raise CapsuleError(f"{label} is outside the allowed range")
    if allowed is not None and value not in allowed:
        raise CapsuleError(f"{label} is unsupported")
    return value


@dataclass(frozen=True)
class CapsuleBuild:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    engine_commit: str


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_environment() -> dict[str, str]:
    allowed = {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
        "LANG", "LC_ALL", "LC_CTYPE",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _git_bytes(repository: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments], env=_git_environment(),
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapsuleError("Could not read committed bootstrap objects") from exc
    if completed.returncode:
        raise CapsuleError("Could not read committed bootstrap objects")
    return completed.stdout


def _committed_member(repository: Path, commit: str, member: str) -> tuple[bytes, int]:
    tree = _git_bytes(repository, ["ls-tree", commit, "--", member]).decode("ascii", "strict").strip()
    fields = tree.split(None, 3)
    if len(fields) != 4 or fields[1] != "blob" or fields[3] != member:
        raise CapsuleError("Bootstrap member is not a committed regular blob")
    if fields[0] not in {"100644", "100755"}:
        raise CapsuleError("Bootstrap member has an unsupported committed mode")
    content = _git_bytes(repository, ["show", f"{commit}:{member}"])
    if len(content) > MAX_CAPSULE_FILE_BYTES:
        raise CapsuleError("Bootstrap member exceeds the fixed size limit")
    return content, (0o755 if fields[0] == "100755" else 0o644)


def build_capsule(
    repository: str | os.PathLike[str], output_root: str | os.PathLike[str], *, revision: str = "HEAD",
) -> CapsuleBuild:
    """Build code-only capsule bytes from exact committed Git objects."""

    repository_path = _assert_safe_path_components(repository, require_leaf=True)
    output = _assert_safe_path_components(output_root)
    commit = _git_bytes(repository_path, ["rev-parse", "--verify", f"{revision}^{{commit}}"] ).decode("ascii").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise CapsuleError("Engine revision did not resolve to an exact commit")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise CapsuleError("Capsule output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    _assert_safe_path_components(output, require_leaf=True)
    members: list[dict[str, object]] = []
    total = 0
    for member in CAPSULE_MODULE_PATHS:
        content, mode = _committed_member(repository_path, commit, member)
        total += len(content)
        if total > MAX_CAPSULE_BYTES:
            raise CapsuleError("Bootstrap capsule exceeds the aggregate size limit")
        destination = output.joinpath(*PurePosixPath(member).parts)
        _assert_safe_path_components(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_safe_path_components(destination.parent, require_leaf=True)
        destination.write_bytes(content)
        os.chmod(destination, mode)
        members.append({"path": member, "size": len(content), "sha256": _sha256(content), "mode": mode})
    manifest = {
        "schema_version": CAPSULE_SCHEMA,
        "engine_commit": commit,
        "files": members,
        "limits": {
            "max_file_bytes": MAX_CAPSULE_FILE_BYTES,
            "max_total_bytes": MAX_CAPSULE_BYTES,
        },
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_path = output / CAPSULE_MANIFEST
    manifest_path.write_bytes(manifest_bytes)
    os.chmod(manifest_path, 0o644)
    return CapsuleBuild(output, manifest_path, _sha256(manifest_bytes), commit)


def _safe_member_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CapsuleError("Capsule manifest contains an unsafe member path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CapsuleError("Capsule manifest contains an unsafe member path")
    normalized = path.as_posix()
    if normalized != value:
        raise CapsuleError("Capsule manifest member paths must be canonical")
    return normalized


def _stat_signature(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _read_regular_file(path: Path, *, maximum: int) -> tuple[bytes, tuple[int, int, int, int]]:
    """Read one stable regular file with no-follow behavior where available."""

    path = _assert_safe_path_components(path, require_leaf=True)
    try:
        before_path = path.stat(follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise CapsuleError("Capsule member is unavailable") from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise CapsuleError("Capsule members must be regular files")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode) or opened_before.st_size > maximum:
                raise CapsuleError("Capsule member is not a bounded regular file")
            chunks: list[bytes] = []
            count = 0
            while True:
                chunk = os.read(descriptor, min(65536, maximum + 1 - count))
                if not chunk:
                    break
                chunks.append(chunk)
                count += len(chunk)
                if count > maximum:
                    raise CapsuleError("Capsule member exceeds the fixed size limit")
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except CapsuleError:
        raise
    except OSError as exc:
        raise CapsuleError("Capsule member could not be read safely") from exc
    try:
        after_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CapsuleError("Capsule member changed while being verified") from exc
    signatures = {_stat_signature(item) for item in (before_path, opened_before, opened_after, after_path)}
    if len(signatures) != 1:
        raise CapsuleError("Capsule member changed while being verified")
    return b"".join(chunks), _stat_signature(after_path)


def _load_manifest(root: Path, expected_digest: str) -> tuple[dict[str, object], bytes]:
    if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(expected_digest):
        raise CapsuleError("Expected capsule manifest digest must be lowercase SHA-256")
    manifest_bytes, _ = _read_regular_file(root / CAPSULE_MANIFEST, maximum=MAX_MANIFEST_BYTES)
    if _sha256(manifest_bytes) != expected_digest:
        raise CapsuleError("Capsule manifest digest mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapsuleError("Capsule manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CAPSULE_SCHEMA:
        raise CapsuleError("Capsule manifest schema is unsupported")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "Capsule manifest")
    if _canonical_json(manifest) != manifest_bytes:
        raise CapsuleError("Capsule manifest is not canonically encoded")
    commit = manifest.get("engine_commit")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise CapsuleError("Capsule manifest engine commit is invalid")
    limits = manifest.get("limits")
    if not isinstance(limits, Mapping):
        raise CapsuleError("Capsule manifest limits are invalid")
    _require_exact_keys(limits, _LIMIT_KEYS, "Capsule manifest limits")
    max_file = _require_exact_int(
        limits.get("max_file_bytes"), label="Capsule manifest max_file_bytes",
        minimum=0, maximum=MAX_CAPSULE_FILE_BYTES,
    )
    max_total = _require_exact_int(
        limits.get("max_total_bytes"), label="Capsule manifest max_total_bytes",
        minimum=0, maximum=MAX_CAPSULE_BYTES,
    )
    if max_file != MAX_CAPSULE_FILE_BYTES or max_total != MAX_CAPSULE_BYTES:
        raise CapsuleError("Capsule manifest limits do not match the verifier contract")
    return manifest, manifest_bytes


def _cleanup_private_scratch(scratch: Path) -> None:
    """Remove private scratch or fail visibly without following a new link."""

    if not os.path.lexists(scratch):
        return
    _assert_safe_path_components(scratch, require_leaf=True)
    shutil.rmtree(scratch)
    if os.path.lexists(scratch):
        raise CapsuleError("Private capsule scratch cleanup failed")


@contextmanager
def verified_capsule_scratch(
    capsule_root: str | os.PathLike[str], expected_manifest_sha256: str, *,
    scratch_parent: str | os.PathLike[str] | None = None,
    after_member_read: Callable[[Path], None] | None = None,
) -> Iterator[Path]:
    """Authenticate capsule code, copy it privately, and remove it afterward.

    ``after_member_read`` is a deterministic race-test seam and must not be
    supplied by provider/runtime input.
    """

    root = _assert_safe_path_components(capsule_root, require_leaf=True)
    parent = _assert_safe_path_components(
        scratch_parent if scratch_parent is not None else tempfile.gettempdir(),
        require_leaf=True,
    )
    if not root.is_dir():
        raise CapsuleError("Capsule root must be a directory")
    manifest, manifest_bytes = _load_manifest(root, expected_manifest_sha256)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(CAPSULE_MODULE_PATHS):
        raise CapsuleError("Capsule manifest requires a file table")
    declared: list[tuple[str, int, str, int]] = []
    for raw_entry in raw_files:
        if not isinstance(raw_entry, Mapping):
            raise CapsuleError("Capsule file entries must be objects")
        _require_exact_keys(raw_entry, _MEMBER_KEYS, "Capsule file entry")
        member = _safe_member_path(raw_entry.get("path"))
        size = _require_exact_int(
            raw_entry.get("size"), label="Capsule manifest contains invalid member size",
            minimum=0, maximum=MAX_CAPSULE_FILE_BYTES,
        )
        digest = raw_entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CapsuleError("Capsule manifest contains an invalid member digest")
        mode = _require_exact_int(
            raw_entry.get("mode"), label="Capsule manifest member mode",
            allowed={0o644, 0o755},
        )
        declared.append((member, size, digest, mode))
    declared_paths = [member for member, _size, _digest, _mode in declared]
    if len(set(declared_paths)) != len(declared_paths) or len(
        {path.casefold() for path in declared_paths}
    ) != len(declared_paths):
        raise CapsuleError("Capsule manifest contains duplicate member paths")
    if tuple(declared_paths) != CAPSULE_MODULE_PATHS:
        raise CapsuleError("Capsule manifest modules are not in canonical order")
    if sum(size for _member, size, _digest, _mode in declared) > MAX_CAPSULE_BYTES:
        raise CapsuleError("Bootstrap capsule exceeds the aggregate size limit")
    entries: list[tuple[str, bytes, int, tuple[int, int, int, int]]] = []
    seen: set[str] = set()
    casefolded: set[str] = set()
    total = 0
    for member, size, digest, mode in declared:
        if member in seen or member.casefold() in casefolded:
            raise CapsuleError("Capsule manifest contains duplicate member paths")
        seen.add(member)
        casefolded.add(member.casefold())
        path = root.joinpath(*PurePosixPath(member).parts)
        content, signature = _read_regular_file(path, maximum=MAX_CAPSULE_FILE_BYTES)
        if len(content) != size or _sha256(content) != digest:
            raise CapsuleError("Capsule member integrity check failed")
        total += len(content)
        if total > MAX_CAPSULE_BYTES:
            raise CapsuleError("Bootstrap capsule exceeds the aggregate size limit")
        entries.append((member, content, mode, signature))
    if tuple(member for member, _content, _mode, _signature in entries) != CAPSULE_MODULE_PATHS:
        raise CapsuleError("Capsule manifest modules are not in canonical order")

    scratch = Path(tempfile.mkdtemp(prefix="synaptic-bootstrap-", dir=parent))
    primary_error: BaseException | None = None
    try:
        try:
            _assert_safe_path_components(scratch, require_leaf=True)
            os.chmod(scratch, 0o700)
            for member, content, mode, signature in entries:
                source = root.joinpath(*PurePosixPath(member).parts)
                if after_member_read:
                    after_member_read(source)
                current = source.stat(follow_symlinks=False)
                if _stat_signature(current) != signature or not stat.S_ISREG(current.st_mode):
                    raise CapsuleError("Capsule member changed before private copy")
                destination = scratch.joinpath(*PurePosixPath(member).parts)
                _assert_safe_path_components(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _assert_safe_path_components(destination.parent, require_leaf=True)
                destination.write_bytes(content)
                os.chmod(destination, mode)
                copied, _ = _read_regular_file(destination, maximum=MAX_CAPSULE_FILE_BYTES)
                if copied != content:
                    raise CapsuleError("Private capsule copy failed integrity verification")
            copied_manifest = scratch / CAPSULE_MANIFEST
            copied_manifest.write_bytes(manifest_bytes)
            os.chmod(copied_manifest, 0o600)
            if _sha256(copied_manifest.read_bytes()) != expected_manifest_sha256:
                raise CapsuleError("Private manifest copy failed integrity verification")
            yield scratch
        except (OSError, ValueError) as exc:
            wrapped = CapsuleError("Capsule could not be copied to private scratch")
            primary_error = wrapped
            raise wrapped from exc
        except BaseException as exc:
            primary_error = exc
            raise
    finally:
        try:
            _cleanup_private_scratch(scratch)
        except Exception as cleanup_exc:
            cleanup_error = CapsuleError("Private capsule scratch cleanup failed")
            if primary_error is not None:
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note("Private capsule scratch cleanup also failed")
            else:
                raise cleanup_error from cleanup_exc


def authenticate_external_input(path: str | os.PathLike[str], expected_sha256: str) -> bytes:
    """Authenticate one separate run input without interpreting its bytes."""

    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise CapsuleError("Expected external input digest must be lowercase SHA-256")
    safe_path = _assert_safe_path_components(path, require_leaf=True)
    content, _ = _read_regular_file(safe_path, maximum=MAX_EXTERNAL_INPUT_BYTES)
    if _sha256(content) != expected_sha256:
        raise CapsuleError("External bootstrap input digest mismatch")
    return content


def invoke_verified_capsule(
    capsule_root: str | os.PathLike[str], expected_manifest_sha256: str, *,
    source_lock_path: str | os.PathLike[str], source_lock_sha256: str,
    checkout_policy_path: str | os.PathLike[str], checkout_policy_sha256: str,
    destination: str | os.PathLike[str], scratch_parent: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Copy verified code, then invoke its run-agnostic external-input contract."""

    _assert_safe_path_components(source_lock_path, require_leaf=True)
    _assert_safe_path_components(checkout_policy_path, require_leaf=True)
    _assert_safe_path_components(destination)
    with verified_capsule_scratch(capsule_root, expected_manifest_sha256, scratch_parent=scratch_parent) as scratch:
        entrypoint = scratch.joinpath(*PurePosixPath("tuner/cloud/bootstrap_capsule.py").parts)
        command = [
            sys.executable, "-I", str(entrypoint), "_run-verified",
            "--source-lock", str(source_lock_path), "--source-lock-sha256", source_lock_sha256,
            "--checkout-policy", str(checkout_policy_path), "--checkout-policy-sha256", checkout_policy_sha256,
            "--destination", str(destination),
        ]
        return subprocess.run(command, env=dict(environment) if environment is not None else None,
                              capture_output=True, text=True, timeout=300)


def _run_verified(args: argparse.Namespace) -> int:
    source_lock_bytes = authenticate_external_input(args.source_lock, args.source_lock_sha256)
    policy_bytes = authenticate_external_input(args.checkout_policy, args.checkout_policy_sha256)
    # Import only after the trusted verifier copied and authenticated capsule
    # code and this entrypoint authenticated the separate per-run inputs.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bootstrap_core import BootstrapError, reconstruct_source_lock_json

    try:
        result = reconstruct_source_lock_json(source_lock_bytes, policy_bytes, args.destination, environment=os.environ)
    except BootstrapError as exc:
        raise CapsuleError(str(exc)) from exc
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synaptic-bootstrap-capsule")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("_run-verified")
    run.add_argument("--source-lock", required=True)
    run.add_argument("--source-lock-sha256", required=True)
    run.add_argument("--checkout-policy", required=True)
    run.add_argument("--checkout-policy-sha256", required=True)
    run.add_argument("--destination", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "_run-verified":
            return _run_verified(args)
        raise CapsuleError("Unsupported capsule command")
    except CapsuleError as exc:
        sys.stderr.write(f"bootstrap failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPSULE_MANIFEST", "CAPSULE_MODULE_PATHS", "CAPSULE_SCHEMA", "CapsuleBuild", "CapsuleError",
    "authenticate_external_input", "build_capsule", "invoke_verified_capsule", "verified_capsule_scratch",
]
