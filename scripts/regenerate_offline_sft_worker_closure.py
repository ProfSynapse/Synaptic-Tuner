"""Refresh the checked-in offline SFT worker closure manifest in place.

The manifest at ``tuner/runtime/manifests/offline-sft-worker-v1.json`` pins the
exact source closure that executes inside the network-disabled training
container: every member by ``size_bytes`` and ``sha256``, plus a whole-closure
``closure_digest``. Editing any member (for example ``Trainers/sft/train_sft.py``)
invalidates it, and every prepared run then fails at staging.

This tool REFRESHES that manifest; it never rebuilds it. The member list is read
from the manifest's own ``members[].path`` and is written back unchanged, so the
closure cannot widen from here: adding a file to the container's executable set
requires a deliberate edit of both this manifest and the independent ``_MEMBERS``
tuple in ``tests/contract/test_offline_sft_worker_closure.py``, which a reviewer
sees as a diff of paths. A filesystem walk would make widening the tool's default
behaviour, so there is deliberately no walk.

Exactly four values are ever rewritten (architecture ruling section 15.5):
``members[<path>].sha256``, ``members[<path>].size_bytes``, the ``payload_bytes``
total, and ``closure_digest``. The six identity fields, the member paths, the
``git_mode`` values, the ordering and ``member_count`` pass through untouched;
the production parser re-checks them verbatim, so touching one fails closed.

Usage (``--check`` is the default; a bare invocation never mutates the artifact)::

    python3 scripts/regenerate_offline_sft_worker_closure.py            # check
    python3 scripts/regenerate_offline_sft_worker_closure.py --write    # refresh

Exit codes: 0 already current, 3 drift, 125 fault (nothing written).

Related: ``tuner/runtime/offline_sft_worker.py`` owns the canonical
serialization, the digest and the parser, all of which this script imports
rather than reimplements. ``tests/contract/test_offline_sft_worker_closure.py``
is the independent round-trip proof and runs this script in ``--check`` mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


_MANIFEST_RELATIVE = "tuner/runtime/manifests/offline-sft-worker-v1.json"


def _authenticated_repo_root() -> Path:
    raw_script = Path(__file__)
    if raw_script.is_symlink():
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    script = raw_script.resolve(strict=True)
    if not script.is_file() or script.name != "regenerate_offline_sft_worker_closure.py":
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    root = script.parents[1]
    if script.parent.name != "scripts" or (root / "scripts" / script.name).resolve(strict=True) != script:
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    for relative in (
        "tuner/runtime/offline_sft_worker.py",
        _MANIFEST_RELATIVE,
    ):
        anchor = root / relative
        if anchor.is_symlink() or not anchor.is_file() or anchor.resolve(strict=True) != anchor.absolute():
            raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    return root


try:
    REPO_ROOT = _authenticated_repo_root()
except (OSError, RuntimeError):
    print(
        "offline SFT worker closure regeneration failed: SCRIPT_IDENTITY_INVALID",
        file=sys.stderr,
    )
    raise SystemExit(125)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuner.runtime.offline_sft_worker import (  # noqa: E402
    OfflineSFTWorkerError,
    _MAX_CLOSURE_BYTES,
    _MAX_MEMBER_BYTES,
    _canonical_json,
    closure_digest,
    parse_offline_sft_worker_manifest,
)


class RegenerationFault(RuntimeError):
    """A fail-closed condition. Carries the reason code printed to stderr."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


def _serialize(document: dict) -> bytes:
    """Return the exact on-disk bytes: canonical JSON PLUS the trailing newline.

    The newline is part of the file and NOT part of the digest input; conflating
    the two is the one mistake that produces a manifest which only fails at run
    time. ``closure_digest`` applies the same canonical call to the document with
    ``closure_digest`` popped and no newline.
    """

    return _canonical_json(document) + b"\n"


def _read_member(root: Path, relative: str) -> bytes:
    """Read one closure member, refusing anything that is not a plain file."""

    if not relative or relative.startswith("/") or ".." in relative.split("/"):
        raise RegenerationFault("MEMBER_PATH_INVALID", relative)
    path = root.joinpath(*relative.split("/"))
    if path.is_symlink():
        raise RegenerationFault("MEMBER_NOT_REGULAR_FILE", relative)
    if not path.is_file():
        raise RegenerationFault("MEMBER_MISSING", relative)
    if path.resolve(strict=True) != path.absolute():
        raise RegenerationFault("MEMBER_NOT_REGULAR_FILE", relative)
    payload = path.read_bytes()
    if len(payload) > _MAX_MEMBER_BYTES:
        raise RegenerationFault("MEMBER_EXCEEDS_BYTE_BOUND", relative)
    return payload


def _git_index_modes(root: Path, paths: list[str]) -> dict[str, str]:
    """Return each member's mode as recorded in the git index.

    Git is the authority for a field named ``git_mode``, and a ``path.stat()``
    executable bit is the wrong oracle anywhere ``core.fileMode`` is false. On a
    Windows-backed DrvFs/9p mount every file reports ``0o777`` while the index
    still records ``100644``, so a stat-based check would refuse all 66 members
    on a tree that is in fact correct (architecture ruling 15.13).

    ``safe.directory`` is passed per invocation with ``-c`` and names only this
    repository root. A checkout whose owner differs from the invoking user (any
    Windows-backed mount read from WSL) otherwise trips git's dubious-ownership
    guard and the oracle is unavailable exactly where it is most needed. The
    override is deliberately NOT a config write: the tool must not mutate the
    user's git configuration as a side effect of running, and the exemption must
    not outlive the one read. Scope is narrow by construction, since the
    generator already reads these members' bytes from this same root, so trusting
    the index of that root adds no reachable surface (architecture ruling 15.13).

    The mode is compared as the literal first whitespace-delimited field. Parsing
    it as an int would drop the leading zero and stop matching ``git_mode``.

    Fails closed: if git is absent, refuses the repository, or does not report a
    member, the caller aborts rather than guessing a mode.
    """

    command = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        "ls-files",
        "-s",
        "-z",
        "--",
        *paths,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RegenerationFault("GIT_ORACLE_UNAVAILABLE", str(exc)) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RegenerationFault("GIT_ORACLE_FAILED", detail[0] if detail else "")

    modes: dict[str, str] = {}
    for record in completed.stdout.decode("utf-8").split("\0"):
        if not record:
            continue
        metadata, _, path = record.partition("\t")
        fields = metadata.split()
        if not path or len(fields) != 3:
            raise RegenerationFault("GIT_ORACLE_UNPARSABLE", record[:80])
        # A path listed more than once means an unmerged index; refuse it.
        if path in modes and modes[path] != fields[0]:
            raise RegenerationFault("GIT_INDEX_UNMERGED", path)
        modes[path] = fields[0]
    return modes


def _mode_conflicts(recorded: dict[str, str], indexed: dict[str, str]) -> list[str]:
    """Return members whose git index mode disagrees with the manifest's git_mode.

    The container rejects a mode mismatch on the staged tree
    (``offline_sft_worker.py:438-441``), so catching it here turns a run-time
    failure into a tool-time one.
    """

    conflicts = []
    for path, git_mode in recorded.items():
        if path not in indexed:
            conflicts.append(f"{path} (not tracked by git)")
        elif indexed[path] != git_mode:
            conflicts.append(f"{path} (index {indexed[path]} != manifest {git_mode})")
    return conflicts


def _refresh(root: Path, document: dict) -> dict:
    """Return a copy of the document with the four content-derived values refreshed."""

    members = document.get("members")
    if not isinstance(members, list) or not members:
        raise RegenerationFault("MANIFEST_MEMBERS_INVALID")

    paths = [member.get("path") for member in members]
    if any(not isinstance(path, str) or not path for path in paths):
        raise RegenerationFault("MEMBER_PATH_INVALID")
    if len(set(paths)) != len(paths):
        raise RegenerationFault("MEMBER_PATH_DUPLICATE")
    if paths != sorted(paths):
        raise RegenerationFault("MEMBER_ORDER_INVALID")

    refreshed_members = []
    recorded_modes: dict[str, str] = {}
    total = 0
    for member in members:
        if not isinstance(member, dict):
            raise RegenerationFault("MANIFEST_MEMBERS_INVALID")
        relative = member["path"]
        git_mode = member.get("git_mode")
        if git_mode not in ("100644", "100755"):
            raise RegenerationFault("MEMBER_MODE_INVALID", relative)
        payload = _read_member(root, relative)
        recorded_modes[relative] = git_mode
        entry = dict(member)
        entry["size_bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        total += len(payload)
        refreshed_members.append(entry)

    conflicts = _mode_conflicts(recorded_modes, _git_index_modes(root, paths))
    if conflicts:
        raise RegenerationFault("MEMBER_MODE_CONFLICT", "; ".join(sorted(conflicts)[:5]))

    if total > _MAX_CLOSURE_BYTES:
        raise RegenerationFault("CLOSURE_EXCEEDS_BYTE_BOUND")

    refreshed = dict(document)
    refreshed["members"] = refreshed_members
    refreshed["payload_bytes"] = total
    # The member set never changes here, so member_count passes through; assert
    # the invariant rather than recomputing it, so a drifted count fails closed.
    if refreshed.get("member_count") != len(refreshed_members):
        raise RegenerationFault("MEMBER_COUNT_CONFLICT")
    if [entry["path"] for entry in refreshed_members] != paths:
        raise RegenerationFault("MEMBER_ORDER_INVALID")
    refreshed["closure_digest"] = closure_digest(refreshed)
    return refreshed


def _describe_drift(current: dict, refreshed: dict) -> list[str]:
    """Name every differing field, so --check explains itself without a diff tool."""

    lines: list[str] = []
    for field in ("payload_bytes", "closure_digest"):
        if current.get(field) != refreshed.get(field):
            lines.append(f"  {field}: {current.get(field)} -> {refreshed.get(field)}")
    current_by_path = {
        member.get("path"): member
        for member in current.get("members", [])
        if isinstance(member, dict)
    }
    for entry in refreshed["members"]:
        before = current_by_path.get(entry["path"], {})
        for field in ("size_bytes", "sha256"):
            if before.get(field) != entry[field]:
                lines.append(
                    f"  members[{entry['path']}].{field}: "
                    f"{before.get(field)} -> {entry[field]}"
                )
    return lines


def _verify_written(manifest: Path, expected: bytes) -> None:
    """Re-read and re-parse through the production verifier after writing.

    The generator must never leave a manifest on disk that the worker would
    reject, so this runs the same authority the container runs.
    """

    observed = manifest.read_bytes()
    if observed != expected:
        raise RegenerationFault("VERIFY_AFTER_WRITE_FAILED", "bytes on disk differ")
    document = json.loads(observed.decode("utf-8"))
    parse_offline_sft_worker_manifest(
        observed,
        source_ref=f"file:{_MANIFEST_RELATIVE}",
        manifest_path=manifest,
        expected_digest=document["closure_digest"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh the manifest in place. Without this flag the tool only checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = REPO_ROOT / Path(_MANIFEST_RELATIVE)

    try:
        current_bytes = manifest.read_bytes()
        current = json.loads(current_bytes.decode("utf-8"))
        if not isinstance(current, dict):
            raise RegenerationFault("MANIFEST_NOT_AN_OBJECT")
        refreshed = _refresh(REPO_ROOT, current)
        refreshed_bytes = _serialize(refreshed)
    except RegenerationFault as exc:
        print(
            f"offline SFT worker closure regeneration failed: {exc.reason_code}"
            + (f" ({exc.detail})" if exc.detail else ""),
            file=sys.stderr,
        )
        return 125
    except OfflineSFTWorkerError as exc:
        print(
            f"offline SFT worker closure regeneration failed: MANIFEST_REJECTED ({exc})",
            file=sys.stderr,
        )
        return 125
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"offline SFT worker closure regeneration failed: MANIFEST_UNREADABLE ({exc})",
            file=sys.stderr,
        )
        return 125

    if refreshed_bytes == current_bytes:
        print(
            json.dumps(
                {
                    "status": "CURRENT",
                    "member_count": refreshed["member_count"],
                    "payload_bytes": refreshed["payload_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0

    drift = _describe_drift(current, refreshed)
    if not args.write:
        print("offline SFT worker closure manifest is STALE:", file=sys.stderr)
        for line in drift:
            print(line, file=sys.stderr)
        print(
            "re-run with --write to refresh it.",
            file=sys.stderr,
        )
        return 3

    try:
        manifest.write_bytes(refreshed_bytes)
    except OSError as exc:
        print(
            f"offline SFT worker closure regeneration failed: MANIFEST_UNWRITABLE ({exc})",
            file=sys.stderr,
        )
        return 125

    try:
        _verify_written(manifest, refreshed_bytes)
    except (RegenerationFault, OfflineSFTWorkerError, OSError, ValueError, KeyError) as exc:
        # Never leave a manifest the worker would reject: put the original back.
        try:
            manifest.write_bytes(current_bytes)
            restored = "original restored"
        except OSError:
            restored = "ORIGINAL COULD NOT BE RESTORED"
        print(
            "offline SFT worker closure regeneration failed: "
            f"VERIFY_AFTER_WRITE_FAILED ({exc}; {restored})",
            file=sys.stderr,
        )
        return 125

    print(
        json.dumps(
            {
                "status": "REFRESHED",
                "member_count": refreshed["member_count"],
                "payload_bytes": refreshed["payload_bytes"],
                "closure_digest": refreshed["closure_digest"],
            },
            sort_keys=True,
        )
    )
    for line in drift:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
