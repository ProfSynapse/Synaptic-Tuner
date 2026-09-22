"""Process-local, one-use admission of explicitly selected local sources.

The snapshotter deliberately keeps host paths and source bytes out of the public
ingestion facade.  Directory traversal is not descriptor-anchored on every
supported Python platform.  It therefore fails closed on symlinks/reparse
points and on identity changes observed before and after traversal/read.  On
Windows, portable Python cannot eliminate ancestor replace-and-restore between
those checks by any actor with directory write access, including a process
running under the same account. This is an observational snapshot, not an atomic
filesystem transaction; no observed substitution is accepted.

Directory identity is structural: ordinary child activity does not replace its
parent. Two bounded discovery passes and stable content reads bind the admitted
file paths, versions and content digests, while retained relevant ancestors bind
the paths used for reading those files.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock

from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.ingestion_facade import (
    AuthorizedSourceRef,
    SourceAdmissionKind,
    SourceMatcher,
)


MAX_SELECTION_ROOTS = 32
MAX_ADMITTED_FILES = 10_000
DEFAULT_MAX_MEMBER_BYTES = 256 * 1024
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024
HARD_MAX_MEMBER_BYTES = 16 * 1024 * 1024
HARD_MAX_TOTAL_BYTES = 32 * 1024 * 1024
# Compatibility aliases for callers that imported the original fixed budgets.
MAX_FILE_BYTES = DEFAULT_MAX_MEMBER_BYTES
MAX_TOTAL_BYTES = DEFAULT_MAX_TOTAL_BYTES
MAX_LOGICAL_PATH_BYTES = 1_024
MAX_LOGICAL_PATH_SEGMENTS = 64
MAX_DISCOVERY_ENTRIES = 100_000
_MAX_PATTERNS = 128
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class LocalSelectionCodeV1(str, Enum):
    INVALID_SELECTION = "invalid_selection"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    SOURCE_UNSAFE = "source_unsafe"
    SOURCE_CHANGED = "source_changed"
    LIMIT_EXCEEDED = "limit_exceeded"


class LocalSelectionErrorV1(ValueError):
    """Closed failure from local selection authorization or consumption."""

    __slots__ = ("code",)

    def __init__(self, code: LocalSelectionCodeV1) -> None:
        if type(code) is not LocalSelectionCodeV1:
            raise TypeError("code must be exact LocalSelectionCodeV1")
        self.code = code
        super().__init__(code.value)


def _raise(code: LocalSelectionCodeV1) -> None:
    raise LocalSelectionErrorV1(code) from None


def _is_reparse(value: os.stat_result) -> bool:
    return bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    # Directory size/mtime describe child namespace activity, not replacement.
    # The admitted file inventory is compared separately after all reads.
    return (value.st_dev, value.st_ino, value.st_mode)


def _safe_kind(value: os.stat_result) -> str:
    if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
    if stat.S_ISREG(value.st_mode):
        return "file"
    if stat.S_ISDIR(value.st_mode):
        return "directory"
    _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)


def _logical_segment(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty exact string")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise ValueError(f"{name} must be NFC normalized")
    if value in {".", ".."} or any(
        character in "/\\:" or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError(f"{name} is not a safe logical path segment")
    return value


def _logical_path(segments: tuple[str, ...]) -> str:
    if not 1 <= len(segments) <= MAX_LOGICAL_PATH_SEGMENTS:
        _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
    normalized = tuple(unicodedata.normalize("NFC", item) for item in segments)
    if normalized != segments or any(
        not item
        or item in {".", ".."}
        or any(
            character in "/\\:" or ord(character) < 32 or ord(character) == 127
            for character in item
        )
        for item in normalized
    ):
        _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
    value = "/".join(normalized)
    if len(value.encode("utf-8")) > MAX_LOGICAL_PATH_BYTES:
        _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
    return value


@dataclass(frozen=True, slots=True)
class LocalSelectionRootV1:
    """One selected file or directory under a caller-chosen logical alias.

    A file alias is its complete logical path.  A directory alias is the first
    logical path segment for every descendant.
    """

    alias: str
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", _logical_segment(self.alias, name="alias"))
        if type(self.path) is not type(Path()):
            raise TypeError("path must be the platform's exact concrete Path type")


@dataclass(frozen=True, slots=True)
class AdmissionLimitsV1:
    """Caller-selected source-admission budgets under fixed safety ceilings."""

    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES

    def __post_init__(self) -> None:
        if (
            type(self.max_member_bytes) is not int
            or type(self.max_total_bytes) is not int
        ):
            raise TypeError("admission limits must be exact integers")
        if not 1 <= self.max_member_bytes <= HARD_MAX_MEMBER_BYTES:
            raise ValueError("max_member_bytes is outside the supported range")
        if not 1 <= self.max_total_bytes <= HARD_MAX_TOTAL_BYTES:
            raise ValueError("max_total_bytes is outside the supported range")
        if self.max_member_bytes > self.max_total_bytes:
            raise ValueError("max_member_bytes must not exceed max_total_bytes")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_member_bytes": self.max_member_bytes,
            "max_total_bytes": self.max_total_bytes,
        }


@dataclass(frozen=True, slots=True)
class LocalDiscoveryPolicyV1:
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    include_hidden: bool = False

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("include", self.include, 1),
            ("exclude", self.exclude, 0),
        ):
            if type(value) is not tuple or any(type(item) is not str for item in value):
                raise TypeError(f"{name} must be an exact tuple of strings")
            if not minimum <= len(value) <= _MAX_PATTERNS:
                raise ValueError(f"{name} has an invalid number of patterns")
            for item in value:
                _validated_pattern(item)
            if len(value) != len(set(value)):
                raise ValueError(f"{name} patterns must be unique")
        if type(self.include_hidden) is not bool:
            raise TypeError("include_hidden must be an exact boolean")


@dataclass(frozen=True, slots=True)
class SnapshotEntryV1:
    logical_path: str
    content: bytes
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.logical_path) is not str or self.logical_path != _logical_path(
            tuple(self.logical_path.split("/"))
        ):
            raise ValueError("logical_path is invalid")
        if type(self.content) is not bytes:
            raise TypeError("content must be exact bytes")
        if type(self.size_bytes) is not int or self.size_bytes != len(self.content):
            raise ValueError("size_bytes must bind content")
        if self.size_bytes > HARD_MAX_MEMBER_BYTES:
            raise ValueError("content exceeds the hard member ceiling")
        expected = hashlib.sha256(self.content).hexdigest()
        if type(self.sha256) is not str or self.sha256 != expected:
            raise ValueError("sha256 must bind content")


@dataclass(frozen=True, slots=True)
class AdmissionReportV1:
    root_count: int
    discovered_file_count: int
    admitted_file_count: int
    excluded_file_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.root_count,
            self.discovered_file_count,
            self.admitted_file_count,
            self.excluded_file_count,
            self.total_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(
                "admission report counts must be nonnegative exact integers"
            )
        if not 1 <= self.root_count <= MAX_SELECTION_ROOTS:
            raise ValueError("root_count is invalid")
        if (
            self.discovered_file_count
            != self.admitted_file_count + self.excluded_file_count
        ):
            raise ValueError("admission report file counts are inconsistent")
        if (
            self.admitted_file_count > MAX_ADMITTED_FILES
            or self.total_bytes > HARD_MAX_TOTAL_BYTES
        ):
            raise ValueError("admission report exceeds snapshot limits")


@dataclass(frozen=True, slots=True)
class ImmutableLocalSnapshotV1:
    project_ref: str
    source_ref: str
    entries: tuple[SnapshotEntryV1, ...]
    report: AdmissionReportV1
    manifest_digest: str

    def __post_init__(self) -> None:
        if type(self.project_ref) is not str or type(self.source_ref) is not str:
            raise TypeError("snapshot references must be exact strings")
        if type(self.entries) is not tuple or any(
            type(item) is not SnapshotEntryV1 for item in self.entries
        ):
            raise TypeError("entries must be an exact tuple of SnapshotEntryV1 values")
        if type(self.report) is not AdmissionReportV1:
            raise TypeError("report must be exact AdmissionReportV1")
        entries = tuple(
            SnapshotEntryV1(
                item.logical_path, item.content, item.size_bytes, item.sha256
            )
            for item in self.entries
        )
        report = AdmissionReportV1(
            self.report.root_count,
            self.report.discovered_file_count,
            self.report.admitted_file_count,
            self.report.excluded_file_count,
            self.report.total_bytes,
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "report", report)
        paths = tuple(item.logical_path for item in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("snapshot entries must be unique and sorted")
        if len(self.entries) != self.report.admitted_file_count:
            raise ValueError("snapshot entries do not bind the admission report")
        if sum(item.size_bytes for item in self.entries) != self.report.total_bytes:
            raise ValueError("snapshot bytes do not bind the admission report")
        expected = _manifest_digest(self.entries)
        if type(self.manifest_digest) is not str or self.manifest_digest != expected:
            raise ValueError("manifest_digest must bind entries")


@dataclass(frozen=True, slots=True)
class _AuthorizedRootV1:
    alias: str
    path: Path
    kind: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _AuthorizedSelectionV1:
    project_ref: str
    source_ref: str
    roots: tuple[_AuthorizedRootV1, ...]
    policy: LocalDiscoveryPolicyV1
    admission_limits: AdmissionLimitsV1


@dataclass(frozen=True, slots=True)
class _RetainedPathV1:
    path: Path
    identity: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _CandidateV1:
    logical_path: str
    path: Path
    identity: tuple[int, int, int, int, int]
    ancestors: tuple[_RetainedPathV1, ...]


def _manifest_digest(entries: tuple[SnapshotEntryV1, ...]) -> str:
    document = [
        {
            "logical_path": item.logical_path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in entries
    ]
    return hashlib.sha256(
        b"synaptic-local-source-manifest/v1\0" + canonical_bytes({"entries": document})
    ).hexdigest()


def _segment_matches(pattern: str, value: str) -> bool:
    previous = [True] + [False] * len(value)
    for token in pattern:
        current = [False] * (len(value) + 1)
        if token == "*":
            current[0] = previous[0]
            for index in range(1, len(value) + 1):
                current[index] = previous[index] or current[index - 1]
        elif token == "?":
            for index in range(1, len(value) + 1):
                current[index] = previous[index - 1]
        else:
            for index in range(1, len(value) + 1):
                current[index] = previous[index - 1] and value[index - 1] == token
        previous = current
    return previous[-1]


def _validated_pattern(pattern: str) -> str:
    validated = SourceMatcher(pattern).pattern
    if any(ord(character) < 32 or ord(character) == 127 for character in validated):
        raise ValueError("pattern contains a control character")
    return validated


def glob_matches_v1(pattern: str, logical_path: str) -> bool:
    """Match the facade's exact whole-path, case-sensitive glob language."""

    validated = _validated_pattern(pattern)
    if type(logical_path) is not str:
        raise TypeError("logical_path must be an exact string")
    if unicodedata.normalize("NFC", logical_path) != logical_path:
        raise ValueError("logical_path must be NFC normalized")
    path_segments = logical_path.split("/")
    if (
        logical_path.startswith("/")
        or logical_path.endswith("/")
        or any(not item or item in {".", ".."} for item in path_segments)
    ):
        raise ValueError("logical_path must be a relative POSIX path")
    if logical_path != _logical_path(tuple(path_segments)):
        raise ValueError("logical_path must be a bounded relative POSIX path")
    pattern_segments = validated.split("/")
    rows = len(pattern_segments) + 1
    columns = len(path_segments) + 1
    table = [[False] * columns for _ in range(rows)]
    table[0][0] = True
    for row in range(1, rows):
        if pattern_segments[row - 1] == "**":
            table[row][0] = table[row - 1][0]
        for column in range(1, columns):
            segment = pattern_segments[row - 1]
            if segment == "**":
                table[row][column] = table[row - 1][column] or table[row][column - 1]
            else:
                table[row][column] = table[row - 1][column - 1] and _segment_matches(
                    segment, path_segments[column - 1]
                )
    return table[-1][-1]


def _stable_regular_read(
    path: Path,
    expected_identity: tuple[int, int, int, int, int],
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> bytes:
    try:
        declared = path.lstat()
        if _safe_kind(declared) != "file":
            _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
        if _identity(declared) != expected_identity:
            _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
        if declared.st_size > max_member_bytes:
            _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
        if path.resolve(strict=True) != path.absolute():
            _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if _safe_kind(before) != "file":
                _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
            chunks: list[bytes] = []
            size = 0
            while size <= max_member_bytes:
                chunk = os.read(descriptor, max_member_bytes + 1 - size)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final = path.lstat()
    except LocalSelectionErrorV1:
        raise
    except BaseException:
        _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
    identities = tuple(_identity(item) for item in (declared, before, after, final))
    if any(item != identities[0] for item in identities[1:]):
        _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
    if len(payload) > max_member_bytes:
        _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
    if len(payload) != before.st_size:
        _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
    return payload


def _prepare_root(root: LocalSelectionRootV1) -> _AuthorizedRootV1:
    try:
        absolute = root.path.absolute()
        declared = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except BaseException:
        _raise(LocalSelectionCodeV1.INVALID_SELECTION)
    if resolved != absolute or _is_reparse(declared) or stat.S_ISLNK(declared.st_mode):
        _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
    return _AuthorizedRootV1(
        root.alias, resolved, _safe_kind(declared), _identity(declared)
    )


def _is_hidden(segments: tuple[str, ...]) -> bool:
    return any(segment.startswith(".") for segment in segments)


def _revalidate_retained(paths: tuple[_RetainedPathV1, ...]) -> None:
    try:
        for retained in paths:
            current = retained.path.lstat()
            if (
                _safe_kind(current) != "directory"
                or _directory_identity(current) != retained.identity
                or retained.path.resolve(strict=True) != retained.path.absolute()
            ):
                _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
    except LocalSelectionErrorV1:
        raise
    except BaseException:
        _raise(LocalSelectionCodeV1.SOURCE_CHANGED)


def _discover(selection: _AuthorizedSelectionV1) -> tuple[_CandidateV1, ...]:
    """One bounded discovery pass; retain no bytes from nonmatching files."""
    candidates: list[_CandidateV1] = []
    seen: set[str] = set()
    observed_entries = 0

    def add_candidate(
        path: Path,
        segments: tuple[str, ...],
        identity: tuple[int, int, int, int, int],
        ancestors: tuple[_RetainedPathV1, ...],
    ) -> None:
        logical = _logical_path(segments)
        if logical in seen:
            _raise(LocalSelectionCodeV1.SOURCE_UNSAFE)
        seen.add(logical)
        candidates.append(_CandidateV1(logical, path, identity, ancestors))

    def walk(
        path: Path,
        logical_prefix: tuple[str, ...],
        ancestors: tuple[_RetainedPathV1, ...],
        expected_identity: tuple[int, int, int],
    ) -> None:
        nonlocal observed_entries
        try:
            before = path.lstat()
            if (
                _safe_kind(before) != "directory"
                or _directory_identity(before) != expected_identity
                or path.resolve(strict=True) != path.absolute()
            ):
                _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
            retained = _RetainedPathV1(path, _directory_identity(before))
            chain = ancestors + (retained,)
            children: list[
                tuple[str, str, Path, str, tuple[int, int, int, int, int]]
            ] = []
            with os.scandir(path) as iterator:
                for child in iterator:
                    observed_entries += 1
                    if observed_entries > MAX_DISCOVERY_ENTRIES:
                        _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
                    child_path = path / child.name
                    declared = child_path.lstat()
                    kind = _safe_kind(declared)
                    normalized_name = unicodedata.normalize("NFC", child.name)
                    _logical_segment(normalized_name, name="source path segment")
                    relative = logical_prefix + (normalized_name,)
                    _logical_path(relative)
                    children.append(
                        (
                            normalized_name,
                            child.name,
                            child_path,
                            kind,
                            _identity(declared),
                        )
                    )
            children.sort(key=lambda item: (item[0], item[1]))
            for (
                normalized_name,
                _raw_name,
                child_path,
                kind,
                child_identity,
            ) in children:
                relative = logical_prefix + (normalized_name,)
                if not selection.policy.include_hidden and _is_hidden(relative[1:]):
                    continue
                if kind == "directory":
                    walk(child_path, relative, chain, child_identity[:3])
                else:
                    add_candidate(child_path, relative, child_identity, chain)
            after = path.lstat()
        except LocalSelectionErrorV1:
            raise
        except BaseException:
            _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
        if (
            _safe_kind(after) != "directory"
            or _directory_identity(before) != _directory_identity(after)
        ):
            _raise(LocalSelectionCodeV1.SOURCE_CHANGED)

    for root in selection.roots:
        try:
            current = root.path.lstat()
        except BaseException:
            _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
        current_identity = (
            _identity(current) if root.kind == "file" else _directory_identity(current)
        )
        expected_identity = root.identity if root.kind == "file" else root.identity[:3]
        if current_identity != expected_identity or _safe_kind(current) != root.kind:
            _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
        if root.kind == "file":
            observed_entries += 1
            if observed_entries > MAX_DISCOVERY_ENTRIES:
                _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
            add_candidate(root.path, (root.alias,), root.identity, ())
        else:
            walk(root.path, (root.alias,), (), root.identity[:3])

    candidates.sort(key=lambda item: item.logical_path)
    return tuple(candidates)


def _admitted_candidates(
    candidates: tuple[_CandidateV1, ...], policy: LocalDiscoveryPolicyV1
) -> tuple[_CandidateV1, ...]:
    admitted: list[_CandidateV1] = []
    for candidate in candidates:
        included = any(
            glob_matches_v1(item, candidate.logical_path) for item in policy.include
        )
        denied = any(
            glob_matches_v1(item, candidate.logical_path) for item in policy.exclude
        )
        if included and not denied:
            if len(admitted) >= MAX_ADMITTED_FILES:
                _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
            admitted.append(candidate)
    return tuple(admitted)


def _inventory(selection: _AuthorizedSelectionV1) -> ImmutableLocalSnapshotV1:
    candidates = _discover(selection)
    admitted = _admitted_candidates(candidates, selection.policy)
    # Only ancestors of admitted files need to remain stable across the entire
    # snapshot. Every traversed entry is still checked for unsafe path types.
    retained_directories = {
        ancestor.path: ancestor
        for candidate in admitted
        for ancestor in candidate.ancestors
    }
    entries: list[SnapshotEntryV1] = []
    total = 0
    for candidate in admitted:
        logical_path = candidate.logical_path
        _revalidate_retained(candidate.ancestors)
        payload = _stable_regular_read(
            candidate.path,
            candidate.identity,
            selection.admission_limits.max_member_bytes,
        )
        _revalidate_retained(candidate.ancestors)
        total += len(payload)
        if total > selection.admission_limits.max_total_bytes:
            _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
        entries.append(
            SnapshotEntryV1(
                logical_path,
                payload,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        )
    # A structural directory identity cannot detect matching children added or
    # removed. Rediscover under the same budgets and compare the exact admitted
    # logical-path -> file-version inventory, including earlier-read files.
    final_admitted = _admitted_candidates(_discover(selection), selection.policy)
    if tuple((item.logical_path, item.identity) for item in admitted) != tuple(
        (item.logical_path, item.identity) for item in final_admitted
    ):
        _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
    # Size/mtime can be restored after an in-place rewrite. Bind the second
    # observation to captured content as well as metadata, using the same
    # bounded, link-rejecting descriptor read and ancestor checks. Inventory
    # equality above also binds the file count and declared aggregate size.
    verified_total = 0
    for candidate, captured in zip(final_admitted, entries, strict=True):
        _revalidate_retained(candidate.ancestors)
        payload = _stable_regular_read(
            candidate.path,
            candidate.identity,
            selection.admission_limits.max_member_bytes,
        )
        _revalidate_retained(candidate.ancestors)
        verified_total += len(payload)
        if verified_total > selection.admission_limits.max_total_bytes:
            _raise(LocalSelectionCodeV1.LIMIT_EXCEEDED)
        if (
            len(payload) != captured.size_bytes
            or hashlib.sha256(payload).hexdigest() != captured.sha256
        ):
            _raise(LocalSelectionCodeV1.SOURCE_CHANGED)
    _revalidate_retained(tuple(retained_directories.values()))
    if not entries:
        _raise(LocalSelectionCodeV1.INVALID_SELECTION)
    frozen_entries = tuple(entries)
    report = AdmissionReportV1(
        len(selection.roots),
        len(candidates),
        len(entries),
        len(candidates) - len(admitted),
        total,
    )
    return ImmutableLocalSnapshotV1(
        selection.project_ref,
        selection.source_ref,
        frozen_entries,
        report,
        _manifest_digest(frozen_entries),
    )


class ProcessLocalSelectionRegistryV1:
    """In-memory authority registry; entries are consumed exactly once."""

    __slots__ = ("_authorizations", "_lock", "_secret")

    def __init__(self) -> None:
        self._authorizations: dict[str, _AuthorizedSelectionV1] = {}
        self._lock = Lock()
        self._secret = secrets.token_bytes(32)

    def authorize(
        self,
        project_ref: str,
        roots: tuple[LocalSelectionRootV1, ...],
        policy: LocalDiscoveryPolicyV1,
        admission_limits: AdmissionLimitsV1 | None = None,
    ) -> AuthorizedSourceRef:
        if type(roots) is not tuple or any(
            type(item) is not LocalSelectionRootV1 for item in roots
        ):
            raise TypeError(
                "roots must be an exact tuple of LocalSelectionRootV1 values"
            )
        if not 1 <= len(roots) <= MAX_SELECTION_ROOTS:
            raise ValueError("roots requires 1 through 32 entries")
        if type(policy) is not LocalDiscoveryPolicyV1:
            raise TypeError("policy must be exact LocalDiscoveryPolicyV1")
        policy = LocalDiscoveryPolicyV1(
            policy.include, policy.exclude, policy.include_hidden
        )
        if (
            admission_limits is not None
            and type(admission_limits) is not AdmissionLimitsV1
        ):
            raise TypeError("admission_limits must be exact AdmissionLimitsV1 or None")
        effective_limits = (
            AdmissionLimitsV1()
            if admission_limits is None
            else AdmissionLimitsV1(
                admission_limits.max_member_bytes,
                admission_limits.max_total_bytes,
            )
        )
        aliases = tuple(item.alias for item in roots)
        if len(aliases) != len(set(aliases)):
            raise ValueError("root aliases must be unique")
        prepared = tuple(_prepare_root(item) for item in roots)
        with self._lock:
            source_ref = "local_" + secrets.token_hex(24)
            while source_ref in self._authorizations:
                source_ref = "local_" + secrets.token_hex(24)
            authority_document = {
                "project_ref": project_ref,
                "source_ref": source_ref,
                "roots": [
                    {
                        "alias": item.alias,
                        "path": str(item.path),
                        "identity": list(item.identity),
                    }
                    for item in prepared
                ],
                "policy": {
                    "include": list(policy.include),
                    "exclude": list(policy.exclude),
                    "include_hidden": policy.include_hidden,
                },
            }
            authority_domain = b"synaptic-process-local-selection-authority/v1\0"
            if admission_limits is not None:
                authority_document["admission_limits"] = effective_limits.to_dict()
                authority_domain = b"synaptic-process-local-selection-authority/v2\0"
            authority_digest = hashlib.sha256(
                authority_domain + self._secret + canonical_bytes(authority_document)
            ).hexdigest()
            reference = AuthorizedSourceRef(
                project_ref,
                SourceAdmissionKind.LOCAL_SELECTION,
                source_ref,
                authority_digest,
            )
            self._authorizations[source_ref] = _AuthorizedSelectionV1(
                reference.project_ref,
                source_ref,
                prepared,
                policy,
                effective_limits,
            )
        return reference

    def consume_snapshot(self, source_ref: str) -> ImmutableLocalSnapshotV1:
        if type(source_ref) is not str:
            raise TypeError("source_ref must be an exact string")
        with self._lock:
            selection = self._authorizations.pop(source_ref, None)
        if selection is None:
            _raise(LocalSelectionCodeV1.AUTHORITY_UNAVAILABLE)
        return _inventory(selection)


__all__ = [
    "AdmissionLimitsV1",
    "AdmissionReportV1",
    "DEFAULT_MAX_MEMBER_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "HARD_MAX_MEMBER_BYTES",
    "HARD_MAX_TOTAL_BYTES",
    "ImmutableLocalSnapshotV1",
    "LocalDiscoveryPolicyV1",
    "LocalSelectionCodeV1",
    "LocalSelectionErrorV1",
    "LocalSelectionRootV1",
    "MAX_ADMITTED_FILES",
    "MAX_FILE_BYTES",
    "MAX_DISCOVERY_ENTRIES",
    "MAX_LOGICAL_PATH_BYTES",
    "MAX_LOGICAL_PATH_SEGMENTS",
    "MAX_SELECTION_ROOTS",
    "MAX_TOTAL_BYTES",
    "ProcessLocalSelectionRegistryV1",
    "SnapshotEntryV1",
    "glob_matches_v1",
]
