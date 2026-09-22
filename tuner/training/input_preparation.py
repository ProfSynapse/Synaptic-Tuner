"""Host-side preparation of private, provider-neutral training inputs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from synaptic_tuner.api.v1.training_facade import TrainingRequest
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1,
    OneUseTrainingInputUploadV1,
    PreparedTrainingInputIdentity,
    PreparedTrainingInputResultV1,
    PreparedTrainingInputV1,
    RetainedPreparedTrainingInputSourceV1,
    TrainingInputSourceV1,
    TrainingNormalizerConfigV1,
    TrainingPreparationConfigV1,
)
from tuner.dataset_prep import (
    CONFIG_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION_V2,
    DatasetPrepConfigV1,
    DatasetPrepConfigV2,
    LocalPreparedTrainingInputSource,
    ROW_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION_V2,
    prepare_dataset_v1,
    prepare_dataset_v2,
    snapshot_prepared_dataset_v1,
    snapshot_prepared_dataset_v2,
)
from tuner.training.contracts import RetainedTrainingInputStreamLease


DATASET_PREP_NORMALIZER_V1 = "synaptic.dataset-prep/v1"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_BUNDLE_MEMBERS = frozenset({"manifest.json", "items.jsonl"})


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError("dataset-prep config contains duplicate fields")
        value[key] = member
    return value


def _reject_nonfinite(_value: str) -> object:
    raise ValueError("dataset-prep config contains a non-finite number")


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def _stable_directory_identity(path: Path, name: str) -> tuple[int, int, int, int, int]:
    identity = _plain_directory_identity(path, name)
    return identity[0], identity[1], identity[2], 0, 0


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _plain_directory_identity(path: Path, name: str) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or _is_reparse(info)
            or path.resolve(strict=True) != path.absolute()
        ):
            raise ValueError
        return _identity(info)
    except Exception:
        raise ValueError(f"{name} must be a stable plain directory") from None


@dataclass(frozen=True, slots=True)
class PrivatePreparedRootAttestationV1:
    path: Path
    identity: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if type(self.path) is not type(Path()) or not self.path.is_absolute():
            raise TypeError("attested private root must be an absolute Path")
        if type(self.identity) is not tuple or len(self.identity) != 5 or any(
            type(value) is not int for value in self.identity
        ):
            raise TypeError("private root identity is invalid")

    def __copy__(self):
        raise TypeError("host-only root attestations are not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("host-only root attestations are not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("host-only root attestations are not serializable")


@runtime_checkable
class PrivatePreparedRootAuthorityV1(Protocol):
    def attest(self, path: Path) -> PrivatePreparedRootAttestationV1: ...

    def verify(self, attestation: PrivatePreparedRootAttestationV1) -> None: ...


class PosixPrivatePreparedRootAuthorityV1:
    """Owner/mode attestor; unsupported on Windows by design."""

    def attest(self, path: Path) -> PrivatePreparedRootAttestationV1:
        if os.name != "posix":
            raise ValueError("POSIX private-root attestation is unavailable")
        if type(path) is not type(Path()) or not path.is_absolute():
            raise TypeError("private root must be an absolute Path")
        absolute = Path(os.path.abspath(path))
        try:
            absolute.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = absolute.lstat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("private root ownership or permissions are unsafe")
        return PrivatePreparedRootAttestationV1(
            absolute, _stable_directory_identity(absolute, "private root")
        )

    def verify(self, attestation: PrivatePreparedRootAttestationV1) -> None:
        if type(attestation) is not PrivatePreparedRootAttestationV1:
            raise TypeError("exact private-root attestation required")
        current = self.attest(attestation.path)
        if current != attestation:
            raise ValueError("private root identity changed")


@dataclass(frozen=True, slots=True)
class DatasetPrepNormalizerConfigV1:
    """Closed path-free adapter for existing dataset-prep v1/v2 documents."""

    canonical_json: str

    def __post_init__(self) -> None:
        if type(self.canonical_json) is not str:
            raise TypeError("dataset-prep config must be exact canonical JSON")
        try:
            document = json.loads(
                self.canonical_json,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite,
            )
            canonical = json.dumps(
                document, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
        except (TypeError, json.JSONDecodeError):
            raise ValueError("dataset-prep config is invalid") from None
        if type(document) is not dict or canonical != self.canonical_json:
            raise ValueError("dataset-prep config must be exact canonical JSON")
        source = document.get("source")
        if type(source) is not dict or set(source) != {"expected_bundle_digest"}:
            raise ValueError("dataset-prep config requires one path-free source digest")
        self._bound(Path(os.path.abspath("__dataset_prep_source__")))

    @property
    def normalizer_ref(self) -> str:
        return DATASET_PREP_NORMALIZER_V1

    def _bound(self, source_path: Path) -> DatasetPrepConfigV1 | DatasetPrepConfigV2:
        document = json.loads(self.canonical_json)
        source = document["source"]
        document["source"] = {
            "bundle_path": str(source_path),
            "expected_bundle_digest": source["expected_bundle_digest"],
        }
        if document.get("schema_version") == CONFIG_SCHEMA_VERSION:
            return DatasetPrepConfigV1.from_dict(document)
        if document.get("schema_version") == CONFIG_SCHEMA_VERSION_V2:
            return DatasetPrepConfigV2.from_dict(document)
        raise ValueError("dataset-prep config schema is unsupported")

    def bind_source(self, source_path: Path) -> DatasetPrepConfigV1 | DatasetPrepConfigV2:
        return self._bound(source_path)


@dataclass(frozen=True, slots=True)
class NormalizedTrainingInputV1:
    """Untrusted normalizer candidate; the service verifies it independently."""

    path: Path
    format: str

    def __post_init__(self) -> None:
        if type(self.path) is not type(Path()) or not self.path.is_absolute():
            raise TypeError("normalized candidate path must be an absolute Path")
        if type(self.format) is not str or not self.format:
            raise ValueError("normalized candidate format is required")

    def __copy__(self):
        raise TypeError("host-only normalized candidates are not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("host-only normalized candidates are not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("host-only normalized candidates are not serializable")


@runtime_checkable
class TrainingInputNormalizerV1(Protocol):
    def prepare(
        self,
        source: TrainingInputSourceV1,
        *,
        config: TrainingNormalizerConfigV1,
        output_root: Path,
    ) -> NormalizedTrainingInputV1: ...


@runtime_checkable
class PreparedInputFormatVerifierV1(Protocol):
    @property
    def format(self) -> str: ...

    def verify(self, path: Path) -> "PreparedInputVerificationV1": ...

    def source(
        self, path: Path, identity: PreparedTrainingInputIdentity
    ) -> RetainedPreparedTrainingInputSourceV1: ...


@dataclass(frozen=True, slots=True)
class PreparedInputVerificationV1:
    """Format-neutral verified publication observation retained by the host."""

    path: Path
    semantic_digest: str
    content_digest: str
    size_bytes: int
    snapshot: bytes

    def __post_init__(self) -> None:
        if type(self.path) is not type(Path()) or not self.path.is_absolute():
            raise TypeError("verified prepared path must be an absolute Path")
        for value, name in (
            (self.semantic_digest, "semantic_digest"),
            (self.content_digest, "content_digest"),
        ):
            if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("verified prepared size must be positive")
        if type(self.snapshot) is not bytes:
            raise TypeError("verified prepared snapshot must be exact bytes")

    def __copy__(self):
        raise TypeError("host-only prepared verifications are not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("host-only prepared verifications are not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("host-only prepared verifications are not serializable")


class DatasetPreparedInputFormatVerifierV1:
    """Current dataset-prep format adapters, isolated from the generic core."""

    __slots__ = ("_format",)

    def __init__(self, format: str) -> None:
        if format not in {ROW_SCHEMA_VERSION, ROW_SCHEMA_VERSION_V2}:
            raise ValueError("dataset-prep prepared format is unsupported")
        self._format = format

    @property
    def format(self) -> str:
        return self._format

    def verify(self, path: Path) -> PreparedInputVerificationV1:
        if self._format == ROW_SCHEMA_VERSION:
            verified, snapshot = snapshot_prepared_dataset_v1(path)
        else:
            verified, snapshot = snapshot_prepared_dataset_v2(path)
        semantic = verified.semantic_identity
        return PreparedInputVerificationV1(
            path=verified.path.resolve(strict=True),
            semantic_digest=semantic.dataset_digest,
            content_digest=semantic.dataset_sha256,
            size_bytes=semantic.dataset_bytes,
            snapshot=snapshot,
        )

    def source(
        self, path: Path, identity: PreparedTrainingInputIdentity
    ) -> LocalPreparedTrainingInputSource:
        if identity.format != self._format:
            raise ValueError("prepared input format differs from verifier")
        return LocalPreparedTrainingInputSource(path, identity)


def default_dataset_format_verifiers_v1() -> dict[str, PreparedInputFormatVerifierV1]:
    return {
        value: DatasetPreparedInputFormatVerifierV1(value)
        for value in (ROW_SCHEMA_VERSION, ROW_SCHEMA_VERSION_V2)
    }


def _bundle_observation(path: Path) -> tuple[object, ...]:
    directory = _plain_directory_identity(path, "normalized bundle")
    members: list[tuple[object, ...]] = []
    try:
        with os.scandir(path) as entries:
            names = tuple(sorted(entry.name for entry in entries))
        if frozenset(names) != _BUNDLE_MEMBERS or len(names) != 2:
            raise ValueError
        for name in names:
            member = path / name
            info = member.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or _is_reparse(info)
                or info.st_nlink != 1
                or member.resolve(strict=True) != member.absolute()
            ):
                raise ValueError
            members.append((name, *_identity(info), info.st_nlink))
    except Exception:
        raise ValueError("normalized bundle contains unsafe or hardlinked members") from None
    if _plain_directory_identity(path, "normalized bundle") != directory:
        raise ValueError("normalized bundle changed during observation")
    return directory, *members


class DatasetPrepTrainingInputNormalizerV1:
    __slots__ = ("_allowed_roots",)

    def __init__(self, *, allowed_source_roots: tuple[Path, ...]) -> None:
        if type(allowed_source_roots) is not tuple or not allowed_source_roots:
            raise ValueError("allowed_source_roots must be a nonempty tuple")
        retained = []
        for root in allowed_source_roots:
            if type(root) is not type(Path()) or not root.is_absolute():
                raise TypeError("allowed source roots must be absolute Paths")
            absolute = Path(os.path.abspath(root))
            retained.append(
                (absolute, _stable_directory_identity(absolute, "allowed source root"))
            )
        self._allowed_roots = tuple(retained)

    def _admit(self, path: Path) -> Path:
        for root, identity in self._allowed_roots:
            if _stable_directory_identity(root, "allowed source root") != identity:
                raise ValueError("allowed source root identity changed")
        absolute = Path(os.path.abspath(path))
        if absolute.resolve(strict=True) != absolute or not any(
            absolute == root or root in absolute.parents for root, _ in self._allowed_roots
        ):
            raise ValueError("training input source is outside stable admitted roots")
        _plain_directory_identity(absolute, "training input source")
        return absolute

    def prepare(
        self,
        source: TrainingInputSourceV1,
        *,
        config: TrainingNormalizerConfigV1,
        output_root: Path,
    ) -> NormalizedTrainingInputV1:
        if type(source) is not LocalTrainingInputPathV1:
            raise TypeError("dataset-prep normalization requires a local bundle path")
        if type(config) is not DatasetPrepNormalizerConfigV1:
            raise TypeError("dataset-prep normalization requires its exact config")
        source_path = self._admit(source.path)
        observed = _bundle_observation(source_path)
        bound = config.bind_source(source_path)
        if type(bound) is DatasetPrepConfigV1:
            prepared = prepare_dataset_v1(bound, output_root)
            format = ROW_SCHEMA_VERSION
        elif type(bound) is DatasetPrepConfigV2:
            prepared = prepare_dataset_v2(bound, output_root)
            format = ROW_SCHEMA_VERSION_V2
        else:  # pragma: no cover - closed config invariant
            raise TypeError("dataset-prep config resolved incorrectly")
        if _bundle_observation(source_path) != observed:
            raise ValueError("normalized bundle changed during preparation")
        self._admit(source_path)
        return NormalizedTrainingInputV1(prepared.path.resolve(strict=True), format)


def _verified_lease_bytes(
    source: RetainedPreparedTrainingInputSourceV1,
    identity: PreparedTrainingInputIdentity,
) -> bytes:
    if source.identity != identity:
        raise ValueError("retained source does not bind the prepared identity")
    lease = source.open_lease()
    if type(lease) is not RetainedTrainingInputStreamLease:
        raise TypeError("format verifier returned an invalid retained source")
    stream = lease.take_stream()
    chunks: list[bytes] = []
    size = 0
    try:
        while size <= identity.size_bytes:
            chunk = stream.read(min(1024 * 1024, identity.size_bytes + 1 - size))
            if type(chunk) is not bytes:
                raise TypeError("prepared source returned invalid bytes")
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        payload = b"".join(chunks)
    finally:
        lease.close()
    if (
        len(payload) != identity.size_bytes
        or hashlib.sha256(payload).hexdigest() != identity.content_digest
    ):
        raise ValueError("prepared source bytes do not bind the prepared identity")
    return payload


class TrainingInputPreparationServiceV1:
    __slots__ = (
        "_format_verifiers", "_normalizers", "_prepared_root", "_root_authority",
    )

    def __init__(
        self,
        *,
        prepared_root: Path,
        normalizers: Mapping[str, TrainingInputNormalizerV1],
        format_verifiers: Mapping[str, PreparedInputFormatVerifierV1],
        root_authority: PrivatePreparedRootAuthorityV1 | None = None,
    ) -> None:
        if type(prepared_root) is not type(Path()) or not prepared_root.is_absolute():
            raise TypeError("prepared_root must be an absolute Path")
        self._prepared_root = Path(os.path.abspath(prepared_root))
        self._root_authority = root_authority
        if root_authority is not None and not isinstance(
            root_authority, PrivatePreparedRootAuthorityV1
        ):
            raise TypeError("root_authority must implement PrivatePreparedRootAuthorityV1")
        self._normalizers = self._registry(normalizers, "normalizer_ref", "normalizers")
        self._format_verifiers = self._registry(
            format_verifiers, "format", "format_verifiers"
        )

    @staticmethod
    def _registry(values: Mapping[str, object], member: str, name: str) -> dict[str, object]:
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"{name} must be a nonempty mapping")
        retained: dict[str, object] = {}
        for key, value in values.items():
            declared = getattr(value, member, key if member == "normalizer_ref" else None)
            if type(key) is not str or not key or (declared is not None and declared != key):
                raise ValueError(f"{name} keys must bind registered identities")
            retained[key] = value
        return retained

    def _attest_root(self) -> PrivatePreparedRootAttestationV1:
        if self._root_authority is None:
            raise ValueError("private prepared-root authority is required")
        attestation = self._root_authority.attest(self._prepared_root)
        if (
            type(attestation) is not PrivatePreparedRootAttestationV1
            or attestation.path != self._prepared_root
        ):
            raise ValueError("private prepared-root attestation is invalid")
        self._root_authority.verify(attestation)
        return attestation

    def prepare(
        self,
        source: TrainingInputSourceV1,
        config: TrainingPreparationConfigV1,
    ) -> PreparedTrainingInputResultV1:
        if type(source) not in (LocalTrainingInputPathV1, OneUseTrainingInputUploadV1):
            raise TypeError("source must be an exact training input source")
        if type(config) is not TrainingPreparationConfigV1:
            raise TypeError("config must be exact TrainingPreparationConfigV1")
        root_attestation = self._attest_root()
        normalizer = self._normalizers.get(config.normalizer_config.normalizer_ref)
        if not isinstance(normalizer, TrainingInputNormalizerV1):
            raise ValueError("training input normalizer is unavailable")
        candidate = normalizer.prepare(
            source,
            config=config.normalizer_config,
            output_root=root_attestation.path,
        )
        if type(candidate) is not NormalizedTrainingInputV1:
            raise TypeError("normalizer returned an invalid candidate")
        PreparedTrainingInputIdentity.validate_format(candidate.format)
        self._root_authority.verify(root_attestation)  # type: ignore[union-attr]
        candidate_path = candidate.path.resolve(strict=True)
        if (
            candidate_path != candidate.path.absolute()
            or candidate_path.parent != root_attestation.path
        ):
            raise ValueError("normalizer candidate is outside the private prepared root")
        verifier = self._format_verifiers.get(candidate.format)
        if not isinstance(verifier, PreparedInputFormatVerifierV1):
            raise ValueError("prepared input format is unsupported")
        verified = verifier.verify(candidate_path)
        if type(verified) is not PreparedInputVerificationV1:
            raise TypeError("format verifier returned an invalid verification")
        self._root_authority.verify(root_attestation)  # type: ignore[union-attr]
        if verified.path.resolve(strict=True) != candidate_path:
            raise ValueError("format verifier substituted the prepared publication")
        if (
            len(verified.snapshot) != verified.size_bytes
            or hashlib.sha256(verified.snapshot).hexdigest() != verified.content_digest
        ):
            raise ValueError("format verifier bytes do not bind its publication")
        identity = PreparedTrainingInputIdentity(
            ref=f"prepared://sha256/{verified.semantic_digest}",
            revision=verified.semantic_digest,
            content_digest=verified.content_digest,
            size_bytes=verified.size_bytes,
            format=verifier.format,
        )
        retained = verifier.source(candidate_path, identity)
        if not isinstance(retained, RetainedPreparedTrainingInputSourceV1):
            raise TypeError("format verifier returned an invalid retained source")
        if _verified_lease_bytes(retained, identity) != verified.snapshot:
            raise ValueError("retained source differs from the verified publication")
        self._root_authority.verify(root_attestation)  # type: ignore[union-attr]
        training_input = TrainingInputV1.from_json(config.canonical_training_json)
        request_document = training_input.to_dict()
        request_document["dataset"] = {"ref": identity.ref}
        canonical_request = json.dumps(
            request_document, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        TrainingInputV1.from_json(canonical_request)
        prepared = PreparedTrainingInputV1(
            identity,
            TrainingRequest(config.request_id, config.project_ref, canonical_request),
        )
        return PreparedTrainingInputResultV1(prepared, retained)


__all__ = [
    "DATASET_PREP_NORMALIZER_V1",
    "DatasetPrepNormalizerConfigV1",
    "DatasetPrepTrainingInputNormalizerV1",
    "DatasetPreparedInputFormatVerifierV1",
    "NormalizedTrainingInputV1",
    "PosixPrivatePreparedRootAuthorityV1",
    "PreparedInputFormatVerifierV1",
    "PreparedInputVerificationV1",
    "PrivatePreparedRootAttestationV1",
    "PrivatePreparedRootAuthorityV1",
    "TrainingInputNormalizerV1",
    "TrainingInputPreparationServiceV1",
    "default_dataset_format_verifiers_v1",
]
