"""Protected, descriptor-only Hugging Face source provisioning operator."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from tuner.cloud.hf_provider_adapter import HFProviderAdapter, HFRemoteMember
from tuner.cloud.hf_provisioning import (
    EVIDENCE_SCHEMA_VERSION,
    HFPreparedSourceTransport,
    canonical_json_bytes,
    document_sha256,
    load_hf_source_transport,
    validate_hf_evidence_binding,
)
from tuner.core.exceptions import CloudProviderError
from tuner.project import ProjectContext


_MEMBERS = (
    "checkout-policy.json",
    "source-lock.json",
    "capsule/synaptic-bootstrap-capsule.json",
    "capsule/tuner/cloud/bootstrap_capsule.py",
    "capsule/tuner/cloud/bootstrap_core.py",
)
_MAX_MEMBER_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class HFProvisioningFailure:
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class HFProvisioningOutcome:
    evidence: Mapping[str, object] | None = None
    evidence_sha256: str | None = None
    mutated: bool = False
    failure: HFProvisioningFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.evidence is not None and self.failure is None


def provision_hf_source_transport(
    context: ProjectContext,
    *,
    transport_root: Path,
    descriptor_uri: str,
    source_lock_uri: str,
    provider: HFProviderAdapter,
    actor: str,
    authority: str = "operator",
    asserted_at: datetime | None = None,
) -> HFProvisioningOutcome:
    """Provision or verify one exact immutable descriptor without submission."""

    prepared = load_hf_source_transport(
        context,
        transport_root=Path(transport_root),
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
    )
    local = _local_members(prepared)
    descriptor = prepared.descriptor
    volume = _mapping(descriptor["volume"])
    bucket_id = str(volume["source"])
    prefix = str(volume["path"])
    evidence = _build_evidence(
        prepared,
        actor=actor,
        authority=authority,
        asserted_at=asserted_at,
    )

    # This probe covers all reads and mutations and must precede create_bucket.
    provider.probe_signatures()
    try:
        canonical_bucket = provider.ensure_private_bucket(bucket_id)
    except Exception:
        # create_bucket(exist_ok=True) can still have created the bucket before
        # a response/readback failure. Its outcome is therefore mutation-ambiguous.
        return _ambiguous_outcome()
    remote = provider.list_members(canonical_bucket, prefix=prefix)
    expected_paths = {
        f"{prefix}/{member}": digest for member, (_content, digest) in local.items()
    }

    if remote:
        _require_exact_remote_paths(remote, expected_paths, prefix=prefix)
        _verify_remote_bytes(provider, canonical_bucket, remote, expected_paths)
        return HFProvisioningOutcome(
            evidence=evidence,
            evidence_sha256=document_sha256(evidence),
            mutated=False,
        )

    additions = [(local[member][0], f"{prefix}/{member}") for member in _MEMBERS]
    for content, destination in additions:
        relative = destination.removeprefix(f"{prefix}/")
        if hashlib.sha256(content).hexdigest() != local[relative][1]:
            raise CloudProviderError("HF JP authenticated upload bytes changed before handoff.")
    try:
        # HF documents batch_bucket_files as non-transactional. From this call
        # onward every error is ambiguous and must never trigger an automatic retry.
        provider.upload_once(canonical_bucket, additions=additions)
        uploaded = provider.list_members(canonical_bucket, prefix=prefix)
        _require_exact_remote_paths(uploaded, expected_paths, prefix=prefix)
        _verify_remote_bytes(provider, canonical_bucket, uploaded, expected_paths)
    except Exception:
        return _ambiguous_outcome()

    return HFProvisioningOutcome(
        evidence=evidence,
        evidence_sha256=document_sha256(evidence),
        mutated=True,
    )


def _ambiguous_outcome() -> HFProvisioningOutcome:
    return HFProvisioningOutcome(
        mutated=True,
        failure=HFProvisioningFailure(
            code="mutation_ambiguous",
            message=(
                "HF JP provider mutation outcome is ambiguous; inspect the exact immutable "
                "target read-only and do not retry automatically."
            ),
            retryable=False,
        ),
    )


def _local_members(prepared: HFPreparedSourceTransport) -> dict[str, tuple[bytes, str]]:
    result: dict[str, tuple[bytes, str]] = {}
    for member in _MEMBERS:
        path = prepared.bundle_root.joinpath(*member.split("/"))
        content = _read_regular(path)
        result[member] = (content, hashlib.sha256(content).hexdigest())
    inventory = [
        {"path": member, "sha256": result[member][1]}
        for member in sorted(result)
    ]
    actual = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    expected = str(_mapping(prepared.descriptor["bundle"])["content_sha256"])
    if actual != expected:
        raise CloudProviderError("HF JP local bundle digest does not match the descriptor.")
    return result


def _require_exact_remote_paths(
    remote: tuple[HFRemoteMember, ...], expected: Mapping[str, str], *, prefix: str
) -> None:
    paths = [item.path for item in remote]
    if len(paths) != len(set(paths)):
        raise CloudProviderError(
            "HF JP immutable prefix contains duplicate or colliding tree entries."
        )
    expected_files = set(expected)
    expected_directories: set[str] = {prefix}
    for file_path in expected_files:
        relative = file_path.removeprefix(f"{prefix}/")
        parts = relative.split("/")
        expected_directories.update(
            f"{prefix}/{'/'.join(parts[:index])}"
            for index in range(1, len(parts))
        )
    actual_files = {item.path for item in remote if item.entry_type == "file"}
    actual_directories = {
        item.path for item in remote if item.entry_type == "directory"
    }
    if actual_files != expected_files or not actual_directories <= expected_directories:
        raise CloudProviderError(
            "HF JP immutable prefix is non-empty and not exactly descriptor-identical."
        )


def _verify_remote_bytes(
    provider: HFProviderAdapter,
    bucket_id: str,
    remote: tuple[HFRemoteMember, ...],
    expected: Mapping[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="synaptic-hf-jp-readback-") as directory:
        root = Path(directory)
        downloads: list[tuple[object, Path]] = []
        local_by_remote: dict[str, Path] = {}
        files = tuple(member for member in remote if member.entry_type == "file")
        for index, member in enumerate(files):
            local = root / f"member-{index}"
            downloads.append((member.provider_object, local))
            local_by_remote[member.path] = local
        provider.download_members(bucket_id, files=downloads)
        for remote_path, expected_digest in expected.items():
            actual = hashlib.sha256(_read_regular(local_by_remote[remote_path])).hexdigest()
            if actual != expected_digest:
                raise CloudProviderError("HF JP remote member digest does not match the descriptor.")


def _build_evidence(
    prepared: HFPreparedSourceTransport,
    *,
    actor: str,
    authority: str,
    asserted_at: datetime | None,
) -> dict[str, object]:
    descriptor = prepared.descriptor
    volume = _mapping(descriptor["volume"])
    capsule = _mapping(descriptor["capsule"])
    timestamp = asserted_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise CloudProviderError("HF JP evidence timestamp must include a timezone.")
    bundle_digest = str(_mapping(descriptor["bundle"])["content_sha256"])
    evidence: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "descriptor": {
            "uri": prepared.descriptor_uri,
            "sha256": prepared.descriptor_sha256,
        },
        "run_id": descriptor["run_id"],
        "provider": descriptor["provider"],
        "profile": descriptor["profile"],
        "volume": {
            "source": volume["source"],
            "path": volume["path"],
            "type": volume["type"],
            "read_only": volume["read_only"],
        },
        "bundle_sha256": bundle_digest,
        "capsule_manifest_sha256": _mapping(capsule["manifest"])["sha256"],
        "source_lock_sha256": _mapping(descriptor["source_lock"])["sha256"],
        "checkout_policy_sha256": _mapping(descriptor["checkout_policy"])["sha256"],
        "status": "provisioned",
        "authority": authority,
        "actor": actor,
        "asserted_at": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provider_receipt_id": f"hf-prefix-{bundle_digest}",
    }
    return validate_hf_evidence_binding(
        descriptor,
        evidence,
        descriptor_uri=prepared.descriptor_uri,
        descriptor_sha256=prepared.descriptor_sha256,
    )


def _read_regular(path: Path) -> bytes:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & 0x400
            or before.st_size > _MAX_MEMBER_BYTES
        ):
            raise CloudProviderError("HF JP member must be a bounded regular file.")
        with path.open("rb") as handle:
            content = handle.read(_MAX_MEMBER_BYTES + 1)
            opened = os.fstat(handle.fileno())
        after = path.lstat()
    except CloudProviderError:
        raise
    except OSError:
        raise CloudProviderError("HF JP member is unavailable.") from None
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if len(content) > _MAX_MEMBER_BYTES or identity(before) != identity(opened) or identity(before) != identity(after):
        raise CloudProviderError("HF JP member changed during verification.")
    return content


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CloudProviderError("HF JP descriptor contains an invalid object.")
    return value


__all__ = [
    "HFProvisioningFailure",
    "HFProvisioningOutcome",
    "provision_hf_source_transport",
]
