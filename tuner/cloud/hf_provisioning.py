"""Pure immutable provisioning contract for Hugging Face source transport.

Preparation and consumption in this module are local, deterministic, and
provider-effect free.  External provisioning and paid submission are separate
protected operations and are intentionally not implemented here.
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote

from jsonschema import Draft202012Validator, FormatChecker

from tuner.cloud.bootstrap_capsule import CAPSULE_MANIFEST, CapsuleError, build_capsule
from tuner.cloud.checkout import (
    CheckoutPolicy,
    SSHCheckoutPolicy,
    _policy_mapping,
    checkout_policy_from_context,
)
from tuner.cloud.hf_volume_transport import (
    HFVerifiedVolume,
    HFVerifiedVolumeSpec,
    validate_read_only_volume_object,
)
from tuner.core.exceptions import CloudProviderError
from tuner.project import ProjectContext
from tuner.project.source_bundle import SourceLock


DESCRIPTOR_SCHEMA_VERSION = "synaptic-hf-source-transport/v1"
EVIDENCE_SCHEMA_VERSION = "synaptic-hf-provisioning-evidence/v1"
PROFILE = "C"
PROVIDER = "hf_jobs"
MOUNT_PATH = "/workspace/synaptic-bootstrap-input"
DESCRIPTOR_FILENAME = "descriptor.json"
BUNDLE_DIRECTORY = "bundle"
SOURCE_LOCK_FILENAME = "source-lock.json"
CHECKOUT_POLICY_FILENAME = "checkout-policy.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SECRET_VALUE_RE = re.compile(
    r"(?:hf_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|"
    r"github_pat_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|"
    r"AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._~+/-]{8,}|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?:^|[?&;,\s])(?:token|secret|password|passwd|api[_-]?key|authorization|credential|"
    r"hf[_-]?token|aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key|"
    r"aws[_-]?session[_-]?token)\s*(?:=|:|%3[dD])\s*[^\s&;,]+",
    re.IGNORECASE,
)
_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schemas"
_DESCRIPTOR_SCHEMA = _SCHEMA_ROOT / "synaptic-hf-source-transport-v1.schema.json"
_EVIDENCE_SCHEMA = _SCHEMA_ROOT / "synaptic-hf-provisioning-evidence-v1.schema.json"


@dataclass(frozen=True)
class HFPreparedSourceTransport:
    """One locally prepared immutable bundle and descriptor."""

    root: Path
    descriptor: Mapping[str, object]
    descriptor_uri: str
    descriptor_sha256: str
    source_lock: SourceLock
    checkout_policy: CheckoutPolicy
    verification_context: ProjectContext

    @property
    def descriptor_path(self) -> Path:
        return self.root / DESCRIPTOR_FILENAME

    @property
    def bundle_root(self) -> Path:
        return self.root / BUNDLE_DIRECTORY


@dataclass(frozen=True)
class HFConsumableSourceTransport:
    """A prepared bundle whose external provisioning evidence is exact."""

    prepared: HFPreparedSourceTransport
    evidence: Mapping[str, object]
    evidence_sha256: str
    volume_spec: HFVerifiedVolumeSpec


def canonical_json_bytes(value: object) -> bytes:
    """Return the one accepted canonical JSON representation."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CloudProviderError("HF provisioning document is not canonical JSON data.") from exc


def document_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CloudProviderError("HF provisioning JSON contains a duplicate key.")
        result[key] = value
    return result


def load_canonical_json(path: Path, *, maximum_bytes: int = 4 * 1024 * 1024) -> dict[str, object]:
    """Read a bounded regular file, reject duplicates, and require canonical bytes."""

    raw = _read_regular(path, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_reject_duplicate_pairs)
    except CloudProviderError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudProviderError("HF provisioning JSON is malformed.") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise CloudProviderError("HF provisioning JSON must use the canonical wire encoding.")
    return value


def validate_hf_source_transport_descriptor(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the closed descriptor schema and immutable URI relationships."""

    document = dict(value)
    _validate_schema(document, _DESCRIPTOR_SCHEMA, label="source-transport descriptor")
    bundle = _mapping(document["bundle"])
    capsule = _mapping(document["capsule"])
    policy = _mapping(document["checkout_policy"])
    bundle_uri = str(bundle["uri"])
    if str(capsule["uri"]) != f"{bundle_uri}/capsule":
        raise CloudProviderError("HF descriptor capsule URI is not bundle-relative.")
    if str(policy["uri"]) != f"{bundle_uri}/{CHECKOUT_POLICY_FILENAME}":
        raise CloudProviderError("HF descriptor checkout-policy URI is not bundle-relative.")
    return document


def validate_hf_bootstrap_volume_config(value: Mapping[str, object]) -> tuple[str, str]:
    """Accept only the two declarative profile-C preparation settings."""

    if set(value) != {"source", "path_prefix"}:
        raise CloudProviderError(
            "hf_jobs.bootstrap_volume accepts exactly source and path_prefix; "
            "local_root, static path, and mutable mount paths are prohibited."
        )
    source = value.get("source")
    prefix = value.get("path_prefix")
    if not isinstance(source, str) or not isinstance(prefix, str):
        raise CloudProviderError("HF bootstrap volume source and path_prefix must be strings.")
    source = source.strip()
    prefix = prefix.strip()
    _validate_run_and_volume("config", source, prefix)
    return source, prefix


def validate_hf_provisioning_evidence(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the bounded, closed external evidence schema."""

    document = dict(value)
    _validate_schema(document, _EVIDENCE_SCHEMA, label="provisioning evidence")
    timestamp = str(document["asserted_at"])
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CloudProviderError("HF provisioning evidence asserted_at is invalid.") from exc
    if parsed.tzinfo is None:
        raise CloudProviderError("HF provisioning evidence asserted_at must include a timezone.")
    for field in ("actor", "provider_receipt_id", "provider_revision"):
        item = document.get(field)
        if isinstance(item, str) and _resembles_known_secret(item):
            raise CloudProviderError(
                f"HF provisioning evidence {field} matches a prohibited known-secret pattern."
            )
    return document


def _resembles_known_secret(value: str) -> bool:
    """Detect bounded known credential forms; this is not a proof of absence."""

    candidates = {value}
    current = value
    for _ in range(2):
        decoded = unquote(current)
        if decoded == current:
            break
        candidates.add(decoded)
        current = decoded
    for candidate in tuple(candidates):
        compact = candidate.strip()
        if 8 <= len(compact) <= 512 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", compact):
            padded = compact.replace("-", "+").replace("_", "/")
            padded += "=" * (-len(padded) % 4)
            try:
                decoded = base64.b64decode(padded, validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if len(decoded) <= 512:
                candidates.add(decoded)
    return any(
        _SECRET_VALUE_RE.search(candidate) or _SECRET_ASSIGNMENT_RE.search(candidate)
        for candidate in candidates
    )


def validate_hf_evidence_binding(
    descriptor: Mapping[str, object],
    evidence: Mapping[str, object],
    *,
    descriptor_uri: str,
    descriptor_sha256: str | None = None,
) -> dict[str, object]:
    """Require byte-for-byte evidence binding to the immutable descriptor."""

    accepted_descriptor = validate_hf_source_transport_descriptor(descriptor)
    accepted_evidence = validate_hf_provisioning_evidence(evidence)
    actual_descriptor_sha256 = document_sha256(accepted_descriptor)
    if descriptor_sha256 is not None and descriptor_sha256 != actual_descriptor_sha256:
        raise CloudProviderError("HF descriptor digest does not match its canonical bytes.")
    descriptor_ref = _mapping(accepted_evidence["descriptor"])
    descriptor_volume = _mapping(accepted_descriptor["volume"])
    evidence_volume = _mapping(accepted_evidence["volume"])
    descriptor_capsule = _mapping(accepted_descriptor["capsule"])
    descriptor_manifest = _mapping(descriptor_capsule["manifest"])
    expected = {
        "descriptor.uri": (descriptor_ref["uri"], descriptor_uri),
        "descriptor.sha256": (descriptor_ref["sha256"], actual_descriptor_sha256),
        "run_id": (accepted_evidence["run_id"], accepted_descriptor["run_id"]),
        "provider": (accepted_evidence["provider"], accepted_descriptor["provider"]),
        "profile": (accepted_evidence["profile"], accepted_descriptor["profile"]),
        "volume.source": (evidence_volume["source"], descriptor_volume["source"]),
        "volume.path": (evidence_volume["path"], descriptor_volume["path"]),
        "volume.type": (evidence_volume["type"], descriptor_volume["type"]),
        "volume.read_only": (evidence_volume["read_only"], descriptor_volume["read_only"]),
        "bundle_sha256": (
            accepted_evidence["bundle_sha256"], _mapping(accepted_descriptor["bundle"])["content_sha256"],
        ),
        "capsule_manifest_sha256": (
            accepted_evidence["capsule_manifest_sha256"], descriptor_manifest["sha256"],
        ),
        "source_lock_sha256": (
            accepted_evidence["source_lock_sha256"], _mapping(accepted_descriptor["source_lock"])["sha256"],
        ),
        "checkout_policy_sha256": (
            accepted_evidence["checkout_policy_sha256"], _mapping(accepted_descriptor["checkout_policy"])["sha256"],
        ),
    }
    mismatch = next((name for name, pair in expected.items() if pair[0] != pair[1]), None)
    if mismatch is not None:
        raise CloudProviderError(f"HF provisioning evidence does not match descriptor binding: {mismatch}.")
    return accepted_evidence


def prepare_hf_source_transport(
    context: ProjectContext,
    *,
    source_lock: SourceLock,
    source_lock_uri: str,
    descriptor_uri: str,
    transport_root: Path,
    volume_source: str,
    path_prefix: str,
    checkout_policy: CheckoutPolicy | None = None,
) -> HFPreparedSourceTransport:
    """Atomically prepare or idempotently reuse one immutable local transport."""

    root = Path(transport_root)
    if not root.is_absolute():
        raise CloudProviderError("HF transport root must be absolute.")
    _validate_run_and_volume(source_lock.run_id, volume_source, path_prefix)
    _validate_tracking_uri(source_lock_uri, label="source-lock URI")
    _validate_tracking_uri(descriptor_uri, label="descriptor URI")
    if not descriptor_uri.endswith(f"/{DESCRIPTOR_FILENAME}"):
        raise CloudProviderError("HF descriptor URI must identify descriptor.json.")
    bundle_uri = descriptor_uri.rsplit("/", 1)[0] + f"/{BUNDLE_DIRECTORY}"
    policy = checkout_policy or checkout_policy_from_context(context, source_lock=source_lock)
    _validate_policy_against_context(context, source_lock, policy)
    root.parent.mkdir(parents=True, exist_ok=True)
    _assert_real_directory_chain(root.parent)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent)
    )
    try:
        bundle_root = temporary / BUNDLE_DIRECTORY
        bundle_root.mkdir()
        capsule = build_capsule(
            context.engine_root,
            bundle_root / "capsule",
            revision=source_lock.engine_source.commit,
        )
        source_lock_bytes = canonical_json_bytes(source_lock.to_dict())
        policy_bytes = canonical_json_bytes(_policy_mapping(policy))
        (bundle_root / SOURCE_LOCK_FILENAME).write_bytes(source_lock_bytes)
        (bundle_root / CHECKOUT_POLICY_FILENAME).write_bytes(policy_bytes)
        inventory = _bundle_inventory(bundle_root)
        bundle_digest = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        volume_path = _volume_path(path_prefix, source_lock.run_id, bundle_digest)
        descriptor: dict[str, object] = {
            "schema_version": DESCRIPTOR_SCHEMA_VERSION,
            "run_id": source_lock.run_id,
            "profile": PROFILE,
            "provider": PROVIDER,
            "source_lock": {
                "uri": source_lock_uri,
                "sha256": hashlib.sha256(source_lock_bytes).hexdigest(),
                "path": SOURCE_LOCK_FILENAME,
            },
            "capsule": {
                "engine_commit": capsule.engine_commit,
                "uri": f"{bundle_uri}/capsule",
                "root": "capsule",
                "manifest": {
                    "path": f"capsule/{CAPSULE_MANIFEST}",
                    "sha256": capsule.manifest_sha256,
                },
            },
            "checkout_policy": {
                "uri": f"{bundle_uri}/{CHECKOUT_POLICY_FILENAME}",
                "path": CHECKOUT_POLICY_FILENAME,
                "sha256": hashlib.sha256(policy_bytes).hexdigest(),
            },
            "bundle": {"uri": bundle_uri, "content_sha256": bundle_digest},
            "volume": {
                "type": "bucket",
                "source": volume_source,
                "path": volume_path,
                "mount_path": MOUNT_PATH,
                "read_only": True,
            },
        }
        validate_hf_source_transport_descriptor(descriptor)
        (temporary / DESCRIPTOR_FILENAME).write_bytes(canonical_json_bytes(descriptor))
        _make_files_read_only(temporary)
        if root.exists():
            _require_identical_trees(root, temporary)
        else:
            try:
                os.replace(temporary, root)
                temporary = None
            except OSError:
                if not root.exists():
                    raise
                _require_identical_trees(root, temporary)
        return load_hf_source_transport(
            context,
            transport_root=root,
            descriptor_uri=descriptor_uri,
            source_lock_uri=source_lock_uri,
        )
    except (CapsuleError, OSError) as exc:
        if isinstance(exc, CloudProviderError):
            raise
        raise CloudProviderError("HF source transport preparation failed.") from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_tree(temporary)


def load_hf_source_transport(
    context: ProjectContext,
    *,
    transport_root: Path,
    descriptor_uri: str,
    source_lock_uri: str,
) -> HFPreparedSourceTransport:
    """Load and independently authenticate a prepared descriptor and bundle."""

    root = Path(transport_root)
    _assert_real_directory_chain(root)
    _validate_tracking_uri(descriptor_uri, label="descriptor URI")
    _validate_tracking_uri(source_lock_uri, label="source-lock URI")
    descriptor = validate_hf_source_transport_descriptor(
        load_canonical_json(root / DESCRIPTOR_FILENAME, maximum_bytes=64 * 1024)
    )
    descriptor_lock = _mapping(descriptor["source_lock"])
    expected_bundle_uri = descriptor_uri.rsplit("/", 1)[0] + f"/{BUNDLE_DIRECTORY}"
    if _mapping(descriptor["bundle"])["uri"] != expected_bundle_uri:
        raise CloudProviderError("HF descriptor bundle URI is not descriptor-relative.")
    if descriptor_lock["uri"] != source_lock_uri:
        raise CloudProviderError("HF descriptor does not preserve the canonical source-lock URI.")
    source_lock_document = load_canonical_json(root / BUNDLE_DIRECTORY / SOURCE_LOCK_FILENAME)
    try:
        source_lock = SourceLock.from_dict(source_lock_document)
        source_lock_bytes = canonical_json_bytes(source_lock.to_dict())
    except Exception as exc:
        raise CloudProviderError("HF prepared SourceLock is invalid.") from exc
    if source_lock_bytes != canonical_json_bytes(source_lock_document):
        raise CloudProviderError("HF prepared SourceLock does not round-trip canonically.")
    if source_lock.run_id != descriptor["run_id"]:
        raise CloudProviderError("HF prepared SourceLock belongs to another run.")
    if hashlib.sha256(source_lock_bytes).hexdigest() != descriptor_lock["sha256"]:
        raise CloudProviderError("HF prepared SourceLock digest mismatch.")
    capsule = _mapping(descriptor["capsule"])
    if source_lock.engine_source.commit != capsule["engine_commit"]:
        raise CloudProviderError("HF descriptor engine commit does not match SourceLock.")
    _verify_rebuilt_capsule(context.engine_root, root / BUNDLE_DIRECTORY / "capsule", str(capsule["engine_commit"]))
    actual_policy = _read_regular(root / BUNDLE_DIRECTORY / CHECKOUT_POLICY_FILENAME)
    policy_ref = _mapping(descriptor["checkout_policy"])
    if hashlib.sha256(actual_policy).hexdigest() != policy_ref["sha256"]:
        raise CloudProviderError("HF checkout policy digest mismatch.")
    policy_document = load_canonical_json(
        root / BUNDLE_DIRECTORY / CHECKOUT_POLICY_FILENAME,
        maximum_bytes=64 * 1024,
    )
    accepted_policy = _checkout_policy_from_mapping(policy_document)
    _validate_policy_against_context(context, source_lock, accepted_policy)
    actual_bundle_digest = hashlib.sha256(
        canonical_json_bytes(_bundle_inventory(root / BUNDLE_DIRECTORY))
    ).hexdigest()
    if actual_bundle_digest != _mapping(descriptor["bundle"])["content_sha256"]:
        raise CloudProviderError("HF prepared bundle content digest mismatch.")
    volume = _mapping(descriptor["volume"])
    expected_path = _volume_path(
        _path_prefix_from_volume_path(str(volume["path"]), source_lock.run_id, actual_bundle_digest),
        source_lock.run_id,
        actual_bundle_digest,
    )
    if volume["path"] != expected_path:
        raise CloudProviderError("HF descriptor volume path is not content addressed.")
    return HFPreparedSourceTransport(
        root=root,
        descriptor=descriptor,
        descriptor_uri=descriptor_uri,
        descriptor_sha256=document_sha256(descriptor),
        source_lock=source_lock,
        checkout_policy=accepted_policy,
        verification_context=context,
    )


def consume_hf_source_transport(
    context: ProjectContext,
    *,
    transport_root: Path,
    descriptor_uri: str,
    source_lock_uri: str,
    evidence: Mapping[str, object],
) -> HFConsumableSourceTransport:
    """Verify local reconstruction and exact external evidence without effects."""

    prepared = load_hf_source_transport(
        context,
        transport_root=transport_root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
    )
    accepted_evidence = validate_hf_evidence_binding(
        prepared.descriptor,
        evidence,
        descriptor_uri=descriptor_uri,
        descriptor_sha256=prepared.descriptor_sha256,
    )
    descriptor = prepared.descriptor
    volume = _mapping(descriptor["volume"])
    capsule = _mapping(descriptor["capsule"])
    manifest = _mapping(capsule["manifest"])
    source_lock = _mapping(descriptor["source_lock"])
    policy = _mapping(descriptor["checkout_policy"])
    spec = HFVerifiedVolumeSpec(
        source=str(volume["source"]),
        path=str(volume["path"]),
        mount_path=str(volume["mount_path"]),
        capsule_path=str(capsule["root"]),
        capsule_manifest_sha256=str(manifest["sha256"]),
        source_lock_path=str(source_lock["path"]),
        source_lock_sha256=str(source_lock["sha256"]),
        checkout_policy_path=str(policy["path"]),
        checkout_policy_sha256=str(policy["sha256"]),
        local_root=prepared.bundle_root.resolve(),
    )
    return HFConsumableSourceTransport(
        prepared=prepared,
        evidence=accepted_evidence,
        evidence_sha256=document_sha256(accepted_evidence),
        volume_spec=spec,
    )


def revalidate_hf_verified_volume(volume: HFVerifiedVolume) -> HFConsumableSourceTransport:
    """Reload every closed binding represented by a submission-time volume.

    This proves only that the descriptor/evidence/volume relationship remains
    intact.  It deliberately does not authorize provider submission.
    """

    required_digests = (
        volume.descriptor_sha256,
        volume.provisioning_evidence_sha256,
    )
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in required_digests):
        raise CloudProviderError(
            "HF source volumes require canonical CONSUMABLE descriptor/evidence digests."
        )
    if (
        not isinstance(volume.verification_context, ProjectContext)
        or not isinstance(volume.transport_root, Path)
        or not volume.transport_root.is_absolute()
        or not isinstance(volume.descriptor_uri, str)
        or not isinstance(volume.source_lock_uri, str)
        or not isinstance(volume.provisioning_evidence, Mapping)
    ):
        raise CloudProviderError(
            "HF source volumes require a complete closed CONSUMABLE verification binding."
        )
    consumed = consume_hf_source_transport(
        volume.verification_context,
        transport_root=volume.transport_root,
        descriptor_uri=volume.descriptor_uri,
        source_lock_uri=volume.source_lock_uri,
        evidence=volume.provisioning_evidence,
    )
    if consumed.prepared.descriptor_sha256 != volume.descriptor_sha256:
        raise CloudProviderError("HF source volume descriptor binding changed before submission.")
    if consumed.evidence_sha256 != volume.provisioning_evidence_sha256:
        raise CloudProviderError("HF source volume evidence binding changed before submission.")
    if consumed.volume_spec != volume.spec:
        raise CloudProviderError("HF source volume tuple changed before submission.")
    validate_read_only_volume_object(volume.provider_volume, consumed.volume_spec)
    return consumed


def _checkout_policy_from_mapping(value: Mapping[str, object]) -> CheckoutPolicy:
    """Reconstruct the exact persisted policy without substituting defaults."""

    allowed = {"allowed_hosts", "allowed_schemes", "nested_submodules", "max_submodule_depth", "ssh"}
    required = allowed - {"ssh"}
    if set(value) - allowed or not required <= set(value):
        raise CloudProviderError("HF checkout policy has an invalid closed shape.")
    hosts = value.get("allowed_hosts")
    schemes = value.get("allowed_schemes")
    nested = value.get("nested_submodules")
    depth = value.get("max_submodule_depth")
    if (
        not isinstance(hosts, list)
        or not hosts
        or any(not isinstance(item, str) or not item or item != item.lower() for item in hosts)
        or hosts != sorted(set(hosts))
        or not isinstance(schemes, list)
        or not schemes
        or any(not isinstance(item, str) or item not in {"https", "ssh"} for item in schemes)
        or schemes != sorted(set(schemes))
        or type(nested) is not bool
        or type(depth) is not int
    ):
        raise CloudProviderError("HF checkout policy fields are invalid or noncanonical.")
    ssh_value = value.get("ssh")
    ssh_policy = None
    if ssh_value is not None:
        if not isinstance(ssh_value, Mapping) or set(ssh_value) != {
            "executable", "agent_socket", "known_hosts",
        }:
            raise CloudProviderError("HF controlled SSH checkout policy shape is invalid.")
        try:
            ssh_policy = SSHCheckoutPolicy(
                ssh_executable=Path(str(ssh_value["executable"])),
                agent_socket=str(ssh_value["agent_socket"]),
                known_hosts=Path(str(ssh_value["known_hosts"])),
            )
        except Exception as exc:
            raise CloudProviderError("HF controlled SSH checkout policy is invalid.") from exc
    try:
        policy = CheckoutPolicy(
            allowed_hosts=frozenset(hosts),
            allowed_schemes=frozenset(schemes),
            nested_submodules=nested,
            max_submodule_depth=depth,
            ssh=ssh_policy,
        )
    except Exception as exc:
        raise CloudProviderError("HF checkout policy is invalid.") from exc
    if canonical_json_bytes(_policy_mapping(policy)) != canonical_json_bytes(dict(value)):
        raise CloudProviderError("HF checkout policy does not round-trip canonically.")
    return policy


def _validate_policy_against_context(
    context: ProjectContext,
    source_lock: SourceLock,
    accepted: CheckoutPolicy,
) -> None:
    """Allow equal or tighter persisted policy, never a broader one."""

    current = checkout_policy_from_context(context, source_lock=source_lock)
    if not accepted.allowed_hosts <= current.allowed_hosts:
        raise CloudProviderError("HF checkout policy permits a host outside the current context.")
    if not accepted.allowed_schemes <= current.allowed_schemes:
        raise CloudProviderError("HF checkout policy permits a scheme outside the current context.")
    if accepted.nested_submodules and not current.nested_submodules:
        raise CloudProviderError("HF checkout policy enables nested submodules outside the current context.")
    if accepted.max_submodule_depth > current.max_submodule_depth:
        raise CloudProviderError("HF checkout policy depth exceeds the current context.")
    if accepted.ssh is not None and "ssh" not in accepted.allowed_schemes:
        raise CloudProviderError("HF controlled SSH policy requires the persisted ssh scheme.")
    for label, source in (
        ("project", source_lock.project_source),
        ("engine", source_lock.engine_source),
    ):
        location = source.location
        if location.host not in accepted.allowed_hosts:
            raise CloudProviderError(
                f"HF checkout policy does not authorize the actual {label} repository host."
            )
        if location.scheme not in accepted.allowed_schemes:
            raise CloudProviderError(
                f"HF checkout policy does not authorize the actual {label} repository scheme."
            )
        if location.scheme == "ssh" and accepted.ssh is None:
            raise CloudProviderError(
                f"HF checkout policy requires controlled SSH for the actual {label} repository."
            )
        try:
            accepted.validate(location)
        except Exception as exc:
            raise CloudProviderError(
                f"HF checkout policy rejects the actual {label} repository location."
            ) from exc


def _validate_schema(value: object, path: Path, *, label: str) -> None:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudProviderError(f"HF {label} schema is unavailable.") from exc
    if errors:
        raise CloudProviderError(f"HF {label} is invalid: {errors[0].message}")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CloudProviderError("HF provisioning document contains an invalid object.")
    return value


def _validate_tracking_uri(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.startswith("tracking://") or "\\" in value:
        raise CloudProviderError(f"HF {label} must be a canonical tracking URI.")
    relative = value.removeprefix("tracking://")
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != relative
    ):
        raise CloudProviderError(f"HF {label} must be a canonical tracking URI.")


def _validate_run_and_volume(run_id: str, source: str, prefix: str) -> None:
    if not _SEGMENT_RE.fullmatch(run_id):
        raise CloudProviderError("HF source transport run_id is not a safe path segment.")
    if not _BUCKET_RE.fullmatch(source):
        raise CloudProviderError("HF source transport requires a namespaced bucket source.")
    _canonical_relative_path(prefix, label="volume path prefix")


def _canonical_relative_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CloudProviderError(f"HF {label} must be a canonical relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} or not _SEGMENT_RE.fullmatch(part) for part in path.parts):
        raise CloudProviderError(f"HF {label} contains an unsafe path segment.")
    if path.as_posix() != value:
        raise CloudProviderError(f"HF {label} must use canonical encoding.")
    return value


def _volume_path(prefix: str, run_id: str, bundle_digest: str) -> str:
    _canonical_relative_path(prefix, label="volume path prefix")
    if not _SEGMENT_RE.fullmatch(run_id) or not _SHA256_RE.fullmatch(bundle_digest):
        raise CloudProviderError("HF volume path identity is invalid.")
    return f"{prefix}/{run_id}/{bundle_digest}"


def _path_prefix_from_volume_path(value: str, run_id: str, digest: str) -> str:
    suffix = f"/{run_id}/{digest}"
    if not value.endswith(suffix):
        raise CloudProviderError("HF descriptor volume path has the wrong run or digest suffix.")
    return _canonical_relative_path(value[: -len(suffix)], label="volume path prefix")


def _read_regular(path: Path, *, maximum_bytes: int = 4 * 1024 * 1024) -> bytes:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_size > maximum_bytes:
            raise CloudProviderError("HF provisioning member must be a bounded regular file.")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except CloudProviderError:
        raise
    except OSError as exc:
        raise CloudProviderError("HF provisioning member is unavailable.") from exc
    if len(data) > maximum_bytes or len(data) != before.st_size or not stat.S_ISREG(after.st_mode):
        raise CloudProviderError("HF provisioning member exceeds its bound.")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
    ):
        raise CloudProviderError("HF provisioning member changed during authentication.")
    return data


def _assert_real_directory_chain(path: Path) -> None:
    if not path.is_absolute():
        raise CloudProviderError("HF transport directory must be absolute.")
    chain: list[Path] = []
    cursor = path
    while cursor != cursor.parent:
        chain.append(cursor)
        cursor = cursor.parent
    for component in reversed(chain):
        try:
            info = component.lstat()
        except OSError as exc:
            raise CloudProviderError("HF transport directory is unavailable.") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise CloudProviderError("HF transport directory cannot traverse links or reparse points.")


def _bundle_inventory(root: Path) -> list[dict[str, str]]:
    expected = {
        SOURCE_LOCK_FILENAME,
        CHECKOUT_POLICY_FILENAME,
        f"capsule/{CAPSULE_MANIFEST}",
        "capsule/tuner/cloud/bootstrap_core.py",
        "capsule/tuner/cloud/bootstrap_capsule.py",
    }
    inventory: list[dict[str, str]] = []
    actual: set[str] = set()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        for name in list(directories):
            candidate = current / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise CloudProviderError("HF bundle cannot contain links or reparse points.")
        for name in files:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            _canonical_relative_path(relative, label="bundle member path")
            actual.add(relative)
            inventory.append({"path": relative, "sha256": hashlib.sha256(_read_regular(candidate)).hexdigest()})
    if actual != expected:
        raise CloudProviderError("HF bundle member inventory is incomplete or contains extensions.")
    return sorted(inventory, key=lambda item: item["path"])


def _verify_rebuilt_capsule(repository: Path, persisted: Path, engine_commit: str) -> None:
    parent = Path(tempfile.mkdtemp(prefix="synaptic-hf-capsule-"))
    try:
        rebuilt = parent / "capsule"
        build_capsule(repository, rebuilt, revision=engine_commit)
        _require_identical_trees(persisted, rebuilt)
    except CapsuleError as exc:
        raise CloudProviderError("HF capsule could not be independently rebuilt.") from exc
    finally:
        _remove_tree(parent)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        for name in directories:
            info = (current / name).lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise CloudProviderError("HF transport tree cannot contain links or reparse points.")
        for name in files:
            candidate = current / name
            result[candidate.relative_to(root).as_posix()] = _read_regular(candidate)
    return result


def _require_identical_trees(left: Path, right: Path) -> None:
    if _tree_bytes(left) != _tree_bytes(right):
        raise CloudProviderError("Existing HF transport conflicts with deterministic preparation.")


def _make_files_read_only(root: Path) -> None:
    for directory, _directories, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            os.chmod(path, path.lstat().st_mode & ~0o222)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, _directories, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            try:
                os.chmod(path, path.lstat().st_mode | stat.S_IWRITE)
            except OSError:
                pass
    shutil.rmtree(root)


__all__ = [
    "DESCRIPTOR_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "HFConsumableSourceTransport",
    "HFPreparedSourceTransport",
    "canonical_json_bytes",
    "consume_hf_source_transport",
    "document_sha256",
    "load_canonical_json",
    "load_hf_source_transport",
    "prepare_hf_source_transport",
    "revalidate_hf_verified_volume",
    "validate_hf_evidence_binding",
    "validate_hf_bootstrap_volume_config",
    "validate_hf_provisioning_evidence",
    "validate_hf_source_transport_descriptor",
]
