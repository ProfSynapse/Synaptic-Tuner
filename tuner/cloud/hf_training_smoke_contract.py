"""Pure closed contracts for the isolated one-step HF training smoke."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from tuner.core.exceptions import CloudProviderError


RUNTIME_LOCK_SCHEMA = "synaptic-hf-training-runtime-lock/v1"
PREFLIGHT_SCHEMA = "synaptic-hf-training-preflight/v1"
APPROVAL_SCHEMA = "synaptic-hf-training-approval/v1"
SUBMISSION_SCHEMA = "synaptic-hf-training-submission-event/v1"
CANCELLATION_SCHEMA = "synaptic-hf-training-cancellation-event/v1"
OBSERVATION_SCHEMA = "synaptic-hf-training-observation-event/v1"
RESULT_SCHEMA = "synaptic-hf-training-result/v1"
RUNTIME_PYTHON_IMPLEMENTATION = "CPython"
RUNTIME_PYTHON_VERSION = "3.11.14"
ARTIFACT_SLOT_INPUT_SCHEMA = "synaptic-hf-training-artifact-slot-input/v1"
ARTIFACT_SLOT_DOMAIN = b"synaptic-hf-training-artifact-slot/v1\x00"
HARDWARE_QUOTE_ENDPOINT = "https://huggingface.co"
HARDWARE_QUOTE_UNIT_LABEL = "minute"
HARDWARE_QUOTE_MAX_AGE_SECONDS = 900
HARDWARE_QUOTE_MAX_UNIT_COST_MICRO_USD = 16_666
HARDWARE_MAX_HOURLY_COST_MICRO_USD = 1_000_000
HARDWARE_MAX_TIMEOUT_COST_MICRO_USD = 500_000

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_RELATIVE_PATH = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
)

TRAINING_DOCUMENT_SCHEMAS = frozenset(
    {
        RUNTIME_LOCK_SCHEMA,
        PREFLIGHT_SCHEMA,
        APPROVAL_SCHEMA,
        SUBMISSION_SCHEMA,
        CANCELLATION_SCHEMA,
        OBSERVATION_SCHEMA,
        RESULT_SCHEMA,
    }
)
BOOTSTRAP_DOCUMENT_SCHEMAS = frozenset(
    {
        "synaptic-hf-run-approval/v1",
        "synaptic-hf-submission-claim/v1",
        "synaptic-hf-cancellation-attempt/v1",
        "synaptic-hf-bootstrap-smoke-result/v1",
    }
)

_SCHEMA_FILES = {
    RUNTIME_LOCK_SCHEMA: "synaptic-hf-training-runtime-lock-v1.schema.json",
    PREFLIGHT_SCHEMA: "synaptic-hf-training-preflight-v1.schema.json",
    APPROVAL_SCHEMA: "synaptic-hf-training-approval-v1.schema.json",
    SUBMISSION_SCHEMA: "synaptic-hf-training-submission-event-v1.schema.json",
    CANCELLATION_SCHEMA: "synaptic-hf-training-cancellation-event-v1.schema.json",
    OBSERVATION_SCHEMA: "synaptic-hf-training-observation-event-v1.schema.json",
    RESULT_SCHEMA: "synaptic-hf-training-result-v1.schema.json",
}
_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schemas"

_MAX_DESCRIPTOR_BYTES = 4 * 1024 * 1024
_INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
_CHILD_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
_CONFIG_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.config.v1+json",
        "application/vnd.docker.container.image.v1+json",
    }
)
_LAYER_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.layer.v1.tar",
        "application/vnd.oci.image.layer.v1.tar+gzip",
        "application/vnd.oci.image.layer.v1.tar+zstd",
        "application/vnd.docker.image.rootfs.diff.tar.gzip",
        "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
    }
)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CloudProviderError("HF training-smoke document is not canonical JSON data") from exc


def document_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_hf_training_artifact_slot(value: Mapping[str, object]) -> str:
    """Derive the approval-bound artifact slot from a closed domain-separated input."""

    slot_input = _mapping(value, "artifact slot input")
    required = {
        "schema_version",
        "experiment_id",
        "run_id",
        "tracking_root_id",
        "source_lock_sha256",
        "workload_digest",
        "runtime_lock_sha256",
        "artifact_bucket_id",
        "artifact_base_prefix",
    }
    if set(slot_input) != required or slot_input.get("schema_version") != ARTIFACT_SLOT_INPUT_SCHEMA:
        raise CloudProviderError("HF training-smoke artifact slot input is not closed")
    for key in ("experiment_id", "run_id"):
        item = slot_input.get(key)
        if not isinstance(item, str) or _SEGMENT.fullmatch(item) is None:
            raise CloudProviderError("HF training-smoke artifact slot identity is invalid")
    for key in (
        "tracking_root_id",
        "source_lock_sha256",
        "workload_digest",
        "runtime_lock_sha256",
    ):
        item = slot_input.get(key)
        if not isinstance(item, str) or _HEX64.fullmatch(item) is None:
            raise CloudProviderError("HF training-smoke artifact slot digest is invalid")
    bucket = slot_input.get("artifact_bucket_id")
    base = slot_input.get("artifact_base_prefix")
    if not isinstance(bucket, str) or _BUCKET_ID.fullmatch(bucket) is None:
        raise CloudProviderError("HF training-smoke artifact slot bucket is invalid")
    if (
        not isinstance(base, str)
        or len(base) > 1024
        or _RELATIVE_PATH.fullmatch(base) is None
        or base != base.rstrip("/")
    ):
        raise CloudProviderError("HF training-smoke artifact base prefix is invalid")
    return hashlib.sha256(ARTIFACT_SLOT_DOMAIN + canonical_json_bytes(slot_input)).hexdigest()


def derive_hf_training_artifact_prefix(base_prefix: str, slot_id: str) -> str:
    if (
        not isinstance(base_prefix, str)
        or _RELATIVE_PATH.fullmatch(base_prefix) is None
        or base_prefix != base_prefix.rstrip("/")
        or not isinstance(slot_id, str)
        or _HEX64.fullmatch(slot_id) is None
    ):
        raise CloudProviderError("HF training-smoke artifact prefix input is invalid")
    prefix = f"{base_prefix}/{slot_id}"
    if len(prefix) > 1024:
        raise CloudProviderError("HF training-smoke artifact prefix is oversized")
    return prefix


def seal_training_document(value: Mapping[str, object]) -> dict[str, object]:
    """Return a detached content-addressed document without mutating ``value``."""

    document = json.loads(canonical_json_bytes(_mapping(value, "document")))
    version = _schema_version(document)
    fields = {
        RUNTIME_LOCK_SCHEMA: "lock_id",
        PREFLIGHT_SCHEMA: "preflight_id",
        APPROVAL_SCHEMA: "authorization_id",
        SUBMISSION_SCHEMA: "event_id",
        CANCELLATION_SCHEMA: "event_id",
        OBSERVATION_SCHEMA: "event_id",
        RESULT_SCHEMA: "result_id",
    }
    field = fields.get(version)
    if field is None:
        raise CloudProviderError("Unsupported HF training-smoke schema")
    document[field] = "0" * 64
    document[field] = _content_id(document, field)
    return validate_training_document_shape(document)


def _canonical_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise CloudProviderError("HF training-smoke timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise CloudProviderError("HF training-smoke timestamp must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    rendered = parsed.isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if rendered != value:
        raise CloudProviderError("HF training-smoke timestamp is not canonical UTC")
    return rendered


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CloudProviderError(f"HF training-smoke {label} must be an object")
    return value


def _closed_keys(
    value: Mapping[str, object], required: set[str], *, optional: set[str] | None = None
) -> None:
    permitted = required | (optional or set())
    if not required <= set(value) or not set(value) <= permitted:
        raise CloudProviderError("HF training-smoke image evidence is not closed")


def _raw_json_object(raw: bytes, label: str) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_DESCRIPTOR_BYTES:
        raise CloudProviderError(f"HF training-smoke {label} bytes are missing or oversized")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CloudProviderError(f"HF training-smoke {label} bytes are not strict JSON") from exc
    if not isinstance(value, dict):
        raise CloudProviderError(f"HF training-smoke {label} must be a JSON object")
    return value


def _raw_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _validate_image_identity(image_value: object) -> Mapping[str, object]:
    image = _mapping(image_value, "image identity")
    if image["provider_reference"] != f'{image["provider_repository"]}@{image["child_digest"]}':
        raise CloudProviderError("HF training-smoke provider image reference is not child-bound")
    layer_digests = [layer["digest"] for layer in image["layers"]]
    if len(layer_digests) != len(set(layer_digests)):
        raise CloudProviderError("HF training-smoke layer digests must be unique")
    if image["requested_kind"] == "index":
        if (
            image["requested_digest"] != image["index_digest"]
            or image["requested_media_type"] != image["index_media_type"]
            or image["requested_digest"] == image["child_digest"]
        ):
            raise CloudProviderError("HF training-smoke index identity is inconsistent")
    elif (
        image["requested_digest"] != image["child_digest"]
        or image["requested_media_type"] != image["child_media_type"]
        or image["index_digest"] is not None
        or image["index_media_type"] is not None
    ):
        raise CloudProviderError("HF training-smoke direct-manifest identity is inconsistent")
    return image


def _validate_manifest_and_config(
    image: Mapping[str, object], *, child_raw: bytes, config_raw: bytes
) -> None:
    if _raw_digest(child_raw) != image["child_digest"]:
        raise CloudProviderError("HF training-smoke child manifest digest disagrees")
    child = _raw_json_object(child_raw, "child manifest")
    _closed_keys(
        child,
        {"schemaVersion", "mediaType", "config", "layers"},
        optional={"annotations"},
    )
    if child["schemaVersion"] != 2 or child["mediaType"] != image["child_media_type"]:
        raise CloudProviderError("HF training-smoke child manifest type disagrees")

    config_descriptor = _mapping(child["config"], "config descriptor")
    _closed_keys(config_descriptor, {"mediaType", "digest", "size"})
    if config_descriptor != {
        "mediaType": image["config_media_type"],
        "digest": image["config_digest"],
        "size": image["config_size"],
    }:
        raise CloudProviderError("HF training-smoke config descriptor disagrees")
    if len(config_raw) != image["config_size"] or _raw_digest(config_raw) != image["config_digest"]:
        raise CloudProviderError("HF training-smoke config bytes disagree")

    manifest_layers = child["layers"]
    if not isinstance(manifest_layers, list) or len(manifest_layers) != len(image["layers"]):
        raise CloudProviderError("HF training-smoke manifest layer inventory disagrees")
    normalized_layers: list[dict[str, object]] = []
    for descriptor in manifest_layers:
        layer = _mapping(descriptor, "layer descriptor")
        _closed_keys(layer, {"mediaType", "digest", "size"})
        if layer["mediaType"] not in _LAYER_MEDIA_TYPES:
            raise CloudProviderError("HF training-smoke layer media type is unsupported")
        normalized_layers.append(
            {"media_type": layer["mediaType"], "digest": layer["digest"], "size": layer["size"]}
        )
    if normalized_layers != image["layers"]:
        raise CloudProviderError("HF training-smoke ordered layer descriptors disagree")

    config = _raw_json_object(config_raw, "image config")
    if (
        config.get("os") != "linux"
        or config.get("architecture") != "amd64"
        or "variant" in config
    ):
        raise CloudProviderError("HF training-smoke image config platform disagrees")
    rootfs = _mapping(config.get("rootfs"), "image config rootfs")
    _closed_keys(rootfs, {"type", "diff_ids"})
    diff_ids = rootfs["diff_ids"]
    if (
        rootfs["type"] != "layers"
        or not isinstance(diff_ids, list)
        or len(diff_ids) != len(normalized_layers)
        or any(
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
            for digest in diff_ids
        )
    ):
        raise CloudProviderError("HF training-smoke image config rootfs disagrees")


def _validate_image_evidence(
    image: Mapping[str, object], *, requested_raw: bytes, child_raw: bytes | None, config_raw: bytes
) -> None:
    if _raw_digest(requested_raw) != image["requested_digest"]:
        raise CloudProviderError("HF training-smoke requested bytes disagree")
    requested = _raw_json_object(requested_raw, "requested image document")
    if requested.get("schemaVersion") != 2 or requested.get("mediaType") != image["requested_media_type"]:
        raise CloudProviderError("HF training-smoke requested document type disagrees")
    if image["requested_kind"] == "manifest":
        if child_raw is not None and child_raw != requested_raw:
            raise CloudProviderError("HF training-smoke direct manifest has divergent child bytes")
        _validate_manifest_and_config(image, child_raw=requested_raw, config_raw=config_raw)
        return

    if child_raw is None:
        raise CloudProviderError("HF training-smoke index evidence requires child bytes")
    _closed_keys(requested, {"schemaVersion", "mediaType", "manifests"}, optional={"annotations"})
    manifests = requested["manifests"]
    if not isinstance(manifests, list) or not 1 <= len(manifests) <= 256:
        raise CloudProviderError("HF training-smoke index manifest inventory is invalid")
    matches: list[Mapping[str, object]] = []
    for descriptor_value in manifests:
        descriptor = _mapping(descriptor_value, "index child descriptor")
        _closed_keys(descriptor, {"mediaType", "digest", "size", "platform"}, optional={"annotations"})
        platform = _mapping(descriptor["platform"], "index child platform")
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64" and "variant" not in platform:
            _closed_keys(platform, {"os", "architecture"})
            matches.append(descriptor)
    if len(matches) != 1:
        raise CloudProviderError("HF training-smoke index must select exactly one linux/amd64 child")
    selected = matches[0]
    if selected != {
        "mediaType": image["child_media_type"],
        "digest": image["child_digest"],
        "size": len(child_raw),
        "platform": {"os": "linux", "architecture": "amd64"},
    }:
        raise CloudProviderError("HF training-smoke selected child descriptor disagrees")
    _validate_manifest_and_config(image, child_raw=child_raw, config_raw=config_raw)


def _schema_version(value: Mapping[str, object]) -> str:
    version = value.get("schema_version")
    if not isinstance(version, str):
        raise CloudProviderError("HF training-smoke schema_version is required")
    if version in BOOTSTRAP_DOCUMENT_SCHEMAS:
        raise CloudProviderError("Bootstrap documents are not training-smoke documents")
    return version


def _validate_schema(value: Mapping[str, object], expected: str) -> dict[str, object]:
    version = _schema_version(value)
    if version != expected:
        if version in TRAINING_DOCUMENT_SCHEMAS:
            raise CloudProviderError("Wrong HF training-smoke document kind")
        raise CloudProviderError("Unsupported HF training-smoke schema")
    try:
        document = json.loads(canonical_json_bytes(value))
        schema = json.loads((_SCHEMA_ROOT / _SCHEMA_FILES[expected]).read_text(encoding="utf-8"))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    except CloudProviderError:
        raise
    except Exception as exc:
        raise CloudProviderError("HF training-smoke document does not match its exact schema") from exc
    for key in ("occurred_at", "issued_at", "expires_at", "fetched_at", "created_at"):
        if key in document:
            _canonical_utc(str(document[key]))
    return document


def _content_id(document: Mapping[str, object], field: str) -> str:
    body = {key: value for key, value in document.items() if key != field}
    return document_sha256(body)


def _require_content_id(document: Mapping[str, object], field: str) -> None:
    if document.get(field) != _content_id(document, field):
        raise CloudProviderError(f"HF training-smoke {field} does not match canonical content")


def _same_identity(current: Mapping[str, object], previous: Mapping[str, object]) -> None:
    immutable = (
        "experiment_id",
        "run_id",
        "tracking_root_id",
        "authorization_id",
        "approval",
    )
    if any(current.get(key) != previous.get(key) for key in immutable):
        raise CloudProviderError("HF training-smoke event changed immutable identity")


def validate_runtime_lock(
    value: Mapping[str, object],
    *,
    requested_raw: bytes | None = None,
    child_raw: bytes | None = None,
    config_raw: bytes | None = None,
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "runtime lock"), RUNTIME_LOCK_SCHEMA)
    _require_content_id(document, "lock_id")
    image = _validate_image_identity(document["image"])
    evidence = (requested_raw, child_raw, config_raw)
    if any(item is not None for item in evidence):
        if requested_raw is None or config_raw is None:
            raise CloudProviderError("Runtime-lock image evidence is incomplete")
        _validate_image_evidence(
            image,
            requested_raw=requested_raw,
            child_raw=child_raw,
            config_raw=config_raw,
        )
    anonymous = _mapping(document["anonymous_loading"], "anonymous loading")
    if anonymous != {
        "token": False,
        "trust_remote_code": False,
        "use_safetensors": True,
    }:
        raise CloudProviderError("Runtime lock must require exact anonymous safe loading")
    return document


def validate_preflight(
    value: Mapping[str, object],
    *,
    requested_raw: bytes | None = None,
    child_raw: bytes | None = None,
    config_raw: bytes | None = None,
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "preflight"), PREFLIGHT_SCHEMA)
    _require_content_id(document, "preflight_id")
    image = _validate_image_identity(document["image"])
    evidence = (requested_raw, child_raw, config_raw)
    if any(item is not None for item in evidence):
        if requested_raw is None or config_raw is None:
            raise CloudProviderError("Preflight image evidence is incomplete")
        _validate_image_evidence(
            image,
            requested_raw=requested_raw,
            child_raw=child_raw,
            config_raw=config_raw,
        )
    if document["status"] != "PASS" or document["job_secrets"] != []:
        raise CloudProviderError("Training preflight must PASS with no remote job secrets")
    if _mapping(document["launcher_auth"], "launcher auth")["mode"] != "explicit_file":
        raise CloudProviderError("Training preflight requires explicit-file launcher auth")
    hardware = _mapping(document["hardware"], "hardware quote")
    unit_cost = hardware["unit_cost_micro_usd"]
    if (
        hardware["endpoint"] != HARDWARE_QUOTE_ENDPOINT
        or hardware["flavor"] != "a10g-small"
        or hardware["unit_label"] != HARDWARE_QUOTE_UNIT_LABEL
        or isinstance(unit_cost, bool)
        or not isinstance(unit_cost, int)
        or not 1 <= unit_cost <= HARDWARE_QUOTE_MAX_UNIT_COST_MICRO_USD
        or hardware["hourly_cost_micro_usd"] != unit_cost * 60
        or hardware["timeout_cost_micro_usd"] != unit_cost * 30
        or hardware["hourly_cost_micro_usd"] > HARDWARE_MAX_HOURLY_COST_MICRO_USD
        or hardware["timeout_cost_micro_usd"] > HARDWARE_MAX_TIMEOUT_COST_MICRO_USD
    ):
        raise CloudProviderError("Training preflight hardware quote is invalid")
    slot_input = _mapping(document["artifact_slot_input"], "artifact slot input")
    slot_id = derive_hf_training_artifact_slot(slot_input)
    if slot_id != document["artifact_slot_id"]:
        raise CloudProviderError("Training preflight artifact slot digest is invalid")
    slot_expected = {
        "experiment_id": document["experiment_id"],
        "run_id": document["run_id"],
        "tracking_root_id": document["tracking_root_id"],
        "source_lock_sha256": document["source"]["source_lock"]["sha256"],
        "workload_digest": document["workload_digest"],
        "runtime_lock_sha256": document["runtime_lock"]["sha256"],
    }
    if any(slot_input[key] != expected for key, expected in slot_expected.items()):
        raise CloudProviderError("Training preflight artifact slot identity is inconsistent")
    source_volume, artifact_volume = document["volumes"]
    source_segments = tuple(str(source_volume["prefix"]).split("/"))
    artifact_segments = tuple(str(artifact_volume["prefix"]).split("/"))
    if source_volume["bucket_id"] == artifact_volume["bucket_id"] and (
        source_segments == artifact_segments
        or source_segments == artifact_segments[: len(source_segments)]
        or artifact_segments == source_segments[: len(artifact_segments)]
    ):
        raise CloudProviderError("Training source and artifact prefixes overlap")
    if (
        artifact_volume["bucket_id"] != slot_input["artifact_bucket_id"]
        or artifact_volume["prefix"]
        != derive_hf_training_artifact_prefix(
            str(slot_input["artifact_base_prefix"]), str(document["artifact_slot_id"])
        )
    ):
        raise CloudProviderError("Training artifact volume does not bind its exclusive slot")
    return document


def validate_approval(
    value: Mapping[str, object], *, preflight: Mapping[str, object] | None = None
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "approval"), APPROVAL_SCHEMA)
    _require_content_id(document, "authorization_id")
    if document["kind"] != "hf.training-smoke" or document["job_secrets"] != []:
        raise CloudProviderError("Training approval kind or remote-secret contract is invalid")
    if document["maximum_submissions"] != 1 or document["maximum_retries"] != 0:
        raise CloudProviderError("Training approval must authorize one submission and no retry")
    if any(document[key] for key in ("publication", "ssh", "ports", "wandb")):
        raise CloudProviderError("Training approval enables a forbidden capability")
    if document["hardware"] != "a10g-small" or document["provider_timeout_seconds"] != 1800:
        raise CloudProviderError("Training approval hardware or timeout is not fixed")
    quote = _mapping(document["hardware_quote"], "approval hardware quote")
    unit_cost = quote["unit_cost_micro_usd"]
    if (
        isinstance(unit_cost, bool)
        or not isinstance(unit_cost, int)
        or not 1 <= unit_cost <= HARDWARE_QUOTE_MAX_UNIT_COST_MICRO_USD
        or quote["hourly_cost_micro_usd"] != unit_cost * 60
        or quote["timeout_cost_micro_usd"] != unit_cost * 30
        or quote["hourly_cost_micro_usd"] > HARDWARE_MAX_HOURLY_COST_MICRO_USD
        or quote["timeout_cost_micro_usd"] > HARDWARE_MAX_TIMEOUT_COST_MICRO_USD
    ):
        raise CloudProviderError("Training approval exceeds its cost envelope")
    issued = datetime.fromisoformat(str(document["issued_at"]).replace("Z", "+00:00"))
    expires = datetime.fromisoformat(str(document["expires_at"]).replace("Z", "+00:00"))
    fetched = datetime.fromisoformat(str(quote["fetched_at"]).replace("Z", "+00:00"))
    if (
        not fetched <= issued < expires
        or (issued - fetched).total_seconds() > HARDWARE_QUOTE_MAX_AGE_SECONDS
    ):
        raise CloudProviderError("Training approval time ordering is invalid")
    if preflight is not None:
        accepted = validate_preflight(preflight)
        reference = _mapping(document["preflight"], "approval preflight")
        if reference["sha256"] != document_sha256(accepted):
            raise CloudProviderError("Training approval does not bind the supplied preflight")
        for key in ("experiment_id", "run_id", "tracking_root_id"):
            if document[key] != accepted[key]:
                raise CloudProviderError("Training approval and preflight identities differ")
        source_volume, artifact_volume = accepted["volumes"]
        source = accepted["source"]
        bindings = document["bindings"]
        if quote != {
            "preflight_sha256": document_sha256(accepted),
            "unit_cost_micro_usd": accepted["hardware"]["unit_cost_micro_usd"],
            "hourly_cost_micro_usd": accepted["hardware"]["hourly_cost_micro_usd"],
            "timeout_cost_micro_usd": accepted["hardware"]["timeout_cost_micro_usd"],
            "fetched_at": accepted["hardware"]["fetched_at"],
        }:
            raise CloudProviderError("Training approval hardware quote does not bind preflight")
        expected = {
            "source_lock_sha256": source["source_lock"]["sha256"],
            "workload_digest": accepted["workload_digest"],
            "runtime_lock_sha256": accepted["runtime_lock"]["sha256"],
            "model_revision": accepted["model"]["revision"],
            "dataset_sha256": accepted["dataset"]["sha256"],
            "image_child_digest": accepted["image"]["child_digest"],
            "remote_argv_sha256": accepted["command"]["remote_argv_sha256"],
            "provider_command_sha256": accepted["command"]["provider_command_sha256"],
            "source_bucket_id": source_volume["bucket_id"],
            "source_prefix": source_volume["prefix"],
            "artifact_bucket_id": artifact_volume["bucket_id"],
            "artifact_base_prefix": accepted["artifact_slot_input"]["artifact_base_prefix"],
            "artifact_slot_id": accepted["artifact_slot_id"],
            "artifact_prefix": artifact_volume["prefix"],
        }
        if any(bindings.get(key) != expected_value for key, expected_value in expected.items()):
            raise CloudProviderError("Training approval bindings do not match preflight")
    return document


def validate_submission_event(
    value: Mapping[str, object],
    *,
    approval: Mapping[str, object] | None = None,
    previous_event: Mapping[str, object] | None = None,
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "submission event"), SUBMISSION_SCHEMA)
    _require_content_id(document, "event_id")
    state = str(document["state"])
    if approval is not None:
        accepted = validate_approval(approval)
        if document["authorization_id"] != accepted["authorization_id"]:
            raise CloudProviderError("Submission event authorization is wrong")
    if previous_event is None:
        if state != "SUBMITTING" or document["sequence"] != 1:
            raise CloudProviderError("First submission event must be SUBMITTING")
    else:
        previous = _validate_schema(_mapping(previous_event, "previous submission event"), SUBMISSION_SCHEMA)
        _require_content_id(previous, "event_id")
        if approval is not None and previous["authorization_id"] != accepted["authorization_id"]:
            raise CloudProviderError("Submission predecessor authorization is wrong")
        ordinary = (
            previous["state"] == "SUBMITTING"
            and previous["sequence"] == 1
            and document["sequence"] == 2
            and state in {"SUBMITTED", "NOT_SUBMITTED", "AMBIGUOUS"}
        )
        recovery = (
            previous["state"] == "AMBIGUOUS"
            and previous["sequence"] == 2
            and document["sequence"] == 3
            and state == "SUBMITTED"
            and document["reason_code"] == "RECOVERY_CONFIRMED_SUBMITTED"
        )
        if not (ordinary or recovery):
            raise CloudProviderError("Submission terminal transition is invalid")
        _same_identity(document, previous)
        if document["previous_event"]["sha256"] != document_sha256(previous):
            raise CloudProviderError("Submission predecessor digest is invalid")
    effect = bool(document["provider_effect_possible"])
    if state in {"SUBMITTING", "SUBMITTED", "AMBIGUOUS"} and not effect:
        raise CloudProviderError("Submission provider-effect evidence is invalid")
    if state == "NOT_SUBMITTED" and effect:
        raise CloudProviderError("NOT_SUBMITTED requires provider effect to be impossible")
    local_reasons = {"CREDENTIAL_REJECTED", "OWNER_MISMATCH", "PREFIX_NOT_EMPTY", "APPROVAL_EXPIRED", "QUOTE_STALE", "LOCAL_PRECALL_FAILURE"}
    ambiguous_reasons = {"PROVIDER_OUTCOME_AMBIGUOUS", "INTERRUPTED_AFTER_CLAIM", "RECOVERY_EVIDENCE_INVALID"}
    if state == "NOT_SUBMITTED" and document["reason_code"] not in local_reasons:
        raise CloudProviderError("NOT_SUBMITTED reason does not prove a pre-call failure")
    if state == "AMBIGUOUS" and document["reason_code"] not in ambiguous_reasons:
        raise CloudProviderError("AMBIGUOUS submission reason is invalid")
    if state == "SUBMITTED":
        expected_reason = "RECOVERY_CONFIRMED_SUBMITTED" if document["sequence"] == 3 else None
        if document["reason_code"] != expected_reason:
            raise CloudProviderError("SUBMITTED reason does not match its transition")
    return document


def validate_cancellation_event(
    value: Mapping[str, object], *, previous_event: Mapping[str, object] | None = None
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "cancellation event"), CANCELLATION_SCHEMA)
    _require_content_id(document, "event_id")
    state = str(document["state"])
    if previous_event is None:
        if state not in {"CLAIMED", "NOT_REQUIRED"} or document["sequence"] != 1:
            raise CloudProviderError("Initial cancellation transition is invalid")
    else:
        previous = validate_cancellation_event(previous_event)
        if previous["state"] != "CLAIMED" or state not in {"REQUESTED", "NOT_REQUIRED", "AMBIGUOUS"}:
            raise CloudProviderError("Cancellation terminal transition is invalid")
        _same_identity(document, previous)
        if document["previous_event"]["sha256"] != document_sha256(previous):
            raise CloudProviderError("Cancellation predecessor digest is invalid")
    if state == "NOT_REQUIRED" and document["reason_code"] != "TERMINAL_ON_REINSPECTION":
        raise CloudProviderError("NOT_REQUIRED requires terminal reinspection evidence")
    if state == "AMBIGUOUS" and document["reason_code"] not in {
        "CANCEL_OUTCOME_AMBIGUOUS", "INTERRUPTED_AFTER_CLAIM", "RECOVERY_EVIDENCE_INVALID"
    }:
        raise CloudProviderError("AMBIGUOUS cancellation reason is invalid")
    return document


def validate_observation_event(
    value: Mapping[str, object], *, previous_event: Mapping[str, object] | None = None
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "observation event"), OBSERVATION_SCHEMA)
    _require_content_id(document, "event_id")
    state = str(document["state"])
    if document["terminal"] is not (state in {"COMPLETED", "ERROR", "CANCELLED"}):
        raise CloudProviderError("Observation terminal evidence is contradictory")
    if previous_event is not None:
        previous = validate_observation_event(previous_event)
        _same_identity(document, previous)
        if previous["state"] != "STOPPED" or state not in {"COMPLETED", "ERROR", "CANCELLED"}:
            raise CloudProviderError("Only STOPPED may be refined by a later observation")
        if document["previous_event"]["sha256"] != document_sha256(previous):
            raise CloudProviderError("Observation predecessor digest is invalid")
    return document


def validate_result(
    value: Mapping[str, object], *, previous_result: Mapping[str, object] | None = None
) -> dict[str, object]:
    document = _validate_schema(_mapping(value, "result"), RESULT_SCHEMA)
    _require_content_id(document, "result_id")
    state = str(document["state"])
    if previous_result is None:
        if state != "VERIFYING" or document["previous_result"] is not None:
            raise CloudProviderError("First artifact result must claim VERIFYING")
    else:
        previous = validate_training_document_shape(previous_result)
        if previous["schema_version"] != RESULT_SCHEMA:
            raise CloudProviderError("Artifact predecessor has wrong document kind")
        allowed = (
            state in {"VERIFIED", "INVALID", "INCONCLUSIVE"}
            if previous["state"] == "VERIFYING"
            else state == "VERIFYING"
            if previous["state"] == "INCONCLUSIVE"
            else False
        )
        if not allowed:
            raise CloudProviderError("Artifact result cannot be replaced")
        for key in (
            "experiment_id",
            "run_id",
            "tracking_root_id",
            "authorization_id",
            "provider_job",
            "approval",
            "submission",
            "observation",
        ):
            if document[key] != previous[key]:
                raise CloudProviderError("Artifact result changed immutable identity")
        for key in ("bucket_id", "base_prefix", "slot_id", "prefix"):
            if document["artifact_prefix"][key] != previous["artifact_prefix"][key]:
                raise CloudProviderError("Artifact result changed immutable prefix identity")
        if document["previous_result"]["sha256"] != document_sha256(previous):
            raise CloudProviderError("Artifact result predecessor digest is invalid")
    artifact = _mapping(document["artifact_prefix"], "artifact prefix")
    if artifact["prefix"] != derive_hf_training_artifact_prefix(
        str(artifact["base_prefix"]), str(artifact["slot_id"])
    ):
        raise CloudProviderError("Artifact result prefix is not slot-bound")
    inventory = document["inventory"]
    if not isinstance(inventory, list):
        raise CloudProviderError("Artifact result inventory is invalid")
    paths = [item["path"] for item in inventory]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CloudProviderError("Artifact result inventory must be normalized and ordered")
    inventory_digest = document_sha256(inventory)
    verified_digest = artifact["verified_inventory_sha256"]
    if verified_digest is not None and verified_digest != inventory_digest:
        raise CloudProviderError("Artifact verifier inventory digest is invalid")
    if state == "VERIFYING" and (inventory or any(
        artifact[key] is not None
        for key in (
            "pre_download_inventory_sha256",
            "post_download_inventory_sha256",
            "verified_inventory_sha256",
        )
    )):
        raise CloudProviderError("Artifact verification claim contains premature evidence")
    if state == "VERIFIED":
        inventory_digests = (
            artifact["pre_download_inventory_sha256"],
            artifact["post_download_inventory_sha256"],
            artifact["verified_inventory_sha256"],
        )
        if any(digest is None for digest in inventory_digests):
            raise CloudProviderError("Verified artifact inventories are incomplete")
        if inventory_digests[0] != inventory_digests[1]:
            raise CloudProviderError("Provider artifact inventory changed during download")
        if verified_digest != inventory_digest:
            raise CloudProviderError("Verified artifact inventory does not bind verifier output")
        proof = _mapping(document["optimizer_proof"], "optimizer proof")
        required = {
            "optimizer_boundaries": 1,
            "global_step": 1,
            "optimizer_step": 1,
            "scheduler_step": 1,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
        }
        if any(
            type(proof.get(key)) is not int or proof.get(key) != expected
            for key, expected in required.items()
        ):
            raise CloudProviderError("Verified result does not prove exactly one optimizer update")
        if type(proof.get("loss")) not in {int, float} or not math.isfinite(float(proof["loss"])):
            raise CloudProviderError("Verified result loss must be finite")
        delta = proof.get("trainable_weight_delta")
        if type(delta) not in {int, float} or not math.isfinite(float(delta)) or float(delta) <= 0:
            raise CloudProviderError("Verified result requires a finite nonzero model delta")
        if proof["post_adapter_sha256"] != proof["checkpoint_adapter_sha256"] or proof["post_adapter_sha256"] != proof["final_adapter_sha256"]:
            raise CloudProviderError("Checkpoint and final adapters do not match post-step identity")
        if proof["pre_adapter_sha256"] == proof["post_adapter_sha256"]:
            raise CloudProviderError("Verified result adapter identity did not change")
    return document


VALIDATORS = {
    RUNTIME_LOCK_SCHEMA: validate_runtime_lock,
    PREFLIGHT_SCHEMA: validate_preflight,
    APPROVAL_SCHEMA: validate_approval,
    SUBMISSION_SCHEMA: validate_submission_event,
    CANCELLATION_SCHEMA: validate_cancellation_event,
    OBSERVATION_SCHEMA: validate_observation_event,
    RESULT_SCHEMA: validate_result,
}


def validate_training_document(value: Mapping[str, object]) -> dict[str, object]:
    version = _schema_version(_mapping(value, "document"))
    validator = VALIDATORS.get(version)
    if validator is None:
        raise CloudProviderError("Unsupported HF training-smoke schema")
    return validator(value)


def validate_training_document_shape(value: Mapping[str, object]) -> dict[str, object]:
    """Validate closed bytes/content identity without resolving predecessor artifacts."""

    version = _schema_version(_mapping(value, "document"))
    document = _validate_schema(value, version)
    identity_fields = {
        RUNTIME_LOCK_SCHEMA: "lock_id",
        PREFLIGHT_SCHEMA: "preflight_id",
        APPROVAL_SCHEMA: "authorization_id",
        SUBMISSION_SCHEMA: "event_id",
        CANCELLATION_SCHEMA: "event_id",
        OBSERVATION_SCHEMA: "event_id",
        RESULT_SCHEMA: "result_id",
    }
    field = identity_fields.get(version)
    if field is None:
        raise CloudProviderError("Unsupported HF training-smoke schema")
    _require_content_id(document, field)
    return document


__all__ = [
    "APPROVAL_SCHEMA",
    "ARTIFACT_SLOT_DOMAIN",
    "ARTIFACT_SLOT_INPUT_SCHEMA",
    "BOOTSTRAP_DOCUMENT_SCHEMAS",
    "CANCELLATION_SCHEMA",
    "OBSERVATION_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "RESULT_SCHEMA",
    "RUNTIME_LOCK_SCHEMA",
    "RUNTIME_PYTHON_IMPLEMENTATION",
    "RUNTIME_PYTHON_VERSION",
    "HARDWARE_MAX_HOURLY_COST_MICRO_USD",
    "HARDWARE_MAX_TIMEOUT_COST_MICRO_USD",
    "HARDWARE_QUOTE_ENDPOINT",
    "HARDWARE_QUOTE_MAX_AGE_SECONDS",
    "HARDWARE_QUOTE_MAX_UNIT_COST_MICRO_USD",
    "HARDWARE_QUOTE_UNIT_LABEL",
    "SUBMISSION_SCHEMA",
    "TRAINING_DOCUMENT_SCHEMAS",
    "canonical_json_bytes",
    "document_sha256",
    "derive_hf_training_artifact_prefix",
    "derive_hf_training_artifact_slot",
    "seal_training_document",
    "validate_approval",
    "validate_cancellation_event",
    "validate_observation_event",
    "validate_preflight",
    "validate_result",
    "validate_runtime_lock",
    "validate_submission_event",
    "validate_training_document",
    "validate_training_document_shape",
]
