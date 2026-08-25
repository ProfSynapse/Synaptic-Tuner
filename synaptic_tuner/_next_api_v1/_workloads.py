"""Pure canonical workload values and sealed resolution provenance."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping

MAX_WORKLOAD_BYTES = 65_536
SFT_WORKLOAD_SCHEMA = "synaptic.sft-workload/v1"
SFT_ENTRYPOINT = "synaptic.sft.train/v1"
WORKLOAD_DIGEST_DOMAIN = b"synaptic.sft-workload/v1\0"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TARGET = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}")
_SELECTOR = re.compile(r"[A-Za-z0-9._/-]{1,512}")
_LIVE_SEAL = object()
_FIXTURE_SEAL = object()
_CANONICAL_SEAL = object()


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _revision(value: str, name: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable 40-character lowercase revision")
    return value


def _repository(value: str, name: str) -> str:
    if not isinstance(value, str) or _REPO.fullmatch(value) is None:
        raise ValueError(f"{name} must be an owner/name repository reference")
    return value


def _ordered_unique(values: Iterable[str], name: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be a non-empty ordered unique tuple")
    if any(not isinstance(value, str) or pattern.fullmatch(value) is None for value in result):
        raise ValueError(f"{name} contains an invalid value")
    return result


@dataclass(frozen=True, slots=True)
class RemoteDatasetBinding:
    repository: str
    revision: str
    split: str
    file_selector: str
    record_format: str = "conversations"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", _repository(self.repository, "dataset.repository"))
        object.__setattr__(self, "revision", _revision(self.revision, "dataset.revision"))
        if not isinstance(self.split, str) or not self.split or len(self.split) > 128 or any(
            ord(c) < 0x21 or ord(c) > 0x7e or c in "?#\\" for c in self.split
        ):
            raise ValueError("dataset.split must be bounded plain text")
        selector = self.file_selector
        if not isinstance(selector, str) or _SELECTOR.fullmatch(selector) is None:
            raise ValueError("dataset.file_selector must be bounded ASCII path text")
        if selector.endswith("/") or any(token in selector for token in ("\\", "://", "?", "#", "*", "[", "]", "{", "}", "//")):
            raise ValueError("dataset.file_selector cannot be a URL, local path, query, or glob")
        path = PurePosixPath(selector)
        if path.is_absolute() or selector.startswith("/") or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("dataset.file_selector must be a normalized relative POSIX path")
        if selector != path.as_posix():
            raise ValueError("dataset.file_selector must be canonical POSIX syntax")
        if path.suffix not in {".json", ".jsonl"}:
            raise ValueError("dataset.file_selector must end in lowercase .json or .jsonl")
        if self.record_format != "conversations":
            raise ValueError("only conversations record_format is supported")


class LiveResolutionUnavailable(RuntimeError):
    code = "live_resolution_unavailable"


@dataclass(frozen=True, slots=True, init=False)
class LiveVerified:
    """Nominal future resolver result; deliberately unconstructible offline."""

    _seal: object

    def __init__(self, *_: object, **__: object) -> None:
        raise LiveResolutionUnavailable(
            "live resolution requires the future capability-sealed production resolver"
        )

@dataclass(frozen=True, slots=True, init=False)
class FixtureVerified:
    fixture_manifest_digest: str
    target_profile_id: str
    target_profile_digest: str
    target_modules: tuple[str, ...]
    _seal: object

    def __init__(self, fixture_manifest_digest: str, target_profile_id: str,
                 target_profile_digest: str, target_modules: tuple[str, ...], *,
                 _seal: object = None) -> None:
        if _seal is not _FIXTURE_SEAL:
            raise TypeError("FixtureVerified values require the fixture factory")
        object.__setattr__(self, "fixture_manifest_digest", _digest(fixture_manifest_digest, "fixture_manifest_digest"))
        if not isinstance(target_profile_id, str) or _PROFILE_ID.fullmatch(target_profile_id) is None:
            raise ValueError("fixture target_profile_id is invalid")
        object.__setattr__(self, "target_profile_id", target_profile_id)
        object.__setattr__(self, "target_profile_digest", _digest(target_profile_digest, "target_profile_digest"))
        object.__setattr__(self, "target_modules", _ordered_unique(target_modules, "target_modules", _TARGET))
        object.__setattr__(self, "_seal", _FIXTURE_SEAL)


def fixture_verified_for_tests(*, fixture_manifest_digest: str, target_profile_id: str,
                               target_profile_digest: str,
                               target_modules: tuple[str, ...]) -> FixtureVerified:
    """Explicit test-composition hook; it cannot produce ``LiveVerified``."""
    return FixtureVerified(fixture_manifest_digest, target_profile_id, target_profile_digest,
                           target_modules, _seal=_FIXTURE_SEAL)


def require_live_verified(value: object) -> LiveVerified:
    del value
    raise LiveResolutionUnavailable("live production resolution is unavailable")


class CanonicalWorkload:
    __slots__ = ("_bytes", "_digest", "_document")

    def __init__(self, payload: bytes, document: Mapping[str, Any], *, _seal: object = None) -> None:
        if _seal is not _CANONICAL_SEAL:
            raise TypeError("CanonicalWorkload values require a canonical factory")
        self._bytes = payload
        self._digest = hashlib.sha256(WORKLOAD_DIGEST_DOMAIN + payload).hexdigest()
        self._document = MappingProxyType(dict(document))

    @property
    def canonical_bytes(self) -> bytes:
        return self._bytes

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def document(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self._bytes.decode("utf-8")))

    def __repr__(self) -> str:
        return f"CanonicalWorkload(digest={self._digest!r}, bytes={len(self._bytes)})"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is prohibited: {value}")


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*(?:\.[0-9]*[1-9])?|0\.[0-9]*[1-9])")
_ROLES = ["workload_record", "training_lineage", "training_metrics", "final_adapter", "tokenizer"]


def _object(value: object, keys: tuple[str, ...], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} field set is invalid")
    return value


def _int(value: object, name: str, zero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if zero else 1):
        raise ValueError(f"{name} is invalid")


def _dec(value: object, name: str, positive: bool = False, ratio: bool = False) -> None:
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical decimal string")
    from decimal import Decimal
    number = Decimal(value)
    if positive and number <= 0 or ratio and number > 1:
        raise ValueError(f"{name} is outside its range")


def _validate_sft(document: dict[str, Any]) -> None:
    _object(document, ("schema_version", "kind", "entrypoint", "provenance", "code", "runtime", "model", "dataset", "objective", "adapter", "optimization", "logging", "checkpoints", "artifacts"), "workload")
    if (document["schema_version"], document["kind"], document["entrypoint"]) != (SFT_WORKLOAD_SCHEMA, "sft", SFT_ENTRYPOINT): raise ValueError("invalid workload identity")
    p = document["provenance"]
    if isinstance(p, dict) and p.get("class") == "fixture_verified": _object(p, ("class", "fixture_manifest_digest"), "provenance"); _digest(p["fixture_manifest_digest"], "fixture_manifest_digest")
    elif isinstance(p, dict) and p.get("class") == "live_verified": _object(p, ("class", "resolution_attestation_digest", "release_manifest_digest"), "provenance"); _digest(p["resolution_attestation_digest"], "attestation"); _digest(p["release_manifest_digest"], "release_manifest")
    else: raise ValueError("invalid provenance")
    code = _object(document["code"], ("engine_commit", "source_digest"), "code"); _revision(code["engine_commit"], "engine_commit"); _digest(code["source_digest"], "source_digest")
    runtime = _object(document["runtime"], ("dependency_lock_digest",), "runtime"); _digest(runtime["dependency_lock_digest"], "dependency_lock_digest")
    model = _object(document["model"], ("repository", "revision", "tokenizer_revision", "trust_remote_code", "weights_format"), "model"); _repository(model["repository"], "model.repository"); _revision(model["revision"], "model.revision"); _revision(model["tokenizer_revision"], "tokenizer_revision")
    if model["trust_remote_code"] is not False or model["weights_format"] != "safetensors": raise ValueError("invalid model policy")
    dataset = _object(document["dataset"], ("repository", "revision", "split", "file_selector", "record_format"), "dataset"); RemoteDatasetBinding(**dataset)
    objective = _object(document["objective"], ("prompt_render", "chat_template", "masking_contract", "loss_scope", "max_seq_length", "packing", "truncation"), "objective"); _int(objective["max_seq_length"], "max_seq_length")
    if (objective["prompt_render"], objective["chat_template"], objective["masking_contract"], objective["packing"], objective["truncation"]) != ("full_conversation", "tokenizer_embedded_required", "synaptic.sft-mask/conversation-prefix-v1", False, "right") or objective["loss_scope"] not in {"assistant_messages", "full_sequence"}: raise ValueError("invalid objective")
    adapter = _object(document["adapter"], ("type", "target_profile_id", "target_profile_digest", "target_modules", "rank", "alpha", "dropout", "bias", "initialization", "use_rslora", "use_dora"), "adapter"); _digest(adapter["target_profile_digest"], "target_profile_digest"); _ordered_unique(adapter["target_modules"], "target_modules", _TARGET); _int(adapter["rank"], "rank"); _int(adapter["alpha"], "alpha"); _dec(adapter["dropout"], "dropout", ratio=True)
    if adapter["type"] != "lora" or not isinstance(adapter["target_profile_id"], str) or _PROFILE_ID.fullmatch(adapter["target_profile_id"]) is None or (adapter["bias"], adapter["initialization"], adapter["use_rslora"], adapter["use_dora"]) != ("none", "default", False, False): raise ValueError("invalid adapter")
    opt = _object(document["optimization"], ("duration", "batch_size", "gradient_accumulation_steps", "learning_rate", "warmup_ratio", "weight_decay", "max_grad_norm", "optimizer", "scheduler", "seed", "dtype", "model_quantization", "gradient_checkpointing"), "optimization"); _int(opt["batch_size"], "batch_size"); _int(opt["gradient_accumulation_steps"], "gradient_accumulation_steps"); _int(opt["seed"], "seed", True); _dec(opt["learning_rate"], "learning_rate", True); _dec(opt["warmup_ratio"], "warmup_ratio", ratio=True); _dec(opt["weight_decay"], "weight_decay"); _dec(opt["max_grad_norm"], "max_grad_norm", True)
    duration = opt["duration"]
    if isinstance(duration, dict) and set(duration) == {"max_steps"}: _int(duration["max_steps"], "max_steps")
    elif isinstance(duration, dict) and set(duration) == {"epochs"}: _dec(duration["epochs"], "epochs", True)
    else: raise ValueError("invalid duration")
    if (opt["optimizer"], opt["scheduler"], opt["dtype"], opt["model_quantization"], opt["gradient_checkpointing"]) != ("adamw_8bit", "linear", "bf16", "none", "unsloth"): raise ValueError("invalid optimization")
    logging = _object(document["logging"], ("structured_metrics_steps", "external_reporting"), "logging"); _int(logging["structured_metrics_steps"], "logging_steps")
    if logging["external_reporting"] != "none": raise ValueError("invalid logging")
    checkpoints = _object(document["checkpoints"], ("strategy", "steps", "limit", "resume"), "checkpoints")
    if checkpoints["resume"] is not None: raise ValueError("resume is prohibited")
    if checkpoints["strategy"] == "none" and (checkpoints["steps"] is not None or checkpoints["limit"] != 0): raise ValueError("invalid checkpoint policy")
    elif checkpoints["strategy"] == "steps": _int(checkpoints["steps"], "checkpoint_steps"); _int(checkpoints["limit"], "checkpoint_limit")
    elif checkpoints["strategy"] != "none": raise ValueError("invalid checkpoint strategy")
    artifacts = _object(document["artifacts"], ("required_roles",), "artifacts")
    if artifacts["required_roles"] != _ROLES: raise ValueError("invalid artifact roles")

def canonical_sft_workload(document: Mapping[str, Any]) -> CanonicalWorkload:
    if not isinstance(document, Mapping):
        raise TypeError("workload document must be a mapping")
    try:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("workload is not closed canonical JSON") from exc
    return decode_sft_workload(encoded)


def decode_sft_workload(payload: bytes) -> CanonicalWorkload:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_WORKLOAD_BYTES:
        raise ValueError("canonical workload bytes must be 1..65536 bytes")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("canonical workload must not contain a BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
        document = json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid canonical workload bytes") from exc
    if not isinstance(document, dict):
        raise ValueError("canonical workload root must be an object")
    _validate_sft(document)
    try:
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                               allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid canonical workload value") from exc
    if canonical != payload:
        raise ValueError("workload bytes are not in exact canonical form")
    return CanonicalWorkload(payload, document, _seal=_CANONICAL_SEAL)


__all__ = [
    "CanonicalWorkload", "FixtureVerified", "LiveResolutionUnavailable", "LiveVerified",
    "MAX_WORKLOAD_BYTES", "RemoteDatasetBinding", "SFT_ENTRYPOINT", "SFT_WORKLOAD_SCHEMA",
    "canonical_sft_workload", "decode_sft_workload", "fixture_verified_for_tests",
    "require_live_verified",
]