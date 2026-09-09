"""Canonical generic-log chunks for the Foundation-native Modal path."""

from __future__ import annotations

from dataclasses import dataclass

from synaptic_tuner.api.v1.runs_facade import RunLogEntry

from ...contracts import safe_ref
from ...foundation_v2.canonical import digest_text
from .contracts import BoundsPolicyV1, _object, canonical_json, sha, strict_int


_SCHEMA = "synaptic.modal-log-chunk/v2"
_NODE_SCHEMA = "synaptic.modal-log-chain-node/v2"
_FIELDS = {
    "schema", "generation", "sequence", "previous_digest", "payload_digest",
    "job_ref", "effect_id", "plan_digest", "invocation_nonce", "records",
}


@dataclass(frozen=True, slots=True)
class ModalCoordinatorLogChunk:
    generation: int
    sequence: int
    previous_digest: str
    payload_digest: str
    job_ref: str
    effect_id: str
    plan_digest: str
    invocation_nonce: str
    records: tuple[RunLogEntry, ...]

    def __post_init__(self) -> None:
        strict_int(self.generation, "generation", minimum=1, maximum=2**31 - 1)
        strict_int(self.sequence, "sequence", maximum=2**31 - 1)
        for name in ("previous_digest", "payload_digest", "plan_digest"):
            digest_text(getattr(self, name), name)
        for name in ("job_ref", "effect_id", "invocation_nonce"):
            safe_ref(getattr(self, name), name)
        if (
            type(self.records) is not tuple or not self.records
            or len(self.records) > BoundsPolicyV1().max_log_records
        ):
            raise ValueError("log chunk requires a nonempty exact record tuple")
        if any(type(record) is not RunLogEntry for record in self.records):
            raise TypeError("log chunk requires exact generic log entries")
        records = tuple(RunLogEntry.from_dict(record.to_dict()) for record in self.records)
        object.__setattr__(self, "records", records)
        expected = sha(canonical_json([record.to_dict() for record in records]))
        if self.payload_digest != expected:
            raise ValueError("log payload digest mismatch")
        if len(self.canonical_bytes) > BoundsPolicyV1().max_log_chunk_bytes:
            raise ValueError("coordinator log chunk exceeds default bound")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json({
            "schema": _SCHEMA,
            "generation": self.generation,
            "sequence": self.sequence,
            "previous_digest": self.previous_digest,
            "payload_digest": self.payload_digest,
            "job_ref": self.job_ref,
            "effect_id": self.effect_id,
            "plan_digest": self.plan_digest,
            "invocation_nonce": self.invocation_nonce,
            "records": [record.to_dict() for record in self.records],
        })

    @classmethod
    def parse(
        cls, data: bytes, *, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> "ModalCoordinatorLogChunk":
        if type(bounds) is not BoundsPolicyV1:
            raise TypeError("exact Modal bounds required")
        if type(data) is not bytes:
            raise TypeError("exact immutable coordinator log bytes required")
        value = _object(data, bounds.max_log_chunk_bytes)
        records = value.get("records")
        if (
            set(value) != _FIELDS or value.get("schema") != _SCHEMA
            or type(records) is not list
            or not 1 <= len(records) <= bounds.max_log_records
        ):
            raise ValueError("invalid coordinator log chunk")
        parsed: list[RunLogEntry] = []
        for record in records:
            if type(record) is not dict:
                raise ValueError("invalid coordinator log record")
            entry = RunLogEntry.from_dict(record)
            for text in (entry.timestamp, entry.level.value, entry.event, entry.message):
                if len(text.encode("utf-8")) > bounds.max_string_bytes:
                    raise ValueError("coordinator log record string exceeds bound")
            parsed.append(entry)
        chunk = cls(
            value["generation"], value["sequence"], value["previous_digest"],
            value["payload_digest"], value["job_ref"], value["effect_id"],
            value["plan_digest"], value["invocation_nonce"], tuple(parsed),
        )
        if chunk.canonical_bytes != data:
            raise ValueError("coordinator log chunk does not round-trip")
        return chunk

    @property
    def chunk_digest(self) -> str:
        return sha(canonical_json({
            "schema": _NODE_SCHEMA,
            "generation": self.generation,
            "sequence": self.sequence,
            "previous_digest": self.previous_digest,
            "payload_digest": self.payload_digest,
            "job_ref": self.job_ref,
            "effect_id": self.effect_id,
            "plan_digest": self.plan_digest,
            "invocation_nonce": self.invocation_nonce,
        }))


def validate_modal_log_chain(
    chunks: tuple[ModalCoordinatorLogChunk, ...], *,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> str:
    """Validate one nonempty identity-bound chunk and record sequence."""
    if type(bounds) is not BoundsPolicyV1:
        raise TypeError("exact Modal bounds required")
    if (
        type(chunks) is not tuple or not chunks
        or len(chunks) > bounds.max_log_records
    ):
        raise ValueError("coordinator log chain must be a nonempty exact tuple")
    previous = "0" * 64
    identity = None
    record_sequence = 0
    for chunk_sequence, chunk in enumerate(chunks):
        if type(chunk) is not ModalCoordinatorLogChunk:
            raise TypeError("coordinator log chain requires exact chunks")
        rebuilt = ModalCoordinatorLogChunk.parse(chunk.canonical_bytes, bounds=bounds)
        if rebuilt != chunk:
            raise ValueError("coordinator log chunk reconstruction mismatch")
        chunk = rebuilt
        if (
            len(chunk.canonical_bytes) > bounds.max_log_chunk_bytes
            or len(chunk.records) > bounds.max_log_records
            or record_sequence + len(chunk.records) > bounds.max_log_records
        ):
            raise ValueError("coordinator log chain exceeds its bounds")
        current = (
            chunk.generation, chunk.job_ref, chunk.effect_id,
            chunk.plan_digest, chunk.invocation_nonce,
        )
        if identity is None:
            identity = current
        if (
            current != identity or chunk.sequence != chunk_sequence
            or chunk.previous_digest != previous
        ):
            raise ValueError("coordinator log chain mismatch")
        for record in chunk.records:
            if any(
                len(text.encode("utf-8")) > bounds.max_string_bytes
                for text in (
                    record.timestamp, record.level.value,
                    record.event, record.message,
                )
            ):
                raise ValueError("coordinator log chain string exceeds bound")
            if record.sequence != record_sequence:
                raise ValueError("coordinator log record sequence mismatch")
            record_sequence += 1
        previous = chunk.chunk_digest
    return previous


__all__: list[str] = []
