"""Create-only, non-authoritative packaged training progress envelopes.

One signed file commits one metric or the closed capture-exhausted status. The
files live outside the completion namespace and become visible through Modal's
ordinary background Volume commits. A missing unseen tail proves no completion.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Event, Thread

from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION, ObservationFamily, ObservationKind,
    ObservationRecordV1, ObservationStreamRef, TrainingMetricObservedPayloadV1,
    TrainingPhaseObservedPayloadV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogEntry

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, parse_canonical_object, safe_ref,
)
from tuner.execution.observation_logs import (
    decode_training_progress_log, encode_training_progress_log,
)


PROGRESS_PURPOSE = "modal-packaged-progress/v1"
PROGRESS_SCHEMA = "synaptic-modal-packaged-progress/v1"
MAX_PROGRESS_FILE_BYTES = 8192
PROGRESS_SHARD_SIZE = 256
MAX_PROGRESS_RECORDS = 10_000
ZERO_DIGEST = "0" * 64


def progress_root(effect_id: str) -> str:
    effect_id = safe_ref(effect_id, "effect_id")
    return "progress-v1-" + hashlib.sha256(effect_id.encode("utf-8")).hexdigest()


def progress_path(effect_id: str, index: int) -> str:
    if type(index) is not int or not 0 <= index <= MAX_PROGRESS_RECORDS:
        raise ValueError("invalid packaged progress index")
    return f"{progress_root(effect_id)}/shards/{index // PROGRESS_SHARD_SIZE:06d}/{index:012d}.json"


def progress_shard(effect_id: str, index: int) -> str:
    return progress_path(effect_id, index).rsplit("/", 1)[0]


def _content_identity(*, effect_id, command_digest, provider_job_ref,
                      execution_binding_digest, run: TrainingRunRef,
                      max_records: int) -> dict[str, object]:
    if type(run) is not TrainingRunRef or type(max_records) is not int \
            or not 1 <= max_records <= MAX_PROGRESS_RECORDS:
        raise ValueError("invalid packaged progress identity")
    return {
        "schema_version": PROGRESS_SCHEMA,
        "effect_id": safe_ref(effect_id, "effect_id"),
        "command_digest": digest_text(command_digest, "command_digest"),
        "provider_job_ref": safe_ref(provider_job_ref, "provider_job_ref"),
        "execution_binding_digest": digest_text(
            execution_binding_digest, "execution_binding_digest",
        ),
        "project_ref": safe_ref(run.project_ref, "project_ref"),
        "run_id": safe_ref(run.run_id, "run_id"),
        "max_records": max_records,
    }


def make_progress_file(
    *, signer, key_ref: str, effect_id: str, command_digest: str,
    provider_job_ref: str, execution_binding_digest: str,
    run: TrainingRunRef, max_records: int, index: int,
    previous_digest: str, record: ObservationRecordV1,
) -> tuple[bytes, str]:
    identity = _content_identity(
        effect_id=effect_id, command_digest=command_digest,
        provider_job_ref=provider_job_ref,
        execution_binding_digest=execution_binding_digest,
        run=run, max_records=max_records,
    )
    if type(index) is not int or not 0 <= index <= max_records:
        raise ValueError("invalid packaged progress index")
    digest_text(previous_digest, "previous_digest")
    if index == 0 and previous_digest != ZERO_DIGEST:
        raise ValueError("packaged progress genesis mismatch")
    entry = encode_training_progress_log(record, expected_run=run)
    exhausted = record.kind is ObservationKind.TRAINING_PHASE_OBSERVED
    if (entry.sequence != index or (exhausted and index != max_records)
            or (not exhausted and index >= max_records)):
        raise ValueError("invalid packaged progress record sequence")
    content = {
        **identity,
        "index": index,
        "previous_digest": previous_digest,
        "kind": "exhausted" if exhausted else "metric",
        "entry": entry.to_dict(),
    }
    payload = canonical_bytes(content)
    try:
        tag = signer.sign(PROGRESS_PURPOSE, payload, key_ref)
    except Exception:
        raise ValueError("packaged progress authentication unavailable") from None
    if type(tag) is not bytes or not 1 <= len(tag) <= 128:
        raise ValueError("packaged progress authentication invalid")
    raw = canonical_bytes({"content": content, "tag_hex": tag.hex()})
    if len(raw) > MAX_PROGRESS_FILE_BYTES:
        raise ValueError("packaged progress file exceeds bound")
    return raw, hashlib.sha256(raw).hexdigest()


def parse_progress_file(
    raw: bytes, *, verifier, key_ref: str, effect_id: str,
    command_digest: str, provider_job_ref: str,
    execution_binding_digest: str, run: TrainingRunRef,
    max_records: int | None, index: int, previous_digest: str | None,
) -> tuple[RunLogEntry, str, int, bool]:
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PROGRESS_FILE_BYTES:
        raise ValueError("packaged progress file invalid")
    envelope = parse_canonical_object(raw, name="packaged progress envelope")
    if set(envelope) != {"content", "tag_hex"} or type(envelope["content"]) is not dict:
        raise ValueError("packaged progress envelope invalid")
    content = envelope["content"]
    declared_max = content.get("max_records")
    if max_records is not None and declared_max != max_records:
        raise ValueError("packaged progress capture bound mismatch")
    identity = _content_identity(
        effect_id=effect_id, command_digest=command_digest,
        provider_job_ref=provider_job_ref,
        execution_binding_digest=execution_binding_digest,
        run=run, max_records=declared_max,
    )
    if set(content) != {*identity, "index", "previous_digest", "kind", "entry"} \
            or any(content[key] != value for key, value in identity.items()) \
            or content["index"] != index \
            or type(content["index"]) is not int \
            or (previous_digest is not None and content["previous_digest"] != previous_digest):
        raise ValueError("packaged progress binding mismatch")
    digest_text(content["previous_digest"], "previous_digest")
    if index == 0 and content["previous_digest"] != ZERO_DIGEST:
        raise ValueError("packaged progress genesis mismatch")
    tag_hex = envelope["tag_hex"]
    if type(tag_hex) is not str or not 2 <= len(tag_hex) <= 256 \
            or len(tag_hex) % 2 or any(ch not in "0123456789abcdef" for ch in tag_hex):
        raise ValueError("packaged progress tag invalid")
    try:
        valid = verifier.verify(
            PROGRESS_PURPOSE, canonical_bytes(content), bytes.fromhex(tag_hex), key_ref,
        )
    except Exception:
        raise ValueError("packaged progress authentication unavailable") from None
    if valid is not True:
        raise ValueError("packaged progress authentication failed")
    kind = content["kind"]
    if kind not in {"metric", "exhausted"} or type(content["entry"]) is not dict:
        raise ValueError("packaged progress kind invalid")
    try:
        entry = RunLogEntry.from_dict(content["entry"])
        record = decode_training_progress_log(entry, expected_run=run)
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("packaged progress entry invalid") from None
    if entry.sequence != index or (kind == "metric" and (
        index >= declared_max or record.kind is not ObservationKind.TRAINING_METRIC_OBSERVED
    )) or (kind == "exhausted" and (
        index != declared_max or record.kind is not ObservationKind.TRAINING_PHASE_OBSERVED
    )):
        raise ValueError("packaged progress entry kind or sequence invalid")
    return entry, hashlib.sha256(raw).hexdigest(), declared_max, kind == "exhausted"


class PackagedProgressPublisher:
    """Bounded best-effort writer; callbacks enqueue without Volume I/O."""

    def __init__(
        self, *, signer, key_ref: str, effect_id: str,
        command_digest: str, provider_job_ref: str,
        execution_binding_digest: str, run: TrainingRunRef,
        max_records: int, claim_directory, write_exclusive,
    ) -> None:
        _content_identity(
            effect_id=effect_id, command_digest=command_digest,
            provider_job_ref=provider_job_ref,
            execution_binding_digest=execution_binding_digest,
            run=run, max_records=max_records,
        )
        self._identity = dict(
            signer=signer, key_ref=key_ref, effect_id=effect_id,
            command_digest=command_digest, provider_job_ref=provider_job_ref,
            execution_binding_digest=execution_binding_digest,
            run=run, max_records=max_records,
        )
        self._claim = claim_directory
        self._write = write_exclusive
        self._queue: Queue[TrainingMetricObservedPayloadV1] = Queue(maxsize=256)
        self._closed = Event()
        self._exhausted = Event()
        self._failed = Event()
        self._thread = Thread(target=self._run, name="packaged-progress", daemon=True)
        self._thread.start()

    def on_metric_frame(self, payload: TrainingMetricObservedPayloadV1) -> None:
        if self._closed.is_set() or self._exhausted.is_set() or self._failed.is_set():
            return
        if type(payload) is not TrainingMetricObservedPayloadV1:
            self._failed.set()
            return
        try:
            self._queue.put_nowait(payload)
        except Full:
            # Queue loss is not exhaustion of the configured capture budget.
            # Keep the already accepted prefix but do not sign an exhausted status.
            self._failed.set()

    def on_metric_exhausted(self) -> None:
        if not self._failed.is_set() and not self._closed.is_set():
            self._exhausted.set()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=0.25)

    def _run(self) -> None:
        identity = self._identity
        root = progress_root(identity["effect_id"])
        index, previous, shard = 0, ZERO_DIGEST, -1
        try:
            self._claim(root)
            self._claim(root + "/shards")
            while True:
                try:
                    payload = self._queue.get(timeout=0.05)
                except Empty:
                    if self._closed.is_set() or self._exhausted.is_set() or self._failed.is_set():
                        break
                    continue
                if index >= identity["max_records"]:
                    self._exhausted.set()
                    break
                payload = TrainingMetricObservedPayloadV1.from_dict(payload.to_dict())
                timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                record = ObservationRecordV1(
                    OBSERVATION_SCHEMA_VERSION,
                    ObservationStreamRef(
                        ObservationFamily.TRAINING,
                        identity["run"].project_ref,
                        identity["run"].run_id,
                    ),
                    index, timestamp, ObservationKind.TRAINING_METRIC_OBSERVED,
                    payload,
                )
                current_shard = index // PROGRESS_SHARD_SIZE
                if current_shard != shard:
                    self._claim(progress_shard(identity["effect_id"], index))
                    shard = current_shard
                raw, previous = make_progress_file(
                    **identity, index=index, previous_digest=previous,
                    record=record,
                )
                self._write(progress_path(identity["effect_id"], index), raw)
                index += 1
                if index >= identity["max_records"]:
                    self._exhausted.set()
            if self._exhausted.is_set() and not self._failed.is_set():
                current_shard = index // PROGRESS_SHARD_SIZE
                if current_shard != shard:
                    self._claim(progress_shard(identity["effect_id"], index))
                timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                status = ObservationRecordV1(
                    OBSERVATION_SCHEMA_VERSION,
                    ObservationStreamRef(
                        ObservationFamily.TRAINING,
                        identity["run"].project_ref,
                        identity["run"].run_id,
                    ),
                    index, timestamp, ObservationKind.TRAINING_PHASE_OBSERVED,
                    TrainingPhaseObservedPayloadV1("metric_capture_exhausted"),
                )
                raw, _ = make_progress_file(
                    **identity, index=index, previous_digest=previous,
                    record=status,
                )
                self._write(progress_path(identity["effect_id"], index), raw)
        except Exception:
            # Progress has no lifecycle authority. Earlier files remain a
            # useful prefix once the provider commits them in the background.
            self._failed.set()


__all__: list[str] = []
