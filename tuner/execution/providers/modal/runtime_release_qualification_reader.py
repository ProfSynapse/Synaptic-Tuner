"""Authenticated authoritative readback for Modal runtime qualification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
import re
import json
import math

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    parse_canonical_object,
    safe_ref,
)

from .contracts import operation_path, provider_entry_identity
from .facade import ExplicitModal154ReadFacade
from .runtime_release_qualification import (
    CURRENT_LAYOUT_LIMITATION,
    MAX_EVIDENCE_BYTES,
    QUALIFICATION_OUTPUT_SCHEMA,
    _volume_id,
    _observe_current_layout,
    ModalRuntimeReleaseQualificationDispatchV1,
    ModalRuntimeReleaseQualificationReceiptV1,
    TRANSPORT_PROBE_SCHEMA, transport_probe_root, transport_probe_content,
    verify_transport_probe_envelope,
)


_CAPTURE_AUTHORITY = object()


def _transport_evidence_bytes(document):
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(raw) > 1048576:
        raise ValueError("CPU transport evidence exceeds aggregate bound")
    return raw


@dataclass(frozen=True, slots=True)
class _TransportWitness:
    raw: bytes
    sha256: str
    authority: object


@dataclass(frozen=True, slots=True)
class ModalCPUTransportCaptureV1:
    """Process-local observed-read authority; retained JSON is not rehydration authority."""
    _raw: bytes
    _sha256: str
    _authority: object

    def __post_init__(self):
        if (type(self._authority) is not _TransportWitness
                or self._authority.authority is not _CAPTURE_AUTHORITY
                or self._authority.raw != self._raw or self._authority.sha256 != self._sha256 or type(self._raw) is not bytes
                or not 0 < len(self._raw) <= 1048576
                or hashlib.sha256(self._raw).hexdigest() != self._sha256):
            raise ValueError("CPU transport capture is not locally attested")

    def to_dict(self):
        if (type(self._authority) is not _TransportWitness
                or self._authority.authority is not _CAPTURE_AUTHORITY
                or self._authority.raw != self._raw or self._authority.sha256 != self._sha256
                or hashlib.sha256(self._raw).hexdigest() != self._sha256):
            raise ValueError("CPU transport capture changed")
        return json.loads(self._raw)


@dataclass(frozen=True, slots=True)
class ModalCPUTransportVerificationV1:
    """Exact verified same-process proof; dictionaries cannot enable capabilities."""
    _raw: bytes
    _sha256: str
    _authority: object

    def __post_init__(self):
        if (type(self._authority) is not _TransportWitness
                or self._authority.authority is not _CAPTURE_AUTHORITY
                or self._authority.raw != self._raw or self._authority.sha256 != self._sha256 or type(self._raw) is not bytes
                or not 0 < len(self._raw) <= 65536
                or hashlib.sha256(self._raw).hexdigest() != self._sha256):
            raise ValueError("CPU transport verification is not attested")

    def to_dict(self):
        if (type(self._authority) is not _TransportWitness
                or self._authority.authority is not _CAPTURE_AUTHORITY
                or self._authority.raw != self._raw or self._authority.sha256 != self._sha256
                or hashlib.sha256(self._raw).hexdigest() != self._sha256):
            raise ValueError("CPU transport verification changed")
        return json.loads(self._raw)


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseQualificationObservation:
    receipt: ModalRuntimeReleaseQualificationReceiptV1
    output: bytes


class ModalRuntimeReleaseQualificationReader:
    def __init__(self, *, facade: ExplicitModal154ReadFacade,
                 deployment_observer, verifier):
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact explicit Modal facade required")
        if not callable(getattr(deployment_observer, "observe", None)) \
                or not callable(getattr(verifier, "verify", None)):
            raise TypeError("qualification reader collaborators are incomplete")
        self._facade, self._observer, self._verifier = facade, deployment_observer, verifier

    def capture_transport_probe(self, dispatch, *, provider_call_id, monotonic=time.monotonic,
                                sleep=time.sleep, host_deadline=None):
        """CPU-only SDK observations; no training or completion evidence is fabricated."""
        from .facade import ModalFunctionCallState
        if type(dispatch) is not ModalRuntimeReleaseQualificationDispatchV1 or dispatch.transport_probe is None:
            raise ValueError("CPU transport probe is not admitted")
        call = safe_ref(provider_call_id, "provider_call_id")
        probe = dispatch.transport_probe
        root, volume = transport_probe_root(dispatch), _volume_id(dispatch.deployment_facts, "control")
        started = monotonic()
        deadline = started + probe.max_host_seconds
        if host_deadline is not None:
            if type(host_deadline) not in (int, float) or not math.isfinite(host_deadline):
                raise ValueError("CPU host deadline invalid")
            deadline = min(deadline, host_deadline)
        from .runtime_build import _bounded
        def provider_read(operation):
            # Bound the actual SDK closure, not only the outer polling loop.
            remaining = max(0, deadline - monotonic())
            return _bounded(operation, deadline=time.monotonic() + min(5, remaining),
                            code="cpu_transport_read_unavailable")
        rows, seen, staged = [], {}, False
        invalid = False
        for poll in range(probe.max_polls):
            if monotonic() >= deadline:
                break
            before = provider_read(lambda: self._facade.observe_known_call_pending(call))
            if before is not ModalFunctionCallState.PENDING:
                break
            observations = []
            page_bytes = 0
            status = "empty"
            try:
                provider_read(lambda: _observe_current_layout(self._observer, dispatch.deployment_facts))
                manifests = provider_read(lambda: self._facade.list_prefix(volume, root + "/manifests", max_entries=10))
                finals = provider_read(lambda: self._facade.list_prefix(volume, root + "/final", max_entries=10))
                if len(manifests) > 9 or len(finals) > 9:
                    raise ValueError("CPU transport inventory exceeds bound")
                expected_manifests = {root + f"/manifests/{i:012d}.json": i for i in range(9)}
                expected_finals = {root + f"/final/{i:012d}.json": i for i in range(9)}
                if (len({row[0] for row in manifests}) != len(manifests)
                        or len({row[0] for row in finals}) != len(finals)
                        or any(row[0] not in expected_manifests for row in manifests)
                        or any(row[0] not in expected_finals for row in finals)):
                    raise ValueError("CPU transport inventory rows invalid")
                for path, size, entry in (*manifests, *finals):
                    if path not in {*expected_manifests, *expected_finals} or not 0 < size <= probe.max_leaf_bytes:
                        raise ValueError("CPU transport inventory invalid")
                    if entry != provider_entry_identity(volume, path, size):
                        raise ValueError("CPU transport entry identity invalid")
                final_names = {path for path, _, _ in finals}
                # One signed manifest names the sealed stage held by the real primitive.
                for path, size, _ in manifests:
                    if expected_manifests[path] != 0 or staged:
                        continue
                    raw = provider_read(lambda: self._facade.read_complete(volume, path, max_bytes=size))
                    page_bytes += len(raw)
                    if len(raw) != size or page_bytes > probe.max_page_bytes:
                        raise ValueError("CPU transport page bound exceeded")
                    manifest = verify_transport_probe_envelope(raw, self._verifier, dispatch.key_ref)
                    base = transport_probe_content(dispatch, call, 0)
                    if set(manifest) != {*base, "stage_path", "final_path", "size_bytes", "sha256"} \
                            or any(manifest[key] != value for key, value in base.items()):
                        raise ValueError("CPU transport manifest binding invalid")
                    stage = manifest["stage_path"]
                    if (type(stage) is not str or re.fullmatch(re.escape(root) + r"/staging/[0-9a-f]{32}\.json", stage) is None
                            or manifest["final_path"] != root + "/final/000000000000.json"
                            or type(manifest["size_bytes"]) is not int or not 0 < manifest["size_bytes"] <= 8192):
                        raise ValueError("CPU transport manifest path invalid")
                    digest_text(manifest["sha256"], "CPU stage digest")
                    listed = provider_read(lambda: self._facade.list_prefix(volume, root + "/staging", max_entries=19))
                    listed = tuple(row for row in listed if row[0] == stage)
                    if listed and manifest["final_path"] not in final_names:
                        body = provider_read(lambda: self._facade.read_complete(volume, stage, max_bytes=8192))
                        page_bytes += len(body)
                        if page_bytes > probe.max_page_bytes:
                            raise ValueError("CPU transport page bound exceeded")
                        if (listed != ((stage, len(body), provider_entry_identity(volume, stage, len(body))),)
                                or len(body) != manifest["size_bytes"]
                                or hashlib.sha256(body).hexdigest() != manifest["sha256"]
                                or verify_transport_probe_envelope(body, self._verifier, dispatch.key_ref) != base):
                            raise ValueError("CPU sealed stage changed")
                        if not any(row[0] == manifest["final_path"] for row in provider_read(
                                lambda: self._facade.list_prefix(volume, root + "/final", max_entries=10))):
                            observations.append({"kind": "sealed_stage_final_absent", "sequence": 0,
                                                 "sha256": manifest["sha256"]})
                for path, size, _ in finals:
                    index = expected_finals[path]
                    raw = provider_read(lambda: self._facade.read_complete(volume, path, max_bytes=size))
                    page_bytes += len(raw)
                    if page_bytes > probe.max_page_bytes:
                        raise ValueError("CPU transport page bound exceeded")
                    if len(raw) != size or verify_transport_probe_envelope(raw, self._verifier, dispatch.key_ref) != transport_probe_content(dispatch, call, index):
                        raise ValueError("CPU transport final binding invalid")
                    observations.append({"kind": "complete_final", "sequence": index,
                                         "sha256": hashlib.sha256(raw).hexdigest()})
                status = "observed" if observations else "empty"
            except ValueError:
                status = "invalid"
                observations = []
                invalid = True
            except Exception:
                status = "read_unavailable"
                observations = []
            after = provider_read(lambda: self._facade.observe_known_call_pending(call))
            elapsed_ms = max(0, int((monotonic() - started) * 1000))
            if monotonic() >= deadline or elapsed_ms > probe.max_host_seconds * 1000:
                status = "deadline"
            elapsed_ms = min(elapsed_ms, probe.max_host_seconds * 1000)
            accepted = before is after is ModalFunctionCallState.PENDING and status in {"empty", "observed"}
            if accepted:
                for item in observations:
                    if item["kind"] == "sealed_stage_final_absent":
                        if 0 not in seen:
                            staged = True
                    else:
                        index = item["sequence"]
                        if index in seen and seen[index] != item["sha256"]:
                            invalid = True
                        seen[index] = item["sha256"]
            row = {"poll": poll, "elapsed_ms": elapsed_ms, "before": before.value, "after": after.value,
                   "status": status, "accepted": accepted, "observations": observations}
            rows.append(row)
            if len(_transport_evidence_bytes(rows)) > probe.max_ledger_bytes - 8192:
                raise ValueError("CPU transport ledger bound exceeded")
            if (invalid or (staged and len(seen) == 9) or (0 in seen and not staged)
                    or after is not ModalFunctionCallState.PENDING or status == "deadline"):
                break
            sleep(1)
        document = {"schema_version": TRANSPORT_PROBE_SCHEMA, "dispatch_digest": dispatch.dispatch_digest,
                "provider_call_id": call,
                "runtime_release_digest": dispatch.runtime_release.manifest_digest,
                "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
                "staged_final_absent": staged,
                "records": [{"sequence": i, "sha256": seen[i]} for i in sorted(seen)],
                "polls": rows, "verdict": "evidence_invalid" if invalid else
                    "observed_pending" if staged and len(seen) == 9 else "inconclusive",
                "training_executed": False, "gpu_qualified": False,
                "transport_qualified": False, "durability_acknowledged": False}
        raw = _transport_evidence_bytes(document)
        return ModalCPUTransportCaptureV1(raw, hashlib.sha256(raw).hexdigest(), _TransportWitness(raw, hashlib.sha256(raw).hexdigest(), _CAPTURE_AUTHORITY))

    def verify_transport_probe(self, dispatch, *, provider_call_id, capture, retained_capture_bytes=None):
        """Mandatory core receipt is checked separately before optional transport proof."""
        if dispatch.transport_probe is None or type(capture) is not ModalCPUTransportCaptureV1:
            raise ValueError("CPU transport evidence is not admitted")
        capture_raw = capture._raw
        capture_sha256 = capture._sha256
        durable = retained_capture_bytes is not None
        if durable and (type(retained_capture_bytes) is not bytes or retained_capture_bytes != capture_raw):
            raise ValueError("CPU transport retained capture differs")
        capture = capture.to_dict()
        call = safe_ref(provider_call_id, "provider_call_id")
        if (capture.get("dispatch_digest") != dispatch.dispatch_digest
                or capture.get("provider_call_id") != call
                or capture.get("runtime_release_digest") != dispatch.runtime_release.manifest_digest
                or capture.get("deployment_facts_digest") != dispatch.deployment_facts.facts_digest):
            raise ValueError("CPU transport capture binding differs")
        # Reconstruct the accepted facts from the bounded retained ledger rather
        # than trusting a caller-supplied summary or verdict.
        polls = capture.get("polls")
        if type(polls) is not list or len(polls) > 180 or len(_transport_evidence_bytes(capture)) > 1048576:
            raise ValueError("CPU transport retained ledger invalid")
        derived, held, invalid = {}, None, False
        held_poll, final_zero_poll, previous_elapsed = None, None, -1
        for index, row in enumerate(polls):
            if (type(row) is not dict or set(row) != {"poll", "elapsed_ms", "before", "after", "status", "accepted", "observations"}
                    or type(row["poll"]) is not int or row["poll"] != index
                    or type(row["elapsed_ms"]) is not int or not max(0, previous_elapsed) <= row["elapsed_ms"] <= 180000
                    or row["before"] not in {"pending", "unknown"} or row["after"] not in {"pending", "unknown"}
                    or row["status"] not in {"empty", "observed", "read_unavailable", "invalid", "deadline"}
                    or type(row["accepted"]) is not bool or type(row["observations"]) is not list
                    or len(row["observations"]) > 10):
                raise ValueError("CPU transport retained poll invalid")
            previous_elapsed = row["elapsed_ms"]
            accepted = row["before"] == row["after"] == "pending" and row["status"] in {"empty", "observed"}
            if accepted != row["accepted"]:
                raise ValueError("CPU transport pending bracket invalid")
            invalid = invalid or row["status"] == "invalid"
            for item in row["observations"]:
                if (type(item) is not dict or set(item) != {"kind", "sequence", "sha256"}
                        or item["kind"] not in {"sealed_stage_final_absent", "complete_final"}
                        or type(item["sequence"]) is not int or not 0 <= item["sequence"] <= 8):
                    raise ValueError("CPU transport retained record invalid")
                digest_text(item["sha256"], "CPU transport retained digest")
                if not accepted:
                    continue
                if item["kind"] == "sealed_stage_final_absent":
                    if item["sequence"] != 0 or (held is not None and held != item["sha256"]):
                        raise ValueError("CPU transport stage evidence conflict")
                    if final_zero_poll is None:
                        held, held_poll = item["sha256"], index
                else:
                    sequence = item["sequence"]
                    if sequence in derived and derived[sequence] != item["sha256"]:
                        raise ValueError("CPU transport final evidence conflict")
                    derived[sequence] = item["sha256"]
                    if sequence == 0 and final_zero_poll is None:
                        final_zero_poll = index
        records = [{"sequence": i, "sha256": derived[i]} for i in sorted(derived)]
        complete = (held is not None and len(derived) == 9 and held == derived.get(0)
                    and final_zero_poll is not None and held_poll < final_zero_poll and not invalid)
        verdict = "evidence_invalid" if invalid else "observed_pending" if complete else "inconclusive"
        if (capture.get("records") != records or capture.get("staged_final_absent") is not (held is not None)
                or capture.get("verdict") != verdict):
            raise ValueError("CPU transport retained summary differs")
        core = self.observe(dispatch, provider_call_id=call)
        path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
        volume = _volume_id(dispatch.deployment_facts, "artifacts")
        prefix = path.rsplit("/", 1)[0]
        listed = self._facade.list_prefix(volume, prefix, max_entries=2)
        raw = self._facade.read_complete(volume, path, max_bytes=8192)
        expected_entry = ((path, len(raw), provider_entry_identity(volume, path, len(raw))),)
        if listed != expected_entry or self._facade.list_prefix(volume, prefix, max_entries=2) != expected_entry:
            raise ValueError("CPU transport terminal receipt inventory changed")
        content = verify_transport_probe_envelope(raw, self._verifier, dispatch.key_ref)
        expected = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
            "effect_id": dispatch.effect_id, "dispatch_digest": dispatch.dispatch_digest,
            "runtime_release_digest": dispatch.runtime_release.manifest_digest,
            "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
            "fixture_sha256": dispatch.fixture.sha256, "provider_call_id": call,
            "probe": dispatch.transport_probe.to_dict(), "evidence": content.get("evidence"),
            "training_executed": False, "gpu_qualified": False}
        if content != expected:
            raise ValueError("CPU transport receipt binding differs")
        evidence = content["evidence"]
        expected_evidence = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
            "root": transport_probe_root(dispatch), "records": capture.get("records"),
            "manual_commit_used": False, "training_executed": False, "gpu_qualified": False}
        passed = (capture.get("verdict") == "observed_pending" and capture.get("staged_final_absent") is True
                  and len(capture.get("records", [])) == 9 and evidence == expected_evidence)
        result = {"schema_version": TRANSPORT_PROBE_SCHEMA, "status": "passed" if passed else "inconclusive",
                "dispatch_digest": dispatch.dispatch_digest, "provider_call_id": call,
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "capture_sha256": capture_sha256,
                "core_receipt_sha256": hashlib.sha256(core.receipt.canonical_bytes).hexdigest(),
                "runtime_release_digest": dispatch.runtime_release.manifest_digest,
                "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
                "probe": dispatch.transport_probe.to_dict(), "transport_qualified": False,
                "partial_write_behavior_qualified": False, "training_executed": False,
                "gpu_qualified": False, "durability_acknowledged": durable}
        result_raw = canonical_bytes(result)
        return ModalCPUTransportVerificationV1(result_raw, hashlib.sha256(result_raw).hexdigest(), _TransportWitness(result_raw, hashlib.sha256(result_raw).hexdigest(), _CAPTURE_AUTHORITY))

    def observe(self, dispatch: ModalRuntimeReleaseQualificationDispatchV1,
                *, provider_call_id: str) -> ModalRuntimeReleaseQualificationObservation:
        if type(dispatch) is not ModalRuntimeReleaseQualificationDispatchV1:
            raise TypeError("exact qualification dispatch required")
        call_id = safe_ref(provider_call_id, "provider_call_id")
        current_digest = _observe_current_layout(
            self._observer, dispatch.deployment_facts,
        )
        facts = dispatch.deployment_facts
        root = operation_path(
            dispatch.effect_id, "runtime-release-qualification", "receipt",
        )
        raw = self._facade.read_complete(
            _volume_id(facts, "control"), root + "/receipt.json",
            max_bytes=64 * 1024,
        )
        tag = self._facade.read_complete(
            _volume_id(facts, "control"), root + "/receipt.mac", max_bytes=128,
        )
        receipt = ModalRuntimeReleaseQualificationReceiptV1.parse(raw)
        try:
            valid = self._verifier.verify(
                "modal-runtime-release-qualification-receipt/v1",
                raw, tag, dispatch.key_ref,
            )
        except Exception:
            valid = False
        if valid is not True:
            raise ValueError("runtime qualification receipt authentication failed")
        if receipt.effect_id != dispatch.effect_id \
                or receipt.dispatch_digest != dispatch.dispatch_digest \
                or receipt.provider_call_id != call_id \
                or receipt.deployment_facts_digest != facts.facts_digest \
                or receipt.current_observation_digest != current_digest:
            raise ValueError("runtime qualification receipt differs from dispatch")
        expected_entry = provider_entry_identity(
            _volume_id(facts, "artifacts"), receipt.output_path, receipt.output_size,
        )
        prefix = receipt.output_path.rsplit("/", 1)[0]
        if receipt.output_provider_entry_id != expected_entry \
                or self._facade.list_prefix(
                    _volume_id(facts, "artifacts"), prefix, max_entries=2,
                ) != ((receipt.output_path, receipt.output_size, expected_entry),):
            raise ValueError("runtime qualification output inventory differs")
        output = self._facade.read_complete(
            _volume_id(facts, "artifacts"), receipt.output_path,
            max_bytes=min(receipt.output_size, MAX_EVIDENCE_BYTES),
        )
        if len(output) != receipt.output_size \
                or hashlib.sha256(output).hexdigest() != receipt.output_sha256:
            raise ValueError("runtime qualification output content differs")
        document = parse_canonical_object(
            output, name="Modal runtime qualification output",
        )
        if set(document) != {
            "schema_version", "status", "effect_id", "dispatch_digest",
            "runtime_release_digest", "deployment_facts_digest",
            "fixture_sha256", "child", "limitations", "training_executed",
            "gpu_qualified",
        } or document.get("schema_version") != QUALIFICATION_OUTPUT_SCHEMA \
                or document.get("status") != "passed" \
                or document.get("effect_id") != dispatch.effect_id \
                or document.get("dispatch_digest") != dispatch.dispatch_digest \
                or document.get("runtime_release_digest") != dispatch.runtime_release.manifest_digest \
                or document.get("deployment_facts_digest") != facts.facts_digest \
                or document.get("fixture_sha256") != dispatch.fixture.sha256 \
                or document.get("limitations") != [CURRENT_LAYOUT_LIMITATION] \
                or document.get("training_executed") is not False \
                or document.get("gpu_qualified") is not False:
            raise ValueError("runtime qualification output is invalid")
        child = document.get("child")
        if type(child) is not dict:
            raise ValueError("runtime qualification child evidence is invalid")
        try:
            trainer_digest = digest_text(
                child.get("trainer_sha256"), "trainer_sha256",
            )
            from tuner.runtime.packaged_training_worker import local_cpu_result
            expected_child = local_cpu_result(dispatch.runtime_release, trainer_digest)
        except Exception:
            raise ValueError("runtime qualification child evidence is invalid") from None
        if child != expected_child:
            raise ValueError("runtime qualification child evidence is invalid")
        # Re-list after the body read so a concurrent replacement cannot be
        # mistaken for the authoritative member retained in the receipt.
        if self._facade.list_prefix(
            _volume_id(facts, "artifacts"), prefix, max_entries=2,
        ) != ((receipt.output_path, receipt.output_size, expected_entry),):
            raise ValueError("runtime qualification output changed during read")
        return ModalRuntimeReleaseQualificationObservation(receipt, output)


__all__ = [
    "ModalRuntimeReleaseQualificationObservation",
    "ModalRuntimeReleaseQualificationReader",
]
