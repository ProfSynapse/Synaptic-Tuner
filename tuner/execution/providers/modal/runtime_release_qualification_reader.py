"""Authenticated authoritative readback for Modal runtime qualification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
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
        """Observe complete SDK-published CPU fixtures before their terminal receipt."""
        from .facade import ModalFunctionCallState
        from .runtime_build import _bounded
        if type(dispatch) is not ModalRuntimeReleaseQualificationDispatchV1 or dispatch.transport_probe is None:
            raise ValueError("CPU transport probe is not admitted")
        call, probe = safe_ref(provider_call_id, "provider_call_id"), dispatch.transport_probe
        root, volume = transport_probe_root(dispatch), _volume_id(dispatch.deployment_facts, "control")
        terminal = operation_path(dispatch.effect_id, "runtime-release-transport-evidence")
        started = monotonic()
        deadline = started + probe.max_host_seconds
        if host_deadline is not None:
            if type(host_deadline) not in (int, float) or not math.isfinite(host_deadline):
                raise ValueError("CPU host deadline invalid")
            deadline = min(deadline, host_deadline)
        def provider_read(operation):
            remaining = max(0, deadline - monotonic())
            return _bounded(operation, deadline=time.monotonic() + min(5, remaining),
                            code="cpu_transport_read_unavailable")
        rows, seen, invalid = [], {}, False
        for poll in range(probe.max_polls):
            if monotonic() >= deadline:
                break
            before = provider_read(lambda: self._facade.observe_known_call_pending(call))
            if before is not ModalFunctionCallState.PENDING:
                break
            observations, page_bytes = [], 0
            absent_before, absent_after, status = False, False, "empty"
            try:
                provider_read(lambda: _observe_current_layout(self._observer, dispatch.deployment_facts))
                absent_before = not provider_read(lambda: self._facade.list_prefix(volume, terminal, max_entries=2))
                if absent_before:
                    finals = provider_read(lambda: self._facade.list_prefix(volume, root + "/final", max_entries=10))
                    expected = {root + f"/final/{i:012d}.json": i for i in range(9)}
                    if (len(finals) > 9 or len({row[0] for row in finals}) != len(finals)
                            or any(row[0] not in expected for row in finals)):
                        raise ValueError("CPU transport inventory invalid")
                    for path, size, entry in finals:
                        if (type(size) is not int or not 0 < size <= probe.max_leaf_bytes
                                or entry != provider_entry_identity(volume, path, size)):
                            raise ValueError("CPU transport entry identity invalid")
                        index = expected[path]
                        raw = provider_read(lambda: self._facade.read_complete(volume, path, max_bytes=size))
                        page_bytes += len(raw)
                        if (page_bytes > probe.max_page_bytes or len(raw) != size
                                or verify_transport_probe_envelope(raw, self._verifier, dispatch.key_ref)
                                != transport_probe_content(dispatch, call, index)):
                            raise ValueError("CPU transport complete publication invalid")
                        observations.append({"kind": "complete_final", "sequence": index,
                                             "sha256": hashlib.sha256(raw).hexdigest()})
                    absent_after = not provider_read(lambda: self._facade.list_prefix(volume, terminal, max_entries=2))
                    status = "observed" if observations else "empty"
                if not absent_before or not absent_after:
                    status = "terminal_visible"
            except ValueError:
                status, observations, invalid = "invalid", [], True
            except Exception:
                status, observations = "read_unavailable", []
            after = provider_read(lambda: self._facade.observe_known_call_pending(call))
            elapsed_ms = max(0, int((monotonic() - started) * 1000))
            if monotonic() >= deadline or elapsed_ms > probe.max_host_seconds * 1000:
                status = "deadline"
            elapsed_ms = min(elapsed_ms, probe.max_host_seconds * 1000)
            accepted = (before is after is ModalFunctionCallState.PENDING
                        and status in {"empty", "observed"} and absent_before and absent_after)
            if accepted:
                for item in observations:
                    index = item["sequence"]
                    if index in seen and seen[index] != item["sha256"]:
                        invalid = True
                    seen[index] = item["sha256"]
            rows.append({"poll": poll, "elapsed_ms": elapsed_ms, "before": before.value,
                "after": after.value, "terminal_absent_before": absent_before,
                "terminal_absent_after": absent_after, "status": status,
                "accepted": accepted, "observations": observations})
            if len(_transport_evidence_bytes(rows)) > probe.max_ledger_bytes - 8192:
                raise ValueError("CPU transport ledger bound exceeded")
            if (invalid or len(seen) == 9 or after is not ModalFunctionCallState.PENDING
                    or status in {"deadline", "terminal_visible"}):
                break
            sleep(1)
        document = {"schema_version": TRANSPORT_PROBE_SCHEMA,
            "publication_strategy": probe.publication_strategy,
            "dispatch_digest": dispatch.dispatch_digest, "provider_call_id": call,
            "runtime_release_digest": dispatch.runtime_release.manifest_digest,
            "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
            "records": [{"sequence": i, "sha256": seen[i]} for i in sorted(seen)],
            "polls": rows, "verdict": "evidence_invalid" if invalid else
                "observed_pending" if len(seen) == 9 else "inconclusive",
            "training_executed": False, "gpu_qualified": False,
            "transport_qualified": False, "durability_acknowledged": False}
        raw = _transport_evidence_bytes(document)
        return ModalCPUTransportCaptureV1(raw, hashlib.sha256(raw).hexdigest(),
            _TransportWitness(raw, hashlib.sha256(raw).hexdigest(), _CAPTURE_AUTHORITY))

    def _cpu_call_completed(self, call):
        # Existing ordinary Function-call projection is for training, not CPU
        # receipts. Authenticate the pinned SDK handle and this lane's closed result.
        from .runtime_release_qualification import QUALIFICATION_RESULT_SCHEMA
        handle = self._facade.sdk.FunctionCall.from_id(call, client=self._facade.client)
        if (handle.hydrate(self._facade.client) is not handle
                or getattr(handle, "is_hydrated", False) is not True
                or getattr(handle, "object_id", None) != call):
            raise ValueError("CPU transport terminal call identity invalid")
        result = handle.get(timeout=0)
        if (type(result) is not dict or set(result) != {"schema_version", "status_code"}
                or result.get("schema_version") != QUALIFICATION_RESULT_SCHEMA
                or result.get("status_code") != "completed"):
            raise ValueError("CPU transport call has not completed successfully")

    def verify_transport_probe(self, dispatch, *, provider_call_id, capture, retained_capture_bytes=None):
        """Core proof, completed call, collision and postcommit remote inventory agree."""
        from .packaged_progress import SDK_PUBLICATION_DIAGNOSTICS
        if dispatch.transport_probe is None or type(capture) is not ModalCPUTransportCaptureV1:
            raise ValueError("CPU transport evidence is not admitted")
        capture_raw, capture_sha256 = capture._raw, capture._sha256
        durable = retained_capture_bytes is not None
        if durable and (type(retained_capture_bytes) is not bytes or retained_capture_bytes != capture_raw):
            raise ValueError("CPU transport retained capture differs")
        capture = capture.to_dict()
        call, probe = safe_ref(provider_call_id, "provider_call_id"), dispatch.transport_probe
        if (capture.get("dispatch_digest") != dispatch.dispatch_digest or capture.get("provider_call_id") != call
                or capture.get("publication_strategy") != probe.publication_strategy
                or capture.get("runtime_release_digest") != dispatch.runtime_release.manifest_digest
                or capture.get("deployment_facts_digest") != dispatch.deployment_facts.facts_digest):
            raise ValueError("CPU transport capture binding differs")
        polls = capture.get("polls")
        if type(polls) is not list or len(polls) > 180 or len(_transport_evidence_bytes(capture)) > 1048576:
            raise ValueError("CPU transport retained ledger invalid")
        derived, invalid, previous_elapsed = {}, False, -1
        for index, row in enumerate(polls):
            if (type(row) is not dict or set(row) != {"poll", "elapsed_ms", "before", "after", "status",
                    "terminal_absent_before", "terminal_absent_after", "accepted", "observations"}
                    or type(row["poll"]) is not int or row["poll"] != index
                    or type(row["elapsed_ms"]) is not int or not max(0, previous_elapsed) <= row["elapsed_ms"] <= 180000
                    or row["before"] not in {"pending", "unknown"} or row["after"] not in {"pending", "unknown"}
                    or row["status"] not in {"empty", "observed", "read_unavailable", "invalid", "deadline", "terminal_visible"}
                    or any(type(row[key]) is not bool for key in ("accepted", "terminal_absent_before", "terminal_absent_after"))
                    or type(row["observations"]) is not list or len(row["observations"]) > 9):
                raise ValueError("CPU transport retained poll invalid")
            previous_elapsed = row["elapsed_ms"]
            accepted = (row["before"] == row["after"] == "pending" and row["status"] in {"empty", "observed"}
                        and row["terminal_absent_before"] and row["terminal_absent_after"])
            if accepted != row["accepted"]:
                raise ValueError("CPU transport pending bracket invalid")
            invalid = invalid or row["status"] == "invalid"
            for item in row["observations"]:
                if (type(item) is not dict or set(item) != {"kind", "sequence", "sha256"}
                        or item["kind"] != "complete_final" or type(item["sequence"]) is not int
                        or not 0 <= item["sequence"] <= 8):
                    raise ValueError("CPU transport retained record invalid")
                digest_text(item["sha256"], "CPU transport retained digest")
                if accepted:
                    sequence = item["sequence"]
                    if sequence in derived and derived[sequence] != item["sha256"]:
                        raise ValueError("CPU transport final evidence conflict")
                    derived[sequence] = item["sha256"]
        records = [{"sequence": i, "sha256": derived[i]} for i in sorted(derived)]
        verdict = "evidence_invalid" if invalid else "observed_pending" if len(derived) == 9 else "inconclusive"
        if capture.get("records") != records or capture.get("verdict") != verdict:
            raise ValueError("CPU transport retained summary differs")
        core = self.observe(dispatch, provider_call_id=call)
        self._cpu_call_completed(call)
        root, volume = transport_probe_root(dispatch), _volume_id(dispatch.deployment_facts, "control")
        prefix = operation_path(dispatch.effect_id, "runtime-release-transport-evidence")
        path = prefix + "/receipt.json"
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
            "probe": probe.to_dict(), "evidence": content.get("evidence"),
            "training_executed": False, "gpu_qualified": False}
        if content != expected:
            raise ValueError("CPU transport receipt binding differs")
        evidence, failure_code = content["evidence"], None
        if type(evidence) is dict and evidence.get("status") == "inconclusive":
            if (set(evidence) != {"status", "failure_code"}
                    or evidence.get("failure_code") not in {*SDK_PUBLICATION_DIAGNOSTICS, "TRANSPORT_PROBE_UNAVAILABLE"}):
                raise ValueError("CPU transport diagnostic invalid")
            failure_code = evidence["failure_code"]
        inventory_verified, collision_verified = False, False
        if failure_code is None:
            leaves = evidence.get("records") if type(evidence) is dict else None
            if type(leaves) is not list or len(leaves) != 9:
                raise ValueError("CPU transport producer inventory invalid")
            collision = {"attempts": 1, "rejected": True, "path": root + "/final/000000000000.json",
                         "original_sha256": leaves[0].get("sha256") if type(leaves[0]) is dict else None}
            expected_evidence = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
                "publication_strategy": probe.publication_strategy, "root": root, "records": leaves,
                "collision": collision, "training_executed": False, "gpu_qualified": False}
            if evidence != expected_evidence:
                raise ValueError("CPU transport collision or strategy evidence invalid")
            expected_final = []
            for i, leaf in enumerate(leaves):
                expected_path = root + f"/final/{i:012d}.json"
                if (type(leaf) is not dict or set(leaf) != {"sequence", "path", "size_bytes", "sha256"}
                        or type(leaf["sequence"]) is not int or leaf["sequence"] != i or leaf["path"] != expected_path
                        or type(leaf["size_bytes"]) is not int or not 0 < leaf["size_bytes"] <= 8192):
                    raise ValueError("CPU transport signed final member invalid")
                digest_text(leaf["sha256"], "CPU transport signed member digest")
                expected_final.append((expected_path, leaf["size_bytes"], provider_entry_identity(volume, expected_path, leaf["size_bytes"])))
            actual = self._facade.list_prefix(volume, root + "/final", max_entries=10)
            if tuple(sorted(actual)) != tuple(expected_final):
                raise ValueError("CPU transport postcommit inventory differs")
            page_bytes = 0
            for i, leaf in enumerate(leaves):
                body = self._facade.read_complete(volume, leaf["path"], max_bytes=leaf["size_bytes"])
                page_bytes += len(body)
                if (page_bytes > probe.max_page_bytes or len(body) != leaf["size_bytes"]
                        or hashlib.sha256(body).hexdigest() != leaf["sha256"]
                        or verify_transport_probe_envelope(body, self._verifier, dispatch.key_ref) != transport_probe_content(dispatch, call, i)):
                    raise ValueError("CPU transport postcommit member differs")
            if tuple(sorted(self._facade.list_prefix(volume, root + "/final", max_entries=10))) != tuple(expected_final):
                raise ValueError("CPU transport postcommit inventory changed during read")
            inventory_verified, collision_verified = True, True
        signed_records = [{"sequence": leaf["sequence"], "sha256": leaf["sha256"]}
                          for leaf in evidence["records"]] if inventory_verified else []
        passed = verdict == "observed_pending" and records == signed_records and inventory_verified and collision_verified
        result = {"schema_version": TRANSPORT_PROBE_SCHEMA, "publication_strategy": probe.publication_strategy,
            "status": "passed" if passed else "inconclusive", "dispatch_digest": dispatch.dispatch_digest,
            "provider_call_id": call, "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "capture_sha256": capture_sha256, "core_receipt_sha256": hashlib.sha256(core.receipt.canonical_bytes).hexdigest(),
            "runtime_release_digest": dispatch.runtime_release.manifest_digest,
            "deployment_facts_digest": dispatch.deployment_facts.facts_digest, "probe": probe.to_dict(),
            "call_completed": True, "final_inventory_verified": inventory_verified, "collision_verified": collision_verified,
            "transport_qualified": False, "partial_write_behavior_qualified": False,
            "training_executed": False, "gpu_qualified": False, "durability_acknowledged": durable}
        if failure_code is not None:
            result["failure_code"] = failure_code
        result_raw = canonical_bytes(result)
        return ModalCPUTransportVerificationV1(result_raw, hashlib.sha256(result_raw).hexdigest(),
            _TransportWitness(result_raw, hashlib.sha256(result_raw).hexdigest(), _CAPTURE_AUTHORITY))

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
