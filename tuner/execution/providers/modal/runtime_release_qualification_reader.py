"""Authenticated authoritative readback for Modal runtime qualification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from tuner.execution.foundation_v2.canonical import (
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
)


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
