"""Explicit-client host operator for one Modal runtime-release qualification.

The operator performs no deployment and has no automatic retry.  A missing
call receipt after ``spawn`` is ambiguous and can only be reconciled through
the consumer-owned call catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from tuner.execution.foundation_v2.canonical import digest_text, safe_ref
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalRuntimeReleaseFixtureReceiptV1,
    _self_check_fact,
    _observe_current_layout,
    _volume_id,
    parse_modal_runtime_release_qualification_dispatch,
)
from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA


class ModalRuntimeQualificationCallCatalog(Protocol):
    def resolve(self, dispatch_digest: str) -> str | None: ...
    def publish_if_absent(self, dispatch_digest: str, provider_call_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModalRuntimeQualificationOutcome:
    disposition: str
    provider_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.disposition not in {"found", "indeterminate"}:
            raise ValueError("qualification outcome disposition is closed")
        if (self.disposition == "found") != (self.provider_call_id is not None):
            raise ValueError("qualification outcome call identity is inconsistent")
        if self.provider_call_id is not None:
            safe_ref(self.provider_call_id, "provider_call_id")


class ModalRuntimeQualificationOperator:
    def __init__(self, *, facade: ExplicitModal154ReadFacade, deployment_observer,
                 verifier, call_catalog: ModalRuntimeQualificationCallCatalog):
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact explicit Modal facade required")
        for value, member in (
            (deployment_observer, "observe"), (verifier, "verify"),
            (call_catalog, "resolve"), (call_catalog, "publish_if_absent"),
        ):
            if not callable(getattr(value, member, None)):
                raise TypeError("runtime qualification operator collaborator is incomplete")
        self._facade, self._observer, self._verifier = facade, deployment_observer, verifier
        self._calls = call_catalog

    def stage_fixture_once(self, *, effect_id: str, deployment_facts):
        _observe_current_layout(self._observer, deployment_facts)
        receipt = ModalRuntimeReleaseFixtureReceiptV1.create(
            effect_id=effect_id,
            artifact_volume_id=_volume_id(deployment_facts, "artifacts"),
        )
        root = receipt.path.rsplit("/", 1)[0]
        if self._facade.list_prefix(
            receipt.artifact_volume_id, root, max_entries=2,
        ):
            raise ValueError("runtime qualification fixture collides")
        volume = self._facade._volume(receipt.artifact_volume_id)
        try:
            from io import BytesIO
            with volume.batch_upload(force=False) as batch:
                batch.put_file(BytesIO(LOCAL_CPU_DATA), receipt.path)
        except Exception:
            raise ValueError("runtime qualification fixture staging failed") from None
        content = self._facade.read_complete(
            receipt.artifact_volume_id, receipt.path,
            max_bytes=receipt.size_bytes,
        )
        listing = self._facade.list_prefix(
            receipt.artifact_volume_id, root, max_entries=2,
        )
        if content != LOCAL_CPU_DATA or hashlib.sha256(content).hexdigest() != receipt.sha256 \
                or listing != ((receipt.path, receipt.size_bytes, receipt.provider_entry_id),):
            raise ValueError("runtime qualification fixture readback differs")
        return receipt

    def submit_once(self, dispatch_bytes: bytes, *, expected_facts):
        dispatch = parse_modal_runtime_release_qualification_dispatch(
            dispatch_bytes, self._verifier,
        )
        if dispatch.deployment_facts != expected_facts:
            raise ValueError("runtime qualification dispatch targets another deployment")
        _observe_current_layout(self._observer, expected_facts)
        retained = self._calls.resolve(dispatch.dispatch_digest)
        if retained is not None:
            return ModalRuntimeQualificationOutcome(
                "found", safe_ref(retained, "provider_call_id"),
            )
        fact = _self_check_fact(expected_facts)
        function_name = safe_ref(fact.spec.name, "self_check_function_name")
        expected_id = safe_ref(fact.function_id, "self_check_function_id")
        function = self._facade._function(
            app_name=expected_facts.app_name, function_name=function_name,
        )
        try:
            function.hydrate(self._facade.client)
            if getattr(function, "is_hydrated", False) is not True \
                    or safe_ref(getattr(function, "object_id", None), "function_id") != expected_id:
                raise ValueError
            call = function.spawn(dispatch_bytes)
            call_id = safe_ref(getattr(call, "object_id", None), "provider_call_id")
            if self._calls.publish_if_absent(dispatch.dispatch_digest, call_id) is not True \
                    or self._calls.resolve(dispatch.dispatch_digest) != call_id:
                raise ValueError
        except Exception:
            return ModalRuntimeQualificationOutcome("indeterminate")
        return ModalRuntimeQualificationOutcome("found", call_id)

    def reconcile(self, dispatch_digest: str):
        digest = digest_text(dispatch_digest, "dispatch_digest")
        retained = self._calls.resolve(digest)
        if retained is None:
            return ModalRuntimeQualificationOutcome("indeterminate")
        return ModalRuntimeQualificationOutcome(
            "found", safe_ref(retained, "provider_call_id"),
        )


__all__ = [
    "ModalRuntimeQualificationCallCatalog", "ModalRuntimeQualificationOperator",
    "ModalRuntimeQualificationOutcome",
]
