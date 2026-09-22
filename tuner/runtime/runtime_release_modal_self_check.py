"""Installed package-owned Modal parent for runtime-release qualification.

The provider SDK remains confined to this parent.  The installed-child
diagnostic that it invokes is provider-neutral and runs in isolated Python.
"""
from __future__ import annotations


def run_runtime_release_self_check(dispatch_bytes: bytes) -> dict[str, object]:
    """Verify and execute one authenticated, network-blocked CPU self-check."""
    failed = {
        "schema_version": "synaptic-modal-runtime-release-qualification-result/v1",
        "status_code": "failed",
    }
    try:
        if type(dispatch_bytes) is not bytes:
            raise TypeError

        import base64
        import binascii
        import os
        from pathlib import Path

        import modal

        from tuner.execution.providers.modal.facade import (
            EXACT_MODAL_SDK_VERSION,
            MODAL_VOLUME_V1,
        )
        from tuner.execution.providers.modal.runtime_release_deployment import (
            ModalRuntimeReleaseDeploymentObservationV1,
        )
        from tuner.execution.providers.modal.runtime_release_qualification import (
            QUALIFICATION_HMAC_ENV_KEY,
            ModalRuntimeQualificationHmacAuthenticator,
            ModalRuntimeReleaseQualificationRoots,
            ModalRuntimeReleaseQualificationWorker,
            _volume_fact,
            parse_modal_runtime_release_qualification_dispatch,
        )

        encoded_key = os.environ.get(QUALIFICATION_HMAC_ENV_KEY)
        if type(encoded_key) is not str or not encoded_key or not encoded_key.isascii():
            raise ValueError
        try:
            key = base64.b64decode(encoded_key, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError from None
        if len(key) != 32 or base64.b64encode(key).decode("ascii") != encoded_key:
            raise ValueError
        authenticator = ModalRuntimeQualificationHmacAuthenticator(key)
        dispatch = parse_modal_runtime_release_qualification_dispatch(
            dispatch_bytes, authenticator,
        )
        facts = dispatch.deployment_facts
        if getattr(modal, "__version__", None) != EXACT_MODAL_SDK_VERSION \
                or os.environ.get("MODAL_IS_REMOTE") != "1" \
                or os.environ.get("MODAL_ENVIRONMENT") \
                != facts.client_binding.environment_ref \
                or os.environ.get("MODAL_IMAGE_ID") != facts.image_id:
            raise ValueError

        client = modal.Client.from_env()
        handles = {}
        roots = {}
        for role in ("control", "artifacts"):
            fact = _volume_fact(facts, role)
            volume = modal.Volume.from_name(
                fact.spec.name,
                environment_name=facts.client_binding.environment_ref,
                create_if_missing=False,
                version=MODAL_VOLUME_V1,
                client=client,
            )
            volume.hydrate(client)
            if getattr(volume, "is_hydrated", False) is not True \
                    or getattr(volume, "object_id", None) != fact.volume_id:
                raise ValueError
            handles[role] = volume
            roots[role] = Path(fact.spec.mount_path)

        class AcknowledgedLayoutObserver:
            def observe(self, expected):
                if expected != facts:
                    raise ValueError
                return ModalRuntimeReleaseDeploymentObservationV1(
                    facts.app_name, facts.client_binding.environment_ref, True,
                    facts.app_id, "", facts.generation,
                    tuple(sorted(
                        (item.spec.name, item.function_id)
                        for item in facts.functions
                    )),
                )

        worker = ModalRuntimeReleaseQualificationWorker(
            expected_facts=facts,
            verifier=authenticator,
            signer=authenticator,
            observer=AcknowledgedLayoutObserver(),
            call_id_provider=modal.current_function_call_id,
            roots=ModalRuntimeReleaseQualificationRoots(
                roots["control"], roots["artifacts"],
            ),
        )
        return worker(
            dispatch_bytes,
            commit_artifacts=handles["artifacts"].commit,
            commit_control=handles["control"].commit,
        )
    except BaseException:
        return failed


__all__ = ["run_runtime_release_self_check"]
