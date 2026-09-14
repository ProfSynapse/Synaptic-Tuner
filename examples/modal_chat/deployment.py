"""Consumer-owned deployment receipt and exact, read-only Modal observers.

Only an acknowledged deployment made by this instance can be observed. The
receipt binds the fixed engine builder's selection to the SDK-returned Function
definition, app, image, and existing Volume identities. It is not a way to adopt
an unrelated deployment by supplying an expected configuration.
"""

from __future__ import annotations

import hashlib

from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.config import (
    ModalProviderProfileV1,
    ModalRuntimeLockV1,
)
from tuner.execution.providers.modal.coordinator_deployment import (
    build_modal_coordinator_deployment,
)
from tuner.execution.providers.modal.deployment_v1 import ModalDeploymentSpecV1
from tuner.execution.providers.modal.facade import (
    ExplicitModal154ReadFacade,
    MODAL_VOLUME_V1,
)
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1

from .storage import ModalChatStorage


class ModalChatDeploymentError(RuntimeError):
    """Closed failure; a deployment attempt is never automatically repeated."""


class ModalChatScope:
    """One explicit SDK client, authenticated workspace, and existing environment.

    Modal's workspace is the billing-account boundary. This consumer uses its
    authenticated slug for both account_ref and workspace_ref; client_ref is a
    non-secret host identity, never a credential or credential-derived value.
    """

    def __init__(self, *, sdk, client, environment_name: str, client_ref: str):
        if getattr(sdk, "__version__", None) != "1.5.4" or client is None:
            raise ModalChatDeploymentError("modal_chat_sdk_invalid")
        self.sdk, self.client = sdk, client
        self.environment_name = safe_ref(environment_name, "environment_name")
        self.client_ref = safe_ref(client_ref, "client_ref")
        workspace, environment_id = self._read()
        self._workspace, self._environment_id = workspace, environment_id
        self.binding = ModalClientBinding(
            workspace, workspace, environment_name, client_ref, "1.5.4"
        )

    def _read(self):
        try:
            workspace = self.sdk.Workspace.from_context(client=self.client)
            workspace.hydrate(self.client)
            environment = self.sdk.Environment.from_name(
                self.environment_name, create_if_missing=False, client=self.client
            )
            environment.hydrate(self.client)
            if workspace.is_hydrated is not True or environment.is_hydrated is not True:
                raise ValueError
            if environment.name != self.environment_name:
                raise ValueError
            return safe_ref(workspace.name, "workspace"), safe_ref(
                environment.object_id, "environment_id"
            )
        except Exception:
            raise ModalChatDeploymentError("modal_chat_scope_unavailable") from None

    def observe(self, supplied_client):
        if supplied_client is not self.client:
            raise ModalChatDeploymentError("modal_chat_scope_mismatch")
        if self._read() != (self._workspace, self._environment_id):
            raise ModalChatDeploymentError("modal_chat_scope_mismatch")
        return (
            self.binding.account_ref,
            self.binding.workspace_ref,
            self.binding.environment_ref,
            self.binding.client_ref,
        )


class ModalChatOwnedDeployment:
    """Single-process deployment ownership; no adoption, mutation retry, or teardown."""

    def __init__(
        self,
        *,
        scope: ModalChatScope,
        storage: ModalChatStorage,
        profile: ModalProviderProfileV1,
        runtime_environment: dict[str, str],
        timeout_seconds: int,
        evidence_environment_key: str,
        evidence_key_ref: str,
        model_token_key: str,
    ):
        if type(scope) is not ModalChatScope or type(storage) is not ModalChatStorage:
            raise TypeError("exact scope and consumer storage required")
        if type(profile) is not ModalProviderProfileV1 or len(profile.secrets) != 1:
            raise TypeError("exact single-secret Modal profile required")
        secret = profile.secrets[0]
        self._scope, self._storage, self._profile = scope, storage, profile
        self._spec = ModalDeploymentSpecV1(
            profile.deployment_ref,
            profile.function_name,
            ModalRuntimeLockV1.packaged().registry_reference,
            profile.control_volume_ref,
            profile.artifact_volume_ref,
            secret.name,
            secret.required_keys,
            runtime_environment,
            timeout_seconds,
        )
        self._selection = ModalDeploymentSelectionV1.from_profile(
            profile,
            binding=scope.binding,
            runtime_environment=self._spec.environment,
            timeout_seconds=timeout_seconds,
        )
        self._keys = dict(
            evidence_environment_key=evidence_environment_key,
            evidence_key_ref=evidence_key_ref,
            model_token_key=model_token_key,
        )
        self._receipt = None
        self._candidate_receipt = None
        self._attempted = False

    @property
    def candidate_receipt(self) -> bytes | None:
        """Non-authorizing diagnostics, including when durable storage fails."""
        return (
            None if self._candidate_receipt is None else bytes(self._candidate_receipt)
        )

    @property
    def selection(self):
        return ModalDeploymentSelectionV1.from_dict(self._selection.to_dict())

    def _function_identity(self, function):
        # Pinned SDK 1.5.4 read-only metadata; never deserialize remote code.
        metadata = function._get_metadata()
        if function.is_hydrated is not True or metadata.web_url:
            raise ValueError("deployment function is unavailable or public")
        if metadata.function_name != self._selection.function_name:
            raise ValueError("deployment function name differs")
        return {
            "function_id": safe_ref(function.object_id, "function_id"),
            "definition_id": safe_ref(metadata.definition_id, "definition_id"),
            "app_id": safe_ref(metadata.app_id, "app_id"),
        }

    def deploy_once(self, *, attempt_ref: str) -> bytes:
        """Deploy the fixed engine builder after a permanent local attempt claim.

        Volumes and Secret must already exist in the selected environment. This
        creates/updates the selected app deployment but does not invoke training.
        Any failed or interrupted attempt retains its claim; no cleanup is inferred.
        """
        if self._attempted:
            raise ModalChatDeploymentError("modal_chat_deployment_already_attempted")
        attempt_ref = safe_ref(attempt_ref, "attempt_ref")
        group = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-deployment-group/v1",
                "account_ref": self._selection.account_ref,
                "workspace_ref": self._selection.workspace_ref,
                "environment_ref": self._selection.environment_ref,
                "app_name": self._selection.app_name,
            }
        )
        self._attempted = True
        # App.deploy replaces the app definition, not merely one function.
        # A new attempt or function name must not evade an unresolved app claim.
        self._storage.attempts.claim(
            "deployment-" + hashlib.sha256(group).hexdigest(), group
        )
        self._storage.attempts.claim(
            attempt_ref,
            canonical_bytes(
                {
                    "schema_version": "synaptic-modal-chat-deployment-attempt/v1",
                    "selection": self._selection.to_dict(),
                }
            ),
        )
        try:
            scope = self._scope
            scope.observe(scope.client)
            volumes = []
            for name in (
                self._profile.control_volume_ref,
                self._profile.artifact_volume_ref,
            ):
                volume = scope.sdk.Volume.from_name(
                    name,
                    environment_name=scope.environment_name,
                    create_if_missing=False,
                    version=MODAL_VOLUME_V1,
                    client=scope.client,
                )
                volume.hydrate(scope.client)
                if volume.is_hydrated is not True:
                    raise ValueError
                volumes.append(safe_ref(volume.object_id, "volume_id"))
            if volumes[0] == volumes[1]:
                raise ValueError
            objects = build_modal_coordinator_deployment(
                sdk=scope.sdk,
                client=scope.client,
                environment_name=scope.environment_name,
                spec=self._spec,
                profile=self._profile,
                deployment_selection_bytes=canonical_bytes(self._selection.to_dict()),
                provider_id="modal",
                profile_ref=self._profile.profile,
                executor_id="modal-coordinator-executor",
                executor_implementation_version="0.1.0",
                control_volume_id=volumes[0],
                artifact_volume_id=volumes[1],
                **self._keys,
            )
            objects.app.deploy(
                client=scope.client, environment_name=scope.environment_name
            )
            dispatch_ack = canonical_bytes(
                {
                    "schema_version": "synaptic-modal-chat-deployment-return/v1",
                    "selection": self._selection.to_dict(),
                    "deployment_returned": True,
                    "definition_identity_verified": False,
                    "provider_shutdown_proof": False,
                }
            )
            returned = self._storage.catalog(
                "deployment-returned", encode=bytes, decode=bytes
            )
            returned.publish_if_absent(attempt_ref, dispatch_ack)
            if returned.resolve(attempt_ref) != dispatch_ack:
                raise ValueError
            identity = self._function_identity(objects.function)
            if objects.app.app_id != identity["app_id"]:
                raise ValueError
            if [
                objects.control_volume.object_id,
                objects.artifact_volume.object_id,
            ] != volumes:
                raise ValueError
            receipt = {
                "schema_version": "synaptic-modal-chat-owned-deployment/v1",
                "selection": self._selection.to_dict(),
                **identity,
                "image_id": safe_ref(objects.image.object_id, "image_id"),
                "control_volume_id": volumes[0],
                "artifact_volume_id": volumes[1],
            }
            # Preserve acknowledged ownership before provider readback can fail.
            # This receipt alone does not claim a successful readback or cleanup.
            payload = canonical_bytes(receipt)
            self._candidate_receipt = payload
            acknowledgements = self._storage.catalog(
                "deployment-acknowledgements", encode=bytes, decode=bytes
            )
            acknowledgements.publish_if_absent(attempt_ref, payload)
            if acknowledgements.resolve(attempt_ref) != payload:
                raise ValueError
            self._receipt = receipt
            self.observe(
                client=scope.client,
                app_name=self._selection.app_name,
                function_name=self._selection.function_name,
                environment_name=scope.environment_name,
            )
            catalog = self._storage.catalog(
                "deployment-results", encode=bytes, decode=bytes
            )
            catalog.publish_if_absent(attempt_ref, payload)
            if catalog.resolve(attempt_ref) != payload:
                raise ValueError
            return payload
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatDeploymentError("modal_chat_deployment_failed") from None

    def observe(self, *, client, app_name, function_name, environment_name):
        if self._receipt is None:
            raise ModalChatDeploymentError("modal_chat_deployment_not_owned")
        scope = self._scope
        try:
            if (
                client is not scope.client
                or app_name != self._selection.app_name
                or function_name != self._selection.function_name
                or environment_name != scope.environment_name
            ):
                raise ValueError
            scope.observe(client)
            function = scope.sdk.Function.from_name(
                app_name,
                function_name,
                environment_name=environment_name,
                client=client,
            )
            function.hydrate(client)
            identity = self._function_identity(function)
            if any(self._receipt[name] != value for name, value in identity.items()):
                raise ValueError
            return self.selection
        except Exception:
            raise ModalChatDeploymentError("modal_chat_deployment_changed") from None

    def facade(self):
        if self._receipt is None:
            raise ModalChatDeploymentError("modal_chat_deployment_not_owned")
        return ExplicitModal154ReadFacade(
            self._scope.binding,
            sdk=self._scope.sdk,
            client=self._scope.client,
            scope_observer=self._scope.observe,
            deployment_observer=self.observe,
            volume_names={
                self._receipt["control_volume_id"]: self._profile.control_volume_ref,
                self._receipt["artifact_volume_id"]: self._profile.artifact_volume_ref,
            },
        )


__all__ = ["ModalChatDeploymentError", "ModalChatScope", "ModalChatOwnedDeployment"]
