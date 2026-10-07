"""Explicit one-shot provisioning for consumer-owned Modal resources."""

from __future__ import annotations

import hashlib

from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.providers.modal.facade import MODAL_VOLUME_V1

from .deployment import ModalChatScope
from .storage import ModalChatStorage


class ModalChatProvisioningError(RuntimeError):
    """Closed provisioning result; a failed group is never retried automatically."""


def _canonical_payload(value):
    if type(value) is not bytes:
        raise TypeError("exact canonical bytes required")
    return value


class ModalChatProvisioner:
    """Create exactly three fresh Volumes and one fresh runtime Secret."""

    def __init__(
        self,
        *,
        scope: ModalChatScope,
        storage: ModalChatStorage,
        training_control: str,
        artifacts: str,
        model_cache: str,
        runtime_secret: str,
        model_key_name: str,
        evidence_key_name: str,
        secret_values: dict[str, str],
    ) -> None:
        if type(scope) is not ModalChatScope or type(storage) is not ModalChatStorage:
            raise TypeError("exact scope and storage required")
        names = tuple(
            safe_ref(value, label)
            for value, label in (
                (training_control, "training_control"),
                (artifacts, "artifacts"),
                (model_cache, "model_cache"),
                (runtime_secret, "runtime_secret"),
                (model_key_name, "model_key_name"),
                (evidence_key_name, "evidence_key_name"),
            )
        )
        volume_names = names[:3]
        key_names = names[4:]
        if len(set(volume_names)) != 3 or len(set(key_names)) != 2:
            raise ValueError("provisioned resource roles must be distinct")
        if any(key in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} for key in key_names):
            raise ValueError("Modal credentials cannot enter the runtime Secret")
        if type(secret_values) is not dict or set(secret_values) != set(key_names):
            raise ValueError("runtime Secret values are incomplete")
        if any(
            type(value) is not str or not value.strip()
            for value in secret_values.values()
        ):
            raise ValueError("runtime Secret values are incomplete")
        self._scope, self._storage = scope, storage
        self._volumes = volume_names
        self._secret_name = names[3]
        self._secret_values = dict(secret_values)
        self._attempted = False
        self._acks = storage.catalog(
            "provisioning-acks",
            encode=_canonical_payload,
            decode=_canonical_payload,
        )
        group = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-provisioning-group/v1",
                "environment": scope.environment_name,
                "volumes": list(volume_names),
                "runtime_secret": self._secret_name,
                "secret_keys": sorted(key_names),
            }
        )
        self._group_evidence = group
        self._group_ref = safe_ref(
            "provision-" + hashlib.sha256(group).hexdigest(),
            "provisioning_group",
        )

    def _record(self, item_ref, *, kind, name, created, object_id=None):
        payload = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-provisioning-ack/v1",
                "kind": kind,
                "name": name,
                "creation_acknowledged": created,
                "object_id": object_id,
            }
        )
        self._acks.publish_if_absent(item_ref, payload)

    def _hydrate(self, kind, name):
        options = (
            {"create_if_missing": False}
            if kind == "Volume"
            else {"required_keys": sorted(self._secret_values)}
        )
        resource = getattr(self._scope.sdk, kind).from_name(
            name,
            environment_name=self._scope.environment_name,
            client=self._scope.client,
            **options,
        )
        resource.hydrate(self._scope.client)
        if resource.is_hydrated is not True:
            raise ValueError
        object_id = safe_ref(resource.object_id, "object_id")
        self._record(
            f"{self._group_ref}-{kind.lower()}-{name}-hydrated",
            kind=kind.lower(),
            name=name,
            created=True,
            object_id=object_id,
        )
        return object_id

    def provision_once(self, *, attempt_ref: str):
        if self._attempted:
            raise ModalChatProvisioningError(
                "modal_chat_provisioning_already_attempted"
            )
        attempt_ref = safe_ref(attempt_ref, "attempt_ref")
        self._attempted = True
        try:
            # The deterministic claim prevents a renamed attempt from retrying this group.
            self._storage.attempts.claim(self._group_ref, self._group_evidence)
            self._storage.attempts.claim(
                attempt_ref,
                canonical_bytes(
                    {
                        "schema_version": "synaptic-modal-chat-provisioning-attempt/v1",
                        "group_ref": self._group_ref,
                    }
                ),
            )
            # Resource rows acknowledge completed calls; the permanent group
            # claim above, not those rows, forbids every renamed retry.
            self._scope.observe(self._scope.client)
            result = {}
            for role, name in zip(
                ("training_control", "artifacts", "model_cache"),
                self._volumes,
                strict=True,
            ):
                self._scope.sdk.Volume.objects.create(
                    name,
                    version=MODAL_VOLUME_V1,
                    allow_existing=False,
                    environment_name=self._scope.environment_name,
                    client=self._scope.client,
                )
                self._record(
                    f"{self._group_ref}-volume-{name}-created",
                    kind="volume",
                    name=name,
                    created=True,
                )
                result[role] = self._hydrate("Volume", name)
                self._scope.observe(self._scope.client)
            self._scope.sdk.Secret.objects.create(
                self._secret_name,
                self._secret_values,
                allow_existing=False,
                environment_name=self._scope.environment_name,
                client=self._scope.client,
            )
            self._record(
                f"{self._group_ref}-secret-{self._secret_name}-created",
                kind="secret",
                name=self._secret_name,
                created=True,
            )
            result["runtime_secret"] = self._hydrate("Secret", self._secret_name)
            self._scope.observe(self._scope.client)
            return result
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatProvisioningError("modal_chat_provisioning_failed") from None


__all__ = ["ModalChatProvisioner", "ModalChatProvisioningError"]
