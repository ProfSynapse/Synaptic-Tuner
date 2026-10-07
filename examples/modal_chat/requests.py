"""Consumer-owned canonical request bridge for the Modal coordinator example."""

from __future__ import annotations

import hashlib
import json

from synaptic_tuner.api.v1.planning import ResolvedTrainingRequest, TrainingPlan
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.training import TrainingService
from tuner.training.contracts import CanonicalDocument
from tuner.training.coordinator_material import (
    CoordinatorResolvedMaterial,
    derive_coordinator_material,
)
from tuner.training.recipes import RecipeRegistry


class ModalChatRequestError(RuntimeError):
    """Closed failure at the consumer request/material boundary."""


class ModalChatTrainingRequests:
    """Compile real inputs and preserve each load's allocated run identity."""

    __slots__ = (
        "_service",
        "_recipes",
        "_project_ref",
        "_identities",
        "_requests",
        "_materials",
    )

    def __init__(
        self,
        *,
        service: TrainingService,
        recipes: RecipeRegistry,
        project_ref: str,
        identities: object,
        request_catalog: object,
        material_catalog: object,
    ) -> None:
        if type(service) is not TrainingService or type(recipes) is not RecipeRegistry:
            raise TypeError("exact configured training service and recipes required")
        self._service = service
        self._recipes = recipes
        self._project_ref = safe_ref(project_ref, "project_ref")
        for value, methods in (
            (identities, ("allocate",)),
            (request_catalog, ("resolve", "publish_if_absent")),
            (material_catalog, ("resolve", "publish_if_absent")),
        ):
            if any(not callable(getattr(value, method, None)) for method in methods):
                raise TypeError("request bridge port is incomplete")
        self._identities = identities
        self._requests = request_catalog
        self._materials = material_catalog

    @staticmethod
    def _request_key(request_id: str) -> str:
        return safe_ref(request_id, "request_id")

    @staticmethod
    def _record(payload: bytes) -> dict[str, object]:
        if type(payload) is not bytes:
            raise ValueError("retained request must be bytes")
        value = json.loads(payload.decode("utf-8"))
        if type(value) is not dict or canonical_bytes(value) != payload:
            raise ValueError("retained request is not canonical")
        if (
            set(value) != {"schema_version", "request", "run"}
            or value["schema_version"] != "synaptic-modal-chat-training-request/v1"
        ):
            raise ValueError("retained request is invalid")
        return value

    def load(self, canonical_json: str) -> TrainingRequest:
        """Validate canonical public input and durably bind its allocated identities."""
        try:
            training_input = TrainingInputV1.from_json(canonical_json)
            if training_input.canonical_json() != canonical_json:
                raise ValueError("training input is not canonical")
            request_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
            allocated = self._identities.allocate(
                project_ref=self._project_ref, request_digest=request_digest
            )
            if (
                type(allocated) is not tuple
                or len(allocated) != 2
                or type(allocated[0]) is not str
                or type(allocated[1]) is not TrainingRunRef
                or allocated[1].project_ref != self._project_ref
            ):
                raise TypeError("identity allocation is invalid")
            request = TrainingRequest(
                safe_ref(allocated[0], "request_id"), self._project_ref, canonical_json
            )
            record = canonical_bytes(
                {
                    "schema_version": "synaptic-modal-chat-training-request/v1",
                    "request": {
                        "request_id": request.request_id,
                        "project_ref": request.project_ref,
                        "canonical_json": request.canonical_json,
                    },
                    "run": allocated[1].to_dict(),
                }
            )
            inserted = self._requests.publish_if_absent(
                self._request_key(request.request_id), record
            )
            if (
                type(inserted) is not bool
                or self._requests.resolve(self._request_key(request.request_id))
                != record
            ):
                raise ValueError("retained request collision")
            return request
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatRequestError("modal_chat_request_invalid") from None

    def _retained(self, request_id: str) -> tuple[TrainingRequest, TrainingRunRef]:
        key = self._request_key(request_id)
        record = self._record(self._requests.resolve(key))
        request = record["request"]
        run = record["run"]
        if type(request) is not dict or type(run) is not dict:
            raise ValueError("retained request is invalid")
        retained = TrainingRequest(**request)
        retained_run = TrainingRunRef.from_dict(run)
        if (
            retained.request_id != key
            or retained.project_ref != self._project_ref
            or retained_run.project_ref != self._project_ref
        ):
            raise ValueError("retained request identity differs")
        return retained, retained_run

    def _validated_material(
        self,
        payload: bytes,
        *,
        request: TrainingRequest,
        run: TrainingRunRef,
    ) -> CoordinatorResolvedMaterial:
        material = CoordinatorResolvedMaterial.parse(payload, self._recipes)
        planning = material.planning_request
        if (
            planning.request_id != request.request_id
            or planning.project_ref != request.project_ref
            or material.run_id != run.run_id
            or material.request_bytes
            != canonical_bytes(json.loads(request.canonical_json))
        ):
            raise ValueError("retained material differs from request")
        return material

    def resolve(self, request: TrainingRequest):
        """Resolve through the configured rich service and retain compiled material."""
        try:
            if type(request) is not TrainingRequest:
                raise TypeError("exact public training request required")
            retained, run = self._retained(request.request_id)
            if retained != request or request.project_ref != self._project_ref:
                raise ValueError("request differs from retained identity")
            key = self._request_key(request.request_id)
            payload = self._materials.resolve(key)
            if payload is None:
                rich_request = self._service.load(
                    CanonicalDocument(request.canonical_json)
                )
                rich_resolved = self._service.resolve(rich_request)
                if rich_resolved.execution_source.run_id != run.run_id:
                    raise ValueError("resolved source differs from allocated run")
                material = derive_coordinator_material(
                    rich_resolved,
                    self._recipes,
                    request_id=request.request_id,
                    project_ref=request.project_ref,
                    run_id=run.run_id,
                )
                payload = material.canonical_bytes
                inserted = self._materials.publish_if_absent(key, payload)
                if (
                    type(inserted) is not bool
                    or self._materials.resolve(key) != payload
                ):
                    raise ValueError("retained material collision")
            material = self._validated_material(payload, request=request, run=run)
            return material.planning_request
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatRequestError("modal_chat_request_invalid") from None

    def for_plan(self, plan: TrainingPlan) -> TrainingRunRef:
        """Return the exact run allocated before the plan's rich resolution."""
        try:
            if type(plan) is not TrainingPlan:
                raise TypeError("exact coordinator plan required")
            request, run = self._retained(plan.basis.request_id)
            if (
                request.project_ref != plan.basis.project_ref
                or run.project_ref != plan.basis.project_ref
            ):
                raise ValueError("plan differs from retained identities")
            material = self._validated_material(
                self._materials.resolve(self._request_key(request.request_id)),
                request=request,
                run=run,
            )
            basis = plan.basis.to_dict()
            basis["schema_version"] = "synaptic-resolved-training-request/v1"
            if material.planning_request != ResolvedTrainingRequest.from_dict(basis):
                raise ValueError("plan differs from retained material")
            return run
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatRequestError("modal_chat_request_invalid") from None


__all__ = ["ModalChatRequestError", "ModalChatTrainingRequests"]
