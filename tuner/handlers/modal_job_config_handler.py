"""Lightweight config-first train handler; importing it cannot install UI extras."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tuner.handlers.base import BaseHandler
from tuner.project import ProjectContext


def _closed_bootstrap_details(error: BaseException) -> dict[str, object] | None:
    """Expose only reviewed, non-authorizing bootstrap diagnostics."""
    try:
        from tuner.training.modal_host_runtime import ModalHostBootstrapUnavailable
        from tuner.training.modal_host_qualification import ModalHostQualificationUnavailable
        from tuner.training.modal_standalone_runner import ModalStandalonePhaseUnavailable
        from tuner.execution.providers.modal.packaged_worker import PACKAGED_WORKER_FAILURE_STAGES
    except Exception:
        return None

    closed = {
        "SOURCE_WHEEL": ("runtime_build.prepare_current_source_wheel", {
            "SOURCE_ARCHIVE_INVALID", "LOCAL_BUILD_FAILED", "SOURCE_STATE_INVALID",
            "BUILDER_SETUP_FAILED", "OFFLINE_WHEEL_TIMEOUT", "OFFLINE_WHEEL_FAILED",
            "WHEEL_INVENTORY_INVALID"}),
        "BUILD_INPUTS": ("runtime_build.prepare_build_inputs", {"INVALID"}),
        "APP_START": ("modal_host_runtime.build_app", {"TIMEOUT", "OPERATION_FAILED"}),
        "APP_CLEANUP": ("modal_host_runtime.close_build_app", {"TIMEOUT", "OPERATION_FAILED"}),
        "IMAGE_BUILD": ("runtime_build.build_image", {
            "TIMEOUT", "OPERATION_FAILED", "IDENTITY_MISSING"}),
        "CAPTURE_CREATE": ("runtime_build.create_capture_sandbox", {"TIMEOUT", "OPERATION_FAILED"}),
        "CAPTURE_OUTPUT": ("runtime_build.capture_output", {
            "TIMEOUT", "OPERATION_FAILED", "INSPECTOR_REJECTED", "OUTPUT_INVALID"}),
        "CAPTURE_CLEANUP": ("runtime_build.cleanup_capture_sandbox", {"TIMEOUT", "OPERATION_FAILED"}),
        "CAPTURE_VALIDATE": ("runtime_build.validate_capture", {"INVALID"}),
        "RELEASE_VALIDATE": ("modal_host_runtime.build_release", {"INVALID"}),
        "RELEASE_OBSERVE": ("modal_host_runtime.observe_release", {
            "SCOPE_UNAVAILABLE", "OBSERVATION_UNAVAILABLE", "OBSERVATION_INVALID"}),
        "RELEASE_ATTEMPT": ("modal_host_runtime.deploy_release", {
            "BOUNDED_DEPLOY_UNAVAILABLE", "SCOPE_UNAVAILABLE",
            "OBSERVATION_UNAVAILABLE", "OBSERVATION_INVALID",
            "RESOURCE_UNAVAILABLE", "CONSTRUCTION_FAILED",
            "DEPLOYMENT_INDETERMINATE", "ACKNOWLEDGEMENT_INVALID"}),
    }
    if type(error) is ModalHostBootstrapUnavailable and error.retry_authorized is False:
        admitted = closed.get(error.phase)
        if (admitted is not None and error.location == admitted[0]
                and error.failure_class in admitted[1]):
            return {
                "phase": error.phase,
                "failure_class": error.failure_class,
                "location": error.location,
                "retry_authorized": False,
            }
    qualification = {
        "FIXTURE_STAGE": ("UNAVAILABLE", "modal_host_qualification.stage_fixture"),
        "DISPATCH_SUBMIT": ("INDETERMINATE", "modal_host_qualification.submit_once"),
        "DISPATCH_FUNCTION_IDENTITY": (
            "UNAVAILABLE", "modal_runtime_qualification_operator.function_identity"),
        "DISPATCH_SPAWN_INDETERMINATE": (
            "INDETERMINATE", "modal_runtime_qualification_operator.spawn"),
        "DISPATCH_CATALOG_INDETERMINATE": (
            "INDETERMINATE", "modal_runtime_qualification_operator.call_catalog"),
        "CALL_OBSERVE": ("UNAVAILABLE", "modal_host_qualification.observe_call"),
        "CALL_PARENT_SETUP": (
            "UNAVAILABLE", "runtime_release_modal_self_check.parent_setup"),
        "CALL_INSTALLED_CHILD": (
            "UNAVAILABLE", "modal_runtime_release_qualification.installed_child"),
        "CALL_PARENT_RELEASE": (
            "UNAVAILABLE", "packaged_training_worker.parent_release"),
        "CALL_CHILD_RESULT": (
            "UNAVAILABLE", "packaged_training_worker.child_result"),
        "RECEIPT_VERIFY": ("UNAVAILABLE", "modal_host_qualification.verify_receipt"),
    }
    if type(error) is ModalHostQualificationUnavailable and error.retry_authorized is False:
        admitted = qualification.get(error.phase)
        if admitted == (error.failure_class, error.location):
            return {
                "phase": error.phase,
                "failure_class": error.failure_class,
                "location": error.location,
                "retry_authorized": False,
            }
    run_phase = {
        "RUN_HOST_ASSEMBLY": ("UNAVAILABLE", "modal_standalone_runner.compose_host"),
        "RUN_PUBLIC_PREPARE": ("UNAVAILABLE", "modal_standalone_runner.public_prepare"),
        "RUN_PUBLIC_LOAD": ("UNAVAILABLE", "modal_standalone_runner.public_load"),
        "RUN_PUBLIC_RESOLVE": ("UNAVAILABLE", "modal_standalone_runner.public_resolve"),
        "RUN_RESOLVE_RICH": ("UNAVAILABLE", "modal_standalone_runner.resolve_rich"),
        "RUN_RESOLVE_DERIVE": ("UNAVAILABLE", "modal_standalone_runner.resolve_derive"),
        "RUN_RESOLVE_REPARSE": ("UNAVAILABLE", "modal_standalone_runner.resolve_reparse"),
        "RUN_PUBLIC_PLAN": ("UNAVAILABLE", "modal_standalone_runner.public_plan"),
        "RUN_PUBLIC_PREFLIGHT": ("UNAVAILABLE", "modal_standalone_runner.public_preflight"),
        "RUN_START_INDETERMINATE": ("INDETERMINATE", "modal_standalone_runner.training_start"),
        "RUN_STAGE_RECONCILE_REQUIRED": ("INDETERMINATE", "modal_standalone_runner.stage_reconcile"),
        "RUN_SUBMIT_RECONCILE_REQUIRED": ("INDETERMINATE", "modal_standalone_runner.submit_reconcile"),
        "RUN_WORKFLOW_FAILED": ("UNAVAILABLE", "modal_standalone_runner.workflow_failed"),
        "RUN_WORKFLOW_CONTRADICTED": ("INDETERMINATE", "modal_standalone_runner.workflow_contradicted"),
        "RUN_WORKFLOW_STATE": ("UNAVAILABLE", "modal_standalone_runner.workflow_state"),
        "RUN_READ_BINDING": ("UNAVAILABLE", "modal_standalone_runner.read_binding"),
        "RUN_CALL_OBSERVE": ("INDETERMINATE", "modal_standalone_runner.call_observe"),
        "RUN_OUTCOME": ("UNAVAILABLE", "modal_standalone_runner.outcome"),
        "RUN_VERIFY": ("UNAVAILABLE", "modal_standalone_runner.verify"),
        "RUN_ARTIFACT_DOWNLOAD": ("UNAVAILABLE", "modal_standalone_runner.artifact_download"),
        "RUN_WORKER_FAILED": ("UNAVAILABLE", "modal_standalone_runner.worker_result"),
    }
    run_phase.update({
        "RUN_WORKER_" + stage: ("UNAVAILABLE", "modal_standalone_runner.worker_result")
        for stage in PACKAGED_WORKER_FAILURE_STAGES
    })
    if type(error) is ModalStandalonePhaseUnavailable and error.retry_authorized is False:
        admitted = run_phase.get(error.phase)
        if admitted == (error.failure_class, error.location):
            return {
                "phase": error.phase,
                "failure_class": error.failure_class,
                "location": error.location,
                "retry_authorized": False,
            }
    return None


class ModalJobConfigHandler(BaseHandler):
    def __init__(self, args: Namespace, context: ProjectContext | None = None) -> None:
        super().__init__(args=args, context=context)

    @property
    def name(self) -> str:
        return "train"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        from tuner.training.modal_recipe import load_modal_sft_recipe, plan_modal_sft_recipe
        import yaml

        requested = getattr(self.args, "job_config", None)
        if type(requested) is not str or not requested:
            self.output_error("Training recipe required", code="TRAIN_RECIPE_REQUIRED")
            return 2
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = self.context.invocation_cwd / candidate
        try:
            recipe_path = candidate.resolve(strict=True)
            profiles_root = self.engine_root / "Trainers" / "runtime_profiles"
            if sum(bool(getattr(self.args, name, False)) for name in (
                    "plan", "quote", "qualify")) > 1:
                self.output_error("Select one Modal recipe mode", code="MODAL_RECIPE_MODE_INVALID")
                return 2
            if getattr(self.args, "quote", False):
                # Billing-rate observation needs only the declared resource,
                # never the Windows-mounted dataset or build intent.
                recipe = load_modal_sft_recipe(recipe_path, profiles_root=profiles_root)
                return self._quote({
                    "accelerator": recipe.accelerator,
                    "accelerator_count": recipe.accelerator_count,
                    "timeout_seconds": recipe.timeout_seconds,
                })
            plan = plan_modal_sft_recipe(
                recipe_path, project_root=self.project_root,
                profiles_root=profiles_root,
            )
            if getattr(self.args, "plan", False):
                self.output(plan.to_dict())
                return 0
            if getattr(self.args, "qualify", False):
                return self._qualify(plan)
            return self._execute(plan)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            self.output_error("Modal recipe or prepared dataset is invalid", code="MODAL_RECIPE_INVALID")
            return 2

    def _quote(self, plan: object) -> int:
        profile = getattr(self.args, "modal_profile", None)
        environment = getattr(self.args, "modal_environment", None)
        if type(profile) is not str or not profile or type(environment) is not str or not environment:
            self.output_error("Named Modal profile and environment required", code="MODAL_QUOTE_SCOPE_REQUIRED")
            return 2
        try:
            import modal
            from tuner.training.modal_host_scope import open_modal_host_scope
            from tuner.training.modal_host_runtime import observe_scoped_modal_gpu_rates

            client, binding = open_modal_host_scope(
                sdk=modal, profile=profile, environment_name=environment,
            )
            rates = observe_scoped_modal_gpu_rates(
                sdk=modal, client=client, client_binding=binding,
            )
            resource = (plan.to_dict()["resource_request"] if callable(getattr(plan, "to_dict", None))
                        else plan)
            self.output({
                "schema_version": "synaptic-modal-sft-rate-observation/v1",
                "resource_request": resource,
                "gpu_hourly_rates_usd": rates,
                "authorization": "read-only observation; not a training quote or billing cap",
            })
            return 0
        except Exception:
            self.output_error("Scoped Modal GPU rates unavailable", code="MODAL_QUOTE_UNAVAILABLE")
            return 2

    def _execute(self, plan: object) -> int:
        profile = getattr(self.args, "modal_profile", None)
        environment = getattr(self.args, "modal_environment", None)
        if type(profile) is not str or not profile or type(environment) is not str or not environment:
            self.output_error("Named Modal profile and environment required", code="MODAL_RUN_SCOPE_REQUIRED")
            return 2
        try:
            from tuner.training.modal_standalone_runner import run_modal_standalone_job

            result = run_modal_standalone_job(
                plan=plan, context=self.context,
                modal_profile=profile, modal_environment=environment,
                fresh_attempt=bool(getattr(self.args, "fresh_attempt", False)),
            )
            self.output({
                "schema_version": "synaptic-modal-sft-run-result/v1",
                "run": result.run.to_dict(),
                "verified_artifacts": [str(path) for path in result.artifact_paths],
                "gpu_only_timeout_estimate_minor_units": result.gpu_only_timeout_estimate_minor_units,
                "operator_maximum_cost_minor_units": result.maximum_cost_minor_units,
                "currency": "USD",
                "authorization_semantics": "operator maximum, not provider billing cap",
                "excluded_billing_dimensions": [
                    "build", "cpu", "memory", "storage", "usage-beyond-timeout",
                ],
            })
            return 0
        except Exception as error:
            self.output_error(
                "Modal training did not complete", code="MODAL_TRAINING_UNAVAILABLE",
                details=_closed_bootstrap_details(error),
            )
            return 2

    def _qualify(self, plan: object) -> int:
        profile = getattr(self.args, "modal_profile", None)
        environment = getattr(self.args, "modal_environment", None)
        if type(profile) is not str or not profile or type(environment) is not str or not environment:
            self.output_error("Named Modal profile and environment required", code="MODAL_RUN_SCOPE_REQUIRED")
            return 2
        try:
            from tuner.training.modal_standalone_runner import qualify_modal_standalone_job

            result = qualify_modal_standalone_job(
                plan=plan, context=self.context,
                modal_profile=profile, modal_environment=environment,
                fresh_attempt=bool(getattr(self.args, "fresh_attempt", False)),
            )
            self.output(result.to_dict())
            return 0
        except Exception as error:
            self.output_error(
                "Modal CPU qualification did not complete",
                code="MODAL_QUALIFICATION_UNAVAILABLE",
                details=_closed_bootstrap_details(error),
            )
            return 2
