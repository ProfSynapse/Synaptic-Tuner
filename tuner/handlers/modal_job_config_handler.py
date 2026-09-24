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
