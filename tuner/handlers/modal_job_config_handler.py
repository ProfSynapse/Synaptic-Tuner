"""Lightweight config-first train handler; importing it cannot install UI extras."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import re
import os

from tuner.handlers.base import BaseHandler
from tuner.project import ProjectContext
from tuner.training.modal_metrics_qualification import MetricsQualificationUnavailable


def _closed_bootstrap_details(error: BaseException) -> dict[str, object] | None:
    """Expose only reviewed, non-authorizing bootstrap diagnostics."""
    try:
        from tuner.training.modal_local_preparation import ModalLocalPreparationUnavailable
        if type(error) is ModalLocalPreparationUnavailable and error.code in error.CODES:
            return {"phase": "LOCAL_" + error.code,
                    "failure_class": "UNAVAILABLE" if error.code.endswith("UNAVAILABLE") else "INVALID",
                    "location": "modal_local_preparation.prepare_local_inputs", "retry_authorized": False}
        from tuner.training.modal_host_runtime import ModalHostBootstrapUnavailable
        from tuner.training.modal_host_qualification import ModalHostQualificationUnavailable
        from tuner.execution.providers.modal.runtime_release_qualification import PARENT_RUNTIME_FAILURE_STAGES
        from tuner.training.modal_standalone_runner import ModalStandalonePhaseUnavailable
        from tuner.execution.providers.modal.packaged_worker import PACKAGED_WORKER_FAILURE_STAGES
    except Exception:
        return None

    closed = {
        "SOURCE_WHEEL": ("runtime_build.prepare_current_source_wheel", {
            "SOURCE_ARCHIVE_INVALID", "LOCAL_BUILD_FAILED", "SOURCE_STATE_INVALID",
            "INPUT_INVALID", "HEAD_BEFORE_UNAVAILABLE", "HEAD_BEFORE_MISMATCH",
            "STATUS_BEFORE_TIMEOUT", "STATUS_BEFORE_UNAVAILABLE", "STATUS_BEFORE_DIRTY",
            "HEAD_AFTER_UNAVAILABLE", "HEAD_AFTER_MISMATCH",
            "STATUS_AFTER_TIMEOUT", "STATUS_AFTER_UNAVAILABLE", "STATUS_AFTER_DIRTY",
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
            details = {
                "phase": error.phase,
                "failure_class": error.failure_class,
                "location": error.location,
                "retry_authorized": False,
            }
            if error.phase == "IMAGE_BUILD" and error.failure_class == "OPERATION_FAILED":
                app_id, image_id = error.build_app_id, error.image_id
                if (app_id is not None and (
                        type(app_id) is not str
                        or re.fullmatch(r"ap-[A-Za-z0-9]{1,64}", app_id) is None)
                        or image_id is not None and (
                            type(image_id) is not str
                            or re.fullmatch(r"im-[A-Za-z0-9]{1,64}", image_id) is None)):
                    return None
                if app_id is not None:
                    details["build_app_id"] = app_id
                if image_id is not None:
                    details["image_id"] = image_id
            return details
    qualification = {
        "FIXTURE_STAGE": ("UNAVAILABLE", "modal_host_qualification.stage_fixture"),
        "TRANSPORT_MARKER_STAGE": ("UNAVAILABLE", "modal_host_qualification.stage_transport_marker"),
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
        **{"CALL_" + stage: ("UNAVAILABLE", "modal_runtime_release_qualification.parent_runtime")
           for stage in PARENT_RUNTIME_FAILURE_STAGES},
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
        "RUN_RETAIN_READ_INPUTS": ("UNAVAILABLE", "modal_standalone_runner.retain_read_inputs"),
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
        "RUN_EVALUATION_READ": ("UNAVAILABLE", "modal_standalone_runner.evaluation_read"),
        "RUN_EVALUATION_GATE": ("UNAVAILABLE", "modal_standalone_runner.evaluation_gate"),
        "RUN_WORKER_FAILED": ("UNAVAILABLE", "modal_standalone_runner.worker_result"),
    }
    run_phase.update({
        "RUN_WORKER_" + stage: ("UNAVAILABLE", "modal_standalone_runner.worker_result")
        for stage in PACKAGED_WORKER_FAILURE_STAGES
    })
    if type(error) is ModalStandalonePhaseUnavailable and error.retry_authorized is False:
        admitted = run_phase.get(error.phase)
        if admitted == (error.failure_class, error.location):
            details = {
                "phase": error.phase,
                "failure_class": error.failure_class,
                "location": error.location,
                "retry_authorized": False,
            }
            from synaptic_tuner.api.v1.results import TrainingRunRef
            retained_run = getattr(error, "retained_run", None)
            if type(retained_run) is TrainingRunRef:
                details["run_ref"] = retained_run.to_dict()
                details["retention_ready"] = True
            if error.phase == "RUN_START_INDETERMINATE":
                try:
                    from tuner.training.modal_standalone_runner import validate_start_failure_causes
                    causes = validate_start_failure_causes(getattr(error, "start_causes", ()))
                except Exception:
                    causes = []
                if causes:
                    details["exception_chain"] = causes
            return details
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
                    "plan", "quote", "qualify", "qualify_metrics", "prepare_local")) > 1:
                self.output_error("Select one Modal recipe mode", code="MODAL_RECIPE_MODE_INVALID")
                return 2
            attempt = getattr(self.args, "metrics_qualification_attempt", None)
            generation_fields = ("qualification_account", "qualification_workspace", "qualification_output")
            execute_admission = getattr(self.args, "qualify_metrics", None)
            invalid_flags = (
                (attempt is not None and not getattr(self.args, "plan", False))
                or (any(getattr(self.args, name, None) is not None for name in generation_fields) and attempt is None)
                or (getattr(self.args, "qualification_sha256", None) is not None and execute_admission is None)
                or (execute_admission is not None and not getattr(self.args, "qualification_sha256", None))
                or ((attempt is not None or execute_admission is not None) and getattr(self.args, "fresh_attempt", False))
                or (getattr(self.args, "prepare_local", False) and getattr(self.args, "fresh_attempt", False))
            )
            if invalid_flags:
                self.output_error("Invalid metrics qualification mode", code="MODAL_RECIPE_MODE_INVALID")
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
                document = plan.to_dict()
                from tuner.training.modal_local_preparation import local_readiness
                from tuner.training.modal_recipe import resolve_modal_sft_build
                from tuner.runtime_profiles import load_runtime_profile
                selected = load_runtime_profile(
                    plan.recipe.runtime_profile, profiles_root,
                ).resolve(model=plan.recipe.model.ref, model_revision=plan.recipe.model.revision, method="sft")
                profile_path, _ = resolve_modal_sft_build(selected, profiles_root, plan.recipe.model.ref, plan.recipe.model.revision)
                document["local_readiness"] = local_readiness(profile_path)
                if attempt is not None:
                    from tuner.training.modal_metrics_qualification import candidate_admission, current_clean_commit, write_candidate
                    import hashlib
                    raw = candidate_admission(
                        plan, engine_commit=current_clean_commit(self.engine_root), attempt_ref=attempt,
                        modal_profile=getattr(self.args, "modal_profile", None),
                        modal_environment=getattr(self.args, "modal_environment", None),
                        account_ref=getattr(self.args, "qualification_account", None),
                        workspace_ref=getattr(self.args, "qualification_workspace", None),
                    )
                    destination = getattr(self.args, "qualification_output", None)
                    if destination is not None:
                        target = Path(os.path.abspath(destination))
                        if target.is_relative_to(self.engine_root.resolve()):
                            raise ValueError
                        write_candidate(target, raw)
                    document["metrics_qualification_candidate"] = {
                        "canonical_json": raw.decode("utf-8"), "sha256": hashlib.sha256(raw).hexdigest(),
                        "admission_path": destination, "execution_authorized": False,
                    }
                self.output(document)
                return 0
            if getattr(self.args, "qualify", False):
                return self._qualify(plan)
            if getattr(self.args, "prepare_local", False):
                try:
                    from tuner.training.modal_standalone_runner import prepare_modal_standalone_local
                    prepared = prepare_modal_standalone_local(plan=plan, context=self.context)
                    try:
                        self.output(prepared.report())
                    finally:
                        prepared.close()
                    return 0
                except Exception as error:
                    self.output_error("Local Modal inputs could not be prepared", code="MODAL_LOCAL_PREPARATION_UNAVAILABLE",
                                      details=_closed_bootstrap_details(error))
                    return 2
            return self._execute(plan)
        except (OSError, TypeError, ValueError, yaml.YAMLError, MetricsQualificationUnavailable):
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
            admission = None
            if getattr(self.args, "qualify_metrics", None) is not None:
                from tuner.training.modal_metrics_qualification import load_admission
                admission = load_admission(Path(self.args.qualify_metrics), self.args.qualification_sha256)

            result = run_modal_standalone_job(
                plan=plan, context=self.context,
                modal_profile=profile, modal_environment=environment,
                fresh_attempt=bool(getattr(self.args, "fresh_attempt", False)),
                **({"metrics_admission": admission} if admission is not None else {}),
            )
            evaluation_passed = getattr(result, "evaluation_passed", None)
            evaluation_path = getattr(result, "evaluation_path", None)
            metrics = getattr(result, "metrics_qualification", None)
            succeeded = evaluation_passed is not False and (metrics is None or metrics["delivery_verdict"] == "delivered_precompletion" and metrics["durability_acknowledged"] is True)
            self.output({
                "schema_version": "synaptic-modal-sft-run-result/v1",
                "run": result.run.to_dict(),
                "verified_artifacts": [str(path) for path in result.artifact_paths],
                "evaluation_path": None if evaluation_path is None else str(evaluation_path),
                "evaluation_passed": evaluation_passed,
                "gpu_only_timeout_estimate_minor_units": result.gpu_only_timeout_estimate_minor_units,
                "operator_maximum_cost_minor_units": result.maximum_cost_minor_units,
                "currency": "USD",
                "authorization_semantics": "operator maximum, not provider billing cap",
                "excluded_billing_dimensions": [
                    "build", "cpu", "memory", "storage", "usage-beyond-timeout",
                ],
                **({"progress": result.progress} if getattr(result, "progress", None) is not None else {}),
                **({"metrics_qualification": metrics} if metrics is not None else {}),
            }, success=succeeded)
            return 0 if succeeded else 2
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
