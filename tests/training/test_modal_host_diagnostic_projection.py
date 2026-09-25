"""Closed, non-retryable host bootstrap details shown by the train handler."""

import pytest

from tuner.handlers.modal_job_config_handler import _closed_bootstrap_details
from tuner.training.modal_host_runtime import ModalHostBootstrapUnavailable
from tuner.training.modal_host_qualification import ModalHostQualificationUnavailable
from tuner.training.modal_standalone_runner import ModalStandalonePhaseUnavailable


@pytest.mark.parametrize("diagnosis", (
    "SOURCE_ARCHIVE_INVALID", "SOURCE_WHEEL_LOCAL_BUILD_FAILED",
    "SOURCE_WHEEL_SOURCE_STATE_INVALID", "SOURCE_WHEEL_BUILDER_SETUP_FAILED",
    "SOURCE_WHEEL_OFFLINE_WHEEL_TIMEOUT", "SOURCE_WHEEL_OFFLINE_WHEEL_FAILED",
    "SOURCE_WHEEL_WHEEL_INVENTORY_INVALID",
    "BUILD_INPUTS_INVALID",
    "APP_START_TIMEOUT", "APP_START_OPERATION_FAILED",
    "APP_CLEANUP_TIMEOUT", "APP_CLEANUP_OPERATION_FAILED",
    "IMAGE_BUILD_TIMEOUT", "IMAGE_BUILD_OPERATION_FAILED", "IMAGE_BUILD_IDENTITY_MISSING",
    "CAPTURE_CREATE_TIMEOUT", "CAPTURE_CREATE_OPERATION_FAILED",
    "CAPTURE_OUTPUT_TIMEOUT", "CAPTURE_OUTPUT_OPERATION_FAILED",
    "CAPTURE_OUTPUT_INSPECTOR_REJECTED", "CAPTURE_OUTPUT_OUTPUT_INVALID",
    "CAPTURE_CLEANUP_TIMEOUT", "CAPTURE_CLEANUP_OPERATION_FAILED",
    "CAPTURE_VALIDATE_INVALID", "RELEASE_VALIDATE_INVALID",
    "RELEASE_OBSERVE_SCOPE_UNAVAILABLE",
    "RELEASE_OBSERVE_OBSERVATION_UNAVAILABLE",
    "RELEASE_OBSERVE_OBSERVATION_INVALID",
    "RELEASE_ATTEMPT_BOUNDED_DEPLOY_UNAVAILABLE",
    "RELEASE_ATTEMPT_SCOPE_UNAVAILABLE",
    "RELEASE_ATTEMPT_OBSERVATION_UNAVAILABLE",
    "RELEASE_ATTEMPT_OBSERVATION_INVALID",
    "RELEASE_ATTEMPT_RESOURCE_UNAVAILABLE",
    "RELEASE_ATTEMPT_CONSTRUCTION_FAILED",
    "RELEASE_ATTEMPT_DEPLOYMENT_INDETERMINATE",
    "RELEASE_ATTEMPT_ACKNOWLEDGEMENT_INVALID",
))
def test_known_bootstrap_diagnosis_projects_only_closed_fields(diagnosis):
    error = ModalHostBootstrapUnavailable(diagnosis)
    assert _closed_bootstrap_details(error) == {
        "phase": error.phase,
        "failure_class": error.failure_class,
        "location": error.location,
        "retry_authorized": False,
    }


def test_bootstrap_diagnosis_subclass_cannot_project_hostile_fields():
    class HostileBootstrapError(ModalHostBootstrapUnavailable):
        @property
        def location(self):
            return "HF_TOKEN=private /home/owner/dataset.jsonl"

    assert _closed_bootstrap_details(HostileBootstrapError()) is None


def test_unknown_bootstrap_exception_has_no_details():
    assert _closed_bootstrap_details(
        RuntimeError("HF_TOKEN=private /home/owner/dataset.jsonl")
    ) is None


@pytest.mark.parametrize("phase", (
    "FIXTURE_STAGE", "DISPATCH_SUBMIT", "DISPATCH_FUNCTION_IDENTITY",
    "DISPATCH_SPAWN_INDETERMINATE", "DISPATCH_CATALOG_INDETERMINATE",
    "CALL_OBSERVE", "CALL_PARENT_SETUP", "CALL_INSTALLED_CHILD",
    "CALL_PARENT_RELEASE", "CALL_CHILD_RESULT",
    "RECEIPT_VERIFY",
))
def test_cpu_qualification_diagnosis_projects_only_closed_fields(phase):
    error = ModalHostQualificationUnavailable(phase)
    assert _closed_bootstrap_details(error) == {
        "phase": error.phase,
        "failure_class": error.failure_class,
        "location": error.location,
        "retry_authorized": False,
    }


@pytest.mark.parametrize("phase", (
    "RUN_HOST_ASSEMBLY", "RUN_PUBLIC_PREPARE", "RUN_PUBLIC_LOAD",
    "RUN_PUBLIC_RESOLVE", "RUN_RESOLVE_RICH", "RUN_RESOLVE_DERIVE",
    "RUN_RESOLVE_REPARSE", "RUN_PUBLIC_PLAN", "RUN_PUBLIC_PREFLIGHT",
    "RUN_START_INDETERMINATE",
))
def test_standalone_run_diagnosis_projects_only_closed_fields(phase):
    error = ModalStandalonePhaseUnavailable(phase)
    assert _closed_bootstrap_details(error) == {
        "phase": error.phase,
        "failure_class": error.failure_class,
        "location": error.location,
        "retry_authorized": False,
    }


def test_standalone_run_diagnosis_subclass_cannot_project_hostile_fields():
    class HostileRunError(ModalStandalonePhaseUnavailable):
        @property
        def location(self):
            return "HF_TOKEN=private /home/owner/dataset.jsonl"

        @location.setter
        def location(self, _value):
            pass

    assert _closed_bootstrap_details(HostileRunError("RUN_HOST_ASSEMBLY")) is None
