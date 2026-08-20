"""Import-light facade for shared cloud abstractions."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "BootstrapError": ".bootstrap_core",
    "reconstruct_source_lock": ".bootstrap_core",
    "reconstruct_source_lock_json": ".bootstrap_core",
    "CAPSULE_MANIFEST": ".bootstrap_capsule",
    "CAPSULE_MODULE_PATHS": ".bootstrap_capsule",
    "CAPSULE_SCHEMA": ".bootstrap_capsule",
    "CapsuleBuild": ".bootstrap_capsule",
    "CapsuleError": ".bootstrap_capsule",
    "authenticate_external_input": ".bootstrap_capsule",
    "build_capsule": ".bootstrap_capsule",
    "invoke_verified_capsule": ".bootstrap_capsule",
    "verified_capsule_scratch": ".bootstrap_capsule",
    "CloudJobSpec": ".hf_jobs",
    "HF_BUCKET_SYNC_OVERLAY_PACKAGES": ".hf_jobs",
    "HFJobExecutor": ".hf_jobs",
    "HFJobSubmission": ".hf_jobs",
    "HFBootstrapSmokeObservation": ".hf_jobs",
    "HFBootstrapSmokeSubmission": ".hf_jobs",
    "RepoCheckoutSpec": ".hf_jobs",
    "build_bash_command": ".hf_jobs",
    "build_hf_job_secrets": ".hf_jobs",
    "build_secrets_from_env": ".hf_jobs",
    "build_repo_checkout_steps": ".hf_jobs",
    "decode_hf_job_label": ".hf_jobs",
    "format_timeout_hours": ".hf_jobs",
    "load_huggingface_hub": ".hf_jobs",
    "resolve_hf_bucket_id": ".hf_jobs",
    "observe_submitted_bootstrap_smoke": ".hf_jobs",
    "sanitize_hf_job_labels": ".hf_jobs",
    "submit_approved_bootstrap_smoke": ".hf_jobs",
    "CheckoutPolicy": ".checkout",
    "SSHCheckoutPolicy": ".checkout",
    "CheckoutResult": ".checkout",
    "build_source_lock": ".checkout",
    "checkout_policy_from_context": ".checkout",
    "checkout_source_lock": ".checkout",
    "standalone_credential_from_environment": ".checkout",
    "ssh_checkout_policy_from_environment": ".checkout",
    "validate_source_lock_for_cloud": ".checkout",
    "RUNTIME_LAYOUT_SCHEMA": ".runtime_layout",
    "CloudRuntimeLayout": ".runtime_layout",
    "RuntimeMount": ".runtime_layout",
    "build_runtime_layout": ".runtime_layout",
}

__all__ = [
    "BootstrapError",
    "reconstruct_source_lock",
    "reconstruct_source_lock_json",
    "CAPSULE_MANIFEST",
    "CAPSULE_MODULE_PATHS",
    "CAPSULE_SCHEMA",
    "CapsuleBuild",
    "CapsuleError",
    "authenticate_external_input",
    "build_capsule",
    "invoke_verified_capsule",
    "verified_capsule_scratch",
    "CloudJobSpec",
    "HF_BUCKET_SYNC_OVERLAY_PACKAGES",
    "HFJobExecutor",
    "HFJobSubmission",
    "HFBootstrapSmokeObservation",
    "HFBootstrapSmokeSubmission",
    "RepoCheckoutSpec",
    "build_bash_command",
    "build_hf_job_secrets",
    "build_secrets_from_env",
    "build_repo_checkout_steps",
    "decode_hf_job_label",
    "format_timeout_hours",
    "load_huggingface_hub",
    "resolve_hf_bucket_id",
    "observe_submitted_bootstrap_smoke",
    "sanitize_hf_job_labels",
    "submit_approved_bootstrap_smoke",
    "CheckoutPolicy",
    "SSHCheckoutPolicy",
    "CheckoutResult",
    "build_source_lock",
    "checkout_policy_from_context",
    "checkout_source_lock",
    "standalone_credential_from_environment",
    "ssh_checkout_policy_from_environment",
    "validate_source_lock_for_cloud",
    "RUNTIME_LAYOUT_SCHEMA",
    "CloudRuntimeLayout",
    "RuntimeMount",
    "build_runtime_layout",
]


def __getattr__(name: str):
    """Load a requested cloud surface without eager provider-side imports."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
