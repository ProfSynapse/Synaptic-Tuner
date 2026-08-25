"""Provider-neutral cloud training API.

This module is the stable boundary used by CLIs, recipe adapters, and host
projects. Provider implementations remain internal and are resolved lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from tuner.backends.registry import TrainingBackendRegistry
from tuner.backends.training.cloud.base_cloud import resolve_cloud_image
from tuner.cloud import (
    build_runtime_layout,
    build_source_lock,
    checkout_policy_from_context,
    ssh_checkout_policy_from_environment,
    standalone_credential_from_environment,
)
from tuner.cloud.hf_jobs import require_current_hf_source_submission_authorization
from tuner.core.interfaces import ExecuteResult
from tuner.project import ProjectContext


CLOUD_PROVIDERS: Mapping[str, Mapping[str, Any]] = {
    "hf_jobs": {
        "name": "HuggingFace Jobs",
        "description": "Managed GPU training via HF infrastructure",
        "install_hint": "pip install --upgrade huggingface_hub>=0.27.0",
    },
    "modal": {
        "name": "Modal",
        "description": "Serverless GPU compute with auto-scaling",
        "install_hint": "pip install modal && modal setup",
    },
    "runpod": {
        "name": "RunPod",
        "description": "On-demand GPU pods with Docker support",
        "install_hint": "pip install runpod",
    },
}


@dataclass(frozen=True)
class CloudTrainingRequest:
    """A provider-neutral request to train one model on one dataset."""

    provider: str
    method: str
    model_name: str | None = None
    dataset_name: str | None = None
    dataset_file: str | None = None
    training: Mapping[str, Any] = field(default_factory=dict)
    lora: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        method = self.method.strip().lower()
        if not provider or not method:
            raise ValueError("Cloud training provider and method are required")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "method", method)

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> "CloudTrainingRequest":
        """Build a request from a declarative cloud-target recipe."""

        model = recipe.get("model", {})
        dataset = recipe.get("dataset", {})
        model = model if isinstance(model, Mapping) else {}
        dataset = dataset if isinstance(dataset, Mapping) else {}
        cloud = recipe.get("cloud", {})
        cloud = cloud if isinstance(cloud, Mapping) else {}
        job = recipe.get("job", {})
        job = job if isinstance(job, Mapping) else {}
        runtime = dict(cloud)
        for source, target in (
            ("flavor", "gpu_type"),
            ("gpu", "gpu_type"),
            ("timeout_hours", "timeout_hours"),
            ("image", "cloud_image"),
            ("image_profile", "image_profile"),
        ):
            if source in job and target not in runtime:
                runtime[target] = job[source]
        return cls(
            provider=str(recipe.get("provider", "")).strip(),
            method=str(recipe.get("method", "")).strip(),
            model_name=_optional_text(model.get("name") or model.get("model_name")),
            dataset_name=_optional_text(dataset.get("name") or dataset.get("dataset_name")),
            dataset_file=_optional_text(
                dataset.get("file") or dataset.get("dataset_file") or dataset.get("local_file")
            ),
            training=_mapping(recipe.get("training")),
            lora=_mapping(recipe.get("lora")),
            runtime=runtime,
            artifacts=_mapping(recipe.get("artifacts")),
            run_id=_optional_text(recipe.get("run_id")),
        )


@dataclass(frozen=True)
class CloudSourceContract:
    """Validated source and filesystem contract shared by every provider."""

    source_lock: Any
    runtime_layout: Any
    checkout_policy: Any


@dataclass
class CloudTrainingPlan:
    """Prepared, inspectable training plan that has not executed yet."""

    request: CloudTrainingRequest
    summary: Mapping[str, Any]
    source: CloudSourceContract
    _backend: Any = field(repr=False)
    _config: Any = field(repr=False)
    _submitted: bool = field(default=False, init=False, repr=False)


@dataclass(frozen=True)
class CloudTrainingResult:
    """Normalized result returned by all provider implementations."""

    provider: str
    method: str
    exit_code: int
    job_id: str | None = None
    artifact_prefix: str | None = None
    artifact_identifier: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class CloudTrainingAPI:
    """Prepare and submit cloud training without coupling callers to a provider."""

    def __init__(
        self,
        context: ProjectContext,
        *,
        hf_authorizer: Callable[..., Any] | None = None,
    ):
        self.context = context
        self._hf_authorizer = (
            hf_authorizer or require_current_hf_source_submission_authorization
        )
        self._authorized_providers: set[str] = set()

    def _authorize_provider(self, provider: str, *, route: str) -> None:
        if provider != "hf_jobs" or provider in self._authorized_providers:
            return
        self._hf_authorizer(route=route)
        self._authorized_providers.add(provider)

    def provider_statuses(self, *, validate_environment: bool = False) -> list[dict[str, Any]]:
        registered = set(TrainingBackendRegistry.list())
        statuses: list[dict[str, Any]] = []
        for provider, metadata in CLOUD_PROVIDERS.items():
            status = {
                "id": provider,
                "name": metadata["name"],
                "registered": provider in registered,
                "env_ready": False,
                "detail": "",
            }
            if provider not in registered:
                status["detail"] = f"Not installed (run: {metadata['install_hint']})"
            elif not validate_environment:
                status["detail"] = "Registered; credentials not checked in inspection mode"
            else:
                try:
                    backend = TrainingBackendRegistry.get(provider, repo_root=self.context.engine_root)
                    ready, error = backend.validate_environment()
                    status["env_ready"] = ready
                    status["detail"] = "" if ready else error
                except Exception as exc:
                    status["detail"] = str(exc)
            statuses.append(status)
        return statuses

    def prepare_source(
        self,
        *,
        run_id: str | None = None,
        mode: str | None = None,
        provider_secret: Any = None,
        credential_helper: Any = None,
    ) -> CloudSourceContract:
        """Validate exact source identity and runtime roots before paid execution."""

        resolved_run_id = run_id or (
            "cloud-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        ssh_policy = ssh_checkout_policy_from_environment(os.environ)
        source_lock = build_source_lock(
            self.context,
            run_id=resolved_run_id,
            mode=mode,
            environment=os.environ,
            provider_secret=provider_secret,
            credential_helper=credential_helper,
            standalone_credential=standalone_credential_from_environment(os.environ),
            ssh_policy=ssh_policy,
        )
        checkout_policy = checkout_policy_from_context(
            self.context,
            ssh_policy=ssh_policy,
            source_lock=source_lock,
        )
        checkout_policy.validate(source_lock.project_source.location)
        checkout_policy.validate(source_lock.engine_source.location)
        return CloudSourceContract(
            source_lock=source_lock,
            runtime_layout=build_runtime_layout(self.context),
            checkout_policy=checkout_policy,
        )

    def provider_methods(
        self, provider: str, *, validate_environment: bool = True
    ) -> list[str]:
        """Return methods supported by one provider through the API boundary."""

        provider = provider.strip().lower()
        if provider not in TrainingBackendRegistry.list():
            raise ValueError(f"Unknown cloud training provider: {provider}")
        self._authorize_provider(
            provider, route="synaptic-api.v1.training.methods"
        )
        backend = TrainingBackendRegistry.get(provider, repo_root=self.context.engine_root)
        if validate_environment:
            ready, error = backend.validate_environment()
            if not ready:
                raise RuntimeError(error or f"{provider} environment is not ready")
        return list(backend.get_available_methods())

    def prepare(
        self,
        request: CloudTrainingRequest,
        *,
        source: CloudSourceContract | None = None,
        validate_environment: bool = True,
    ) -> CloudTrainingPlan:
        """Resolve a request into a provider plan without starting a job."""

        source = source or self.prepare_source(run_id=request.run_id)
        if request.provider not in TrainingBackendRegistry.list():
            raise ValueError(f"Unknown cloud training provider: {request.provider}")
        self._authorize_provider(
            request.provider, route="synaptic-api.v1.training.prepare"
        )
        backend = TrainingBackendRegistry.get(
            request.provider,
            repo_root=self.context.engine_root,
        )
        if validate_environment:
            ready, error = backend.validate_environment()
            if not ready:
                raise RuntimeError(error or f"{request.provider} environment is not ready")
        methods = backend.get_available_methods()
        if request.method not in methods:
            raise ValueError(
                f"Provider {request.provider!r} does not support method {request.method!r}; "
                f"available methods: {', '.join(methods)}"
            )
        config = backend.load_config(request.method)
        self.apply_request(config, request)
        requested_profile = request.runtime.get("image_profile")
        if requested_profile:
            config.cloud_image, config.cloud_image_profile = resolve_cloud_image(
                self.context.engine_root / "Trainers" / "cloud" / "cloud_config.yaml",
                requested_profile=str(requested_profile),
                fallback_image=getattr(config, "cloud_image", ""),
            )
        config.source_lock = source.source_lock
        config.runtime_layout = source.runtime_layout
        config.checkout_policy = source.checkout_policy
        summary = {
            "provider": request.provider,
            "method": request.method,
            "model": config.model_name,
            "dataset_name": getattr(config, "dataset_name", None),
            "dataset_file": config.dataset_file,
            "gpu": getattr(config, "gpu_type", None),
            "timeout_hours": getattr(config, "timeout_hours", None),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_steps": getattr(config, "max_steps", None),
            "artifact_backend": getattr(config, "artifact_backend", None),
            "artifact_identifier": getattr(config, "artifact_identifier", None),
            "source_commit": getattr(
                getattr(source.source_lock, "engine_source", None), "commit", None
            ),
        }
        return CloudTrainingPlan(request, summary, source, backend, config)

    def submit(self, plan: CloudTrainingPlan) -> CloudTrainingResult:
        """Execute one prepared plan. A plan cannot be submitted twice."""

        if plan._submitted:
            raise RuntimeError("Cloud training plan has already been submitted")
        plan._submitted = True
        raw_result = plan._backend.execute(plan._config, python_path="")
        if isinstance(raw_result, ExecuteResult):
            return CloudTrainingResult(
                provider=plan.request.provider,
                method=plan.request.method,
                exit_code=raw_result.exit_code,
                job_id=raw_result.job_id,
                artifact_prefix=raw_result.artifact_prefix,
                artifact_identifier=raw_result.bucket_id,
                details=dict(raw_result.extras),
            )
        return CloudTrainingResult(
            provider=plan.request.provider,
            method=plan.request.method,
            exit_code=int(raw_result),
            artifact_identifier=getattr(plan._config, "artifact_identifier", None),
        )

    @staticmethod
    def apply_request(config: Any, request: CloudTrainingRequest) -> Any:
        """Apply validated request fields to a provider-loaded configuration."""

        if request.model_name:
            config.model_name = request.model_name
        if request.dataset_name:
            config.dataset_name = request.dataset_name
        if request.dataset_file:
            config.dataset_file = request.dataset_file

        training_fields = {
            "epochs": "epochs",
            "num_train_epochs": "epochs",
            "batch_size": "batch_size",
            "per_device_train_batch_size": "batch_size",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "save_steps": "save_steps",
            "save_total_limit": "save_total_limit",
            "max_steps": "max_steps",
            "max_seq_length": "max_seq_length",
            "seed": "seed",
            "beta": "beta",
            "chat_template_kwargs": "chat_template_kwargs",
            "load_in_4bit": "load_in_4bit",
            "evolutionary_enabled": "evolutionary_enabled",
            "evolutionary_candidates": "evolutionary_candidates",
            "evolutionary_eval_batch_size": "evolutionary_eval_batch_size",
            "evolutionary_validation_config": "evolutionary_validation_config",
            "evolutionary_strategy": "evolutionary_strategy",
            "evolutionary_noise_scale": "evolutionary_noise_scale",
            "evolutionary_max_grad_norm": "evolutionary_max_grad_norm",
            "evolutionary_scale_factors": "evolutionary_scale_factors",
            "evolutionary_selection_method": "evolutionary_selection_method",
            "evolutionary_min_improvement": "evolutionary_min_improvement",
            "evolutionary_min_relative_improvement": "evolutionary_min_relative_improvement",
            "evolutionary_noise_floor_epsilon": "evolutionary_noise_floor_epsilon",
            "evolutionary_eval_frequency": "evolutionary_eval_frequency",
            "evolutionary_warmup_steps": "evolutionary_warmup_steps",
            "evolutionary_cache_baseline": "evolutionary_cache_baseline",
            "evolutionary_log_candidates": "evolutionary_log_candidates",
            "evolutionary_log_selected": "evolutionary_log_selected",
        }
        _apply_fields(config, request.training, training_fields, "training")
        lora_fields = {
            "r": "lora_r",
            "lora_r": "lora_r",
            "alpha": "lora_alpha",
            "lora_alpha": "lora_alpha",
            "dropout": "lora_dropout",
            "lora_dropout": "lora_dropout",
            "target_modules": "lora_target_modules",
            "use_dora": "use_dora",
            "use_rslora": "use_rslora",
            "init_lora_weights": "init_lora_weights",
        }
        _apply_fields(config, request.lora, lora_fields, "lora")
        runtime_fields = {
            "gpu": "gpu_type",
            "gpu_type": "gpu_type",
            "flavor": "gpu_type",
            "timeout_hours": "timeout_hours",
            "cloud_image": "cloud_image",
            "image": "cloud_image",
            "image_profile": "cloud_image_profile",
        }
        _apply_fields(config, request.runtime, runtime_fields, "runtime")
        if getattr(config, "provider", None) == "hf_jobs" and "gpu_type" in request.runtime:
            config.hf_flavor = config.gpu_type
        artifact_fields = {
            "publish_final_model": "publish_final_model",
            "publish_target_repo": "publish_target_repo",
            "backend": "artifact_backend",
            "identifier": "artifact_identifier",
            "mount_path": "artifact_mount_path",
        }
        _apply_fields(config, request.artifacts, artifact_fields, "artifacts")
        return config


def _mapping(value: Any) -> Mapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_fields(config: Any, values: Mapping[str, Any], fields: Mapping[str, str], section: str) -> None:
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ValueError(f"Unknown {section} option(s): {', '.join(unknown)}")
    for source, value in values.items():
        if value is not None:
            setattr(config, fields[source], value)


__all__ = [
    "CLOUD_PROVIDERS",
    "CloudSourceContract",
    "CloudTrainingAPI",
    "CloudTrainingPlan",
    "CloudTrainingRequest",
    "CloudTrainingResult",
]
