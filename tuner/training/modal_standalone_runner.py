"""Internal same-process host for one config-first packaged Modal run."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import secrets
import stat
import time

from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1, TrainingPreparationConfigV1,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.coordinator_v1.model import ProviderReadPurposeV1
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.packaged_composition import (
    ModalPackagedAuthorityPorts, ModalPackagedCatalogPorts,
    ModalPackagedSourcePorts, compose_modal_packaged_adapter,
)
from tuner.execution.providers.modal.packaged_deployment import ModalPackagedDeploymentObserver
from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
from tuner.execution.providers.modal.packaged_staging import ModalPackagedInputStager
from tuner.execution.providers.modal.packaged_worker import InstalledPackagedSFTTrainerExecutor
from tuner.execution.providers.modal.model_snapshot import prepare_model_snapshot
from tuner.execution.providers.modal.runtime_release_qualification import ModalRuntimeReleaseQualificationReceiptV1
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import CanonicalDocument, ResolvedTrainingComponents, ResourceSpec, RuntimeSpec
from tuner.training.input_preparation import (
    PUBLISHED_PREPARED_DATASET_NORMALIZER_V1,
    PosixPrivatePreparedRootAuthorityV1,
    PublishedPreparedDatasetNormalizerConfigV1,
    PublishedPreparedDatasetNormalizerV1,
    TrainingInputPreparationServiceV1, default_dataset_format_verifiers_v1,
)
from tuner.training.modal_host_authority import HMACAuthenticator, UTCClock
from tuner.training.modal_host_composition import compose_modal_packaged_reference_host
from tuner.training.modal_host_download import download_verified_modal_artifact
from tuner.training.modal_host_reader import ModalPackagedReadUnavailable
from tuner.training.modal_host_requests import ModalPackagedResolutionUnavailable
from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
from tuner.training.modal_host_prepared_copy import stage_published_modal_dataset
from tuner.training.modal_host_qualification import (
    ModalHostCPUQualificationV1, ModalHostQualificationUnavailable,
    qualify_modal_runtime_for_host,
)
from tuner.training.modal_host_runtime import (
    ModalHostBootstrapUnavailable, prepare_modal_runtime_for_host,
)
from tuner.training.modal_host_scope import observe_modal_host_scope, open_modal_host_scope
from tuner.training.modal_host_storage import ModalHostStorageV1
from tuner.training.modal_recipe import ModalSFTRecipePlanV1
from tuner.training.packaged_compilation import (
    PACKAGED_CONTEXT_SCHEMA, compile_packaged_sft_workload,
    packaged_artifact_policy_digest, packaged_configuration_digest,
)


_QUALIFICATION_SECRET_NAME = "synaptic-qualification"
_HF_TOKEN_SECRET_NAME = "synaptic-training-hf-token"
_APP_NAME = "synaptic-training"


class ModalStandaloneRunUnavailable(RuntimeError):
    """Fixed host diagnostic; no private path, provider text or credential."""


_RUN_PHASE_DIAGNOSTICS = {
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
}


class ModalStandalonePhaseUnavailable(RuntimeError):
    """Closed, non-authorizing diagnosis after a validated CPU receipt."""

    def __init__(self, phase: str):
        if phase not in _RUN_PHASE_DIAGNOSTICS:
            raise ValueError("invalid_modal_standalone_phase")
        self.phase = phase
        self.failure_class, self.location = _RUN_PHASE_DIAGNOSTICS[phase]
        self.retry_authorized = False
        super().__init__("modal_standalone_phase_unavailable")


class _HostSecrets:
    def __init__(self, generated: dict[str, str]):
        self._generated = dict(generated)

    def resolve(self, ref: SecretRef) -> str:
        if type(ref) is not SecretRef or ref.provider != "env":
            raise ModalStandaloneRunUnavailable("modal_host_secret_unavailable")
        value = self._generated.get(ref.name, os.environ.get(ref.name))
        if type(value) is not str or not value.strip():
            raise ModalStandaloneRunUnavailable("modal_host_secret_unavailable")
        return value


def _private_root() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    if not base.is_absolute():
        raise ModalStandaloneRunUnavailable("modal_host_state_root_invalid")
    root = base / "synaptic-training"
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or root.resolve(strict=True) != root):
            raise ValueError
        return root
    except Exception:
        raise ModalStandaloneRunUnavailable("modal_host_state_root_invalid") from None


def _attempt_journal(root: Path, attempt_ref: str, *, fresh_attempt: bool) -> Path:
    if type(fresh_attempt) is not bool:
        raise ModalStandaloneRunUnavailable("modal_host_attempt_mode_invalid")
    if not fresh_attempt:
        return root / "modal-host.sqlite3"
    attempt_dir = root / ("attempt-" + attempt_ref)
    try:
        attempt_dir.mkdir(mode=0o700, exist_ok=False)
        info = attempt_dir.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError
        return attempt_dir / "modal-host.sqlite3"
    except Exception:
        raise ModalStandaloneRunUnavailable("modal_host_attempt_unavailable") from None


@dataclass(frozen=True, slots=True)
class ModalStandaloneRunResultV1:
    run: TrainingRunRef
    artifact_paths: tuple[Path, ...]
    gpu_only_timeout_estimate_minor_units: int
    maximum_cost_minor_units: int


@dataclass(frozen=True, slots=True)
class ModalStandaloneCPUQualificationV1:
    runtime_release_digest: str
    deployment_facts_digest: str
    qualification_output_sha256: str
    training_executed: bool = False
    gpu_qualified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "synaptic-modal-sft-cpu-qualification/v1",
            "runtime_release_digest": self.runtime_release_digest,
            "deployment_facts_digest": self.deployment_facts_digest,
            "qualification_output_sha256": self.qualification_output_sha256,
            "training_executed": False,
            "gpu_qualified": False,
            "authorization": "CPU-only installed-worker check; not GPU training authority",
        }


def _qualified_runtime(*, modal: object, plan: ModalSFTRecipePlanV1,
                       context: object, resources: ResourceSpec,
                       storage: ModalHostStorageV1, effect_id: str,
                       modal_profile: str, modal_environment: str,
                       builder_cache_root: Path):
    secrets_port = _HostSecrets({
        "SYNAPTIC_TRAIN_AUTHORITY_KEY": secrets.token_hex(32),
        "SYNAPTIC_TRAIN_READER_KEY": secrets.token_hex(32),
    })
    hf_ref = SecretRef("env", "HF_TOKEN")
    secrets_port.resolve(hf_ref)
    client, client_binding = open_modal_host_scope(
        sdk=modal, profile=modal_profile, environment_name=modal_environment,
    )
    profile_path = (context.engine_root / "Trainers" / "image_profiles"
                    / "qwen35_4b_packaged_sft_3360351c" / "profile.yaml")
    runtime = prepare_modal_runtime_for_host(
        sdk=modal, client=client, client_binding=client_binding,
        profile_path=profile_path,
        runtime_material_intent_digest=plan.runtime_material_intent_digest,
        recipe_resource=resources,
        maximum_cost_minor_units=plan.recipe.maximum_cost_minor_units,
        private_storage=storage, secret_resolver=secrets_port,
        hf_token_ref=hf_ref,
        qualification_secret_name=_QUALIFICATION_SECRET_NAME,
        hf_token_secret_name=_HF_TOKEN_SECRET_NAME,
        app_name=_APP_NAME, environment_name=modal_environment,
        builder_cache_root=builder_cache_root,
    )
    qualification = qualify_modal_runtime_for_host(
        sdk=modal, client=client, client_binding=client_binding,
        runtime=runtime, private_storage=storage,
        effect_id=effect_id, environment_name=modal_environment,
    )
    if (type(qualification) is not ModalHostCPUQualificationV1
            or type(qualification.receipt) is not ModalRuntimeReleaseQualificationReceiptV1
            or qualification.runtime_release_digest != runtime.release.manifest_digest
            or qualification.deployment_facts_digest != runtime.deployment_facts.facts_digest
            or qualification.output_sha256 != qualification.receipt.output_sha256
            or qualification.provider_call_id != qualification.receipt.provider_call_id
            or qualification.receipt.effect_id != effect_id
            or qualification.receipt.deployment_facts_digest != runtime.deployment_facts.facts_digest
            or qualification.training_executed is not False
            or qualification.gpu_qualified is not False):
        raise ModalStandaloneRunUnavailable("modal_host_cpu_qualification_unavailable")
    return secrets_port, client, client_binding, runtime, qualification


def qualify_modal_standalone_job(*, plan: ModalSFTRecipePlanV1, context: object,
                                 modal_profile: str,
                                 modal_environment: str,
                                 fresh_attempt: bool = False) -> ModalStandaloneCPUQualificationV1:
    """Build/deploy and check the installed CPU worker; never stage training input."""
    if (os.name != "posix" or platform.python_version() != "3.11.14"
            or type(plan) is not ModalSFTRecipePlanV1
            or plan.recipe.maximum_cost_minor_units is None):
        raise ModalStandaloneRunUnavailable("modal_host_qualification_unavailable")
    try:
        import modal
        if modal.__version__ != "1.5.4":
            raise ValueError
        root = _private_root()
        resources = ResourceSpec(
            plan.recipe.accelerator, plan.recipe.accelerator_count,
            plan.recipe.timeout_seconds,
        )
        effect_id = "cpu-qual-" + secrets.token_hex(12)
        with ModalHostStorageV1(
            _attempt_journal(root, effect_id, fresh_attempt=fresh_attempt),
            "standalone-training",
        ) as storage:
            _, _, _, runtime, qualification = _qualified_runtime(
                modal=modal, plan=plan, context=context, resources=resources,
                storage=storage, effect_id=effect_id,
                modal_profile=modal_profile, modal_environment=modal_environment,
                builder_cache_root=root / "modal-wheel-builder",
            )
        return ModalStandaloneCPUQualificationV1(
            runtime.release.manifest_digest,
            runtime.deployment_facts.facts_digest,
            qualification.output_sha256,
        )
    except (ModalHostBootstrapUnavailable, ModalHostQualificationUnavailable):
        raise
    except Exception:
        raise ModalStandaloneRunUnavailable("modal_host_qualification_unavailable") from None


def run_modal_standalone_job(*, plan: ModalSFTRecipePlanV1, context: object,
                             modal_profile: str, modal_environment: str,
                             fresh_attempt: bool = False) -> ModalStandaloneRunResultV1:
    """Prepare, quote, build, submit, authenticate, verify, and save once.

    No user callback, manually built wheel or provider-specific public API is
    part of this command.  The public TrainingAPI and RunsAPI own the run.
    """
    if (os.name != "posix" or platform.python_version() != "3.11.14"):
        raise ModalStandaloneRunUnavailable("modal_host_python_31114_posix_required")
    if (type(plan) is not ModalSFTRecipePlanV1
            or type(modal_profile) is not str or not modal_profile
            or type(modal_environment) is not str or not modal_environment
            or plan.recipe.maximum_cost_minor_units is None):
        raise ModalStandaloneRunUnavailable("modal_host_configuration_invalid")
    phase: str | None = None
    try:
        import modal
        if modal.__version__ != "1.5.4":
            raise ValueError

        recipe = plan.recipe
        root = _private_root()
        run_id = "modal-" + secrets.token_hex(12)
        project_ref = "standalone-training"
        run = TrainingRunRef(run_id, project_ref)
        prepared_root = root / run_id
        source = context.project_root / recipe.dataset_locator
        copied = stage_published_modal_dataset(
            source.absolute(), private_root=prepared_root,
            expected_dataset_digest=plan.prepared_identity.revision,
            expected_content_digest=plan.prepared_identity.content_digest,
        )
        normalizer = PublishedPreparedDatasetNormalizerV1(authorized_root=prepared_root)
        preparation = TrainingInputPreparationServiceV1(
            prepared_root=prepared_root,
            normalizers={PUBLISHED_PREPARED_DATASET_NORMALIZER_V1: normalizer},
            format_verifiers=default_dataset_format_verifiers_v1(),
            root_authority=PosixPrivatePreparedRootAuthorityV1(),
        )
        configuration = TrainingPreparationConfigV1(
            "request-" + run_id, project_ref,
            recipe.training_input(plan.prepared_identity.ref).canonical_json(),
            PublishedPreparedDatasetNormalizerConfigV1(
                recipe.dataset_digest, recipe.train_rows, recipe.validation_rows,
            ),
        )
        source_port = LocalTrainingInputPathV1(copied)
        prepared = preparation.prepare(source_port, configuration)
        if prepared.prepared.identity != plan.prepared_identity:
            raise ValueError
        identity = prepared.prepared.identity
        resolved_config = recipe.packaged_config(identity)
        workload = compile_packaged_sft_workload(resolved_config=resolved_config)
        if (workload.fingerprint != plan.workload_digest
                or packaged_configuration_digest(resolved_config) != plan.configuration_digest):
            raise ValueError
        resources = ResourceSpec(recipe.accelerator, recipe.accelerator_count,
                                 recipe.timeout_seconds)
        with ModalHostStorageV1(
            _attempt_journal(root, run_id, fresh_attempt=fresh_attempt),
            project_ref,
        ) as storage:
            secrets_port, client, client_binding, runtime, qualification = _qualified_runtime(
                modal=modal, plan=plan, context=context, resources=resources,
                storage=storage, effect_id="cpu-qual-" + run_id,
                modal_profile=modal_profile, modal_environment=modal_environment,
                builder_cache_root=root / "modal-wheel-builder",
            )
            phase = "RUN_HOST_ASSEMBLY"
            release, facts = runtime.release, runtime.facts
            provider_binding = facts.build_provider_binding(release)
            execution = PackagedExecutionBindingV1.build(
                run_ref=run_id, runtime_release=release,
                provider_runtime_binding=provider_binding,
                prepared_input_ref=identity.ref,
                prepared_input_revision=identity.revision,
                prepared_input_content_digest=identity.content_digest,
                prepared_input_size_bytes=identity.size_bytes,
                prepared_input_format=identity.format,
                workload_digest=workload.fingerprint,
                configuration_digest=packaged_configuration_digest(resolved_config),
                artifact_policy_digest=packaged_artifact_policy_digest(recipe.artifact_policy()),
            )
            components = ResolvedTrainingComponents(
                execution, CanonicalDocument.from_mapping({
                    "schema_version": PACKAGED_CONTEXT_SCHEMA,
                    "runtime_release": release.to_dict(),
                }), resolved_config,
                RuntimeSpec(None, release.installed_distributions_digest,
                            release.python_version, "modal_build", release.material_digest),
                resources, recipe.artifact_policy(),
            )
            signer_ref = "modal-runtime-release-qualification-hmac-v1"
            signer = HMACAuthenticator(
                {signer_ref: runtime.qualification_key},
                allowed_purposes=frozenset({
                    "modal-packaged-dispatch/v1", "modal-packaged-completion/v1",
                }),
            )
            effects = ModalPackagedHostEffectsV1(
                storage=storage, runtime_release=release,
                provider_binding=provider_binding, provider_facts=facts,
                execution_binding=execution, retained_source=prepared.retained_source,
                workload_bytes=workload.canonical_bytes,
                artifact_policy=recipe.artifact_policy(),
                signer=signer, key_ref=signer_ref,
                maximum_cost_minor_units=recipe.maximum_cost_minor_units,
            )
            deployment = ModalPackagedDeploymentObserver(
                sdk=modal, client=client, client_binding=client_binding,
                reader=runtime.deployment_reader,
            )
            facade = ExplicitModal154ReadFacade(
                client_binding, sdk=modal, client=client,
                scope_observer=lambda supplied: observe_modal_host_scope(
                    sdk=modal, client=supplied, binding=client_binding,
                ),
                deployment_observer=runtime.deployment_reader.observe,
                volume_names=dict(runtime.volume_names_by_id),
            )
            reader = ModalPackagedReader(
                facade=facade, deployment_observer=deployment,
                verifier=signer, key_ref=signer_ref,
            )
            namespace = domain_digest("synaptic-modal-namespace/v1", canonical_bytes({
                "workspace_ref": facts.workspace_ref,
                "environment_ref": facts.environment_ref,
            }))
            adapter = compose_modal_packaged_adapter(
                profile_ref=recipe.runtime_profile, account_ref=facts.account_ref,
                namespace_ref=namespace, sdk=modal, client=client,
                facade=facade, deployment_observer=deployment,
                stager=ModalPackagedInputStager(facade),
                catalogs=ModalPackagedCatalogPorts(
                    effects.bindings, effects.stage_receipts, effects.calls,
                ),
                sources=ModalPackagedSourcePorts(effects.stages, effects.dispatches),
                authorities=ModalPackagedAuthorityPorts(effects, signer),
                clock=UTCClock(),
                trainer=InstalledPackagedSFTTrainerExecutor(model_preparer=prepare_model_snapshot),
                reader=reader,
            )
            host = compose_modal_packaged_reference_host(
                adapter=adapter, effects=effects,
                prepared_request=prepared.prepared.request, run=run,
                components=components, context=context,
                input_preparation=preparation, clock=UTCClock(),
                secrets=secrets_port,
                authority_secret=SecretRef("env", "SYNAPTIC_TRAIN_AUTHORITY_KEY"),
                reader_secret=SecretRef("env", "SYNAPTIC_TRAIN_READER_KEY"),
                profile_digest=plan.profile_sha256.removeprefix("sha256:"),
                quote_digest=runtime.quote_digest,
                maximum_cost_minor_units=recipe.maximum_cost_minor_units,
            )
            api = host.api
            phase = "RUN_PUBLIC_PREPARE"
            publicly_prepared = api.training.prepare(source_port, configuration)
            if publicly_prepared.prepared != prepared.prepared:
                raise ValueError
            phase = "RUN_PUBLIC_LOAD"
            request = api.training.load(publicly_prepared.prepared.request.canonical_json)
            phase = "RUN_PUBLIC_RESOLVE"
            resolved = api.training.resolve(request)
            phase = "RUN_PUBLIC_PLAN"
            training_plan = api.training.plan(resolved, ProviderRef("modal", recipe.runtime_profile))
            phase = "RUN_PUBLIC_PREFLIGHT"
            preflight = api.training.preflight(training_plan)
            phase = "RUN_START_INDETERMINATE"
            started = api.training.start(training_plan, preflight)
            if started.accepted is not True:
                raise ValueError
            phase = None
            workflow = host.composition.stores.workflow_store.get(started.run)
            read_request = host.composition.runs._request(
                workflow, ProviderReadPurposeV1.OBSERVE,
            )
            binding = host.reader._binding(read_request, ProviderReadPurposeV1.OBSERVE)
            provider_job_ref = read_request.provider_run.reference.provider_job_ref
            deadline = time.monotonic() + recipe.timeout_seconds + 120
            while True:
                try:
                    state, _ = host.reader._poll_packaged_call(
                        binding, provider_job_ref, deadline=deadline,
                    )
                except ModalPackagedReadUnavailable:
                    raise
                if state is ModalFunctionCallState.RETURNED:
                    break
                if state is not ModalFunctionCallState.PENDING or time.monotonic() >= deadline:
                    raise ModalStandaloneRunUnavailable("modal_packaged_call_unavailable")
                time.sleep(min(5, max(0, deadline - time.monotonic())))
            outcome = api.runs.outcome(started.run)
            if outcome.state.value != "succeeded" or api.runs.verify(started.run).verified is not True:
                raise ValueError
            artifact_root = root / (run_id + "-artifacts")
            artifact_paths = tuple(download_verified_modal_artifact(
                api.runs, started.run, role=role, output_root=artifact_root,
            ) for role in sorted((
                "workload_record", "training_lineage", "training_metrics",
                "final_model", "tokenizer",
            )))
            return ModalStandaloneRunResultV1(
                started.run, artifact_paths,
                runtime.quote.gpu_only_timeout_estimate_minor_units,
                recipe.maximum_cost_minor_units,
            )
    except (ModalHostBootstrapUnavailable, ModalHostQualificationUnavailable):
        raise
    except ModalPackagedResolutionUnavailable as failure:
        if phase == "RUN_PUBLIC_RESOLVE" and type(failure) is ModalPackagedResolutionUnavailable:
            stage_phase = {
                "RICH": "RUN_RESOLVE_RICH",
                "DERIVE": "RUN_RESOLVE_DERIVE",
                "REPARSE": "RUN_RESOLVE_REPARSE",
            }.get(failure.stage)
            if stage_phase is not None:
                raise ModalStandalonePhaseUnavailable(stage_phase) from None
        if phase is not None:
            raise ModalStandalonePhaseUnavailable(phase) from None
        raise ModalStandaloneRunUnavailable("modal_standalone_run_unavailable") from None
    except Exception:
        if phase is not None:
            raise ModalStandalonePhaseUnavailable(phase) from None
        raise ModalStandaloneRunUnavailable("modal_standalone_run_unavailable") from None
