"""Provider-free qualification for closed packaged Modal bindings."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from synaptic_tuner.api.v1.providers import ProviderRef
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import (
    CanonicalProviderPayloadV1,
    build_stage_command,
)
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.providers.modal.packaged_binding import (
    MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA,
    CommittedModalPackagedBindingCatalog,
    ModalPackagedCommandBinding,
    ModalPackagedRuntimeFactsV1,
    parse_modal_packaged_runtime_facts,
)
from tuner.runtime.releases import (
    PackagedExecutionBindingV1,
    PackagedTrainingRuntimeReleaseV1,
)
from tuner.training.packaged_compilation import compile_packaged_sft_workload

from tests.training.test_packaged_execution_material import packaged_fixture


DIGEST = "d" * 64


def _release_and_execution(payload: bytes | None = None):
    _, components = packaged_fixture()
    release = PackagedTrainingRuntimeReleaseV1.from_dict(
        components.execution_context.to_dict()["runtime_release"]
    )
    original = components.execution_source
    facts = ModalPackagedRuntimeFactsV1(
        account_ref="acct", workspace_ref="workspace", environment_ref="env",
        client_ref="client", sdk_version="1.5.4",
        app_name="synaptic-training-v1", app_id="ap-owned",
        deployment_generation=7, function_name="packaged-worker",
        function_id="fu-worker", self_check_function_name="packaged-self-check",
        self_check_function_id="fu-selfcheck", deployment_spec_digest="e" * 64,
        image_id="im-runtime",
        image_digest=release.image_digest, package_digest=release.package_digest,
        installed_distributions_digest=release.installed_distributions_digest,
        worker_entrypoint=release.worker_entrypoint,
        worker_closure_digest=release.worker_closure_digest,
        control_volume_id="vo-control", artifact_volume_id="vo-artifact",
    )
    provider = facts.build_provider_binding(release)
    prepared_content_digest = (
        original.prepared_input_content_digest
        if payload is None else hashlib.sha256(payload).hexdigest()
    )
    prepared_size_bytes = (
        original.prepared_input_size_bytes if payload is None else len(payload)
    )
    execution = PackagedExecutionBindingV1.build(
        run_ref=original.run_ref, runtime_release=release,
        provider_runtime_binding=provider,
        prepared_input_ref=original.prepared_input_ref,
        prepared_input_revision=original.prepared_input_revision,
        prepared_input_content_digest=prepared_content_digest,
        prepared_input_size_bytes=prepared_size_bytes,
        prepared_input_format=original.prepared_input_format,
        workload_digest=original.workload_digest,
        configuration_digest=original.configuration_digest,
        artifact_policy_digest=original.artifact_policy_digest,
    )
    return release, facts, provider, execution, components


def _binding(payload: bytes | None = None) -> ModalPackagedCommandBinding:
    release, facts, provider, execution, _ = _release_and_execution(payload)
    namespace = domain_digest(
        "synaptic-modal-namespace/v1",
        canonical_bytes({"workspace_ref": facts.workspace_ref,
                         "environment_ref": facts.environment_ref}),
    )
    preparation = CanonicalPreparationV2.build(
        provider=ProviderRef("modal", "modal-packaged-v1"),
        scope=ExecutionScopeV1(facts.account_ref, namespace),
        project_ref="project", run_id=execution.run_ref,
        plan_fingerprint="1" * 64, source_digest=execution.binding_digest,
        workload_digest=execution.workload_digest, runtime_digest="2" * 64,
        resource_digest="3" * 64, artifact_contract_digest="4" * 64,
        quote_digest="5" * 64, secret_requirements_digest="6" * 64,
        execution_binding_digest="7" * 64,
    )
    payload = CanonicalProviderPayloadV1.build(
        "modal", "stage-payload/v2", execution.workload_digest,
    )
    command = build_stage_command(
        preparation, "nonce", payload,
        ExecutorDescriptorV1("modal", "modal-packaged-executor", "0.1.0"),
    )
    return ModalPackagedCommandBinding(
        command.canonical_bytes, release.canonical_bytes(),
        provider.canonical_bytes(), facts.canonical_bytes,
        execution.canonical_bytes(),
    )


def test_closed_modal_facts_round_trip_and_bind_release_exactly() -> None:
    release, facts, provider, _, _ = _release_and_execution()
    assert parse_modal_packaged_runtime_facts(facts.canonical_bytes) == facts
    assert facts.build_provider_binding(release) == provider
    assert provider.provider_ref == "modal"
    assert provider.provider_facts_schema == MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA
    assert provider.provider_facts_digest == facts.facts_digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("image_digest", "0" * 64),
        ("package_digest", "1" * 64),
        ("installed_distributions_digest", "2" * 64),
        ("worker_entrypoint", "other.package:main"),
        ("worker_closure_digest", "3" * 64),
    ),
)
def test_release_or_build_identity_drift_is_rejected(field: str, value: str) -> None:
    release, facts, _, _, _ = _release_and_execution()
    with pytest.raises(ValueError, match="differ"):
        replace(facts, **{field: value}).validate_release(release)


def test_provider_facts_are_closed_private_and_secret_free() -> None:
    _, facts, _, _, _ = _release_and_execution()
    document = json.loads(facts.canonical_bytes)
    assert document["deployment"]["private"] is True
    encoded = facts.canonical_bytes.lower()
    for forbidden in (
        b"token", b"password", b"credential", b"secret", b"api_key",
        b"modal_token_id", b"modal_token_secret",
    ):
        assert forbidden not in encoded
    document["runtime"]["secret_value"] = "must-not-enter-facts"
    with pytest.raises(ValueError, match="identity is invalid"):
        ModalPackagedRuntimeFactsV1.from_dict(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("deployment_generation", True),
        ("app_id", "ap/unsafe"),
        ("function_id", "fu/unsafe"),
        ("self_check_function_id", "fu/unsafe"),
        ("image_id", "im/unsafe"),
        ("artifact_volume_id", "vo-control"),
        ("sdk_version", "1.5.5"),
    ),
)
def test_modal_facts_reject_ambiguous_or_unsafe_identity(field: str, value: object) -> None:
    _, facts, _, _, _ = _release_and_execution()
    with pytest.raises((TypeError, ValueError)):
        replace(facts, **{field: value})


def test_complete_command_binding_reconstructs_every_commitment() -> None:
    binding = _binding()
    rebuilt = binding.reconstructed()
    assert rebuilt == binding
    assert rebuilt.command_digest == binding.command_digest
    assert rebuilt.execution_binding.validate_bindings(
        rebuilt.runtime_release, rebuilt.provider_binding,
    ) is None
    assert rebuilt.provider_facts.build_provider_binding(
        rebuilt.runtime_release,
    ) == rebuilt.provider_binding


def test_committed_catalog_reconstructs_and_rejects_substitution() -> None:
    binding = _binding()
    calls: list[str] = []
    catalog = CommittedModalPackagedBindingCatalog(
        lambda digest: calls.append(digest) or binding,
    )
    assert catalog.resolve(binding.command_digest) == binding
    assert calls == [binding.command_digest]
    with pytest.raises(ValueError, match="substituted"):
        catalog.resolve("0" * 64)


def test_provider_native_facts_never_enter_shared_workload() -> None:
    _, facts, _, _, components = _release_and_execution()
    workload = compile_packaged_sft_workload(
        resolved_config=components.resolved_config,
    ).canonical_bytes
    for modal_value in (
        facts.app_id, facts.function_id, facts.image_id,
        facts.control_volume_id, facts.artifact_volume_id,
    ):
        assert modal_value.encode("utf-8") not in workload
    document = json.loads(workload)
    keys: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(document)
    assert not keys.intersection({
        "app_id", "function_id", "image_id", "control_volume_id",
        "artifact_volume_id", "deployment_generation", "call_id",
    })


def test_rebound_provider_facts_cannot_replace_committed_execution() -> None:
    binding = _binding()
    changed = replace(binding.provider_facts, deployment_generation=8)
    changed_provider = changed.build_provider_binding(binding.runtime_release)
    with pytest.raises(ValueError, match="cross-binding"):
        ModalPackagedCommandBinding(
            binding.command_bytes, binding.runtime_release_bytes,
            changed_provider.canonical_bytes(), changed.canonical_bytes,
            binding.execution_binding_bytes,
        )
