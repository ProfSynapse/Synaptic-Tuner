"""Pure packaged material admission; no provider dispatch or recipe selection."""
from dataclasses import replace
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.releases import PackagedExecutionBindingV1, parse_packaged_runtime_release
from tuner.training.contracts import (
    ArtifactPolicy, CanonicalDocument, ResolvedTrainingComponents, ResolvedTrainingRequest,
    ResourceSpec, RuntimeSpec, TrainingRequest,
)
from tuner.training.packaged_boundary import (
    compile_bound_packaged_workload, derive_packaged_coordinator_material,
    packaged_method_configuration_digest, parse_packaged_coordinator_material,
    validate_packaged_material,
)
from tuner.training.packaged_compilation import packaged_artifact_policy_digest
from tuner.training.packaged_grpo_compilation import _declared_input, compile_packaged_grpo_workload
from tuner.training import default_recipe_registry
from tests.runtime.test_packaged_grpo_compilation import _fixture
from tests.runtime.test_packaged_runtime_releases import _provider


def fixture():
    core, context = _fixture()
    public = _declared_input(core.document)
    config = CanonicalDocument.from_mapping({'schema_version': 'synaptic-packaged-grpo-config/v1',
        'method': 'grpo', 'execution': {'mode': 'packaged_runtime'},
        'core_workload': core.document, 'limits': {'child_max_seconds': 60}})
    context = CanonicalDocument.from_mapping(context)
    workload = compile_packaged_grpo_workload(core_workload=core, context=context.to_dict(), child_max_seconds=60)
    release = parse_packaged_runtime_release(context.to_dict()['runtime_release'])
    identity = PreparedTrainingInputIdentity.from_dict(core.document['prepared_input'])
    policy = ArtifactPolicy(public.artifacts.required_kinds, public.artifacts.retain_checkpoints)
    binding = PackagedExecutionBindingV1.build(run_ref='run-grpo', runtime_release=release,
        provider_runtime_binding=_provider(release), **identity.execution_binding_fields(),
        workload_digest=workload.fingerprint,
        configuration_digest=packaged_method_configuration_digest(config, execution_context=context),
        artifact_policy_digest=packaged_artifact_policy_digest(policy))
    components = ResolvedTrainingComponents(binding, context, config,
        RuntimeSpec(release.image_ref, release.installed_distributions_digest, release.python_version),
        ResourceSpec('cpu'), policy)
    return public, components, workload


def test_exact_public_core_context_and_binding_admit():
    public, components, expected = fixture()
    request = TrainingRequest(CanonicalDocument(public.canonical_json()))
    assert validate_packaged_material(request=request, material=components) == expected
    assert compile_bound_packaged_workload(components.resolved_config, components.execution_source,
        execution_context=components.execution_context) == expected


@pytest.mark.parametrize('field', ['model', 'hyperparameters', 'rewards', 'dataset', 'artifacts'])
def test_public_declaration_cannot_relabel_same_bound_core(field):
    public, components, _ = fixture()
    changed = public.to_dict()
    if field == 'model': changed[field]['revision'] = 'b'*40
    elif field == 'hyperparameters': changed[field]['learning_rate'] = 0.0001
    elif field == 'rewards': changed[field]['documents']['exact']['validation'][0]['match'] = '^changed$'
    elif field == 'dataset': changed[field]['ref'] = 'prepared://sha256/'+'e'*64
    else: changed[field]['retain_checkpoints'] = True
    request = TrainingRequest(CanonicalDocument(TrainingInputV1.from_dict(changed).canonical_json()))
    with pytest.raises((ValueError, TypeError)):
        validate_packaged_material(request=request, material=components)


@pytest.mark.parametrize('change', ['limit', 'context', 'extra', 'developer', 'unknown', 'missing_context'])
def test_bound_wrapper_rejects_changed_or_unsupported_material(change):
    _, components, _ = fixture()
    config, context = components.resolved_config.to_dict(), components.execution_context.to_dict()
    if change == 'limit': config['limits']['child_max_seconds'] = 61
    elif change == 'context': context['runtime_release']['release_ref'] = 'runtime:changed'
    elif change == 'extra': config['extra'] = True
    elif change == 'developer': config['core_workload']['execution_source'] = {}
    elif change == 'unknown': config['method'] = 'other'
    with pytest.raises((ValueError, TypeError, KeyError)):
        compile_bound_packaged_workload(CanonicalDocument.from_mapping(config), components.execution_source,
            execution_context=None if change == 'missing_context' else CanonicalDocument.from_mapping(context))


def test_cold_material_rebuild_uses_original_context_and_preserves_public_basis():
    public, components, workload = fixture()
    request = TrainingRequest(CanonicalDocument(public.canonical_json()))
    resolved = ResolvedTrainingRequest(request, components.execution_source, components.execution_context,
        components.resolved_config, CanonicalDocument(workload.canonical_bytes.decode()),
        components.runtime, components.resources, components.artifact_policy)
    material = derive_packaged_coordinator_material(resolved, default_recipe_registry(),
        request_id='request-grpo', project_ref='project-grpo', run_id='run-grpo')
    cold = parse_packaged_coordinator_material(material.canonical_bytes, default_recipe_registry())
    assert cold.canonical_bytes == material.canonical_bytes
    # Actual context is immutable detached input, not current source/recipe lookup.
    changed = copy.deepcopy(components.execution_context.to_dict())
    changed['runtime_release']['release_ref'] = 'runtime:changed'
    with pytest.raises(ValueError):
        validate_packaged_material(request=request, material=replace(components,
            execution_context=CanonicalDocument.from_mapping(changed)))


def test_sft_keeps_exact_bytes_and_lazy_grpo_import_boundary():
    root = Path(__file__).resolve().parents[2]
    code = f"""import sys
sys.path.insert(0, {str(root)!r})
from tests.training.test_packaged_execution_material import packaged_fixture
from tuner.training.packaged_compilation import compile_packaged_sft_workload, packaged_configuration_digest
from tuner.training.packaged_boundary import compile_bound_packaged_workload, packaged_method_configuration_digest
_, material = packaged_fixture()
expected = compile_packaged_sft_workload(resolved_config=material.resolved_config)
assert compile_bound_packaged_workload(material.resolved_config, material.execution_source).canonical_bytes == expected.canonical_bytes
assert packaged_method_configuration_digest(material.resolved_config) == packaged_configuration_digest(material.resolved_config)
assert not any(name.startswith(('tuner.training.grpo', 'tuner.training.packaged_grpo', 'Trainers.grpo')) for name in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True)
