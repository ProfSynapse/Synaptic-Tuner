"""Actual Docker adapters, public APIs and SQLite with scripted CLI replies.

No daemon, training process or GPU is used. Source bytes and five artifact
roles are explicit synthetic fixtures; cold compositions retain original keys.
"""
from dataclasses import replace
import json
import hashlib
import sys
from pathlib import Path
import time

import pytest

from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest
from tuner.cloud.derived_training_image import DockerLaunchAuthority, _validate_docker_inputs
from tuner.cloud.hf_training_image_lock import CommandResult
from tuner.training import default_recipe_registry
from tuner.training.contracts import CanonicalDocument, ResolvedTrainingRequest, TrainingRequest
from tuner.training.coordinator_material import derive_coordinator_material
from tuner.training.docker_host_composition import compose_docker_reference_host
from tests.execution.providers.docker_provider_v1.conftest import profile as profile
from tests.execution.providers.docker_provider_v1.test_local_source import material as material

CID = 'f' * 64
SECRET = SecretRef('env', 'DOCKER_HOST_AUTHORITY')
EVIDENCE = SecretRef('env', 'DOCKER_HOST_EVIDENCE')


class Clock:
    def __init__(self):
        self.epoch = 1787788800

    def now_epoch(self):
        return self.epoch

    def now(self):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self.epoch, timezone.utc).isoformat().replace('+00:00', 'Z')


class Secrets:
    def resolve(self, ref):
        return {SECRET: 'original-host-secret-' + 'a' * 40,
                EVIDENCE: 'original-evidence-secret-' + 'b' * 40}[ref]


class Grants:
    def __init__(self):
        self.bound = []

    def authorize(self, requirements):
        return ExecutionGrant('host-grant')

    def bind(self, grant, *, operation, requirements):
        self.bound.append(operation.operation.effect.kind.value)
        return object()


@pytest.fixture
def host_fixture(material, tmp_path):
    _, _, _, source, raw = material
    rich = source['training_plan']
    from tests.runtime.test_artifact_verification import _runtime_workload
    from tuner.training.methods.sft import compile_sft_workload
    from tuner.runtime.dispatch import (WorkerControlLocationV1, CanonicalWorkloadFileLocationV1,
        materialize_worker_bundle, build_source_worker_invocation)
    from tuner.execution.providers.docker_provider_v1.model import DockerWorkloadV1
    from pathlib import PurePosixPath
    execution = replace(rich.execution_source, python_executable=str(Path(sys.executable).resolve()),
                        python_version='.'.join(map(str, sys.version_info[:3])))
    rich = replace(rich, execution_source=execution,
                   runtime=replace(rich.runtime, python_version=execution.python_version))
    config = rich.resolved_config.to_dict()
    native_config = _runtime_workload(raw_text=True).document['configuration']['document']
    config['model'], config['sft'] = native_config['model'], native_config['sft']
    canonical = CanonicalDocument.from_mapping(config)
    compiled = compile_sft_workload(resolved_config=canonical, execution_source=rich.execution_source)
    rich = replace(rich, resolved_config=canonical, workload=CanonicalDocument(compiled.canonical_bytes.decode()))
    control_root = WorkerControlLocationV1(PurePosixPath('/source/control'))
    bundle = materialize_worker_bundle(build_source_worker_invocation(rich, control_root,
        CanonicalWorkloadFileLocationV1(control_root.control_root)))
    profile = replace(source['prepared_plan'].profile,
        workload=DockerWorkloadV1(bundle.dispatch.argv, tuple(sorted(dict(bundle.dispatch.environment))),
                                  bundle.workload_sha256),
        descriptor=replace(source['prepared_plan'].profile.descriptor,
            capabilities=replace(source['prepared_plan'].profile.descriptor.capabilities, logs=False)))
    recipes = default_recipe_registry()
    resolved = ResolvedTrainingRequest(
        TrainingRequest(CanonicalDocument.from_mapping({'fixture': 'explicit-sft-request'})),
        rich.execution_source, rich.execution_context, rich.resolved_config,
        rich.workload, rich.runtime, rich.resources, rich.artifact_policy)
    retained = derive_coordinator_material(resolved, recipes,
        request_id='request', project_ref='project', run_id=rich.execution_source.run_id)
    source_inputs = {k: v for k, v in source.items() if k not in (
        'training_plan', 'prepared_plan', 'sign_seal')}
    source_inputs['bundle'] = bundle
    executable, config = tmp_path / 'docker', tmp_path / 'docker-config'
    executable.write_bytes(b'fixture executable identity; never executed')
    config.mkdir()
    commands, state = [], {'exists': False, 'status': 'created', 'code': 0,
        'labels': None, 'unknown_create': False, 'claims': None}
    authority = DockerLaunchAuthority({}, _validate_docker_inputs(executable, config))

    def runner(spec):
        args = spec.argv[5:]
        commands.append(args)
        if args[0] == 'info':
            result = 'fixture-daemon'
        elif args[:2] == ('image', 'inspect'):
            result = json.dumps(profile.image.image_digest) + '\n[]'
        elif args[:2] == ('container', 'ls'):
            result = CID if state['exists'] else ''
        elif args[:2] == ('container', 'create'):
            # Source publication preceded create; exact input bytes are present.
            assert any(p.read_bytes() == raw for p in source['artifact_root'].rglob('*.jsonl'))
            assert state['claims']() == ('stage', 'submit')
            labels = {}
            for i, value in enumerate(args):
                if value == '--label':
                    key, val = args[i + 1].split('=', 1)
                    labels[key] = val
            state['labels'] = labels
            state['name'] = args[args.index('--name') + 1]
            state['exists'] = True
            if state['unknown_create']:
                raise TimeoutError('lost reply, fixture only')
            result = CID
        elif args[:2] == ('container', 'start'):
            state['status'] = 'running'
            result = CID
        elif args[:2] == ('container', 'inspect'):
            name = state['name']
            result = '\n'.join(map(json.dumps, [CID, '/' + name, profile.image.image_digest,
                state['labels'], state['status'], state['code']]))
        elif args[:2] == ('container', 'stop'):
            state['status'] = 'exited'
            result = CID
        else:
            raise AssertionError(args)
        return CommandResult(result.encode())

    cli = dict(authority=authority, endpoint='unix:///fixture.sock', daemon_id='fixture-daemon',
        image_refs={profile.image.image_ref: profile.image.image_digest},
        source_roots={profile.roots.source_ref: (source['source_root'], '/source')},
        artifact_roots={profile.roots.artifact_ref: (source['artifact_root'], '/artifacts')},
        environment=dict(source['bundle'].dispatch.environment),
        deadline=time.monotonic() + 120, runner=runner)
    roles = {role: 'native/' + role + '.bin' for role in profile.artifacts.roles}
    clock, grants = Clock(), Grants()
    state['clock'] = clock
    kwargs = dict(profile=profile, rich_plan=rich, material=retained, recipes=recipes,
        authority_secret=SECRET, evidence_secret=EVIDENCE, cli_configuration=cli,
        source_inputs=source_inputs, artifact_root=source['artifact_root'], role_paths=roles,
        allow_current_cancel_authorization=True)
    def compose(store, *, cold=False):
        host = compose_docker_reference_host(**{**kwargs, 'source_inputs': None if cold else source_inputs},
            ports=ReferenceHostPortsV1(store, store, clock, grants, Secrets()))
        if not cold:
            # Use exact current workflow rather than private planning caches.
            def claims():
                planning = host.composition.coordinator._bindings
                workflow = host.composition.stores.workflow_store.get(planning.run)
                return tuple(kind for kind, intent in (('stage', workflow.stage), ('submit', workflow.submit))
                    if intent is not None and host.composition.foundation.get(intent.effect_id).attempt_count == 1)
            state['claims'] = claims
        return host
    return compose, kwargs, commands, state, roles, grants, tmp_path


def start(host, kwargs):
    api = host.api
    request = api.training.load(kwargs['material'].request_bytes.decode())
    resolved = api.training.resolve(request)
    plan = api.training.plan(resolved, kwargs['profile'].provider)
    preflight = api.training.preflight(plan)
    result = api.training.start(plan, preflight)
    return api, plan, preflight, result.run


def native_values(kwargs, monkeypatch):
    """Production invocation/evidence builders, synthetic trainer payloads only."""
    from Trainers.sft import runtime_v1 as runtime
    from tuner.training.recipes import CompiledWorkload
    from tests.runtime.test_artifact_verification import _lineage, _tar, _safetensors
    rich = kwargs['rich_plan']
    doc = rich.workload.to_dict()
    workload = CompiledWorkload(doc['method'], doc['schema_version'], doc['entrypoint'],
                               rich.workload.canonical_json.encode())
    roots = runtime.RuntimeRoots(**{key: Path(value)
        for key, value in doc['execution_source']['runtime']['roots'].items()})
    dataset_path = roots.state / 'prepared-inputs' / doc['configuration']['document']['dataset']['content_digest'] / 'private-dataset.jsonl'
    raw = kwargs['source_inputs']['prepared_lease'].identity
    # The stager's actual immutable dataset bytes have already been checked.
    dataset_bytes = next(p.read_bytes() for p in kwargs['artifact_root'].rglob('*.jsonl')
                         if p.stat().st_size == raw.size_bytes)
    model = doc['configuration']['document']['model']
    snapshot = roots.cache / 'model' / ('models--' + model['ref'].replace('/', '--')) / 'snapshots' / model['revision']
    allowed = {dataset_path, roots.engine / 'Trainers/sft/train_sft.py',
               roots.engine / 'tuner/runtime/offline_sft_worker.py'}
    def regular(path, root, **_):
        assert path in allowed
        return path
    def read(path, **_):
        assert path == dataset_path
        return dataset_bytes
    bundle = kwargs['source_inputs']['bundle']
    environment = {'SYNAPTIC_WORKER_CLOSURE_MANIFEST': bundle.closure_manifest_runtime_path.as_posix(),
                   'SYNAPTIC_WORKER_CLOSURE_DIGEST': bundle.closure_digest}
    with monkeypatch.context() as patch:
        patch.setattr(runtime, '_require_local_model_snapshot', lambda *_: snapshot)
        patch.setattr(runtime, '_require_contained_regular', regular)
        patch.setattr(runtime, '_read_regular', read)
        invocation = runtime.build_trainer_invocation(workload, roots, environment)
    try:
        trainer = json.loads(_lineage(workload))['trainer_lineage']
        trainer.update(run_directory=invocation.run_dir.as_posix(),
            dataset={'source': invocation.argv[invocation.argv.index('--local-file') + 1]},
            synaptic_runtime_projection=dict(invocation.expected_projection))
        evidence = runtime.TrainerEvidence(0, invocation.final_model_dir,
            invocation.tokenizer_dir, trainer, invocation.expected_projection, {})
        execution = runtime._build_execution_evidence(workload, invocation, evidence)
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        lineage = canonical({'schema_version': 'synaptic-sft-training-lineage/v1',
            'workload_fingerprint': workload.fingerprint,
            'execution_source': doc['execution_source'], 'configuration_revision': doc['configuration']['revision'],
            'identities': doc['identities'], 'trainer_exit_code': 0, 'execution_evidence': execution,
            'execution_evidence_sha256': hashlib.sha256(canonical(execution)).hexdigest(), 'trainer_lineage': trainer})
        return {'workload_record': workload.canonical_bytes, 'training_lineage': lineage,
            'training_metrics': b'{}',
            'final_model': _tar({'adapter_config.json': b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
                                 'adapter_model.safetensors': _safetensors()}),
            'tokenizer': _tar({'tokenizer_config.json': b'{"tokenizer_class":"Fixture"}',
                'tokenizer.json': b'{"model":{"type":"BPE","vocab":{"x":0}},"version":"1.0"}'})}
    finally:
        runtime._close_retained_fds(invocation.retained_fds)


def test_public_lifecycle_native_artifacts_and_cold_sqlite(host_fixture, monkeypatch):
    compose, kwargs, commands, state, roles, _, tmp_path = host_fixture
    path = tmp_path / 'workflow.sqlite3'
    with SQLiteDurableStoreV1(path) as store, compose(store) as host:
        assert commands == []
        assert list(kwargs['source_inputs']['source_root'].iterdir()) == []
        api, plan, _, run = start(host, kwargs)
        assert api.runs.outcome(run).state.value == 'running'
        state['status'] = 'exited'
        native = kwargs['artifact_root'] / 'native'
        native.mkdir(mode=0o700)
        values = native_values(kwargs, monkeypatch)
        for role, relative in roles.items():
            (kwargs['artifact_root'] / relative).write_bytes(values[role])
        assert api.runs.outcome(run).state.value == 'succeeded'
        verified = api.runs.verify(run)
        assert verified.verified is True
        original = host.composition.stores.workflow_store.get(run)
        assert len([x for x in commands if x[:2] == ('container', 'create')]) == 1
    before = len(commands)
    with SQLiteDurableStoreV1(path, create=False) as store, compose(store, cold=True) as cold:
        assert len(commands) == before
        assert cold.api.runs.show(run).state.value == 'succeeded'
        assert cold.api.runs.outcome(run).state.value == 'succeeded'
        assert cold.composition.stores.workflow_store.get(run) == original
        for role in roles:
            chunks = cold.api.runs.artifacts(RunArtifactRequest(run, role, 1048576))
            assert b''.join(chunks.iter_bytes()) == values[role]
        assert cold.api.training.preflight(plan).ready is False
    assert len([x for x in commands if x[:2] == ('container', 'create')]) == 1


def test_cold_cancel_once_does_not_replay_submit(host_fixture):
    compose, kwargs, commands, state, _, _, tmp_path = host_fixture
    path = tmp_path / 'workflow.sqlite3'
    with SQLiteDurableStoreV1(path) as store, compose(store) as host:
        _, _, _, run = start(host, kwargs)
    state['clock'].epoch += 1800
    with SQLiteDurableStoreV1(path, create=False) as store, compose(store, cold=True) as cold:
        assert cold.api.runs.cancel(run, 'operator request').state.value == 'cancel_requested'
        cold.api.runs.cancel(run, 'operator request')
        # Stop acknowledgement cannot invent a causal cancelled state: this
        # fixture's actual exit code is zero, so ordinary completion wins.
        assert cold.api.runs.outcome(run).state.value == 'succeeded'
    assert len([x for x in commands if x[:2] == ('container', 'stop')]) == 1
    assert len([x for x in commands if x[:2] == ('container', 'create')]) == 1


def test_unknown_submit_reopen_never_creates_again(host_fixture):
    compose, kwargs, commands, state, _, _, tmp_path = host_fixture
    state['unknown_create'] = True
    path = tmp_path / 'workflow.sqlite3'
    with SQLiteDurableStoreV1(path) as store, compose(store) as host:
        api, plan, preflight, run = start(host, kwargs)
        before = len(commands)
        api.training.start(plan, preflight)
        assert len(commands) == before
    with SQLiteDurableStoreV1(path, create=False) as store, compose(store, cold=True) as cold:
        cold.api.runs.show(run)
        cold.api.training.start(plan, preflight)
    assert len([x for x in commands if x[:2] == ('container', 'create')]) == 1


@pytest.mark.parametrize('attack', ['training_lineage', 'workload_record', 'final_model', 'tokenizer', 'failed_exit'])
def test_hash_consistent_invalid_native_artifacts_never_verify(host_fixture, monkeypatch, attack):
    compose, kwargs, _, state, roles, _, tmp_path = host_fixture
    with SQLiteDurableStoreV1(tmp_path / 'workflow.sqlite3') as store, compose(store) as host:
        api, _, _, run = start(host, kwargs)
        state['status'] = 'exited'
        values = native_values(kwargs, monkeypatch)
        if attack != 'failed_exit':
            values[attack] = b'{}'
        (kwargs['artifact_root'] / 'native').mkdir(mode=0o700)
        for role, relative in roles.items():
            (kwargs['artifact_root'] / relative).write_bytes(values[role])
        # Publish successful terminal observation, then alter only actual exit
        # evidence. Native bytes remain valid in the failed-exit control.
        api.runs.outcome(run)
        if attack == 'failed_exit':
            state['code'] = 7
        assert api.runs.verify(run).verified is False
        workflow = host.composition.stores.workflow_store.get(run)
        assert workflow.phase.value == 'verification_failed'
        assert workflow.verification_receipts[-1].content.verified_artifacts == ()


def test_stage_claim_precedes_lease_and_both_submit_checks(host_fixture, monkeypatch):
    from tuner.training.contracts import RetainedTrainingInputStreamLease
    from tuner.execution.providers.docker_provider_v1.local_source import DockerLocalSourceSealsV1
    compose, kwargs, _, _, _, _, tmp_path = host_fixture
    original_take = RetainedTrainingInputStreamLease.take_stream
    original_check = DockerLocalSourceSealsV1.revalidate_for_submit
    checks = []
    with SQLiteDurableStoreV1(tmp_path / 'workflow.sqlite3') as store, compose(store) as host:
        def take(lease):
            planning = host.composition.coordinator._bindings
            workflow = host.composition.stores.workflow_store.get(planning.run)
            assert workflow.stage is not None and workflow.submit is None
            record = host.composition.foundation.get(workflow.stage.effect_id)
            assert record.attempt_count == 1
            return original_take(lease)
        def check(port, **values):
            checks.append(values['labels'].digest)
            return original_check(port, **values)
        monkeypatch.setattr(RetainedTrainingInputStreamLease, 'take_stream', take)
        monkeypatch.setattr(DockerLocalSourceSealsV1, 'revalidate_for_submit', check)
        start(host, kwargs)
    assert len(checks) == 2 and checks[0] == checks[1]


def test_unstarted_host_closes_owned_lease(host_fixture):
    compose, kwargs, commands, _, _, _, tmp_path = host_fixture
    lease = kwargs['source_inputs']['prepared_lease']
    with SQLiteDurableStoreV1(tmp_path / 'workflow.sqlite3') as store:
        with compose(store):
            assert lease._state == 'available'
        assert lease._state == 'closed'
    assert commands == []


@pytest.mark.parametrize('field', ['execution_context', 'resolved_config', 'resources', 'request'])
def test_material_mismatch_rejects_before_lease_or_cli(host_fixture, field):
    from tuner.execution.providers.docker_provider_v1.preparation import DockerTrainingPreparationBridgeV1
    from tuner.training.methods.sft import compile_sft_workload
    from synaptic_tuner.api.v1.planning import TrainingPlan, TrainingPlanBasisV1, ProviderPlanContextV1, ProviderPlanRef
    from synaptic_tuner.api.v1.results import TrainingRunRef
    _, kwargs, commands, _, _, _, _ = host_fixture
    original = kwargs['material']
    rich = kwargs['rich_plan']
    profile = kwargs['profile']
    bridge = DockerTrainingPreparationBridgeV1(profile)
    _, resolved = bridge.coordinator_material(rich, original, kwargs['recipes'])
    basis = TrainingPlanBasisV1.from_resolved(resolved)
    context = ProviderPlanContextV1('synaptic-provider-plan-context/v1', profile.provider,
        basis.basis_digest, profile.descriptor.descriptor_digest, profile.profile_digest)
    old_plan = TrainingPlan('synaptic-training-plan/v2', basis, ProviderPlanRef(context.provider_context_digest))
    request = json.loads(original.request_bytes)
    if field == 'resources':
        rich = replace(rich, resources=replace(rich.resources, timeout_seconds=rich.resources.timeout_seconds + 1))
    elif field == 'request':
        request['selected_request'] = 'different'
    elif field == 'execution_context':
        context_doc = rich.execution_context.to_dict()
        context_doc['different'] = 'context'
        rich = replace(rich, execution_context=CanonicalDocument.from_mapping(context_doc))
    else:
        config = rich.resolved_config.to_dict()
        config['sft']['seed'] += 1
        canonical = CanonicalDocument.from_mapping(config)
        compiled = compile_sft_workload(resolved_config=canonical, execution_source=rich.execution_source)
        rich = replace(rich, resolved_config=canonical, workload=CanonicalDocument(compiled.canonical_bytes.decode()))
    actual = ResolvedTrainingRequest(TrainingRequest(CanonicalDocument.from_mapping(request)),
        rich.execution_source, rich.execution_context, rich.resolved_config,
        rich.workload, rich.runtime, rich.resources, rich.artifact_policy)
    changed = derive_coordinator_material(actual, kwargs['recipes'],
        request_id=resolved.request_id, project_ref=resolved.project_ref, run_id=original.run_id)
    from pathlib import PurePosixPath
    from tuner.runtime.dispatch import (WorkerControlLocationV1, CanonicalWorkloadFileLocationV1,
        materialize_worker_bundle, build_source_worker_invocation)
    from tuner.execution.providers.docker_provider_v1.model import DockerWorkloadV1
    from tuner.execution.providers.docker_provider_v1.preparation import _resource_digest
    control = WorkerControlLocationV1(PurePosixPath('/source/control'))
    bundle = materialize_worker_bundle(build_source_worker_invocation(rich, control,
        CanonicalWorkloadFileLocationV1(control.control_root)))
    changed_profile = replace(profile,
        workload=DockerWorkloadV1(bundle.dispatch.argv, tuple(sorted(dict(bundle.dispatch.environment))), bundle.workload_sha256),
        runtime=replace(profile.runtime, timeout_seconds=rich.resources.timeout_seconds),
        resource_digest=_resource_digest(rich))
    changed_bridge = DockerTrainingPreparationBridgeV1(changed_profile)
    _, projected = changed_bridge.coordinator_material(rich, changed, kwargs['recipes'])
    assert projected.resolved_config_digest != resolved.resolved_config_digest
    with pytest.raises(ValueError):
        changed_bridge.coordinator_prepared(plan=rich, material=changed, coordinator_plan=old_plan,
            run=TrainingRunRef(original.run_id, resolved.project_ref), recipes=kwargs['recipes'])
    assert kwargs['source_inputs']['prepared_lease']._state == 'available'
    assert commands == []


@pytest.mark.parametrize('field', ['source_host', 'source_destination', 'artifact_host', 'artifact_destination'])
def test_mount_projection_mismatch_rejects_before_daemon_and_lease(host_fixture, field):
    _, kwargs, commands, _, _, grants, tmp_path = host_fixture
    cli = dict(kwargs['cli_configuration'])
    target = 'source_roots' if field.startswith('source') else 'artifact_roots'
    mappings = dict(cli[target])
    ref, (host, destination) = next(iter(mappings.items()))
    replacement = tmp_path / 'different-root'
    replacement.mkdir(mode=0o700)
    mappings[ref] = (replacement, destination) if field.endswith('host') else (host, '/different')
    cli[target] = mappings
    with SQLiteDurableStoreV1(tmp_path / 'workflow.sqlite3') as store:
        with pytest.raises(ValueError):
            compose_docker_reference_host(**{**kwargs, 'cli_configuration': cli},
                ports=ReferenceHostPortsV1(store, store, Clock(), grants, Secrets()))
    assert kwargs['source_inputs']['prepared_lease']._state == 'available'
    assert commands == []


@pytest.mark.parametrize('missing', ['coordinator_plan', 'coordinator_material'])
def test_partial_source_plan_binding_rejects_before_lease(host_fixture, missing):
    from tuner.execution.providers.docker_provider_v1.local_source import DockerLocalSourceSealsV1
    compose, kwargs, commands, _, _, _, tmp_path = host_fixture
    with SQLiteDurableStoreV1(tmp_path / 'workflow.sqlite3') as store, compose(store) as host:
        planning = host.composition.coordinator._bindings
        source = dict(kwargs['source_inputs'])
        source.update(training_plan=kwargs['rich_plan'], prepared_plan=planning.prepared,
            sign_seal=host.source._evidence.source_seal, coordinator_plan=planning.plan,
            coordinator_material=planning.material, recipes=kwargs['recipes'])
        source[missing] = None
        with pytest.raises(ValueError, match='docker_source_plan_invalid'):
            DockerLocalSourceSealsV1(**source)
        assert kwargs['source_inputs']['prepared_lease']._state == 'available'
        assert commands == []


@pytest.mark.parametrize('change', ['dependency_lock_digest', 'retain_checkpoints'])
def test_full_material_projection_preserves_runtime_and_policy(host_fixture, change):
    from synaptic_tuner.api.v1.planning import TrainingPlan, TrainingPlanBasisV1, ProviderPlanContextV1, ProviderPlanRef
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from tuner.execution.providers.docker_provider_v1.preparation import DockerTrainingPreparationBridgeV1
    _, kwargs, commands, _, _, _, _ = host_fixture
    rich, material, recipes, profile = (kwargs[k] for k in ('rich_plan', 'material', 'recipes', 'profile'))
    bridge = DockerTrainingPreparationBridgeV1(profile)
    _, before = bridge.coordinator_material(rich, material, recipes)
    if change == 'dependency_lock_digest':
        changed = replace(rich, runtime=replace(rich.runtime, dependency_lock_digest='7' * 64))
    else:
        changed = replace(rich, artifact_policy=replace(rich.artifact_policy,
                           retain_checkpoints=not rich.artifact_policy.retain_checkpoints))
    actual = ResolvedTrainingRequest(TrainingRequest(CanonicalDocument(material.request_bytes.decode())),
        changed.execution_source, changed.execution_context, changed.resolved_config,
        changed.workload, changed.runtime, changed.resources, changed.artifact_policy)
    newer = derive_coordinator_material(actual, recipes,
        request_id=before.request_id, project_ref=before.project_ref, run_id=material.run_id)
    _, after = bridge.coordinator_material(changed, newer, recipes)
    assert before.resolved_config_digest != after.resolved_config_digest
    assert before.workload_digest != material.planning_request.workload_digest
    def plan(resolved):
        basis = TrainingPlanBasisV1.from_resolved(resolved)
        context = ProviderPlanContextV1('synaptic-provider-plan-context/v1', profile.provider,
            basis.basis_digest, profile.descriptor.descriptor_digest, profile.profile_digest)
        return TrainingPlan('synaptic-training-plan/v2', basis, ProviderPlanRef(context.provider_context_digest))
    old, new = plan(before), plan(after)
    assert old.plan_fingerprint != new.plan_fingerprint
    with pytest.raises(ValueError):
        bridge.coordinator_prepared(plan=changed, material=newer, coordinator_plan=old,
            run=TrainingRunRef(material.run_id, before.project_ref), recipes=recipes)
    assert kwargs['source_inputs']['prepared_lease']._state == 'available'
    assert commands == []
