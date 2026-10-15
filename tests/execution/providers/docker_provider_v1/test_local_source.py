"""Real bundle/closure/lease staging; all source authority is declared fixture data."""
from dataclasses import replace
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import time

import pytest

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.training.contracts import RetainedTrainingInputStreamLease, CanonicalDocument
from tuner.training.methods.sft import compile_sft_workload
from tuner.runtime.dispatch import (build_source_worker_invocation, materialize_worker_bundle,
    WorkerControlLocationV1, CanonicalWorkloadFileLocationV1)
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.execution.providers.docker_provider_v1.local_source import DockerLocalSourceSealsV1
from tuner.execution.providers.docker_provider_v1.preparation import DockerTrainingPreparationBridgeV1
from tuner.execution.providers.docker_provider_v1.model import (
    DockerWorkloadV1, DockerEffectIdentityV1, DockerSourceSealRequestV1,
    DockerSourceSealLookupRequestV1, AuthenticatedDockerSourceSealV1,
    DockerProviderError, labels_for,
)
from tests.execution.providers.docker_provider_v1.test_preparation import _public_bridge_values

KEY = b'local-source-fixture-key-only'


def sign(content):
    tag = hmac.new(KEY, content.content_digest.encode(), hashlib.sha256).hexdigest()
    return AuthenticatedDockerSourceSealV1(content, 'source-fixture', 'key-v1', tag)


@pytest.fixture
def material(profile, tmp_path):
    training, aligned, run, _ = _public_bridge_values(profile)
    raw = b'{"text":"declared synthetic row"}\n'
    digest = hashlib.sha256(raw).hexdigest()
    identity = PreparedTrainingInputIdentity('prepared://sha256/' + digest, digest, digest, len(raw), 'syntunia-sft-row/v1')
    config = training.resolved_config.to_dict()
    config['dataset'] = identity.to_dict()
    # This fixture now selects authoritative raw input rather than a custom
    # project adapter, so bind the portable trainer's prepared controls.
    config['sft'].pop('custom_row_adapter')
    config['sft'].update(dataset_format='raw_text', completion_only_loss=False,
        assistant_only_loss=False, use_preassigned_splits=True, split_dataset=False)
    config = CanonicalDocument.from_mapping(config)
    roots = dict(training.execution_source.roots)
    roots.update(engine='/source/engine', project='/source/project')
    for name in ('state', 'tracking', 'cache', 'tmp', 'artifacts'):
        roots[name] = '/artifacts/run/' + name
    env = dict(training.execution_source.environment)
    for name, value in roots.items():
        env['SYNAPTIC_' + ('ARTIFACT' if name == 'artifacts' else name.upper()) + '_ROOT'] = value
    env.update(PYTHONPATH=roots['engine'], HF_HOME=roots['cache'] + '/huggingface', TRANSFORMERS_CACHE=roots['cache'] + '/transformers')
    source = replace(training.execution_source, roots=roots, writable_capability_root='/artifacts', environment=env)
    compiled = compile_sft_workload(resolved_config=config, execution_source=source)
    training = replace(training, execution_source=source, resolved_config=config,
                       workload=CanonicalDocument(compiled.canonical_bytes.decode()))
    control = WorkerControlLocationV1(PurePosixPath('/source/control'))
    bundle = materialize_worker_bundle(build_source_worker_invocation(training, control,
        CanonicalWorkloadFileLocationV1(control.control_root)))
    aligned = replace(aligned, workload=DockerWorkloadV1(bundle.dispatch.argv,
        tuple(sorted(dict(bundle.dispatch.environment))), bundle.workload_sha256))
    bridge = DockerTrainingPreparationBridgeV1(aligned)
    prep = bridge.prepare(training, run, source.fingerprint)
    prepared = bridge.prepared(preparation=prep, plan=training, run=run, source_digest=source.fingerprint)
    closure = parse_offline_sft_worker_manifest(bundle.closure_manifest_bytes,
        source_ref=bundle.closure_manifest_source, manifest_path=Path(bundle.closure_manifest_runtime_path))
    import tuner
    engine = Path(tuner.__file__).resolve().parent.parent
    members = {member.path.as_posix(): (engine / member.path).read_bytes() for member in closure.closure.members}
    source_root, artifact_root = tmp_path / 'source', tmp_path / 'artifacts'
    source_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    lease = RetainedTrainingInputStreamLease(identity, io.BytesIO(raw))
    args = dict(training_plan=training, prepared_plan=prepared, bundle=bundle,
        closure_members=members, prepared_lease=lease, source_root=source_root,
        source_container_root=PurePosixPath('/source'), artifact_root=artifact_root,
        artifact_container_root=PurePosixPath('/artifacts'), deadline=time.monotonic() + 120,
        sign_seal=sign)
    port = DockerLocalSourceSealsV1(**args)
    request = DockerSourceSealRequestV1(DockerEffectIdentityV1('a' * 64, 'stage-effect', 'stage', prepared),
        aligned.roots.source_ref, source.fingerprint)
    labels = labels_for(DockerEffectIdentityV1('b' * 64, 'submit-effect', 'submit', prepared))
    yield port, request, labels, args, raw
    port.close()


def test_real_bundle_and_one_use_input_stage(material):
    port, request, labels, args, raw = material
    seal = port.seal_read_only(request)
    assert seal == sign(seal.content)
    assert seal.content.source_digest == args['training_plan'].execution_source.fingerprint
    assert seal.content.source_digest != seal.content.evidence_digest
    key = port._input_key
    assert (port._roots[key[0]].path / key[1]).read_bytes() == raw
    assert stat_mode(port._roots[key[0]].path / key[1]) == 0o600
    port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)
    assert port.lookup(DockerSourceSealLookupRequestV1(request, 1)).seal == seal
    with pytest.raises(ValueError, match='consumed'):
        args['prepared_lease'].take_stream()
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)


def stat_mode(path):
    return path.stat().st_mode & 0o777


@pytest.mark.parametrize('member', ['engine', 'control', 'input', 'extra', 'source_root'])
def test_stage_mutation_rejects_submit(material, member, tmp_path):
    port, request, labels, _, _ = material
    port.seal_read_only(request)
    if member == 'extra':
        (port._roots['source'].path / 'extra').write_bytes(b'wrong')
    elif member == 'source_root':
        path = port._roots['source'].path
        path.rename(tmp_path / 'old-source')
        path.mkdir(mode=0o700)
    else:
        key = (port._input_key if member == 'input' else next(key for key in port._files
               if ('engine/' in key[1]) == (member == 'engine')))
        (port._roots[key[0]].path / key[1]).write_bytes(b'wrong')
    with pytest.raises(DockerProviderError):
        port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)
    assert port.lookup(DockerSourceSealLookupRequestV1(request, 1)).disposition.value == 'indeterminate'


def test_expired_submit_retains_original_seal(material):
    port, request, labels, _, _ = material
    seal = port.seal_read_only(request)
    port._deadline = time.monotonic() - 1
    with pytest.raises(DockerProviderError):
        port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)
    assert port._seal == seal


def test_partial_publication_retained_no_replay(material, monkeypatch):
    port, request, _, _, _ = material
    original = os.write
    calls = 0
    def fail(fd, raw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('private fixture fault')
        return original(fd, raw)
    monkeypatch.setattr(os, 'write', fail)
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)
    assert any(port._roots['source'].path.rglob('*'))
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)
    assert calls == 2


@pytest.mark.parametrize('kind', ['member', 'bundle', 'project_dataset', 'mount', 'lease'])
def test_invalid_material_fails_before_publication(material, kind):
    port, _, _, args, _ = material
    modified = dict(args)
    if kind == 'member':
        members = dict(args['closure_members'])
        members[next(iter(members))] = b'wrong'
        modified['closure_members'] = members
    elif kind == 'bundle':
        bundle = materialize_worker_bundle(build_source_worker_invocation(
            args['training_plan'], WorkerControlLocationV1(PurePosixPath('/source/control')),
            CanonicalWorkloadFileLocationV1(PurePosixPath('/source/control'))))
        object.__setattr__(bundle, 'workload_sha256', '0' * 64)
        modified['bundle'] = bundle
    elif kind == 'project_dataset':
        training = args['training_plan']
        config = training.resolved_config.to_dict()
        config['dataset'] = {'ref': 'project://data/rows.jsonl', 'revision': '9' * 40,
                            'content_digest': 'e' * 64, 'split': 'train', 'format': 'configured_chat_rows/v2'}
        modified['training_plan'] = replace(training, resolved_config=CanonicalDocument.from_mapping(config))
    elif kind == 'mount':
        modified['source_container_root'] = PurePosixPath('/unmapped')
    else:
        modified['prepared_lease'] = RetainedTrainingInputStreamLease(
            replace(args['prepared_lease'].identity, content_digest='c' * 64), io.BytesIO(b'wrong'))
    with pytest.raises((ValueError, DockerProviderError)):
        DockerLocalSourceSealsV1(**modified)
    assert list(port._roots['source'].path.iterdir()) == []


def test_existing_empty_input_slot_collision_not_adopted(material):
    port, request, _, _, _ = material
    root = port._roots[port._input_key[0]].path
    (root / port._input_key[1]).parent.mkdir(parents=True, mode=0o700)
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)
    assert port.lookup(DockerSourceSealLookupRequestV1(request, 1)).disposition.value == 'indeterminate'


def test_bad_lease_bytes_no_publication(material):
    port, request, _, args, _ = material
    args['prepared_lease']._stream = io.BytesIO(b'wrong')
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)
    assert list(port._roots['source'].path.iterdir()) == []
    with pytest.raises(ValueError):
        args['prepared_lease'].take_stream()


def test_signer_deadline_failure_keeps_partial_material(material):
    port, request, _, _, _ = material
    def delayed(content):
        result = sign(content)
        port._deadline = time.monotonic() - 1
        return result
    port._sign = delayed
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)
    assert len(port._retained) == len(port._files) + 1
    assert port._seal is None
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)


def test_original_receipt_lookup_after_lost_return(material):
    port, request, _, _, _ = material
    seal = port.seal_read_only(request)
    # Caller loses its return; lookup uses retained receipt, never consumes again.
    assert port.lookup(DockerSourceSealLookupRequestV1(request, 2)).seal is seal


@pytest.mark.parametrize('kind', ['source', 'input'])
def test_private_member_mode_mutation_rejects(material, kind):
    port, request, labels, _, _ = material
    port.seal_read_only(request)
    key = port._input_key if kind == 'input' else next(iter(port._files))
    path = port._roots[key[0]].path / key[1]
    path.chmod(0o644)
    with pytest.raises(DockerProviderError):
        port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)


def test_retained_parent_descriptors_survive_stage(material):
    port, request, _, _, _ = material
    port.seal_read_only(request)
    assert port._held_parents
    for chain in port._held_parents.values():
        for parent, _, child in chain:
            assert os.fstat(parent).st_ino > 0
            assert os.fstat(child).st_ino > 0
    retained = [fd for chain in port._held_parents.values() for p, _, c in chain for fd in (p, c)]
    port.close()
    for fd in retained:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize('mutation', ['none', 'before_submit', 'during_image', 'invalid_callback'])
def test_signed_effects_and_immediate_create_check(material, tmp_path, mutation):
    from tuner.cloud.derived_training_image import DockerLaunchAuthority, _validate_docker_inputs
    from tuner.cloud.hf_training_image_lock import CommandResult
    from tuner.execution.providers.docker_provider_v1.cli_transport import DockerCLISessionV1
    from tuner.execution.providers.docker_provider_v1.effects import DockerEffectExecutorV1
    from tests.execution.providers.docker_provider_v1.test_effects import make_binding, request as effect_request, payload
    from tests.execution.providers.docker_provider_v1.conftest import Catalog, BindingAuthority
    port, _, _, _, _ = material
    prepared = port._plan
    stage = make_binding('stage', 'a' * 64, 'stage-effect', prepared)
    submit = make_binding('submit', 'b' * 64, 'submit-effect', prepared)
    labels = labels_for(submit.identity)
    class BindingSigner(BindingAuthority):
        def tag_for(self, value):
            return hmac.new(KEY, value.binding_digest.encode(), hashlib.sha256).hexdigest()
    catalog = Catalog(BindingSigner())
    catalog.values.update({stage.command_digest: stage, submit.command_digest: submit})
    class Authority:
        def authenticate_source_seal(self, value):
            return value == sign(value.content)
    executable, config = tmp_path / 'docker', tmp_path / 'docker-config'
    executable.write_bytes(b'declared fixture executable, never executed')
    config.mkdir()
    calls, state = [], {'exists': False, 'status': 'created'}
    cid = 'f' * 64
    changed = False
    def runner(spec):
        nonlocal changed
        calls.append(spec.argv)
        args = spec.argv[5:]
        if args[0] == 'info':
            output = 'daemon-fixture'
        elif args[:2] == ('image', 'inspect'):
            output = json.dumps(prepared.profile.image.image_digest) + '\n[]'
            if mutation == 'during_image' and port._seal is not None and not changed:
                key = next(iter(port._files))
                (port._roots[key[0]].path / key[1]).write_bytes(b'changed')
                changed = True
        elif args[:2] == ('container', 'ls'):
            output = cid if state['exists'] else ''
        elif args[:2] == ('container', 'create'):
            state['exists'] = True
            output = cid
        elif args[:2] == ('container', 'start'):
            state['status'] = 'running'
            output = cid
        elif args[:2] == ('container', 'inspect'):
            output = '\n'.join(map(json.dumps, [cid, '/' + labels.container_name,
                prepared.profile.image.image_digest, DockerCLISessionV1._labels(labels), state['status'], 0]))
        else:
            raise AssertionError(args)
        return CommandResult(output.encode())
    check = (lambda **_: False) if mutation == 'invalid_callback' else port.revalidate_for_submit
    control = DockerCLISessionV1(authority=DockerLaunchAuthority({}, _validate_docker_inputs(executable, config)),
        endpoint='unix:///fixture.sock', daemon_id='daemon-fixture',
        image_refs={prepared.profile.image.image_ref: prepared.profile.image.image_digest},
        source_roots={prepared.profile.roots.source_ref: (port._roots['source'].path, '/source')},
        artifact_roots={prepared.profile.roots.artifact_ref: (port._roots['artifact'].path, '/artifacts')},
        environment=dict(material[3]['bundle'].dispatch.environment), deadline=time.monotonic() + 120,
        runner=runner, profile=prepared.profile, source_submit_check=check)
    executor = DockerEffectExecutorV1(prepared.profile, catalog, catalog.binding_authority,
        control, port, control, object(), Authority(), source_submit_check=check)
    result = executor.execute_once(payload(prepared.profile, 'stage'), effect_request(prepared.profile, stage))
    assert result.disposition.value == 'found'
    assert port._seal is not None
    if mutation == 'before_submit':
        key = next(iter(port._files))
        (port._roots[key[0]].path / key[1]).write_bytes(b'changed')
    result = executor.execute_once(payload(prepared.profile, 'submit'), effect_request(prepared.profile, submit))
    assert result.disposition.value == ('found' if mutation == 'none' else 'indeterminate')
    assert sum('create' in c for c in calls) == (1 if mutation == 'none' else 0)
    assert sum('start' in c for c in calls) == (1 if mutation == 'none' else 0)
    executor.execute_once(payload(prepared.profile, 'submit'), effect_request(prepared.profile, submit))
    assert sum('create' in c for c in calls) == (1 if mutation == 'none' else 0)
