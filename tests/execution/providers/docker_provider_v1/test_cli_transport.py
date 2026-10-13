"""Synthetic process replies; this does not qualify a live daemon or SFT."""
import json
import time

import pytest

from tuner.cloud.derived_training_image import DockerLaunchAuthority, _validate_docker_inputs
from tuner.cloud.hf_training_image_lock import CommandResult
from tuner.execution.providers.docker_provider_v1.cli_transport import DockerCLISessionV1
from tuner.execution.providers.docker_provider_v1.model import (
    DockerLabelsV1, DockerImageV1, DockerRuntimeV1, DockerWorkloadV1,
    DockerLookupRequestV1, DockerLookupPurposeV1,
)
from tuner.training.contracts import AcceleratorDeviceRequestV1

CID = 'f' * 64
IMAGE = 'sha256:' + 'a' * 64


@pytest.fixture
def setup(tmp_path):
    executable = tmp_path / 'docker'
    executable.write_bytes(b'fixture executable identity only')
    config = tmp_path / 'config'
    config.mkdir()
    source, artifacts = tmp_path / 'source', tmp_path / 'artifacts'
    source.mkdir()
    artifacts.mkdir()
    authority = DockerLaunchAuthority({}, _validate_docker_inputs(executable, config))
    labels = DockerLabelsV1('a' * 64, 'docker', 'profile', 'account', 'namespace',
                           'project', 'run', 'b' * 64, 'c' * 64, 'effect', 'submit',
                           'd' * 64, 'e' * 64)
    commands = []
    state = {'exists': False, 'status': 'created', 'daemon': 'daemon-exact'}

    def runner(spec):
        commands.append(spec)
        args = spec.argv[5:]
        assert spec.argv[3:5] == ('--host', 'unix:///fixture.sock')
        assert spec.timeout_seconds <= 30
        assert spec.maximum_output_bytes == 65536
        if args[0] == 'info':
            output = state['daemon']
        elif args[:2] == ('image', 'inspect'):
            output = json.dumps(IMAGE) + '\n' + json.dumps([IMAGE])
        elif args[:2] == ('container', 'ls'):
            output = CID if state['exists'] else ''
        elif args[:2] == ('container', 'create'):
            state['exists'] = True
            output = CID
        elif args[:2] == ('container', 'start'):
            state['status'] = 'running'
            output = CID
        elif args[:2] == ('container', 'inspect'):
            output = '\n'.join(map(json.dumps, [CID, '/synaptic-' + labels.digest,
                               IMAGE, DockerCLISessionV1._labels(labels), state['status'], 0]))
        else:
            raise AssertionError(args)
        return CommandResult(output.encode())

    env = {'EXPLICIT': 'quotes " and $literal'}
    session = DockerCLISessionV1(authority=authority, endpoint='unix:///fixture.sock',
        daemon_id='daemon-exact', image_refs={'image': IMAGE},
        source_roots={'source': (source, '/input')},
        artifact_roots={'artifacts': (artifacts, '/output')}, environment=env,
        deadline=time.monotonic() + 120, runner=runner)
    image = DockerImageV1('image', IMAGE)
    runtime = DockerRuntimeV1(1, 512 * 1024**2, 120, AcceleratorDeviceRequestV1('cpu', (), ()))
    workload = DockerWorkloadV1(('/explicit/program', 'one argument', '$literal'), ('EXPLICIT',), 'a' * 64)
    kwargs = dict(labels=labels, image=image, runtime=runtime, workload=workload,
                  source_ref='source', artifact_ref='artifacts')
    return session, kwargs, commands, state, env


def test_create_start_lookup_explicit_arguments(setup):
    session, kwargs, commands, state, env = setup
    env['EXPLICIT'] = 'changed after construction'
    assert session.require_present(kwargs['image']) is True
    result = session.create_once(**kwargs)
    assert result.disposition.value == 'created'
    create = next(s for s in commands if 'create' in s.argv)
    assert create.argv[-3:] == (IMAGE, 'one argument', '$literal')
    assert create.argv[create.argv.index('--entrypoint') + 1] == '/explicit/program'
    assert '--pull=never' in create.argv
    assert 'EXPLICIT=quotes " and $literal' in create.argv
    assert 'changed after construction' not in repr(create.argv)
    assert not any(k.startswith('HF_') for k in create.env)
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'started'
    found = session.lookup(DockerLookupRequestV1(kwargs['labels'], DockerLookupPurposeV1.OBSERVE, 1))
    assert found.phase.value == 'running'
    assert sum('create' in s.argv for s in commands) == 1
    assert sum('start' in s.argv for s in commands) == 1
    assert session.create_once(**kwargs).disposition.value == 'collision'
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'collision'


def test_collision_never_adopted(setup):
    session, kwargs, commands, state, _ = setup
    state['exists'] = True
    assert session.create_once(**kwargs).disposition.value == 'collision'
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'collision'
    assert not any('create' in s.argv or 'start' in s.argv for s in commands)


def test_daemon_changes_after_create_consumes_attempt(setup):
    session, kwargs, commands, state, _ = setup
    original = session._runner
    def changed(spec):
        result = original(spec)
        if 'create' in spec.argv:
            state['daemon'] = 'replacement'
        return result
    session._runner = changed
    assert session.create_once(**kwargs).disposition.value == 'indeterminate'
    assert session.create_once(**kwargs).disposition.value == 'collision'
    assert sum('create' in s.argv for s in commands) == 1


@pytest.mark.parametrize('failure', [TimeoutError, OSError, ValueError])
def test_unknown_create_never_replayed(setup, failure):
    session, kwargs, commands, _, _ = setup
    original = session._runner
    def fail(spec):
        if 'create' in spec.argv:
            commands.append(spec)
            raise failure('secret raw exception')
        return original(spec)
    session._runner = fail
    result = session.create_once(**kwargs)
    assert result.disposition.value == 'indeterminate'
    assert 'secret' not in repr(result)
    assert session.create_once(**kwargs).disposition.value == 'collision'
    assert sum('create' in s.argv for s in commands) == 1


def test_empty_lookup_is_not_authenticated_absence(setup):
    session, kwargs, _, _, _ = setup
    assert session.lookup(DockerLookupRequestV1(kwargs['labels'], DockerLookupPurposeV1.OBSERVE, 1)).disposition.value == 'indeterminate'


def test_deadline_before_effect(setup):
    session, kwargs, commands, _, _ = setup
    session._deadline = time.monotonic() - 1
    assert session.create_once(**kwargs).disposition.value == 'indeterminate'
    assert commands == []


def test_present_image_mismatch_no_create(setup):
    session, kwargs, commands, _, _ = setup
    original = session._runner
    def wrong(spec):
        if 'image' in spec.argv:
            return CommandResult((json.dumps('sha256:' + 'b' * 64) + '\n[]').encode())
        return original(spec)
    session._runner = wrong
    assert session.require_present(kwargs['image']) is False
    assert session.create_once(**kwargs).disposition.value == 'indeterminate'
    assert not any('create' in s.argv for s in commands)


def test_image_repo_digest_resolves_exact_content_id(setup):
    session, kwargs, commands, _, _ = setup
    session._images = {'image': 'repository@' + IMAGE}
    original = session._runner
    def manifest(spec):
        if 'image' in spec.argv:
            return CommandResult((json.dumps('sha256:' + 'b' * 64) + '\n' + json.dumps(['repository@' + IMAGE])).encode())
        if 'inspect' in spec.argv and 'container' in spec.argv:
            result = original(spec)
            return CommandResult(result.stdout.replace(IMAGE.encode(), ('sha256:' + 'b' * 64).encode()))
        return original(spec)
    session._runner = manifest
    assert session.create_once(**kwargs).disposition.value == 'created'
    create = next(s for s in commands if 'create' in s.argv)
    assert create.argv[-3] == 'sha256:' + 'b' * 64


def test_unknown_start_consumed(setup):
    session, kwargs, commands, _, _ = setup
    assert session.create_once(**kwargs).disposition.value == 'created'
    original = session._runner
    def fail(spec):
        if 'start' in spec.argv:
            commands.append(spec)
            raise TimeoutError('private')
        return original(spec)
    session._runner = fail
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'indeterminate'
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'collision'
    assert sum('start' in s.argv for s in commands) == 1


@pytest.mark.parametrize('kind', ['oversized', 'wrong_labels', 'invalid_json', 'multiple'])
def test_malformed_or_ambiguous_lookup_closed(setup, kind):
    session, kwargs, _, state, _ = setup
    state['exists'] = True
    original = session._runner
    def malformed(spec):
        if 'ls' in spec.argv and kind == 'multiple':
            return CommandResult((CID + '\n' + 'e' * 64).encode())
        if 'inspect' in spec.argv:
            if kind == 'oversized':
                return CommandResult(b'x' * 65537)
            if kind == 'invalid_json':
                return CommandResult(b'invalid')
            result = original(spec)
            return CommandResult(result.stdout.replace(b'"project"', b'"other"'))
        return original(spec)
    session._runner = malformed
    result = session.lookup(DockerLookupRequestV1(kwargs['labels'], DockerLookupPurposeV1.OBSERVE, 1))
    assert result.disposition.value == ('multiple' if kind == 'multiple' else 'indeterminate')


def test_environment_key_mismatch_no_process(setup):
    session, kwargs, commands, _, _ = setup
    kwargs['workload'] = DockerWorkloadV1(('/program',), (), 'a' * 64)
    assert session.create_once(**kwargs).disposition.value == 'indeterminate'
    assert commands == []


def test_executable_authority_mutation_rejects_before_command(setup):
    session, kwargs, commands, _, _ = setup
    session._authority._authority.executable.write_bytes(b'replaced executable')
    assert session.require_present(kwargs['image']) is False
    assert session.create_once(**kwargs).disposition.value == 'indeterminate'
    assert commands == []


def test_no_start_ack_when_state_still_created(setup):
    session, kwargs, _, _, _ = setup
    assert session.create_once(**kwargs).disposition.value == 'created'
    original = session._runner
    def no_start(spec):
        if 'start' in spec.argv:
            return CommandResult(CID.encode())
        return original(spec)
    session._runner = no_start
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'indeterminate'
    assert session.start_once(CID, kwargs['labels']).disposition.value == 'collision'
