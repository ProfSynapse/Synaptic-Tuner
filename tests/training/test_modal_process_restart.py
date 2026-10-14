"""Actual host-process loss with retained, explicit provider-free SDK fixtures.

The mock provider is private fixture bytes, not a running Modal service. These
checks qualify host reconstruction and authenticated reads, not live training.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tuner.training import modal_retained_reader as reader
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_host_authority import HMACAuthenticator
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
from tests.training.test_modal_standalone_runner import _setup


_SOURCE = Path(reader.__file__).resolve().parents[2]
_TEST_ROOT = Path(__file__).resolve().parents[2]
_CREDENTIAL_NAMES = ('HF_TOKEN', 'HF_API_KEY', 'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET')
_BOOTSTRAP = '''
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from tests.training.test_modal_process_restart import _child
_child(sys.argv[3], Path(sys.argv[4]), Path(sys.argv[2]))
'''


def _write(path, document):
    raw = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
    assert len(raw) < 128 * 1024
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _snapshot(root):
    return json.loads((root / 'provider.json').read_bytes())


def _publish_provider_fixture(root, binding, call_ref, retained):
    from tuner.execution.providers.modal.contracts import operation_path, provider_entry_identity
    from tuner.execution.foundation_v2.canonical import canonical_bytes
    facts = binding.provider_facts
    effect = binding.command.operation.effect.effect_id
    files, members = {}, []
    for role in sorted(('workload_record', 'training_lineage', 'training_metrics', 'final_model', 'tokenizer')):
        payload = ('declared fixture artifact: ' + role).encode()
        path = operation_path(effect, 'output', role)
        files[path] = base64.b64encode(payload).decode()
        members.append({'role': role, 'path': path, 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                        'provider_entry_id': provider_entry_identity(facts.artifact_volume_id, path, len(payload))})
    completion = canonical_bytes({
        'schema_version': 'synaptic-modal-packaged-completion/v1', 'effect_id': effect,
        'command_digest': binding.command_digest, 'provider_job_ref': call_ref,
        'runtime_release_digest': binding.runtime_release.manifest_digest,
        'provider_runtime_binding_digest': binding.provider_binding.binding_digest,
        'execution_binding_digest': binding.execution_binding.binding_digest,
        'stage_receipt_sha256': '1' * 64, 'inventory_sha256': '2' * 64, 'terminal_sha256': '3' * 64,
        'members': members,
    })
    key_ref = 'modal-runtime-release-qualification-hmac-v1'
    signer = HMACAuthenticator({key_ref: retained.qualification_key}, allowed_purposes=frozenset({'modal-packaged-completion/v1'}))
    tag = signer.sign('modal-packaged-completion/v1', completion, key_ref)
    evidence = operation_path(effect, 'evidence')
    _write(root / 'provider.json', {
        'phase': 'pending', 'facts': facts.to_dict(), 'client_binding': retained.inputs['client_binding'],
        'volume_names': retained.inputs['volume_names'], 'run': retained.run.to_dict(),
        'call_result': {'schema_version': 'synaptic-modal-packaged-worker-result/v1', 'effect_id': effect,
                        'status_code': 'completed', 'completion_sha256': hashlib.sha256(completion).hexdigest()},
        'volumes': {
            facts.control_volume_id: {evidence + '/packaged-completion.json': base64.b64encode(completion).decode(),
                                      evidence + '/packaged-completion.mac': base64.b64encode(tag).decode()},
            facts.artifact_volume_id: files,
        },
        'artifact_hashes': {item['role']: item['sha256'] for item in members},
    })


def _producer(root):
    patch = pytest.MonkeyPatch()
    (root / 'host').mkdir(mode=0o700)
    plan, context, events = _setup(root / 'host', patch)
    def ready(self, binding, call_ref, **kwargs):
        from synaptic_tuner.api.v1.results import TrainingRunRef
        run = TrainingRunRef(binding.execution_binding.run_ref, binding.command.preparation.project_ref)
        run_root = root / 'host/state/synaptic-training' / run.run_id
        retained = reader.load_retained_modal_run(run_root, expected_run=run)
        _publish_provider_fixture(root, binding, call_ref, retained)
        print(json.dumps({'ready': True, 'pid': os.getpid(), 'run': run.to_dict(), 'effects': events,
                          'census_digest': _census_digest(root, run.to_dict())}), flush=True)
        # The parent kills this exact Popen process while the original host's
        # SQLite connections and runner stack are still alive.
        signal.alarm(60)
        signal.pause()
        raise AssertionError('producer must be killed before runner returns')
    patch.setattr(ModalPackagedCoordinatorReaderV1, '_poll_packaged_call', ready)
    runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')


def _consumer(root, role):
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from synaptic_tuner.api.v1.training_facade import TrainingAPI
    from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
    from tuner.training.modal_host_download import download_verified_modal_artifact
    state = _snapshot(root)
    run = TrainingRunRef.from_dict(state['run'])
    scope = reader.ModalClientBinding(**state['client_binding'])
    facts = reader.ModalPackagedRuntimeFactsV1.from_dict(state['facts'])
    client = object()
    reads = []
    class Volume:
        def __init__(self, volume_id):
            self.object_id, self.is_hydrated = volume_id, False
        @staticmethod
        def from_name(name, **kwargs):
            assert kwargs.get('create_if_missing') is False and kwargs['client'] is client
            matches = [identifier for identifier, stored_name in state['volume_names'] if stored_name == name]
            assert len(matches) == 1
            return Volume(matches[0])
        def hydrate(self, supplied):
            assert supplied is client
            self.is_hydrated = True
            return self
        def read_file(self, path):
            reads.append('file')
            payload = base64.b64decode(state['volumes'][self.object_id][path], validate=True)
            midpoint = max(1, len(payload) // 2)
            yield payload[:midpoint]
            if midpoint < len(payload):
                yield payload[midpoint:]
        def iterdir(self, prefix, recursive=True):
            reads.append('list')
            for path, encoded in sorted(state['volumes'][self.object_id].items()):
                if path.startswith(prefix):
                    yield SimpleNamespace(path=path, size=len(base64.b64decode(encoded)), type=1, mtime=1)
    class Call:
        is_hydrated = True
        object_id = state.get('call_ref', 'fc-owned')
        def __init__(self):
            self.client = client
        def cancel(self, *, terminate_containers):
            assert role == 'cancel' and terminate_containers is True
            with (root / 'cancel-events').open('a') as stream:
                stream.write(self.object_id + '\n')
            if state.get('cancel_lost'):
                raise RuntimeError('controlled cancellation reply loss')
        def hydrate(self, supplied):
            assert supplied is client
            return self
        def get(self, *, timeout):
            assert timeout == 0
            assert state['phase'] != 'pending'
            return state['call_result']
    class Calls:
        @staticmethod
        def from_id(reference, *, client):
            assert reference == state.get('call_ref', 'fc-owned')
            return Call()
    sdk = SimpleNamespace(__version__='1.5.4', Volume=Volume, FunctionCall=Calls)
    patch = pytest.MonkeyPatch()
    def forbidden(*args, **kwargs):
        raise AssertionError('consumer cannot create execution authority or effects')
    patch.setattr(TrainingAPI, 'start', forbidden)
    mutation = role in {'cancel', 'reconcile'}
    if not mutation:
        patch.setattr(ModalPackagedHostEffectsV1, 'authorize', forbidden)
        patch.setattr(ModalPackagedHostEffectsV1, 'bind', forbidden)
    patch.setattr(runner, '_qualified_runtime', forbidden)
    patch.setattr(reader, 'validate_local_launcher', lambda: None)
    from tuner.execution.providers.modal.facade import ModalFunctionCallState
    def status_read(*args, **kwargs):
        reads.append('poll')
        return (ModalFunctionCallState.PENDING if state['phase'] == 'pending'
                else ModalFunctionCallState.RETURNED)
    patch.setattr(reader.ExplicitModal154ReadFacade, 'observe_known_call_status', status_read)
    patch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: (client, scope))
    patch.setattr(reader, 'observe_modal_host_scope', lambda **kwargs: (scope.account_ref, scope.workspace_ref, scope.environment_ref, scope.client_ref))
    patch.setattr(reader._CurrentPackagedDeploymentReader, 'observe', lambda *args, **kwargs: facts)
    if role == 'cancel' and state.get('cancel_receipt_gap'):
        from tuner.execution.providers.modal.packaged_effects import ModalPackagedFoundationEffectExecutor
        original_execute = ModalPackagedFoundationEffectExecutor.execute_once
        def lost_receipt(self, payload, request):
            original_execute(self, payload, request)
            assert payload.payload_kind == 'cancel-payload/v2'
            raise RuntimeError('controlled post-ack receipt loss')
        patch.setattr(ModalPackagedFoundationEffectExecutor, 'execute_once', lost_receipt)
    run_root = root / 'host/state/synaptic-training' / run.run_id
    from tuner.handlers.runs_handler import _CurrentCancellationGrant
    opener = reader.open_retained_modal_lifecycle if mutation else reader.open_retained_modal_runs
    with opener(root=run_root, run=run, modal_profile='fixture', modal_environment='main', sdk=sdk,
                **({'cancellation_grants': _CurrentCancellationGrant(run)} if mutation else {})) as view:
        before = view.runs.show(run).state.value
        if role == 'cancel':
            if state.get('cancel_receipt_gap'):
                from synaptic_tuner.api.v1.runs_facade import RunOperationError
                with pytest.raises(RunOperationError) as failure:
                    view.runs.cancel(run, 'operator stop')
                assert failure.value.code.value == 'state_conflict'
                workflow = view.runs._operations._workflows.get(run)
                record = view.runs._operations._operations._foundation.get(workflow.cancel.effect_id)
                assert workflow.phase.value == 'cancel_intent_recorded'
                assert record.attempt_count == 1 and record.dispatch.value == 'orphaned_unproven'
                assert not record.results
                return {'pid': os.getpid(), 'before': before, 'state': workflow.phase.value,
                        'cancel_intent': workflow.cancel.to_dict(), 'attempt_count': record.attempt_count,
                        'dispatch': record.dispatch.value, 'census_digest': _census_digest(root, run.to_dict())}
            assert view.runs.cancel(run, 'operator stop').state.value == 'reconcile_required'
        elif role == 'reconcile':
            assert view.runs.reconcile(run).state.value == 'reconcile_required'
        elif role == 'pending':
            outcome = view.runs.outcome(run)
            assert outcome.state.value == before == state.get('expected_state', 'queued')
            assert reads == ['poll']
        elif role == 'mismatch':
            from synaptic_tuner.api.v1.runs_facade import RunOperationError
            with pytest.raises(RunOperationError) as failure:
                view.runs.outcome(run)
            assert failure.value.code.value == 'provider_read_invalid'
            assert view.runs.show(run).state.value == before == state.get('expected_state', 'queued')
            assert not view.runs.show(run).artifacts
        elif role == 'verified':
            restored = view.runs.show(run)
            assert restored.state.value == 'succeeded'
            assert {item.role: item.sha256 for item in restored.artifacts} == state['artifact_hashes']
            from synaptic_tuner.api.v1.runs_facade import RunListRequest
            page = view.runs.list(RunListRequest(run.project_ref))
            assert len(page.outcomes) == 1 and page.outcomes[0].run == run
            assert not reads
        else:
            assert view.runs.outcome(run).state.value == 'succeeded'
            assert reads.count('poll') == 1
            assert view.runs.verify(run).verified
            observed = {}
            for artifact_role in state['artifact_hashes']:
                path = download_verified_modal_artifact(view.runs, run, role=artifact_role,
                    output_root=root / 'downloads', maximum_bytes=state.get('maximum_artifact_bytes', 4096))
                observed[artifact_role] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert observed == state['artifact_hashes']
        restored = view.runs.show(run)
        workflow = view.runs._operations._workflows.get(run)
        cancel_intent = None if workflow.cancel is None else workflow.cancel.to_dict()
        if state.get('expected_cancel') is not None:
            original = view.runs._operations._workflows.get(run)
            if role == 'reconcile':
                prior = state['expected_cancel']
                current = original.cancel.to_dict()
                for key in ('command_digest', 'effect_id', 'kind'):
                    assert current[key] == prior[key]
                for key in ('foundation_binding_digests', 'foundation_outcome_digests'):
                    assert current[key][:len(prior[key])] == prior[key]
            else:
                assert original.cancel.to_dict() == state['expected_cancel']
            assert original.bound_cancellation is None
    return {'pid': os.getpid(), 'before': before, 'state': restored.state.value,
            'verified_artifacts': len(restored.artifacts), 'reads': reads,
            'cancel_intent': cancel_intent,
            'census_digest': _census_digest(root, run.to_dict())}


def _child(role, root, expected_source):
    assert not any(os.environ.get(name) for name in _CREDENTIAL_NAMES)
    assert Path(reader.__file__).resolve().parents[2] == expected_source
    assert Path(runner.__file__).resolve().parents[2] == expected_source
    if _TEST_ROOT != expected_source:
        assert not Path(reader.__file__).resolve().is_relative_to(_TEST_ROOT)
        assert Path(reader.__file__).resolve().is_relative_to(Path(sys.prefix))
    if role == 'producer':
        _producer(root)
    else:
        print(json.dumps(_consumer(root, role)), flush=True)


def _command(root, role):
    return [sys.executable, '-I', '-B', '-c', _BOOTSTRAP, str(_TEST_ROOT), str(_SOURCE), role, str(root)]


def _census(root, run):
    path = root / 'host/state/synaptic-training' / run['run_id'] / 'workflow.sqlite3'
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        return tuple(db.execute('SELECT partition,key,revision,canonical FROM records ORDER BY partition,key')), tuple(db.execute('SELECT partition,stream_key,sequence,canonical FROM streams ORDER BY partition,stream_key,sequence'))


def _census_digest(root, run):
    digest = hashlib.sha256()
    for table, rows in zip(('records', 'streams'), _census(root, run), strict=True):
        digest.update(table.encode())
        for row in rows:
            for value in row:
                raw = value if type(value) is bytes else json.dumps(value).encode()
                digest.update(len(raw).to_bytes(8, 'big'))
                digest.update(raw)
    return digest.hexdigest()


@pytest.mark.skipif(os.name != 'posix', reason='finite SIGKILL host-loss proof is POSIX-only')
def test_killed_host_reconnects_in_fresh_processes_with_real_signed_artifacts(tmp_path):
    root = tmp_path / 'process-proof'
    root.mkdir(mode=0o700)
    scratch = root / 'child-scratch'
    scratch.mkdir(mode=0o700)
    environment = {'PATH': os.defpath, 'TMPDIR': str(scratch)}
    producer = subprocess.Popen(_command(root, 'producer'), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(producer.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=60), 'producer did not reach retained known-call checkpoint'
            payload = os.read(producer.stdout.fileno(), 4096)
            if not payload:
                producer.wait(timeout=5)
                raise AssertionError(producer.stderr.read(8192).decode())
            ready = json.loads(payload)
        assert ready['ready'] is True and ready['pid'] == producer.pid
        assert ready['effects'] == ['bootstrap', 'cpu', 'stage', 'spawn']
        producer.kill()
        assert producer.wait(timeout=5) == -signal.SIGKILL
    finally:
        if producer.poll() is None:
            producer.kill()
            producer.wait(timeout=5)
        producer.stdout.close()
        producer.stderr.close()
    import shutil
    for path in (root / 'host').iterdir():
        if path.name == 'state':
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    run_root = root / 'host/state/synaptic-training' / ready['run']['run_id']
    for path in run_root.iterdir():
        if path.name in {'authority-keys.json', 'readback-inputs.json', 'workflow.sqlite3',
                         'workflow.sqlite3-wal', 'workflow.sqlite3-shm', 'workflow.sqlite3-journal'}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    pending = subprocess.run(_command(root, 'pending'), env=environment, capture_output=True, timeout=60)
    assert pending.returncode == 0, pending.stderr.decode()
    first = json.loads(pending.stdout)
    assert first['pid'] != producer.pid and first['state'] == 'queued'
    assert first['census_digest'] == ready['census_digest']
    # The parent does not open either database until the first fresh consumer
    # has reopened retained state and completed its public pending observation.
    assert _census_digest(root, ready['run']) == ready['census_digest']
    snapshot = _snapshot(root)
    snapshot['phase'] = 'completed'
    temporary = root / 'completed-provider.json'
    _write(temporary, snapshot)
    temporary.replace(root / 'provider.json')
    completed = subprocess.run(_command(root, 'completed'), env=environment, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr.decode()
    second = json.loads(completed.stdout)
    assert second['pid'] not in (producer.pid, first['pid'])
    assert second['before'] == 'queued' and second['state'] == 'succeeded'
    assert second['verified_artifacts'] == 5
    assert second['census_digest'] != ready['census_digest']
    verified_census = _census_digest(root, ready['run'])
    verified = subprocess.run(_command(root, 'verified'), env=environment, capture_output=True, timeout=60)
    assert verified.returncode == 0, verified.stderr.decode()
    third = json.loads(verified.stdout)
    assert third['pid'] not in (producer.pid, first['pid'], second['pid'])
    assert third['verified_artifacts'] == 5 and not third['reads']
    assert third['census_digest'] == verified_census
    assert _census_digest(root, ready['run']) == verified_census
