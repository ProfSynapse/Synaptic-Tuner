"""Known retained calls reconnect without restoring submission authority."""
from __future__ import annotations

import shutil
import sys

import pytest

from synaptic_tuner.api.v1.runs_facade import RunOperationCode, RunOperationError
from tuner.training import modal_retained_reader as reader
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1, ModalPackagedReadUnavailable
from tests.training.test_modal_standalone_runner import _setup


def _active(tmp_path, monkeypatch):
    original_poll = ModalPackagedCoordinatorReaderV1._poll_packaged_call
    plan, context, effects = _setup(tmp_path, monkeypatch)
    captured = []
    with monkeypatch.context() as stop:
        def interrupted(self, binding, reference, **kwargs):
            captured.append((binding, reference))
            raise ModalPackagedReadUnavailable('modal_packaged_call_read_unavailable')
        stop.setattr(ModalPackagedCoordinatorReaderV1, '_poll_packaged_call', interrupted)
        with pytest.raises(runner.ModalStandalonePhaseUnavailable) as unavailable:
            runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
        assert unavailable.value.phase == 'RUN_CALL_OBSERVE'
        run = unavailable.value.retained_run
    binding, call_ref = captured[0]
    root = tmp_path / 'state' / 'synaptic-training' / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    scope = reader.ModalClientBinding(**retained.inputs['client_binding'])
    client = object()
    polls = []
    state = ['pending']
    class Call:
        is_hydrated = True
        object_id = call_ref
        def hydrate(self, supplied):
            assert supplied is client
            return self
        def get(self, *, timeout):
            assert timeout == 0
            polls.append(state[0])
            if state[0] == 'pending':
                raise TimeoutError()
            if state[0] == 'unknown':
                raise RuntimeError('untrusted provider text')
            if state[0] == 'none':
                return None
            if state[0] == 'failed':
                return {'schema_version': 'synaptic-modal-packaged-worker-result/v1', 'effect_id': 'unavailable', 'status_code': 'failed', 'completion_sha256': '0' * 64}
            return {'schema_version': 'synaptic-modal-packaged-worker-result/v1', 'effect_id': binding.command.operation.effect.effect_id, 'status_code': 'completed', 'completion_sha256': 'f' * 64}
    class Calls:
        @staticmethod
        def from_id(reference, *, client):
            assert reference == call_ref
            return Call()
    monkeypatch.setattr(sys.modules['modal'], 'FunctionCall', Calls, raising=False)
    monkeypatch.setattr(ModalPackagedCoordinatorReaderV1, '_poll_packaged_call', original_poll)
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: (client, scope))
    # Remove all owned mutable source/config/data inputs. Only private retained
    # state remains; every read context opens new SQLite connections.
    for path in tmp_path.iterdir():
        if path.name == 'state':
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return run, root, effects, state, polls


def _open(run, root):
    return reader.open_retained_modal_runs(root=root, run=run, modal_profile='fixture', modal_environment='main', sdk=sys.modules['modal'])


def test_pending_then_completed_across_fresh_contexts(tmp_path, monkeypatch):
    run, root, effects, state, polls = _active(tmp_path, monkeypatch)
    with _open(run, root) as view:
        before = view.runs.show(run)
        assert before.state.value == 'queued'
        assert view.runs.outcome(run) == before
        with pytest.raises(RunOperationError):
            view.runs.verify(run)
    assert polls == ['pending']
    state[0] = 'completed'
    with _open(run, root) as view:
        assert view.runs.outcome(run).state.value == 'succeeded'
    assert polls == ['pending', 'completed']
    with _open(run, root) as view:
        assert view.runs.show(run).state.value == 'succeeded'
    assert effects == ['bootstrap', 'cpu', 'stage', 'spawn']


@pytest.mark.parametrize('status', ['unknown', 'failed', 'none'])
def test_unknown_or_failed_call_preserves_original_checkpoint(tmp_path, monkeypatch, status):
    run, root, effects, state, polls = _active(tmp_path, monkeypatch)
    state[0] = status
    with _open(run, root) as view:
        before = view.runs.show(run)
        with pytest.raises(RunOperationError) as unavailable:
            view.runs.outcome(run)
        assert unavailable.value.code is RunOperationCode.PROVIDER_READ_INVALID
        assert view.runs.show(run) == before
    with _open(run, root) as view:
        assert view.runs.show(run) == before
    assert polls == [status]
    assert effects == ['bootstrap', 'cpu', 'stage', 'spawn']
