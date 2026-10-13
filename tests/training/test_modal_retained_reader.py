"""New terminal retained runs reconstruct existing public reads without sources."""
import sys
import copy
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.runs_facade import RunListRequest, RunArtifactRequest, RunLogsRequest
from tuner.training import modal_standalone_runner as runner
from tuner.training import modal_retained_reader as reader
from tests.training.test_modal_standalone_runner import _setup


def test_terminal_reader_replays_new_stores_and_exposes_public_reads(tmp_path, monkeypatch):
    plan, context, events = _setup(tmp_path, monkeypatch)
    report = runner.run_modal_standalone_job(
        plan=plan, context=context, modal_profile="fixture", modal_environment="main",
    )
    run = report.run
    root = tmp_path / "state" / "synaptic-training" / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    reader._typed_inputs(retained)  # pure reconstruction failures remain visible in this provider-free test
    monkeypatch.setattr(reader, "validate_local_launcher", lambda: None)
    binding = reader.ModalClientBinding(**retained.inputs["client_binding"])
    monkeypatch.setattr(reader, "open_modal_host_scope", lambda **kwargs: (object(), binding))
    with reader.open_retained_modal_runs(
        root=root, run=run, modal_profile="fixture", modal_environment="main", sdk=sys.modules["modal"],
    ) as view:
        assert view.run == run and not hasattr(view, "training")
        assert view.runs.show(run).state.value == "succeeded"
        assert view.runs.list(RunListRequest(run.project_ref)).outcomes[0].run == run
        assert view.runs.outcome(run).state.value == "succeeded"
        assert b"".join(view.runs.artifacts(RunArtifactRequest(run, "training_metrics", 1024)).iter_bytes()) == b"data"
        for operation in (
            lambda: view.runs.logs(RunLogsRequest(run)),
            lambda: view.runs.cancel(run, "operator request"),
            lambda: view.runs.reconcile(run),
            lambda: view.runs.reverify(run),
        ):
            with pytest.raises(reader.RunOperationError) as unavailable:
                operation()
            assert unavailable.value.code is reader.RunOperationCode.CAPABILITY_UNAVAILABLE
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


@pytest.mark.parametrize('field', ['request_id', 'request_json', 'resources', 'runtime_image', 'cost', 'authority_key', 'reader_key', 'quote'])
def test_retained_changes_reject_before_sdk_scope(tmp_path, monkeypatch, field):
    plan, context, events = _setup(tmp_path, monkeypatch)
    report = runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
    run = report.run
    root = tmp_path / 'state' / 'synaptic-training' / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    inputs = copy.deepcopy(dict(retained.inputs))
    keys = dict(retained.signing_keys)
    qualification = retained.qualification_key
    if field == 'request_id':
        inputs['prepared_request']['request_id'] = 'different-request'
    elif field == 'request_json':
        inputs['prepared_request']['canonical_json'] = '{}'
    elif field == 'resources':
        inputs['components']['resources']['timeout_seconds'] += 1
    elif field == 'runtime_image':
        inputs['components']['runtime']['image'] = 'different-image'
    elif field == 'cost':
        inputs['maximum_cost_minor_units'] += 1
    elif field == 'authority_key':
        keys['SYNAPTIC_TRAIN_AUTHORITY_KEY'] = '0' * 64
    elif field == 'reader_key':
        keys['SYNAPTIC_TRAIN_READER_KEY'] = '0' * 64
    elif field == 'quote':
        inputs['quote_digest'] = '0' * 64
    changed = replace(retained, inputs=inputs, signing_keys=keys, qualification_key=qualification)
    monkeypatch.setattr(reader, 'load_retained_modal_run', lambda *args, **kwargs: changed)
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    calls = []
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: calls.append(kwargs))
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with reader.open_retained_modal_runs(root=root, run=run, modal_profile='fixture', modal_environment='main', sdk=sys.modules['modal']):
            pytest.fail('changed retention admitted')
    assert not calls
    assert events == ['bootstrap', 'cpu', 'stage', 'spawn']


def test_unverified_terminal_run_can_verify_after_local_verifier_failure(tmp_path, monkeypatch):
    import json
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from tuner.training.modal_host_artifacts import ModalPackagedArtifactVerifierV1

    plan, context, events = _setup(tmp_path, monkeypatch)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(ModalPackagedArtifactVerifierV1, 'verify', lambda *args: (_ for _ in ()).throw(ValueError('fixture rejection')))
        with pytest.raises(runner.ModalStandalonePhaseUnavailable):
            runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
    roots = list((tmp_path / 'state' / 'synaptic-training').glob('*/authority-keys.json'))
    assert len(roots) == 1
    root = roots[0].parent
    run = TrainingRunRef.from_dict(json.loads(roots[0].read_bytes())['run'])
    retained = reader.load_retained_modal_run(root, expected_run=run)
    binding = reader.ModalClientBinding(**retained.inputs['client_binding'])
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: (object(), binding))
    with reader.open_retained_modal_runs(root=root, run=run, modal_profile='fixture', modal_environment='main', sdk=sys.modules['modal']) as view:
        assert view.runs.verify(run).verified
        with pytest.raises(reader.RunOperationError) as unavailable:
            view.runs.verify(run)
        assert unavailable.value.code is reader.RunOperationCode.READ_INELIGIBLE
    assert events == ['bootstrap', 'cpu', 'stage', 'spawn']


def test_wrong_qualification_key_refuses_signed_remote_completion(tmp_path, monkeypatch):
    from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
    original_completion = ModalPackagedReader.observe_completion
    plan, context, events = _setup(tmp_path, monkeypatch)
    report = runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
    run = report.run
    root = tmp_path / 'state' / 'synaptic-training' / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    scope = reader.ModalClientBinding(**retained.inputs['client_binding'])
    facts = reader.ModalPackagedRuntimeFactsV1.from_dict(retained.inputs['runtime_facts'])
    key_ref = 'modal-runtime-release-qualification-hmac-v1'
    signer = reader.HMACAuthenticator({key_ref: retained.qualification_key}, allowed_purposes=frozenset({'modal-packaged-completion/v1'}))
    raw = b'{}'
    tag = signer.sign('modal-packaged-completion/v1', raw, key_ref)
    assert signer.verify('modal-packaged-completion/v1', raw, tag, key_ref) is True
    verified = []
    original_verify = reader.HMACAuthenticator.verify
    def verify(self, purpose, payload, authentication, supplied_key):
        result = original_verify(self, purpose, payload, authentication, supplied_key)
        if purpose == 'modal-packaged-completion/v1':
            verified.append(result)
        return result
    monkeypatch.setattr(reader.HMACAuthenticator, 'verify', verify)
    changed = replace(retained, qualification_key=bytes(32))
    monkeypatch.setattr(reader, 'load_retained_modal_run', lambda *args, **kwargs: changed)
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: (object(), scope))
    monkeypatch.setattr(reader.ModalPackagedDeploymentObserver, 'observe', lambda *args, **kwargs: facts)
    monkeypatch.setattr(ModalPackagedReader, 'observe_completion', original_completion)
    reads = []
    def read_complete(self, volume, path, **kwargs):
        reads.append(path)
        return tag if path.endswith('.mac') else raw
    monkeypatch.setattr(reader.ExplicitModal154ReadFacade, 'read_complete', read_complete)
    with reader.open_retained_modal_runs(root=root, run=run, modal_profile='fixture', modal_environment='main', sdk=sys.modules['modal']) as view:
        with pytest.raises(reader.RunOperationError) as invalid:
            list(view.runs.artifacts(RunArtifactRequest(run, 'training_metrics', 1024)).iter_bytes())
        assert invalid.value.code is reader.RunOperationCode.ARTIFACT_CONTENT_INVALID
    assert len(reads) == 2
    assert verified == [False]
    assert events == ['bootstrap', 'cpu', 'stage', 'spawn']


@pytest.mark.parametrize('missing', ['workflow', 'journal'])
def test_missing_retained_database_rejects_before_scope_without_recreation(tmp_path, monkeypatch, missing):
    plan, context, events = _setup(tmp_path, monkeypatch)
    report = runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
    run = report.run
    root = tmp_path / 'state' / 'synaptic-training' / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    path = root / reader.WORKFLOW_FILE if missing == 'workflow' else reader.Path(retained.inputs['attempt_journal'])
    path.unlink()
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: pytest.fail('SDK accessed before store validation'))
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with reader.open_retained_modal_runs(root=root, run=run, modal_profile='fixture', modal_environment='main', sdk=sys.modules['modal']):
            pytest.fail('missing database admitted')
    assert not path.exists()
    assert events == ['bootstrap', 'cpu', 'stage', 'spawn']


def test_nonterminal_retained_run_refuses_before_sdk(tmp_path, monkeypatch):
    import json
    from synaptic_tuner.api.v1.results import TrainingRunRef
    from tuner.execution.coordinator_v1.operations import RunOperationsV1
    plan, context, events = _setup(tmp_path, monkeypatch)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(RunOperationsV1, 'outcome', lambda *args: (_ for _ in ()).throw(ValueError('fixture interruption')))
        with pytest.raises(runner.ModalStandalonePhaseUnavailable):
            runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
    keys = list((tmp_path / 'state' / 'synaptic-training').glob('*/authority-keys.json'))
    assert len(keys) == 1
    run = TrainingRunRef.from_dict(json.loads(keys[0].read_bytes())['run'])
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: pytest.fail('nonterminal SDK read'))
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with reader.open_retained_modal_runs(root=keys[0].parent, run=run, modal_profile='fixture', modal_environment='main', sdk=sys.modules['modal']):
            pytest.fail('nonterminal admitted')
    assert events == ['bootstrap', 'cpu', 'stage', 'spawn']


def test_historical_log_descriptor_cannot_enable_read_view_operations():
    descriptor = reader._modal_packaged_descriptor(logs=True)
    assert descriptor.capabilities.logs
    # Even an underlying historical host advertising logs cannot delegate any
    # of these verbs through the restricted view.
    class Historical:
        def __getattr__(self, name):
            pytest.fail('blocked verb delegated to historical host')
    restricted = reader._TerminalOperations(Historical(), None, None)
    for callback in (lambda: restricted.logs(None), lambda: restricted.cancel(None, 'request'), lambda: restricted.reconcile(None), lambda: restricted.reverify(None)):
        with pytest.raises(reader.RunOperationError) as denied:
            callback()
        assert denied.value.code is reader.RunOperationCode.CAPABILITY_UNAVAILABLE


@pytest.mark.parametrize('invalid', ['profile', 'launcher', 'missing_call'])
def test_pre_provider_admission_refusals(tmp_path, monkeypatch, invalid):
    import sqlite3
    plan, context, events = _setup(tmp_path, monkeypatch)
    report = runner.run_modal_standalone_job(plan=plan, context=context, modal_profile='fixture', modal_environment='main')
    run = report.run
    root = tmp_path / 'state' / 'synaptic-training' / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    monkeypatch.setattr(reader, 'validate_local_launcher', lambda: None)
    if invalid == 'launcher':
        monkeypatch.setattr(reader, 'validate_local_launcher', lambda: (_ for _ in ()).throw(ValueError('unsupported launcher')))
    if invalid == 'missing_call':
        with sqlite3.connect(retained.inputs['attempt_journal']) as db:
            db.execute("DELETE FROM catalogs WHERE catalog_ref = 'packaged-calls-v1'")
    monkeypatch.setattr(reader, 'open_modal_host_scope', lambda **kwargs: pytest.fail('SDK opened on invalid admission'))
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with reader.open_retained_modal_runs(root=root, run=run, modal_profile='wrong' if invalid == 'profile' else 'fixture', modal_environment='main', sdk=sys.modules['modal']):
            pytest.fail('invalid admission succeeded')
    assert events == ['bootstrap', 'cpu', 'stage', 'spawn']
