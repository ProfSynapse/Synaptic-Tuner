"""Real caller-owned records and signed receipts; scripted CLI only, no daemon."""
from dataclasses import asdict, replace
import hashlib
import hmac
import json
import time
from threading import Event, Thread

import pytest

from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.cloud.derived_training_image import DockerLaunchAuthority, _validate_docker_inputs
from tuner.cloud.hf_training_image_lock import CommandResult
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.docker_provider_v1.cli_transport import DockerCLISessionV1
from tuner.execution.providers.docker_provider_v1.cancellation_transport import DockerCancellationTransportV1
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerCancellationEvidenceV1, DockerCancellationLookupRequestV1,
    DockerCancellationRequestV1, DockerEffectIdentityV1, DockerProviderError,
    DockerLookupRequestV1, DockerLookupPurposeV1, PreparedDockerPlanV1, labels_for,
)

CID = 'f' * 64
IMAGE = 'sha256:' + 'a' * 64


class Signer:
    key = b'declared deterministic fixture key, not a credential'
    def tag(self, content):
        return hmac.new(self.key, canonical_bytes(asdict(content)), hashlib.sha256).hexdigest()
    def issue(self, content):
        return AuthenticatedDockerCancellationEvidenceV1(content, 'fixture-docker', 'fixture-key', self.tag(content))
    def authenticate_cancellation(self, evidence):
        return (type(evidence) is AuthenticatedDockerCancellationEvidenceV1
                and evidence.authority_ref == 'fixture-docker' and evidence.key_ref == 'fixture-key'
                and hmac.compare_digest(evidence.tag, self.tag(evidence.content)))


@pytest.fixture
def setup(tmp_path, profile, plan, run):
    prepared = PreparedDockerPlanV1(profile, run.project_ref, run.run_id,
        plan.plan_fingerprint, plan.basis.source_digest, 'b' * 64)
    labels = labels_for(DockerEffectIdentityV1('c' * 64, 'submit-effect', 'submit', prepared))
    request = DockerCancellationRequestV1(
        DockerEffectIdentityV1('d' * 64, 'cancel-effect', 'cancel', prepared),
        labels, CID, 'e' * 64, 'a' * 64,
    )
    executable = tmp_path / 'docker'
    executable.write_bytes(b'fixture executable never run')
    config = tmp_path / 'docker-config'
    config.mkdir()
    authority = DockerLaunchAuthority({}, _validate_docker_inputs(executable, config))
    commands = []
    state = {'status': 'running', 'code': 0, 'daemon': 'fixture-daemon', 'image': IMAGE,
             'container': CID, 'labels': labels, 'stop_error': None, 'post': 'exited'}
    def runner(spec):
        args = spec.argv[5:]
        commands.append(args)
        assert spec.argv[3:5] == ('--host', 'unix:///fixture.sock')
        assert 1 <= spec.timeout_seconds <= 30 and spec.maximum_output_bytes == 65536
        if args[0] == 'info':
            output = state['daemon']
        elif args[:2] == ('image', 'inspect'):
            output = json.dumps(IMAGE) + '\n[]'
        elif args[:2] == ('container', 'inspect'):
            output = '\n'.join(map(json.dumps, [state['container'], '/' + state['labels'].container_name,
                state['image'], DockerCLISessionV1._labels(state['labels']), state['status'], state['code']]))
        elif args[:2] == ('container', 'ls'):
            output = CID
        elif args[:2] == ('container', 'stop'):
            assert args[2:] == ('--timeout', '5', CID)
            if state['stop_error']:
                raise state['stop_error']('controlled private failure')
            state['status'] = state['post']
            output = CID
        else:
            raise AssertionError(args)
        return CommandResult(output.encode())
    session = DockerCLISessionV1(authority=authority, endpoint='unix:///fixture.sock',
        daemon_id='fixture-daemon', image_refs={profile.image.image_ref: IMAGE},
        source_roots={}, artifact_roots={}, environment={}, deadline=time.monotonic() + 120,
        runner=runner, profile=profile)
    signer = Signer()
    store = InMemoryDurableRecordStoreV1()
    wrapper = DockerCancellationTransportV1(session=session, records=store,
        issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=5)
    return wrapper, request, commands, state, signer, store


def stops(commands):
    return sum(args[:2] == ('container', 'stop') for args in commands)


def test_stop_receipt_committed_then_duplicate_and_lookup_never_call_cli(setup):
    wrapper, request, commands, state, signer, store = setup
    evidence = wrapper.stop_once(request)
    assert signer.authenticate_cancellation(evidence)
    assert evidence.content.request_digest == request.digest
    assert stops(commands) == 1
    before = list(commands)
    wrapper._session._deadline = time.monotonic() - 1
    assert wrapper.stop_once(request) == evidence
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).evidence == evidence
    assert commands == before
    row = store.read(partition='effects', key=wrapper._key(request))
    assert row.revision == 2


@pytest.mark.parametrize('fault', ['wrong_id', 'wrong_image', 'wrong_labels', 'wrong_daemon',
                                 'created', 'exited', 'dead'])
def test_wrong_or_already_terminal_target_never_stops(setup, fault):
    wrapper, request, commands, state, _, _ = setup
    if fault == 'wrong_id':
        state['container'] = '1' * 64
    elif fault == 'wrong_image':
        state['image'] = 'sha256:' + 'b' * 64
    elif fault == 'wrong_labels':
        state['labels'] = replace(request.submit_labels, namespace_ref='other')
    elif fault == 'wrong_daemon':
        state['daemon'] = 'other'
    else:
        state['status'] = fault
    with pytest.raises(DockerProviderError, match='docker_stop_indeterminate'):
        wrapper.stop_once(request)
    assert stops(commands) == 0
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'


@pytest.mark.parametrize('failure', [TimeoutError, RuntimeError])
def test_unknown_stop_is_never_replayed(setup, failure):
    wrapper, request, commands, state, _, _ = setup
    state['stop_error'] = failure
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    before = list(commands)
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert commands == before and stops(commands) == 1
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'


def test_host_signing_has_separate_time_semantics_from_cli_deadline(setup):
    wrapper, request, commands, _, signer, store = setup
    def host_issue(content):
        wrapper._session._deadline = time.monotonic() - 1
        return signer.issue(content)
    wrapper._issue = host_issue
    retained = wrapper.stop_once(request)
    assert signer.authenticate_cancellation(retained)
    before = list(commands)
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).evidence == retained
    assert commands == before and stops(commands) == 1
def test_changed_request_for_same_effect_is_not_a_second_stop(setup):
    wrapper, request, commands, _, _, _ = setup
    wrapper.stop_once(request)
    before = list(commands)
    changed = replace(request, reason_digest='1' * 64)
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(changed)
    assert wrapper.lookup(DockerCancellationLookupRequestV1(changed, 1)).disposition.value == 'indeterminate'
    assert commands == before


def test_cas_failure_does_not_return_unretained_receipt_or_retry(setup, monkeypatch):
    wrapper, request, commands, _, _, store = setup
    monkeypatch.setattr(type(store), 'compare_and_swap', lambda *args, **kwargs: False)
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    before = list(commands)
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert commands == before and stops(commands) == 1


def test_signature_tamper_is_not_recovered_or_replayed(setup):
    wrapper, request, commands, _, _, store = setup
    wrapper.stop_once(request)
    row = store.read(partition='effects', key=wrapper._key(request))
    document = json.loads(row.canonical)
    document['evidence']['tag'] = '0' * 64
    assert store.compare_and_swap(partition='effects', key=row.key, expected_revision=row.revision,
        canonical=canonical_bytes(document))
    before = list(commands)
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert commands == before


def test_valid_signature_for_wrong_stop_observation_is_not_accepted(setup):
    wrapper, request, commands, _, signer, _ = setup
    wrapper._issue = lambda content: signer.issue(replace(content, evidence_digest='0' * 64))
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    before = list(commands)
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert commands == before and stops(commands) == 1


@pytest.mark.parametrize('failure', ['wrong_ack', 'daemon_drift'])
def test_acknowledgment_fault_cannot_publish_a_receipt(setup, failure):
    wrapper, request, commands, state, _, _ = setup
    original = wrapper._session._runner
    def changed(spec):
        result = original(spec)
        if spec.argv[5:7] == ('container', 'stop'):
            if failure == 'wrong_ack':
                return CommandResult(b'other-container')
            state['daemon'] = 'replacement-daemon'
        return result
    wrapper._session._runner = changed
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    before = list(commands)
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert commands == before and stops(commands) == 1
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'


@pytest.mark.parametrize('lost_reply', [False, True])
def test_actual_sqlite_reopen_preserves_receipt_or_uncertainty(setup, tmp_path, lost_reply):
    wrapper, request, commands, state, signer, _ = setup
    tmp_path.chmod(0o700)
    path = tmp_path / 'caller.sqlite3'
    with SQLiteDurableStoreV1(path) as records:
        first = DockerCancellationTransportV1(session=wrapper._session, records=records,
            issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=5)
        if lost_reply:
            state['stop_error'] = RuntimeError
            with pytest.raises(DockerProviderError):
                first.stop_once(request)
        else:
            evidence = first.stop_once(request)
    before = list(commands)
    with SQLiteDurableStoreV1(path, create=False) as records:
        cold = DockerCancellationTransportV1(session=wrapper._session, records=records,
            issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=5)
        lookup = cold.lookup(DockerCancellationLookupRequestV1(request, 2))
        if lost_reply:
            assert lookup.disposition.value == 'indeterminate'
            with pytest.raises(DockerProviderError):
                cold.stop_once(request)
        else:
            assert lookup.evidence == evidence
            assert cold.stop_once(request) == evidence
    assert commands == before and stops(commands) == 1


@pytest.mark.parametrize('code', [0, 137])
def test_completed_race_retains_existing_lookup_success_or_failure(setup, code):
    wrapper, request, commands, state, _, _ = setup
    state['code'] = code
    wrapper.stop_once(request)
    observed = wrapper._session.lookup(DockerLookupRequestV1(
        request.submit_labels, DockerLookupPurposeV1.OBSERVE, 1))
    assert observed.disposition.value == 'found'
    assert observed.phase.value == ('succeeded' if code == 0 else 'failed')
    assert stops(commands) == 1


def test_concurrent_duplicate_cannot_issue_a_second_stop(setup):
    wrapper, request, commands, _, _, _ = setup
    original = wrapper._session._runner
    entered, release = Event(), Event()
    results = []
    def held(spec):
        if spec.argv[5:7] == ('container', 'stop'):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('controlled test hold')
        return original(spec)
    wrapper._session._runner = held
    def first():
        try:
            results.append(wrapper.stop_once(request))
        except BaseException as error:
            results.append(error)
    worker = Thread(target=first)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(DockerProviderError):
            wrapper.stop_once(request)
        assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(results) == 1 and type(results[0]) is AuthenticatedDockerCancellationEvidenceV1
    assert stops(commands) == 1


def test_receipt_commit_with_lost_return_recovers_lookup_without_cli(setup, monkeypatch):
    wrapper, request, commands, _, _, store = setup
    original = type(store).compare_and_swap
    def lost_return(self, **kwargs):
        assert original(self, **kwargs)
        raise OSError('controlled lost storage response')
    monkeypatch.setattr(type(store), 'compare_and_swap', lost_return)
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    before = list(commands)
    recovered = wrapper.lookup(DockerCancellationLookupRequestV1(request, 1))
    assert recovered.disposition.value == 'found'
    assert wrapper.stop_once(request) == recovered.evidence
    assert commands == before and stops(commands) == 1


@pytest.mark.parametrize('post', ['running', 'created'])
def test_stop_ack_without_nonrunning_observation_stays_uncertain(setup, post):
    wrapper, request, commands, state, _, _ = setup
    state['post'] = post
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert stops(commands) == 1
    assert wrapper.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'


@pytest.mark.parametrize('grace', [None, True, 0, 11, 1.5])
def test_grace_is_an_explicit_finite_integer(setup, grace):
    wrapper, _, commands, _, signer, store = setup
    with pytest.raises(TypeError):
        DockerCancellationTransportV1(session=wrapper._session, records=store,
            issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=grace)
    assert commands == []


def test_insufficient_deadline_never_issues_stop(setup):
    wrapper, request, commands, _, _, _ = setup
    wrapper._session._deadline = time.monotonic() + 5
    with pytest.raises(DockerProviderError):
        wrapper.stop_once(request)
    assert stops(commands) == 0


def test_actual_sqlite_receipt_write_failure_keeps_claim_after_reopen(setup, tmp_path):
    wrapper, request, commands, _, signer, _ = setup
    tmp_path.chmod(0o700)
    path = tmp_path / 'failed-receipt.sqlite3'
    with SQLiteDurableStoreV1(path) as records:
        records._connection.execute("CREATE TEMP TRIGGER fail_receipt BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'controlled receipt fault'); END")
        first = DockerCancellationTransportV1(session=wrapper._session, records=records,
            issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=5)
        with pytest.raises(DockerProviderError):
            first.stop_once(request)
        row = records.read(partition='effects', key=first._key(request))
        assert row.revision == 1 and json.loads(row.canonical)['evidence'] is None
    before = list(commands)
    with SQLiteDurableStoreV1(path, create=False) as records:
        cold = DockerCancellationTransportV1(session=wrapper._session, records=records,
            issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=5)
        assert cold.lookup(DockerCancellationLookupRequestV1(request, 1)).disposition.value == 'indeterminate'
        with pytest.raises(DockerProviderError):
            cold.stop_once(request)
    assert commands == before and stops(commands) == 1


def test_sqlite_ownership_remains_on_host_thread(setup, tmp_path):
    wrapper, request, commands, _, signer, _ = setup
    tmp_path.chmod(0o700)
    with SQLiteDurableStoreV1(tmp_path / 'thread-owned.sqlite3') as records:
        bound = DockerCancellationTransportV1(session=wrapper._session, records=records,
            issue_evidence=signer.issue, evidence_authority=signer, grace_seconds=5)
        errors = []
        def off_owner():
            try:
                bound.stop_once(request)
            except DockerProviderError as error:
                errors.append(error)
        worker = Thread(target=off_owner)
        worker.start()
        worker.join(5)
        assert not worker.is_alive() and len(errors) == 1
        assert records.read(partition='effects', key=bound._key(request)) is None
        assert commands == []


@pytest.mark.parametrize('exit_code', [0, 137])
def test_existing_executor_receipt_and_reader_preserve_completion_race(setup, profile, plan, run, exit_code):
    from types import SimpleNamespace
    from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1
    from tuner.execution.foundation_v2.observations import ObservationDisposition
    from tuner.execution.providers.docker_provider_v1.effects import DockerEffectExecutorV1
    from tuner.execution.providers.docker_provider_v1.model import AuthenticatedDockerCommandBindingV1
    from tests.execution.providers.docker_provider_v1.test_effects import prepared, make_binding, payload, request as resolution
    from tests.execution.providers.docker_provider_v1.test_reader import BypassReader, Authority as ReadAuthority

    wrapper, _, commands, state, signer, _ = setup
    prepared_plan = prepared(profile, plan, run)
    submit = make_binding('submit', '1' * 64, 'submit-effect', prepared_plan)
    labels = labels_for(submit.identity)
    binding = make_binding('cancel', '2' * 64, 'cancel-effect', prepared_plan,
        original_submit_command_bytes=submit.command_bytes, cancel_submit_labels=labels,
        cancel_container_ref=CID, cancel_reason_digest='3' * 64, cancel_authorization_digest='4' * 64)
    class BindingSigner:
        authority_ref, key_ref = 'fixture-binding', 'fixture-key'
        def tag(self, value):
            return hmac.new(signer.key, value.binding_digest.encode(), hashlib.sha256).hexdigest()
        def authenticate(self, value):
            return (type(value) is AuthenticatedDockerCommandBindingV1
                and value.binding_digest == value.content.binding_digest
                and value.authority_ref == self.authority_ref and value.key_ref == self.key_ref
                and hmac.compare_digest(value.tag, self.tag(value.content)))
    authority = BindingSigner()
    authenticated = AuthenticatedDockerCommandBindingV1(binding, binding.binding_digest,
        authority.authority_ref, authority.key_ref, authority.tag(binding))
    class Catalog:
        def resolve(self, digest):
            assert digest == binding.command_digest
            return authenticated
    state['labels'], state['code'] = labels, exit_code
    executor = DockerEffectExecutorV1(profile, Catalog(), authority, None, None,
        wrapper._session, wrapper, signer)
    acknowledged = executor.execute_once(payload(profile, 'cancel'), resolution(profile, binding))
    assert acknowledged.disposition is ObservationDisposition.FOUND
    assert stops(commands) == 1
    # Reuse the existing reader fixture's admitted request seam: full read
    # authorization is separately tested; the production observe/phase path
    # and actual CLI lookup are exercised here, not a parallel classification.
    ref = ScopedProviderRunRefV1(profile.provider.provider_id, profile.provider.profile_ref,
        profile.scope.account_ref, profile.scope.namespace_ref, CID)
    read = BypassReader(submit, ref, labels, wrapper._session, ReadAuthority())
    request = SimpleNamespace(run=run, request_digest='5' * 64,
        source_workflow_record_digest='6' * 64, source_revision=1,
        provider_run=SimpleNamespace(binding_digest='7' * 64))
    observed = read.observe(request)
    assert observed.content.phase.value == ('succeeded' if exit_code == 0 else 'failed')
    assert observed.content.phase.value != 'cancelled'
