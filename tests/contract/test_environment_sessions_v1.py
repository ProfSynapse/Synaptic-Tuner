"""Actual retained-input fixture feedback, without opening public starts."""
from dataclasses import replace
import hashlib
import json
import os
import signal
import shutil

import pytest
import yaml

from test_environment_admission_v1 import harness  # noqa: F401
from test_reference_data_v1 import FakeLocalClient
from shared.environments.bounded_local_runtime import BoundedLocalEnvironmentRuntime
from shared.environments.local_worker import WorkerLifecycleError, WorkerEnvironmentRuntime
from shared.llm.metering import MeteredLLMClient
from shared.llm.usage import UsageAccumulator
from synaptic_tuner.api.v1.data_facade import DataAPI, DataMode, DataOperationError, DataRunState
from synaptic_tuner.api.v1.environment_execution import EnvironmentMode
from synaptic_tuner.api.v1.reference.environment_sessions import CapturedEnvironmentSessionFactory


def response(name, **arguments):
    return {'role': 'assistant', 'content': None,
            'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]}


class ScriptedClient(FakeLocalClient):
    def __init__(self, responses, scores):
        super().__init__(responses, structured=[{'repair_score': score} for score in scores])
        self.prompts = []
        self.judge_calls = 0

    def chat(self, messages, **kwargs):
        self.prompts.append(json.dumps(messages))
        return super().chat(messages, **kwargs)

    def structured_output(self, messages, schema, **kwargs):
        self.judge_calls += 1
        self.prompts.append(json.dumps(messages))
        return super().structured_output(messages, schema, **kwargs)


@pytest.fixture(params=['bounded_local', 'worker'])
def selected(harness, tmp_path, monkeypatch, request):  # noqa: F811
    runtime_kind = request.param
    operations, data_request, ports, host, clients, scenario = harness
    import SynthChat.config.format_resolver as formats
    monkeypatch.setattr('SynthChat.engine.MAX_RETRIES', 1)
    monkeypatch.setattr(formats, '_CONFIG_DIR', ports.config_dir)
    rubrics = tmp_path / 'rubrics'
    rubrics.mkdir()
    (rubrics / 'repair.yaml').write_text(yaml.safe_dump({'name': 'repair', 'scope': 'response',
        'pass_threshold': .8, 'judge_prompt': 'Check the environment_result before scoring.',
        'improver_prompt': 'Repair the response using environment_result. Return an assistant message.',
        'output_format': {'type': 'assistant_message'}}))
    policy = json.loads(host.raw['policy'])
    policy['execution'] = {'tool_action_hints': {'files_fetch': 'read', 'files_store': 'write'},
        'candidate_projection': {'response': ['conversations', -1], 'system': ['conversations', 0, 'content']}}
    host.raw['policy'] = json.dumps(policy).encode()
    host.raw['tool-schema'] = json.dumps({'tools': {'files': [
        {'name': 'fetch', 'params': {'required': ['path']}},
        {'name': 'store', 'params': {'required': ['path', 'content']}}]}}).encode()
    host.raw['fixture-schema'] = json.dumps({'type': 'object', 'additionalProperties': False,
        'properties': {'fixture': {'type': 'object'}, 'assertions': {'type': 'array'}}}).encode()
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(host.raw[item.ref]).hexdigest(),
        size_bytes=len(host.raw[item.ref])) for item in host.dependencies)
    environment = {'fixture': {'files': {}}, 'assertions': [{'type': 'file_contains', 'path': 'result.txt', 'text': 'fixed'}]}
    scenario.write_text(yaml.safe_dump({'scenarios': {'selected': {'type': 'behavioral',
        'environment_execution': 'required', 'environment': environment,
        'prompts': {'system': 'system', 'user': 'user', 'assistant': 'assistant'},
        'rubrics': {'response': ['repair']}}}}))
    scratch = tmp_path / 'scratch'
    scratch.mkdir(mode=0o700)
    runtimes, prepared = [], []
    def runtime_factory(*, limits, boundary):
        boundary()
        runtime = (BoundedLocalEnvironmentRuntime(scratch, limits) if runtime_kind == 'bounded_local'
                   else WorkerEnvironmentRuntime(scratch, limits, boundary=boundary))
        runtimes.append(runtime)
        return runtime
    class Factory(CapturedEnvironmentSessionFactory):
        def prepare(self, *args):
            validator = super().prepare(*args)
            prepared.append(validator)
            return validator
    ports = replace(ports, rubrics_dir=rubrics, environment_sessions=Factory(runtime_factory))
    operations._ports = ports
    data_request = replace(data_request, max_iterations=2)
    return operations, data_request, ports, host, clients, scenario, environment, scratch, runtimes, prepared


def forbidden_defaults(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Selected session consulted mutable/default inputs')
    for name in ('load_tool_call_formats', 'load_workspace_formats', 'load_label_mappings', 'load_privacy_profiles', 'load_yaml'):
        monkeypatch.setattr('SynthChat.generator.' + name, forbidden)
    monkeypatch.setattr('shared.validation.parsing.configured_formats.get_configured_wrapper_specs', forbidden)
    monkeypatch.setattr('shared.environments.tool_executor.cli_command_catalog', forbidden)
    monkeypatch.setattr('shared.environments.tool_executor.load_cli_command_catalog', forbidden)


def test_actual_generation_missing_path_feedback_and_fresh_repair(selected, monkeypatch):
    operations, request, ports, _, clients, scenario, _, scratch, runtimes, prepared = selected
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    scenario.unlink()
    shutil.rmtree(ports.config_dir)
    shutil.rmtree(ports.rubrics_dir)
    forbidden_defaults(monkeypatch)
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt')),
                           json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    metered = MeteredLLMClient(inner, UsageAccumulator())
    rows = []
    operations._generate(plan, metered, None, lambda ref, row, passed, writer: rows.append((row, passed)),
                         lambda: None, lambda *args: None, admitted_inputs=capture)
    assert rows[0][1] and inner.chat_calls == 4 and clients.calls == 0
    assert rows[0][0]['conversations'][-1]['tool_calls'][0]['function']['name'] == 'files_store'
    assert any('missing_path' in prompt for prompt in inner.prompts)
    assert len(runtimes) >= 2 and len({id(runtime) for runtime in runtimes}) == len(runtimes)
    assert prepared[0].last_evaluation['passed']
    assert prepared[0].last_evaluation['mode'] == 'execute'
    assert not list(scratch.iterdir())


def test_actual_improvement_reexecutes_changed_candidate_from_retained_input(selected, tmp_path, monkeypatch):
    operations, request, ports, host, clients, _, environment, scratch, runtimes, prepared = selected
    policy = json.loads(host.raw['policy'])
    policy['execution']['candidate_projection']['environment'] = ['environment']
    host.raw['policy'] = json.dumps(policy).encode()
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(host.raw[item.ref]).hexdigest(),
        size_bytes=len(host.raw[item.ref])) for item in host.dependencies)
    dataset = tmp_path / 'seed.jsonl'
    candidate = {'conversations': [{'role': 'system', 'content': 'bound system'},
        {'role': 'user', 'content': 'bound user'}, response('files_fetch', path='result.txt')], 'environment': environment}
    dataset.write_text(json.dumps(candidate) + '\n')
    request = replace(request, mode=DataMode.IMPROVE, scenarios=(), rubric_refs=('repair',), input_dataset_ref=str(dataset))
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    dataset.unlink()
    shutil.rmtree(ports.config_dir)
    shutil.rmtree(ports.rubrics_dir)
    forbidden_defaults(monkeypatch)
    inner = ScriptedClient([json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    rows = []
    operations._improve(plan, MeteredLLMClient(inner, UsageAccumulator()), None,
        lambda ref, row, passed, writer: rows.append((row, passed)), lambda: None, lambda *args: None,
        admitted_inputs=capture)
    assert rows[0][1] and inner.chat_calls == 1 and clients.calls == 0
    assert len(runtimes) == 2 and prepared[0].last_evaluation['passed']
    assert any('missing_path' in prompt for prompt in inner.prompts)
    raw = json.dumps(rows[0][0], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert prepared[0].last_evaluation['candidate_digest'] == hashlib.sha256(raw).hexdigest()
    assert not list(scratch.iterdir())


def test_prepare_and_composed_preflight_are_pure(selected):
    operations, request, ports, _, clients, _, _, _, runtimes, _ = selected
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    validator = ports.environment_sessions.prepare(plan, capture, lambda: None)
    assert validator.catalog == {} and not runtimes
    assert operations.preflight(plan).ready
    assert not runtimes and clients.calls == 0


def update_host(host, **policy_updates):
    policy = json.loads(host.raw['policy'])
    policy.update(policy_updates)
    host.raw['policy'] = json.dumps(policy).encode()
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(host.raw[item.ref]).hexdigest(),
        size_bytes=len(host.raw[item.ref])) for item in host.dependencies)


def test_explicit_simulation_never_invokes_inferred_write(selected):
    operations, request, ports, host, _, scenario, _, _, _, _ = selected
    host.selection = replace(host.selection, mode=EnvironmentMode.SIMULATE)
    update_host(host, mode='simulate')
    data = yaml.safe_load(scenario.read_text())
    del data['scenarios']['selected']['environment_execution']
    scenario.write_text(yaml.safe_dump(data))
    request = replace(request, environment_selection=host.selection)
    plan = operations.plan(request)
    validator = ports.environment_sessions.prepare(plan, operations._verify_plan(plan), lambda: None)
    result = validator.validate_response('', response('files_store', path='result.txt', content='fixed'), {'fixture': {'files': {}}})
    assert result.passed and result.to_dict()['mode'] == 'simulate'
    assert result.snapshot['files'] == [] and result.executed_tools[0].output == 'simulated'


def test_denied_action_and_empty_strict_schema_never_execute(selected):
    operations, request, ports, host, _, _, _, _, _, _ = selected
    # Keep schema/action interpretation valid while narrowing the trusted policy.
    host.resolve = lambda selection, project_ref: replace(
        type(host).resolve(host, selection, project_ref=project_ref), allowed_actions=('read',))
    update_host(host, allowed_actions=['read'])
    plan = operations.plan(request)
    validator = ports.environment_sessions.prepare(plan, operations._verify_plan(plan), lambda: None)
    result = validator.validate_response('', response('files_store', path='result.txt', content='fixed'), {'fixture': {'files': {}}})
    assert not result.passed and result.snapshot['files'] == []
    assert result.issues[0].code == 'action_denied'
    validator.tool_schema = {'tools': {}}
    result = validator.validate_response('', response('files_fetch', path='result.txt'), {'fixture': {'files': {}}})
    assert not result.passed and result.executed_tools[0].status == 'error'
    assert result.issues[0].code == 'tool_not_in_schema'


def test_cleanup_failure_preserves_pending_ownership_and_blocks_execution(selected):
    operations, request, ports, _, _, _, _, _, _, _ = selected
    plan = operations.plan(request)
    validator = ports.environment_sessions.prepare(plan, operations._verify_plan(plan), lambda: None)
    session = validator.start_session('', {'fixture': {'files': {}}})
    runtime = session.runtime.runtime
    original = runtime.teardown
    runtime.teardown = lambda: (_ for _ in ()).throw(WorkerLifecycleError('worker_cleanup_uncertain'))
    with pytest.raises(WorkerLifecycleError) as caught:
        session.close()
    assert caught.value.code == 'worker_cleanup_uncertain'
    assert not session.closed and validator._pending_session is session
    with pytest.raises(DataOperationError, match='environment_unavailable'):
        session.execute_response(response('files_store', path='result.txt', content='fixed'))
    with pytest.raises(DataOperationError, match='environment_unavailable'):
        validator.start_session('', {})
    runtime.teardown = original
    session.close()
    assert session.closed and validator._pending_session is None


def test_private_runtime_error_text_never_reaches_candidate_feedback(selected):
    operations, request, ports, _, _, _, environment, _, _, _ = selected
    plan = operations.plan(request)
    validator = ports.environment_sessions.prepare(plan, operations._verify_plan(plan), lambda: None)
    real_factory = validator.runtime_factory
    def failing_read(**kwargs):
        runtime = real_factory(**kwargs)
        runtime.read_text = lambda path: (_ for _ in ()).throw(OSError('private /credential/location'))
        return runtime
    validator.runtime_factory = failing_read
    candidate = {'conversations': [{'role': 'system', 'content': ''}, response('files_fetch', path='result.txt')]}
    feedback = validator.candidate_evaluator(environment)(candidate, 'a' * 64, {})
    assert 'private' not in json.dumps(feedback) and '/credential' not in json.dumps(feedback)
    assert any(issue['code'] == 'assertion_read_failed' for issue in feedback['environment_result']['issues'])


def test_workflow_deadline_is_not_renewed_for_new_candidates(selected, monkeypatch):
    operations, request, ports, _, _, _, _, _, runtimes, _ = selected
    plan = operations.plan(request)
    validator = ports.environment_sessions.prepare(plan, operations._verify_plan(plan), lambda: None)
    deadline = validator._deadline
    monkeypatch.setattr('synaptic_tuner.api.v1.reference.environment_sessions.time.monotonic', lambda: deadline)
    with pytest.raises(TimeoutError):
        validator.start_session('', {})
    assert not runtimes


@pytest.mark.parametrize('field,value', [('tool_action_hints', {'files_fetch': 'unknown_action'}),
                                       ('verb_rules', {'unknown_action': ['fetch']})])
def test_malformed_execution_policy_rejects_before_runtime_or_client(selected, field, value):
    operations, request, ports, host, clients, _, _, _, runtimes, _ = selected
    policy = json.loads(host.raw['policy'])
    policy['execution'][field] = value
    update_host(host, execution=policy['execution'])
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    with pytest.raises(DataOperationError, match='environment_invalid'):
        ports.environment_sessions.prepare(plan, capture, lambda: None)
    assert not runtimes and clients.calls == 0


def test_bound_cli_catalog_expands_wrapper_without_defaults(selected, monkeypatch):
    operations, request, ports, host, _, _, _, _, _, _ = selected
    catalog = {'tools': [{'agent': 'files', 'tool': 'store', 'command': 'fixture store', 'arguments': [
        {'name': 'path', 'type': 'string', 'positional': True, 'required': True},
        {'name': 'content', 'type': 'string', 'positional': True, 'required': True}]}]}
    update_host(host, cli_catalog=catalog)
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    # Both caller mutation and a changed host dependency must not replace capture.
    catalog['tools'].clear()
    host.raw['policy'] = b'changed after capture'
    forbidden_defaults(monkeypatch)
    validator = ports.environment_sessions.prepare(plan, capture, lambda: None)
    wrapped = response('useTools', sessionId='bound', workspaceId='bound', memory='bound', goal='bound',
                       tool='fixture store result.txt fixed')
    result = validator.validate_response('', wrapped, {'fixture': {'files': {}},
        'assertions': [{'type': 'file_contains', 'path': 'result.txt', 'text': 'fixed'}]})
    assert result.passed and result.executed_tools[0].name == 'files_store'
    assert result.executed_tools[0].arguments == {'path': 'result.txt', 'content': 'fixed'}


@pytest.mark.parametrize('interrupt', ['fatal_transport', 'cancel'])
def test_generation_propagates_lifecycle_interruption_without_model_retry_or_output(selected, interrupt):
    from synaptic_tuner.api.v1.reference.data import _Cancelled

    operations, request, ports, _, _, _, _, scratch, runtimes, prepared = selected
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    factory = ports.environment_sessions
    original = factory.runtime_factory
    cancelled = False
    def boundary():
        if cancelled:
            raise _Cancelled()
    def interrupted_runtime(**kwargs):
        runtime = original(**kwargs)
        def interrupted_read(path):
            nonlocal cancelled
            if interrupt == 'cancel':
                cancelled = True
                boundary()
            raise WorkerLifecycleError('worker_protocol_invalid')
        runtime.read_text = interrupted_read
        return runtime
    factory.runtime_factory = interrupted_runtime
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt'))], [])
    rows = []
    expected = _Cancelled if interrupt == 'cancel' else WorkerLifecycleError
    with pytest.raises(expected):
        operations._generate(plan, MeteredLLMClient(inner, UsageAccumulator()), None,
            lambda *args: rows.append(args), boundary, lambda *args: None, admitted_inputs=capture)
    assert not rows and inner.chat_calls == 3 and inner.judge_calls == 0
    assert len(runtimes) == 1 and prepared[0]._pending_session is None
    assert not list(scratch.iterdir())


def public_client(selected, inner, *, on_create=None):
    operations, _, ports, _, _, _, _, _, runtimes, _ = selected
    created = []
    class Factory:
        def create(self, **kwargs):
            assert not runtimes, 'No fixture may open before client construction and run claim'
            created.append(kwargs)
            if on_create is not None:
                on_create()
            return inner
    operations._ports = replace(ports, llm_clients=Factory())
    return DataAPI(operations), created


def test_public_generation_claims_before_effects_and_never_replays_after_recomposition(selected):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1, data_run_ref

    operations, request, ports, _, _, scenario, _, scratch, runtimes, _ = selected
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt')),
                           json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    def claimed_and_delete_sources():
        head = operations._require_head(data_run_ref(plan))
        assert head.result.state is DataRunState.PLANNED
        scenario.unlink()
        shutil.rmtree(ports.config_dir)
        shutil.rmtree(ports.rubrics_dir)
    api, created = public_client(selected, inner, on_create=claimed_and_delete_sources)
    plan = api.plan(request)
    assert api.preflight(plan).ready
    start = api.start(plan)
    outcome = api.show(start.run)
    assert outcome.state is DataRunState.SUCCEEDED and outcome.rows_written == 1
    assert len(created) == 1 and inner.chat_calls == 4 and not list(scratch.iterdir())
    assert len(runtimes) == 3 and not operations._environment_owners
    cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                     clock=operations._clock, ports=operations._ports)
    assert DataAPI(cold).show(start.run) == outcome
    assert DataAPI(cold).start(plan).run == start.run
    assert len(created) == 1 and inner.chat_calls == 4


def test_public_improvement_executes_revised_candidate(selected, tmp_path):
    operations, request, _, host, _, _, environment, scratch, runtimes, _ = selected
    policy = json.loads(host.raw['policy'])
    policy['execution']['candidate_projection']['environment'] = ['environment']
    update_host(host, execution=policy['execution'])
    dataset = tmp_path / 'public-seed.jsonl'
    dataset.write_text(json.dumps({'conversations': [{'role': 'system', 'content': 'bound'},
        response('files_fetch', path='result.txt')], 'environment': environment}) + '\n')
    request = replace(request, mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(dataset), rubric_refs=('repair',))
    inner = ScriptedClient([json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    api, _ = public_client(selected, inner, on_create=dataset.unlink)
    plan = api.plan(request)
    start = api.start(plan)
    outcome = api.show(start.run)
    assert outcome.state is DataRunState.SUCCEEDED and outcome.rows_written == 1
    assert len(runtimes) == 2 and not list(scratch.iterdir())
    assert not operations._environment_owners


def test_public_cancel_during_client_creation_is_not_overwritten(selected):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1, data_run_ref

    operations, request, _, _, _, _, _, scratch, runtimes, _ = selected
    inner = ScriptedClient([], [])
    def cancel_from_cold_host():
        cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                         clock=operations._clock, ports=operations._ports)
        assert DataAPI(cold).cancel(data_run_ref(plan), 'stop').state is DataRunState.CANCEL_REQUESTED
    api, created = public_client(selected, inner, on_create=cancel_from_cold_host)
    plan = api.plan(request)
    outcome = api.show(api.start(plan).run)
    assert outcome.state is DataRunState.CANCELLED and outcome.rows_written == 0
    assert len(created) == 1 and inner.chat_calls == 0 and not runtimes
    assert not list(scratch.iterdir())


def test_public_cancel_after_chat_prevents_next_model_request_and_row(selected):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1, data_run_ref

    operations, request, _, _, _, _, _, _, runtimes, _ = selected
    inner = ScriptedClient(['bound system'], [])
    original = inner.chat
    def cancelling_chat(*args, **kwargs):
        result = original(*args, **kwargs)
        cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                         clock=operations._clock, ports=operations._ports)
        DataAPI(cold).cancel(data_run_ref(plan), 'stop after this request')
        return result
    inner.chat = cancelling_chat
    api, _ = public_client(selected, inner)
    plan = api.plan(request)
    outcome = api.show(api.start(plan).run)
    assert outcome.state is DataRunState.CANCELLED and outcome.rows_written == 0
    assert inner.chat_calls == 1 and inner.judge_calls == 0 and not runtimes


def test_public_cleanup_uncertainty_retains_owner_and_no_restart_replay(selected):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1

    operations, request, ports, _, _, _, _, _, runtimes, _ = selected
    original = ports.environment_sessions.runtime_factory
    def uncertain_cleanup(**kwargs):
        runtime = original(**kwargs)
        runtime.original_teardown = runtime.teardown
        runtime.teardown = lambda: (_ for _ in ()).throw(WorkerLifecycleError('worker_cleanup_uncertain'))
        return runtime
    ports.environment_sessions.runtime_factory = uncertain_cleanup
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt'))], [])
    api, created = public_client(selected, inner)
    plan = api.plan(request)
    try:
        start = api.start(plan)
        outcome = api.show(start.run)
        assert outcome.state is DataRunState.RUNNING and outcome.diagnostic_code == 'worker_cleanup_uncertain'
        assert outcome.rows_written == 0 and start.run.run_id in operations._environment_owners
        cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                         clock=operations._clock, ports=operations._ports)
        assert DataAPI(cold).start(plan).run == start.run
        assert len(created) == 1 and inner.chat_calls == 3 and len(runtimes) == 1
        assert DataAPI(cold).cancel(start.run, 'stop unknown attempt').state is DataRunState.CANCEL_REQUESTED
        assert api.show(start.run).diagnostic_code == 'worker_cleanup_uncertain'
    finally:
        # Explicit cleanup of exact test-owned instances, never replay.
        for runtime in runtimes:
            runtime.teardown = runtime.original_teardown
            runtime.teardown()
        for owner in operations._environment_owners.values():
            owner.close()


def test_cancel_landing_before_final_transition_is_not_overwritten(selected, monkeypatch):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1

    operations, request, _, _, _, _, _, _, _, _ = selected
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt')),
                           json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    api, _ = public_client(selected, inner)
    original = operations._transition
    def late_cancel(run, mutate, **kwargs):
        if mutate.__name__ == 'finalize':
            cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                             clock=operations._clock, ports=operations._ports)
            DataAPI(cold).cancel(run, 'cancel before final head update')
        return original(run, mutate, **kwargs)
    monkeypatch.setattr(operations, '_transition', late_cancel)
    plan = api.plan(request)
    outcome = api.show(api.start(plan).run)
    assert outcome.state is DataRunState.CANCELLED and outcome.rows_written == 1


def test_failed_final_cas_remains_claimed_and_never_reexecutes(selected, monkeypatch):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1, data_run_ref

    operations, request, _, _, _, _, _, _, _, _ = selected
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt')),
                           json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    api, created = public_client(selected, inner)
    store = operations._records
    original = type(store).compare_and_swap
    def fail_final(self, **kwargs):
        document = json.loads(kwargs['canonical'])
        if self is store and document.get('result', {}).get('state') == 'succeeded':
            return False
        return original(self, **kwargs)
    monkeypatch.setattr(type(store), 'compare_and_swap', fail_final)
    plan = api.plan(request)
    with pytest.raises(DataOperationError, match='state_conflict'):
        api.start(plan)
    run = data_run_ref(plan)
    assert api.show(run).state is DataRunState.RUNNING
    cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                     clock=operations._clock, ports=operations._ports)
    assert DataAPI(cold).start(plan).run == run and len(created) == 1 and inner.chat_calls == 4
    # The orphan history slot cannot be guessed/adopted as cancellation or replay.
    with pytest.raises(DataOperationError, match='state_conflict'):
        DataAPI(cold).cancel(run, 'cancel uncertain persistence')


def test_boundary_integrity_failure_is_not_reported_as_deadline(selected):
    from synaptic_tuner.api.v1.data_facade import DataOperationCode

    operations, request, ports, _, _, _, _, _, runtimes, _ = selected
    inner = ScriptedClient(['bound system'], [])
    original = ports.environment_sessions.prepare
    def prepare(*args):
        validator = original(*args)
        boundary = validator.check_boundary
        def interrupted():
            boundary()
            if inner.chat_calls:
                raise DataOperationError(DataOperationCode.INTEGRITY_ERROR)
        validator.check_boundary = interrupted
        return validator
    ports.environment_sessions.prepare = prepare
    api, _ = public_client(selected, inner)
    outcome = api.show(api.start(api.plan(request)).run)
    assert outcome.state is DataRunState.FAILED and outcome.diagnostic_code == 'integrity_error'
    assert inner.chat_calls == 1 and inner.judge_calls == 0 and not runtimes


@pytest.mark.parametrize('code', ['worker_start_uncertain', 'worker_termination_uncertain', 'worker_cleanup_uncertain'])
def test_unknown_worker_signal_stays_nonterminal_without_restart_replay(selected, code):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1

    operations, request, ports, _, _, _, _, scratch, _, _ = selected
    original = ports.environment_sessions.runtime_factory
    def unknown_runtime(**kwargs):
        runtime = original(**kwargs)
        runtime.read_text = lambda path: (_ for _ in ()).throw(WorkerLifecycleError(code))
        return runtime
    ports.environment_sessions.runtime_factory = unknown_runtime
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt'))], [])
    api, created = public_client(selected, inner)
    plan = api.plan(request)
    start = api.start(plan)
    outcome = api.show(start.run)
    assert outcome.state is DataRunState.RUNNING and outcome.diagnostic_code == code
    assert not list(scratch.iterdir())
    cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                     clock=operations._clock, ports=operations._ports)
    assert DataAPI(cold).start(plan).run == start.run and len(created) == 1
    assert inner.chat_calls == 3 and inner.judge_calls == 0


def test_public_cancel_reaps_stalled_owned_child_from_parent_poll(selected, request):
    from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1, data_run_ref

    if request.node.callspec.params['selected'] != 'worker':
        pytest.skip('Actual process cancellation applies to the supervised worker')
    operations, data_request, ports, _, _, _, _, scratch, runtimes, _ = selected
    factory = ports.environment_sessions
    original_prepare, original_runtime = factory.prepare, factory.runtime_factory
    armed = cancelled = False
    stopped_processes = []
    def prepare(plan, capture, boundary):
        def parent_poll():
            nonlocal cancelled
            if armed and not cancelled:
                cold = ReferenceDataOperationsV1(records=operations._records, streams=operations._streams,
                                                 clock=operations._clock, ports=operations._ports)
                DataAPI(cold).cancel(data_run_ref(plan), 'cancel stopped owned worker')
                cancelled = True
            boundary()
        return original_prepare(plan, capture, parent_poll)
    def runtime_factory(**kwargs):
        runtime = original_runtime(**kwargs)
        original_read = runtime.read_text
        def stalled_read(path):
            nonlocal armed
            process = runtime._process
            os.kill(process.pid, signal.SIGSTOP)
            stopped_processes.append(process)
            armed = True
            return original_read(path)
        runtime.read_text = stalled_read
        return runtime
    factory.prepare, factory.runtime_factory = prepare, runtime_factory
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt'))], [])
    api, _ = public_client(selected, inner)
    try:
        plan = api.plan(data_request)
        outcome = api.show(api.start(plan).run)
        assert outcome.state is DataRunState.CANCELLED and outcome.rows_written == 0
        assert armed and cancelled and stopped_processes
        assert all(process.poll() is not None for process in stopped_processes)
        assert inner.chat_calls == 3 and inner.judge_calls == 0 and not list(scratch.iterdir())
        assert not operations._environment_owners
    finally:
        for runtime in runtimes:
            runtime.teardown()


def test_uncertainty_survives_later_artifact_failure(selected, monkeypatch):
    from synaptic_tuner.api.v1.reference import data as data_module

    operations, request, ports, _, _, _, _, scratch, runtimes, _ = selected
    original_prepare = ports.environment_sessions.prepare
    def prepare(*args, **kwargs):
        validator = original_prepare(*args, **kwargs)
        validator.close = lambda: (_ for _ in ()).throw(WorkerLifecycleError('worker_cleanup_uncertain'))
        return validator
    monkeypatch.setattr(ports.environment_sessions, 'prepare', prepare)
    monkeypatch.setattr(data_module, '_verified_artifact',
                        lambda *args: (_ for _ in ()).throw(OSError('private artifact failure')))
    inner = ScriptedClient(['bound system', 'bound user', json.dumps(response('files_fetch', path='result.txt')),
                           json.dumps(response('files_store', path='result.txt', content='fixed'))], [.1, .95, .95])
    api, _ = public_client(selected, inner)
    plan = api.plan(request)
    try:
        outcome = api.show(api.start(plan).run)
        assert outcome.rows_written == 1 and not outcome.artifacts
        assert outcome.state is DataRunState.RUNNING
        assert outcome.diagnostic_code == 'worker_cleanup_uncertain'
        assert not list(scratch.iterdir())
    finally:
        for runtime in runtimes:
            runtime.teardown()
        operations._environment_owners.clear()


def test_guarded_data_proxy_preserves_completion_identity_usage_and_pre_dispatch_boundary(selected, monkeypatch):
    from shared.llm.usage import LLMCompletionV1, LLMStructuredV1, measured_usage
    from synaptic_tuner.api.v1.reference.data import _EnvironmentInterrupted

    operations, request, _, _, _, _, _, _, _, _ = selected
    usage = measured_usage(2, 3)
    chat_result = LLMCompletionV1('reply', usage)
    structured_result = LLMStructuredV1({'value': 'reply'}, usage)
    inner = ScriptedClient([], [])
    dispatched = []
    inner.chat = lambda *a, **k: dispatched.append('chat') or chat_result
    inner.structured_output = lambda *a, **k: dispatched.append('structured') or structured_result
    api, _ = public_client(selected, inner)
    def execute(plan, run, client, meter, scope, *, admitted_inputs, environment_validator):
        try:
            assert client.chat([]) is chat_result
            assert client.structured_output([], {}) is structured_result
            assert meter.calls == 2 and meter.measured and not meter.indeterminate
            record = meter.record()
            assert record.input_tokens == 4 and record.output_tokens == 6
            def cancelled():
                raise TimeoutError('boundary')
            environment_validator.check_boundary = cancelled
            for call in (lambda: client.chat([]), lambda: client.structured_output([], {})):
                with pytest.raises(_EnvironmentInterrupted):
                    call()
            assert dispatched == ['chat', 'structured']
            assert meter.calls == 2 and not meter.indeterminate
        finally:
            environment_validator.close()
            operations._environment_owners.pop(run.run_id, None)
    monkeypatch.setattr(operations, '_execute', execute)
    api.start(api.plan(request))
