"""Generic feedback follows exact candidates without changing legacy execution."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from SynthChat.engine import ImprovementEngine


def digest(candidate):
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def engine(candidates=(), scores=(1,), mutate_context=False):
    instance = ImprovementEngine.__new__(ImprovementEngine)
    rubric = {'key': 'quality', 'name': 'quality', 'scope': 'response', 'pass_threshold': .8}
    instance.rubric_repo = SimpleNamespace(get_rubric=lambda key: rubric)
    instance.scope_config = SimpleNamespace(scope_processing_order=['response'],
        judge=SimpleNamespace(score_field_suffix='_score'),
        validation=SimpleNamespace(schema_score_suffix='_schema'))
    instance.logger = Mock()
    instance.interaction_logger = Mock()
    instance.validation_service = SimpleNamespace(validate_example=lambda *args: {})
    instance.judge_service = Mock()
    instance.judge_service.judge.side_effect = [{'quality_score': score} for score in scores]
    instance._call_with_retry = lambda callback, name: callback()
    prompts, improvements = [], []
    replacements = iter(candidates)
    def build(candidate, rubrics, validation, prompt_context=None):
        prompts.append((deepcopy(candidate), deepcopy(prompt_context)))
        if mutate_context and prompt_context is not None:
            prompt_context['nested']['values'].append('judge mutation')
        return 'system', 'judge prompt'
    def improve(candidate, scope, rubrics, judgment, prompt_context=None):
        improvements.append(deepcopy(prompt_context))
        if mutate_context and prompt_context is not None:
            prompt_context['nested']['values'].append('improver mutation')
        return deepcopy(next(replacements, candidate)), 'improve prompt', 'response', 'system'
    instance._build_judge_prompt = build
    instance._improve_scope_with_logging = improve
    return instance, prompts, improvements


def evaluator(calls, *, mutate=False):
    def evaluate(candidate, candidate_digest, base_context):
        calls.append((deepcopy(candidate), candidate_digest, deepcopy(base_context)))
        if mutate:
            candidate['value'] = 'callback mutation'
            base_context['nested']['values'].append('callback mutation')
        return {'feedback': calls[-1][0]['value'], 'nested': {'values': []}}
    return evaluate


def test_initial_pass_uses_current_feedback_and_no_legacy_overlay():
    instance, prompts, _ = engine()
    candidate, base, calls = {'value': 'initial'}, {'stale': 'old feedback'}, []
    result = instance.run(candidate, ['quality'], prompt_context=base, candidate_evaluator=evaluator(calls))
    assert result.passed and result.improved_example == candidate
    assert len(calls) == 1
    assert prompts[0][1] == {'feedback': 'initial', 'nested': {'values': []}, 'candidate_digest': digest(candidate)}
    assert base == {'stale': 'old feedback'}


def test_changed_candidate_evaluated_before_rejudge_and_selected_at_limit():
    initial, changed, calls = {'value': 'initial'}, {'value': 'changed'}, []
    instance, prompts, improvements = engine([changed], scores=(0, 1))
    result = instance.run(initial, ['quality'], max_iterations=1, candidate_evaluator=evaluator(calls))
    assert result.passed and result.improved_example == changed
    assert [entry[0] for entry in calls] == [initial, changed]
    assert [context['feedback'] for _, context in prompts] == ['initial', 'changed']
    assert prompts[-1][1]['candidate_digest'] == digest(result.improved_example)
    assert improvements[0]['candidate_digest'] == digest(initial)


def test_unchanged_candidate_reuses_only_current_run_feedback():
    candidate, calls = {'value': 'unchanged'}, []
    instance, prompts, _ = engine(scores=(0, 0))
    result = instance.run(candidate, ['quality'], max_iterations=2, candidate_evaluator=evaluator(calls))
    assert not result.passed and len(calls) == 1 and len(prompts) == 2
    instance, _, _ = engine()
    instance.run(candidate, ['quality'], candidate_evaluator=evaluator(calls))
    assert len(calls) == 2  # No cross-run cache.


@pytest.mark.parametrize('stage', ['initial', 'changed'])
def test_callback_failure_propagates_without_quality_result_or_retry(stage):
    initial, changed = {'value': 'initial'}, {'value': 'changed'}
    instance, prompts, _ = engine([changed], scores=(0, 1))
    error, calls = RuntimeError('callback failed'), []
    def evaluate(candidate, candidate_digest, base_context):
        calls.append(candidate)
        if stage == 'initial' or candidate == changed: raise error
        return {'feedback': 'initial'}
    with pytest.raises(RuntimeError) as caught:
        instance.run(initial, ['quality'], candidate_evaluator=evaluate)
    assert caught.value is error
    assert len(prompts) == (0 if stage == 'initial' else 1)
    assert len(calls) == (1 if stage == 'initial' else 2)


def test_callback_and_downstream_mutations_do_not_change_candidate_or_base():
    initial, changed, base = {'value': 'initial'}, {'value': 'changed'}, {'nested': {'values': []}}
    instance, prompts, improvements = engine([changed], scores=(0, 1), mutate_context=True)
    calls = []
    result = instance.run(initial, ['quality'], max_iterations=1, prompt_context=base,
        candidate_evaluator=evaluator(calls, mutate=True))
    assert initial == {'value': 'initial'} and result.improved_example == changed
    assert base == {'nested': {'values': []}}
    assert all(entry[2] == base for entry in calls)
    assert all(context['nested']['values'] == [] for _, context in prompts)
    assert improvements[0]['nested']['values'] == []


@pytest.mark.parametrize('feedback', [None, [], {'candidate_digest': 'wrong'}])
def test_invalid_callback_result_cannot_fallback(feedback):
    instance, prompts, _ = engine()
    with pytest.raises((TypeError, ValueError)):
        instance.run({'value': 'candidate'}, ['quality'], candidate_evaluator=lambda *args: feedback)
    assert prompts == []


def test_legacy_absent_callback_preserves_context_and_call_counts():
    candidate, context = {'value': 'legacy'}, {'legacy': {'nested': []}}
    instance, prompts, _ = engine(scores=(1,))
    result = instance.run(candidate, ['quality'], prompt_context=context)
    assert result.improved_example is candidate
    assert result.__dict__ == {'improved_example': candidate, 'iterations': 1, 'passed': True,
        'final_scores': {'quality': 1}, 'scopes_improved': []}
    assert prompts == [(candidate, context)]
    assert instance.judge_service.judge.call_count == 1


def test_max_iterations_retains_final_failed_candidate_feedback():
    initial, first, final, calls = {'value': 'initial'}, {'value': 'first'}, {'value': 'final'}, []
    instance, prompts, _ = engine([first, final], scores=(0, 0, 0, 0))
    result = instance.run(initial, ['quality'], max_iterations=2, candidate_evaluator=evaluator(calls))
    assert not result.passed and result.improved_example == final
    assert [row[0] for row in calls] == [initial, first, final]
    assert prompts[-1][1]['candidate_digest'] == digest(final)


class CustomDict(dict):
    pass


class CustomList(list):
    pass


@pytest.mark.parametrize('candidate', [
    {1: 'coerced key'}, {'value': (1, 2)}, CustomDict(value='custom'),
    {'value': CustomList([1])}, {'value': float('nan')}, {'value': float('inf')},
])
def test_non_json_candidates_reject_before_callback_or_judgment(candidate):
    instance, prompts, _ = engine()
    callback = Mock(return_value={})
    with pytest.raises((TypeError, ValueError)):
        instance.run(candidate, ['quality'], candidate_evaluator=callback)
    callback.assert_not_called()
    assert prompts == []


def test_cycle_rejects_before_callback():
    candidate = {'value': []}
    candidate['value'].append(candidate)
    instance, _, _ = engine()
    callback = Mock(return_value={})
    with pytest.raises(ValueError, match='cycle'):
        instance.run(candidate, ['quality'], candidate_evaluator=callback)
    callback.assert_not_called()


def test_fractional_json_candidate_remains_supported():
    candidate, calls = {'value': .125}, []
    instance, _, _ = engine()
    result = instance.run(candidate, ['quality'], candidate_evaluator=evaluator(calls))
    assert result.passed and calls[0][1] == digest(candidate)


def test_non_json_legacy_candidate_is_not_newly_rejected():
    candidate = {'value': ('legacy',)}
    instance, _, _ = engine()
    assert instance.run(candidate, ['quality']).improved_example is candidate


def test_multiple_scopes_reuse_current_feedback_and_refresh_changed_candidate():
    initial, changed, calls = {'value': 'initial'}, {'value': 'changed'}, []
    instance, prompts, _ = engine([changed])
    scopes = ['first', 'second', 'third']
    instance.scope_config.scope_processing_order = scopes
    instance.rubric_repo = SimpleNamespace(get_rubric=lambda key: {
        'key': key, 'name': key, 'scope': key, 'pass_threshold': .8})
    instance.judge_service.judge.side_effect = [
        {'first_score': 1}, {'second_score': 0}, {'second_score': 1}, {'third_score': 1}]
    result = instance.run(initial, scopes, max_iterations=1, candidate_evaluator=evaluator(calls))
    assert result.passed and result.improved_example == changed
    assert [entry[0] for entry in calls] == [initial, changed]
    assert [context['feedback'] for _, context in prompts] == ['initial', 'initial', 'changed', 'changed']
    assert prompts[-1][1]['candidate_digest'] == digest(result.improved_example)
