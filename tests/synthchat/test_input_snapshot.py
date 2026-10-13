"""Explicit admitted documents are consumed without reopening source paths."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest
import yaml

from SynthChat.config.config_loader import ConfigLoader
from SynthChat.config.format_resolver import load_tool_call_formats, load_workspace_formats, load_label_mappings
from SynthChat.config.privacy import load_privacy_profiles
from SynthChat.engine import ImprovementEngine
from SynthChat.generator import ScenarioLoader, SynthChatGenerator
from SynthChat.input_snapshot import WorkflowInputSnapshot
from shared.validation.rubric import RubricRepository

ROOT = Path(__file__).resolve().parents[2] / 'SynthChat'
SELECTED = {'prompts': {'assistant_system': 'bound'}, 'assistant_generation': {}}
RUBRIC = {'name': 'selected', 'scope': 'response', 'pass_threshold': .8}


def snapshot(*, validation=True, privacy=True, selected=None, extra_scenario=None):
    documents = []
    if validation:
        documents.append(('validation', 'config:validation', (ROOT / 'config/validation.yaml').read_bytes()))
    if privacy:
        documents.append(('privacy', 'config:privacy_profiles', (ROOT / 'config/privacy_profiles.yaml').read_bytes()))
    for name in ('tool_call_formats', 'workspace_formats', 'label_mappings'):
        documents.append(('registry', 'registry:' + name, (ROOT / 'config' / (name + '.yaml')).read_bytes()))
    scenarios = {'selected': deepcopy(SELECTED if selected is None else selected)}
    if extra_scenario is not None: scenarios['unselected'] = extra_scenario
    documents.append(('scenario', 'scenario:source-file', yaml.safe_dump({'scenarios': scenarios}).encode()))
    documents.append(('rubric', 'rubric:selected', yaml.safe_dump(RUBRIC).encode()))
    return WorkflowInputSnapshot(tuple(documents))


def generator(inputs, *, client=None, validation=True, engine=None, **kwargs):
    return SynthChatGenerator(Path('/absent/config'), Path('/absent/scenarios'), Path('/absent/rubrics'),
        client or Mock(provider_name='host'), engine=engine, enable_stage_validation=validation,
        input_snapshot=inputs, scenario_refs=('selected',), **kwargs)


def test_parsers_match_path_mode_and_selected_ids_only(tmp_path):
    inputs = snapshot(extra_scenario={'system': {'template': '/forbidden/unselected'}})
    client = Mock(provider_name='host')
    current = generator(inputs, client=client)
    assert current.scenario_loader.list_scenarios() == ['selected']
    assert current.scenario_loader.get_scenario('selected') == SELECTED
    assert current.scenario_loader.get_scenario('unselected') is None
    assert current._tool_call_formats == load_tool_call_formats()
    assert current._workspace_formats == load_workspace_formats()
    assert current._label_mappings == load_label_mappings()
    assert current._privacy_profiles == load_privacy_profiles()
    assert current.engine.scope_config == ConfigLoader(ROOT / 'config/validation.yaml').load()
    rubric_dir = tmp_path / 'rubrics'; rubric_dir.mkdir()
    (rubric_dir / 'selected.yaml').write_text(yaml.safe_dump(RUBRIC))
    assert current.engine.rubric_repo.get_rubric('selected') == RubricRepository(rubric_dir).get_rubric('selected')
    assert current.engine.llm_client is client
    assert type(current.engine.rubric_repo) is RubricRepository


def test_source_mutation_deletion_and_no_constructor_input_or_output_io(tmp_path, monkeypatch):
    original = snapshot()
    source = tmp_path / 'source'; source.mkdir()
    documents = []
    for index, (role, ref, raw) in enumerate(original.documents):
        path = source / str(index); path.write_bytes(raw)
        documents.append((role, ref, path.read_bytes()))
    inputs = WorkflowInputSnapshot(tuple(documents))
    for path in source.iterdir(): path.write_text('changed')
    shutil.rmtree(source)
    def forbidden(*args, **kwargs): raise AssertionError('snapshot constructor attempted path I/O')
    for method in ('open', 'glob', 'mkdir', 'read_text', 'read_bytes'):
        monkeypatch.setattr(Path, method, forbidden)
    current = generator(inputs)
    assert current.scenario_loader.get_scenario('selected') == SELECTED
    assert current.engine.rubric_repo.get_rubric('selected')['name'] == 'selected'
    assert current.engine.interaction_logger.enabled is False
    explicit = Mock()
    engine = ImprovementEngine(Mock(), Path('/unused'), input_snapshot=inputs, interaction_logger=explicit)
    assert engine.interaction_logger is explicit


def test_independent_snapshots_and_fresh_copies():
    first = snapshot(); second = snapshot(selected={'prompts': {'assistant_system': 'other'}})
    one, two = generator(first), generator(second)
    one.scenario_loader.get_scenario('selected')['prompts']['assistant_system'] = 'mutated'
    one.engine.rubric_repo.get_rubric('selected')['scope'] = 'mutated'
    one._tool_call_formats['default']['wrapper_name'] = 'mutated'
    assert generator(first)._tool_call_formats == two._tool_call_formats
    assert first.document('scenario', 'scenario:source-file')['scenarios']['selected'] == SELECTED
    assert two.scenario_loader.get_scenario('selected')['prompts']['assistant_system'] == 'other'
    with pytest.raises(FrozenInstanceError): first.documents = ()
    assert 'assistant_system' not in repr(first)


@pytest.mark.parametrize('field', ['registry:tool_call_formats', 'registry:workspace_formats', 'registry:label_mappings'])
def test_missing_registry_cannot_fallback(field):
    inputs = snapshot()
    inputs = replace(inputs, documents=tuple(item for item in inputs.documents if item[1] != field))
    with pytest.raises(ValueError, match='missing'): generator(inputs)


@pytest.mark.parametrize('raw', [b'{', b'{}', b'value: .nan', b'a: 1\na: 2', b'a: &a [*a]'])
def test_corrupt_documents_reject_at_snapshot_construction(raw):
    with pytest.raises(ValueError): WorkflowInputSnapshot((('scenario', 'scenario:bad', raw),))


def test_bounded_exact_document_interface():
    with pytest.raises(ValueError): WorkflowInputSnapshot([])
    with pytest.raises(ValueError): WorkflowInputSnapshot((('dataset', 'input:dataset', b'a: 1'),))
    with pytest.raises(ValueError): WorkflowInputSnapshot((('scenario', 'scenario:/absolute', b'a: 1'),))
    with pytest.raises(ValueError): WorkflowInputSnapshot((('scenario', 'scenario:a', b'a: 1'),) * 2)
    with pytest.raises(ValueError): WorkflowInputSnapshot((('scenario', 'scenario:a', bytearray(b'a: 1')),))
    with pytest.raises(ValueError): WorkflowInputSnapshot((('scenario', 'scenario:a', b'x' * (4 * 1024 * 1024 + 1)),))


@pytest.mark.parametrize('refs', [None, (), ('selected', 'selected'), ('missing',), ['selected']])
def test_selection_must_be_exact_known_unique(refs):
    with pytest.raises(ValueError): snapshot().validate_generation(refs)


def test_max_one_generation_requires_no_unbound_validation():
    inputs = snapshot(validation=False)
    assert inputs.validate_generation(('selected',), max_iterations=1)['selected'] == SELECTED
    current = generator(inputs, validation=False)
    assert current.engine is None and current.enable_stage_validation is False
    current._validate_snapshot_selection('selected', SELECTED, max_iterations=1)
    with pytest.raises(ValueError): current._validate_snapshot_selection('selected', SELECTED, max_iterations=2)
    with pytest.raises(ValueError): generator(inputs, validation=True)


@pytest.mark.parametrize('raw', [b'ignored: true', b'scopes: {}\nscope_processing_order: []',
    b'scopes: {only: {}}\nscope_processing_order: [other]'])
def test_semantically_empty_validation_rejects_without_affecting_path_defaults(raw):
    inputs = snapshot()
    inputs = replace(inputs, documents=tuple((role, ref, raw if ref == 'config:validation' else data)
                                             for role, ref, data in inputs.documents))
    with pytest.raises(ValueError): inputs.validate_generation(('selected',), max_iterations=2)


def test_selected_rubric_scopes_must_be_declared():
    inputs = snapshot()
    inputs = replace(inputs, documents=tuple((role, ref, b'name: selected\nscope: unknown' if role == 'rubric' else raw)
                                             for role, ref, raw in inputs.documents))
    with pytest.raises(ValueError, match='scope'): inputs.validate_improvement()


@pytest.mark.parametrize('patch', [
    {'system': {'template': '/missing'}}, {'assistant_generation': {'model': 'raw'}},
    {'user_generation': {'fallback_models': ['raw']}}, {'system_generation': {'judge': {'provider': 'raw'}}},
    {'judge': {'in_loop': {'provider_routing': {'order': ['raw']}}}},
    {'tool_call_format': 'missing'}, {'workspace_format': 'missing'},
])
def test_unsupported_selected_configuration_rejects_purely_before_factory(patch, monkeypatch):
    factory = Mock(side_effect=AssertionError('raw factory called'))
    monkeypatch.setattr('SynthChat.generator.create_client', factory)
    inputs = snapshot(selected={**SELECTED, **patch})
    with pytest.raises(ValueError): inputs.validate_generation(('selected',))
    with pytest.raises(ValueError): generator(inputs)
    factory.assert_not_called()


def test_missing_or_unknown_selected_privacy_profiles_reject():
    with pytest.raises(ValueError): snapshot(privacy=False).validate_generation(('selected',),
        privacy_settings={'enabled': True, 'profile': 'mask_only'})
    with pytest.raises(ValueError): snapshot().validate_generation(('selected',),
        privacy_settings={'enabled': True, 'profile': 'missing'})


def test_supplied_engine_must_share_snapshot_and_injected_client():
    inputs, client = snapshot(), Mock(provider_name='host')
    engine = ImprovementEngine(client, Path('/unused'), input_snapshot=inputs)
    assert generator(inputs, client=client, engine=engine).engine is engine
    with pytest.raises(ValueError): generator(inputs, client=Mock(), engine=engine)
    with pytest.raises(ValueError): generator(snapshot(), client=client, engine=engine)


def test_injected_client_retries_never_open_raw_fallback(monkeypatch):
    inputs, client = snapshot(), Mock(provider_name='host')
    engine = ImprovementEngine(client, Path('/unused'), input_snapshot=inputs)
    factory = Mock(side_effect=AssertionError('raw factory called'))
    monkeypatch.setattr('shared.llm.create_client', factory)
    monkeypatch.setattr('SynthChat.generator.create_client', factory)
    monkeypatch.setattr('SynthChat.engine.MAX_RETRIES', 2)
    monkeypatch.setattr('SynthChat.engine.time.sleep', lambda _: None)
    error = RuntimeError('injected client failed')
    callback = Mock(side_effect=error)
    with pytest.raises(RuntimeError) as caught: engine._call_with_retry(callback, 'test')
    assert caught.value is error and callback.call_count == 2 and engine.llm_client is client
    current = generator(inputs, client=client, engine=engine)
    assert current._get_stage_llm_clients({}) == [client]
    with pytest.raises(ValueError): current._get_stage_llm_clients({'model': 'raw'})
    with pytest.raises(ValueError): current._get_or_create_llm_client({'provider': 'raw'})
    factory.assert_not_called()


def test_snapshot_rubrics_and_scope_config_never_reload_paths():
    current = generator(snapshot())
    with pytest.raises(ValueError): current.engine.rubric_repo.reload()
    config = ConfigLoader.from_mapping(snapshot().document('validation', 'config:validation'))
    with pytest.raises(ValueError): config.reload()


def test_none_keeps_legacy_missing_registry_fallbacks(tmp_path):
    assert load_tool_call_formats(str(tmp_path / 'missing'))['default']
    loader = ScenarioLoader(tmp_path)
    assert loader.list_scenarios() == []


def test_unselected_sibling_conflicts_do_not_expand_or_block_selection():
    original = snapshot(extra_scenario={'system': {'template': '/unselected'}})
    extra = ('scenario', 'scenario:another-file', yaml.safe_dump({'scenarios': {
        'unselected': {'assistant_generation': {'model': 'unselected'}}}}).encode())
    inputs = replace(original, documents=original.documents + (extra,))
    assert generator(inputs).scenario_loader.list_scenarios() == ['selected']
    ambiguous = ('scenario', 'scenario:duplicate-file', yaml.safe_dump({'scenarios': {'selected': SELECTED}}).encode())
    with pytest.raises(ValueError, match='conflict'):
        replace(original, documents=original.documents + (ambiguous,)).validate_generation(('selected',))


def test_external_scenario_mutation_cannot_replace_admitted_inputs():
    current = generator(snapshot(), validation=False)
    altered = deepcopy(SELECTED)
    altered['assistant_generation']['model'] = 'raw'
    with pytest.raises(ValueError): current._validate_snapshot_selection('selected', altered)
    current.scenario_loader.scenarios['selected'] = altered
    with pytest.raises(ValueError): current._validate_snapshot_selection('selected', altered)


def test_missing_privacy_profile_rejects_before_preprocessor_factory():
    factory = Mock(side_effect=AssertionError('privacy fallback used'))
    current = generator(snapshot(privacy=False), validation=False, privacy_preprocessor_factory=factory)
    with pytest.raises(ValueError): current._get_privacy_preprocessor('missing')
    factory.assert_not_called()


def test_missing_requested_rubric_rejects_before_client_factory(monkeypatch):
    factory = Mock(side_effect=AssertionError('raw factory called'))
    monkeypatch.setattr('SynthChat.generator.create_client', factory)
    inputs = snapshot(selected={**SELECTED, 'rubrics': {'response': ['selected', 'missing']}})
    with pytest.raises(ValueError, match='rubric'):
        inputs.validate_generation(('selected',), max_iterations=2)
    with pytest.raises(ValueError, match='rubric'):
        generator(inputs)
    with pytest.raises(ValueError, match='rubric'):
        snapshot().validate_improvement(('selected', 'missing'))
    client = Mock(provider_name='host')
    engine = ImprovementEngine(client, Path('/unused'), input_snapshot=snapshot())
    with pytest.raises(ValueError, match='rubric'):
        engine.run({'value': 'candidate'}, ['selected', 'missing'])
    assert client.mock_calls == []
    factory.assert_not_called()


@pytest.mark.parametrize('path', ['generate_single', 'prepare_seed_bundle'])
def test_admitted_unselected_sibling_cannot_expand_constructor_selection(path):
    sibling = {'prompts': {'assistant_system': 'unselected'}, 'assistant_generation': {}}
    client = Mock(provider_name='host')
    current = generator(snapshot(extra_scenario=sibling), validation=False, client=client)
    with pytest.raises(ValueError, match='outside the selected'):
        if path == 'generate_single':
            current.generate_single('unselected', sibling, max_iterations=1, randomize_params=False)
        else:
            current.prepare_seed_bundle(scenario_key='unselected', scenario=sibling, randomize_params=False)
    assert client.mock_calls == []


@pytest.mark.parametrize(('field', 'registry'), [('tool_call_format', 'tool_call_formats'),
                                               ('workspace_format', 'workspace_formats')])
def test_selected_empty_named_format_cannot_fallback_to_default(field, registry, monkeypatch):
    factory = Mock(side_effect=AssertionError('raw factory called'))
    monkeypatch.setattr('SynthChat.generator.create_client', factory)
    inputs = snapshot(selected={**SELECTED, field: 'empty'})
    data = inputs.document('registry', 'registry:' + registry)
    data['formats']['empty'] = {}
    inputs = replace(inputs, documents=tuple((role, ref, yaml.safe_dump(data).encode()
        if ref == 'registry:' + registry else raw) for role, ref, raw in inputs.documents))
    with pytest.raises(ValueError, match='format'): inputs.validate_generation(('selected',))
    with pytest.raises(ValueError, match='format'): generator(inputs)
    factory.assert_not_called()


def test_selected_empty_privacy_profile_cannot_fallback():
    inputs = snapshot()
    profile = yaml.safe_dump({'profiles': {'empty': {}}}).encode()
    inputs = replace(inputs, documents=tuple((role, ref, profile if role == 'privacy' else raw)
        for role, ref, raw in inputs.documents))
    with pytest.raises(ValueError, match='privacy'):
        inputs.validate_generation(('selected',), privacy_settings={'enabled': True, 'profile': 'empty'})


def test_generator_parsing_retains_independent_snapshot_wrapper_contexts(monkeypatch):
    import json

    def forbidden(*args, **kwargs):
        raise AssertionError('Snapshot parsing consulted ambient wrapper formats')
    monkeypatch.setattr('shared.validation.parsing.configured_formats.get_configured_wrapper_specs', forbidden)
    generators = []
    for wrapper, field in (('dispatch_alpha', 'tenant'), ('dispatch_beta', 'account')):
        inputs = snapshot(validation=False)
        registry = {'formats': {'default': {'wrapper_name': wrapper,
            'argument_fields': {'required': [field, 'command'], 'properties': {
                field: {'type': 'string'}, 'command': {'type': 'string'}}}, 'command_field': 'command'}}}
        inputs = replace(inputs, documents=tuple((role, ref, yaml.safe_dump(registry).encode()
            if ref == 'registry:tool_call_formats' else raw) for role, ref, raw in inputs.documents))
        current = generator(inputs, validation=False)
        registry['formats']['default']['wrapper_name'] = 'mutated'
        generators.append((current, wrapper, field))
    for current, wrapper, field in generators:
        content = json.dumps({'tool_calls': [{'function': {'name': wrapper,
            'arguments': {field: 'bound', 'command': 'configured command'}}}]})
        parsed = current._parse_assistant_response(content, SELECTED)
        assert json.loads(parsed['tool_calls'][0]['function']['arguments'])[field] == 'bound'
        assert current._wrapper_specs[0]['wrapper_name'] == wrapper
