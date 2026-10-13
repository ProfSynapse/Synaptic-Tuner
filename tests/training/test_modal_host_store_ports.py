"""Optional generic host-store injection rejects before named secrets/effects."""
import pytest

from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.execution.providers.modal.packaged_composition import compose_modal_packaged_adapter
from tuner.project.context import ProjectContext
from tuner.training.modal_host_composition import compose_modal_packaged_reference_host
from tests.execution.providers.test_modal_packaged_composition import _components
from tests.training.test_packaged_execution_material import packaged_fixture


class UTCClock:
    def now(self):
        return '2026-10-13T12:00:00Z'

    def now_epoch(self):
        from datetime import datetime, timezone
        return int(datetime(2026, 10, 13, 12, tzinfo=timezone.utc).timestamp())


class _Secrets:
    def resolve(self, ref):
        return 'host-only-test-authority-value-' + 'x' * 40


class _Effects:
    bindings = object()

    def authorize(self, requirements):
        pytest.fail('composition must not authorize paid effects')

    def bind(self, grant, *, operation, requirements):
        pytest.fail('composition must not bind paid effects')

    def authenticate(self, binding):
        pytest.fail('composition must not read retained bindings')


def arguments(tmp_path):
    _, adapter_arguments, events, _ = _components()
    adapter = compose_modal_packaged_adapter(**adapter_arguments)
    public, components = packaged_fixture()
    return dict(
        adapter=adapter, effects=_Effects(),
        prepared_request=TrainingRequest('request', 'project', public.canonical_json()),
        run=TrainingRunRef('run-packaged', 'project'), components=components,
        context=ProjectContext.standalone(engine_root=tmp_path), input_preparation=None,
        clock=UTCClock(), secrets=_Secrets(),
        authority_secret=SecretRef('env', 'SYNAPTIC_TEST_AUTHORITY'),
        reader_secret=SecretRef('env', 'SYNAPTIC_TEST_READER'),
        profile_digest='a' * 64, quote_digest='b' * 64, maximum_cost_minor_units=100,
    ), events


def test_default_stores_remain_in_memory(tmp_path):
    kwargs, events = arguments(tmp_path)
    host = compose_modal_packaged_reference_host(**kwargs)
    workflow = host.composition.stores.workflow_store
    assert type(workflow._records) is InMemoryDurableRecordStoreV1
    assert type(workflow._streams) is InMemoryDurableStreamStoreV1
    assert events == []


def test_exact_supplied_objects_reach_generic_workflow_store(tmp_path):
    kwargs, events = arguments(tmp_path)
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    host = compose_modal_packaged_reference_host(**kwargs, records=records, streams=streams)
    assert host.composition.stores.workflow_store._records is records
    assert host.composition.stores.workflow_store._streams is streams
    assert events == []


@pytest.mark.parametrize('pair', ['records_only', 'streams_only', 'bad_records', 'bad_streams', 'properties'])
def test_incomplete_or_property_store_ports_reject_before_secret_resolution(tmp_path, pair):
    kwargs, events = arguments(tmp_path)
    class ForbiddenSecrets:
        def resolve(self, ref):
            pytest.fail('invalid stores reached secret resolution')
    class PropertyRecords:
        @property
        def create(self):
            pytest.fail('store capability detection invoked a property')
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    options = {'records': records, 'streams': streams}
    if pair == 'records_only':
        options.pop('streams')
    elif pair == 'streams_only':
        options.pop('records')
    elif pair == 'bad_records':
        options['records'] = object()
    elif pair == 'bad_streams':
        options['streams'] = object()
    else:
        options['records'] = PropertyRecords()
    kwargs['secrets'] = ForbiddenSecrets()
    with pytest.raises(TypeError):
        compose_modal_packaged_reference_host(**kwargs, **options)
    assert events == []
