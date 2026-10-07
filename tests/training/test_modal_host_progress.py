"""Authenticated host projection works before packaged completion is present."""
import pytest

from tests.execution.test_observation_logs import metric_record
from tests.training.test_modal_host_reader import _reader
from tuner.execution.coordinator_v1.model import ProviderLogQueryV1
from tuner.execution.observation_logs import encode_training_metric_log
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.packaged_reader import (
    ModalPackagedProgressPage, _progress_page_evidence,
)
from tuner.training.modal_host_reader import ModalPackagedReadUnavailable


def _page(reader, request, *, count=1, after=None):
    entries = tuple(encode_training_metric_log(metric_record(i, run=request.run),
                   expected_run=request.run) for i in range(count))
    values = (
        request.run, reader.binding.command.operation.effect.effect_id,
        reader.binding.command_digest, request.provider_run.reference.provider_job_ref,
        reader.binding.execution_binding.binding_digest, entries, after,
        count - 1 if count else (-1 if after is None else after),
        "f" * 64 if count else None, False, False,
    )
    return ModalPackagedProgressPage(*values, _progress_page_evidence(*values))


def test_pending_progress_is_authenticated_without_polling_completion(monkeypatch):
    reader, request, authority = _reader(ModalFunctionCallState.PENDING)
    page = _page(reader, request)
    seen = []

    def read(binding, **kwargs):
        assert binding == reader.binding
        seen.append(kwargs)
        return page

    reader._reader.read_progress = read
    monkeypatch.setattr(reader, "_completion", lambda *args: pytest.fail("completion poll"))
    result = reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert authority.authenticate_log_page(result)
    assert result.content.entries == page.entries
    assert result.content.canonical_evidence == page.canonical_evidence
    assert seen == [dict(provider_job_ref="fc-1", expected_run=request.run,
                         after_sequence=None, limit=2, maximum_bytes=4096)]


def test_identical_query_can_grow_from_empty_to_nonempty():
    reader, request, authority = _reader(ModalFunctionCallState.PENDING)
    pages = iter((_page(reader, request, count=0), _page(reader, request)))
    reader._reader.read_progress = lambda *args, **kwargs: next(pages)
    query = ProviderLogQueryV1(None, 2, 4096)
    first, second = reader.logs(request, query), reader.logs(request, query)
    assert first.content.entries == ()
    assert len(second.content.entries) == 1
    assert authority.authenticate_log_page(first) and authority.authenticate_log_page(second)


@pytest.mark.parametrize("change", [
    {"command_digest": "0" * 64}, {"provider_job_ref": "wrong-call"},
    {"execution_binding_digest": "0" * 64}, {"effect_id": "wrong-effect"},
])
def test_wrong_bound_progress_is_rejected_without_private_error(change):
    reader, request, _ = _reader(ModalFunctionCallState.PENDING)
    page = _page(reader, request)
    for key, value in change.items():
        object.__setattr__(page, key, value)
    object.__setattr__(page, "canonical_evidence", page.expected_evidence)
    reader._reader.read_progress = lambda *args, **kwargs: page
    with pytest.raises(ModalPackagedReadUnavailable) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "modal_packaged_progress_read_invalid"


def test_tampered_canonical_evidence_rejected():
    reader, request, _ = _reader(ModalFunctionCallState.PENDING)
    page = _page(reader, request)
    object.__setattr__(page, "canonical_evidence", b"{}")
    reader._reader.read_progress = lambda *args, **kwargs: page
    with pytest.raises(ModalPackagedReadUnavailable, match="progress_read_invalid"):
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))


def test_reader_failure_does_not_call_training_or_echo_private_details():
    reader, request, _ = _reader(ModalFunctionCallState.PENDING)

    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE_SENTINEL")

    reader._reader.read_progress = fail
    with pytest.raises(ModalPackagedReadUnavailable) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "modal_packaged_progress_read_invalid"
    assert caught.value.__suppress_context__
