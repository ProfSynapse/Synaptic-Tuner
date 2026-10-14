"""Scripted public HF lifecycle; no live backend or source-free restart claim."""

from datetime import datetime, timezone

import pytest

sdk = pytest.importorskip("huggingface_hub")

from tests.execution.providers.test_hf_jobs_public_family import (
    material as material,
    _job,
    Clock,
    Grants,
    Secrets,
)
from tuner.execution.providers.hf_jobs_provider_v1 import preparation as prep_module
from tuner.execution.providers.hf_jobs_provider_v1.composition import (
    compose_hf_jobs_host,
)
from synaptic_tuner.api.v1.providers import ProviderCapabilities
from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
from synaptic_tuner.api.v1.reference import (
    compose_reference_authority,
    ReferenceRequestPortsV1,
)
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1,
    InMemoryDurableStreamStoreV1,
)
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef

pytestmark = pytest.mark.skipif(
    sdk.__version__ != "1.33.0", reason="exact isolated HF host SDK"
)


class MutableClock(Clock):
    def __init__(self):
        self.epoch = super().now_epoch()

    def now_epoch(self):
        return self.epoch

    def now(self):
        return (
            datetime.fromtimestamp(self.epoch, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )


class CurrentGrants(Grants):
    def __init__(self):
        super().__init__()
        self.requirements = []

    def authorize(self, requirements):
        self.requirements.append(requirements)
        return super().authorize(requirements)


@pytest.fixture
def enabled_material(tmp_path, monkeypatch):
    # Scripted test composition chooses capabilities BEFORE the original plan
    # and descriptor binding. Production defaults remain off; no historical run
    # digest is changed and this is not qualification/capability promotion.
    monkeypatch.setattr(
        prep_module,
        "ProviderCapabilities",
        lambda *args: ProviderCapabilities(True, False, True, True, True, False),
    )
    root = tmp_path / "enabled"
    root.mkdir(mode=0o700)
    return material.__wrapped__(root)


def host(
    value,
    *,
    records=None,
    streams=None,
    clock=None,
    grants=None,
    current=False,
    deadline=30,
):
    prep = value.preparation
    clock, grants = clock or MutableClock(), grants or CurrentGrants()
    ports = ReferenceHostPortsV1(
        records or InMemoryDurableRecordStoreV1(),
        streams or InMemoryDurableStreamStoreV1(),
        clock,
        grants,
        Secrets(),
    )
    authority = compose_reference_authority(
        clock=clock,
        secrets=ports.secrets,
        authority_secret=SecretRef("env", "AUTHORITY"),
    )
    loader = type(
        "Loader",
        (),
        {
            "load": lambda self, raw: TrainingRequest(
                prep.resolved.request_id, prep.resolved.project_ref, raw
            )
        },
    )()
    resolver = type("Resolver", (), {"resolve": lambda self, request: prep.resolved})()
    identity = type(
        "Identity",
        (),
        {
            "for_plan": lambda self, plan: TrainingRunRef(
                prep.rich_plan.execution_source.run_id, prep.resolved.project_ref
            )
        },
    )()
    composed = compose_hf_jobs_host(
        preparation=prep,
        sdk=sdk,
        api=value.provider,
        ports=ports,
        requests=ReferenceRequestPortsV1(loader, resolver, identity),
        authority=authority,
        scratch_root=value.scratch,
        reader_authority_ref="fixture-reader",
        reader_key_ref="fixture-reader-key",
        reader_key=b"r" * 32,
        allow_current_cancel_authorization=current,
        deadline_seconds=deadline,
    )
    return composed.api(), composed, ports


def start(api, value):
    plan = api.training.plan(
        api.training.resolve(api.training.load("{}")),
        value.preparation.selection.provider,
    )
    ready = api.training.preflight(plan)
    return api.training.start(plan, ready).run, plan, ready


def cancellation(value, *, lost=False):
    def cancel_job(*, job_id, namespace, token):
        assert (job_id, namespace) == (value.provider.job.id, "fixture-owner")
        assert token == Secrets().resolve(SecretRef("env", "HOST_TOKEN"))
        value.provider.calls.append(("cancel", job_id))
        if lost:
            raise TimeoutError("scripted lost acknowledgement")
        return None

    value.provider.cancel_job = cancel_job


def status(value, phase):
    arguments = next(
        arguments
        for operation, arguments in value.provider.calls
        if operation == "submit"
    )
    value.provider.job = _job(arguments, phase)


def test_pinned_sdk_cancel_ack_and_inspect_terminal_are_distinct(monkeypatch):
    import httpx
    import huggingface_hub.hf_api as module

    calls = []
    phase = "RUNNING"

    class Session:
        def post(self, url, **kwargs):
            calls.append(("POST", url))
            return httpx.Response(200, request=httpx.Request("POST", url))

        def get(self, url, **kwargs):
            calls.append(("GET", url))
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={
                    "id": "known-job",
                    "owner": {"id": "owner", "name": "fixture-owner", "type": "user"},
                    "dockerImage": "python:3.12",
                    "command": ["python"],
                    "arguments": [],
                    "environment": {},
                    "flavor": "cpu-basic",
                    "labels": {},
                    "volumes": [],
                    "status": {"stage": phase},
                    "secrets": {},
                },
            )

    monkeypatch.setattr(module, "get_session", Session)
    api = sdk.HfApi(endpoint="https://huggingface.co")
    monkeypatch.setattr(
        api, "whoami", lambda **kwargs: pytest.fail("implicit account lookup")
    )
    assert (
        api.cancel_job(
            job_id="known-job", namespace="fixture-owner", token="fixture-token"
        )
        is None
    )
    for phase in ("RUNNING", "CANCELED", "COMPLETED", "ERROR"):
        observed = api.inspect_job(
            job_id="known-job", namespace="fixture-owner", token="fixture-token"
        )
        assert observed.id == "known-job" and observed.owner.name == "fixture-owner"
        assert observed.status.stage == phase
    assert (
        calls
        == [("POST", "https://huggingface.co/api/jobs/fixture-owner/known-job/cancel")]
        + [("GET", "https://huggingface.co/api/jobs/fixture-owner/known-job")] * 4
    )


@pytest.mark.parametrize(
    "phase, expected",
    [
        ("RUNNING", "cancel_requested"),
        ("CANCELED", "cancelled"),
        ("COMPLETED", "succeeded"),
        ("ERROR", "failed"),
    ],
)
def test_public_acknowledged_cancel_preserves_actual_job_terminal(
    enabled_material, phase, expected
):
    value = enabled_material
    cancellation(value)
    api, composition, ports = host(value, current=True)
    run, _, ready = start(api, value)
    original = ready.to_dict()
    ports.clock.epoch += 3600
    result = api.runs.cancel(run, "stop exact run")
    assert result.state.value == "cancel_requested"
    assert ready.to_dict() == original
    assert ports.grants.requirements[-1][0].operation == "training.cancel"
    status(value, phase)
    assert api.runs.outcome(run).state.value == expected
    assert len([call for call in value.provider.calls if call[0] == "cancel"]) == 1
    workflow = composition.stores.workflow_store.get(run)
    assert workflow.cancel.foundation_outcomes[-1].disposition.value == "found"


def test_default_flags_and_expiry_are_not_weakened(material, enabled_material):
    # The explicit test-enabled second fixture must not relabel original off
    # descriptor runs. Inspect the first original constructor's flags directly.
    assert material.preparation.descriptor.capabilities.cancel is False
    cancellation(material)
    default_api, _, _ = host(material, current=True)
    default_run, _, _ = start(default_api, material)
    with pytest.raises(Exception):
        default_api.runs.cancel(default_run, "default descriptor stays disabled")
    assert not [call for call in material.provider.calls if call[0] == "cancel"]
    cancellation(enabled_material)
    api, _, ports = host(enabled_material, current=False)
    run, _, _ = start(api, enabled_material)
    ports.clock.epoch += 3600
    with pytest.raises(Exception):
        api.runs.cancel(run, "expired start has no fresh cancellation opt-in")
    assert not [call for call in enabled_material.provider.calls if call[0] == "cancel"]


@pytest.mark.parametrize("lost", [False, True])
def test_real_sqlite_reopen_preserves_ack_or_unknown_without_reissuing(
    enabled_material, tmp_path, lost
):
    value, clock = enabled_material, MutableClock()
    cancellation(value, lost=lost)
    path = tmp_path / "host.sqlite3"
    with SQLiteDurableStoreV1(path) as store:
        api, composition, ports = host(
            value, records=store, streams=store, clock=clock, current=True
        )
        run, plan, ready = start(api, value)
        clock.epoch += 3600
        api.runs.cancel(run, "one stop")
        assert api.runs.show(run).state.value == (
            "reconcile_required" if lost else "cancel_requested"
        )
        api.runs.cancel(run, "one stop")
    before = list(value.provider.calls)
    status(value, "CANCELED")
    with SQLiteDurableStoreV1(path, create=False) as store:
        cold, composed, ports = host(
            value, records=store, streams=store, clock=clock, current=True
        )
        outcome = cold.runs.reconcile(run)
        assert outcome.state.value == (
            "reconcile_required" if lost else "cancel_requested"
        )
        record = composed.stores.workflow_store.get(run)
        assert record.cancel.foundation_outcomes[-1].disposition.value == (
            "indeterminate" if lost else "found"
        )
        assert ports.grants.calls == (["authorize"] if lost else [])
    assert [
        call
        for call in value.provider.calls
        if call[0] in {"cancel", "submit", "upload"}
    ] == [call for call in before if call[0] in {"cancel", "submit", "upload"}]


@pytest.mark.parametrize("field", ("id", "docker_image", "labels", "command", "owner"))
def test_wrong_provider_target_never_sends_cancel(enabled_material, field):
    value = enabled_material
    cancellation(value)
    api, _, _ = host(value, current=True)
    run, _, _ = start(api, value)
    if field == "owner":
        value.provider.job.owner.name = "other-owner"
    else:
        setattr(
            value.provider.job,
            field,
            {"changed": "target"}
            if field == "labels"
            else ["changed"]
            if field == "command"
            else "other-target",
        )
    assert api.runs.cancel(run, "stop exact job").state.value == "reconcile_required"
    assert not [call for call in value.provider.calls if call[0] == "cancel"]


def test_cancel_receipt_store_failure_keeps_claim_on_reopen(
    enabled_material, tmp_path, monkeypatch
):
    value = enabled_material
    cancellation(value)
    path = tmp_path / "fault.sqlite3"
    with SQLiteDurableStoreV1(path) as store:
        api, composition, _ = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
        original = store.create

        def create(*args, **kwargs):
            if "/cancel/" in kwargs.get("key", ""):
                raise RuntimeError("scripted receipt store failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "create", create)
        assert api.runs.cancel(run, "one stop").state.value == "reconcile_required"
        assert (
            composition.stores.workflow_store.get(run)
            .cancel.foundation_outcomes[-1]
            .disposition.value
            == "indeterminate"
        )
    with SQLiteDurableStoreV1(path, create=False) as reopened:
        cold, _, _ = host(value, records=reopened, streams=reopened, current=True)
        assert cold.runs.reconcile(run).state.value == "reconcile_required"
    assert len([call for call in value.provider.calls if call[0] == "cancel"]) == 1


def test_late_cancel_cannot_publish_receipt_after_store_closes(
    enabled_material, tmp_path
):
    from threading import Event

    value = enabled_material
    entered, release, finished = Event(), Event(), Event()

    def cancel_job(**kwargs):
        value.provider.calls.append(("cancel", kwargs["job_id"]))
        entered.set()
        assert release.wait(5)
        finished.set()
        return None

    value.provider.cancel_job = cancel_job
    path = tmp_path / "late.sqlite3"
    try:
        with SQLiteDurableStoreV1(path) as store:
            api, _, _ = host(
                value, records=store, streams=store, current=True, deadline=1
            )
            run, _, _ = start(api, value)
            assert (
                api.runs.cancel(run, "bounded stop").state.value == "reconcile_required"
            )
            assert entered.is_set() and not finished.is_set()
        release.set()
        assert finished.wait(2)
        with SQLiteDurableStoreV1(path, create=False) as reopened:
            cold, _, _ = host(value, records=reopened, streams=reopened, current=True)
            assert cold.runs.reconcile(run).state.value == "reconcile_required"
        assert len([call for call in value.provider.calls if call[0] == "cancel"]) == 1
    finally:
        release.set()
        assert finished.wait(2)


@pytest.mark.parametrize("invalid", (1, None, "true"))
def test_cancel_opt_in_requires_exact_bool(material, invalid):
    with pytest.raises(Exception):
        host(material, current=invalid)


@pytest.mark.parametrize("kind", ("stage", "submit"))
def test_unknown_effect_lookup_never_discovers_or_replays(enabled_material, kind):
    value = enabled_material
    method = "batch_bucket_files" if kind == "stage" else "run_job"
    original = getattr(value.provider, method)

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("scripted effect reply loss")

    setattr(value.provider, method, lost)
    api, composition, _ = host(value, current=True)
    run, _, _ = start(api, value)
    assert api.runs.show(run).state.value == "reconcile_required"
    before = list(value.provider.calls)
    assert api.runs.reconcile(run).state.value == "reconcile_required"
    assert value.provider.calls == before
    workflow = composition.stores.workflow_store.get(run)
    assert (workflow.stage if kind == "stage" else workflow.submit).foundation_outcomes[
        -1
    ].disposition.value == "indeterminate"


def test_known_private_submit_receipt_reconciles_without_resubmit(
    enabled_material, tmp_path, monkeypatch
):
    from dataclasses import replace
    from tuner.execution.foundation_v2.identities import EffectKind
    from tuner.execution.foundation_v2.observations import ObservationDisposition
    from tuner.execution.providers.hf_jobs_provider_v1.composition import _Executor

    original = _Executor.execute_once

    def lost_after_private_receipt(self, payload, request):
        result = original(self, payload, request)
        if request.effect_kind == EffectKind.SUBMIT.value:
            # Actual SDK submission and durable private receipt occurred; model
            # loss before the Foundation result receives that known reference.
            return replace(
                result,
                disposition=ObservationDisposition.INDETERMINATE,
                provider_run=None,
            )
        return result

    monkeypatch.setattr(_Executor, "execute_once", lost_after_private_receipt)
    value = enabled_material
    path = tmp_path / "known.sqlite3"
    with SQLiteDurableStoreV1(path) as store:
        api, _, _ = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
        assert api.runs.show(run).state.value == "reconcile_required"
    before = list(value.provider.calls)
    with SQLiteDurableStoreV1(path, create=False) as reopened:
        cold, _, ports = host(value, records=reopened, streams=reopened, current=True)
        assert cold.runs.reconcile(run).state.value == "queued"
        assert cold.runs.outcome(run).state.value == "running"
        assert not [call for call in ports.grants.calls if call in {"stage", "submit"}]
    assert [
        call for call in value.provider.calls if call[0] in {"upload", "submit"}
    ] == [call for call in before if call[0] in {"upload", "submit"}]


def test_acknowledged_completion_race_keeps_native_verification_and_streams(
    enabled_material,
):
    import hashlib
    from tests.execution.providers.test_hf_jobs_public_family import (
        _native_values,
        _complete,
    )
    from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest

    value = enabled_material
    cancellation(value)
    api, composition, _ = host(value, current=True)
    run, _, _ = start(api, value)
    assert api.runs.cancel(run, "completion may race").state.value == "cancel_requested"
    values = _native_values(value.preparation)
    _complete(value, values)
    assert api.runs.outcome(run).state.value == "succeeded"
    assert api.runs.verify(run).verified is True
    assert api.runs.show(run).state.value == "succeeded"
    workflow = composition.stores.workflow_store.get(run)
    assert workflow.phase.value == "verified"
    assert workflow.cancel.foundation_outcomes[-1].disposition.value == "found"
    for role, (_, expected) in values.items():
        stream = api.runs.artifacts(RunArtifactRequest(run, role, len(expected)))
        actual = b"".join(stream.iter_bytes())
        assert (
            actual == expected
            and hashlib.sha256(actual).hexdigest()
            == hashlib.sha256(expected).hexdigest()
        )
    assert len([call for call in value.provider.calls if call[0] == "cancel"]) == 1
    assert len([call for call in value.provider.calls if call[0] == "submit"]) == 1
