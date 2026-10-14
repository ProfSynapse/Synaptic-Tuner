"""Restricted admission and delivery proof, without SDK/provider effects."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
import os

import pytest
from synaptic_tuner.api.v1.observations import (
    ObservationKind,
    TrainingPhaseObservedPayloadV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.observation_logs import encode_training_progress_log
from tests.execution.test_observation_logs import metric_record
from tests.training.test_run_progress_consumer import Pages
from tuner.training.run_progress_consumer import RunProgressConsumer
from tuner.training.modal_metrics_qualification import (
    ADMISSION_SCHEMA,
    MetricsQualificationAdmission,
    MetricsQualificationUnavailable,
    MetricsQualificationLedger,
    current_clean_commit,
    load_admission,
    write_candidate,
)

RUN = TrainingRunRef("run-progress", "project-progress")


def document():
    return dict(
        schema_version=ADMISSION_SCHEMA,
        attempt_ref="metrics-test-1",
        engine_commit="a" * 40,
        profile_sha256="b" * 64,
        runtime_material_intent_digest="c" * 64,
        workload_digest="d" * 64,
        configuration_digest="e" * 64,
        dataset_digest="f" * 64,
        dataset_content_digest="0" * 64,
        modal_profile="selected",
        modal_environment="main",
        account_ref="workspace",
        workspace_ref="workspace",
        accelerator="L40S",
        accelerator_count=1,
        timeout_seconds=900,
        maximum_cost_minor_units=500,
        max_steps=32,
        capture_max_records=32,
    )


def admission(document=None):
    raw = canonical_bytes(document or globals()["document"]())
    return MetricsQualificationAdmission(raw, hashlib.sha256(raw).hexdigest())


@pytest.fixture
def root(tmp_path):
    tmp_path.chmod(0o700)
    return tmp_path


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_steps", True),
        ("timeout_seconds", 901),
        ("accelerator_count", 2),
        ("capture_max_records", 1),
        ("maximum_cost_minor_units", 501),
        ("engine_commit", "short"),
        ("account_ref", None),
        ("unknown", "extra"),
    ],
)
def test_strict_admission_rejects_malformed(field, value):
    d = document()
    d[field] = value
    with pytest.raises(MetricsQualificationUnavailable):
        admission(d)


def test_canonical_digest_duplicate_and_bounded_file_admission(root):
    expected = admission()
    path = root / "admission.json"
    write_candidate(path, expected.canonical)
    assert load_admission(path, expected.sha256) == expected
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(MetricsQualificationUnavailable):
        load_admission(path, "0" * 64)
    raw = expected.canonical[:-1] + b',"max_steps":32}'
    with pytest.raises(MetricsQualificationUnavailable):
        MetricsQualificationAdmission(raw, hashlib.sha256(raw).hexdigest())
    path.write_bytes(b" " + expected.canonical)
    with pytest.raises(MetricsQualificationUnavailable):
        load_admission(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("kind", ["fifo", "symlink", "oversize"])
def test_untrusted_admission_leaves_reject_without_blocking(root, kind):
    path = root / "input"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to(root / "target")
    else:
        path.write_bytes(b"x" * 8193)
    with pytest.raises(MetricsQualificationUnavailable):
        load_admission(path, "a" * 64)


def test_actual_engine_root_must_match_package_source(root):
    with pytest.raises(MetricsQualificationUnavailable):
        current_clean_commit(root)


def test_scope_rejects_account_workspace_environment():
    bound = admission()
    for field in ("account_ref", "workspace_ref", "environment_ref"):
        values = dict(
            account_ref="workspace", workspace_ref="workspace", environment_ref="main"
        )
        values[field] = "another"
        with pytest.raises(MetricsQualificationUnavailable):
            bound.validate_scope(SimpleNamespace(**values))


def entries():
    result = []
    for number in range(2):
        record = metric_record(number, run=RUN)
        record = replace(record, payload=replace(record.payload, loss=-0.03125))
        result.append(encode_training_progress_log(record, expected_run=RUN))
    return result


def ledger_with_records(root, *, before="pending", after="pending", page_entries=None):
    source = Pages([(entries() if page_entries is None else page_entries, None)])
    consumer = RunProgressConsumer(source, RUN, root)
    ledger = MetricsQualificationLedger(admission(), RUN, "fc-owned", root)
    consumer.poll()
    ledger.record(
        consumer,
        before_state=before,
        after_state=after,
        receipt_time="2026-10-12T10:00:00Z",
        requested_cursor=None,
    )
    consumer.close()
    ledger.compare_retained(consumer)
    return ledger, consumer


def test_fractional_numeric_pending_proof_is_provisional_until_durable_return(root):
    ledger, consumer = ledger_with_records(root)
    result = ledger.finish(verified=True)
    assert result["delivery_verdict"] == "delivered_precompletion"
    assert result["precompletion_numeric_records"] == 2
    assert result["durability_acknowledged"] is True
    assert result["retained_file_compared"] is True
    assert (
        result["retained_file_sha256"]
        == hashlib.sha256(consumer.path.read_bytes()).hexdigest()
    )
    saved = json.loads(ledger.final_path.read_bytes())
    assert saved["durability_acknowledged"] is False
    assert (
        result["report_sha256"]
        == hashlib.sha256(ledger.final_path.read_bytes()).hexdigest()
    )
    assert (
        saved["transport_qualified"] is False
        and saved["partial_write_behavior_qualified"] is False
    )


@pytest.mark.parametrize(
    "before,after,verified",
    [
        ("returned", "returned", True),
        ("pending", "returned", True),
        ("pending", "pending", False),
    ],
)
def test_terminal_only_or_unverified_is_inconclusive(root, before, after, verified):
    ledger, _ = ledger_with_records(root, before=before, after=after)
    assert ledger.finish(verified=verified)["delivery_verdict"] == "inconclusive"


def test_status_only_and_duplicates_do_not_prove_two_numeric_records(root):
    record = replace(
        metric_record(0, run=RUN),
        kind=ObservationKind.TRAINING_PHASE_OBSERVED,
        payload=TrainingPhaseObservedPayloadV1("metric_capture_exhausted"),
    )
    status = encode_training_progress_log(record, expected_run=RUN)
    ledger, consumer = ledger_with_records(root, page_entries=[status])
    ledger.record(
        consumer,
        before_state="pending",
        after_state="pending",
        receipt_time="2026-10-12T10:00:01Z",
        requested_cursor=None,
    )
    result = ledger.finish(verified=True)
    assert (
        result["delivery_verdict"] == "inconclusive"
        and result["precompletion_numeric_records"] == 0
    )


@pytest.mark.parametrize("target", ["progress", "ledger"])
def test_retained_file_corruption_blocks_success(root, target):
    ledger, consumer = ledger_with_records(root)
    path = consumer.path if target == "progress" else ledger.path
    path.write_bytes(b"partial")
    ledger.compare_retained(consumer)
    assert ledger.finish(verified=True)["delivery_verdict"] == "evidence_invalid"


def test_final_fsync_failure_cannot_create_an_acknowledged_success_report(
    root, monkeypatch
):
    ledger, _ = ledger_with_records(root)
    original = os.fsync

    def fail_file(fd):
        if os.readlink(f"/proc/self/fd/{fd}").endswith("-result.json"):
            raise OSError("private failure")
        return original(fd)

    monkeypatch.setattr(os, "fsync", fail_file)
    result = ledger.finish(verified=True)
    assert (
        result["delivery_verdict"] == "evidence_invalid"
        and not result["durability_acknowledged"]
    )
    assert (
        json.loads(ledger.final_path.read_bytes())["durability_acknowledged"] is False
    )


def qualification_setup(tmp_path, monkeypatch):
    from tests.training.test_modal_standalone_runner import _setup
    from tuner.training import modal_metrics_qualification as metrics
    from tuner.training.modal_metrics_qualification import candidate_admission

    plan, context, events = _setup(
        tmp_path, monkeypatch, capture=True, metrics_qualification=True
    )
    monkeypatch.setattr(metrics, "current_clean_commit", lambda _root: "a" * 40)
    raw = candidate_admission(
        plan,
        engine_commit="a" * 40,
        attempt_ref="one-approved-attempt",
        modal_profile="explicit",
        modal_environment="main",
        account_ref="workspace",
        workspace_ref="workspace",
    )
    return (
        plan,
        context,
        events,
        MetricsQualificationAdmission(raw, hashlib.sha256(raw).hexdigest()),
    )


def signed_public_transport(monkeypatch, *, torn_first=False):
    from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
    from tuner.execution.providers.modal.packaged_deployment import (
        ModalPackagedDeploymentObserver,
    )
    from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
    from tuner.execution.providers.modal.packaged_progress import (
        make_progress_file,
        ZERO_DIGEST,
        progress_path,
    )
    from tuner.execution.providers.modal.contracts import provider_entry_identity

    original = ModalPackagedReader.read_progress
    files = {}
    calls = []
    monkeypatch.setattr(
        ModalPackagedDeploymentObserver, "observe", lambda _self, facts, **_kw: facts
    )

    def read(self, binding, **kwargs):
        calls.append(binding.command_digest)
        files.clear()
        previous = ZERO_DIGEST
        for index in range(2):
            record = metric_record(index, run=kwargs["expected_run"])
            record = replace(record, payload=replace(record.payload, loss=-0.03125))
            raw, previous = make_progress_file(
                signer=self._verifier,
                key_ref=self._key_ref,
                effect_id=binding.command.operation.effect.effect_id,
                command_digest=binding.command_digest,
                provider_job_ref=kwargs["provider_job_ref"],
                execution_binding_digest=binding.execution_binding.binding_digest,
                run=kwargs["expected_run"],
                max_records=10,
                index=index,
                previous_digest=previous,
                record=record,
            )
            if torn_first and len(calls) == 1 and index == 1:
                raw = raw[:10]
            files[progress_path(binding.command.operation.effect.effect_id, index)] = (
                raw
            )
        return original(self, binding, **kwargs)

    monkeypatch.setattr(ModalPackagedReader, "read_progress", read)

    def listed(_self, volume, prefix, *, max_entries):
        return tuple(
            (path, len(raw), provider_entry_identity(volume, path, len(raw)))
            for path, raw in sorted(files.items())
            if path.startswith(prefix)
        )

    monkeypatch.setattr(ExplicitModal154ReadFacade, "list_prefix", listed)
    monkeypatch.setattr(
        ExplicitModal154ReadFacade,
        "read_complete",
        lambda _self, volume, path, **_kw: files[path],
    )
    return calls


@pytest.mark.parametrize("torn_first", [False, True])
def test_real_restricted_host_public_signed_read_and_one_shot_ledger(
    tmp_path, monkeypatch, torn_first
):
    from tuner.training import modal_standalone_runner as runner
    from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
    from tuner.execution.providers.modal.facade import ModalFunctionCallState as State

    plan, context, events, bound = qualification_setup(tmp_path, monkeypatch)
    calls = signed_public_transport(monkeypatch, torn_first=torn_first)
    states = iter(
        [State.PENDING, State.PENDING] * (2 if torn_first else 1)
        + [State.RETURNED, State.RETURNED]
    )
    monkeypatch.setattr(
        ModalPackagedCoordinatorReaderV1,
        "_poll_packaged_call",
        lambda *_a, **_kw: (next(states, State.RETURNED), "f" * 64),
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _n: None)
    result = runner.run_modal_standalone_job(
        plan=plan,
        context=context,
        modal_profile="explicit",
        modal_environment="main",
        metrics_admission=bound,
    )
    evidence = result.metrics_qualification
    assert (
        evidence["delivery_verdict"] == "delivered_precompletion"
        and evidence["durability_acknowledged"]
    )
    assert (
        evidence["precompletion_numeric_records"] == 2
        and evidence["read_rejection_observed"] == torn_first
    )
    assert (
        evidence["terminal_artifacts_verified"] and not evidence["transport_qualified"]
    )
    assert len(calls) == (2 if torn_first else 1)
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
    prior = list(events)
    with pytest.raises(runner.ModalStandaloneRunUnavailable):
        runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="explicit",
            modal_environment="main",
            metrics_admission=bound,
        )
    assert events == prior


@pytest.mark.parametrize("change", ["source", "scope", "plan", "fresh"])
def test_unbound_admission_rejects_before_effects(tmp_path, monkeypatch, change):
    from tuner.training import modal_standalone_runner as runner

    plan, context, events, bound = qualification_setup(tmp_path, monkeypatch)
    d = bound.document
    fresh = False
    if change == "source":
        d["engine_commit"] = "b" * 40
    elif change == "scope":
        d["modal_environment"] = "other"
    elif change == "plan":
        d["workload_digest"] = "f" * 64
    else:
        fresh = True
    altered = admission(d)
    with pytest.raises(runner.ModalStandaloneRunUnavailable):
        runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="explicit",
            modal_environment="main",
            metrics_admission=altered,
            fresh_attempt=fresh,
        )
    assert events == []
    assert not (tmp_path / "state" / "synaptic-training").exists()


def test_actual_account_mismatch_blocks_build_after_readonly_scope(
    tmp_path, monkeypatch
):
    from tuner.training import modal_standalone_runner as runner

    plan, context, events, bound = qualification_setup(tmp_path, monkeypatch)
    d = bound.document
    d["account_ref"] = "other"
    with pytest.raises(runner.ModalStandaloneRunUnavailable):
        runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="explicit",
            modal_environment="main",
            metrics_admission=admission(d),
        )
    assert events == []


def test_ordinary_composition_rejects_qualification_authority():
    from tuner.training.modal_host_composition import (
        compose_modal_packaged_reference_host,
    )

    with pytest.raises(TypeError):
        compose_modal_packaged_reference_host(qualification_admission=admission())


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--qualification-sha256", "a" * 64],
        ["train", "--job-config", "recipe.yaml", "--qualify-metrics", "admission.json"],
        ["train", "--job-config", "recipe.yaml", "--qualification-output", "file"],
        [
            "train",
            "--job-config",
            "recipe.yaml",
            "--metrics-qualification-attempt",
            "attempt",
        ],
        [
            "train",
            "--job-config",
            "recipe.yaml",
            "--qualify-metrics",
            "file",
            "--qualification-sha256",
            "a" * 64,
            "--fresh-attempt",
        ],
        [
            "train",
            "--job-config",
            "recipe.yaml",
            "--qualify-metrics",
            "file",
            "--qualification-sha256",
            "a" * 64,
            "--quote",
        ],
        [
            "cloud-run",
            "--job-config",
            "recipe.yaml",
            "--qualify-metrics",
            "file",
            "--qualification-sha256",
            "a" * 64,
        ],
    ],
)
def test_misplaced_flags_cannot_fall_back_to_training(arguments):
    from tuner.cli.parser import create_parser

    with pytest.raises(SystemExit):
        create_parser().parse_args(arguments)


@pytest.mark.parametrize("fault", ["truncate", "substitute"])
def test_partial_or_replaced_progress_file_rejects_proof(root, fault):
    ledger, consumer = ledger_with_records(root)
    if fault == "truncate":
        consumer.path.write_bytes(consumer.path.read_bytes()[:-1])
    else:
        raw = consumer.path.read_bytes()
        replacement = root / "replacement"
        replacement.write_bytes(raw)
        replacement.replace(consumer.path)
    ledger.compare_retained(consumer)
    assert ledger.finish(verified=True)["delivery_verdict"] == "evidence_invalid"


@pytest.mark.parametrize("phase", ["consumer", "ledger"])
def test_progress_or_ledger_fsync_failure_cannot_prove_delivery(
    root, monkeypatch, phase
):
    original = os.fsync

    def fail_file(fd):
        target = os.readlink(f"/proc/self/fd/{fd}")
        if (phase == "consumer" and "/progress-" in target) or (
            phase == "ledger" and "qualification-" in target
        ):
            raise OSError("private error")
        return original(fd)

    monkeypatch.setattr(os, "fsync", fail_file)
    ledger, _ = ledger_with_records(root)
    assert ledger.finish(verified=True)["delivery_verdict"] == "evidence_invalid"


@pytest.mark.parametrize(
    "binding_change", ["runtime_material_intent_digest", "run", "scope"]
)
def test_restricted_factory_revalidates_claimed_runtime_run_scope(
    tmp_path, monkeypatch, binding_change
):
    from tuner.training import modal_standalone_runner as runner

    plan, context, events, bound = qualification_setup(tmp_path, monkeypatch)
    original = runner.compose_modal_metrics_qualification_host

    def changed(*, admission, **kwargs):
        if binding_change == "runtime_material_intent_digest":
            d = admission.document
            d[binding_change] = "f" * 64
            admission = globals()["admission"](d)
        elif binding_change == "run":
            kwargs["run"] = TrainingRunRef("another-run", kwargs["run"].project_ref)
        else:
            d = admission.document
            d["workspace_ref"] = "another"
            admission = globals()["admission"](d)
        return original(admission=admission, **kwargs)

    monkeypatch.setattr(runner, "compose_modal_metrics_qualification_host", changed)
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as caught:
        runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="explicit",
            modal_environment="main",
            metrics_admission=bound,
        )
    assert caught.value.phase == "RUN_HOST_ASSEMBLY"
    assert events == ["bootstrap", "cpu"]


def test_candidate_generation_has_no_authority_effects(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler

    plan, context, events, _ = qualification_setup(tmp_path, monkeypatch)
    # This control starts with an admitted fixture plan; recipe dispatch is
    # covered separately. Patch the current planning port, not its SFT delegate.
    monkeypatch.setattr(
        "tuner.training.modal_recipe.plan_modal_recipe", lambda *_a, **_k: plan
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "admission.json"
    # The fixture engine_root is tmp_path. Use a separate private sibling so
    # admission output cannot make its tracked engine source dirty.
    outside = tmp_path.parent / (tmp_path.name + "-private")
    outside.mkdir(mode=0o700)
    output = outside / "admission.json"
    handler = ModalJobConfigHandler(
        Namespace(
            job_config=__file__,
            plan=True,
            metrics_qualification_attempt="candidate",
            qualification_account="workspace",
            qualification_workspace="workspace",
            qualification_output=str(output),
            modal_profile="explicit",
            modal_environment="main",
            json=True,
        ),
        context,
    )
    assert handler.handle() == 0
    candidate = json.loads(capsys.readouterr().out)["data"][
        "metrics_qualification_candidate"
    ]
    assert output.read_bytes() == candidate["canonical_json"].encode()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == candidate["sha256"]
    assert candidate["execution_authorized"] is False and events == []
    assert not (tmp_path / "state").exists()


def test_duplicate_pending_page_does_not_increase_proof_count(root):
    ledger, consumer = ledger_with_records(root)
    ledger.record(
        consumer,
        before_state="pending",
        after_state="pending",
        receipt_time="2026-10-12T10:00:01Z",
        requested_cursor=None,
    )
    assert ledger.finish(verified=True)["precompletion_numeric_records"] == 2


def test_later_canonical_conflict_invalidates_proof(root):
    initial = entries()
    altered = replace(metric_record(1, run=RUN), occurred_at="2026-10-13T10:00:00Z")
    source = Pages(
        [
            (initial, None),
            ([encode_training_progress_log(altered, expected_run=RUN)], None),
        ]
    )
    consumer = RunProgressConsumer(source, RUN, root)
    ledger = MetricsQualificationLedger(admission(), RUN, "fc-owned", root)
    consumer.poll()
    ledger.record(
        consumer,
        before_state="pending",
        after_state="pending",
        receipt_time="2026-10-12T10:00:00Z",
        requested_cursor=None,
    )
    consumer.poll()
    ledger.record(
        consumer,
        before_state="pending",
        after_state="pending",
        receipt_time="2026-10-12T10:00:01Z",
        requested_cursor=None,
    )
    consumer.close()
    ledger.compare_retained(consumer)
    assert consumer.status == "progress_sequence_conflict"
    assert ledger.finish(verified=True)["delivery_verdict"] == "evidence_invalid"
