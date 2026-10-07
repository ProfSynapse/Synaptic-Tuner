"""Provider-free structural checks; artifact authenticity is a separate port."""

from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from tuner.training.continuation_contracts import (
    ContinuationMode, ContinuationRequestV1, ParentTrainingStateV1,
    SchedulePolicy,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64
_MEMBERS = tuple((name, _A) for name in (
    "data_cursor", "model", "optimizer", "rng", "scheduler",
))


def _parent(role: str = "trainer_state") -> ParentTrainingStateV1:
    return ParentTrainingStateV1(
        TrainingRunRef("parent", "project"), VerifiedArtifact(role, _A, 123),
        _B, _C, _D[:40], _E, _F, 4, 10,
        _MEMBERS if role == "trainer_state" else (),
    )


def _request(mode: ContinuationMode = ContinuationMode.FULL_STATE_RESUME,
             parent: ParentTrainingStateV1 | None = None) -> ContinuationRequestV1:
    source = parent or _parent("final_model" if mode is ContinuationMode.ADAPTER_WARM_START else "trainer_state")
    return ContinuationRequestV1(
        mode, source, TrainingRunRef("child", "project"), source.dataset_digest,
        source.training_invariants_digest, source.base_model_revision,
        source.adapter_layout_digest, source.runtime_digest, 10,
        SchedulePolicy.RESET if mode is ContinuationMode.ADAPTER_WARM_START else SchedulePolicy.PRESERVE,
        mode is ContinuationMode.ADAPTER_WARM_START,
    )


def test_full_state_resume_preserves_dataset_and_schedule() -> None:
    request = _request()
    assert request.child_total_steps == request.parent.planned_total_steps
    with pytest.raises(ValueError, match="dataset or training identity"):
        replace(request, child_dataset_digest=_A)
    with pytest.raises(ValueError, match="dataset or training identity"):
        replace(request, child_training_invariants_digest=_A)
    with pytest.raises(ValueError, match="dataset or training identity"):
        replace(request, child_runtime_digest=_A)
    with pytest.raises(ValueError, match="planned horizon"):
        replace(request, child_total_steps=11)


def test_explicit_longer_horizon_requires_scheduler_transition() -> None:
    request = _request()
    extended = replace(request, child_total_steps=12,
                       schedule_policy=SchedulePolicy.EXPLICIT_EXTENSION,
                       schedule_transition_digest=_A)
    assert extended.child_total_steps == 12
    with pytest.raises(ValueError, match="longer horizon and transition"):
        replace(extended, schedule_transition_digest=None)
    with pytest.raises(ValueError, match="longer horizon and transition"):
        replace(extended, child_total_steps=10)


def test_full_state_rejects_adapter_only_and_missing_state() -> None:
    with pytest.raises(ValueError, match="requires trainer state"):
        _request(parent=_parent("final_model"))
    with pytest.raises(ValueError, match="requires model, optimizer"):
        replace(_parent(), state_member_digests=_MEMBERS[:-1])
    with pytest.raises(ValueError, match="no remaining steps"):
        replace(_request(), child_total_steps=4)


def test_warm_start_allows_new_data_but_resets_optimizer_and_schedule() -> None:
    request = _request(ContinuationMode.ADAPTER_WARM_START)
    changed = replace(request, child_dataset_digest=_A,
                      child_training_invariants_digest=_A, child_runtime_digest=_A)
    assert changed.child_dataset_digest != changed.parent.dataset_digest
    with pytest.raises(ValueError, match="optimizer reset"):
        replace(request, optimizer_reset=False)
    with pytest.raises(ValueError, match="fresh schedule"):
        replace(request, schedule_policy=SchedulePolicy.PRESERVE)


def test_modes_cannot_cross_artifact_roles_or_model_identity() -> None:
    with pytest.raises(ValueError, match="final adapter"):
        _request(ContinuationMode.ADAPTER_WARM_START, _parent())
    with pytest.raises(ValueError, match="base model revision"):
        replace(_request(), child_base_model_revision=_A)
    with pytest.raises(ValueError, match="adapter layout"):
        replace(_request(), child_adapter_layout_digest=_A)


def test_parent_and_child_run_must_be_distinct_and_same_project() -> None:
    request = _request()
    with pytest.raises(ValueError, match="distinct child run"):
        replace(request, child_run=request.parent.run)
    with pytest.raises(ValueError, match="cross-project"):
        replace(request, child_run=TrainingRunRef("child", "other"))


def test_state_members_are_exact_and_canonical() -> None:
    with pytest.raises(ValueError, match="sorted, unique"):
        replace(_parent(), state_member_digests=tuple(reversed(_MEMBERS)))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_parent(), dataset_digest="A" * 64)
    with pytest.raises(ValueError, match="final model is not a full trainer state"):
        replace(_parent("final_model"), state_member_digests=_MEMBERS)
