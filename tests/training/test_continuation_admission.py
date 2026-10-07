"""Public continuation intent and read-only source evidence checks."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunVerification, RunsAPI
from synaptic_tuner.api.v1.training_input import (
    TrainingContinuationInputV1, TrainingInputV1,
)
from tests.contract.test_public_training_input_v1 import _document
from tests.training.test_modal_recipe import _load
from tuner.training.continuation_contracts import verify_continuation_source


_DATA = b"verified adapter bytes"
_SHA = hashlib.sha256(_DATA).hexdigest()


def _intent(mode: str = "adapter_warm_start") -> dict[str, object]:
    return {
        "schema_version": "synaptic-training-continuation/v1",
        "mode": mode,
        "parent_run": {"run_id": "parent", "project_ref": "project"},
        "artifact": {"role": "final_model" if mode == "adapter_warm_start" else "trainer_state",
                     "sha256": _SHA, "size_bytes": len(_DATA)},
        "schedule_policy": "reset" if mode == "adapter_warm_start" else "preserve",
        "schedule_transition_digest": None,
    }


def test_absent_continuation_keeps_legacy_public_canonical_bytes() -> None:
    legacy = TrainingInputV1.from_dict(_document())
    assert "continuation" not in legacy.to_dict()
    assert TrainingInputV1.from_json(legacy.canonical_json()) == legacy
    present_null = _document()
    present_null["continuation"] = None
    with pytest.raises(TypeError, match="continuation must be an object"):
        TrainingInputV1.from_dict(present_null)


@pytest.mark.parametrize("mode", ["adapter_warm_start", "full_state_resume"])
def test_public_intent_round_trip_and_role_guard(mode: str) -> None:
    document = _document()
    document["continuation"] = _intent(mode)
    parsed = TrainingInputV1.from_dict(document)
    assert TrainingInputV1.from_json(parsed.canonical_json()) == parsed
    bad = deepcopy(document)
    bad["continuation"]["artifact"]["role"] = "tokenizer"
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(bad)


def test_extension_requires_explicit_transition_and_null_is_not_an_omission() -> None:
    document = _document()
    intent = _intent("full_state_resume")
    intent["schedule_policy"] = "explicit_extension"
    document["continuation"] = intent
    with pytest.raises(ValueError, match="transition"):
        TrainingInputV1.from_dict(document)
    intent["schedule_transition_digest"] = "a" * 64
    assert TrainingInputV1.from_dict(document).continuation.schedule_policy == "explicit_extension"


@pytest.mark.parametrize("field", ["schema_version", "mode", "schedule_policy"])
def test_direct_constructor_rejects_string_subclasses_and_equality_spoofs(field: str) -> None:
    class StrSubclass(str):
        pass

    class EqualToEverything:
        def __eq__(self, _other):
            return True

    intent = TrainingContinuationInputV1.from_dict(_intent())
    for invalid in (StrSubclass(getattr(intent, field)), EqualToEverything()):
        with pytest.raises(TypeError, match="exact strings"):
            replace(intent, **{field: invalid})


def test_modal_recipe_carries_intent_but_execution_is_explicitly_unavailable(monkeypatch) -> None:
    recipe = _load(monkeypatch, lambda data: data.update(continuation=_intent()))
    assert recipe.continuation is not None
    public = recipe.training_input("prepared://sha256/" + recipe.dataset_digest)
    assert public.continuation == recipe.continuation
    from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity

    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest, recipe.dataset_digest,
        "a" * 64, 123, "syntunia-sft-row/v2",
    )
    config = recipe.packaged_config(identity)
    assert config.to_dict()["continuation"] == recipe.continuation.to_dict()
    from tuner.training.packaged_compilation import compile_packaged_sft_workload

    with pytest.raises(ValueError, match="continuation execution unavailable"):
        compile_packaged_sft_workload(resolved_config=config)


class _Stream:
    def __init__(self, run, artifact, maximum_bytes, chunks):
        self.run, self.artifact, self.maximum_bytes = run, artifact, maximum_bytes
        self._chunks = chunks

    def iter_bytes(self):
        yield from self._chunks


def _runs(chunks=(_DATA,), *, verified=True, descriptor=None):
    parent = TrainingRunRef("parent", "project")
    artifact = descriptor or VerifiedArtifact("final_model", _SHA, len(_DATA))
    operations = SimpleNamespace(
        verify=lambda run: RunVerification(run, verified, "2026-10-03T00:00:00Z"),
        artifacts=lambda request: _Stream(parent, artifact, request.maximum_bytes, chunks),
    )
    return RunsAPI(operations)


def test_source_admission_reads_exact_verified_public_stream() -> None:
    intent = TrainingContinuationInputV1.from_dict(_intent())
    receipt = verify_continuation_source(_runs(), intent, project_ref="project",
                                         maximum_bytes=len(_DATA))
    assert receipt.artifact.sha256 == intent.artifact_sha256
    assert receipt.run.run_id == intent.parent_run_id


def test_source_admission_rejects_unverified_changed_or_corrupt_stream() -> None:
    intent = TrainingContinuationInputV1.from_dict(_intent())
    with pytest.raises(ValueError, match="not verified"):
        verify_continuation_source(_runs(verified=False), intent, project_ref="project", maximum_bytes=100)
    with pytest.raises(ValueError, match="descriptor changed"):
        verify_continuation_source(_runs(descriptor=VerifiedArtifact("final_model", "b" * 64, len(_DATA))), intent, project_ref="project", maximum_bytes=100)
    with pytest.raises(ValueError, match="differs from its descriptor"):
        verify_continuation_source(_runs(chunks=(b"wrong",)), intent, project_ref="project", maximum_bytes=100)
    with pytest.raises(ValueError, match="exceeds its read bound"):
        verify_continuation_source(_runs(), intent, project_ref="project", maximum_bytes=1)
    with pytest.raises(ValueError, match="child project"):
        verify_continuation_source(_runs(), intent, project_ref="other", maximum_bytes=100)
