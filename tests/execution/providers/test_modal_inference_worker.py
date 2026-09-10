"""Provider-free preparation of a real mounted Modal chat model target."""

from __future__ import annotations
import os
from pathlib import Path
import pytest
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from tests.execution.providers.modal_inference_worker_fixtures import (
    mounted_launch_case,
)
from tuner.execution.providers.modal.contracts import ArtifactMemberV1, ArtifactRole
from tuner.execution.providers.modal.inference_artifacts import (
    ModalMountedInferenceArtifactReader,
)
from tuner.execution.providers.modal.inference_preparation import (
    _validate_preparation_snapshot,
)
from tuner.execution.providers.modal.inference_worker import (
    ModalInferenceWorkerError,
    prepare_modal_chat_worker,
)
from tuner.execution.providers.modal import inference_worker as worker_module


@pytest.mark.parametrize("model_kind", ("full", "lora"))
def test_real_signed_launch_materializes_exact_mounted_target(
    monkeypatch, tmp_path, model_kind
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind=model_kind)
    target = prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    target.validate()
    assert target.retrieved.run.to_dict() == case.source["run"]
    assert tuple(item.to_dict() for item in target.retrieved.artifacts) == tuple(
        case.source["artifacts"]
    )
    assert target.retrieved.model_ref == case.workload["model_ref"]
    assert target.retrieved.model_revision == case.workload["model_revision"]
    assert target.retrieved.tokenizer_revision == case.workload["tokenizer_revision"]
    assert len(case.preparer.calls) == (1 if model_kind == "lora" else 0)


def test_invalid_launch_denies_before_filesystem_or_preparer(monkeypatch, tmp_path):
    case = mounted_launch_case(tmp_path, monkeypatch)
    calls = []

    class DuckPath:
        def is_absolute(self):
            calls.append("path")
            raise AssertionError

    class DuckPreparer:
        def prepare(self, *args, **kwargs):
            calls.append("prepare")
            raise AssertionError

    with pytest.raises(
        ModalInferenceWorkerError, match="^modal_inference_worker_invalid$"
    ):
        prepare_modal_chat_worker(
            case.envelope.argument_bytes + b" ",
            **(case.kwargs | {"destination": DuckPath(), "preparer": DuckPreparer()}),
        )
    assert calls == []


@pytest.mark.parametrize("alias", ("lexical", "symlink"))
def test_destination_cannot_overlap_or_be_a_symlink_to_mount(
    monkeypatch, tmp_path, alias
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    destination = case.roots["artifacts"] / "child"
    if alias == "symlink":
        destination = case.roots["destination"]
        destination.rmdir()
        os.symlink(case.roots["artifacts"], destination, target_is_directory=True)
    with pytest.raises(
        ModalInferenceWorkerError, match="^modal_inference_worker_invalid$"
    ):
        prepare_modal_chat_worker(
            case.envelope.argument_bytes, **(case.kwargs | {"destination": destination})
        )


@pytest.mark.parametrize("attack", ("corrupt", "truncate", "symlink"))
def test_mounted_artifact_attack_is_rejected_and_attempt_cleaned(
    monkeypatch, tmp_path, attack
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    member = case.source["members"][0]
    path = case.roots["artifacts"] / member["path"]
    if attack == "symlink":
        path.unlink()
        path.symlink_to(case.roots["artifacts"] / case.source["members"][1]["path"])
    else:
        value = path.read_bytes()
        path.write_bytes(value[:-1] if attack == "truncate" else value + b"x")
    with pytest.raises(
        ModalInferenceWorkerError, match="^modal_inference_worker_invalid$"
    ):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert not list(case.roots["destination"].glob(".retrieved-*"))


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
def test_admission_control_flow_precedes_filesystem(monkeypatch, tmp_path, control):
    case = mounted_launch_case(tmp_path, monkeypatch)

    class Verifier:
        def verify(self, purpose, payload, tag, key_ref):
            raise control()

    with pytest.raises(control):
        prepare_modal_chat_worker(
            case.envelope.argument_bytes, **(case.kwargs | {"verifier": Verifier()})
        )


def test_all_worker_owned_root_descriptors_close(monkeypatch, tmp_path):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    real_open_directory = worker_module._open_directory
    real_close = os.close
    opened = []
    closed = []

    def tracked_open_directory(path):
        descriptor, identity = real_open_directory(path)
        opened.append(descriptor)
        return descriptor, identity

    def tracked_close(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(worker_module, "_open_directory", tracked_open_directory)
    monkeypatch.setattr(os, "close", tracked_close)
    prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert len(opened) == 4 and set(opened).issubset(closed)


@pytest.mark.parametrize("mutation", ("expectation", "model"))
def test_materializer_callback_mutation_stops_before_base_preparer(
    monkeypatch, tmp_path, mutation
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="lora")
    real = worker_module._materialize_admitted_sft_model

    def changed(**kwargs):
        retrieved = real(**kwargs)
        if mutation == "expectation":
            object.__setattr__(case.kwargs["expectation"], "artifact_root", "/other")
        else:
            object.__setattr__(retrieved, "model_revision", "0" * 40)
        return retrieved

    monkeypatch.setattr(worker_module, "_materialize_admitted_sft_model", changed)
    with pytest.raises(
        ModalInferenceWorkerError, match="^modal_inference_worker_invalid$"
    ):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert case.preparer.calls == []


def test_serving_target_callback_cannot_mutate_authenticated_retrieved_model(
    monkeypatch, tmp_path
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    real = worker_module.prepare_serving_target

    def changed(retrieved, preparer):
        target = real(retrieved, preparer)
        object.__setattr__(retrieved, "model_revision", "0" * 40)
        return target

    monkeypatch.setattr(worker_module, "prepare_serving_target", changed)
    with pytest.raises(
        ModalInferenceWorkerError, match="^modal_inference_worker_invalid$"
    ):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert case.preparer.calls == []


@pytest.mark.parametrize("root_name", ("control", "cache"))
@pytest.mark.parametrize("phase", ("materializer", "preparer"))
def test_root_replacement_during_callbacks_is_detected(
    monkeypatch, tmp_path, root_name, phase
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="lora")
    root = case.roots[root_name]
    original_identity = (root.stat().st_dev, root.stat().st_ino)

    def replace_root():
        root.rename(root.with_name(root.name + "-old"))
        root.mkdir()

    if phase == "materializer":
        real = worker_module._materialize_admitted_sft_model

        def materialize(**kwargs):
            result = real(**kwargs)
            replace_root()
            return result

        monkeypatch.setattr(
            worker_module, "_materialize_admitted_sft_model", materialize
        )
    else:
        real = case.preparer.prepare

        def prepare(*, model_ref, revision):
            result = real(model_ref=model_ref, revision=revision)
            replace_root()
            return result

        case.preparer.prepare = prepare
    with pytest.raises(ModalInferenceWorkerError):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert len(case.preparer.calls) == (1 if phase == "preparer" else 0)
    assert root.with_name(root.name + "-old").is_dir()
    assert (root.stat().st_dev, root.stat().st_ino) != original_identity


def test_second_real_materialization_cannot_replace_authenticated_result(
    monkeypatch, tmp_path
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    materialize = worker_module._materialize_admitted_sft_model
    prepare = worker_module.prepare_serving_target
    completed = []

    def substitute(retrieved, preparer):
        snapshot = _validate_preparation_snapshot(case.envelope.preparation_snapshot)
        source = snapshot["chat_input"]["source"]
        artifact_fd = os.open(case.roots["artifacts"], os.O_RDONLY | os.O_DIRECTORY)
        destination_fd = os.open(
            case.roots["destination"], os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            run = TrainingRunRef.from_dict(source["run"])
            artifacts = tuple(
                VerifiedArtifact.from_dict(value) for value in source["artifacts"]
            )
            members = tuple(
                ArtifactMemberV1(
                    ArtifactRole(value["role"]),
                    value["path"],
                    value["size"],
                    value["sha256"],
                    value["provider_entry_id"],
                )
                for value in source["members"]
            )
            reader = ModalMountedInferenceArtifactReader(
                root=case.roots["artifacts"],
                root_fd=artifact_fd,
                run=run,
                artifact_volume_id=source["artifact_volume_id"],
                effect_id=source["effect_id"],
                artifacts=artifacts,
                members=members,
            )
            other = materialize(
                run=run,
                artifacts=artifacts,
                root=case.roots["destination"],
                root_fd=destination_fd,
                read_artifact=reader.read_artifact,
            )
        finally:
            os.close(destination_fd)
            os.close(artifact_fd)
        assert other is not retrieved
        target = prepare(other, None)
        target.validate()
        completed.append(target)
        return target

    monkeypatch.setattr(worker_module, "prepare_serving_target", substitute)
    with pytest.raises(ModalInferenceWorkerError):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert len(completed) == 1


@pytest.mark.parametrize("phase", ("materializer", "target"))
def test_duck_callback_results_are_rejected_without_method_access(
    monkeypatch, tmp_path, phase
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    calls = []

    class Duck:
        def validate(self):
            calls.append("validate")
            raise AssertionError

    monkeypatch.setattr(
        worker_module,
        (
            "_materialize_admitted_sft_model"
            if phase == "materializer"
            else "prepare_serving_target"
        ),
        lambda *args, **kwargs: Duck(),
    )
    with pytest.raises(ModalInferenceWorkerError):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert calls == []


@pytest.mark.parametrize("phase", ("materializer", "preparer"))
@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
def test_control_flow_after_all_root_opens_closes_every_owned_fd(
    monkeypatch, tmp_path, phase, control
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="lora")
    opened = []
    closed = []
    real_open = worker_module._open_directory
    real_close = os.close

    def opened_dir(path):
        value = real_open(path)
        opened.append(value[0])
        return value

    def closed_fd(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(worker_module, "_open_directory", opened_dir)
    monkeypatch.setattr(os, "close", closed_fd)
    if phase == "materializer":
        monkeypatch.setattr(
            worker_module,
            "_materialize_admitted_sft_model",
            lambda **kwargs: (_ for _ in ()).throw(control()),
        )
    else:
        case.preparer.prepare = lambda **kwargs: (_ for _ in ()).throw(control())
    with pytest.raises(control):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert len(opened) == 4 and set(opened).issubset(closed)


def test_duplicate_physical_root_identity_denies_before_materializer(
    monkeypatch, tmp_path
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    real = worker_module._open_directory
    descriptors = []
    calls = []

    def opened(path):
        if len(descriptors) == 1:
            fd = os.dup(descriptors[0])
            descriptors.append(fd)
            info = os.fstat(fd)
            return fd, (info.st_dev, info.st_ino)
        value = real(path)
        descriptors.append(value[0])
        return value

    monkeypatch.setattr(worker_module, "_open_directory", opened)
    monkeypatch.setattr(
        worker_module,
        "_materialize_admitted_sft_model",
        lambda **kwargs: calls.append("materialize"),
    )
    with pytest.raises(ModalInferenceWorkerError):
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert calls == []
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("phase", ("materializer", "preparer"))
def test_same_class_collaborator_error_is_redacted(monkeypatch, tmp_path, phase):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="lora")

    def private(*args, **kwargs):
        raise ModalInferenceWorkerError("private-collaborator-detail")

    if phase == "materializer":
        monkeypatch.setattr(worker_module, "_materialize_admitted_sft_model", private)
    else:
        case.preparer.prepare = private
    with pytest.raises(
        ModalInferenceWorkerError, match="^modal_inference_worker_invalid$"
    ) as caught:
        prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert caught.value.__cause__ is None
