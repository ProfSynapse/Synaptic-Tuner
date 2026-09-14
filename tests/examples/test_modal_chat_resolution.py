from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil

import pytest

from examples.modal_chat.resolution import (
    ModalChatResolutionError,
    ModalChatRichTrainingResolver,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.providers.modal.config import ModalRuntimeLockV1
from tuner.execution.providers.modal.deployment_identity import modal_function_name
from tuner.execution.providers.modal.resolution import (
    ModalDeploymentSelectionV1,
    ModalExecutionSourceResolutionV1,
)
from tuner.project.context import ProjectContext
from tuner.project.git_verification import GitCliLocalSourceInspector
from tuner.training.contracts import CanonicalDocument, TrainingRequest
from tuner.training import default_recipe_registry
from tuner.training.coordinator_material import derive_coordinator_material
from tuner.training.methods.sft import SFT_ENTRYPOINT
from tuner.training.service import TrainingService

from tests.execution.providers.test_modal_source_resolution import _finalizer, _source

ROOT = Path(__file__).resolve().parents[2]


def _document(
    dataset_ref: str = "project://data/train.jsonl",
    model_ref: str = "organization/model",
) -> dict[str, object]:
    return {
        "schema_version": "synaptic-training-input/v1",
        "method": "sft",
        "model": {
            "ref": model_ref,
            "revision": "c" * 40,
            "tokenizer_revision": "d" * 40,
        },
        "dataset": {"ref": dataset_ref},
        "hyperparameters": {
            "schema_version": "synaptic-sft-hyperparameters/v1",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0002,
            "duration": {"max_steps": 10, "num_epochs": None},
            "max_seq_length": 2048,
            "seed": 42,
            "save_steps": 5,
            "save_total_limit": 2,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["k_proj", "q_proj", "v_proj"],
            "use_dora": False,
            "use_rslora": True,
            "init_lora_weights": True,
            "split_dataset": False,
        },
        "artifacts": {
            "required_kinds": ["final_model", "training_lineage"],
            "retain_checkpoints": True,
        },
    }


def _deployment() -> ModalDeploymentSelectionV1:
    lock = ModalRuntimeLockV1.packaged()
    deployment_ref = "modal-deployment-" + "1" * 32
    return ModalDeploymentSelectionV1(
        account_ref="acct",
        workspace_ref="workspace",
        environment_ref="env",
        client_ref="client",
        app_name="synaptic-training-v1",
        function_name=modal_function_name(deployment_ref),
        deployment_ref=deployment_ref,
        image_digest=lock.image_digest,
        dependency_lock_digest=lock.locked_digest("dependency_lock"),
        wrapper_digest=lock.locked_digest("deployment_wrapper"),
        runtime_digest=lock.locked_digest("sft_runtime"),
        python_version=lock.python_version,
        python_executable=lock.python_executable,
        python_executable_digest=lock.python_executable_digest,
        secret_requirements_digest="6" * 64,
        provider_runtime_requirements_digest="7" * 64,
        runtime_environment={"LANG": "C.UTF-8", "PATH": "/opt/conda/bin"},
    )


def _context(tmp_path: Path) -> ProjectContext:
    project = tmp_path / "consumer"
    engine = project / "vendor" / "engine"
    (project / "data").mkdir(parents=True)
    (project / "data" / "train.jsonl").write_text("fixture bytes\n", encoding="utf-8")
    destination = engine / SFT_ENTRYPOINT
    destination.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / SFT_ENTRYPOINT, destination)
    return ProjectContext.host(engine_root=engine, project_root=project)


@pytest.mark.parametrize("model_ref", ["organization/model", "organization/modèle"])
def test_training_service_resolves_and_derives_coordinator_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_ref: str
) -> None:
    context = _context(tmp_path)
    inspected = _source()
    inspections: list[ProjectContext] = []

    def inspect_fixture(
        _self: GitCliLocalSourceInspector, *, context: ProjectContext
    ) -> object:
        inspections.append(context)
        return inspected

    monkeypatch.setattr(GitCliLocalSourceInspector, "inspect", inspect_fixture)
    deployment = _deployment()
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=True,
        deployment=deployment,
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    recipes = default_recipe_registry()
    service = TrainingService(context=context, resolver=resolver, recipes=recipes)
    request = service.load(
        CanonicalDocument.from_mapping(_document(model_ref=model_ref))
    )
    result = service.resolve(request)
    config = result.resolved_config.to_dict()

    assert inspections == [context]
    assert config["schema_version"] == "synaptic-sft-config/v1"
    assert config["dataset"]["revision"] == "a" * 40
    assert config["dataset"]["content_digest"]
    assert config["model"]["revision"] == "c" * 40
    assert config["sft"]["max_steps"] == 10
    assert "duration" not in config["sft"]
    assert result.execution_source.source_evidence.source_lock_binding.to_dict() == (
        result.execution_context.to_dict()["source_lock_binding"]
    )
    assert result.runtime.image == ModalRuntimeLockV1.packaged().registry_reference
    assert result.resources.accelerator == "A10"
    material = derive_coordinator_material(
        result,
        recipes,
        request_id="request-1",
        project_ref="project",
        run_id="run-1",
    )
    assert material.run_id == "run-1"
    assert material.planning_request.project_ref == "project"


def test_rejects_request_that_does_not_name_configured_consumer_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector,
        "inspect",
        lambda _self, *, context: inspected,
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )

    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(
                CanonicalDocument.from_mapping(_document("project://data/other.jsonl"))
            ),
            context=context,
        )


def test_constructor_rejects_deployment_that_is_not_packaged_runtime() -> None:
    deployment = _deployment()
    value = deployment.to_dict()
    value["image_digest"] = "9" * 64

    with pytest.raises(ValueError, match="packaged runtime lock"):
        ModalChatRichTrainingResolver(
            context=ProjectContext.host(
                engine_root=ROOT,
                project_root=ROOT.parent,
            ),
            run=TrainingRunRef("run-1", "project"),
            created_at="2026-08-25T12:00:00Z",
            dataset_project_path=Path("data/train.jsonl"),
            load_in_4bit=False,
            deployment=ModalDeploymentSelectionV1.from_dict(value),
            source_finalizer=_finalizer(_source()),
            audience_ref="project/run-1",
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value["model"].update(revision="main"),
        lambda value: value["model"].update(tokenizer_revision="latest"),
        lambda value: value["dataset"].update(revision="a" * 40),
    ],
)
def test_rejects_mutable_or_unsupported_input_commitments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change
) -> None:
    context = _context(tmp_path)
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    document = _document()
    change(document)

    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(document)), context=context
        )


def test_rejects_symlink_dataset_and_different_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    request = TrainingRequest(CanonicalDocument.from_mapping(_document()))
    other = ProjectContext.host(
        engine_root=context.engine_root, project_root=context.project_root
    )
    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(request, context=other)

    dataset = context.project_root / "data" / "train.jsonl"
    target = context.project_root / "data" / "target.jsonl"
    dataset.rename(target)
    dataset.symlink_to(target)
    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(request, context=context)


def test_rejects_finalizer_result_with_substituted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    inner = _finalizer(inspected)

    class SubstitutingFinalizerFixture:
        def finalize(self, source_lock, **kwargs):
            result = inner.finalize(source_lock, **kwargs)
            return ModalExecutionSourceResolutionV1(
                replace(result.execution_source, run_id="run-substituted"),
                result.deployment,
            )

    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=SubstitutingFinalizerFixture(),
        audience_ref="project/run-1",
    )
    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(_document())),
            context=context,
        )


def test_rejects_nonregular_dataset_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    dataset = context.project_root / "data" / "train.jsonl"
    dataset.unlink()
    os.mkfifo(dataset)
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(_document())),
            context=context,
        )


def test_nested_dataset_checks_only_absolute_consumer_ancestors(tmp_path, monkeypatch):
    from examples.modal_chat import resolution
    import hashlib

    root = tmp_path / "consumer"
    parent = root / "data" / "nested"
    parent.mkdir(parents=True)
    (parent / "train.jsonl").write_bytes(b"dataset")
    actual = resolution._directory_identity
    visited = []

    def inspect(path):
        assert path.is_absolute()
        assert path == root or root in path.parents
        visited.append(path)
        return actual(path)

    monkeypatch.setattr(resolution, "_directory_identity", inspect)
    assert resolution._regular_digest(root, Path("data/nested/train.jsonl"), 100) == (
        7,
        hashlib.sha256(b"dataset").hexdigest(),
    )
    assert set(visited) == {root, root / "data", parent}


def test_rejects_dataset_changed_while_its_bytes_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    dataset = context.project_root / "data" / "train.jsonl"
    dataset.write_bytes(b"x" * (2 * 1024 * 1024))
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    actual_read = os.read
    calls = 0

    def changing_read(descriptor: int, count: int) -> bytes:
        nonlocal calls
        value = actual_read(descriptor, count)
        calls += 1
        if calls == 1:
            with dataset.open("ab") as stream:
                stream.write(b"changed")
        return value

    monkeypatch.setattr(os, "read", changing_read)
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=Path("data/train.jsonl"),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(_document())),
            context=context,
        )
