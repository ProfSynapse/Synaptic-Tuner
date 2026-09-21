from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType

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
from tuner.training.methods.sft import compile_sft_workload
from tuner.dataset_prep import (
    DatasetPublicationUncertainV1,
    DatasetSemanticIdentityV1,
    VerifiedPreparedDatasetV1,
    prepare_dataset_v2,
)
from tuner.training.service import TrainingService

from tests.execution.providers.test_modal_source_resolution import _finalizer, _source
from tests.dataset_prep.test_dataset_prep_v1 import (
    _bundle as _prepared_bundle,
    _config as _prepared_config,
    _prepare_reconciled,
)
from tests.dataset_prep.test_context_messages_v2 import _config as _v2_config

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


def _git(project: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _initialize_git_policy(context: ProjectContext, *, ignored: bool) -> None:
    _git(context.project_root, "init", "--quiet")
    _git(context.project_root, "config", "user.email", "fixture@example.invalid")
    _git(context.project_root, "config", "user.name", "Fixture")
    if ignored:
        (context.project_root / ".gitignore").write_text(
            "private/\n", encoding="utf-8", newline="\n"
        )
        policy_file = ".gitignore"
    else:
        (context.project_root / ".git-policy-fixture").write_text(
            "fixture\n", encoding="utf-8", newline="\n"
        )
        policy_file = ".git-policy-fixture"
    _git(context.project_root, "add", "--", policy_file)
    _git(context.project_root, "commit", "--quiet", "-m", "policy fixture")


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
    assert resolver.private_dataset_bytes is None


def test_prepared_dataset_resolves_verified_identity_and_retains_only_private_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=True)
    structure, bundle = _prepared_bundle(
        tmp_path / "prepared-source", ("PRIVATE DATASET SENTINEL",)
    )
    prepared = _prepare_reconciled(
        _prepared_config(bundle, structure), context.project_root / "private"
    )
    relative = prepared.path.relative_to(context.project_root)
    identity = prepared.semantic_identity
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    inner = _finalizer(inspected)

    class CapturingFinalizer:
        source_lock = None

        def finalize(self, source_lock, **kwargs):
            self.source_lock = source_lock
            return inner.finalize(source_lock, **kwargs)

    finalizer = CapturingFinalizer()
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=relative,
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=finalizer,
        audience_ref="project/run-1",
    )
    dataset_ref = f"prepared://sha256/{identity.dataset_digest}"
    result = resolver.resolve(
        TrainingRequest(CanonicalDocument.from_mapping(_document(dataset_ref))),
        context=context,
    )

    dataset = result.resolved_config.to_dict()["dataset"]
    assert dataset == {
        "content_digest": identity.dataset_sha256,
        "format": "syntunia-sft-row/v1",
        "ref": dataset_ref,
        "revision": identity.dataset_digest,
        "size_bytes": identity.dataset_bytes,
    }
    assert resolver.private_dataset_bytes == (
        prepared.path / "dataset.jsonl"
    ).read_bytes()
    public = result.resolved_config.canonical_json + json.dumps(
        finalizer.source_lock.to_dict(), sort_keys=True
    )
    assert "PRIVATE DATASET SENTINEL" not in public
    assert relative.as_posix() not in public


def test_authoritative_v2_prepared_dataset_reaches_provider_neutral_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=True)
    _bundle, _ids, _documents, config = _v2_config(tmp_path / "v2-source")
    try:
        prepared = prepare_dataset_v2(config, context.project_root / "private")
    except DatasetPublicationUncertainV1:
        prepared = prepare_dataset_v2(config, context.project_root / "private")
    relative = prepared.path.relative_to(context.project_root)
    identity = prepared.semantic_identity
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=relative,
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    dataset_ref = f"prepared://sha256/{identity.dataset_digest}"
    document = _document(dataset_ref)
    document["hyperparameters"].update(  # type: ignore[union-attr]
        {
            "max_seq_length": 32768,
            "dataset_format": "messages",
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "use_preassigned_splits": True,
            "prompt_render": "prompt_completion",
            "packing": False,
            "require_memory_efficient_loss": True,
        }
    )
    result = resolver.resolve(
        TrainingRequest(CanonicalDocument.from_mapping(document)), context=context
    )

    resolved = result.resolved_config.to_dict()
    assert resolved["dataset"]["format"] == "syntunia-sft-row/v2"
    assert resolved["sft"]["dataset_format"] == "messages"
    assert resolved["sft"]["packing"] is False
    assert resolved["sft"]["completion_only_loss"] is True
    assert resolved["sft"]["assistant_only_loss"] is False
    assert resolved["sft"]["use_preassigned_splits"] is True
    assert resolved["sft"]["prompt_render"] == "prompt_completion"
    assert resolved["sft"]["require_memory_efficient_loss"] is True
    assert resolver.private_dataset_bytes == (prepared.path / "dataset.jsonl").read_bytes()
    workload = compile_sft_workload(
        resolved_config=result.resolved_config,
        execution_source=result.execution_source,
    )
    assert workload.document["configuration"]["document"]["dataset"]["format"] == (
        "syntunia-sft-row/v2"
    )


def test_large_prepared_dataset_retains_source_instead_of_inline_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import examples.modal_chat.resolution as resolution_module

    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=True)
    dataset_digest = "a" * 64
    relative = Path("private") / f"dataset-{dataset_digest}"
    prepared_path = context.project_root / relative
    prepared_path.mkdir(parents=True)
    (prepared_path / "manifest.json").write_text("{}", encoding="utf-8")
    (prepared_path / "dataset.jsonl").write_text("placeholder\n", encoding="utf-8")
    semantic = DatasetSemanticIdentityV1(
        f"dataset-{dataset_digest}",
        dataset_digest,
        1,
        2 * 1024 * 1024 + 1,
        "b" * 64,
        "c" * 64,
        MappingProxyType({"train": 1}),
    )
    verified = VerifiedPreparedDatasetV1(prepared_path, semantic)
    monkeypatch.setattr(
        resolution_module, "verify_prepared_dataset_v1", lambda path: verified,
    )
    monkeypatch.setattr(
        resolution_module,
        "snapshot_prepared_dataset_v1",
        lambda path: (_ for _ in ()).throw(AssertionError("must not snapshot inline")),
    )
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected,
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=relative,
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    result = resolver.resolve(
        TrainingRequest(CanonicalDocument.from_mapping(
            _document(f"prepared://sha256/{dataset_digest}")
        )),
        context=context,
    )
    assert result.resolved_config.to_dict()["dataset"]["size_bytes"] == semantic.dataset_bytes
    assert resolver.private_dataset_bytes is None
    assert resolver.prepared_input_source is not None
    assert resolver.prepared_input_source.identity.content_digest == semantic.dataset_sha256


def test_authoritative_v2_prepared_dataset_rejects_legacy_sft_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=True)
    _bundle, _ids, _documents, config = _v2_config(tmp_path / "v2-source")
    try:
        prepared = prepare_dataset_v2(config, context.project_root / "private")
    except DatasetPublicationUncertainV1:
        prepared = prepare_dataset_v2(config, context.project_root / "private")
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=prepared.path.relative_to(context.project_root),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    dataset_ref = f"prepared://sha256/{prepared.semantic_identity.dataset_digest}"

    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(_document(dataset_ref))),
            context=context,
        )
    assert resolver.private_dataset_bytes is None


def test_prepared_dataset_reference_must_match_verified_semantic_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=True)
    structure, bundle = _prepared_bundle(tmp_path / "prepared-source")
    prepared = _prepare_reconciled(
        _prepared_config(bundle, structure), context.project_root / "private"
    )
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=prepared.path.relative_to(context.project_root),
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )

    with pytest.raises(ModalChatResolutionError, match="modal_chat_resolution_invalid"):
        resolver.resolve(
            TrainingRequest(
                CanonicalDocument.from_mapping(
                    _document("prepared://sha256/" + "0" * 64)
                )
            ),
            context=context,
        )
    assert resolver.private_dataset_bytes is None


@pytest.mark.parametrize("git_state", ["tracked", "staged", "unignored"])
def test_prepared_dataset_requires_untracked_ignored_private_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_state: str
) -> None:
    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=git_state != "unignored")
    structure, bundle = _prepared_bundle(tmp_path / "prepared-source")
    prepared = _prepare_reconciled(
        _prepared_config(bundle, structure), context.project_root / "private"
    )
    relative = prepared.path.relative_to(context.project_root)
    if git_state in {"tracked", "staged"}:
        _git(context.project_root, "add", "-f", "--", relative.as_posix())
    if git_state == "tracked":
        _git(context.project_root, "commit", "--quiet", "-m", "fixture")
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=relative,
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    dataset_ref = "prepared://sha256/" + prepared.semantic_identity.dataset_digest

    with pytest.raises(ModalChatResolutionError) as raised:
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(_document(dataset_ref))),
            context=context,
        )
    assert raised.value.args == ("modal_chat_resolution_invalid",)
    assert str(prepared.path) not in repr(raised.value)
    assert resolver.private_dataset_bytes is None


def test_prepared_dataset_mutation_after_git_policy_check_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.modal_chat import resolution as resolution_module

    context = _context(tmp_path)
    _initialize_git_policy(context, ignored=True)
    structure, bundle = _prepared_bundle(
        tmp_path / "prepared-source", ("PRIVATE MUTATION SENTINEL",)
    )
    prepared = _prepare_reconciled(
        _prepared_config(bundle, structure), context.project_root / "private"
    )
    relative = prepared.path.relative_to(context.project_root)
    inspected = _source()
    monkeypatch.setattr(
        GitCliLocalSourceInspector, "inspect", lambda _self, *, context: inspected
    )
    actual_snapshot = resolution_module.snapshot_prepared_dataset_v1

    def mutate_then_snapshot(path: Path):
        dataset = path / "dataset.jsonl"
        dataset.chmod(0o600)
        dataset.write_bytes(b"PRIVATE POST-CHECK SUBSTITUTION\n")
        return actual_snapshot(path)

    monkeypatch.setattr(
        resolution_module, "snapshot_prepared_dataset_v1", mutate_then_snapshot
    )
    resolver = ModalChatRichTrainingResolver(
        context=context,
        run=TrainingRunRef("run-1", "project"),
        created_at="2026-08-25T12:00:00Z",
        dataset_project_path=relative,
        load_in_4bit=False,
        deployment=_deployment(),
        source_finalizer=_finalizer(inspected),
        audience_ref="project/run-1",
    )
    dataset_ref = "prepared://sha256/" + prepared.semantic_identity.dataset_digest

    with pytest.raises(ModalChatResolutionError) as raised:
        resolver.resolve(
            TrainingRequest(CanonicalDocument.from_mapping(_document(dataset_ref))),
            context=context,
        )
    assert raised.value.args == ("modal_chat_resolution_invalid",)
    assert "PRIVATE" not in repr(raised.value)
    assert str(prepared.path) not in repr(raised.value)
    assert resolver.private_dataset_bytes is None


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
