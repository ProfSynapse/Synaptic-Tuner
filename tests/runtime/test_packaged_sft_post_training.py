"""One-use SFT-to-evaluation handoff over the retained local model copy."""

from __future__ import annotations

import pytest

from tuner.runtime import packaged_sft_execution as seam
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import CanonicalDocument
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload, packaged_configuration_digest,
)
from tests.runtime.test_packaged_sft_execution import FakeRunner, material, prepare, seal  # noqa: F401  pytest fixtures registered by import
from tests.training.test_modal_post_training_compilation import _evaluation


def _opt_in(material):  # noqa: F811  pytest fixture shadows imported fixture
    config = seam._document(material["workload_bytes"])["configuration"]["document"]
    config["post_training"] = _evaluation()
    compiled = compile_packaged_sft_workload(
        resolved_config=CanonicalDocument.from_mapping(config),
    )
    original = material["execution_binding"]
    identity = config["dataset"]
    material["execution_binding"] = PackagedExecutionBindingV1.build(
        run_ref=original.run_ref,
        runtime_release=material["runtime_release"],
        provider_runtime_binding=material["provider_binding"],
        prepared_input_ref=identity["ref"],
        prepared_input_revision=identity["revision"],
        prepared_input_content_digest=identity["content_digest"],
        prepared_input_size_bytes=identity["size_bytes"],
        prepared_input_format=identity["format"],
        workload_digest=compiled.fingerprint,
        configuration_digest=packaged_configuration_digest(
            CanonicalDocument.from_mapping(config),
        ),
        artifact_policy_digest=original.artifact_policy_digest,
    )
    material["workload_bytes"] = compiled.canonical_bytes
    return material


def test_post_training_uses_one_preparation_and_context_expires(material, seal):  # noqa: F811  pytest fixtures
    _opt_in(material)
    calls = []
    captured = []

    def preparer(model, cache):
        calls.append((model, cache))
        return prepare(model, cache)

    def evaluate(result, context):
        context.validate()
        captured.append(context)
        assert context.base_model_path.is_dir()
        assert context.adapter_path.is_dir()
        assert context.tokenizer_path.is_dir()
        assert context.adapter_path == context.tokenizer_path
        assert context.bindings["workload_digest"] == result.workload_fingerprint
        assert context.bindings["adapter_digest"]
        assert context.python_executable == material["runtime_release"].python_executable

    admitted = seam.admit_packaged_sft(**material)
    result = seam.execute_admitted_packaged_sft(
        admitted, model_preparer=preparer, runner=FakeRunner(),
        on_training_complete=evaluate,
    )
    assert len(calls) == len(captured) == 1
    assert tuple(item["role"] for item in result.artifacts) == seam._ROLES
    assert result.inventory_path.is_file() and result.terminal_path.is_file()
    with pytest.raises(ValueError):
        captured[0].validate()
    with pytest.raises(seam.PackagedSFTExecutionError):
        seam.execute_admitted_packaged_sft(
            admitted, model_preparer=preparer, runner=FakeRunner(),
            on_training_complete=evaluate,
        )
    assert len(calls) == 1


def test_evaluation_failure_retains_completed_training_artifacts(material, seal):  # noqa: F811  pytest fixtures
    _opt_in(material)
    admitted = seam.admit_packaged_sft(**material)

    def fail(result, context):
        context.validate()
        assert result.inventory_path.is_file()
        raise RuntimeError("private evaluation failure")

    with pytest.raises(seam.PackagedSFTExecutionError) as caught:
        seam.execute_admitted_packaged_sft(
            admitted, model_preparer=prepare, runner=FakeRunner(),
            on_training_complete=fail,
        )
    assert caught.value.stage == "POST_TRAINING"
    assert (admitted.paths.artifacts / "final_model.tar").is_file()
    assert (admitted.paths.state / "runtime-v1-inventory.json").is_file()
    assert (admitted.paths.state / "packaged-terminal.json").is_file()
    with pytest.raises(seam.PackagedSFTExecutionError):
        seam.execute_admitted_packaged_sft(
            admitted, model_preparer=prepare, runner=FakeRunner(),
            on_training_complete=fail,
        )


def test_changed_adapter_is_rejected_while_context_is_live(material, seal):  # noqa: F811  pytest fixtures
    _opt_in(material)
    admitted = seam.admit_packaged_sft(**material)

    def mutate(_result, context):
        context.validate()
        adapter_config = context.adapter_path / "adapter_config.json"
        adapter_config.write_bytes(adapter_config.read_bytes() + b" ")
        context.validate()

    with pytest.raises(seam.PackagedSFTExecutionError) as caught:
        seam.execute_admitted_packaged_sft(
            admitted, model_preparer=prepare, runner=FakeRunner(),
            on_training_complete=mutate,
        )
    assert caught.value.stage == "POST_TRAINING"
    assert (admitted.paths.artifacts / "final_model.tar").is_file()
