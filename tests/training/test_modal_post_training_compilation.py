"""Provider-free binding checks for the opt-in single-job evaluation plan."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import CanonicalDocument
from tuner.training.modal_recipe import load_modal_sft_recipe
from tuner.training.packaged_boundary import compile_bound_packaged_workload
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload,
    packaged_artifact_policy_digest, packaged_configuration_digest,
)
from tests.runtime.test_packaged_runtime_releases import _provider, _release


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
COMBINED_RECIPE = ROOT / "Trainers/recipes/qwen35_4b_modal_train_eval_smoke.yaml"
PROFILES = ROOT / "Trainers/runtime_profiles"


def _evaluation() -> dict:
    return {
        "mode": "same_job",
        "evaluation": {
            "scenarios": [{
                "id": "fixed_chat_smoke",
                "question": "Say hello in one sentence.",
                "correct": {"all": [{"assertions": [{"type": "contains", "value": "hello"}]}]},
            }],
            "min_pass_rate": 1.0,
            "max_cases": 1,
            "startup_timeout_seconds": 120.0,
            "timeout_seconds": 180.0,
            "served_model_name": "rehearsal-lora",
            "generation": {"max_tokens": 64, "temperature": 0.0, "top_p": 1.0},
            "vllm": {
                "expected_version": "0.26.0", "dtype": "bfloat16",
                "max_model_len": 4096, "tensor_parallel_size": 1,
                "max_num_seqs": 4, "max_num_batched_tokens": 4096,
                "language_model_only": True, "max_lora_rank": 32,
            },
        },
    }


def _config(recipe):
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest,
        recipe.dataset_digest, "a" * 64, 123, "syntunia-sft-row/v2",
    )
    return recipe.packaged_config(identity)


def test_opt_in_evaluation_changes_bound_workload_and_preserves_training_artifacts():
    recipe = load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)
    baseline = _config(recipe)
    requested = _config(replace(recipe, post_training=_evaluation()))
    ordinary = compile_packaged_sft_workload(resolved_config=baseline)
    combined = compile_packaged_sft_workload(resolved_config=requested)

    assert "post_training" not in baseline.to_dict()
    assert requested.to_dict()["post_training"] == _evaluation()
    assert ordinary.fingerprint != combined.fingerprint
    assert packaged_configuration_digest(baseline) != packaged_configuration_digest(requested)
    assert ordinary.document["artifacts"] == combined.document["artifacts"]


def test_checked_in_combined_recipe_compiles_exact_inline_evaluation():
    recipe = load_modal_sft_recipe(COMBINED_RECIPE, profiles_root=PROFILES)
    config = _config(recipe)
    compiled = compile_packaged_sft_workload(resolved_config=config)

    evaluation = config.to_dict()["post_training"]["evaluation"]
    assert [case["id"] for case in evaluation["scenarios"]] == [
        "greeting_prose", "observation_prose", "explanation_prose",
    ]
    assert evaluation["max_cases"] == 3
    assert evaluation["vllm"]["expected_version"] == "0.26.0"
    assert compiled.document["configuration"]["document"] == config.to_dict()


def test_combined_recipe_cases_are_bound_to_host_execution_material():
    recipe = load_modal_sft_recipe(COMBINED_RECIPE, profiles_root=PROFILES)
    config = _config(recipe)
    workload = compile_packaged_sft_workload(resolved_config=config)
    identity = config.to_dict()["dataset"]
    release = _release()
    binding = PackagedExecutionBindingV1.build(
        run_ref="run-combined-smoke", runtime_release=release,
        provider_runtime_binding=_provider(release),
        prepared_input_ref=identity["ref"],
        prepared_input_revision=identity["revision"],
        prepared_input_content_digest=identity["content_digest"],
        prepared_input_size_bytes=identity["size_bytes"],
        prepared_input_format=identity["format"],
        workload_digest=workload.fingerprint,
        configuration_digest=packaged_configuration_digest(config),
        artifact_policy_digest=packaged_artifact_policy_digest(recipe.artifact_policy()),
    )
    assert compile_bound_packaged_workload(config, binding) == workload

    changed = config.to_dict()
    changed["post_training"]["evaluation"]["scenarios"][0]["question"] += " Please."
    with pytest.raises(ValueError, match="binding differs"):
        compile_bound_packaged_workload(CanonicalDocument.from_mapping(changed), binding)


def test_opt_in_evaluation_rejects_unbound_or_invalid_controls():
    recipe = load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)
    valid = _config(replace(recipe, post_training=_evaluation())).to_dict()
    for post_training in (None, {"mode": "parallel", "evaluation": _evaluation()["evaluation"]}):
        candidate = dict(valid)
        candidate["post_training"] = post_training
        with pytest.raises(ValueError):
            compile_packaged_sft_workload(
                resolved_config=CanonicalDocument.from_mapping(candidate),
            )
