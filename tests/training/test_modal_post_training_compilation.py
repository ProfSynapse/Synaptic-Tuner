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
    # This is still a completed-text smoke, not a style-quality qualification.
    recipe = load_modal_sft_recipe(COMBINED_RECIPE, profiles_root=PROFILES)
    config = _config(recipe).to_dict()
    assert config["sft"]["chat_template_kwargs"] == {"enable_thinking": False}
    generation = config["post_training"]["evaluation"]["generation"]
    assert generation["max_tokens"] is None
    assert generation["chat_template_kwargs"] == config["sft"]["chat_template_kwargs"]


@pytest.mark.parametrize("content,finish_reason,expected", [
    ("Welcome to the neighborhood!", "stop", True),
    ("Welcome to the neighborhood!", "length", False),
    ("Thinking Process: decide how to greet the neighbor", "stop", False),
    ("<think>Decide what to say</think>", "stop", False),
    ("", "stop", False),
])
def test_combined_recipe_rejects_truncated_or_thinking_only_responses(content, finish_reason, expected):
    from Evaluator.response_view import build_response_view
    from shared.verifiers.builtins.assertion_verifier import evaluate_correctness

    recipe = load_modal_sft_recipe(COMBINED_RECIPE, profiles_root=PROFILES)
    scenarios = recipe.post_training["evaluation"]["scenarios"]
    raw = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    view = build_response_view(content, raw)
    assert all(evaluate_correctness(case["correct"], view).passed is expected for case in scenarios)


def test_combined_recipe_config_is_bound_to_host_execution_material():
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


@pytest.mark.parametrize("limit", [None, 1, 8192])
def test_generation_transport_controls_are_bound_without_legacy_byte_changes(limit):
    from tuner.training.post_training import validate_post_training_config
    from tuner.training.recipes import canonical_json_bytes
    recipe = load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)
    legacy = _evaluation()
    assert canonical_json_bytes(validate_post_training_config(legacy)) == canonical_json_bytes(legacy)
    requested = _evaluation()
    requested["evaluation"]["generation"].update(max_tokens=limit, chat_template_kwargs={"enable_thinking": False})
    requested["evaluation"]["vllm"]["max_model_len"] = 16384
    normalized = validate_post_training_config(requested)
    assert normalized["evaluation"]["generation"]["max_tokens"] == limit
    baseline = _config(replace(recipe, post_training=legacy))
    changed = _config(replace(recipe, post_training=normalized))
    assert packaged_configuration_digest(changed) != packaged_configuration_digest(baseline)
    assert compile_packaged_sft_workload(resolved_config=changed).fingerprint != compile_packaged_sft_workload(resolved_config=baseline).fingerprint


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, 262145, 4096])
def test_post_training_token_limit_rejects_invalid_or_context_sized_values(limit):
    from tuner.training.post_training import validate_post_training_config
    config = _evaluation()
    config["evaluation"]["generation"]["max_tokens"] = limit
    with pytest.raises(ValueError):
        validate_post_training_config(config)


@pytest.mark.parametrize("kwargs", [None, {}, {"tokenize": False}, {"value": float("inf")},
                                     {"value": "x" * 4097}])
def test_post_training_template_kwargs_use_shared_bounded_validator(kwargs):
    from tuner.training.post_training import validate_post_training_config
    config = _evaluation()
    config["evaluation"]["generation"]["chat_template_kwargs"] = kwargs
    with pytest.raises(ValueError):
        validate_post_training_config(config)


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
