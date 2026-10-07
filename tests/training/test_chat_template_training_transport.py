"""Generic chat-template kwargs stay explicit across public and packaged SFT."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tests.contract.test_public_training_input_v1 import _document
from tuner.training.modal_recipe import load_modal_sft_recipe
from tuner.training.packaged_compilation import compile_packaged_sft_workload
from Trainers.sft.runtime_v1 import (
    RuntimeV1Error, _append_sft_arguments, build_trainer_invocation,
)
from tuner.runtime.verification import _expected_trainer_argv


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
PROFILES = ROOT / "Trainers/runtime_profiles"


def _recipe(monkeypatch, kwargs=None):
    from tuner.training import modal_recipe

    document = deepcopy(yaml.safe_load(RECIPE.read_text(encoding="utf-8")))
    if kwargs is not None:
        document["training"]["chat_template_kwargs"] = kwargs
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: document)
    return load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)


def test_omitted_kwargs_preserve_legacy_training_documents(monkeypatch):
    recipe = _recipe(monkeypatch)
    assert "chat_template_kwargs" not in recipe.hyperparameters.to_dict()
    assert "chat_template_kwargs" not in recipe.training_input("dataset://test").to_dict()["hyperparameters"]


def test_legacy_conversation_hyperparameters_accept_optional_template_kwargs():
    document = _document()
    document["hyperparameters"]["chat_template_kwargs"] = {"enable_thinking": False}
    public = TrainingInputV1.from_dict(document)
    assert public.to_dict() == document
    assert TrainingInputV1.from_json(public.canonical_json()).canonical_json() == public.canonical_json()


def test_kwargs_survive_public_recipe_compilation_and_offline_argv(monkeypatch):
    kwargs = {"enable_thinking": False, "nested_option": {"numbers": [1, 0.5]}}
    recipe = _recipe(monkeypatch, kwargs)
    kwargs["enable_thinking"] = True
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest, recipe.dataset_digest,
        "a" * 64, 123, "syntunia-sft-row/v2",
    )
    public = recipe.training_input(identity.ref)
    assert public.hyperparameters.chat_template_kwargs["enable_thinking"] is False
    assert TrainingInputV1.from_json(public.canonical_json()).canonical_json() == public.canonical_json()
    config = recipe.packaged_config(identity)
    compiled = compile_packaged_sft_workload(resolved_config=config)
    sft = compiled.document["configuration"]["document"]["sft"]
    assert sft["chat_template_kwargs"] == {
        "enable_thinking": False, "nested_option": {"numbers": [1, 0.5]},
    }
    projected = dict(sft)
    projected.pop("schema_version")
    duration = projected.pop("duration")
    projected.update({key: value for key, value in duration.items() if value is not None})
    argv = []
    _append_sft_arguments(argv, projected, config.to_dict()["model"])
    position = argv.index("--chat-template-kwargs")
    assert argv[position + 1] == '{"enable_thinking":false,"nested_option":{"numbers":[1,0.5]}}'
    projected_config = config.to_dict()
    projected_config["sft"] = projected
    expected = _expected_trainer_argv(
        "/usr/bin/python3", {"engine": "/engine", "cache": "/cache", "state": "/state"},
        "/data/dataset.jsonl", projected_config,
        SimpleNamespace(fingerprint=compiled.fingerprint,
                        document={"configuration": {"revision": "r"}}),
    )
    assert expected[expected.index("--chat-template-kwargs") + 1] == argv[position + 1]


@pytest.mark.parametrize("kwargs", [
    {}, {"tokenize": False}, {"enable_thinking": float("nan")},
    {"enable_thinking": object()}, {"context": "x" * 5000},
])
def test_recipe_rejects_invalid_or_unbounded_kwargs(monkeypatch, kwargs):
    with pytest.raises((TypeError, ValueError)):
        _recipe(monkeypatch, kwargs)


@pytest.mark.parametrize("kwargs", [
    {"tokenize": True}, {"enable_thinking": float("inf")},
    {"enable_thinking": object()}, {"context": "x" * 5000},
])
def test_direct_runtime_rejects_unchecked_renderer_kwargs(kwargs):
    with pytest.raises(RuntimeV1Error, match="chat_template_kwargs"):
        _append_sft_arguments(
            [], {"max_steps": 1, "chat_template_kwargs": kwargs},
            {"load_in_4bit": False},
        )


def test_direct_runtime_rejects_raw_text_kwargs_before_model_access(monkeypatch):
    recipe = _recipe(monkeypatch, {"enable_thinking": False})
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest, recipe.dataset_digest,
        "a" * 64, 123, "syntunia-sft-row/v2",
    )
    config = recipe.packaged_config(identity).to_dict()
    sft = dict(config["sft"])
    sft.pop("schema_version")
    duration = sft.pop("duration")
    sft.update({key: value for key, value in duration.items() if value is not None})
    for key in ("prompt_render", "packing", "require_memory_efficient_loss"):
        sft.pop(key)
    sft.update(dataset_format="raw_text", completion_only_loss=False)
    config["dataset"]["format"] = "syntunia-sft-row/v1"
    config["sft"] = sft
    workload = SimpleNamespace(document={"configuration": {"document": config}})
    with pytest.raises(RuntimeV1Error, match="chat_template_kwargs"):
        build_trainer_invocation(workload, None, {})


def test_verified_qwen_template_prompt_completion_boundary_offline():
    """Opt-in artifact-backed proof; no Hub or model weights are accessed."""
    archive_path = os.environ.get("SYNAPTIC_TEST_QWEN35_TOKENIZER_ARTIFACT")
    if not archive_path:
        pytest.skip("set SYNAPTIC_TEST_QWEN35_TOKENIZER_ARTIFACT to a verified local tokenizer artifact")
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from shared.sft_preprocessing import materialize_sft_example

    with tarfile.open(archive_path, "r:") as archive:
        assert sorted(archive.getnames()) == [
            "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
        ]
        template = archive.extractfile("chat_template.jinja").read()
        tokenizer_json = archive.extractfile("tokenizer.json").read()
        tokenizer_config = json.load(archive.extractfile("tokenizer_config.json"))
    assert hashlib.sha256(template).hexdigest() == (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(tokenizer_json.decode("utf-8")),
        chat_template=template.decode("utf-8"),
        eos_token=tokenizer_config["eos_token"],
    )
    question = [{"role": "user", "content": "What happened next?"}]
    default_prompt = tokenizer.apply_chat_template(
        question, tokenize=False, add_generation_prompt=True,
    )
    prose_prompt = tokenizer.apply_chat_template(
        question, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    assert default_prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert prose_prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    completion = "The door opened, and Mara stepped into the garden."
    example = materialize_sft_example(
        tokenizer=tokenizer,
        record={"messages": [*question, {"role": "assistant", "content": completion}]},
        max_seq_length=4096,
        assistant_only_loss=True,
        chat_template_kwargs={"enable_thinking": False},
        prompt_render="prompt_completion",
    )
    prompt_ids = tokenizer.encode(prose_prompt, add_special_tokens=False)
    assert example.input_ids[:len(prompt_ids)] == prompt_ids
    assert example.labels[:len(prompt_ids)] == [-100] * len(prompt_ids)
    target_ids = [label for label in example.labels if label != -100]
    assert target_ids == tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert "<think>" not in tokenizer.decode(target_ids)
