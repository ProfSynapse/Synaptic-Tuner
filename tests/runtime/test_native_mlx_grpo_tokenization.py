"""Pinned tokenizer return contract; no Transformers/ML imports or model use."""
from dataclasses import replace

import pytest

from tests.runtime.test_native_mlx_grpo import inputs
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.runtime.native_mlx_grpo import prepared_prompts, tokenize_prompts
from tuner.runtime.native_mlx_material import canonical
from tuner.training.native_mlx_grpo_compilation import compile_native_grpo


class MappingDefaultTokenizer:
    """Focused Transformers5.20.2 apply_chat_template contract double.

    Pinned tokenization_utils_base.py SHA256
    b16c2ae2a1296f1160f9ab6d73db9da6f6725b4e227b2167597b52add3e23108:
    return_dict defaults True (3017);3188–3219 returns out/input_ids.
    Rendering/encoding are deterministic stubs, not actual tokenizer evidence.
    """
    eos_token_id = 99

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize=True, return_dict=True,
                            add_generation_prompt=False, **kwargs):
        self.calls.append((messages, tokenize, add_generation_prompt, kwargs))
        assert kwargs == {"enable_thinking": False}
        ids = [41, 42, 43] if add_generation_prompt else [41, 42]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)} if return_dict else ids


def test_mapping_default_reproduces_old_failure_explicit_list_keeps_ids_and_scaffold():
    training, source, profile, raw = inputs()
    projection = compile_native_grpo(training, source, profile)
    prompts = prepared_prompts(raw, projection)
    tokenizer = MappingDefaultTokenizer()
    old = tuple(tokenizer.apply_chat_template(prompts[0], tokenize=True,
        add_generation_prompt=True, **training.hyperparameters.chat_template_kwargs))
    assert old == ("input_ids", "attention_mask")
    # Original worker rejects the iterated mapping keys, not the underlying IDs.
    assert any(type(token) is not int for token in old)
    old_scaffold = tokenizer.calls[-1]
    encoded, eos = tokenize_prompts(prompts, tokenizer=tokenizer, projection=projection)
    assert encoded == ((41, 42, 43), (41, 42, 43)) and eos == 99
    assert tokenizer.calls[1] == old_scaffold
    assert tokenizer.calls[2] == (prompts[1], True, True, {"enable_thinking": False})
    assert all("SECRET-GOLD" not in str(call) for call in tokenizer.calls)


@pytest.mark.parametrize("key,value", [("return_dict", True), ("return_dict", False),
    ("tokenize", False), ("add_generation_prompt", False), ("return_tensors", "np")])
def test_configured_kwargs_cannot_override_list_ids_or_generation_boundary(key, value):
    training, source, profile, _ = inputs()
    document = training.to_dict()
    document["hyperparameters"]["chat_template_kwargs"][key] = value
    with pytest.raises(ValueError, match="renderer controls"):
        TrainingInputV1.from_dict(document)
    projection = compile_native_grpo(training, source, profile)
    forged = replace(projection, tokenization=replace(projection.tokenization,
        chat_template_json=canonical({key: value})))
    tokenizer = MappingDefaultTokenizer()
    with pytest.raises(ValueError, match="renderer controls"):
        tokenize_prompts(([{"role": "user", "content": "fixture"}],), tokenizer=tokenizer, projection=forged)
    assert not tokenizer.calls
