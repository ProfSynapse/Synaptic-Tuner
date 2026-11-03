"""Real CPU grammar decoding and generic conditional-model dispatch."""
import json
import sys

import pytest

torch = pytest.importorskip("torch")
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

from tuner.batch.engines.hf_batched import HFBatchedGenerateEngine
from tuner.batch.engines.base import GenerateItem
from shared.utilities.text_model import load_text_generation_model


def test_missing_grammar_dependency_refuses_before_model_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgrammar", None)
    with pytest.raises(RuntimeError, match="optional xgrammar"):
        HFBatchedGenerateEngine("must-not-be-loaded", json_schema={"type": "object"})


def test_real_schema_decoding_resets_matcher_between_batches_and_calls():
    pytest.importorskip("xgrammar")
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update({"[EOS]": 256, "[PAD]": 257})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]")
    cfg = LlamaConfig(vocab_size=258, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      eos_token_id=256, pad_token_id=257, max_position_embeddings=256)
    torch.manual_seed(17)
    model = LlamaForCausalLM(cfg).eval()
    # Keep the random fixture from spending its entire token allowance on
    # schema-permitted whitespace. Grammar still supplies every JSON constraint.
    model.lm_head.bias = torch.nn.Parameter(torch.zeros(258))
    with torch.no_grad():
        for char in (" ", "\t", "\r", "\n"):
            model.lm_head.bias[tok.encode(char, add_special_tokens=False)] = -100
    schema = {"type": "object", "properties": {"answer": {"const": "ok"}},
              "required": ["answer"], "additionalProperties": False}
    engine = HFBatchedGenerateEngine("unused", model=model, tokenizer=tok, device="cpu",
                                     json_schema=schema, structured_output_backend="xgrammar", max_new_tokens=64)
    items = [GenerateItem(id=str(i), prompt=prompt) for i, prompt in enumerate(["Hi", "A longer prompt"])]
    first = engine.generate(items, batch_size=1)
    again = engine.generate(list(reversed(items)), batch_size=1)
    for row in first + again:
        assert json.loads(row.completion_text) == {"answer": "ok"}
        assert row.finish_reason == "eos"
    assert {r.id: r.completion_token_ids for r in first} == {r.id: r.completion_token_ids for r in again}
    assert engine.provenance()["structured_output_backend"] == "xgrammar"


def test_conditional_model_dispatch_is_registered_and_does_not_retry_weight_errors(monkeypatch):
    import transformers as tf
    class ConditionalConfig:
        pass
    cfg = ConditionalConfig()
    monkeypatch.setattr(tf.AutoConfig, "from_pretrained", lambda *a, **k: cfg)
    monkeypatch.setattr(tf.AutoModelForCausalLM, "_model_mapping", {})
    monkeypatch.setattr(tf.AutoModelForImageTextToText, "_model_mapping", {ConditionalConfig: object})
    calls = []
    def conditional(name, **kwargs):
        calls.append((name, kwargs))
        raise OSError("bad checkpoint")
    monkeypatch.setattr(tf.AutoModelForImageTextToText, "from_pretrained", conditional)
    monkeypatch.setattr(tf.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("wrong loader"))
    with pytest.raises(OSError, match="bad checkpoint"):
        load_text_generation_model("local-checkpoint", revision="a" * 40, dtype="bfloat16")
    assert len(calls) == 1 and calls[0][1]["config"] is cfg and calls[0][1]["revision"] == "a" * 40
