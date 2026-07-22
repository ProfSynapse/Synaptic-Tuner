from __future__ import annotations

import json
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft"))

from configs.config_loader import (  # noqa: E402
    TokenizerConfig,
    load_model_config,
    load_tokenizer_config,
    resolve_model_revision,
    validate_model_revision_evidence,
)
import src.special_tokens as special_tokens_module  # noqa: E402
from src.special_tokens import (  # noqa: E402
    _adapter_only_live_state,
    bind_adapter_artifact_lineage,
    prepare_special_tokens,
    restore_portable_adapter_base_provenance,
    restore_verified_selective_token_deltas,
    save_adapter_without_base_vocab,
    save_special_token_artifacts,
    verify_saved_adapter_base_provenance,
    verify_peft_trainable_token_adapters,
    verify_merged_model_roundtrip,
    verify_saved_adapter_roundtrip,
    verify_saved_special_tokenizer,
    write_special_token_lineage,
)


class FakeTokenizer:
    def __init__(self, vocab=None, additional_special_tokens=None):
        self.vocab = dict(vocab or {"alpha": 0, "beta": 1})
        self.additional_special_tokens = list(additional_special_tokens or [])

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def all_special_tokens(self):
        return list(self.additional_special_tokens)

    def add_special_tokens(self, mapping, replace_additional_special_tokens=False):
        assert replace_additional_special_tokens is False
        added = 0
        for token in mapping["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, token, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.vocab[token]] if token in self.vocab else []

    def save_pretrained(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "fake_tokenizer.json").write_text(
            json.dumps(
                {
                    "vocab": self.vocab,
                    "additional_special_tokens": self.additional_special_tokens,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, output_dir, local_files_only=False):
        assert local_files_only is True
        payload = json.loads((Path(output_dir) / "fake_tokenizer.json").read_text(encoding="utf-8"))
        return cls(**payload)


class CurrentKeywordFakeTokenizer(FakeTokenizer):
    def add_special_tokens(self, mapping, replace_extra_special_tokens=True):
        assert replace_extra_special_tokens is False
        return self._add(mapping)

    def _add(self, mapping):
        added = 0
        for token in mapping["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added


class UnsafeKeywordFakeTokenizer(FakeTokenizer):
    def add_special_tokens(self, mapping):
        return 0


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=2, hidden_size=4, *, tied=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=tied, vocab_size=vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, size):
        old_embed = self.embed
        old_head = self.lm_head
        hidden = old_embed.embedding_dim
        tied = old_head.weight is old_embed.weight
        self.embed = nn.Embedding(size, hidden)
        self.lm_head = nn.Linear(hidden, size, bias=False)
        with torch.no_grad():
            self.embed.weight[: old_embed.num_embeddings].copy_(old_embed.weight)
            self.lm_head.weight[: old_head.out_features].copy_(old_head.weight)
        if tied:
            self.lm_head.weight = self.embed.weight
        self.config.vocab_size = size
        return self.embed


def _config(**overrides):
    values = {
        "additional_special_tokens": ["<MODE_A>", "<MODE_B>"],
        "existing_token_policy": "error",
        "initialization": "mean_existing_rows",
        "train_new_embedding_rows": True,
        "train_new_lm_head_rows": True,
        "verify_tokenizer_roundtrip": True,
        "verify_adapter_roundtrip": True,
        "merged_model_save_method": "merged_16bit",
    }
    values.update(overrides)
    return TokenizerConfig(**values)


def _clone_state_dict(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_state_dict_equal(model, expected):
    actual = model.state_dict()
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def _saved_adapter_state(output_dir):
    safetensors = pytest.importorskip("safetensors.torch")
    return safetensors.load_file(str(Path(output_dir) / "adapter_model.safetensors"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bind_test_base_provenance(model, metadata, revision=None):
    evidence = None
    if revision is not None:
        evidence = {
            "requested_repo": "test-org/test-model",
            "requested_revision": revision,
            "resolved_commit": revision,
        }
    return restore_portable_adapter_base_provenance(
        model,
        metadata,
        requested_repo="test-org/test-model",
        requested_revision=revision,
        revision_evidence=evidence,
    )


def _set_merged_test_provenance(metadata):
    revision = "a" * 40
    metadata["base_model_provenance"] = {
        "requested_repo": "test-org/test-model",
        "requested_revision": revision,
        "resolved_commit": revision,
        "adapters": ["default"],
        "portable": True,
        "runtime_snapshot_path_persisted": False,
    }


def test_absent_tokens_are_a_strict_noop():
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    input_before = model.embed.weight.detach().clone()
    assert prepare_special_tokens(model, tokenizer, TokenizerConfig()) is None
    assert tokenizer.get_vocab() == {"alpha": 0, "beta": 1}
    assert torch.equal(model.embed.weight, input_before)


def test_pinned_peft_save_without_special_tokens_restores_portable_provenance(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=2,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
        )
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"]),
    )
    model.peft_config["default"].base_model_name_or_path = (
        "/cache/models--test-org--test-model/snapshots/" + "a" * 40
    )
    revision = "a" * 40
    provenance = restore_portable_adapter_base_provenance(
        model,
        None,
        requested_repo="test-org/test-model",
        requested_revision=revision,
        revision_evidence={
            "requested_repo": "test-org/test-model",
            "requested_revision": revision,
            "resolved_commit": revision,
        },
    )

    output_dir = tmp_path / "adapter"
    model.save_pretrained(output_dir)
    report = verify_saved_adapter_base_provenance(output_dir, provenance)
    saved = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))

    assert saved["base_model_name_or_path"] == "test-org/test-model"
    assert saved["revision"] == revision
    assert report["result"] == "passed"


def test_unpinned_local_model_path_save_is_not_rewritten_or_rejected(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    local_model_path = str((tmp_path / "local-model").resolve())
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=2,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
        )
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"]),
    )
    model.peft_config["default"].base_model_name_or_path = local_model_path

    provenance = restore_portable_adapter_base_provenance(
        model,
        None,
        requested_repo=local_model_path,
        requested_revision=None,
        revision_evidence=None,
    )
    output_dir = tmp_path / "adapter"
    model.save_pretrained(output_dir)
    saved = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))

    assert provenance is None
    assert model.peft_config["default"].base_model_name_or_path == local_model_path
    assert saved["base_model_name_or_path"] == local_model_path
    assert saved.get("revision") is None


def test_ordered_add_resize_mean_init_and_row_plan():
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    input_mean = model.embed.weight.detach().mean(dim=0)
    output_mean = model.lm_head.weight.detach().mean(dim=0)

    metadata = prepare_special_tokens(model, tokenizer, _config())

    assert [entry["token"] for entry in metadata["configured_tokens"]] == ["<MODE_A>", "<MODE_B>"]
    assert [entry["token_id"] for entry in metadata["configured_tokens"]] == [2, 3]
    assert metadata["trainable_token_indices"] == {"embed": [2, 3], "lm_head": [2, 3]}
    assert torch.allclose(model.embed.weight[2], input_mean)
    assert torch.allclose(model.embed.weight[3], input_mean)
    assert torch.allclose(model.lm_head.weight[2], output_mean)
    assert torch.allclose(model.lm_head.weight[3], output_mean)
    assert len(metadata["config_sha256"]) == 64
    assert len(metadata["vocab_sha256_after"]) == 64


@pytest.mark.parametrize("tokenizer_cls", [FakeTokenizer, CurrentKeywordFakeTokenizer])
def test_both_safe_no_replacement_keyword_signatures_preserve_existing_specials(tokenizer_cls):
    tokenizer = tokenizer_cls(
        vocab={"alpha": 0, "<OLD_SPECIAL>": 1},
        additional_special_tokens=["<OLD_SPECIAL>"],
    )
    model = TinyCausalLM(vocab_size=2)
    metadata = prepare_special_tokens(
        model,
        tokenizer,
        _config(additional_special_tokens=["<MODE_A>"]),
    )
    assert tokenizer.all_special_tokens == ["<OLD_SPECIAL>", "<MODE_A>"]
    assert tokenizer.convert_tokens_to_ids("<OLD_SPECIAL>") == 1
    assert metadata["configured_tokens"][0]["token_id"] == 2


def test_tokenizer_without_safe_no_replacement_keyword_fails_before_addition():
    tokenizer = UnsafeKeywordFakeTokenizer()
    with pytest.raises(TypeError, match="no safe no-replacement"):
        prepare_special_tokens(
            TinyCausalLM(),
            tokenizer,
            _config(additional_special_tokens=["<MODE_A>"]),
        )
    assert "<MODE_A>" not in tokenizer.get_vocab()


def test_padded_model_vocab_is_never_shrunk():
    model = TinyCausalLM(vocab_size=6)
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(model, tokenizer, _config())
    assert metadata["resize_applied"] is False
    assert metadata["model_vocab_size_after"] == 6
    assert metadata["new_token_ids"] == [2, 3]


def test_collision_policy_is_explicit_and_reuse_never_trains_old_row():
    tokenizer = FakeTokenizer(vocab={"alpha": 0, "<MODE_A>": 1})
    model = TinyCausalLM(vocab_size=2)
    with pytest.raises(ValueError, match="already exist"):
        prepare_special_tokens(
            model,
            tokenizer,
            _config(additional_special_tokens=["<MODE_A>"], existing_token_policy="error"),
        )

    metadata = prepare_special_tokens(
        model,
        tokenizer,
        _config(additional_special_tokens=["<MODE_A>"], existing_token_policy="reuse"),
    )
    assert metadata["configured_tokens"][0]["status"] == "reused"
    assert metadata["new_token_ids"] == []
    assert metadata["trainable_token_indices"] == {}


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"additional_special_tokens": ["<MODE_A>", "<MODE_A>"]}, "duplicates"),
        ({"additional_special_tokens": [""]}, "non-empty strings"),
        ({"additional_special_tokens": ["<MODE_A>"], "existing_token_policy": "guess"}, "must be one of"),
        ({"additional_special_tokens": ["<MODE_A>"], "initialization": "random"}, "must be one of"),
        ({"additional_special_tokens": ["<MODE_A>"], "merged_model_save_method": "auto"}, "must be one of"),
        ({"additional_special_tokens": ["<MODE_A>"], "merged_model_save_method": []}, "must be one of"),
        ({"additional_special_tokens": ["<MODE_A>"], "train_new_embedding_rows": "false"}, "YAML boolean"),
        ({"additional_special_tokens": ["<MODE_A>"], "unknown_knob": True}, "Unknown model.tokenizer"),
        ({"additional_special_tokens": ["<MODE_A>"], "verify_save_reload_roundtrip": True}, "Unknown model.tokenizer"),
    ],
)
def test_invalid_config_fails_before_model_mutation(payload, message):
    with pytest.raises(ValueError, match=message):
        load_tokenizer_config(payload)


def test_direct_yaml_model_loader_rejects_falsy_non_mapping_tokenizer_block():
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_model_config(
            {
                "model_name": "example/model",
                "max_seq_length": 16,
                "dtype": None,
                "load_in_4bit": False,
                "tokenizer": [],
            }
        )


def test_model_revision_is_optional_validated_and_preserved():
    base = {
        "model_name": "example/model",
        "max_seq_length": 16,
        "dtype": None,
        "load_in_4bit": False,
    }
    assert load_model_config(base).revision is None
    revision = "cad0bedfdd862093a12af478cb974ab2addd0e0a"
    assert load_model_config({**base, "revision": revision}).revision == revision
    for invalid in ("", " cad0bedf", "cad0bedf", 123, False, True):
        with pytest.raises(ValueError, match="model.revision"):
            load_model_config({**base, "revision": invalid})


def test_legacy_custom_model_config_without_revision_is_an_exact_noop():
    legacy_model_config = SimpleNamespace(model_name="example/model")
    assert resolve_model_revision(legacy_model_config) is None
    assert not hasattr(legacy_model_config, "revision")


def test_revision_lineage_evidence_binds_requested_repo_and_full_commit():
    revision = "cad0bedfdd862093a12af478cb974ab2addd0e0a"
    evidence = {
        "requested_repo": "unsloth/Qwen3-4B-bnb-4bit",
        "requested_revision": revision,
        "resolved_snapshot_path": f"/cache/snapshots/{revision}",
        "resolved_commit": revision,
        "resolution_method": "huggingface_hub.snapshot_download_local_snapshot",
    }
    assert validate_model_revision_evidence(
        "unsloth/Qwen3-4B-bnb-4bit", revision, evidence
    )["resolved_commit"] == revision
    with pytest.raises(RuntimeError, match="resolved_commit"):
        validate_model_revision_evidence(
            "unsloth/Qwen3-4B-bnb-4bit",
            revision,
            {**evidence, "resolved_commit": "0" * 40},
        )


def test_tied_head_rejects_incoherent_training_flags():
    model = TinyCausalLM(tied=True)
    with pytest.raises(ValueError, match="are tied"):
        prepare_special_tokens(
            model,
            FakeTokenizer(),
            _config(train_new_embedding_rows=True, train_new_lm_head_rows=False),
        )


def test_declared_tied_but_actual_untied_fails_before_tokenizer_mutation():
    model = TinyCausalLM(tied=False)
    model.config.tie_word_embeddings = True
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="do not share storage"):
        prepare_special_tokens(model, tokenizer, _config())
    assert tokenizer.get_vocab() == {"alpha": 0, "beta": 1}


def test_tokenizer_save_reload_roundtrip_and_lineage(tmp_path):
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(model, tokenizer, _config())
    save_special_token_artifacts(tokenizer, tmp_path, metadata)
    assert (tmp_path / "special_tokens_lineage.json").is_file()
    reloaded = FakeTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.encode("<MODE_A>", add_special_tokens=False) == [2]


def test_merged_model_roundtrip_is_default_off():
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(model, tokenizer, _config())
    assert verify_merged_model_roundtrip(model, tokenizer, "/unused", metadata) is None


def test_forced_4bit_merge_candidate_uses_fresh_pinned_base_and_saved_adapter(
    tmp_path, monkeypatch
):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    quantization = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
    }
    base = TinyCausalLM(vocab_size=2)
    base.config.quantization_config = dict(quantization)
    base.is_loaded_in_4bit = True
    load_calls = []

    def fake_base_load(repo, **kwargs):
        load_calls.append((repo, kwargs))
        return base

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_base_load
    )
    adapter_calls = []

    def fake_adapter_load(loaded_base, path, **kwargs):
        adapter_calls.append((loaded_base, path, kwargs))
        return loaded_base

    monkeypatch.setattr(
        peft.PeftModel, "from_pretrained", staticmethod(fake_adapter_load)
    )
    verified = []
    monkeypatch.setattr(
        special_tokens_module,
        "verify_peft_trainable_token_adapters",
        lambda candidate, metadata, require_trainable: verified.append(
            (candidate, require_trainable)
        ),
    )
    fake_unsloth = ModuleType("unsloth")
    fake_unsloth_save = ModuleType("unsloth.save")

    def fake_patch(candidate):
        candidate.save_pretrained_merged = lambda *args, **kwargs: None
        return candidate

    fake_unsloth_save.patch_saving_functions = fake_patch
    fake_unsloth.save = fake_unsloth_save
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.save", fake_unsloth_save)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    revision = "a" * 40
    metadata = {
        "base_model_provenance": {
            "requested_repo": "test-org/test-model",
            "requested_revision": revision,
        },
        "model_vocab_size_after": 4,
    }

    candidate, device, invariants = (
        special_tokens_module._fresh_forced_4bit_merge_candidate(tmp_path, metadata)
    )

    assert candidate is base
    assert device == "cuda:2"
    assert invariants == quantization
    assert load_calls == [
        (
            "test-org/test-model",
            {
                "revision": revision,
                "local_files_only": True,
                "trust_remote_code": False,
                "device_map": {"": 2},
                "dtype": "auto",
                "low_cpu_mem_usage": True,
            },
        )
    ]
    assert base.get_input_embeddings().weight.shape[0] == 4
    assert adapter_calls == [
        (
            base,
            str(tmp_path),
            {"is_trainable": False, "local_files_only": True},
        )
    ]
    assert verified == [(base, False)]
    assert callable(candidate.save_pretrained_merged)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quant_method", "gptq"),
        ("load_in_4bit", False),
        ("bnb_4bit_quant_type", ""),
        ("bnb_4bit_quant_type", "int4"),
    ],
)
def test_forced_4bit_invariants_fail_closed(field, value):
    config = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
    }
    config[field] = value
    with pytest.raises(RuntimeError, match="not a supported BitsAndBytes 4-bit"):
        special_tokens_module._require_bnb_4bit_invariants(
            config, description="test model"
        )


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_forced_4bit_invariants_accept_supported_source_types(quant_type):
    config = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": quant_type,
    }
    assert special_tokens_module._require_bnb_4bit_invariants(
        config, description="fresh base"
    ) == config


def test_forced_4bit_invariants_require_exact_source_match():
    source = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
    }
    saved = dict(source, bnb_4bit_quant_type="fp4")
    with pytest.raises(RuntimeError, match="differ from the fresh pinned base"):
        special_tokens_module._require_bnb_4bit_invariants(
            saved, description="saved model", expected=source
        )


@pytest.mark.parametrize(
    "save_method,quant_type",
    [
        ("merged_16bit", None),
        ("merged_4bit_forced", "nf4"),
        ("merged_4bit_forced", "fp4"),
    ],
)
def test_merged_model_roundtrip_saves_reloads_compares_and_cleans(
    tmp_path, monkeypatch, save_method, quant_type
):
    transformers = pytest.importorskip("transformers")
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(
        model,
        tokenizer,
        _config(
            verify_merged_model_roundtrip=True,
            merged_model_save_method=save_method,
        ),
    )
    _set_merged_test_provenance(metadata)
    reloaded = TinyCausalLM(vocab_size=len(tokenizer))
    reloaded.load_state_dict(model.state_dict())
    saved_quantization = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": quant_type,
    }
    if save_method == "merged_4bit_forced":
        reloaded.config.quantization_config = dict(saved_quantization)
        reloaded.is_loaded_in_4bit = True
    live_topology = [(name, id(module)) for name, module in model.named_modules()]
    live_input_rows = model.get_input_embeddings().weight[2:4].detach().clone()
    live_output_rows = model.get_output_embeddings().weight[2:4].detach().clone()
    save_targets = []

    def fake_save(output_dir, saved_tokenizer, save_method):
        assert save_method == metadata["resolved_config"]["merged_model_save_method"]
        save_targets.append("copy" if target is merge_candidate else "live")
        saved_tokenizer.save_pretrained(output_dir)
        (Path(output_dir) / "model.safetensors").write_bytes(b"merged")
        config = (
            {"quantization_config": saved_quantization}
            if save_method == "merged_4bit_forced"
            else {}
        )
        (Path(output_dir) / "config.json").write_text(json.dumps(config))

    target = model
    merge_candidate = None
    model.save_pretrained_merged = fake_save
    if save_method == "merged_4bit_forced":
        merge_candidate = TinyCausalLM(vocab_size=len(tokenizer))
        merge_candidate.load_state_dict(model.state_dict())
        merge_candidate.save_pretrained_merged = fake_save
        target = merge_candidate
        monkeypatch.setattr(
            special_tokens_module,
            "_fresh_forced_4bit_merge_candidate",
            lambda output_dir, resolved_metadata: (
                merge_candidate,
                "cuda:0",
                dict(saved_quantization),
            ),
        )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda path, local_files_only, trust_remote_code: FakeTokenizer.from_pretrained(
            path, local_files_only=local_files_only
        ),
    )

    def fake_model_reload(path, **kwargs):
        assert Path(path).name.startswith(".merged-roundtrip-")
        assert kwargs == {
            "local_files_only": True,
            "trust_remote_code": False,
            "device_map": {"": "cpu"},
            "dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        return reloaded

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_model_reload
    )

    output_dir = tmp_path / "final_model"
    report = verify_merged_model_roundtrip(model, tokenizer, output_dir, metadata)
    assert report["result"] == "passed"
    assert report["save_method"] == save_method
    assert report["requested_save_method"] == save_method
    assert report["saved_representation"] == (
        f"bitsandbytes_{quant_type}_4bit"
        if save_method == "merged_4bit_forced"
        else "unquantized_16bit"
    )
    assert report["merge_source"] == (
        "fresh_pinned_base_plus_saved_adapter"
        if save_method == "merged_4bit_forced"
        else "live_model_nondestructive_export"
    )
    assert save_targets == [
        "copy" if save_method == "merged_4bit_forced" else "live"
    ]
    assert [(name, id(module)) for name, module in model.named_modules()] == live_topology
    assert torch.equal(model.get_input_embeddings().weight[2:4], live_input_rows)
    assert torch.equal(model.get_output_embeddings().weight[2:4], live_output_rows)
    assert report["reload_loader"] == "transformers.AutoModelForCausalLM"
    assert report["reload_device"] == "cpu"
    assert report["trust_remote_code"] is False
    assert report["reload_is_loaded_in_4bit"] is (
        save_method == "merged_4bit_forced"
    )
    assert report["forward_executed"] is False
    assert report["live_model_configured_rows_preserved"] is True
    assert report["published"] is False
    assert report["temporary_artifacts_removed"] is True
    assert (output_dir / "merged_model_roundtrip.json").is_file()
    assert metadata["merged_model_roundtrip"]["result"] == "passed"
    assert not list(tmp_path.glob(".merged-roundtrip-*"))


@pytest.mark.parametrize("save_method", ["merged_16bit", "merged_4bit_forced"])
def test_merged_model_roundtrip_requires_saved_files_and_cleans(
    tmp_path, monkeypatch, save_method
):
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(
        model,
        tokenizer,
        _config(
            verify_merged_model_roundtrip=True,
            merged_model_save_method=save_method,
        ),
    )
    _set_merged_test_provenance(metadata)
    received_methods = []

    def fake_save(output_dir, saved_tokenizer, *, save_method):
        received_methods.append(save_method)

    save_target = model
    if save_method == "merged_4bit_forced":
        save_target = TinyCausalLM(vocab_size=len(tokenizer))
        save_target.load_state_dict(model.state_dict())
        monkeypatch.setattr(
            special_tokens_module,
            "_fresh_forced_4bit_merge_candidate",
            lambda output_dir, resolved_metadata: (
                save_target,
                "cuda:0",
                {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                },
            ),
        )
    save_target.save_pretrained_merged = fake_save
    with pytest.raises(RuntimeError, match="produced no files"):
        verify_merged_model_roundtrip(model, tokenizer, tmp_path / "final", metadata)
    assert received_methods == [save_method]
    assert not list(tmp_path.glob(".merged-roundtrip-*"))


def test_merged_model_roundtrip_rejects_unsupported_resolved_method_before_save(
    tmp_path,
):
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(
        model, tokenizer, _config(verify_merged_model_roundtrip=True)
    )
    _set_merged_test_provenance(metadata)
    metadata["resolved_config"]["merged_model_save_method"] = "auto"
    save_called = False

    def fake_save(*args, **kwargs):
        nonlocal save_called
        save_called = True

    model.save_pretrained_merged = fake_save
    with pytest.raises(ValueError, match="explicit supported"):
        verify_merged_model_roundtrip(model, tokenizer, tmp_path, metadata)
    assert save_called is False
    assert not list(tmp_path.parent.glob(".merged-roundtrip-*"))


def test_merged_model_roundtrip_fails_closed_and_cleans_on_row_mismatch(
    tmp_path, monkeypatch
):
    transformers = pytest.importorskip("transformers")
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(
        model, tokenizer, _config(verify_merged_model_roundtrip=True)
    )
    _set_merged_test_provenance(metadata)
    mismatched = TinyCausalLM(vocab_size=len(tokenizer))

    def fake_save(output_dir, saved_tokenizer, save_method):
        saved_tokenizer.save_pretrained(output_dir)
        (Path(output_dir) / "model.safetensors").write_bytes(b"merged")
        (Path(output_dir) / "config.json").write_text("{}")

    model.save_pretrained_merged = fake_save
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda path, local_files_only, trust_remote_code: FakeTokenizer.from_pretrained(
            path, local_files_only=local_files_only
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: mismatched,
    )

    with pytest.raises(RuntimeError, match="input embedding rows differ"):
        verify_merged_model_roundtrip(
            model, tokenizer, tmp_path / "final_model", metadata
        )
    assert not (tmp_path / "final_model" / "merged_model_roundtrip.json").exists()
    assert not list(tmp_path.glob(".merged-roundtrip-*"))


def test_peft_selective_rows_keep_old_vocab_frozen_and_save_reload(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")

    config = transformers.GPT2Config(
        vocab_size=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=False,
    )
    model = transformers.GPT2LMHeadModel(config)
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(model, tokenizer, _config())
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    lora_config = peft.LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["c_attn"],
        trainable_token_indices=metadata["trainable_token_indices"],
    )
    model = peft.get_peft_model(model, lora_config)
    verify_peft_trainable_token_adapters(model, metadata)

    embed_wrapper = next(module for name, module in model.named_modules() if name.endswith("transformer.wte") and hasattr(module, "token_adapter"))
    head_wrapper = next(module for name, module in model.named_modules() if name.endswith("lm_head") and hasattr(module, "token_adapter"))
    embed_old = embed_wrapper.weight[:2].detach().clone()
    head_old = head_wrapper.weight[:2].detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
        weight_decay=0.1,
    )
    loss = model(input_ids=torch.tensor([[2, 3]]), labels=torch.tensor([[3, 2]])).loss
    loss.backward()
    optimizer.step()
    assert torch.equal(embed_wrapper.weight[:2], embed_old)
    assert torch.equal(head_wrapper.weight[:2], head_old)

    trained_embed_rows = embed_wrapper.weight[2:4].detach().clone()
    trained_head_rows = head_wrapper.weight[2:4].detach().clone()
    _bind_test_base_provenance(model, metadata)
    model.save_pretrained(tmp_path)
    artifact_report = save_adapter_without_base_vocab(model, tmp_path, metadata)
    assert artifact_report["save_embedding_layers"] is False
    tokenizer.save_pretrained(tmp_path)
    live_before_verify = _clone_state_dict(model)
    report = verify_saved_adapter_roundtrip(model, tmp_path, metadata)
    assert report["compared_tensor_count"] == 4
    assert "__special_token_roundtrip_verify__" not in model.peft_config
    assert model.active_adapters == ["default"]
    _assert_state_dict_equal(model, live_before_verify)
    assert not any("token_adapter.base_layer.weight" in key for key in _saved_adapter_state(tmp_path))
    reloaded_tokenizer = FakeTokenizer.from_pretrained(tmp_path, local_files_only=True)
    fresh_config = transformers.GPT2Config(
        vocab_size=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=False,
    )
    reloaded_base = transformers.GPT2LMHeadModel(fresh_config)
    reloaded_base.resize_token_embeddings(len(reloaded_tokenizer))
    reloaded_base.load_state_dict(base_state)
    reloaded = peft.PeftModel.from_pretrained(reloaded_base, tmp_path)
    reloaded_embed = reloaded.get_input_embeddings()
    reloaded_head = reloaded.get_output_embeddings()
    assert torch.equal(reloaded_embed.weight[2:4], trained_embed_rows)
    assert torch.equal(reloaded_head.weight[2:4], trained_head_rows)
    probe_hidden = torch.randn(1, config.n_embd)
    assert torch.equal(reloaded_head(probe_hidden), model.get_output_embeddings()(probe_hidden))
    assert FakeTokenizer.from_pretrained(tmp_path, local_files_only=True).encode(
        "<MODE_B>", add_special_tokens=False
    ) == [3]


def test_peft_tied_rows_share_one_delta_train_and_save_reload(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")

    config = transformers.GPT2Config(
        vocab_size=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=True,
    )
    model = transformers.GPT2LMHeadModel(config)
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(model, tokenizer, _config())
    assert metadata["weights_tied"] is True
    assert metadata["trainable_token_indices"] == {"transformer.wte": [2, 3]}
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    lora_config = peft.LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["c_attn"],
        trainable_token_indices=metadata["trainable_token_indices"],
    )
    model = peft.get_peft_model(model, lora_config)
    verify_peft_trainable_token_adapters(model, metadata)

    input_wrapper = model.get_input_embeddings()
    output_wrapper = model.get_output_embeddings()
    input_adapter = input_wrapper.token_adapter
    output_adapter = output_wrapper.token_adapter
    assert output_adapter.tied_adapter is input_adapter
    assert output_adapter.trainable_tokens_delta is input_adapter.trainable_tokens_delta
    assert output_adapter.trainable_tokens_delta["default"] is input_adapter.trainable_tokens_delta["default"]

    old_rows = input_wrapper.weight[:2].detach().clone()
    new_rows_before = input_wrapper.weight[2:4].detach().clone()
    probe_hidden = torch.randn(1, config.n_embd)
    probe_logits_before = output_wrapper(probe_hidden).detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
        weight_decay=0.1,
    )
    loss = model(input_ids=torch.tensor([[2, 3]]), labels=torch.tensor([[3, 2]])).loss
    loss.backward()
    optimizer.step()

    assert torch.equal(input_wrapper.weight[:2], old_rows)
    assert torch.equal(output_wrapper.weight[:2], old_rows)
    assert not torch.equal(input_wrapper.weight[2:4], new_rows_before)
    assert torch.equal(output_wrapper.weight[2:4], input_wrapper.weight[2:4])
    probe_logits_after = output_wrapper(probe_hidden).detach()
    assert torch.equal(probe_logits_after[:, :2], probe_logits_before[:, :2])
    assert not torch.equal(probe_logits_after[:, 2:4], probe_logits_before[:, 2:4])

    trained_rows = input_wrapper.weight[2:4].detach().clone()
    _bind_test_base_provenance(model, metadata)
    model.save_pretrained(tmp_path)
    artifact_report = save_adapter_without_base_vocab(model, tmp_path, metadata)
    assert artifact_report["save_embedding_layers"] is False
    tokenizer.save_pretrained(tmp_path)
    live_before_verify = _clone_state_dict(model)
    report = verify_saved_adapter_roundtrip(model, tmp_path, metadata)
    assert report["compared_tensor_count"] == 3
    assert "__special_token_roundtrip_verify__" not in model.peft_config
    assert model.active_adapters == ["default"]
    _assert_state_dict_equal(model, live_before_verify)
    assert not any("token_adapter.base_layer.weight" in key for key in _saved_adapter_state(tmp_path))
    reloaded_tokenizer = FakeTokenizer.from_pretrained(tmp_path, local_files_only=True)
    fresh_config = transformers.GPT2Config(
        vocab_size=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=True,
    )
    reloaded_base = transformers.GPT2LMHeadModel(fresh_config)
    reloaded_base.resize_token_embeddings(len(reloaded_tokenizer))
    reloaded_base.load_state_dict(base_state)
    reloaded = peft.PeftModel.from_pretrained(reloaded_base, tmp_path, is_trainable=True)
    verify_peft_trainable_token_adapters(reloaded, metadata)
    assert torch.equal(reloaded.get_input_embeddings().weight[2:4], trained_rows)
    assert torch.equal(reloaded.get_output_embeddings().weight[2:4], trained_rows)
    assert torch.equal(reloaded.get_output_embeddings()(probe_hidden), model.get_output_embeddings()(probe_hidden))
    assert FakeTokenizer.from_pretrained(tmp_path, local_files_only=True).encode(
        "<MODE_A>", add_special_tokens=False
    ) == [2]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("padded_vocab", [False, True])
def test_adapter_only_artifact_matrix_has_no_full_vocab_and_restores_live_model(
    tmp_path, tied, padded_vocab
):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=6 if padded_vocab else 2,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
            tie_word_embeddings=tied,
        )
    )
    metadata = prepare_special_tokens(model, FakeTokenizer(), _config())
    assert metadata["resize_applied"] is (not padded_vocab)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            trainable_token_indices=metadata["trainable_token_indices"],
        ),
    )
    verify_peft_trainable_token_adapters(model, metadata)
    _bind_test_base_provenance(model, metadata)
    model.save_pretrained(tmp_path)
    report = save_adapter_without_base_vocab(model, tmp_path, metadata)
    artifact_state = _saved_adapter_state(tmp_path)
    assert report["tensor_count"] == len(artifact_state)
    assert any("lora_A" in key for key in artifact_state)
    assert any("lora_B" in key for key in artifact_state)
    assert any("trainable_tokens_delta" in key for key in artifact_state)
    assert not any("token_adapter.base_layer" in key for key in artifact_state)
    assert not any(
        key.endswith(("transformer.wte.weight", "lm_head.weight"))
        for key in artifact_state
    )

    live_before = _clone_state_dict(model)
    active_before = list(model.active_adapters)
    verify_saved_adapter_roundtrip(model, tmp_path, metadata)
    _assert_state_dict_equal(model, live_before)
    assert model.active_adapters == active_before
    assert set(model.peft_config) == {"default"}


def test_adapter_roundtrip_preserves_float32_selective_rows_over_bf16_padded_tied_base(
    tmp_path,
):
    """Match the Qwen runtime layout omitted by the original float32 fixtures."""
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")

    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=6,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
            tie_word_embeddings=True,
        )
    ).to(dtype=torch.bfloat16)
    metadata = prepare_special_tokens(base, FakeTokenizer(), _config())
    assert metadata["resize_applied"] is False
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            trainable_token_indices=metadata["trainable_token_indices"],
        ),
    )
    deltas = restore_verified_selective_token_deltas(model, metadata)
    assert len(deltas) == 1
    assert deltas[0].dtype == torch.float32
    with torch.no_grad():
        # Perturb every floating adapter tensor off the bf16 grid. This covers
        # LoRA A/B and the selective replacement rows, not just the tensor that
        # exposed the original failure.
        live_adapter_state = _adapter_only_live_state(model, "default")
        for index, tensor in enumerate(live_adapter_state.values(), start=1):
            if not tensor.is_floating_point():
                continue
            offsets = torch.linspace(
                index * 1e-5,
                index * 1e-5 + 7e-5,
                tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
            ).reshape_as(tensor)
            tensor.add_(offsets)
            assert not torch.equal(
                tensor,
                tensor.to(torch.bfloat16).to(dtype=tensor.dtype),
            )

    _bind_test_base_provenance(model, metadata)
    save_adapter_without_base_vocab(model, tmp_path, metadata)
    source_state = {
        key: tensor.detach().clone()
        for key, tensor in _adapter_only_live_state(model, "default").items()
    }
    assert any("lora_A" in key for key in source_state)
    assert any("lora_B" in key for key in source_state)
    assert any("trainable_tokens_delta" in key for key in source_state)
    live_before_verify = _clone_state_dict(model)
    topology_before_verify = {
        name: id(module) for name, module in model.named_modules()
    }
    report = verify_saved_adapter_roundtrip(model, tmp_path, metadata)
    assert report["result"] == "passed"
    assert report["parameter_preparation"] == "peft_add_adapter_then_cast_before_load"
    assert report["compared_tensor_count"] == len(source_state)
    assert report["compared_keys"] == sorted(source_state)
    _assert_state_dict_equal(model, live_before_verify)
    assert {name: id(module) for name, module in model.named_modules()} == topology_before_verify
    assert model.active_adapters == ["default"]
    assert set(model.peft_config) == {"default"}

    fresh_base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=6,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
            tie_word_embeddings=True,
        )
    ).to(dtype=torch.bfloat16)
    fresh_metadata = prepare_special_tokens(fresh_base, FakeTokenizer(), _config())
    reloaded = peft.PeftModel.from_pretrained(fresh_base, tmp_path)
    fresh_initial_state = _adapter_only_live_state(reloaded, "default")
    assert set(fresh_initial_state) == set(source_state)
    for key in source_state:
        assert torch.equal(source_state[key], fresh_initial_state[key])
    verify_peft_trainable_token_adapters(
        reloaded, fresh_metadata, require_trainable=False
    )


def test_corrupt_adapter_delta_fails_closed_without_mutating_live_model(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=2,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
            tie_word_embeddings=False,
        )
    )
    metadata = prepare_special_tokens(base, FakeTokenizer(), _config())
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            trainable_token_indices=metadata["trainable_token_indices"],
        ),
    )
    _bind_test_base_provenance(model, metadata)
    model.save_pretrained(tmp_path)
    save_adapter_without_base_vocab(model, tmp_path, metadata)
    artifact_path = tmp_path / "adapter_model.safetensors"
    corrupt_state = safetensors.load_file(str(artifact_path))
    delta_key = next(key for key in corrupt_state if "trainable_tokens_delta" in key)
    corrupt_state[delta_key] = corrupt_state[delta_key].clone()
    corrupt_state[delta_key].view(-1)[0] += 1
    safetensors.save_file(corrupt_state, str(artifact_path))

    live_before = _clone_state_dict(model)
    with pytest.raises(RuntimeError, match="On-disk adapter tensor .* differs"):
        verify_saved_adapter_roundtrip(model, tmp_path, metadata)
    _assert_state_dict_equal(model, live_before)
    assert model.active_adapters == ["default"]
    assert set(model.peft_config) == {"default"}


def test_both_lineages_bind_exact_adapter_hashes_and_verification(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")
    output_dir = tmp_path / "model"
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=2,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=8,
            tie_word_embeddings=False,
        )
    )
    tokenizer = FakeTokenizer()
    metadata = prepare_special_tokens(base, tokenizer, _config())
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            trainable_token_indices=metadata["trainable_token_indices"],
        ),
    )
    revision = "b" * 40
    _bind_test_base_provenance(model, metadata, revision=revision)
    model.save_pretrained(output_dir)
    save_report = save_adapter_without_base_vocab(model, output_dir, metadata)
    verify_saved_special_tokenizer(tokenizer, output_dir, metadata)
    verification = verify_saved_adapter_roundtrip(model, output_dir, metadata)
    record = bind_adapter_artifact_lineage(output_dir, metadata, save_report, verification)
    write_special_token_lineage(output_dir, metadata)
    training_lineage = {"model": {"special_tokens": dict(metadata)}}
    training_lineage_path = tmp_path / "training_lineage.json"
    training_lineage_path.write_text(json.dumps(training_lineage), encoding="utf-8")

    saved_adapter_config = json.loads(
        (output_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert saved_adapter_config["base_model_name_or_path"] == "test-org/test-model"
    assert saved_adapter_config["revision"] == revision
    assert "snapshots" not in saved_adapter_config["base_model_name_or_path"]

    assert record["verification"]["result"] == "passed"
    assert record["verification"]["source_adapter"] == "default"
    assert record["verification"]["temporary_adapter"] == "__special_token_roundtrip_verify__"
    assert record["full_base_vocab_tensors_present"] is False
    for lineage_metadata in (
        json.loads((output_dir / "special_tokens_lineage.json").read_text(encoding="utf-8")),
        json.loads(training_lineage_path.read_text(encoding="utf-8"))["model"]["special_tokens"],
    ):
        artifact = lineage_metadata["adapter_artifact"]
        assert artifact["tensor_manifest_sha256"] == record["tensor_manifest_sha256"]
        for file_record in artifact["files"]:
            assert file_record["sha256"] == _sha256(output_dir / file_record["path"])

    failed_metadata = dict(metadata)
    failed_metadata.pop("adapter_artifact")
    with pytest.raises(RuntimeError, match="before round-trip verification passes"):
        bind_adapter_artifact_lineage(
            output_dir,
            failed_metadata,
            save_report,
            {"result": "failed"},
        )
    assert "adapter_artifact" not in failed_metadata


def test_runtime_freeze_is_repaired_for_exact_verified_deltas_only():
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")

    config = transformers.GPT2Config(
        vocab_size=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=False,
    )
    base = transformers.GPT2LMHeadModel(config)
    metadata = prepare_special_tokens(base, FakeTokenizer(), _config())
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            trainable_token_indices=metadata["trainable_token_indices"],
        ),
    )
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="is frozen"):
        verify_peft_trainable_token_adapters(model, metadata)

    restored = restore_verified_selective_token_deltas(model, metadata)
    assert len(restored) == 2
    assert all(parameter.requires_grad and parameter.dtype == torch.float32 for parameter in restored)
    restored_ids = {id(parameter) for parameter in restored}
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name or id(parameter) in restored_ids for name, parameter in trainable)
    assert restored_ids.issubset({id(parameter) for _, parameter in trainable})

    input_wrapper = model.get_input_embeddings()
    output_wrapper = model.get_output_embeddings()
    input_old = input_wrapper.weight[:2].detach().clone()
    output_old = output_wrapper.weight[:2].detach().clone()
    input_new = input_wrapper.weight[2:4].detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=0.05, weight_decay=0.1
    )
    loss = model(input_ids=torch.tensor([[2, 3]]), labels=torch.tensor([[3, 2]])).loss
    loss.backward()
    optimizer.step()
    assert torch.equal(input_wrapper.weight[:2], input_old)
    assert torch.equal(output_wrapper.weight[:2], output_old)
    assert not torch.equal(input_wrapper.weight[2:4], input_new)


@pytest.mark.parametrize("tied", [False, True])
def test_merge_and_unload_preserves_selective_rows_and_output_logits(tied):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")
    config = transformers.GPT2Config(
        vocab_size=2,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=tied,
    )
    base = transformers.GPT2LMHeadModel(config)
    metadata = prepare_special_tokens(base, FakeTokenizer(), _config())
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            trainable_token_indices=metadata["trainable_token_indices"],
        ),
    )
    verify_peft_trainable_token_adapters(model, metadata)
    with torch.no_grad():
        seen = set()
        for module_name in metadata["trainable_token_indices"]:
            wrapper = next(
                module
                for name, module in model.named_modules()
                if name.endswith(module_name) and hasattr(module, "token_adapter")
            )
            delta = wrapper.token_adapter.trainable_tokens_delta["default"]
            if id(delta) not in seen:
                delta.add_(0.25)
                seen.add(id(delta))
    input_rows_before = model.get_input_embeddings().weight[2:4].detach().clone()
    probe_hidden = torch.randn(1, config.n_embd)
    logits_before = model.get_output_embeddings()(probe_hidden).detach().clone()
    merged = model.merge_and_unload(safe_merge=True)
    assert torch.equal(merged.get_input_embeddings().weight[2:4], input_rows_before)
    assert torch.allclose(merged.get_output_embeddings()(probe_hidden), logits_before)


def test_model_loader_restores_deltas_after_unsloth_peft_application():
    source = (ROOT / "Trainers" / "sft" / "src" / "model_loader.py").read_text(encoding="utf-8")
    assert source.index("FastLanguageModel.get_peft_model") < source.index(
        "restore_verified_selective_token_deltas"
    )


def test_model_loader_resolves_revision_before_unsloth_consumption():
    source = (ROOT / "Trainers" / "sft" / "src" / "model_loader.py").read_text(encoding="utf-8")
    assert "resolve_pinned_model_snapshot(" in source
    assert "FastLanguageModel.from_pretrained(**load_kwargs)" in source
    assert 'load_kwargs["revision"]' not in source


def _load_model_loader_with_fake_unsloth(monkeypatch):
    calls = []

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            model = SimpleNamespace(
                config=SimpleNamespace(_name_or_path=kwargs["model_name"])
            )
            tokenizer = FakeTokenizer()
            tokenizer.chat_template = None
            tokenizer.loaded_from = kwargs["model_name"]
            return model, tokenizer

    fake_unsloth = ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = FakeFastLanguageModel
    fake_unsloth.is_bfloat16_supported = lambda: False
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    path = ROOT / "Trainers" / "sft" / "src" / "model_loader.py"
    spec = importlib.util.spec_from_file_location("test_revision_model_loader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module, calls


def test_model_loader_absent_revision_is_exact_noop(monkeypatch):
    module, calls = _load_model_loader_with_fake_unsloth(monkeypatch)
    model, tokenizer = module.load_model_and_tokenizer(
        "example/model", load_in_4bit=False, revision=None
    )
    assert calls[0]["model_name"] == "example/model"
    assert "revision" not in calls[0]
    assert not hasattr(model, "_synaptic_revision_evidence")


def test_model_loader_consumes_pin_via_exact_local_snapshot(monkeypatch, tmp_path):
    revision = "cad0bedfdd862093a12af478cb974ab2addd0e0a"
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    hub_calls = []
    import huggingface_hub

    def fake_snapshot_download(**kwargs):
        hub_calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    module, calls = _load_model_loader_with_fake_unsloth(monkeypatch)
    model, tokenizer = module.load_model_and_tokenizer(
        "unsloth/Qwen3-4B-bnb-4bit",
        load_in_4bit=False,
        revision=revision,
        hf_token="token",
    )
    assert hub_calls == [
        {
            "repo_id": "unsloth/Qwen3-4B-bnb-4bit",
            "revision": revision,
            "token": "token",
        }
    ]
    assert calls[0]["model_name"] == str(snapshot)
    assert model.config._name_or_path == str(snapshot)
    assert tokenizer.loaded_from == str(snapshot)
    assert "revision" not in calls[0]
    assert model._synaptic_revision_evidence == {
        "requested_repo": "unsloth/Qwen3-4B-bnb-4bit",
        "requested_revision": revision,
        "resolved_snapshot_path": str(snapshot),
        "resolved_commit": revision,
        "resolution_method": "huggingface_hub.snapshot_download_local_snapshot",
    }


@pytest.mark.parametrize("failure", ["missing", "mismatch"])
def test_model_loader_pinned_snapshot_missing_or_mismatch_fails_closed(
    monkeypatch, tmp_path, failure
):
    revision = "cad0bedfdd862093a12af478cb974ab2addd0e0a"
    resolved = tmp_path / "snapshots" / (
        "0" * 40 if failure == "mismatch" else revision
    )
    if failure == "mismatch":
        resolved.mkdir(parents=True)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(resolved)
    )
    module, calls = _load_model_loader_with_fake_unsloth(monkeypatch)
    message = "missing directory" if failure == "missing" else "commit mismatch"
    with pytest.raises(RuntimeError, match=message):
        module.load_model_and_tokenizer(
            "unsloth/Qwen3-4B-bnb-4bit",
            load_in_4bit=False,
            revision=revision,
        )
    assert calls == []


def test_sft_cli_forwards_and_persists_model_revision_conditionally():
    source = (ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--model-revision"' in source
    assert "config.model.revision = args.model_revision" in source
    assert "model_revision = resolve_model_revision(config.model)" in source
    assert "revision=model_revision" in source
    assert 'model_info["revision_resolution"]' in source


def test_cached_qwen3_tokenizer_order_atomicity_and_padded_vocab_plan():
    transformers = pytest.importorskip("transformers")
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--unsloth--Qwen3-4B-bnb-4bit/snapshots"
        / "cad0bedfdd862093a12af478cb974ab2addd0e0a"
    )
    if not snapshot.is_dir():
        pytest.skip("cached Qwen3 tokenizer snapshot is not available")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prior_specials = {
        token: tokenizer.convert_tokens_to_ids(token) for token in tokenizer.all_special_tokens
    }
    model = TinyCausalLM(vocab_size=151936, hidden_size=2, tied=True)
    metadata = prepare_special_tokens(model, tokenizer, _config())
    assert [entry["token"] for entry in metadata["configured_tokens"]] == [
        "<MODE_A>",
        "<MODE_B>",
    ]
    assert metadata["new_token_ids"] == [151669, 151670]
    assert [tokenizer.encode(token, add_special_tokens=False) for token in ("<MODE_A>", "<MODE_B>")] == [
        [151669],
        [151670],
    ]
    assert metadata["resize_applied"] is False
    assert metadata["model_vocab_size_after"] == 151936
    assert all(tokenizer.convert_tokens_to_ids(token) == token_id for token, token_id in prior_specials.items())
