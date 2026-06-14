"""Registry validation — bad blocks must raise ValueError naming the offending key.

CONTEXT: ``Trainers/embedding/src/registry.py`` is the SSOT for how each base
model is loaded/prompted/adapted. Its loader validates every block and raises
``ValueError`` (naming the offending registry key) on any schema violation, so a
typo fails loudly at load time. This suite drives every documented validation
rule (CONTRACTS §1.2) plus the happy-path seed registry.

Imports are ISOLATED via ``_isolated_import.load_embedding_src`` — the embedding
``registry`` module is loaded by explicit file path under a namespaced key so it
never collides with the bare ``import data_loader`` / ``import registry`` that
sibling trainer tests (sft) rely on under combined pytest collection.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolated_import import load_embedding_src  # noqa: E402

registry = load_embedding_src("registry")


def _write_registry(tmp_path: Path, models: dict) -> Path:
    import yaml

    path = tmp_path / "model_registry.yaml"
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return path


# A minimal valid block we can perturb one field at a time.
_BASE_BLOCK = {
    "hf_id": "org/model",
    "family": "bert",
    "embedding_type": "bi_encoder",
    "pooling": "mean",
    "normalize": True,
    "max_seq_length": 512,
    "default_dim": 768,
}


# ---------------------------------------------------------------------------
# Happy path: the committed seed registry validates fully
# ---------------------------------------------------------------------------

def test_seed_registry_loads_and_validates():
    specs = registry.load_registry()
    # The 4 seed models from CONTRACTS §1.1.
    assert set(specs) == {"bge-base-en", "e5-base", "gte-base", "qwen3-embedding-0.6b"}
    bge = specs["bge-base-en"]
    assert bge.hf_id == "BAAI/bge-base-en-v1.5"
    assert bge.family == "bert"
    # Frozen dataclass — cannot mutate after validation.
    with pytest.raises(Exception):
        bge.hf_id = "x"  # type: ignore[misc]


def test_list_models_is_sorted():
    assert registry.list_models() == sorted(registry.list_models())


def test_get_spec_unknown_name_raises_keyerror():
    with pytest.raises(KeyError) as exc:
        registry.get_spec("does-not-exist")
    assert "does-not-exist" in str(exc.value)


# ---------------------------------------------------------------------------
# Per-field validation: each violation names the offending registry key
# ---------------------------------------------------------------------------

def test_bad_family_raises_naming_key(tmp_path):
    block = {**_BASE_BLOCK, "family": "reranker"}
    path = _write_registry(tmp_path, {"my-model": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    msg = str(exc.value)
    assert "my-model" in msg and "family" in msg


def test_bad_pooling_raises_naming_key(tmp_path):
    block = {**_BASE_BLOCK, "pooling": "max"}
    path = _write_registry(tmp_path, {"pool-model": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "pool-model" in str(exc.value) and "pooling" in str(exc.value)


def test_bad_embedding_type_raises_naming_key(tmp_path):
    block = {**_BASE_BLOCK, "embedding_type": "tri_encoder"}
    path = _write_registry(tmp_path, {"et-model": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "et-model" in str(exc.value) and "embedding_type" in str(exc.value)


def test_missing_hf_id_raises(tmp_path):
    block = {k: v for k, v in _BASE_BLOCK.items() if k != "hf_id"}
    path = _write_registry(tmp_path, {"no-id": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "no-id" in str(exc.value) and "hf_id" in str(exc.value)


def test_empty_hf_id_raises(tmp_path):
    block = {**_BASE_BLOCK, "hf_id": ""}
    path = _write_registry(tmp_path, {"blank-id": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "blank-id" in str(exc.value)


def test_unknown_key_raises_naming_key(tmp_path):
    # config-driven discipline: a typo'd key fails loudly.
    block = {**_BASE_BLOCK, "poling": "mean"}  # typo of "pooling"
    path = _write_registry(tmp_path, {"typo-model": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "typo-model" in str(exc.value) and "poling" in str(exc.value)


def test_matryoshka_dim_exceeding_default_raises(tmp_path):
    block = {**_BASE_BLOCK, "default_dim": 512, "matryoshka_dims": [768, 256]}
    path = _write_registry(tmp_path, {"mrl-too-big": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "mrl-too-big" in str(exc.value)


def test_matryoshka_dims_not_descending_raises(tmp_path):
    block = {**_BASE_BLOCK, "default_dim": 768, "matryoshka_dims": [256, 512]}
    path = _write_registry(tmp_path, {"mrl-unsorted": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "mrl-unsorted" in str(exc.value)


def test_negative_default_dim_raises(tmp_path):
    block = {**_BASE_BLOCK, "default_dim": -1}
    path = _write_registry(tmp_path, {"neg-dim": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "neg-dim" in str(exc.value)


def test_prompt_required_without_prompts_raises(tmp_path):
    block = {**_BASE_BLOCK, "prompt_required": True, "query_prompt": "", "passage_prompt": ""}
    path = _write_registry(tmp_path, {"needs-prompt": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "needs-prompt" in str(exc.value)


def test_decoder_wrong_task_type_raises(tmp_path):
    block = {
        **_BASE_BLOCK,
        "family": "decoder",
        "pooling": "last_token",
        "lora_task_type": "SEQ_CLS",  # wrong; decoder needs FEATURE_EXTRACTION
    }
    path = _write_registry(tmp_path, {"bad-decoder": block})
    with pytest.raises(ValueError) as exc:
        registry.load_registry(path)
    assert "bad-decoder" in str(exc.value)


def test_decoder_missing_task_type_warns_and_defaults(tmp_path):
    block = {**_BASE_BLOCK, "family": "decoder", "pooling": "last_token"}
    path = _write_registry(tmp_path, {"warn-decoder": block})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = registry.load_registry(path)
    assert specs["warn-decoder"].lora_task_type == "FEATURE_EXTRACTION"
    assert any("FEATURE_EXTRACTION" in str(w.message) for w in caught)


def test_empty_models_mapping_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        registry.load_registry(path)


def test_missing_registry_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.load_registry(tmp_path / "nope.yaml")


def test_resolved_fast_path_id_prefers_mirror(tmp_path):
    block = {**_BASE_BLOCK, "fast_path_hf_id": "unsloth/mirror"}
    path = _write_registry(tmp_path, {"mirror-model": block})
    spec = registry.load_registry(path)["mirror-model"]
    assert spec.resolved_fast_path_id() == "unsloth/mirror"


def test_resolved_fast_path_id_falls_back_to_hf_id(tmp_path):
    path = _write_registry(tmp_path, {"plain-model": dict(_BASE_BLOCK)})
    spec = registry.load_registry(path)["plain-model"]
    assert spec.resolved_fast_path_id() == "org/model"
