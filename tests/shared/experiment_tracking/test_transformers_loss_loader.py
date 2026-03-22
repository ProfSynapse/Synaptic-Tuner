import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import shared.experiment_tracking.transformers_loss_loader as loss_loader
from shared.experiment_tracking.transformers_loss_loader import load_transformers_loss_model


def _fake_model():
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config = SimpleNamespace(use_cache=True)
    return model


def test_load_transformers_loss_model_loads_lora_adapter_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    base_model = _fake_model()
    peft_model = _fake_model()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = "<eos>"

    with patch.object(loss_loader, "_default_loss_dtype", return_value=torch.float16):
        with patch(
            "shared.experiment_tracking.transformers_loss_loader.AutoModelForCausalLM.from_pretrained",
            return_value=base_model,
        ) as mock_model_loader:
            with patch(
                "shared.experiment_tracking.transformers_loss_loader.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as mock_tokenizer_loader:
                with patch.object(loss_loader, "PeftModel") as mock_peft_class:
                    mock_peft_class.from_pretrained.return_value = peft_model
                    model, loaded_tokenizer = load_transformers_loss_model(model_dir, device="cpu")

    assert model is peft_model
    assert loaded_tokenizer is tokenizer
    mock_model_loader.assert_called_once_with(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mock_tokenizer_loader.assert_called_once_with(str(model_dir), trust_remote_code=True, use_fast=True)
    mock_peft_class.from_pretrained.assert_called_once_with(base_model, str(model_dir), is_trainable=False)
    assert peft_model.to.called
    assert peft_model.eval.called
    assert peft_model.config.use_cache is False
    assert tokenizer.pad_token == tokenizer.eos_token


def test_load_transformers_loss_model_loads_merged_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    model = _fake_model()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = "<eos>"

    with patch.object(loss_loader, "_default_loss_dtype", return_value=torch.float16):
        with patch(
            "shared.experiment_tracking.transformers_loss_loader.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as mock_model_loader:
            with patch(
                "shared.experiment_tracking.transformers_loss_loader.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as mock_tokenizer_loader:
                with patch.object(loss_loader, "PeftModel") as mock_peft_class:
                    loaded_model, loaded_tokenizer = load_transformers_loss_model(model_dir, device="cpu")

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    mock_model_loader.assert_called_once_with(
        str(model_dir),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mock_tokenizer_loader.assert_called_once_with(str(model_dir), trust_remote_code=True, use_fast=True)
    mock_peft_class.from_pretrained.assert_not_called()
    assert model.to.called
    assert model.eval.called
