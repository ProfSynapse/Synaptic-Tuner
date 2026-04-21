from pathlib import Path

import pytest

safetensors_torch = pytest.importorskip("safetensors.torch")

from Trainers.cloud.scripts.filter_lora_adapter import filter_adapter_directory


def test_filter_adapter_directory_keeps_only_language_model_tensors(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source_dir / "README.md").write_text("adapter", encoding="utf-8")

    safetensors_torch.save_file(
        {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": __import__("torch").zeros((2, 2)),
            "base_model.model.model.audio_tower.layers.0.self_attn.q_proj.linear.lora_A.weight": __import__("torch").zeros((2, 2)),
            "base_model.model.model.vision_tower.encoder.layers.0.mlp.down_proj.linear.lora_A.weight": __import__("torch").zeros((2, 2)),
        },
        str(source_dir / "adapter_model.safetensors"),
    )

    dest_dir = tmp_path / "dest"
    summary = filter_adapter_directory(
        source_dir=source_dir,
        dest_dir=dest_dir,
        include_substrings=[".language_model."],
        exclude_substrings=[],
    )

    assert summary["kept_tensors"] == 1
    assert summary["dropped_tensors"] == 2
    assert (dest_dir / "adapter_config.json").exists()
    assert (dest_dir / "README.md").exists()

    from safetensors import safe_open

    with safe_open(str(dest_dir / "adapter_model.safetensors"), framework="pt") as handle:
        assert list(handle.keys()) == [
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        ]
