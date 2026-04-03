from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_sft_model_loader_supports_qwen35_vl_runtime_path() -> None:
    source = (REPO_ROOT / "Trainers" / "sft" / "src" / "model_loader.py").read_text(encoding="utf-8")

    assert "FastVisionModel" in source
    assert '"qwen3.5"' in source
    assert "_is_vision_model(model_name)" in source
    assert "Extracting text tokenizer for text-only SFT training" in source
    assert "peft_api = FastVisionModel if is_vision_model else FastLanguageModel" in source


def test_train_sft_warns_about_qwen35_4bit_configs() -> None:
    source = (REPO_ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")

    assert "_is_qwen35_family" in source
    assert "_check_qwen35_4bit" in source
    assert "Qwen3.5 is not recommended for QLoRA / 4-bit training" in source
