"""Qwen3.5 SFT support as implemented in this lineage.

The SFT loader and trainer are hash-locked members of the offline SFT worker
closure. Qwen3.5 is loaded through the shared FastLanguageModel path: a
processor returned for the vision-language checkpoint is preserved, and the
memory-efficient loss guard rejects the stock ``ForConditionalGeneration`` loss
fallback. Avoiding 4-bit (QLoRA) Qwen3.5 runs is expressed in the checked-in
configuration rather than by a trainer warning.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.trainers.sft.test_model_revision import _load, _unsloth_loss


REPO_ROOT = Path(__file__).resolve().parents[3]
QWEN35_MODEL = "Qwen/Qwen3.5-4B"


def test_sft_model_loader_supports_qwen35_vl_runtime_path(monkeypatch, tmp_path: Path) -> None:
    module, calls = _load(monkeypatch, None, tmp_path)

    class TextTokenizer:
        def __len__(self):
            return 3

    processor = SimpleNamespace(chat_template="x", tokenizer=TextTokenizer())

    def from_pretrained(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(config=SimpleNamespace(_name_or_path=kwargs["model_name"])), processor

    monkeypatch.setattr(module.FastLanguageModel, "from_pretrained", from_pretrained)

    _, returned = module.load_model_and_tokenizer(QWEN35_MODEL, load_in_4bit=False)

    assert [call["model_name"] for call in calls] == [QWEN35_MODEL]
    assert calls[0]["load_in_4bit"] is False
    assert returned is processor

    unsloth_loss = _unsloth_loss()
    module.require_unsloth_memory_efficient_loss(
        SimpleNamespace(loss_function=unsloth_loss),
        loss_mapping={"ForCausalLM": unsloth_loss, "ForConditionalGeneration": unsloth_loss},
    )

    def stock_conditional_generation_loss():
        return None

    with pytest.raises(RuntimeError, match="SFT_MEMORY_EFFICIENT_LOSS_REQUIRED"):
        module.require_unsloth_memory_efficient_loss(
            SimpleNamespace(loss_function=stock_conditional_generation_loss),
            loss_mapping={
                "ForCausalLM": unsloth_loss,
                "ForConditionalGeneration": stock_conditional_generation_loss,
            },
        )


def _qwen35_training_configs():
    configs = []
    for path in sorted((REPO_ROOT / "Trainers" / "recipes").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        model = data.get("model") if isinstance(data.get("model"), dict) else {}
        steps = " ".join(str(step) for step in ((data.get("run") or {}).get("steps") or []))
        if "Qwen3.5" in str(model.get("name", "")) or "--model-name Qwen/Qwen3.5" in steps:
            configs.append((path, model.get("load_in_4bit"), steps))
    for path in sorted((REPO_ROOT / "Trainers" / "cloud" / "experiments").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        training = ((data or {}).get("experiment") or {}).get("training") or {}
        if "Qwen3.5" in str(training.get("model_name", "")):
            configs.append((path, training.get("load_in_4bit"), ""))
    return configs


def test_qwen35_training_configs_disable_4bit_loading() -> None:
    configs = _qwen35_training_configs()
    assert len(configs) >= 5

    for path, load_in_4bit, steps in configs:
        relative = path.relative_to(REPO_ROOT)
        assert "--load-in-4bit" not in steps.replace("--no-load-in-4bit", ""), relative
        assert load_in_4bit is False or "--no-load-in-4bit" in steps, relative
