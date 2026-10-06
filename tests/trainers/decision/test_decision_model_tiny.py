"""DecisionModel + DecisionTrainer on a tiny random Llama (CPU, no downloads).

Exercises both readouts, the frozen-readout KL path, save/load, the HF Trainer
subclass, calibration/evaluation, `decide`, and train_decision.main end to end.
Skips when transformers/peft/tokenizers are not installed.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DECISION_DIR = REPO_ROOT / "Trainers" / "decision"
for p in (REPO_ROOT, DECISION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from decision_core.collate import CollatorConfig, DecisionCollator  # noqa: E402
from decision_core.examples import DecisionExample, write_jsonl  # noqa: E402
from decision_core.modeling import DecisionModel, DecisionModelConfig  # noqa: E402
from decision_core.prompting import HEADERS  # noqa: E402

COLORS = ["red", "green", "blue", "yellow", "purple"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def make_examples(n: int, seed: int = 0) -> list[DecisionExample]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        color = rng.choice(COLORS)
        state = f"the sky is {color} today"
        kind = ("choice", "noul", "score")[i % 3]
        if kind == "choice":
            k = rng.randint(2, len(COLORS))
            opts = rng.sample([c for c in COLORS if c != color], k - 1) + [color]
            rng.shuffle(opts)
            rows.append(DecisionExample("choice", state, "which color is mentioned", [[c, c] for c in opts],
                                        opts.index(color), task="color_choice"))
        elif kind == "noul":
            rows.append(DecisionExample("noul", state, "is the color red",
                                        [["false", "not red"], ["true", "red"]], int(color == "red"),
                                        task="is_red"))
        else:
            level = COLORS.index(color) % 3
            rows.append(DecisionExample("score", state, "how warm is the color",
                                        [["0", "cold"], ["1", "neutral"], ["2", "warm"]], level,
                                        task="warmth"))
    return rows


def build_tokenizer(save_dir: Path | None = None):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = set(re.findall(r"\w+|[^\w\s]+", " ".join([
        "<state> </state> <question type=\"choice\"> </question> <options> </options> <answer>",
        " ".join(HEADERS.values()),
        "the sky is today which color is mentioned is the color red how warm is the color",
        "false true not red cold neutral warm — . noul choice score",
        " ".join(COLORS),
    ])))
    words |= {str(i) for i in range(10)} | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   eos_token="[PAD]")
    if save_dir is not None:
        fast.save_pretrained(str(save_dir))
    return fast


def build_base(tokenizer, save_dir: Path | None = None):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
        tie_word_embeddings=True, pad_token_id=0,
    )
    lm = transformers.LlamaForCausalLM(config)
    if save_dir is not None:
        lm.save_pretrained(str(save_dir))
    return lm


def build_model(readout: str, marker_style: str = "numbers", base_dir: str = "tiny") -> DecisionModel:
    tok = build_tokenizer()
    cfg = DecisionModelConfig(hf_id=base_dir, lora_targets=LORA_TARGETS, readout=readout,
                              marker_style=marker_style, pointer_dim=32, head_dropout=0.0, max_length=256,
                              torch_dtype="float32", lora_r=4, lora_alpha=8, lora_dropout=0.0)
    return DecisionModel.wrap(cfg, build_base(tok), tok)


def batch_for(model: DecisionModel, rows, train: bool = False):
    coll = DecisionCollator(model.tokenizer, CollatorConfig(max_length=256,
                                                            marker_style=model.decision_config.marker_style),
                            train=train)
    return coll(rows)


TENSORS = ("input_ids", "attention_mask", "option_index", "answer_index", "n_options")

# The shared callbacks and lineage import shared.training_capacity, which needs the
# Unix-only `resource` module (trainers run in Linux containers).
needs_unix = pytest.mark.skipif(sys.platform == "win32", reason="shared.training_capacity needs `resource`")


@pytest.mark.parametrize("readout,style", [("pointer", "numbers"), ("letter_logits", "letters")])
def test_forward_masks_padded_slots(readout, style):
    model = build_model(readout, style)
    rows = make_examples(6)
    batch = batch_for(model, rows)
    logits = model(**{k: batch[k] for k in TENSORS})
    assert logits.shape == batch["option_index"].shape
    for i, ex in enumerate(rows):
        assert (logits[i, ex.n_options:] <= -1e3).all()
        assert torch.isfinite(logits[i, : ex.n_options]).all()


def test_marker_ids_stop_at_first_multi_token_marker():
    model = build_model("letter_logits", "numbers")
    assert model.max_marker_options == 9
    assert build_model("letter_logits", "letters").max_marker_options == 26


def test_frozen_readout_equals_letter_readout_before_training():
    # LoRA B starts at zero, so the adapted torso equals the frozen one at init.
    model = build_model("letter_logits", "numbers")
    model.eval()
    batch = batch_for(model, make_examples(6))
    tensors = {k: batch[k] for k in TENSORS}
    assert torch.allclose(model(**tensors), model.frozen_marker_logits(**tensors), atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    tok = build_tokenizer(tmp_path / "base")
    build_base(tok, tmp_path / "base")
    cfg = DecisionModelConfig(hf_id=str(tmp_path / "base"), lora_targets=LORA_TARGETS, readout="pointer",
                              pointer_dim=32, max_length=256, torch_dtype="float32", lora_r=4, lora_alpha=8,
                              headers={"noul": "Yes or no?", "choice": "Pick one.", "score": "Rate it."})
    model = DecisionModel.from_base(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    model.eval()
    model.decision_config.temperature_by_kind = {"choice": 1.7}
    model.save(tmp_path / "final")
    assert (tmp_path / "final" / "readout_head.safetensors").exists()
    loaded = DecisionModel.load(tmp_path / "final")
    assert loaded.decision_config.temperature_by_kind == {"choice": 1.7}
    # Configured prompt headers travel with the checkpoint.
    assert loaded.decision_config.headers["choice"] == "Pick one."
    batch = batch_for(model, make_examples(5))
    tensors = {k: batch[k] for k in TENSORS}
    assert torch.allclose(model(**tensors), loaded(**tensors), atol=1e-5)


@needs_unix
@pytest.mark.parametrize("readout,style,kl", [("pointer", "numbers", 0.3), ("letter_logits", "letters", 0.0)])
def test_trainer_learns_and_checkpoints_are_small(tmp_path, readout, style, kl):
    from transformers import TrainingArguments

    from decision_core.callbacks import DecisionMetricsCallback
    from decision_core.losses import LossConfig
    from decision_core.trainer import DecisionTrainer, ExampleDataset

    model = build_model(readout, style)
    rows = make_examples(16)
    coll_cfg = CollatorConfig(max_length=256, marker_style=style, seed=0)
    args = TrainingArguments(
        output_dir=str(tmp_path / "checkpoints"), max_steps=150, per_device_train_batch_size=8,
        learning_rate=1e-2, lr_scheduler_type="constant", weight_decay=0.0,
        logging_steps=10, save_steps=150, save_strategy="steps",
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0, use_cpu=True, seed=0,
    )
    trainer = DecisionTrainer(
        model=model, args=args, train_dataset=ExampleDataset(rows), eval_dataset=ExampleDataset(rows),
        data_collator=DecisionCollator(model.tokenizer, coll_cfg, train=True),
        loss_config=LossConfig(brier_weight=0.1, kl_frozen_weight=kl), head_learning_rate=1e-2,
        eval_collator=DecisionCollator(model.tokenizer, coll_cfg, train=False),
        train_lengths=[len(r.state) for r in rows],
        callbacks=[DecisionMetricsCallback(log_every_n_steps=10, output_dir=str(tmp_path))],
    )
    before = trainer.evaluate()["eval_loss"]
    trainer.train()
    after = trainer.evaluate()["eval_loss"]
    # A random 2-layer torso cannot generalise this task, so measure learning on the
    # training rows themselves, rendered canonically (training saw them shuffled).
    # The letter readout has no head and the toy LoRA is rank 4 on attention only,
    # so it moves less (observed drops: 0.15 on torch 2.11, 0.29 on torch 2.4);
    # either bar is far above fixed-eval-set noise.
    min_drop = 0.3 if readout == "pointer" else 0.1
    assert after < before - min_drop, (before, after)
    parts = [e for e in trainer.state.log_history if "loss_ce" in e]
    assert parts and ("loss_kl_frozen" in parts[-1]) == (kl > 0)
    ckpt = tmp_path / "checkpoints" / "checkpoint-150"
    assert (ckpt / "decision_config.json").exists() and (ckpt / "adapter_config.json").exists()
    assert not list(ckpt.glob("model*.safetensors")), "checkpoint must not hold base weights"


def test_calibrate_evaluate_and_decide():
    from decision_core.evaluate import evaluate, fit_temperatures, predict
    from decision_core.inference import decide

    model = build_model("pointer")
    rows = make_examples(90, seed=1)
    pred = predict(model, rows, max_length=256, batch_size=16)
    assert len(pred.logits) == 90 and all(len(l) == r.n_options for l, r in zip(pred.logits, rows))
    temps = fit_temperatures(pred, min_rows_per_kind=10)
    assert set(temps) == {"choice", "noul", "score"} and all(t > 0 for t in temps.values())
    report = evaluate(model, rows, temps, max_length=256, batch_size=16, shuffle_trials=1)
    assert report["n_examples"] == 90
    assert set(report["by_kind"]) == {"choice", "noul", "score"}
    assert 0.0 <= report["order_consistency"]["answer_change_rate"] <= 1.0

    model.decision_config.temperature_by_kind = temps
    answers = decide(model, "the sky is blue today", {
        "red": {"type": "noul", "instructions": "is the color red"},
        "color": {"type": "choice", "instructions": "which color is mentioned",
                  "criteria": {"red": "red", "blue": "blue"}},
        "warmth": {"type": "score", "instructions": "how warm is the color",
                   "criteria": ["cold", "neutral", "warm"]},
    })
    assert 0.0 <= answers["red"]["noul"] <= 1.0
    assert answers["color"]["choice"] in {"red", "blue"}
    assert sum(answers["color"]["probabilities"].values()) == pytest.approx(1.0)
    assert 0.0 <= answers["warmth"]["score"] <= 2.0


@needs_unix
def test_train_decision_main_end_to_end(tmp_path, monkeypatch):
    import train_decision
    from shared.experiment_tracking import registry

    from shared import env_bootstrap

    monkeypatch.setattr(registry, "_default_registry_path", lambda: tmp_path / "registry.jsonl")
    # The real bootstrap rewraps sys.stdout for UTF-8, which closes pytest's capture.
    monkeypatch.setattr(env_bootstrap, "init_trainer_env", lambda **_: None)
    base = tmp_path / "base"
    build_base(build_tokenizer(base), base)
    write_jsonl(tmp_path / "train.jsonl", make_examples(120))
    write_jsonl(tmp_path / "heldout.jsonl", make_examples(60, seed=7))
    registry = tmp_path / "registry.yaml"
    registry.write_text(json.dumps({"models": {"tiny-llama": {
        "hf_id": str(base), "revision": None, "loader": "causal_lm", "causal_lm_class": None,
        "torch_dtype": "float32", "max_length": 256, "attn_implementation": None,
        "lora_target_modules": LORA_TARGETS, "notes": "test torso"}}}), encoding="utf-8")
    config = {
        "model": {"registry_name": "tiny-llama", "registry_path": str(registry), "readout": "pointer",
                  "pointer_dim": 32},
        "lora": {"r": 4, "alpha": 8},
        "data": {"train_files": [str(tmp_path / "train.jsonl")], "val_fraction": 0.1,
                 "eval_files": [str(tmp_path / "heldout.jsonl")]},
        "loss": {"kl_frozen_weight": 0.3},
        "training": {"max_steps": 8, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1,
                     "logging_steps": 2, "eval_steps": 4, "save_steps": 8, "bf16": False,
                     "gradient_checkpointing": False},
        "calibration": {"min_rows_per_kind": 5},
        "evaluation": {"shuffle_trials": 1},
        "output": {"output_root": str(tmp_path / "out")},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    assert train_decision.main(["--config", str(cfg_path), "--dry-run"]) == 0
    assert train_decision.main(["--config", str(cfg_path), "--run-timestamp", "unit"]) == 0

    run = tmp_path / "out" / "unit"
    lineage = json.loads((run / "training_lineage.json").read_text(encoding="utf-8"))
    assert lineage["training_type"] == "DECISION"
    assert lineage["decision"]["readout"] == "pointer"
    assert lineage["model"]["registry_name"] == "tiny-llama"
    saved = json.loads((run / "final_model" / "decision_config.json").read_text(encoding="utf-8"))
    assert saved["temperature_by_kind"] == lineage["decision"]["temperature_by_kind"]
    report = json.loads((run / "evaluation" / "decision_eval.json").read_text(encoding="utf-8"))
    assert report["n_examples"] > 0
    assert (tmp_path / "registry.jsonl").exists()

    # Re-evaluate the saved checkpoint as a separate stage.
    assert train_decision.main(["--config", str(cfg_path), "--stage", "evaluate",
                                "--checkpoint", str(run / "final_model"), "--run-timestamp", "re"]) == 0
    assert (tmp_path / "out" / "re" / "evaluation" / "decision_eval.json").exists()
