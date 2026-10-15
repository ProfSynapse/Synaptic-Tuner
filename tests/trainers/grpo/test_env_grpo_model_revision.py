"""env-GRPO honours model.model_revision and training.chat_template_kwargs.

Runs ``train_env_grpo.run`` end to end with fake ``transformers`` / ``trl``
modules and a tiny in-memory dataset, so no torch, TRL, model or network is
needed. Checks that the tokenizer and the trainer's model load are pinned to
the configured revision, and that the configured chat-template kwargs reach the
dataset prompt render and the rollout builder.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
GRPO = ROOT / "Trainers" / "grpo"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@dataclasses.dataclass
class FakeGRPOConfig:
    output_dir: str = ""
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 2
    max_prompt_length: int = 512
    max_completion_length: int = 64
    temperature: float = 1.0
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    num_train_epochs: int = 1
    max_steps: int = -1
    beta: float = 0.0
    logging_steps: int = 1
    save_steps: int = 1
    save_total_limit: int = 1
    report_to: str = "none"
    fp16: bool = False
    bf16: bool = False
    optim: str = "adamw_torch"
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    seed: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    model_init_kwargs: Optional[Dict[str, Any]] = None


class _RowsDataset:
    """The slice of ``datasets.Dataset`` that ``run`` uses (no fingerprint hashing)."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def map(self, fn, desc=None):
        return _RowsDataset([fn(row) for row in self.rows])

    def select(self, indices):
        return _RowsDataset([self.rows[i] for i in indices])


class FakeTokenizer:
    def __init__(self):
        self.template_kwargs: List[Dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.template_kwargs.append(dict(kwargs))
        return "|".join(str(m.get("content")) for m in messages) + "|<gen>"

    def save_pretrained(self, _path):
        pass


@pytest.fixture
def env_grpo(monkeypatch, tmp_path):
    """Import train_env_grpo with fake heavy deps; restore sys.modules afterwards."""
    calls: Dict[str, Any] = {"tokenizer_loads": [], "trainer_kwargs": None, "rollout_kwargs": None}
    tokenizer = FakeTokenizer()

    def from_pretrained(name, **kwargs):
        calls["tokenizer_loads"].append((name, kwargs))
        return tokenizer

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)

    class FakeGRPOTrainer:
        def __init__(self, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def train(self):
            pass

        def save_model(self, _path):
            pass

    trl = types.ModuleType("trl")
    trl.GRPOConfig = FakeGRPOConfig
    trl.GRPOTrainer = FakeGRPOTrainer

    callbacks = types.ModuleType("src.training_callbacks")
    callbacks.DASHBOARD_AVAILABLE = False
    callbacks.RICH_AVAILABLE = False
    callbacks.LiveDashboardCallback = lambda **kw: object()
    callbacks.MetricsTableCallback = lambda **kw: object()

    tracking = types.ModuleType("shared.experiment_tracking.adapters")
    tracking.register_grpo_run = lambda *a, **kw: None

    saved_src = {name: mod for name, mod in sys.modules.items() if name == "src" or name.startswith("src.")}
    for name in saved_src:
        del sys.modules[name]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "shared.experiment_tracking.adapters", tracking)
    sys.modules["src.training_callbacks"] = callbacks
    monkeypatch.syspath_prepend(str(GRPO))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))

    try:
        spec = importlib.util.spec_from_file_location("train_env_grpo_under_test", GRPO / "train_env_grpo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rows = _RowsDataset([
            {
                "prompt_messages": [{"role": "user", "content": "hello"}],
                "resolved_environment_config": {"loop": {"max_turns": 2}},
                "task_context": {"k": "v"},
                "metadata": {"scenario": "s"},
            }
        ])
        monkeypatch.setattr(module, "load_env_rollout_dataset", lambda **kw: rows)
        monkeypatch.setattr(module, "filter_env_rollout_dataset", lambda ds, **kw: ds)
        monkeypatch.setattr(module, "format_dataset_for_env_grpo_with_options", lambda ds, **kw: ds)
        monkeypatch.setattr(
            module,
            "detect_openenv_runtime_support",
            lambda: {"has_rollout_func": True, "has_env_mask": True},
        )

        def fake_build_rollout_func(**kwargs):
            calls["rollout_kwargs"] = kwargs
            return lambda prompts, trainer: {}

        monkeypatch.setattr(module, "build_rollout_func", fake_build_rollout_func)
        yield module, calls, tokenizer
    finally:
        for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(saved_src)


def _write_config(tmp_path, *, model_revision=None, chat_template_kwargs=None, extra_args=None):
    config = yaml.safe_load((GRPO / "configs" / "env_config.yaml").read_text(encoding="utf-8"))
    config["model"]["model_revision"] = model_revision
    config["training"]["chat_template_kwargs"] = chat_template_kwargs
    config["training"]["output_dir"] = str(tmp_path / "out")
    if extra_args is not None:
        config["training"]["extra_args"] = extra_args
    config["lora"]["enabled"] = False
    config["env_training"]["debug_rollouts_path"] = None
    path = tmp_path / "env_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(module, config_path, tmp_path, *extra):
    args = module.parse_args(["--config", str(config_path), "--output-dir", str(tmp_path / "run"), *extra])
    return module.run(args)


def test_configured_revision_pins_tokenizer_and_model_load(env_grpo, tmp_path):
    module, calls, _tokenizer = env_grpo
    _run(module, _write_config(tmp_path, model_revision=REVISION), tmp_path)

    assert calls["tokenizer_loads"] == [("professorsynapse/Nexus-Quark-L2.5.28", {"revision": REVISION})]
    trainer_kwargs = calls["trainer_kwargs"]
    assert trainer_kwargs["model"] == "professorsynapse/Nexus-Quark-L2.5.28"
    assert trainer_kwargs["args"].model_init_kwargs == {"revision": REVISION}


def test_cli_revision_override_wins(env_grpo, tmp_path):
    module, calls, _tokenizer = env_grpo
    _run(module, _write_config(tmp_path, model_revision="a" * 40), tmp_path, "--model-revision", REVISION)

    assert calls["tokenizer_loads"][0][1] == {"revision": REVISION}
    assert calls["trainer_kwargs"]["args"].model_init_kwargs == {"revision": REVISION}


def test_unset_revision_leaves_default_branch(env_grpo, tmp_path):
    module, calls, _tokenizer = env_grpo
    _run(module, _write_config(tmp_path), tmp_path)

    assert calls["tokenizer_loads"][0][1] == {"revision": None}
    assert calls["trainer_kwargs"]["args"].model_init_kwargs is None


def test_revision_merges_into_extra_model_init_kwargs_and_refuses_conflicts(env_grpo, tmp_path):
    module, calls, _tokenizer = env_grpo
    extra = {"top_p": 0.9, "model_init_kwargs": {"dtype": "bfloat16"}}
    _run(module, _write_config(tmp_path, model_revision=REVISION, extra_args=extra), tmp_path)
    assert calls["trainer_kwargs"]["args"].model_init_kwargs == {"dtype": "bfloat16", "revision": REVISION}

    conflicting = {"model_init_kwargs": {"revision": "b" * 40}}
    with pytest.raises(ValueError, match="conflicts with model.model_revision"):
        _run(module, _write_config(tmp_path, model_revision=REVISION, extra_args=conflicting), tmp_path)


def test_chat_template_kwargs_reach_dataset_render_and_rollout(env_grpo, tmp_path):
    module, calls, tokenizer = env_grpo
    _run(module, _write_config(tmp_path, chat_template_kwargs={"enable_thinking": False}), tmp_path)

    assert tokenizer.template_kwargs
    assert all(kw == {"enable_thinking": False} for kw in tokenizer.template_kwargs)
    assert calls["rollout_kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls["rollout_kwargs"]["use_vllm"] is False


def test_invalid_chat_template_kwargs_are_refused(env_grpo, tmp_path):
    module, _calls, _tokenizer = env_grpo
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        _run(module, _write_config(tmp_path, chat_template_kwargs={"add_generation_prompt": True}), tmp_path)
