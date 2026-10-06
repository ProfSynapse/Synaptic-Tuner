"""Every launcher emits only flags its target trainer's argparse accepts.

A launcher that builds trainer argv (HF Jobs, RunPod, local-run, RTX, Mac,
the flywheel loops) is exercised per supported method with every optional
setting populated. The resulting argv is parsed by that trainer's real
argument parser, loaded from source with ``ast`` so no torch/unsloth/trl import
is needed. An unknown flag makes argparse exit, which fails the test with the
offending flag in the message, so a launcher can no longer pass a flag the
trainer silently rejects at job start.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List

import pytest

from tests.cloud.test_hf_jobs_backend import (  # noqa: F401  (autouse fixture)
    _cloud_config,
    verified_source_preparation,
)

ROOT = Path(__file__).resolve().parents[2]

# trainer key -> (source file, parser function, helper functions it calls)
TRAINER_PARSERS = {
    "sft": ("Trainers/sft/train_sft.py", "parse_args", ("_parse_init_lora_weights",)),
    "kto": ("Trainers/kto/train_kto.py", "build_arg_parser", ()),
    "dpo": ("Trainers/dpo/train_dpo.py", "build_arg_parser", ()),
    "grpo": ("Trainers/grpo/train_grpo.py", "parse_args", ()),
    "env_grpo": ("Trainers/grpo/train_env_grpo.py", "parse_args", ()),
    "embedding": ("Trainers/embedding/train_embedding.py", "parse_args", ()),
    "ace_step": ("Trainers/ace_step/train_ace_step.py", "parse_args", ()),
    "mlx_sft": ("Trainers/mlx_sft_mac/train_sft.py", "parse_args", ()),
}
SCRIPT_TO_TRAINER = {
    "Trainers/sft/train_sft.py": "sft",
    "Trainers/kto/train_kto.py": "kto",
    "Trainers/dpo/train_dpo.py": "dpo",
    "Trainers/grpo/train_grpo.py": "grpo",
    "Trainers/grpo/train_env_grpo.py": "env_grpo",
    "Trainers/embedding/train_embedding.py": "embedding",
    "Trainers/ace_step/train_ace_step.py": "ace_step",
}


def _load_parser(key: str) -> Callable[[List[str]], argparse.Namespace]:
    relative, name, helpers = TRAINER_PARSERS[key]
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {name, *helpers}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == wanted, f"{relative}: missing {wanted}"
    namespace: Dict[str, object] = {
        "argparse": argparse, "json": json, "os": os, "Path": path.__class__,
        "__file__": str(path),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    function = namespace[name]
    if name == "build_arg_parser":
        return lambda argv: function().parse_args(argv)
    if key == "mlx_sft":  # parse_args() reads sys.argv
        def parse(argv):
            saved = sys.argv
            sys.argv = [str(path), *argv]
            try:
                return function()
            finally:
                sys.argv = saved
        return parse
    return lambda argv: function(argv)


def assert_trainer_accepts(trainer: str, argv: List[str], context: str) -> argparse.Namespace:
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            return _load_parser(trainer)(list(argv))
    except SystemExit:
        pytest.fail(
            f"{context}: {TRAINER_PARSERS[trainer][0]} rejects launcher argv\n"
            f"{stderr.getvalue().strip()}\nargv={argv}"
        )


def _split_trainer_argv(command: str, script_name: str) -> List[str]:
    """Return the argv after ``script_name`` in a ``&&``-joined shell command."""
    step = next(part for part in command.split(" && ") if f" {script_name}" in part)
    return shlex.split(step.split(script_name, 1)[1])


# ---------------------------------------------------------------------------
# HF Jobs
# ---------------------------------------------------------------------------

HF_ALL_OPTIONAL = dict(
    dataset_name="org/data",
    dataset_file="org/data/train.jsonl",
    gradient_accumulation_steps=2,
    seed=7,
    beta=0.1,
    chat_template_kwargs={"enable_thinking": False},
    save_steps=10,
    save_total_limit=2,
    max_steps=5,
    max_seq_length=1024,
    load_in_4bit=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    use_dora=True,
    use_rslora=True,
    init_lora_weights="gaussian",
    lora_target_modules=["q_proj", "v_proj"],
    evolutionary_enabled=True,
    evolutionary_candidates=4,
    evolutionary_eval_batch_size=2,
    evolutionary_validation_config="configs/fitness/tool_calling.yaml",
    evolutionary_strategy="gradient_noise",
    evolutionary_noise_scale=0.03,
    evolutionary_max_grad_norm=1.0,
    evolutionary_scale_factors=[0.5, 1.0],
    evolutionary_selection_method="best",
    evolutionary_min_improvement=0.01,
    evolutionary_min_relative_improvement=0.0,
    evolutionary_noise_floor_epsilon=1e-6,
    evolutionary_eval_frequency=5,
    evolutionary_warmup_steps=10,
    evolutionary_cache_baseline=True,
    evolutionary_log_candidates=True,
    evolutionary_log_selected=False,
    publish_final_model=True,
    publish_target_repo="org/model",
    split_dataset=True,
    test_size=0.2,
    validation_group_key="metadata.scenario",
)
# Validation split settings exist only for the SFT/KTO/DPO trainers.
NO_SPLIT = dict(split_dataset=None, test_size=None, validation_group_key=None)


def _assert_split_forwarded(args: argparse.Namespace) -> None:
    assert (args.split_dataset, args.test_size, args.validation_group_key) == (
        True, 0.2, "metadata.scenario",
    )


def _hf_command(method: str, **overrides) -> str:
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend

    trainer_dir = ROOT / "Trainers" / method
    config_path = trainer_dir / "configs" / ("env_config.yaml" if method == "grpo" else "config.yaml")
    settings = {**HF_ALL_OPTIONAL, **overrides}
    config = _cloud_config(method=method, config_path=config_path, trainer_dir=trainer_dir, **settings)
    return HFJobsBackend(ROOT)._build_training_command(config, timestamp="20260101_000000")


@pytest.mark.parametrize("method", ["sft", "kto", "dpo"])
def test_hf_jobs_trainer_argv_is_accepted(method):
    command = _hf_command(method)
    args = assert_trainer_accepts(method, _split_trainer_argv(command, f"train_{method}.py"), f"HF Jobs {method}")
    _assert_split_forwarded(args)


@pytest.mark.parametrize("method", ["sft", "kto", "dpo"])
def test_cloud_pipeline_split_overrides_reach_the_trainer(method):
    # cloud / cloud-pipeline: --train-* split overrides -> HF Jobs trainer argv.
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.cli.parser import create_parser
    from tuner.handlers.cloud_train_handler import CloudTrainHandler

    cli = create_parser().parse_args([
        "cloud-pipeline", "--json", "--train-split-dataset", "--train-test-size", "0.2",
        "--train-validation-group-key", "metadata.scenario",
    ])
    trainer_dir = ROOT / "Trainers" / method
    config = _cloud_config(
        method=method, config_path=trainer_dir / "configs" / "config.yaml", trainer_dir=trainer_dir,
    )
    config = CloudTrainHandler(args=cli)._apply_training_overrides(config)
    command = HFJobsBackend(ROOT)._build_training_command(config, timestamp="20260101_000000")
    args = assert_trainer_accepts(method, _split_trainer_argv(command, f"train_{method}.py"), f"cloud-pipeline {method}")
    _assert_split_forwarded(args)


@pytest.mark.parametrize("method", ["sft", "kto", "dpo"])
def test_run_experiment_dataset_split_reaches_the_trainer(method, monkeypatch, tmp_path):
    # run-experiment: experiment.dataset split settings -> HF training stage -> trainer argv.
    from shared.experiment_tracking.experiment_spec import ExperimentSpec
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.handlers.stages import hf_training_stage

    trainer_dir = ROOT / "Trainers" / method
    backend = HFJobsBackend(ROOT)
    built = {}

    class _Stop(Exception):
        pass

    def _prepare_source(config, run_id):
        built["command"] = backend._build_training_command(config, timestamp="20260101_000000")
        raise _Stop

    backend.load_config = lambda m: _cloud_config(
        method=m, config_path=trainer_dir / "configs" / "config.yaml", trainer_dir=trainer_dir,
    )
    backend.prepare_source = _prepare_source
    monkeypatch.setattr(hf_training_stage.TrainingBackendRegistry, "get", lambda *a, **k: backend)
    spec = ExperimentSpec.from_dict({"experiment": {
        "name": "contract", "provider": "hf_jobs", "method": method,
        "dataset": {
            "source": "org/data", "file": "train.jsonl", "split_dataset": True,
            "test_size": 0.2, "validation_group_key": "metadata.scenario",
        },
        "training": {"model_name": "org/model"},
    }})
    tracking = SimpleNamespace(
        project_context=None, base_dir=tmp_path, tracking_uri=lambda path: "tracking://x",
    )
    runner = hf_training_stage.HFTrainingStageRunner(repo_root=ROOT, tracking_service=tracking)
    monkeypatch.setattr(runner, "_recover_existing_training", lambda **_: None)
    with pytest.raises(_Stop):
        runner.run(spec, SimpleNamespace(experiment_id="exp-1", source_transport_state=None))
    args = assert_trainer_accepts(
        method, _split_trainer_argv(built["command"], f"train_{method}.py"), f"run-experiment {method}",
    )
    _assert_split_forwarded(args)


def test_hf_jobs_env_grpo_argv_is_accepted():
    # max_seq_length has no env-GRPO meaning and is refused by the builder.
    command = _hf_command("grpo", max_seq_length=None, **NO_SPLIT)
    argv = _split_trainer_argv(command, "train_env_grpo.py")
    args = assert_trainer_accepts("env_grpo", argv, "HF Jobs env-GRPO")
    assert (args.seed, args.save_steps, args.save_total_limit) == (7, 10, 2)


@pytest.mark.parametrize("method", ["embedding", "ace_step"])
def test_hf_jobs_refuses_methods_without_its_trainer_contract(method):
    from tuner.core.exceptions import CloudProviderError

    with pytest.raises(CloudProviderError, match="HF Jobs training supports"):
        _hf_command(method)


# ---------------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sft", "kto"])
def test_runpod_trainer_argv_is_accepted(method):
    from tuner.backends.training.cloud.runpod_backend import RunPodBackend

    backend = RunPodBackend.__new__(RunPodBackend)
    backend.repo_root = ROOT
    config = _cloud_config(
        method=method, provider="runpod", artifact_mount_path="/workspace",
        publish_final_model=True, publish_target_repo="org/model",
        split_dataset=True, test_size=0.2, validation_group_key="metadata.scenario",
    )
    command = backend._build_startup_command(config, {})
    args = assert_trainer_accepts(method, _split_trainer_argv(command, f"train_{method}.py"), f"RunPod {method}")
    _assert_split_forwarded(args)


# ---------------------------------------------------------------------------
# local-run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sft", "kto", "dpo"])
def test_local_run_trainer_argv_is_accepted(method, tmp_path):
    from tests.cli.test_local_run_handler import _compile_local_command

    model_config = {"size": "7b", "load_in_4bit": True}
    if method == "sft":  # only the SFT trainer pins a Hub revision
        model_config["revision"] = "a" * 40
    command = _compile_local_command(
        tmp_path,
        method=method,
        trainer=f"Trainers/{method}/train_{method}.py",
        training={
            "batch_size": 2, "gradient_accumulation": 2, "learning_rate": 1e-5,
            "seed": 3, "num_epochs": 1, "max_steps": 4, "save_steps": 2,
            "save_total_limit": 1, "beta": 0.1, "tier": "quick",
            "resume_from_checkpoint": "ckpt", "max_seq_length": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_render": "full_conversation",
        },
        lora={
            "r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"],
            "init_lora_weights": "gaussian", "use_dora": True, "use_rslora": True,
        },
        aux_head={
            "enabled": False, "freeze_base": True, "layer": 1, "token_position": "last",
            "target_field": "target", "loss": "bce", "head_type": "linear",
            "out_activation": "sigmoid", "input_norm": "none", "lm_loss_weight": 0.0,
            "head_lr": 1e-4,
        },
        dataset_config={
            "split_dataset": True, "test_size": 0.2, "validation_group_key": "metadata.scenario",
            "name": "org/data", "file": "train.jsonl",
        },
        model_config=model_config,
    )
    script = f"train_{method}.py"
    argv = command[command.index(script) + 1:]
    args = assert_trainer_accepts(method, argv, f"local-run {method}")
    _assert_split_forwarded(args)


@pytest.mark.parametrize("method", ["kto", "dpo"])
def test_local_run_refuses_model_revision_for_trainers_without_a_pin(method, tmp_path):
    from tests.cli.test_local_run_handler import _compile_local_command
    from tuner.handlers.local_run_handler import LocalRunError

    with pytest.raises(LocalRunError, match="model.revision is supported only"):
        _compile_local_command(
            tmp_path, method=method, trainer=f"Trainers/{method}/train_{method}.py",
            training={"max_steps": 1}, model_config={"revision": "a" * 40},
        )


def test_local_run_sft_runtime_profile_argv_is_accepted(tmp_path):
    from argparse import Namespace

    import yaml

    from tuner.handlers.local_run_handler import LocalRunHandler

    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(yaml.safe_dump({
        "name": "profiled-local-sft",
        "provider": "local_docker",
        "job": {"runtime_profile": "qwen35-sft-v1", "transfer": "copy"},
        "run": {"method": "sft", "dry_run": True},
        "model": {
            "name": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "load_in_4bit": False,
        },
        "dataset": {"local_file": str(dataset)},
        "training": {"max_steps": 1},
        "artifacts": {
            "output_root": "toolset-training-artifacts/runs/local_docker/sft/profiled",
            "run_timestamp": "unit",
        },
    }), encoding="utf-8")
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    command = handler._compile(config, handler._load_yaml(config))["command"]
    argv = command[command.index("train_sft.py") + 1:]
    args = assert_trainer_accepts("sft", argv, "local-run sft runtime profile")
    assert args.runtime_profile_name == "qwen35-sft-v1"


def test_local_run_refuses_grpo_without_explicit_command(tmp_path):
    from tests.cli.test_local_run_handler import _compile_local_command
    from tuner.handlers.local_run_handler import LocalRunError

    with pytest.raises(LocalRunError, match="supports run.method"):
        _compile_local_command(
            tmp_path, method="grpo", trainer="Trainers/grpo/train_grpo.py",
            training={"batch_size": 2, "learning_rate": 1e-5}, lora={"r": 8},
        )


# ---------------------------------------------------------------------------
# cloud-run / local-run recipes that spell out trainer commands
# ---------------------------------------------------------------------------

BASENAME_TO_TRAINER = {Path(script).name: key for script, key in SCRIPT_TO_TRAINER.items()}


def _recipe_trainer_invocations():
    import re

    import yaml

    for recipe in sorted((ROOT / "Trainers" / "recipes").rglob("*.yaml")):
        data = yaml.safe_load(recipe.read_text(encoding="utf-8")) or {}
        run = data.get("run") or {}
        steps = list(run.get("steps") or [])
        command = run.get("command")
        if isinstance(command, list):
            steps.append(shlex.join(str(part) for part in command))
        elif isinstance(command, str):
            steps.append(command)
        for step in steps:
            for match in re.finditer(r"\b(train_[a-z_]+\.py)\b", str(step)):
                if match.group(1) not in BASENAME_TO_TRAINER:
                    continue
                tail = re.split(r"&&|;|\|\||\||>", str(step)[match.end():], maxsplit=1)[0]
                yield recipe.relative_to(ROOT).as_posix(), match.group(1), shlex.split(tail)


RECIPE_INVOCATIONS = list(_recipe_trainer_invocations())


def test_recipes_with_trainer_commands_are_covered():
    assert RECIPE_INVOCATIONS, "expected checked-in recipes that spell out trainer commands"


@pytest.mark.parametrize(
    "recipe, script, argv", RECIPE_INVOCATIONS,
    ids=[f"{recipe}:{script}" for recipe, script, _ in RECIPE_INVOCATIONS],
)
def test_recipe_trainer_commands_are_accepted(recipe, script, argv):
    assert_trainer_accepts(BASENAME_TO_TRAINER[script], argv, f"{recipe} run step")


# ---------------------------------------------------------------------------
# SFT runtime / packaged / protected-smoke argv
# ---------------------------------------------------------------------------


# Placeholders for computed values of choice-constrained flags.
_CHOICE_VALUES = {"--aux-head-prompt-render": "full_conversation"}


def _argv_literals(path: Path, function_name: str, variable: str) -> List[str]:
    """The trainer argv a function assembles in ``variable``.

    Reads ``variable = [...]`` plus ``variable.extend((...))`` /
    ``variable.append(...)`` literals. String constants are kept (the script
    path is skipped); computed values become a placeholder every int/float/str
    option accepts.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    groups: List[list] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
            and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)
        ):
            groups.append(node.value.elts)
        elif (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == variable
            and node.func.attr in {"extend", "append"} and node.args
        ):
            arg = node.args[0]
            groups.append(arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg])
    argv: List[str] = []
    for elements in groups:
        for element in elements:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                if not element.value.endswith(".py"):
                    argv.append(element.value)
            elif not isinstance(element, ast.Starred):
                previous = argv[-1] if argv else ""
                argv.append(_CHOICE_VALUES.get(previous, "1"))
    # Drop the interpreter placeholder that precedes the script path.
    while argv and not argv[0].startswith("--"):
        argv.pop(0)
    return argv


@pytest.mark.parametrize(
    "relative, function_name, variable",
    [
        ("tuner/runtime/packaged_sft_execution.py", "_invocation_spec", "args"),
        ("tuner/cloud/hf_training_smoke_remote_entry.py", "run", "command"),
    ],
)
def test_sft_runtime_launchers_argv_is_accepted(relative, function_name, variable):
    # The shared hyperparameter tail (runtime_v1._append_sft_arguments) is
    # covered by test_runtime_v1_trainer_invocation_is_accepted.
    argv = _argv_literals(ROOT / relative, function_name, variable)
    assert len(argv) > 4, f"{relative}:{function_name} has no trainer flags"
    assert_trainer_accepts("sft", argv, f"{relative}:{function_name}")


def test_offline_worker_flag_allowlist_is_accepted_by_sft():
    from tuner.runtime import offline_sft_worker as worker

    values = {
        "--init-lora-weights": "gaussian",
        "--aux-head-prompt-render": "full_conversation",
        "--runtime-v1-dataset-schema": "syntunia-sft-row/v2",
        "--runtime-v1-dataset-format": "messages",
    }
    for flag in sorted(worker._VALUE_FLAGS):
        assert_trainer_accepts("sft", [flag, values.get(flag, "1")], f"offline worker {flag}")
    for flag in sorted(worker._BOOLEAN_FLAGS):
        assert_trainer_accepts("sft", [flag], f"offline worker {flag}")


@pytest.mark.parametrize("helper", ["_raw_trainer_arguments_from_runtime", "_message_trainer_arguments_from_runtime"])
def test_runtime_v1_trainer_invocation_is_accepted(helper, tmp_path):
    from tests.trainers.sft import test_runtime_v1

    _, trainer_arguments = getattr(test_runtime_v1, helper)(tmp_path)
    assert_trainer_accepts("sft", trainer_arguments, f"runtime_v1 {helper}")


# ---------------------------------------------------------------------------
# RTX / Mac local backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, config_name, trainer",
    [
        ("sft", "config.yaml", "sft"),
        ("kto", "config.yaml", "kto"),
        ("dpo", "config.yaml", "dpo"),
        ("grpo", "env_config.yaml", "env_grpo"),
        ("grpo", "config.yaml", "grpo"),
        ("embedding", "config.yaml", "embedding"),
        ("ace_step", "config.yaml", "ace_step"),
    ],
)
def test_rtx_backend_trainer_argv_is_accepted(method, config_name, trainer, monkeypatch):
    from tuner.backends.training import rtx_backend
    from tuner.core.config import TrainingConfig

    captured = {}

    class _Popen:
        def __init__(self, cmd, cwd=None):
            captured["cmd"] = cmd

        def wait(self):
            return 0

    monkeypatch.setattr(rtx_backend.subprocess, "Popen", _Popen)
    monkeypatch.setitem(
        sys.modules, "env_runtime",
        SimpleNamespace(ensure_local_openenv_runtime=lambda *a, **k: "python"),
    )
    trainer_dir = ROOT / "Trainers" / method
    config = TrainingConfig(
        method=method, platform="rtx", config_path=trainer_dir / "configs" / config_name,
        trainer_dir=trainer_dir, model_name="m", dataset_file="d", epochs=1,
        batch_size=1, learning_rate=1e-5,
    )
    rtx_backend.RTXBackend(ROOT).execute(config, "python")
    cmd = captured["cmd"]
    assert_trainer_accepts(trainer, cmd[2:], f"RTX {method}/{config_name}")
    assert cmd[1] == Path(TRAINER_PARSERS[trainer][0]).name


def test_mac_backend_trainer_argv_is_accepted(monkeypatch):
    from tuner.backends.training import mac_backend
    from tuner.core.config import TrainingConfig

    captured = {}

    class _Popen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stdout = io.StringIO("")

        def wait(self):
            return 0

    monkeypatch.setattr(mac_backend.subprocess, "Popen", _Popen)
    monkeypatch.setitem(sys.modules, "Trainers.shared.ui.training_progress", None)
    trainer_dir = ROOT / "Trainers" / "mlx_sft_mac"
    config = TrainingConfig(
        method="sft", platform="mac", config_path=trainer_dir / "config" / "config.yaml",
        trainer_dir=trainer_dir, model_name="m", dataset_file="d", epochs=1,
        batch_size=1, learning_rate=1e-5,
    )
    mac_backend.MacBackend(ROOT).execute(config, "python")
    assert_trainer_accepts("mlx_sft", captured["cmd"][2:], "Mac sft")


# ---------------------------------------------------------------------------
# Flywheel loops
# ---------------------------------------------------------------------------


def test_flywheel_orchestrator_trainer_argv_is_accepted():
    source = (ROOT / "shared" / "flywheel" / "orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            script, argv = node.elts
            if (
                isinstance(script, ast.Constant)
                and isinstance(script.value, str)
                and script.value in SCRIPT_TO_TRAINER
                and isinstance(argv, ast.List)
            ):
                flags = [elt.value for elt in argv.elts if isinstance(elt, ast.Constant)]
                calls.append((script.value, flags))
    assert calls, "no trainer invocations found in shared/flywheel/orchestrator.py"
    for script, flags in calls:
        argv = [flag if flag.startswith("--") else flag for flag in flags]
        argv = [value for flag in argv for value in (flag, "x")]  # each flag takes a value
        assert_trainer_accepts(SCRIPT_TO_TRAINER[script], argv, f"flywheel orchestrator {script}")


@pytest.mark.parametrize("trainer_type", ["sft", "kto"])
def test_flywheel_experiment_loop_trainer_argv_is_accepted(trainer_type, tmp_path, monkeypatch):
    from shared.flywheel import experiment_loop
    from shared.flywheel.experiment_config import (
        TRAINER_BOOLEAN_OVERRIDE_FLAGS,
        TRAINER_OVERRIDE_FLAGS,
        ExperimentConfig,
    )

    captured = {}

    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise experiment_loop.subprocess.TimeoutExpired(cmd, 0)

    monkeypatch.setattr(experiment_loop.subprocess, "run", _run)
    loop = experiment_loop.ExperimentLoop(ExperimentConfig(
        trainer_type=trainer_type, output_dir=str(tmp_path), dataset_path="data.jsonl",
    ))
    overrides = {key: 1 for key in TRAINER_OVERRIDE_FLAGS[trainer_type]}
    overrides.update({key: True for key in TRAINER_BOOLEAN_OVERRIDE_FLAGS[trainer_type]})
    loop._run_single_experiment("exp-1", overrides)
    cmd = captured["cmd"]
    script = experiment_loop._trainer_script(trainer_type)
    argv = cmd[cmd.index(script) + 1:]
    assert "--config" not in argv
    assert_trainer_accepts(trainer_type, argv, f"flywheel experiment loop {trainer_type}")


def test_flywheel_experiment_loop_refuses_hyperparameters_without_a_flag(tmp_path):
    from shared.flywheel.experiment_config import ExperimentConfig
    from shared.flywheel.experiment_loop import ExperimentLoop

    config = ExperimentConfig(output_dir=str(tmp_path), search_space={"weight_decay": [0.0, 0.1]})
    assert any("weight_decay" in issue for issue in config.validate())
    with pytest.raises(ValueError, match="weight_decay"):
        ExperimentLoop(config)._run_single_experiment("exp-1", {"weight_decay": 0.1})
