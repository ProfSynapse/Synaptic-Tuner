import json
from argparse import Namespace

import pytest
import yaml

from tuner.handlers.local_run_handler import LocalRunHandler


def test_local_run_sft_config_compiles_repo_relative_dataset(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "unit-local-sft",
                "provider": "local_docker",
                "job": {"transfer": "copy"},
                "run": {"method": "sft"},
                "model": {"name": "Qwen/Qwen3.5-2B", "load_in_4bit": False},
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
                "artifacts": {
                    "output_root": "toolset-training-artifacts/runs/local_docker/sft/unit-local-sft",
                    "run_timestamp": "unit",
                },
            }
        ),
        encoding="utf-8",
    )

    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))

    assert plan["transfer"] == "copy"
    assert plan["command"][:3] == ["python", "train_sft.py", "--model-name"]
    local_file_index = plan["command"].index("--local-file") + 1
    assert plan["command"][local_file_index].startswith("../../")
    assert plan["host_artifact_path"].name == "unit"


def _compile_local_command(
    tmp_path,
    *,
    method,
    trainer,
    training,
    lora=None,
    aux_head=None,
    tokenizer=None,
    model_revision=None,
):
    """Compile a local-docker recipe and return the built trainer command list."""
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    recipe = {
        "name": "unit-local",
        "provider": "local_docker",
        "job": {"transfer": "copy"},
        "run": {"method": method, "trainer": trainer},
        "model": {"name": "Qwen/Qwen3.5-2B", "load_in_4bit": False},
        "dataset": {"local_file": str(dataset)},
        "training": training,
        "artifacts": {
            "output_root": "toolset-training-artifacts/runs/local_docker/unit",
            "run_timestamp": "unit",
        },
    }
    if lora is not None:
        recipe["lora"] = lora
    if aux_head is not None:
        recipe["aux_head"] = aux_head
    if tokenizer is not None:
        recipe["model"]["tokenizer"] = tokenizer
    if model_revision is not None:
        recipe["model"]["revision"] = model_revision
    config.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))
    return plan["command"]


def test_local_run_forwards_seed(tmp_path):
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "seed": 1234},
    )
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "1234"


def test_local_run_forwards_seed_zero(tmp_path):
    # seed=0 must survive forwarding; _append_flag skips only None, not falsy 0.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "seed": 0},
    )
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "0"


def test_local_run_omits_seed_when_absent(tmp_path):
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--seed" not in command


def test_local_run_omits_beta_for_sft(tmp_path):
    # beta is gated to dpo/kto; the SFT trainer has no --beta flag, so a stray beta
    # in an sft recipe must not be forwarded as a command argument.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "beta": 0.5},
    )
    assert "--beta" not in command


# Run-control flags with no experimental meaning that dpo/kto do not expose as CLI
# args. The handler gates these to sft so dpo/kto commands omit them (scope v2 (A)).
RUN_CONTROL_SFT_ONLY_FLAGS = (
    "--save-steps",
    "--save-total-limit",
    "--load-in-4bit",
    "--no-load-in-4bit",
    "--no-dashboard",
    "--quiet",
)

# LoRA budget flags that now have parity on all three trainers (scope v2 (B)). The
# recipe's LoRA budget is load-bearing (the identical-budget confound control), so
# these MUST flow to dpo/kto, not be gated.
LORA_PARITY_FLAGS = (
    "--lora-r",
    "--lora-alpha",
    "--lora-dropout",
    "--lora-target-modules",
)

_LORA_BLOCK = {"r": 64, "alpha": 128, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}


def test_local_run_dispatches_dpo_method(tmp_path):
    # local-run dispatches dpo through the generic builder (no longer SFT-only).
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.05},
    )
    assert command[:2] == ["python", "train_dpo.py"]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--beta") + 1] == "0.05"


def test_local_run_dispatches_kto_method(tmp_path):
    command = _compile_local_command(
        tmp_path, method="kto", trainer="Trainers/kto/train_kto.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.5},
    )
    assert command[:2] == ["python", "train_kto.py"]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--beta") + 1] == "0.5"


def test_local_run_dpo_omits_run_control_flags(tmp_path):
    # Run-control flags (dashboard/quiet/save/4bit) have no experimental meaning and
    # the dpo trainer's argparse rejects them, so a dpo command must omit them.
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.05, "save_steps": 50},
        lora=_LORA_BLOCK,
    )
    for flag in RUN_CONTROL_SFT_ONLY_FLAGS:
        assert flag not in command, f"dpo command leaked run-control flag {flag}"


def test_local_run_dpo_carries_lora_budget(tmp_path):
    # The LoRA budget is load-bearing (identical-budget confound control): a dpo
    # recipe's lora block MUST flow to the trainer, not silently fall back to the
    # trainer config default. Verified end-to-end via the parity CLI flags.
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.05},
        lora=_LORA_BLOCK,
    )
    for flag in LORA_PARITY_FLAGS:
        assert flag in command, f"dpo command dropped load-bearing LoRA flag {flag}"
    assert command[command.index("--lora-r") + 1] == "64"
    assert command[command.index("--lora-alpha") + 1] == "128"
    assert command[command.index("--lora-target-modules") + 1] == "q_proj,v_proj"


def test_local_run_kto_carries_lora_budget(tmp_path):
    command = _compile_local_command(
        tmp_path, method="kto", trainer="Trainers/kto/train_kto.py",
        training={"max_steps": 1, "seed": 7, "beta": 0.5},
        lora=_LORA_BLOCK,
    )
    assert command[command.index("--lora-r") + 1] == "64"
    assert command[command.index("--lora-alpha") + 1] == "128"


def test_local_run_sft_still_emits_its_flags(tmp_path):
    # The sft path is byte-unchanged: it still receives the gated run-control flags.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "save_steps": 50},
    )
    assert "--quiet" in command
    assert "--no-dashboard" in command
    assert "--save-steps" in command


def test_local_run_forwards_beta_zero_for_dpo(tmp_path):
    # beta uses is-not-None semantics (mirroring --seed): an explicit beta: 0.0 is
    # honored, not silently dropped to the trainer default. (Provenance: matrix
    # never uses 0, but a silent swap is the cardinal sin this guards against.)
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.0},
    )
    assert "--beta" in command
    assert command[command.index("--beta") + 1] == "0.0"


def test_local_run_omits_beta_when_absent_for_dpo(tmp_path):
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "seed": 7},
    )
    assert "--beta" not in command


def test_local_run_rejects_unregistered_method(tmp_path):
    # The dispatch guard still rejects methods outside {sft, dpo, kto} (absent an
    # explicit run.command), rather than silently building a command.
    import pytest

    from tuner.handlers.local_run_handler import LocalRunError

    with pytest.raises(LocalRunError):
        _compile_local_command(
            tmp_path, method="grpo", trainer="Trainers/grpo/train_grpo.py",
            training={"max_steps": 1},
        )


def test_local_run_sft_serializes_chat_template_kwargs_as_json(tmp_path):
    # chat_template_kwargs is a nested mapping, so it cannot ride the scalar
    # _append_flag path; the handler JSON-encodes it onto --chat-template-kwargs.
    # The trainer parses the JSON back into config.training.chat_template_kwargs.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert "--chat-template-kwargs" in command
    payload = command[command.index("--chat-template-kwargs") + 1]
    assert json.loads(payload) == {"enable_thinking": False}


def test_local_run_sft_omits_chat_template_kwargs_when_absent(tmp_path):
    # Byte-identical for recipes that do not set the key: no flag emitted.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--chat-template-kwargs" not in command


def test_local_run_dpo_omits_chat_template_kwargs(tmp_path):
    # --chat-template-kwargs is sft-only: the dpo/kto trainers template internally
    # via TRL and expose no such flag, so a stray key in a dpo recipe must not be
    # forwarded (argparse would reject it).
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.05, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert "--chat-template-kwargs" not in command


def test_local_run_sft_forwards_tokenizer_config_as_json(tmp_path):
    tokenizer_config = {
        "additional_special_tokens": ["<MODE_A>", "<MODE_B>"],
        "existing_token_policy": "error",
        "initialization": "mean_existing_rows",
        "train_new_embedding_rows": True,
        "train_new_lm_head_rows": True,
        "verify_tokenizer_roundtrip": True,
        "verify_adapter_roundtrip": True,
    }
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
        tokenizer=tokenizer_config,
    )
    assert "--tokenizer-config" in command
    payload = command[command.index("--tokenizer-config") + 1]
    assert json.loads(payload) == tokenizer_config


def test_local_run_omits_tokenizer_config_when_absent(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--tokenizer-config" not in command


def test_local_run_sft_forwards_model_revision_and_omits_it_when_absent(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
        model_revision="cad0bedfdd862093a12af478cb974ab2addd0e0a",
    )
    index = command.index("--model-revision")
    assert command[index + 1] == "cad0bedfdd862093a12af478cb974ab2addd0e0a"
    absent = _compile_local_command(
        tmp_path,
        method="sft",
        trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert "--model-revision" not in absent


def test_local_run_non_sft_does_not_emit_unsupported_model_revision(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="dpo",
        trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.05},
        model_revision="cad0bedfdd862093a12af478cb974ab2addd0e0a",
    )
    assert "--model-revision" not in command


@pytest.mark.parametrize(
    "invalid_revision",
    [False, True, 123, "", " cad0bedfdd862093a12af478cb974ab2addd0e0a", "cad0bedf"],
)
def test_local_run_rejects_invalid_model_revision_types_and_values(
    tmp_path, invalid_revision
):
    from tuner.handlers.local_run_handler import LocalRunError

    with pytest.raises(LocalRunError, match="model.revision"):
        _compile_local_command(
            tmp_path,
            method="sft",
            trainer="Trainers/sft/train_sft.py",
            training={"max_steps": 1},
            model_revision=invalid_revision,
        )


def test_local_run_rejects_falsy_non_mapping_tokenizer_config(tmp_path):
    import pytest

    from tuner.handlers.local_run_handler import LocalRunError

    with pytest.raises(LocalRunError, match="model.tokenizer must be a mapping"):
        _compile_local_command(
            tmp_path,
            method="sft",
            trainer="Trainers/sft/train_sft.py",
            training={"max_steps": 1},
            tokenizer=[],
        )


def test_local_run_dpo_omits_tokenizer_config(tmp_path):
    command = _compile_local_command(
        tmp_path,
        method="dpo",
        trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.05},
        tokenizer={"additional_special_tokens": ["<MODE_A>"]},
    )
    assert "--tokenizer-config" not in command


def test_local_run_sft_omits_aux_head_flags_when_absent(tmp_path):
    # Byte-identical for recipes with no aux_head block and no training.prompt_render:
    # zero --aux-head-* flags emitted (the whole point of the gated forwarding).
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
    )
    assert not any(arg.startswith("--aux-head-") for arg in command)
    assert "--no-aux-head-enabled" not in command


def test_local_run_sft_forwards_aux_head_block(tmp_path):
    # An aux_head block forwards field-by-field; enabled/freeze_base use the
    # tri-state --flag/--no-flag form; prompt_render rides training, not aux_head.
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "prompt_render": "prompt_completion"},
        aux_head={
            "enabled": True,
            "layer": 35,
            "token_position": "end_of_prompt",
            "target_field": "score",
            "loss": "bce",
            "head_type": "linear",
            "out_activation": "sigmoid",
            "input_norm": "layernorm",
            "freeze_base": False,
            "lm_loss_weight": 1.0,
            "head_lr": 0.005,
        },
    )
    assert "--aux-head-enabled" in command
    assert "--no-aux-head-freeze-base" in command
    assert command[command.index("--aux-head-layer") + 1] == "35"
    assert command[command.index("--aux-head-token-position") + 1] == "end_of_prompt"
    assert command[command.index("--aux-head-target-field") + 1] == "score"
    assert command[command.index("--aux-head-loss") + 1] == "bce"
    assert command[command.index("--aux-head-head-type") + 1] == "linear"
    assert command[command.index("--aux-head-out-activation") + 1] == "sigmoid"
    assert command[command.index("--aux-head-input-norm") + 1] == "layernorm"
    assert command[command.index("--aux-head-lm-loss-weight") + 1] == "1.0"
    assert command[command.index("--aux-head-head-lr") + 1] == "0.005"
    assert command[command.index("--aux-head-prompt-render") + 1] == "prompt_completion"


def test_local_run_sft_forwards_falsy_aux_head_values(tmp_path):
    # A set-but-falsy lm_loss_weight (0.0) and an explicit enabled: false must
    # still forward (provenance: the trainer must run the recipe's exact values).
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1},
        aux_head={"enabled": False, "lm_loss_weight": 0.0},
    )
    assert "--no-aux-head-enabled" in command
    assert command[command.index("--aux-head-lm-loss-weight") + 1] == "0.0"


def test_local_run_sft_forwards_prompt_render_without_aux_head_block(tmp_path):
    # prompt_render is a training-config knob, forwarded independently of the
    # aux_head block (a plain SFT recipe may want the faithful render).
    command = _compile_local_command(
        tmp_path, method="sft", trainer="Trainers/sft/train_sft.py",
        training={"max_steps": 1, "prompt_render": "prompt_completion"},
    )
    assert command[command.index("--aux-head-prompt-render") + 1] == "prompt_completion"


def test_local_run_dpo_omits_aux_head_flags(tmp_path):
    # aux_head forwarding is sft-only: a stray aux_head block in a dpo recipe must
    # not be forwarded (the dpo trainer exposes no --aux-head-* flags).
    command = _compile_local_command(
        tmp_path, method="dpo", trainer="Trainers/dpo/train_dpo.py",
        training={"max_steps": 1, "beta": 0.05, "prompt_render": "prompt_completion"},
        aux_head={"enabled": True, "layer": 35},
    )
    assert not any(arg.startswith("--aux-head-") for arg in command)
