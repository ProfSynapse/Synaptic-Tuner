"""CLI parsing + runner-level config-hash gating for the batch verbs.

Location: tests/batch/test_cli_and_config.py

Fast, no model. Asserts the parser registers the two verbs and their flags, and
that the runner refuses to --resume across a changed config with a clear error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tuner.cli.parser import create_parser  # noqa: E402
from tuner.batch import runner as batch_runner  # noqa: E402
from tuner.batch.persistence import ConfigMismatchError  # noqa: E402


def test_parser_registers_batch_generate():
    parser = create_parser()
    args = parser.parse_args(
        [
            "batch-generate",
            "--prompts", "p.jsonl",
            "--model", "some/model",
            "--model-revision", "abc123",
            "--out-dir", "out",
            "--engine", "hf-batched",
            "--batch-size", "32",
            "--max-new-tokens", "48",
            "--seed", "7",
            "--do-sample",
            "--temperature", "0.7",
            "--top-p", "0.9",
            "--stop-string", "\n\n",
            "--stop-string", "END",
            "--resume",
            "--sync-every", "100",
            "--sync-cmd", "echo hi",
        ]
    )
    assert args.command == "batch-generate"
    assert args.prompts == "p.jsonl"
    assert args.model == "some/model"
    assert args.model_revision == "abc123"
    assert args.out_dir == "out"
    assert args.engine == "hf-batched"
    assert args.batch_size == 32
    assert args.max_new_tokens == 48
    assert args.seed == 7
    assert args.do_sample is True
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.stop_strings == ["\n\n", "END"]
    assert args.resume is True
    assert args.sync_every == 100
    assert args.sync_cmd == "echo hi"


def test_parser_registers_batch_capture():
    parser = create_parser()
    args = parser.parse_args(
        [
            "batch-capture",
            "--rows", "r.jsonl",
            "--model", "some/model",
            "--model-revision", "def456",
            "--out-dir", "out",
            "--engine", "vllm",
            "--layers", "20,22",
            "--persist-dtype", "bfloat16",
            "--batch-size", "8",
        ]
    )
    assert args.command == "batch-capture"
    assert args.rows == "r.jsonl"
    assert args.model_revision == "def456"
    assert args.layers == "20,22"
    assert args.persist_dtype == "bfloat16"
    assert args.engine == "vllm"
    assert args.batch_size == 8


def test_engine_choice_is_validated():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["batch-generate", "--engine", "nope"])


def test_runner_refuses_resume_on_changed_config(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": "hi"}) + "\n")
    out = tmp_path / "out"

    # Seed a checkpoint by faking a run's config directly (no model needed):
    from tuner.batch.persistence import RunCheckpoint
    RunCheckpoint.load_or_create(
        out,
        {
            "verb": "batch-generate",
            "model": "m",
            "model_revision": None,
            "engine": "hf-batched",
            "max_new_tokens": 48,
            "min_new_tokens": 0,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "seed": 1,
            "extra_eos_tokens": None,
            "stop": None,
            "dtype": None,
        },
        resume=False,
    )

    # Now resume with a DIFFERENT seed -> must refuse before touching a model.
    with pytest.raises(ConfigMismatchError):
        batch_runner.run_batch_generate(
            prompts_path=prompts, out_dir=out, model="m", seed=2, resume=True,
            log=lambda m: None,
        )
