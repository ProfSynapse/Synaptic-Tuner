"""CLI parsing + runner-level config-hash gating for the batch verbs.

Location: tests/batch/test_cli_and_config.py

Fast, no model. Asserts the parser registers the two verbs and their flags, and
that the runner refuses to --resume across a changed config with a clear error.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tuner.cli.parser import create_parser  # noqa: E402
from tuner.batch import runner as batch_runner  # noqa: E402
from tuner.batch.persistence import ConfigMismatchError  # noqa: E402
from tuner.batch.engines.base import GenerateResult  # noqa: E402
from tuner.handlers.batch_generate_handler import BatchGenerateHandler  # noqa: E402


def test_parser_registers_batch_generate():
    parser = create_parser()
    args = parser.parse_args(
        [
            "batch-generate",
            "--prompts", "p.jsonl",
            "--model", "some/model",
            "--model-revision", "abc123",
            "--tokenizer-revision", "tok456",
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
            "--suppress-token", "<turn|>",
            "--suppress-token", "<|tool_response>",
            "--json-schema", "schema.json",
            "--structured-output-backend", "xgrammar",
            "--structured-output-disable-any-whitespace",
            "--expected-vllm-version", "0.23.0",
            "--min-compute-capability", "8.0",
            "--tensor-parallel-size", "2",
            "--max-num-seqs", "64",
            "--max-num-batched-tokens", "8192",
            "--max-model-len", "2048",
            "--limit-mm-per-prompt", '{"image":0,"audio":0}',
            "--gpu-memory-utilization", "0.90",
            "--resume",
            "--sync-every", "100",
            "--sync-cmd", "echo hi",
        ]
    )
    assert args.command == "batch-generate"
    assert args.prompts == "p.jsonl"
    assert args.model == "some/model"
    assert args.model_revision == "abc123"
    assert args.tokenizer_revision == "tok456"
    assert args.out_dir == "out"
    assert args.engine == "hf-batched"
    assert args.batch_size == 32
    assert args.max_new_tokens == 48
    assert args.seed == 7
    assert args.do_sample is True
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.stop_strings == ["\n\n", "END"]
    assert args.suppress_tokens == ["<turn|>", "<|tool_response>"]
    assert args.json_schema == "schema.json"
    assert args.structured_output_backend == "xgrammar"
    assert args.structured_output_disable_any_whitespace is True
    assert args.expected_vllm_version == "0.23.0"
    assert args.min_compute_capability == "8.0"
    assert args.tensor_parallel_size == 2
    assert args.max_num_seqs == 64
    assert args.max_num_batched_tokens == 8192
    assert args.max_model_len == 2048
    assert args.limit_mm_per_prompt == '{"image":0,"audio":0}'
    assert args.gpu_memory_utilization == 0.90
    assert args.trust_remote_code is False
    assert args.resume is True
    assert args.sync_every == 100
    assert args.sync_cmd == "echo hi"


def test_parser_suppress_tokens_default_to_none():
    args = create_parser().parse_args(["batch-generate"])
    assert args.suppress_tokens is None


def test_handler_forwards_suppress_tokens(monkeypatch, tmp_path, capsys):
    received = {}

    def fake_run_batch_generate(**kwargs):
        received.update(kwargs)
        return {"newly_processed": 0, "artifact": "out.jsonl"}

    monkeypatch.setattr(
        batch_runner, "run_batch_generate", fake_run_batch_generate
    )
    args = Namespace(
        prompts=str(tmp_path / "prompts.jsonl"),
        out_dir=str(tmp_path / "out"),
        model="model",
        engine="vllm",
        suppress_tokens=["<turn|>", "<|tool_response>"],
        json=False,
    )
    assert BatchGenerateHandler(args).handle() == 0
    assert received["suppress_tokens"] == ["<turn|>", "<|tool_response>"]
    capsys.readouterr()


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


def test_vllm_provenance_and_resume_hash_cover_schema_and_scheduler(monkeypatch, tmp_path):
    engine_calls = []

    class FakeVLLMEngine:
        def generate(self, items, *, batch_size, on_oom=None):
            return [
                GenerateResult(
                    id=item.id,
                    completion_text='{"answer":"ok"}',
                    completion_token_ids=[1, 2],
                    prompt_token_ids_sha256="a" * 64,
                    prompt_token_len=3,
                    finish_reason="stop",
                    passthrough=item.passthrough,
                )
                for item in items
            ]

        def provenance(self):
            return {
                "vllm_version": "0.23.0",
                "vllm_batch_invariant": True,
                "structured_outputs": True,
            }

        def close(self):
            pass

    def fake_engine(*args, **kwargs):
        engine_calls.append(kwargs)
        return FakeVLLMEngine()

    monkeypatch.setattr(batch_runner, "get_generate_engine", fake_engine)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": "hi"}) + "\n")
    out = tmp_path / "out"
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    batch_runner.run_batch_generate(
        prompts_path=prompts,
        out_dir=out,
        model="m",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        engine="vllm",
        json_schema=schema,
        structured_output_backend="xgrammar",
        structured_output_disable_any_whitespace=True,
        expected_vllm_version="0.23.0",
        min_compute_capability="8.0",
        tensor_parallel_size=1,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        gpu_memory_utilization=0.90,
        suppress_tokens=["<turn|>", "<|tool_response>"],
        batch_size=20,
        dtype="bfloat16",
        log=lambda message: None,
    )

    provenance = json.loads((out / "provenance.json").read_text())
    config = provenance["config"]
    canonical_schema = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    import hashlib

    assert config["json_schema_sha256"] == hashlib.sha256(canonical_schema).hexdigest()
    assert config["structured_output_backend"] == "xgrammar"
    assert config["structured_output_disable_any_whitespace"] is True
    assert config["tokenizer_revision"] == "tokenizer-sha"
    assert config["expected_vllm_version"] == "0.23.0"
    assert config["min_compute_capability"] == "8.0"
    assert config["vllm_batch_invariant"] is True
    assert config["tensor_parallel_size"] == 1
    assert config["max_num_seqs"] == 64
    assert config["max_num_batched_tokens"] == 8192
    assert config["max_model_len"] == 2048
    assert config["limit_mm_per_prompt"] == {"image": 0, "audio": 0}
    assert config["gpu_memory_utilization"] == 0.90
    assert config["suppress_tokens"] == ["<turn|>", "<|tool_response>"]
    assert config["trust_remote_code"] is False
    assert config["prompts_sha256"] == hashlib.sha256(prompts.read_bytes()).hexdigest()
    assert config["batch_size"] == 20
    assert engine_calls[0]["suppress_tokens"] == [
        "<turn|>",
        "<|tool_response>",
    ]

    completion = json.loads((out / "completions.jsonl").read_text())
    assert completion["prompt_token_ids_sha256"] == "a" * 64
    assert completion["prompt_sha256"] == hashlib.sha256(b"hi").hexdigest()

    prompts.write_text(json.dumps({"id": "a", "prompt": "changed"}) + "\n")
    with pytest.raises(ConfigMismatchError):
        batch_runner.run_batch_generate(
            prompts_path=prompts,
            out_dir=out,
            model="m",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            engine="vllm",
            json_schema=schema,
            structured_output_backend="xgrammar",
            structured_output_disable_any_whitespace=True,
            expected_vllm_version="0.23.0",
            min_compute_capability="8.0",
            tensor_parallel_size=1,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "audio": 0},
            gpu_memory_utilization=0.90,
            suppress_tokens=["<turn|>", "<|tool_response>"],
            batch_size=20,
            dtype="bfloat16",
            resume=True,
            log=lambda message: None,
        )
    prompts.write_text(json.dumps({"id": "a", "prompt": "hi"}) + "\n")

    with pytest.raises(ConfigMismatchError):
        batch_runner.run_batch_generate(
            prompts_path=prompts,
            out_dir=out,
            model="m",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            engine="vllm",
            json_schema=schema,
            structured_output_backend="xgrammar",
            structured_output_disable_any_whitespace=True,
            expected_vllm_version="0.23.0",
            min_compute_capability="8.0",
            tensor_parallel_size=1,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "audio": 0},
            gpu_memory_utilization=0.90,
            suppress_tokens=["<turn|>"],
            batch_size=20,
            dtype="bfloat16",
            resume=True,
            log=lambda message: None,
        )

    with pytest.raises(ConfigMismatchError):
        batch_runner.run_batch_generate(
            prompts_path=prompts,
            out_dir=out,
            model="m",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            engine="vllm",
            json_schema=schema,
            structured_output_backend="xgrammar",
            structured_output_disable_any_whitespace=True,
            expected_vllm_version="0.23.0",
            min_compute_capability="8.0",
            tensor_parallel_size=1,
            max_num_seqs=32,
            max_num_batched_tokens=8192,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "audio": 0},
            gpu_memory_utilization=0.90,
            suppress_tokens=["<turn|>", "<|tool_response>"],
            batch_size=20,
            dtype="bfloat16",
            resume=True,
            log=lambda message: None,
        )


def test_runner_rejects_suppress_tokens_for_non_vllm(tmp_path):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt": "hi"}) + "\n")
    with pytest.raises(ValueError, match="require engine='vllm'"):
        batch_runner.run_batch_generate(
            prompts_path=prompts,
            out_dir=tmp_path / "out",
            model="m",
            engine="hf-batched",
            suppress_tokens=["<turn|>"],
            log=lambda message: None,
        )
