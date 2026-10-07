import importlib
import io

import Evaluator.cli as evaluator_cli


parse_args = evaluator_cli.parse_args


class _CaptureLikeStream(io.StringIO):
    def __init__(self, *, tty: bool = False):
        super().__init__()
        self.tty = tty
        self.reconfigured_to = None

    def isatty(self):
        return self.tty

    def reconfigure(self, *, encoding):
        self.reconfigured_to = encoding


def test_evaluator_cli_accepts_vllm_backend():
    args = parse_args(["--backend", "vllm", "--model", "finetuned", "--scenario", "behavior_prompts.yaml"])
    assert args.backend == "vllm"


def test_evaluator_cli_accepts_optional_loss_flags():
    args = parse_args(
        [
            "--backend",
            "unsloth",
            "--model",
            "/tmp/final_model",
            "--preset",
            "full",
            "--with-loss",
            "--loss-dataset-name",
            "professorsynapse/claudesidian-synthetic-dataset",
            "--loss-dataset-file",
            "train.jsonl",
            "--loss-output-jsonl",
            "/tmp/per_example_losses.jsonl",
        ]
    )
    assert args.with_loss is True
    assert args.loss_dataset_name == "professorsynapse/claudesidian-synthetic-dataset"
    assert args.loss_dataset_file == "train.jsonl"


def test_import_reload_preserves_caller_owned_streams(monkeypatch):
    stdout = _CaptureLikeStream()
    stderr = _CaptureLikeStream()
    monkeypatch.setattr(evaluator_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluator_cli.sys, "stderr", stderr)

    reloaded = importlib.reload(evaluator_cli)

    assert reloaded.sys.stdout is stdout
    assert reloaded.sys.stderr is stderr
    assert not stdout.closed
    assert not stderr.closed
    stdout.write("still writable")
    stderr.write("still writable")


def test_utf8_bootstrap_reconfigures_in_place_without_owning_streams(monkeypatch):
    stdout = _CaptureLikeStream(tty=True)
    stderr = _CaptureLikeStream(tty=True)
    monkeypatch.setattr(evaluator_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluator_cli.sys, "stderr", stderr)
    monkeypatch.setattr(evaluator_cli.sys, "platform", "win32")

    evaluator_cli._configure_utf8_console()

    assert evaluator_cli.sys.stdout is stdout
    assert evaluator_cli.sys.stderr is stderr
    assert stdout.reconfigured_to == "utf-8"
    assert stderr.reconfigured_to == "utf-8"
    assert not stdout.closed
    assert not stderr.closed
    stdout.write("still writable")
    stderr.write("still writable")
