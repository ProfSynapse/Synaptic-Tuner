from dataclasses import replace
import itertools
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shared.sft_preprocessing import PreparedSFTExample, materialize_sft_example
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_mlx_sft import (
    MLXSFTConfig, iter_prepared_batches, make_mlx_sft_hooks,
    prepare_mlx_sft, run_prepared_mlx_sft,
)


def config(**changes):
    values = dict(
        batch_size=2, max_seq_length=8, max_batch_tokens=16, iterations=3,
        seed=17, steps_per_report=1, steps_per_eval=2, steps_per_save=3,
    )
    values.update(changes)
    return MLXSFTConfig.from_dict(values)


def example(tokens=(1, 2, 3, 4), labels=(-100, 2, -100, 4)):
    return PreparedSFTExample(
        list(tokens), [1] * len(tokens), list(labels), "messages",
        "assistant_only", False,
    )


def prepared(*, train=None, validation=None, controls=None):
    return prepare_mlx_sft(
        source=PreparedTrainingInputIdentity(
            "prepared://sha256/" + "a" * 64, "a" * 64, "b" * 64, 10,
            "synaptic-conversation-sft-row/v1",
        ),
        tokenizer_revision="c" * 40,
        train=train if train is not None else tuple(
            (f"train-{i}", example()) for i in range(3)
        ),
        validation=validation if validation is not None else (
            ("heldout-0", example()), ("heldout-1", example((1, 3), (-100, 3))),
        ),
        config=controls or config(),
    )


def test_import_has_no_ml_or_provider_effects():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import tuner.runtime.native_mlx_sft; "
            "assert not any(k.split('.')[0] in "
            "{'mlx', 'mlx_lm', 'torch', 'transformers', 'modal', 'numpy'} "
            "for k in sys.modules)"
        )], cwd=root, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_copy_keeps_arbitrary_masks_splits_and_digest_stable():
    row = example()
    data = prepared(train=(("train", row),))
    before = data.canonical_bytes()
    row.input_ids[0] = 99
    row.labels[1] = -100
    assert data.train[0].input_ids == (1, 2, 3, 4)
    assert data.train[0].labels == (-100, 2, -100, 4)
    assert data.canonical_bytes() == before
    assert len(data.digest) == 64
    assert data.digest != prepared(train=(("different-id", example()),)).digest
    assert data.digest != replace(data, tokenizer_revision="d" * 40).digest


def test_shared_raw_text_materialization_is_not_retokenized_or_given_extra_eos():
    class Tokenizer:
        eos_token_id = 7

        def encode(self, text, add_special_tokens):
            assert text == "text" and add_special_tokens is False
            return [1, 2]

    row = materialize_sft_example(
        tokenizer=Tokenizer(),
        record={"schema_version": "syntunia-sft-row/v1", "format": "raw_text", "text": "text",
                "split": "train"},
        max_seq_length=8, assistant_only_loss=False,
    )
    data = prepared(train=(("raw", row),))
    assert data.train[0].input_ids == (1, 2, 7)
    assert data.train[0].labels == (1, 2, 7)


def test_shared_prompt_completion_keeps_template_options_and_terminal_boundary():
    class Tokenizer:
        eos_token_id = 999
        all_special_ids = [ord("!")]

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False and kwargs == {"enable_thinking": False}
            text = "|".join(message["content"] for message in messages)
            return text + ("|assistant:" if add_generation_prompt else "!")

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(map(ord, text))

    row = materialize_sft_example(
        tokenizer=Tokenizer(), record={
            "schema_version": "syntunia-sft-row/v2", "format": "messages",
            "split": "train", "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "target"},
            ],
        },
        max_seq_length=64, assistant_only_loss=True, prompt_render="prompt_completion",
        chat_template_kwargs={"enable_thinking": False},
    )
    data = prepared(train=(("message", row),), controls=config(
        max_seq_length=64, max_batch_tokens=128,
    ))
    assert data.train[0].input_ids == tuple(map(ord, "prompt|assistant:target!"))
    assert data.train[0].labels == (-100,) * len("prompt|assistant:") + tuple(map(ord, "target!"))
    assert 999 not in data.train[0].input_ids


@pytest.mark.parametrize("changes", [
    {"truncation_applied": True}, {"drop_reason": "no_supervised_tokens"},
    {"mask_prefix_mismatch": True}, {"mask_fallback_reason": "fallback"},
    {"input_ids": [1, True, 3, 4]}, {"labels": [-100, 99, -100, 4]},
    {"labels": [-100, True, -100, 4]}, {"attention_mask": [1, 0, 1, 1]},
    {"attention_mask": [1, 1]}, {"labels": [1, -100, -100, -100]},
    {"input_ids": [1] * 9, "attention_mask": [1] * 9, "labels": [1] * 9},
])
def test_invalid_prepared_rows_fail_without_repair(changes):
    row = replace(example(), **changes)
    with pytest.raises((ValueError, TypeError)):
        prepared(train=(("bad", row),))


def test_rejects_empty_splits_and_holdout_id_overlap():
    with pytest.raises(ValueError, match="nonempty"):
        prepared(validation=())
    with pytest.raises(ValueError, match="more than once"):
        prepared(train=(("heldout-0", example()),))


@pytest.mark.parametrize("changes", [
    {"batch_size": True}, {"iterations": 0}, {"seed": -1},
    {"max_batch_tokens": 15}, {"gradient_accumulation_steps": 2},
    {"packing": True}, {"method": "grpo"},
])
def test_controls_fail_closed(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_batches_keep_tail_all_validation_and_exact_shiftable_labels():
    data = prepared()
    batches = list(iter_prepared_batches(data, split="train"))
    assert sorted(i for batch in batches for i in batch.row_ids) == [
        "train-0", "train-1", "train-2",
    ]
    assert [len(batch.row_ids) for batch in batches] == [2, 1]
    assert batches == list(iter_prepared_batches(data, split="train"))
    validation = list(iter_prepared_batches(data, split="validation"))
    assert validation[0].row_ids == ("heldout-0", "heldout-1")
    assert validation[0].input_ids == ((1, 2, 3, 4), (1, 3, 0, 0))
    assert validation[0].labels == ((-100, 2, -100, 4), (-100, 3, -100, -100))
    looped = list(itertools.islice(iter_prepared_batches(data, split="train", loop=True), 4))
    assert sorted(i for batch in looped[2:] for i in batch.row_ids) == [
        "train-0", "train-1", "train-2",
    ]
    with pytest.raises(ValueError):
        list(iter_prepared_batches(data, split="validation", loop=True))


class Matrix:
    """Small CPU array double; CE below uses independent scalar probability math."""

    def __init__(self, rows, dtype=None):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, key):
        rows, columns = key
        return Matrix([row[columns] for row in self.rows[rows]])

    def __ne__(self, value):
        return Matrix([[item != value for item in row] for row in self.rows])

    def sum(self):
        return sum(sum(row) for row in self.rows)

    def astype(self, dtype):
        return self


def where(mask, left, right):
    return Matrix([
        [(left.rows[i][j] if isinstance(left, Matrix) else left) if item
         else (right.rows[i][j] if isinstance(right, Matrix) else right)
         for j, item in enumerate(row)]
        for i, row in enumerate(mask.rows)
    ])


MX = SimpleNamespace(array=Matrix, where=where, int32="int32", float32="float32")


def test_loss_uses_shifted_per_token_targets_and_supervised_token_denominator():
    data = prepared()
    observed = {}

    def model(inputs):
        observed["inputs"] = inputs.rows
        # Precomputed logits are log probabilities; every position sums to one.
        return [[[math.log(p) for p in (.1, .1, .2, .3, .3)]] * 3] * 2

    def cross_entropy(logits, targets):
        observed["targets"] = targets.rows
        # Reject negative gather indices, just as the real loss must never see -100.
        assert all(t >= 0 for row in targets.rows for t in row)
        return Matrix([
            [math.log(sum(math.exp(x) for x in vector)) - vector[target]
             for vector, target in zip(row, target_row)]
            for row, target_row in zip(logits, targets.rows)
        ])

    iterate, loss = make_mlx_sft_hooks(
        data, mx=MX, nn=SimpleNamespace(losses=SimpleNamespace(cross_entropy=cross_entropy)),
    )
    batch = next(iterate(data.validation, 2, 8))
    value, count = loss(model, *batch)
    assert observed["inputs"] == [[1, 2, 3], [1, 3, 0]]
    assert observed["targets"] == [[2, 0, 4], [3, 0, 0]]
    assert count == 3
    assert value == pytest.approx((-math.log(.2) - 2 * math.log(.3)) / 3)


def test_hooks_refuse_changed_or_distributed_upstream_calls():
    data = prepared()
    iterate, _ = make_mlx_sft_hooks(data, mx=MX, nn=None)
    for kwargs in (
        {"dataset": list(data.train)},
        {"batch_size": 1}, {"max_seq_length": 7}, {"seed": 19},
        {"comm_group": SimpleNamespace(size=lambda: 2, rank=lambda: 0)},
    ):
        args = dict(dataset=data.train, batch_size=2, max_seq_length=8)
        args.update(kwargs)
        with pytest.raises(ValueError):
            list(iterate(**args))


def test_worker_joins_pinned_upstream_hooks_without_text_loader(monkeypatch, tmp_path):
    data = prepared()
    calls = {}
    mx = ModuleType("mlx.core")
    mx.__dict__.update(vars(MX))
    mx.random = SimpleNamespace(seed=lambda seed: calls.update(seed=seed))
    nn = ModuleType("mlx.nn")
    trainer = ModuleType("mlx_lm.tuner.trainer")
    trainer.TrainingArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    trainer.train = lambda **kwargs: calls.update(kwargs)
    mlx = ModuleType("mlx")
    mlx.core, mlx.nn = mx, nn
    for name, module in (
        ("mlx", mlx), ("mlx.core", mx), ("mlx.nn", nn),
        ("mlx_lm", ModuleType("mlx_lm")),
        ("mlx_lm.tuner", ModuleType("mlx_lm.tuner")),
        ("mlx_lm.tuner.trainer", trainer),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    model, optimizer, callback = object(), object(), object()
    runtime = object()
    def inspect(expected, *, dependency_lock):
        assert expected is runtime and dependency_lock == b"reviewed lock"
        calls["inspected"] = True
    monkeypatch.setattr("tuner.runtime.native_mlx_sft.inspect_mlx_runtime_candidate", inspect)
    run_prepared_mlx_sft(
        data, model=model, optimizer=optimizer,
        adapter_file=str(tmp_path / "adapter.safetensors"), training_callback=callback,
        runtime=runtime, dependency_lock=b"reviewed lock",
    )
    assert calls["model"] is model and calls["optimizer"] is optimizer
    assert calls["train_dataset"] is data.train
    assert calls["val_dataset"] is data.validation
    assert calls["training_callback"] is callback
    assert calls["seed"] == 17
    assert calls["args"].iters == 3 and calls["args"].val_batches == -1
    assert calls["args"].grad_accumulation_steps == 1
    assert calls["inspected"]
    assert callable(calls["loss"]) and callable(calls["iterate_batches"])


def test_runtime_rejection_happens_before_worker_imports(monkeypatch, tmp_path):
    def reject(*args, **kwargs):
        raise ValueError("candidate mismatch")
    monkeypatch.setattr("tuner.runtime.native_mlx_sft.inspect_mlx_runtime_candidate", reject)
    with pytest.raises(ValueError, match="candidate mismatch"):
        run_prepared_mlx_sft(
            prepared(), model=None, optimizer=None,
            adapter_file=str(tmp_path / "adapter.safetensors"),
            runtime=None, dependency_lock=b"bad",
        )
