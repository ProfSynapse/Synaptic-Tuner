"""Internal prepared-token SFT boundary for the candidate MLX runtime.

This module is deliberately not a provider or launcher. The host must authenticate
the prepared input, tokenizer, runtime and model, retain the attempt, and execute
the worker in an owned bounded child. Importing/preparing here loads no ML library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Iterator

from shared.sft_preprocessing import PreparedSFTExample
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_mlx_runtime import MLXRuntimeExpectation, inspect_mlx_runtime_candidate


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"native SFT {name} is invalid")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= 512:
        raise ValueError(f"native SFT {name} is invalid")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"native SFT {name} is invalid")
    return value


@dataclass(frozen=True)
class MLXSFTConfig:
    """Explicit worker controls; optimizer/model/LoRA preparation stays external.

    One batch is one update. Accumulation is intentionally unavailable: upstream
    averages microbatch means, which differs from a token-weighted accumulated
    objective for unequal target lengths. No implicit schedule or epoch count.
    """

    batch_size: int
    max_seq_length: int
    max_batch_tokens: int
    iterations: int
    seed: int
    steps_per_report: int
    steps_per_eval: int
    steps_per_save: int

    def __post_init__(self) -> None:
        for name, maximum in (
            ("batch_size", 1024), ("max_seq_length", 1048576),
            ("max_batch_tokens", 1048576), ("iterations", 1000000),
            ("steps_per_report", 1000000), ("steps_per_eval", 1000000),
            ("steps_per_save", 1000000),
        ):
            _integer(getattr(self, name), name, 1, maximum)
        _integer(self.seed, "seed", 0, 2**32 - 1)
        if self.batch_size * self.max_seq_length > self.max_batch_tokens:
            raise ValueError("native SFT batch exceeds the configured token bound")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> MLXSFTConfig:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native SFT controls have missing or unknown fields")
        return cls(**value)


@dataclass(frozen=True)
class _Row:
    row_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class PreparedMLXSFT:
    """Immutable projection of host-verified rows, not a new dataset publication.

    The binding records the host's provenance; it does not authenticate that the
    caller actually obtained these examples from that source/tokenizer.
    """

    source: PreparedTrainingInputIdentity
    tokenizer_revision: str
    config: MLXSFTConfig
    train: tuple[_Row, ...]
    validation: tuple[_Row, ...]

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {
                "source": self.source.to_dict(),
                "tokenizer_revision": self.tokenizer_revision,
                "config": asdict(self.config),
                "train": [asdict(row) for row in self.train],
                "validation": [asdict(row) for row in self.validation],
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def prepare_mlx_sft(
    *,
    source: PreparedTrainingInputIdentity,
    tokenizer_revision: str,
    train: tuple[tuple[str, PreparedSFTExample], ...],
    validation: tuple[tuple[str, PreparedSFTExample], ...],
    config: MLXSFTConfig,
) -> PreparedMLXSFT:
    """Copy the shared preprocessor's exact tokens and masks without rendering.

    Splits and row IDs come from the host's retained prepared publication. No
    tokenizer, EOS insertion, text loader, split generation, filtering, packing,
    or truncation runs here. Unsupported attention masks fail rather than being
    discarded: this first causal-model boundary supports unpadded rows only.
    """
    if type(source) is not PreparedTrainingInputIdentity or type(config) is not MLXSFTConfig:
        raise TypeError("native SFT requires exact source identity and controls")
    _text(tokenizer_revision, "tokenizer revision")
    seen: set[str] = set()
    total_tokens = 0

    def split(rows: tuple[tuple[str, PreparedSFTExample], ...]) -> tuple[_Row, ...]:
        nonlocal total_tokens
        if type(rows) is not tuple or not 1 <= len(rows) <= 100000:
            raise ValueError("native SFT requires nonempty bounded declared splits")
        result = []
        for item in rows:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("native SFT row must pair its retained ID with an example")
            row_id, example = item
            _text(row_id, "row ID")
            if row_id in seen:
                raise ValueError("native SFT row ID occurs more than once")
            seen.add(row_id)
            if type(example) is not PreparedSFTExample:
                raise TypeError("native SFT requires shared prepared examples")
            if (
                example.truncation_applied is not False
                or example.drop_reason is not None
                or example.mask_prefix_mismatch is not False
                or example.mask_fallback_reason is not None
            ):
                raise ValueError("native SFT rejects incomplete or failed prepared examples")
            if any(type(value) is not list for value in (
                example.input_ids, example.labels, example.attention_mask,
            )):
                raise TypeError("native SFT prepared token arrays must be lists")
            size = len(example.input_ids)
            if not 2 <= size <= config.max_seq_length:
                raise ValueError("native SFT row does not fit; truncation is forbidden")
            total_tokens += size
            if total_tokens > 10000000:
                raise ValueError("native SFT prepared token budget exceeded")
            if len(example.labels) != size or len(example.attention_mask) != size:
                raise ValueError("native SFT token and mask lengths differ")
            for token, label, attention in zip(
                example.input_ids, example.labels, example.attention_mask,
            ):
                _integer(token, "token ID", 0, 2**31 - 1)
                if type(label) is not int or label not in (-100, token):
                    raise ValueError("native SFT labels must preserve tokens or ignore them")
                if type(attention) is not int or attention != 1:
                    raise ValueError("native SFT supports only unpadded causal rows")
            if not any(label != -100 for label in example.labels[1:]):
                raise ValueError("native SFT row has no next-token supervision")
            result.append(_Row(row_id, tuple(example.input_ids), tuple(example.labels)))
        return tuple(result)

    return PreparedMLXSFT(source, tokenizer_revision, config, split(train), split(validation))


@dataclass(frozen=True)
class MLXSFTBatch:
    row_ids: tuple[str, ...]
    input_ids: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]


def iter_prepared_batches(
    prepared: PreparedMLXSFT, *, split: str, loop: bool = False,
) -> Iterator[MLXSFTBatch]:
    """Deterministic train shuffle, ordered full validation, no dropped tail.

    Padding uses token zero only at the right edge; every padded label is ignored.
    Causal attention cannot expose that suffix to preceding supervised tokens.
    """
    if type(prepared) is not PreparedMLXSFT or split not in ("train", "validation"):
        raise ValueError("native SFT split selection is invalid")
    if type(loop) is not bool or (loop and split != "train"):
        raise ValueError("native SFT validation cannot loop")
    rows = getattr(prepared, split)
    rng = random.Random(prepared.config.seed)
    while True:
        order = list(range(len(rows)))
        if split == "train":
            rng.shuffle(order)
        for start in range(0, len(order), prepared.config.batch_size):
            batch = tuple(rows[index] for index in order[start:start + prepared.config.batch_size])
            width = max(len(row.input_ids) for row in batch)
            yield MLXSFTBatch(
                tuple(row.row_id for row in batch),
                tuple(row.input_ids + (0,) * (width - len(row.input_ids)) for row in batch),
                tuple(row.labels + (-100,) * (width - len(row.labels)) for row in batch),
            )
        if not loop:
            return


def make_mlx_sft_hooks(prepared: PreparedMLXSFT, *, mx, nn):
    """Adapt MLX-LM 0.32.2's custom batch/loss ABI without global monkeypatches.

    ``mx``/``nn`` are the qualified child's imported modules. Explicit injection
    also permits CPU algebra checks without installing or importing MLX.
    """
    if type(prepared) is not PreparedMLXSFT:
        raise TypeError("native SFT requires a prepared projection")

    def iterate_batches(
        dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None,
    ):
        # Identity is intentional: callers cannot accidentally feed validation to
        # training, a transformed sequence, or a different prepared publication.
        if dataset is prepared.train:
            split = "train"
        elif dataset is prepared.validation:
            split = "validation"
        else:
            raise ValueError("native SFT dataset is not a retained split")
        if (
            type(batch_size) is not int or batch_size != prepared.config.batch_size
            or type(max_seq_length) is not int or max_seq_length != prepared.config.max_seq_length
            or (seed is not None and (type(seed) is not int or seed != prepared.config.seed))
        ):
            raise ValueError("native SFT iterator controls differ from preparation")
        if comm_group is not None and (comm_group.size() != 1 or comm_group.rank() != 0):
            raise ValueError("native SFT distributed execution is unqualified")
        for batch in iter_prepared_batches(prepared, split=split, loop=loop):
            yield mx.array(batch.input_ids, dtype=mx.int32), mx.array(batch.labels, dtype=mx.int32)

    def loss(model, tokens, labels):
        targets = labels[:, 1:]
        mask = targets != -100
        # Never gather at -100: masking after a negative gather can fail or read
        # a real vocabulary entry. Also use where rather than 0 * nonfinite CE.
        safe_targets = mx.where(mask, targets, 0)
        logits = model(tokens[:, :-1])
        ce = nn.losses.cross_entropy(logits, safe_targets)
        ntokens = mask.sum()
        total = mx.where(mask, ce, 0).astype(mx.float32).sum()
        return total / ntokens, ntokens

    return iterate_batches, loss


def run_prepared_mlx_sft(
    prepared: PreparedMLXSFT, *, model, optimizer, adapter_file: str,
    runtime: MLXRuntimeExpectation, dependency_lock: bytes,
    training_callback=None,
) -> None:
    """Private child worker hook; no admission, model loading or process launch.

    Invoke only inside the future host-owned qualified child. The caller owns the
    bounded process, already prepared model/optimizer, fresh private output path,
    artifact verification and callback. This function is not registered in API v1.
    """
    if type(prepared) is not PreparedMLXSFT:
        raise TypeError("native SFT requires a prepared projection")
    from pathlib import Path

    if type(adapter_file) is not str or not Path(adapter_file).is_absolute():
        raise ValueError("native SFT adapter output must be a host-owned absolute path")
    inspect_mlx_runtime_candidate(runtime, dependency_lock=dependency_lock)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.trainer import TrainingArgs, train

    config = prepared.config
    mx.random.seed(config.seed)
    iterate, loss = make_mlx_sft_hooks(prepared, mx=mx, nn=nn)
    args = TrainingArgs(
        batch_size=config.batch_size, iters=config.iterations,
        val_batches=-1, max_seq_length=config.max_seq_length,
        steps_per_report=config.steps_per_report,
        steps_per_eval=config.steps_per_eval, steps_per_save=config.steps_per_save,
        adapter_file=adapter_file, grad_checkpoint=False,
        grad_accumulation_steps=1, clear_cache_threshold=0,
    )
    train(
        model=model, optimizer=optimizer, train_dataset=prepared.train,
        val_dataset=prepared.validation, args=args, loss=loss,
        iterate_batches=iterate, training_callback=training_callback,
    )
