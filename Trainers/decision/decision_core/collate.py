"""
Batching for decision training: option permutation, tokenisation, targets.

Location: Trainers/decision/decision_core/collate.py
Used by:  trainer.py (training/validation loaders), evaluate.py.

Genericity is enforced here. Every time a row is drawn its options are
re-permuted and the label follows, so a readout that memorised "slot 0 is the
positive class" would be wrong most of the time. Score levels are ordered, so
they are never shuffled -- only reversed as a whole, which preserves meaning.

Over-long prompts keep their tail: the question, the options and ``<answer>``
are the last tokens, so the state is cut from the front. A row whose option
list itself does not fit raises rather than scoring options it cannot see.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .examples import KINDS, DecisionExample
from .prompting import HEADERS, RenderedPrompt, last_token_in_span, render_prompt

KIND_IDS = {k: i for i, k in enumerate(KINDS)}


@dataclass
class CollatorConfig:
    max_length: int = 4096
    marker_style: str = "numbers"
    headers: dict[str, str] = field(default_factory=lambda: dict(HEADERS))
    shuffle_options: bool = True
    reverse_score_prob: float = 0.5
    # Target mass moved to adjacent levels for score rows: one level off is a
    # smaller error than four off, which plain cross-entropy cannot express.
    ordinal_smoothing: float = 0.1
    vary_instructions: bool = True
    seed: int = 0


@dataclass(frozen=True)
class EncodedPrompt:
    input_ids: list[int]
    option_index: list[int]
    rendered: RenderedPrompt


def encode_prompt(tokenizer: Any, rendered: RenderedPrompt, max_length: int) -> EncodedPrompt:
    """Tokenise a rendered prompt and locate each option's last token.

    When the prompt is longer than ``max_length`` the leading tokens (the start
    of the state) are dropped.
    """
    enc = tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    option_index = [last_token_in_span(offsets, span) for span in rendered.option_spans]
    cut = max(0, len(ids) - max_length)
    if cut:
        if min(option_index) < cut:
            raise ValueError(
                f"prompt is {len(ids)} tokens and its option list does not fit in max_length={max_length}"
            )
        ids = ids[cut:]
        option_index = [i - cut for i in option_index]
    return EncodedPrompt(input_ids=ids, option_index=option_index, rendered=rendered)


def target_distribution(
    ex: DecisionExample, order: Sequence[int], ordinal_smoothing: float
) -> list[float]:
    """Soft target over rendered slots. One-hot except for smoothed score rows."""
    n = ex.n_options
    mass = {ex.label: 1.0}
    if ex.kind == "score" and ordinal_smoothing > 0:
        neighbours = [lv for lv in (ex.label - 1, ex.label + 1) if 0 <= lv < n]
        if neighbours:
            mass = {ex.label: 1.0 - ordinal_smoothing}
            for lv in neighbours:
                mass[lv] = ordinal_smoothing / len(neighbours)
    order = list(order)
    dist = [0.0] * n
    for level, m in mass.items():
        dist[order.index(level)] += m
    return dist


class DecisionCollator:
    """Turns DecisionExamples into a padded batch. ``train=False`` renders canonically."""

    def __init__(self, tokenizer: Any, config: CollatorConfig, *, train: bool = True):
        self.tok = tokenizer
        self.cfg = config
        self.train = train
        self.rng = random.Random(config.seed)
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise ValueError("tokenizer has neither a pad nor an eos token")
        self.pad_id = int(pad)

    def option_order(self, ex: DecisionExample) -> list[int]:
        n = ex.n_options
        identity = list(range(n))
        if not self.train or not self.cfg.shuffle_options:
            return identity
        if ex.kind == "score":
            return identity[::-1] if self.rng.random() < self.cfg.reverse_score_prob else identity
        self.rng.shuffle(identity)
        return identity

    def instruction(self, ex: DecisionExample) -> str:
        if not self.train or not self.cfg.vary_instructions or not ex.instruction_variants:
            return ex.instructions
        return self.rng.choice(ex.all_instructions())

    def encode(self, ex: DecisionExample, order: Sequence[int] | None = None) -> EncodedPrompt:
        order = list(order) if order is not None else self.option_order(ex)
        rendered = render_prompt(
            ex.state,
            ex.kind,
            self.instruction(ex),
            ex.options,
            order=order,
            marker_style=self.cfg.marker_style,
            headers=self.cfg.headers,
        )
        return encode_prompt(self.tok, rendered, self.cfg.max_length)

    def __call__(
        self, batch: list[DecisionExample], orders: list[Sequence[int]] | None = None
    ) -> dict[str, Any]:
        if not batch:
            raise ValueError("empty batch")
        rows = []
        for i, ex in enumerate(batch):
            order = list(orders[i]) if orders is not None else self.option_order(ex)
            enc = self.encode(ex, order)
            rows.append((ex, order, enc))

        seq_len = max(len(enc.input_ids) for _, _, enc in rows)
        width = max(ex.n_options for ex, _, _ in rows)
        bsz = len(rows)

        input_ids = torch.full((bsz, seq_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long)
        option_index = torch.full((bsz, width), -1, dtype=torch.long)
        target = torch.zeros((bsz, width), dtype=torch.float32)
        labels = torch.zeros(bsz, dtype=torch.long)
        n_options = torch.zeros(bsz, dtype=torch.long)
        answer_index = torch.zeros(bsz, dtype=torch.long)
        weights = torch.zeros(bsz, dtype=torch.float32)
        kinds = torch.zeros(bsz, dtype=torch.long)

        for i, (ex, order, enc) in enumerate(rows):
            n_tok = len(enc.input_ids)
            input_ids[i, :n_tok] = torch.tensor(enc.input_ids, dtype=torch.long)
            attention_mask[i, :n_tok] = 1
            answer_index[i] = n_tok - 1
            option_index[i, : ex.n_options] = torch.tensor(enc.option_index, dtype=torch.long)
            dist = target_distribution(ex, order, self.cfg.ordinal_smoothing)
            target[i, : ex.n_options] = torch.tensor(dist, dtype=torch.float32)
            labels[i] = order.index(ex.label)
            n_options[i] = ex.n_options
            weights[i] = ex.weight
            kinds[i] = KIND_IDS[ex.kind]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "option_index": option_index,
            "answer_index": answer_index,
            "n_options": n_options,
            "labels": labels,
            "target": target,
            "weights": weights,
            "kinds": kinds,
            # Non-tensor bookkeeping for evaluation: canonical index shown at each slot.
            "orders": [order for _, order, _ in rows],
        }
