"""Loading ONE split's parquet from a multi-split HF repo must not enforce the
README's expected-splits contract.

REPRODUCTION APPROACH (real, not monkeypatched): we build a tiny on-disk HF
dataset repo on tmp_path that mirrors the real `<org>/<dataset>` shape
closely enough to trigger the actual failure:

  data/train-00000-of-00001.parquet   (a couple rows)
  data/test-00000-of-00001.parquet     (one row)
  README.md  -- YAML front-matter `dataset_info` declaring BOTH `train` and
                `test` splits.

When you ask the builder for only the train shard via
`data_files="data/train-00000-of-00001.parquet"`, the README's split contract is
enforced and the builder raises
`datasets.exceptions.ExpectedMoreSplitsError: {'test'}` — even though the parquet
loads fine. The fix adds `verification_mode=VerificationMode.NO_CHECKS` to that
single `load_dataset(...)` call in `load_and_prepare_tokenized_dataset`.

Test 1 PROVES the repro is faithful: a raw `load_dataset(..., data_files=train)`
(no verification_mode) raises ExpectedMoreSplitsError on this fixture. If that
ever stops raising, the regression test below is vacuous, so we assert it.

Test 2 PROVES the fix: `load_and_prepare_tokenized_dataset(...)` over the same
fixture returns the train rows and does NOT raise.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

from data_loader import load_and_prepare_tokenized_dataset  # noqa: E402

# Reuse the Qwen-like stub from the sibling test (no model download).
from test_native_tool_calls import _EmptyThinkInjectingTokenizer  # noqa: E402

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


_TRAIN_ROWS = [
    {"messages": [
        {"role": "user", "content": "hello one"},
        {"role": "assistant", "content": "world one"},
    ]},
    {"messages": [
        {"role": "user", "content": "hello two"},
        {"role": "assistant", "content": "world two"},
    ]},
]
_TEST_ROWS = [
    {"messages": [
        {"role": "user", "content": "held out"},
        {"role": "assistant", "content": "held out answer"},
    ]},
]

# README front-matter declaring BOTH splits. `messages` is a list<struct> with
# {role, content}; the split list is what drives ExpectedMoreSplitsError when only
# `train` is requested. num_bytes/num_examples are nominal — the split NAMES are
# what the contract checks.
_README = textwrap.dedent("""\
    ---
    dataset_info:
      features:
      - name: messages
        list:
        - name: role
          dtype: string
        - name: content
          dtype: string
      splits:
      - name: train
        num_bytes: 1000
        num_examples: 2
      - name: test
        num_bytes: 500
        num_examples: 1
      download_size: 1500
      dataset_size: 1500
    configs:
    - config_name: default
      data_files:
      - split: train
        path: data/train-*
      - split: test
        path: data/test-*
    ---
    # tiny multi-split fixture
    """)


def _write_table(rows, path: Path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _build_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "mini_repo"
    data = repo / "data"
    data.mkdir(parents=True)
    _write_table(_TRAIN_ROWS, data / "train-00000-of-00001.parquet")
    _write_table(_TEST_ROWS, data / "test-00000-of-00001.parquet")
    (repo / "README.md").write_text(_README, encoding="utf-8")
    return repo


def test_fixture_reproduces_expected_more_splits_error_without_fix(tmp_path):
    """Guard against a vacuous regression test: the raw builder call (no
    verification_mode) MUST raise ExpectedMoreSplitsError on this fixture, proving
    it faithfully mirrors the cloud failure. If this stops raising, test 2 below
    no longer proves anything and must be revisited."""
    from datasets import load_dataset
    from datasets.exceptions import ExpectedMoreSplitsError

    repo = _build_repo(tmp_path)
    with pytest.raises(ExpectedMoreSplitsError):
        load_dataset(str(repo), data_files="data/train-00000-of-00001.parquet")


def test_loader_loads_single_split_from_multisplit_repo(tmp_path):
    """The fix: load_and_prepare_tokenized_dataset over the multi-split repo,
    requesting only the train shard, returns the train rows and does NOT raise
    ExpectedMoreSplitsError."""
    from datasets.exceptions import ExpectedMoreSplitsError

    repo = _build_repo(tmp_path)
    tok = _EmptyThinkInjectingTokenizer()
    try:
        train, eval_ds = load_and_prepare_tokenized_dataset(
            dataset_name=str(repo),
            data_files="data/train-00000-of-00001.parquet",
            tokenizer=tok,
            max_seq_length=10_000,
            loss_mask_mode="assistant_only",
            split_dataset=False,
        )
    except ExpectedMoreSplitsError as exc:  # pragma: no cover - the bug, if unfixed
        pytest.fail(f"split-count verification still enforced: {exc}")

    assert len(train) == len(_TRAIN_ROWS), "exactly the train shard rows are loaded"
    assert eval_ds is None
    # the tokenized contract is intact (rows materialized to trained features)
    assert set(train.column_names) >= {"input_ids", "attention_mask", "labels"}
    assert len(train[0]["input_ids"]) > 0
