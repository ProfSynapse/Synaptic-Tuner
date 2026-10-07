"""Grouped train/validation split (shared/training_utils.py)."""

import pytest
from datasets import Dataset

from shared.training_utils import (
    DEFAULT_SPLIT_SEED,
    extract_dataset_group_values,
    extract_group_values,
    grouped_split_indices,
    split_train_validation,
)


def _rows(n_groups=10, per_group=4):
    return [
        {"text": f"group {g} variant {v}", "metadata": {"scenario": f"s{g:02d}"}}
        for g in range(n_groups)
        for v in range(per_group)
    ]


def test_no_group_ever_lands_on_both_sides():
    rows = _rows()
    dataset = Dataset.from_list(rows)
    groups = extract_dataset_group_values(dataset, "metadata.scenario")
    train, validation = split_train_validation(dataset, test_size=0.2, group_values=groups)

    train_groups = {row["metadata"]["scenario"] for row in train}
    validation_groups = {row["metadata"]["scenario"] for row in validation}
    assert train_groups and validation_groups
    assert train_groups.isdisjoint(validation_groups)
    assert len(train) + len(validation) == len(rows)
    # Ratio is applied over groups: ceil(0.2 * 10) = 2 groups of 4 rows.
    assert len(validation_groups) == 2
    assert len(validation) == 8


def test_grouped_split_is_deterministic_and_order_independent():
    values = [f"g{i % 7}" for i in range(50)]
    first = grouped_split_indices(values, 0.3, seed=DEFAULT_SPLIT_SEED)
    second = grouped_split_indices(values, 0.3, seed=DEFAULT_SPLIT_SEED)
    assert first == second

    reversed_values = list(reversed(values))
    _, test_idx = grouped_split_indices(values, 0.3)
    _, rev_test_idx = grouped_split_indices(reversed_values, 0.3)
    assert {values[i] for i in test_idx} == {reversed_values[i] for i in rev_test_idx}


def test_different_seed_can_choose_different_groups():
    values = [f"g{i}" for i in range(40)]
    chosen = {
        frozenset(values[i] for i in grouped_split_indices(values, 0.25, seed=seed)[1])
        for seed in range(5)
    }
    assert len(chosen) > 1


def test_missing_group_key_fails_loudly_with_row_index():
    rows = _rows(n_groups=3, per_group=2)
    rows[4] = {"text": "no metadata here", "metadata": {"other": "x"}}
    with pytest.raises(ValueError, match=r"row index 4"):
        extract_group_values(rows, "metadata.scenario")

    # HF datasets fill absent struct keys with None; that is missing too.
    dataset = Dataset.from_list(rows)
    with pytest.raises(ValueError, match=r"row index 4"):
        extract_dataset_group_values(dataset, "metadata.scenario")


def test_missing_top_level_column_fails_on_row_zero():
    dataset = Dataset.from_list([{"text": "a"}, {"text": "b"}])
    with pytest.raises(ValueError, match=r"row index 0"):
        extract_dataset_group_values(dataset, "metadata.scenario")


def test_empty_string_group_value_is_missing():
    with pytest.raises(ValueError, match=r"row index 1"):
        extract_group_values([{"k": "a"}, {"k": "  "}], "k")


def test_single_group_cannot_be_split():
    with pytest.raises(ValueError, match="at least 2 distinct groups"):
        grouped_split_indices(["same"] * 5, 0.2)


def test_ungrouped_split_matches_historical_random_split():
    dataset = Dataset.from_list([{"i": i} for i in range(30)])
    train, validation = split_train_validation(dataset, test_size=0.1)
    expected = dataset.train_test_split(test_size=0.1, seed=42)
    assert train["i"] == expected["train"]["i"]
    assert validation["i"] == expected["test"]["i"]


def test_non_string_group_values_are_canonicalized():
    values = extract_group_values([{"g": 1}, {"g": {"b": 2, "a": 1}}], "g")
    assert values == ["1", '{"a": 1, "b": 2}']
