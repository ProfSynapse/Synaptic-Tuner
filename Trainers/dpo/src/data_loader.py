"""
Data loading and preprocessing for DPO training.

Consumes the on-disk DPO preference-pair schema (one JSON object per line):

    {"prompt":   [{"role": "system", "content": "..."},
                  {"role": "user",   "content": "<question>"}],
     "chosen":   [{"role": "assistant", "content": "<desirable completion>"}],
     "rejected": [{"role": "assistant", "content": "<undesirable completion>"}]}

This is the TRL "conversational" DPO convention (prompt/chosen/rejected as
message lists). Unlike KTO (Trainers/kto/src/data_loader.py), DPO is PAIRED and
unweighted, so there is NO interleaving and NO True/False label column — the
chosen/rejected pairing carries the preference signal directly. The KTO loader's
ChatML->prompt/completion/label transform is replaced here by a
prompt/chosen/rejected pass-through with structural validation.
"""

from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset


# Columns TRL's DPOTrainer expects in conversational mode.
REQUIRED_DPO_COLUMNS = ["prompt", "chosen", "rejected"]


def _is_message_list(value) -> bool:
    """True if value is a non-empty list of {role, content} message dicts."""
    if not isinstance(value, list) or not value:
        return False
    for msg in value:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
    return True


def load_and_prepare_dataset(
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
    test_size: float = 0.1,
    split_dataset: bool = False
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare a preference-pair dataset for DPO training.

    Args:
        dataset_name: HuggingFace dataset name
        data_files: Specific file within the dataset
        local_file: Path to local JSONL file (prompt/chosen/rejected schema)
        num_proc: Number of processes for dataset loading (1 for Windows)
        test_size: Fraction of data for validation
        split_dataset: Whether to create a train/val split

    Returns:
        Tuple of (train_dataset, eval_dataset or None) with columns
        prompt / chosen / rejected.
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    # Load raw dataset
    if local_file:
        print(f"Loading from local file: {local_file}")
        raw_datasets = load_dataset("json", data_files=local_file, split="train")
    elif dataset_name:
        print(f"Loading from HuggingFace: {dataset_name}")
        if data_files:
            print(f"Using file: {data_files}")
            raw_datasets = load_dataset(
                dataset_name,
                data_files=data_files,
                num_proc=num_proc
            )
        else:
            raw_datasets = load_dataset(dataset_name, num_proc=num_proc)
        raw_datasets = raw_datasets["train"]
    else:
        raise ValueError("Must provide either dataset_name or local_file")

    print(f"\nRaw dataset size: {len(raw_datasets)} examples")

    # Keep only the DPO columns; drop any extras the builder may have emitted
    # (e.g. provenance fields) so the Dataset matches what DPOTrainer expects.
    extra_cols = [c for c in raw_datasets.column_names if c not in REQUIRED_DPO_COLUMNS]
    if extra_cols:
        print(f"Dropping non-DPO columns: {extra_cols}")
        raw_datasets = raw_datasets.remove_columns(extra_cols)

    train_dataset = raw_datasets

    # Optional train/validation split
    eval_dataset = None
    if split_dataset and test_size > 0:
        print(f"\nCreating train/validation split ({1-test_size:.0%}/{test_size:.0%})")
        split = train_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        print(f"  Training set: {len(train_dataset)} examples")
        print(f"  Validation set: {len(eval_dataset)} examples")
    else:
        print(f"\nReady for training: {len(train_dataset)} examples")

    print("=" * 60)

    return train_dataset, eval_dataset


def validate_dpo_dataset(dataset: Dataset) -> bool:
    """
    Validate that a DPO dataset has the required prompt/chosen/rejected columns
    and that each row's values are well-formed message lists.

    Args:
        dataset: Dataset to validate

    Returns:
        True if valid, False otherwise
    """
    print("\nValidating DPO dataset...")

    # Check required columns
    missing_cols = [col for col in REQUIRED_DPO_COLUMNS if col not in dataset.column_names]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return False

    print(f"✓ All required columns present: {REQUIRED_DPO_COLUMNS}")

    # Check each row is structurally valid (message lists, non-empty)
    bad_prompt = 0
    bad_chosen = 0
    bad_rejected = 0
    for ex in dataset:
        if not _is_message_list(ex["prompt"]):
            bad_prompt += 1
        if not _is_message_list(ex["chosen"]):
            bad_chosen += 1
        if not _is_message_list(ex["rejected"]):
            bad_rejected += 1

    if bad_prompt or bad_chosen or bad_rejected:
        print(f"✗ Malformed rows -> prompt: {bad_prompt}, chosen: {bad_chosen}, rejected: {bad_rejected}")
        print("  Each of prompt/chosen/rejected must be a non-empty list of {role, content} messages.")
        return False

    print(f"✓ All {len(dataset)} rows have well-formed prompt/chosen/rejected message lists")
    print("✓ Dataset validation passed\n")
    return True


def print_dataset_samples(dataset: Dataset, num_samples: int = 3):
    """Print sample examples from the dataset."""
    print("\nDataset samples:")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        prompt_text = example["prompt"][-1]["content"] if example["prompt"] else ""
        chosen_text = example["chosen"][-1]["content"] if example["chosen"] else ""
        rejected_text = example["rejected"][-1]["content"] if example["rejected"] else ""
        print(f"\nExample {i+1}:")
        print(f"Prompt:   {prompt_text[:160]}...")
        print(f"Chosen:   {chosen_text[:160]}...")
        print(f"Rejected: {rejected_text[:160]}...")
        print("-" * 60)
