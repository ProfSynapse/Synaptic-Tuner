"""
Data loading and preprocessing for KTO training.
Handles ChatML to KTO format conversion.
"""

from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
import os


def prepare_kto_format(example: Dict) -> Optional[Dict]:
    """
    Convert ChatML format to KTO format.

    Args:
        example: Dictionary with 'conversations' and 'label' keys

    Returns:
        Dictionary with 'prompt', 'completion', and 'label' keys,
        or None if conversion fails
    """
    conversations = example.get("conversations", [])

    # Extract user and assistant messages
    user_msgs = [msg for msg in conversations if msg["role"] == "user"]
    assistant_msgs = [msg for msg in conversations if msg["role"] == "assistant"]

    # Validation
    if not user_msgs or not assistant_msgs:
        return None

    return {
        "prompt": user_msgs[0]["content"],
        "completion": assistant_msgs[0]["content"],
        "label": example["label"]
    }


def load_and_prepare_dataset(
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
    test_size: float = 0.1,
    split_dataset: bool = False
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare dataset for KTO training.

    Args:
        dataset_name: HuggingFace dataset name
        data_files: Specific file within the dataset
        local_file: Path to local JSONL file
        num_proc: Number of processes for dataset loading (1 for Windows)
        test_size: Fraction of data for validation
        split_dataset: Whether to create train/val split

    Returns:
        Tuple of (train_dataset, eval_dataset or None)
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

    # Convert to KTO format
    print("\nConverting ChatML to KTO format...")
    processed_examples = []

    for example in raw_datasets:
        kto_example = prepare_kto_format(example)
        if kto_example:
            processed_examples.append(kto_example)

    # Calculate statistics
    desirable = sum(1 for ex in processed_examples if ex["label"])
    undesirable = len(processed_examples) - desirable

    print(f"\nProcessed dataset:")
    print(f"  Total: {len(processed_examples)} examples")
    print(f"  Desirable (True): {desirable}")
    print(f"  Undesirable (False): {undesirable}")
    print(f"  Ratio: {desirable/undesirable:.2f}:1 (desirable:undesirable)")

    # Create HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "prompt": [ex["prompt"] for ex in processed_examples],
        "completion": [ex["completion"] for ex in processed_examples],
        "label": [ex["label"] for ex in processed_examples],
    })

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


def validate_kto_dataset(dataset: Dataset) -> bool:
    """
    Validate that KTO dataset has required format and balanced labels.

    Args:
        dataset: Dataset to validate

    Returns:
        True if valid, False otherwise
    """
    print("\nValidating KTO dataset...")

    # Check required columns
    required_cols = ["prompt", "completion", "label"]
    missing_cols = [col for col in required_cols if col not in dataset.column_names]

    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return False

    print(f"✓ All required columns present: {required_cols}")

    # Check for empty examples
    empty_prompts = sum(1 for ex in dataset if not ex["prompt"].strip())
    empty_completions = sum(1 for ex in dataset if not ex["completion"].strip())

    if empty_prompts > 0:
        print(f"⚠ Warning: {empty_prompts} examples with empty prompts")

    if empty_completions > 0:
        print(f"⚠ Warning: {empty_completions} examples with empty completions")

    # Check label distribution
    labels = dataset["label"]
    true_count = sum(labels)
    false_count = len(labels) - true_count

    print(f"\nLabel distribution:")
    print(f"  True: {true_count} ({true_count/len(labels)*100:.1f}%)")
    print(f"  False: {false_count} ({false_count/len(labels)*100:.1f}%)")

    if true_count == 0 or false_count == 0:
        print("✗ Dataset must have both True and False labels for KTO training")
        return False

    # Warn if severely imbalanced
    ratio = max(true_count, false_count) / min(true_count, false_count)
    if ratio > 10:
        print(f"⚠ Warning: Severely imbalanced dataset (ratio {ratio:.1f}:1)")
        print("  Consider using desirable_weight/undesirable_weight to balance")
    else:
        print(f"✓ Label distribution acceptable (ratio {ratio:.1f}:1)")

    print("✓ Dataset validation passed\n")
    return True


def print_dataset_samples(dataset: Dataset, num_samples: int = 3):
    """Print sample examples from the dataset."""
    print("\nDataset samples:")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\nExample {i+1}:")
        print(f"Label: {example['label']}")
        print(f"Prompt: {example['prompt'][:200]}...")
        print(f"Completion: {example['completion'][:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    # Test dataset loading
    train_ds, eval_ds = load_and_prepare_dataset(
        dataset_name="professorsynapse/claudesidian-synthetic-dataset",
        data_files="syngen_tools_11.14.25.jsonl",
        split_dataset=False
    )

    # Validate
    validate_kto_dataset(train_ds)

    # Print samples
    print_dataset_samples(train_ds)
