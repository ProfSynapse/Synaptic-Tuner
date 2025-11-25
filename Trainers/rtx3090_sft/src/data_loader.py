"""
Data loading and preprocessing for SFT training.
SFT uses conversational format natively - much simpler than KTO!
"""

from typing import Optional, Tuple
from datasets import load_dataset, Dataset


def load_and_prepare_dataset(
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
    test_size: float = 0.1,
    split_dataset: bool = False,
    filter_desirable: bool = False
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare dataset for SFT training.

    Args:
        dataset_name: HuggingFace dataset name
        data_files: Specific file within the dataset
        local_file: Path to local JSONL file
        num_proc: Number of processes for dataset loading (1 for Windows)
        test_size: Fraction of data for validation
        split_dataset: Whether to create train/val split
        filter_desirable: Filter for label=True examples only (if dataset has labels)

    Returns:
        Tuple of (train_dataset, eval_dataset or None)
    """
    print("=" * 60)
    print("LOADING DATASET FOR SFT")
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

    # Convert 'conversations' to 'messages' (TRL 0.15.0+ requirement)
    if "conversations" in raw_datasets.column_names and "messages" not in raw_datasets.column_names:
        print("Converting 'conversations' key to 'messages' (TRL 0.15.0+ requirement)")
        raw_datasets = raw_datasets.rename_column("conversations", "messages")

    # Optional: Filter for desirable examples only
    if filter_desirable and "label" in raw_datasets.column_names:
        print("\nFiltering for desirable examples (label=True)...")
        original_size = len(raw_datasets)
        raw_datasets = raw_datasets.filter(lambda x: x["label"] == True)
        filtered_count = len(raw_datasets)
        print(f"Filtered: {original_size} → {filtered_count} examples")
        print(f"Removed: {original_size - filtered_count} undesirable examples")

    # SFTTrainer handles "conversations" format natively
    # No conversion needed - just use the dataset as-is!
    print("\nSFT uses conversational format natively - no preprocessing needed!")
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


def print_dataset_samples(dataset: Dataset, num_samples: int = 3):
    """Print sample examples from the dataset."""
    print("\nDataset samples:")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\nExample {i+1}:")

        # Check if this is conversational format
        if "conversations" in example:
            print(f"Format: Conversational ({len(example['conversations'])} messages)")
            for msg in example['conversations']:
                role = msg['role']
                # Handle both text content and tool calls
                if msg.get('content'):
                    content = msg['content'][:100]
                    print(f"  [{role}]: {content}...")
                elif msg.get('tool_calls'):
                    num_tools = len(msg['tool_calls'])
                    tool_names = [tc['function']['name'] for tc in msg['tool_calls']]
                    print(f"  [{role}]: <{num_tools} tool call(s): {', '.join(tool_names[:2])}...>")
                else:
                    print(f"  [{role}]: <empty>")
        elif "messages" in example:
            print(f"Format: Conversational ({len(example['messages'])} messages)")
            for msg in example['messages']:
                role = msg['role']
                # Handle both text content and tool calls
                if msg.get('content'):
                    content = msg['content'][:100]
                    print(f"  [{role}]: {content}...")
                elif msg.get('tool_calls'):
                    num_tools = len(msg['tool_calls'])
                    tool_names = [tc['function']['name'] for tc in msg['tool_calls']]
                    print(f"  [{role}]: <{num_tools} tool call(s): {', '.join(tool_names[:2])}...>")
                else:
                    print(f"  [{role}]: <empty>")
        elif "text" in example:
            print(f"Format: Text")
            print(f"Content: {example['text'][:200]}...")
        else:
            print(f"Format: Unknown - columns: {list(example.keys())}")

        if "label" in example:
            print(f"Label: {example['label']}")

        print("-" * 60)


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    train_ds, eval_ds = load_and_prepare_dataset(
        local_file="../../Datasets/syngen_tools_sft_11.18.25.jsonl",
        split_dataset=False,
        filter_desirable=False  # Already filtered
    )

    # Print samples
    print_dataset_samples(train_ds, num_samples=2)

    print(f"\n✓ Dataset loading test passed!")
    print(f"  Total examples: {len(train_ds)}")
    print(f"  Columns: {train_ds.column_names}")
