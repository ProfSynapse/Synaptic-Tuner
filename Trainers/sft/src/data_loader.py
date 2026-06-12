"""
Data loading and preprocessing for SFT training.
"""

from typing import Optional, Tuple, Any
from datasets import load_dataset, Dataset

from preprocessing import (
    ASSISTANT_ONLY,
    load_and_prepare_sft_dataset,
    sanitize_conversations as sanitize_prepared_conversations,
)


def _map_num_proc(num_proc: int) -> Optional[int]:
    return num_proc if num_proc and num_proc > 1 else None


def sanitize_conversations(messages: list) -> list:
    """
    Sanitize conversations to handle None content and tool_calls.
    Converts tool_calls to text format for compatibility with chat templates.
    """
    return sanitize_prepared_conversations(messages)


def load_and_prepare_dataset(
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
    test_size: float = 0.1,
    split_dataset: bool = False,
    filter_desirable: bool = False,
    tokenizer: Any = None,
    apply_chat_template: bool = False
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
        tokenizer: Tokenizer for applying chat template (required if apply_chat_template=True)
        apply_chat_template: If True, preprocesses dataset into a canonical `text`
            column using the active chat template

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

    # Apply chat template preprocessing when requested. This produces a stable
    # `text` dataset shape that works across newer TRL/Unsloth stacks for both
    # packed and non-packed SFT runs.
    if apply_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer is required when apply_chat_template=True")

        print("\nApplying chat template preprocessing...")
        messages_key = "messages" if "messages" in raw_datasets.column_names else "conversations"

        def format_example(example):
            """Apply chat template to create 'text' field."""
            msgs = example[messages_key]
            sanitized_msgs = sanitize_conversations(msgs)
            text = tokenizer.apply_chat_template(
                sanitized_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        raw_datasets = raw_datasets.map(
            format_example,
            num_proc=_map_num_proc(num_proc),
            remove_columns=raw_datasets.column_names,
            desc="Applying chat template"
        )
        print("Converted dataset to canonical 'text' format")
    else:
        print("\nKeeping raw conversational dataset format")

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


def load_and_prepare_tokenized_dataset(
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
    test_size: float = 0.1,
    split_dataset: bool = False,
    filter_desirable: bool = False,
    tokenizer: Any = None,
    max_seq_length: int = 2048,
    loss_mask_mode: str = ASSISTANT_ONLY,
    chat_template_kwargs: Optional[dict] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare dataset into explicit tokenized SFT features.

    This is the repo-owned prepared-dataset path for trainers that want to
    consume ``input_ids`` / ``attention_mask`` / ``labels`` directly instead of
    relying on implicit TRL preprocessing behavior.
    """
    print("=" * 60)
    print("LOADING TOKENIZED DATASET FOR SFT")
    print("=" * 60)

    if tokenizer is None:
        raise ValueError("tokenizer is required for tokenized dataset preparation")

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

    if "conversations" in raw_datasets.column_names and "messages" not in raw_datasets.column_names:
        print("Converting 'conversations' key to 'messages' (TRL 0.15.0+ requirement)")
        raw_datasets = raw_datasets.rename_column("conversations", "messages")

    if filter_desirable and "label" in raw_datasets.column_names:
        print("\nFiltering for desirable examples (label=True)...")
        original_size = len(raw_datasets)
        raw_datasets = raw_datasets.filter(lambda x: x["label"] == True)
        filtered_count = len(raw_datasets)
        print(f"Filtered: {original_size} → {filtered_count} examples")
        print(f"Removed: {original_size - filtered_count} undesirable examples")

    print("\nPreparing explicit tokenized SFT features...")
    train_dataset = load_and_prepare_sft_dataset(
        dataset=raw_datasets,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        loss_mask_mode=loss_mask_mode,
        num_proc=num_proc,
        include_text=False,
        chat_template_kwargs=chat_template_kwargs,
    )

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
        elif "input_ids" in example and "labels" in example:
            print("Format: Tokenized SFT")
            print(f"Input IDs: {len(example['input_ids'])} tokens")
            supervised = sum(1 for label in example["labels"] if label != -100)
            print(f"Supervised tokens: {supervised}")
            if "loss_mask_mode" in example:
                print(f"Loss mask mode: {example['loss_mask_mode']}")
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
