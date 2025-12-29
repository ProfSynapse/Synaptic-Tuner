"""
Data loading and preprocessing for MLX SFT training.
Handles JSONL datasets with conversational format.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import mlx.core as mx


@dataclass
class Example:
    """A single training example."""
    input_ids: mx.array
    attention_mask: mx.array
    labels: mx.array  # Same as input_ids for SFT, with padding masked


@dataclass
class Batch:
    """A batch of training examples."""
    input_ids: mx.array      # (batch_size, seq_length)
    attention_mask: mx.array  # (batch_size, seq_length)
    labels: mx.array         # (batch_size, seq_length)


class SFTDataset:
    """Dataset for SFT training."""

    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


class SFTDataLoader:
    """DataLoader for batching SFT examples."""

    def __init__(
        self,
        dataset: SFTDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)
            self.seed += 1

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_examples = [self.dataset[idx] for idx in batch_indices]

            # Stack arrays
            input_ids = mx.stack([ex.input_ids for ex in batch_examples])
            attention_mask = mx.stack([ex.attention_mask for ex in batch_examples])
            labels = mx.stack([ex.labels for ex in batch_examples])

            yield Batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


def sanitize_message(msg: dict) -> dict:
    """Sanitize a message, handling None content and tool_calls."""
    new_msg = dict(msg)

    # Handle None content
    content = new_msg.get("content")
    if content is None:
        content = ""

    # If there are tool_calls, render them as text
    if "tool_calls" in new_msg and new_msg["tool_calls"]:
        tool_parts = []
        for tc in new_msg["tool_calls"]:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")

            if isinstance(args, str):
                try:
                    args_obj = json.loads(args)
                    args_formatted = json.dumps(args_obj, indent=2)
                except json.JSONDecodeError:
                    args_formatted = args
            else:
                args_formatted = json.dumps(args, indent=2)

            tool_parts.append(f"tool_call: {name}\narguments: {args_formatted}")

        if content:
            content = content + "\n\n" + "\n\n".join(tool_parts)
        else:
            content = "\n\n".join(tool_parts)

    new_msg["content"] = content

    # Remove tool_calls since we've rendered them
    if "tool_calls" in new_msg:
        del new_msg["tool_calls"]

    return new_msg


def get_tokenizer(tokenizer_wrapper):
    """Extract the underlying tokenizer from mlx_lm wrapper."""
    # mlx_lm returns a TokenizerWrapper, get the underlying tokenizer
    if hasattr(tokenizer_wrapper, 'tokenizer'):
        return tokenizer_wrapper.tokenizer
    if hasattr(tokenizer_wrapper, '_tokenizer'):
        return tokenizer_wrapper._tokenizer
    return tokenizer_wrapper


def format_for_qwen(messages: List[dict], tokenizer) -> str:
    """
    Format messages for Qwen3 using ChatML template.

    Qwen3 uses ChatML format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """
    # Sanitize messages
    sanitized = [sanitize_message(msg) for msg in messages]

    # Get underlying tokenizer
    tok = get_tokenizer(tokenizer)

    # Use tokenizer's chat template if available
    if hasattr(tok, 'apply_chat_template'):
        try:
            formatted = tok.apply_chat_template(
                sanitized,
                tokenize=False,
                add_generation_prompt=False
            )
            return formatted
        except Exception as e:
            print(f"[WARN] Chat template failed: {e}, using manual format")

    # Manual ChatML formatting as fallback
    formatted_parts = []
    for msg in sanitized:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return "\n".join(formatted_parts)


def load_jsonl(file_path: str) -> List[dict]:
    """Load JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(data)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}")
    return examples


def load_and_prepare_dataset(
    dataset_path: str,
    tokenizer,
    max_seq_length: int = 2048,
    train_split: float = 0.9,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[SFTDataset, Optional[SFTDataset]]:
    """
    Load and prepare dataset for SFT training.

    Args:
        dataset_path: Path to JSONL file
        tokenizer: Tokenizer for encoding
        max_seq_length: Maximum sequence length
        train_split: Fraction for training (rest is validation)
        shuffle: Whether to shuffle data
        seed: Random seed

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"Loading dataset from: {dataset_path}")

    # Load raw data
    raw_data = load_jsonl(dataset_path)
    print(f"Loaded {len(raw_data)} examples")

    # Determine message key
    if raw_data and "messages" in raw_data[0]:
        msg_key = "messages"
    elif raw_data and "conversations" in raw_data[0]:
        msg_key = "conversations"
    else:
        raise ValueError("Dataset must have 'messages' or 'conversations' key")

    print(f"Using message key: '{msg_key}'")

    # Optional: filter for positive examples only (for datasets with labels)
    if raw_data and "label" in raw_data[0]:
        original_count = len(raw_data)
        raw_data = [ex for ex in raw_data if ex.get("label", True) is True]
        filtered_count = len(raw_data)
        print(f"Filtered for positive labels: {original_count} -> {filtered_count}")

    # Shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(raw_data)

    # Get underlying tokenizer for encoding
    tok = get_tokenizer(tokenizer)

    # Ensure pad token is set
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Process examples
    print("Processing examples...")
    examples = []

    for i, data in enumerate(raw_data):
        messages = data[msg_key]

        # Format conversation
        formatted = format_for_qwen(messages, tokenizer)

        # Tokenize
        encoding = tok(
            formatted,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

        input_ids = mx.array(encoding['input_ids'][0])
        attention_mask = mx.array(encoding['attention_mask'][0])

        # Labels: same as input_ids, but mask padding with -100
        labels = mx.where(attention_mask == 0, -100, input_ids)

        examples.append(Example(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ))

        # Progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(raw_data)} examples")

    print(f"Processed {len(examples)} examples total")

    # Split into train/eval
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:] if split_idx < len(examples) else None

    train_dataset = SFTDataset(train_examples)
    eval_dataset = SFTDataset(eval_examples) if eval_examples else None

    print(f"Train: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval: {len(eval_dataset)} examples")

    return train_dataset, eval_dataset


def print_dataset_samples(dataset: SFTDataset, num_samples: int = 2):
    """Print sample examples from the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SAMPLES")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\nExample {i + 1}:")
        print(f"  Input IDs shape: {example.input_ids.shape}")
        print(f"  Attention mask shape: {example.attention_mask.shape}")

        # Count non-padding tokens
        non_pad = mx.sum(example.attention_mask).item()
        print(f"  Non-padding tokens: {int(non_pad)}")

    print("=" * 60)
