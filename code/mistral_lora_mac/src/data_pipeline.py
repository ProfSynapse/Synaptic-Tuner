"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/src/data_pipeline.py

Data Pipeline for MLX Fine-Tuning System

This module handles all data loading, validation, and preprocessing:
- JSONL dataset loading from local files
- Schema validation and error handling
- Conversation formatting for Mistral Instruct template
- Tokenization with padding/truncation
- Train/validation splitting with stratification
- Batch creation and MLX array conversion
- Dataset statistics and reporting

Dependencies:
- transformers: Mistral tokenizer
- mlx: Array conversion
- json: JSONL parsing
- numpy: Data manipulation

Related Files:
- config/config_manager.py: Data configuration (DataConfig)
- src/utils.py: Logging utilities
- src/trainer.py: Consumes batches from this pipeline
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterator, Optional
from dataclasses import dataclass
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from tqdm import tqdm


@dataclass
class Example:
    """Single training example."""
    conversations: List[Dict[str, str]]
    label: bool
    formatted_text: str
    input_ids: mx.array
    attention_mask: mx.array
    labels: mx.array


@dataclass
class Batch:
    """Training batch."""
    input_ids: mx.array  # Shape: (batch_size, seq_length)
    attention_mask: mx.array  # Shape: (batch_size, seq_length)
    labels: mx.array  # Shape: (batch_size, seq_length)
    metadata: Dict[str, Any]  # Original labels, lengths, etc.


class DataValidator:
    """Validates JSONL dataset format and content."""

    def __init__(self, logger):
        """
        Initialize data validator.

        Args:
            logger: StructuredLogger for error reporting
        """
        self.logger = logger
        self.error_counts = {
            'total': 0,
            'missing_conversations': 0,
            'missing_label': 0,
            'invalid_conversation_format': 0,
            'empty_conversations': 0,
            'invalid_role': 0
        }

    def validate_example(self, data: Dict[str, Any], line_num: int) -> bool:
        """
        Validate a single example.

        Args:
            data: Parsed JSON object
            line_num: Line number in file (for error reporting)

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if 'conversations' not in data:
                self.error_counts['missing_conversations'] += 1
                self.logger.warning(f"Line {line_num}: Missing 'conversations' field")
                return False

            if 'label' not in data:
                self.error_counts['missing_label'] += 1
                self.logger.warning(f"Line {line_num}: Missing 'label' field")
                return False

            # Validate conversations
            conversations = data['conversations']
            if not isinstance(conversations, list):
                self.error_counts['invalid_conversation_format'] += 1
                self.logger.warning(f"Line {line_num}: 'conversations' must be a list")
                return False

            if len(conversations) == 0:
                self.error_counts['empty_conversations'] += 1
                self.logger.warning(f"Line {line_num}: 'conversations' cannot be empty")
                return False

            # Validate each message
            for i, msg in enumerate(conversations):
                if not isinstance(msg, dict):
                    self.error_counts['invalid_conversation_format'] += 1
                    self.logger.warning(f"Line {line_num}: Message {i} is not a dict")
                    return False

                # Check for 'role' or 'from' field
                role = msg.get('role') or msg.get('from')
                if not role:
                    self.error_counts['invalid_conversation_format'] += 1
                    self.logger.warning(f"Line {line_num}: Message {i} missing role/from field")
                    return False

                # Check for 'content' or 'value' field
                content = msg.get('content') or msg.get('value')
                if not content:
                    self.error_counts['invalid_conversation_format'] += 1
                    self.logger.warning(f"Line {line_num}: Message {i} missing content/value field")
                    return False

                # Validate role
                if role not in ['user', 'assistant', 'system']:
                    self.error_counts['invalid_role'] += 1
                    self.logger.warning(f"Line {line_num}: Invalid role '{role}' in message {i}")
                    return False

            # Validate label
            label = data['label']
            if not isinstance(label, bool):
                # Try to convert string to bool
                if isinstance(label, str):
                    if label.lower() in ['true', 'desirable']:
                        data['label'] = True
                    elif label.lower() in ['false', 'undesirable']:
                        data['label'] = False
                    else:
                        self.logger.warning(f"Line {line_num}: Invalid label value '{label}'")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Line {line_num}: Validation error: {e}")
            self.error_counts['total'] += 1
            return False

    def check_error_threshold(self, total_examples: int, threshold: float = 0.05):
        """
        Check if error rate exceeds threshold.

        Args:
            total_examples: Total number of examples processed
            threshold: Maximum acceptable error rate (default 5%)

        Raises:
            ValueError: If error rate exceeds threshold
        """
        if total_examples == 0:
            return

        error_rate = self.error_counts['total'] / total_examples
        if error_rate > threshold:
            raise ValueError(
                f"Error rate {error_rate:.2%} exceeds threshold {threshold:.2%}. "
                f"Found {self.error_counts['total']} errors in {total_examples} examples."
            )

    def get_error_summary(self) -> str:
        """Get summary of validation errors."""
        lines = ["Validation Error Summary:"]
        for error_type, count in self.error_counts.items():
            if count > 0:
                lines.append(f"  {error_type}: {count}")
        return "\n".join(lines)


class ConversationFormatter:
    """Formats conversations to Mistral Instruct template."""

    @staticmethod
    def format_for_mistral(conversations: List[Dict[str, str]]) -> str:
        """
        Format conversation to Mistral Instruct template.

        Mistral format:
        <s>[INST] {user_message} [/INST] {assistant_response}</s>

        With system message:
        <s>[INST] {system_message}\n{user_message} [/INST] {assistant_response}</s>

        Multi-turn:
        <s>[INST] {user_1} [/INST] {assistant_1}</s>
        <s>[INST] {user_2} [/INST] {assistant_2}</s>

        Args:
            conversations: List of message dictionaries

        Returns:
            Formatted string ready for tokenization
        """
        formatted_parts = []
        system_message = None
        current_user_msg = None

        for msg in conversations:
            # Get role and content (support both 'role'/'content' and 'from'/'value')
            role = msg.get('role') or msg.get('from')
            content = msg.get('content') or msg.get('value')

            if role == 'system':
                system_message = content
            elif role == 'user':
                current_user_msg = content
            elif role == 'assistant':
                if current_user_msg is not None:
                    # Build instruction part
                    if system_message and not formatted_parts:
                        # Include system message only in first turn
                        inst_part = f"[INST] {system_message}\n{current_user_msg} [/INST]"
                    else:
                        inst_part = f"[INST] {current_user_msg} [/INST]"

                    # Add turn
                    turn = f"<s>{inst_part} {content}</s>"
                    formatted_parts.append(turn)

                    current_user_msg = None

        # Join all turns
        return " ".join(formatted_parts)


class TokenizerWrapper:
    """Wrapper for Hugging Face tokenizer with MLX array conversion."""

    def __init__(self, tokenizer_name: str, max_seq_length: int, logger):
        """
        Initialize tokenizer.

        Args:
            tokenizer_name: Hugging Face model name
            max_seq_length: Maximum sequence length
            logger: StructuredLogger
        """
        self.logger = logger
        self.max_seq_length = max_seq_length

        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.warning("pad_token not set, using eos_token")

        self.logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        self.logger.info(f"Special tokens: PAD={self.tokenizer.pad_token_id}, "
                        f"EOS={self.tokenizer.eos_token_id}, "
                        f"BOS={self.tokenizer.bos_token_id}")

    def tokenize(self, text: str) -> Dict[str, mx.array]:
        """
        Tokenize text and convert to MLX arrays.

        Args:
            text: Input text

        Returns:
            Dictionary with input_ids, attention_mask, and labels as MLX arrays
        """
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'  # Return numpy arrays
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoding['input_ids'][0])
        attention_mask = mx.array(encoding['attention_mask'][0])

        # Create labels (shifted input_ids with padding masked)
        labels = input_ids.copy()
        # Mask padding tokens in labels (-100 is ignore index)
        labels = mx.where(attention_mask == 0, -100, labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class MLXDataset:
    """Dataset class for MLX fine-tuning."""

    def __init__(self, examples: List[Example]):
        """
        Initialize dataset.

        Args:
            examples: List of Example objects
        """
        self.examples = examples

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        """Get example by index."""
        return self.examples[idx]

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels."""
        distribution = {'desirable': 0, 'undesirable': 0}
        for example in self.examples:
            if example.label:
                distribution['desirable'] += 1
            else:
                distribution['undesirable'] += 1
        return distribution


class MLXDataLoader:
    """DataLoader for batching and iteration."""

    def __init__(self, dataset: MLXDataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        """
        Initialize data loader.

        Args:
            dataset: MLXDataset instance
            batch_size: Number of examples per batch
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        """Get number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        # Shuffle indices if needed
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.indices)
            self.seed += 1  # Increment seed for next epoch

        # Generate batches
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_examples = [self.dataset[idx] for idx in batch_indices]

            # Stack arrays
            input_ids = mx.stack([ex.input_ids for ex in batch_examples])
            attention_mask = mx.stack([ex.attention_mask for ex in batch_examples])
            labels = mx.stack([ex.labels for ex in batch_examples])

            # Metadata
            metadata = {
                'labels': [ex.label for ex in batch_examples],
                'num_examples': len(batch_examples)
            }

            yield Batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                metadata=metadata
            )


class DataPipeline:
    """
    Main data pipeline orchestrator.

    Handles:
    - Loading JSONL dataset
    - Validation
    - Formatting
    - Tokenization
    - Train/val splitting
    - DataLoader creation
    """

    def __init__(self, config, logger):
        """
        Initialize data pipeline.

        Args:
            config: Config object with data, model, and training settings
            logger: StructuredLogger
        """
        self.config = config
        self.logger = logger
        self.validator = DataValidator(logger)
        self.formatter = ConversationFormatter()
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load JSONL dataset from file.

        Args:
            dataset_path: Path to JSONL file

        Returns:
            List of parsed JSON objects

        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If too many validation errors
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.logger.info(f"Loading dataset from: {dataset_path}")

        raw_data = []
        valid_count = 0
        error_count = 0

        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Validate
                    if self.validator.validate_example(data, line_num):
                        raw_data.append(data)
                        valid_count += 1
                    else:
                        error_count += 1

                except json.JSONDecodeError as e:
                    self.logger.error(f"Line {line_num}: JSON decode error: {e}")
                    error_count += 1

        self.logger.info(f"Loaded {valid_count} valid examples, {error_count} errors")

        # Check error threshold
        self.validator.check_error_threshold(valid_count + error_count)

        if error_count > 0:
            self.logger.warning(self.validator.get_error_summary())

        return raw_data

    def prepare_examples(self, raw_data: List[Dict[str, Any]]) -> List[Example]:
        """
        Prepare examples: format and tokenize.

        Args:
            raw_data: List of raw JSON objects

        Returns:
            List of Example objects
        """
        self.logger.info("Preparing examples: formatting and tokenizing...")

        examples = []

        for data in tqdm(raw_data, desc="Processing examples"):
            # Format conversation
            formatted_text = self.formatter.format_for_mistral(data['conversations'])

            # Tokenize
            tokenized = self.tokenizer.tokenize(formatted_text)

            # Create example
            example = Example(
                conversations=data['conversations'],
                label=data['label'],
                formatted_text=formatted_text,
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                labels=tokenized['labels']
            )

            examples.append(example)

        self.logger.info(f"Prepared {len(examples)} examples")
        return examples

    def create_train_val_split(self, examples: List[Example]) -> Tuple[List[Example], List[Example]]:
        """
        Split examples into train and validation sets.

        Args:
            examples: List of Example objects

        Returns:
            Tuple of (train_examples, val_examples)
        """
        self.logger.info(f"Splitting dataset: {self.config.data.train_split:.0%} train, "
                        f"{1 - self.config.data.train_split:.0%} validation")

        # Separate by label for stratified split
        desirable = [ex for ex in examples if ex.label]
        undesirable = [ex for ex in examples if not ex.label]

        self.logger.info(f"Desirable examples: {len(desirable)}")
        self.logger.info(f"Undesirable examples: {len(undesirable)}")

        # Shuffle with seed
        random.seed(self.config.data.seed)
        random.shuffle(desirable)
        random.shuffle(undesirable)

        # Split each group
        train_split = self.config.data.train_split
        n_train_desirable = int(len(desirable) * train_split)
        n_train_undesirable = int(len(undesirable) * train_split)

        train_examples = desirable[:n_train_desirable] + undesirable[:n_train_undesirable]
        val_examples = desirable[n_train_desirable:] + undesirable[n_train_undesirable:]

        # Shuffle combined sets
        random.shuffle(train_examples)
        random.shuffle(val_examples)

        self.logger.info(f"Train set: {len(train_examples)} examples")
        self.logger.info(f"Validation set: {len(val_examples)} examples")

        return train_examples, val_examples

    def initialize(self):
        """Initialize tokenizer."""
        self.logger.info("Initializing data pipeline...")

        # Load tokenizer
        self.tokenizer = TokenizerWrapper(
            tokenizer_name=self.config.model.name,
            max_seq_length=self.config.data.max_seq_length,
            logger=self.logger
        )

    def load_and_prepare(self) -> Tuple[MLXDataLoader, MLXDataLoader]:
        """
        Load dataset and create train/val data loaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load raw data
        raw_data = self.load_dataset(self.config.data.dataset_path)

        # Prepare examples
        examples = self.prepare_examples(raw_data)

        # Split
        train_examples, val_examples = self.create_train_val_split(examples)

        # Create datasets
        self.train_dataset = MLXDataset(train_examples)
        self.val_dataset = MLXDataset(val_examples)

        # Log statistics
        self.logger.info("Train set label distribution:")
        train_dist = self.train_dataset.get_label_distribution()
        for label, count in train_dist.items():
            self.logger.info(f"  {label}: {count}")

        self.logger.info("Validation set label distribution:")
        val_dist = self.val_dataset.get_label_distribution()
        for label, count in val_dist.items():
            self.logger.info(f"  {label}: {count}")

        # Create data loaders
        train_loader = MLXDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.training.per_device_batch_size,
            shuffle=self.config.data.shuffle,
            seed=self.config.data.seed
        )

        val_loader = MLXDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.training.per_device_batch_size,
            shuffle=False,
            seed=self.config.data.seed
        )

        self.logger.info(f"Created data loaders: {len(train_loader)} train batches, "
                        f"{len(val_loader)} validation batches")

        return train_loader, val_loader
