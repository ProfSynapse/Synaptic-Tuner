"""
Dataset loading and formatting for GRPO / GSPO training.

Expected dataset shape for tool-calling:
  - prompt: list[{'role': ..., 'content': ...}] OR a pre-formatted prompt string
  - additional columns used by reward functions (e.g. ground_truth_tool)
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset


def load_raw_dataset(
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    local_file: Optional[str] = None,
    num_proc: int = 1,
) -> Dataset:
    print("=" * 60)
    print("LOADING DATASET FOR GRPO")
    print("=" * 60)

    if local_file:
        print(f"Loading from local file: {local_file}")
        dataset = load_dataset("json", data_files=local_file, split="train")
    elif dataset_name:
        print(f"Loading from HuggingFace: {dataset_name}")
        if data_files:
            print(f"Using file: {data_files}")
            dataset = load_dataset(dataset_name, data_files=data_files, num_proc=num_proc)["train"]
        else:
            dataset = load_dataset(dataset_name, num_proc=num_proc)["train"]
    else:
        raise ValueError("Must provide either dataset_name or local_file")

    print(f"\nRaw dataset size: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print("=" * 60)
    return dataset


def format_dataset_for_grpo(
    dataset: Dataset,
    tokenizer: object,
    prompt_column: str = "prompt",
    num_proc: int = 1,
) -> Dataset:
    """
    Ensure dataset has a string `prompt` field usable by GRPOTrainer.
    """

    def _format_example(example):
        prompt_value = example.get(prompt_column)
        if prompt_value is None:
            raise KeyError(f"Dataset missing prompt column '{prompt_column}'")

        if isinstance(prompt_value, str):
            prompt_text = prompt_value
        else:
            prompt_text = tokenizer.apply_chat_template(
                prompt_value,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Keep all original fields (for rewards), but replace prompt with string
        result = dict(example)
        result["prompt"] = prompt_text
        return result

    remove_columns = [prompt_column] if prompt_column in dataset.column_names else []

    formatted = dataset.map(
        _format_example,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Formatting for GRPO",
    )

    print(f"\nDataset formatted: {len(formatted)} examples")
    print(f"Columns: {formatted.column_names}")
    print(f"Sample formatted prompt (truncated):\n{formatted[0]['prompt'][:300]}")

    return formatted


def print_dataset_samples(dataset: Dataset, num_samples: int = 2):
    print("\nDataset samples:")
    print("=" * 60)
    for i in range(min(num_samples, len(dataset))):
        ex = dataset[i]
        keys = list(ex.keys())
        print(f"\nExample {i+1}: keys={keys}")
        if "prompt" in ex:
            p = ex["prompt"]
            if isinstance(p, str):
                print(f"prompt: {p[:200]}...")
            else:
                print(f"prompt (messages): {p[:2]} ...")
        if "ground_truth_tool" in ex:
            print(f"ground_truth_tool: {ex['ground_truth_tool']}")
        if "ground_truth_args_json" in ex:
            args_preview = ex['ground_truth_args_json'][:200] if isinstance(ex['ground_truth_args_json'], str) else str(ex['ground_truth_args_json'])[:200]
            print(f"ground_truth_args_json: {args_preview}...")
        print("-" * 60)
