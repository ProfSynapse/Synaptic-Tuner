"""
shared/experiment_tracking/per_example_loss.py

Computes per-example cross entropy losses given a model and a dataset.
This replicates TRL's DataCollatorForCompletionOnlyLM behavior by masking
instruction/prompt tokens when computing loss.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .schema import LossResult

logger = logging.getLogger(__name__)

def _hash_jsonl_line(line: str) -> str:
    """Returns the first 8 characters of the SHA-256 hash of a stripped JSONL line."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()[:8]

def compute_per_example_losses(
    model: Any,
    tokenizer: Any,
    dataset_path: Path | str,
    max_seq_length: int = 2048,
    completion_only: bool = True,
    device: str | None = None,
) -> list[LossResult]:
    """Compute per-example loss for each record in a JSONL dataset.
    
    Args:
        model: Hugging Face model in evaluation mode.
        tokenizer: Tokenizer with chat template.
        dataset_path: Path to JSONL dataset.
        max_seq_length: Truncation threshold.
        completion_only: If True, masks everything before response_template.
        device: Device to place tensors on. Defaults to model device.
        
    Returns:
        List of LossResult objects in the same order as dataset.
    """
    if device is None:
        device = next(model.parameters()).device

    dataset_path = Path(dataset_path)
    
    results = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Computing losses"):
        line_hash = _hash_jsonl_line(line)
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        conv = record.get("conversations", [])
        if not conv:
            conv = record.get("messages", [])
            
        if not conv:
            continue

        if completion_only:
            # Mask everything up to the assistant's response.
            if len(conv) > 0 and conv[-1].get("role") == "assistant":
                prompt_messages = conv[:-1]
                prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
                
                full_str = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                full_tokens = tokenizer.encode(full_str, add_special_tokens=False)
                
                input_ids = full_tokens[:max_seq_length]
                labels = list(input_ids)
                
                # Check overlap to find the prompt boundary reliably
                mask_len = min(len(prompt_tokens), len(labels))
                for j in range(mask_len):
                    if labels[j] == prompt_tokens[j]:
                        labels[j] = -100
                    else:
                        break
            else:
                full_str = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                full_tokens = tokenizer.encode(full_str, add_special_tokens=False)
                input_ids = full_tokens[:max_seq_length]
                labels = list(input_ids)
        else:
            full_str = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer.encode(full_str, add_special_tokens=False)[:max_seq_length]
            labels = list(input_ids)

        # Skip empty or degenerate sequences
        if not input_ids:
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        labels_tensor = torch.tensor([labels], dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, labels=labels_tensor)
            loss = outputs.loss.item()
            
        num_completion_tokens = sum(1 for l in labels if l != -100)
        num_total_tokens = len(input_ids)
        
        results.append(LossResult(
            index=i,
            loss=loss,
            num_completion_tokens=num_completion_tokens,
            num_total_tokens=num_total_tokens,
            jsonl_hash=line_hash,
        ))
            
    return results

def save_losses(losses: list[LossResult], out_path: Path | str) -> None:
    """Save a list of LossResult to a JSONL file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for loss in losses:
            import dataclasses
            f.write(json.dumps(dataclasses.asdict(loss)) + "\n")

def load_losses(in_path: Path | str) -> list[LossResult]:
    """Load a list of LossResult from a JSONL file."""
    results = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # handle cases where index is string or anything
            results.append(LossResult(**data))
    return results

