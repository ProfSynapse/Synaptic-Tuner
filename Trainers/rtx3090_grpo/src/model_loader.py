"""
Model loading helpers for GRPO / GSPO training using Unsloth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from unsloth import FastLanguageModel

# Import shared merge utilities
from shared.model_loading import resolve_model_path

# Optional: FastVisionModel for VL models
try:
    from unsloth import FastVisionModel
    _VISION_AVAILABLE = True
except Exception:
    FastVisionModel = None
    _VISION_AVAILABLE = False


def _is_vision_model(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    indicators = [
        "vl",
        "vision",
        "qwen2-vl",
        "qwen3-vl",
        "llava",
        "pixtral",
        "paligemma",
        "idefics",
    ]
    return any(ind in model_name_lower for ind in indicators)


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[str] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple[object, object, bool]:
    """
    Load a model + tokenizer/processor using Unsloth.

    Returns:
        (model, tokenizer_or_processor, is_vision_model)
    """
    is_vl = _is_vision_model(model_name)

    print("=" * 60)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"4-bit quantization: {load_in_4bit}")
    print(f"dtype: {dtype if dtype else 'auto-detect'}")
    if is_vl:
        print("Detected: Vision-Language model")

    if is_vl:
        if not _VISION_AVAILABLE:
            raise ImportError(
                f"Model '{model_name}' appears to be a vision-language model, "
                "but FastVisionModel is not available in this environment. "
                "Install VL support: pip install --upgrade unsloth unsloth_zoo"
            )
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
        tokenizer_or_processor = processor
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
        tokenizer_or_processor = tokenizer

    print(f"\n✓ Model loaded: {getattr(model.config, '_name_or_path', model_name)}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠ WARNING: CUDA not available!")
    print("=" * 60)

    return model, tokenizer_or_processor, is_vl


def get_text_tokenizer(tokenizer_or_processor: object) -> object:
    """
    If a VL processor is provided, extract its underlying text tokenizer.
    Otherwise, return the tokenizer as-is.
    """
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def apply_lora_adapters(
    model: object,
    is_vision_model: bool,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    bias: str,
    target_modules: list[str],
    use_gradient_checkpointing: str,
    random_state: int,
    use_rslora: bool = False,
    use_dora: bool = False,
) -> object:
    """
    Apply LoRA adapters using Unsloth.
    """
    print("\n" + "=" * 60)
    print("APPLYING LORA ADAPTERS")
    print("=" * 60)
    print(f"LoRA rank: {r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")
    print(f"Target modules: {', '.join(target_modules)}")

    peft_api = FastVisionModel if is_vision_model else FastLanguageModel

    model = peft_api.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        use_dora=use_dora,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params if total_params else 0.0
    print(f"\n✓ LoRA adapters applied")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)

    return model


def load_from_sft_checkpoint(
    base_model_name: str,
    lora_path: str,
    max_seq_length: int = 2048,
    dtype: Optional[str] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple[object, object, bool]:
    """
    Load a merged SFT model for continued GRPO training.

    If given a LoRA checkpoint, automatically merges it first.

    Args:
        base_model_name: Name of base model (for VL detection)
        lora_path: Path to merged model or LoRA checkpoint
        max_seq_length: Maximum sequence length
        dtype: Data type (auto-detect if None)
        load_in_4bit: Whether to use 4-bit quantization
        hf_token: Optional HuggingFace token

    Returns:
        (model, tokenizer, is_vision_model)
    """
    # Use shared merge utilities to resolve path (auto-merge if LoRA)
    model_path, was_merged = resolve_model_path(lora_path, max_seq_length)

    print("=" * 60)
    print("LOADING MERGED SFT MODEL")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Max sequence length: {max_seq_length}")

    is_vl = _is_vision_model(base_model_name)

    # Load the merged model
    if is_vl:
        if not _VISION_AVAILABLE:
            raise ImportError("Vision model support not available")
        model, processor = FastVisionModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
        tokenizer_or_processor = processor
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
        tokenizer_or_processor = tokenizer

    print(f"\n✓ Merged model loaded")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU: {gpu_name}")
    print("=" * 60)

    return model, tokenizer_or_processor, is_vl


def check_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

        print("\nGPU Memory Status:")
        print(f"  Device: {gpu_stats.name}")
        print(f"  Total: {max_memory} GB")
        print(f"  Reserved: {reserved} GB")
        print(f"  Available: {max_memory - reserved} GB")
    else:
        print("CUDA not available")
