"""
Model loading with Unsloth optimizations for RTX 3090.
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from typing import Tuple, Optional
import torch


# Mistral-specific chat template (for models using [INST] format)
# Official Mistral format: <s>[INST] user [/INST] assistant</s>
# System messages are prepended before first [INST] (Mistral doesn't have native system support)
MISTRAL_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{% if loop.index == 1 %}{{ message['content'] + ' ' }}{% endif %}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% endif %}{% endfor %}"""

# Generic fallback template for other models (Zephyr/ChatML style)
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"""


def _is_mistral_model(model_name: str) -> bool:
    """Detect if a model is a Mistral model based on name."""
    model_name_lower = model_name.lower()
    return 'mistral' in model_name_lower


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[str] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None
) -> Tuple:
    """
    Load model and tokenizer with Unsloth optimizations.

    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length (1024-4096)
        dtype: Data type (None for auto-detection)
        load_in_4bit: Whether to use 4-bit quantization
        hf_token: HuggingFace token for gated models

    Returns:
        Tuple of (model, tokenizer)
    """
    print("=" * 60)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"4-bit quantization: {load_in_4bit}")
    print(f"dtype: {dtype if dtype else 'auto-detect'}")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=hf_token,
    )

    # Note: Chat template is now applied via Unsloth's get_chat_template() in train_sft.py
    # This ensures proper handling for all model types including VL models
    if tokenizer.chat_template is not None:
        print("✓ Chat template already configured")
    else:
        print("ℹ Chat template will be applied via get_chat_template()")

    # Verify model loaded correctly
    print(f"\n✓ Model loaded: {model.config._name_or_path}")
    # Handle both tokenizers and processors (VL models)
    try:
        vocab_size = len(tokenizer)
        print(f"✓ Tokenizer vocab size: {vocab_size}")
    except TypeError:
        # Vision-Language models use Processors instead of Tokenizers
        if hasattr(tokenizer, 'tokenizer'):
            print(f"✓ Processor tokenizer vocab size: {len(tokenizer.tokenizer)}")
        else:
            print(f"✓ Processor loaded (VL model)")

    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠ WARNING: CUDA not available!")

    print("=" * 60)

    return model, tokenizer


def apply_lora_adapters(
    model,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    bias: str = "none",
    target_modules: list = None,
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 3407
):
    """
    Apply LoRA adapters to the model using Unsloth.

    Args:
        model: Base model to add LoRA to
        r: LoRA rank (3B: 32, 7B: 64, 20B: 128)
        lora_alpha: LoRA alpha scaling factor (typically 2x rank)
        lora_dropout: Dropout probability
        bias: Bias configuration
        target_modules: Modules to apply LoRA to
        use_gradient_checkpointing: Gradient checkpointing mode
        random_state: Random seed

    Returns:
        Model with LoRA adapters applied
    """
    print("\n" + "=" * 60)
    print("APPLYING LORA ADAPTERS")
    print("=" * 60)
    print(f"LoRA rank: {r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    print(f"Target modules: {', '.join(target_modules)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"\n✓ LoRA adapters applied")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)

    return model


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


def create_reference_model(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[str] = None,
    load_in_4bit: bool = True,
    hf_token: Optional[str] = None
):
    """
    Create a reference model for KTO training.

    The reference model is a frozen copy of the base model used to compute
    KL divergence. It should NOT have LoRA adapters applied.

    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length
        dtype: Data type (None for auto-detection)
        load_in_4bit: Whether to use 4-bit quantization
        hf_token: HuggingFace token for gated models

    Returns:
        Reference model (frozen, no LoRA)
    """
    print("\n" + "=" * 60)
    print("CREATING REFERENCE MODEL FOR KTO")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("Note: Reference model will be frozen (no LoRA)")

    # Load reference model (same as policy model but will stay frozen)
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=hf_token,
    )

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    # Set to eval mode
    ref_model.eval()

    print("✓ Reference model created and frozen")
    print("=" * 60)

    return ref_model


def prepare_model_for_inference(model, tokenizer, chat_template: str = "chatml"):
    """
    Prepare model for inference by setting it to inference mode
    and configuring the chat template.

    Args:
        model: Model to prepare
        tokenizer: Tokenizer to configure
        chat_template: Chat template name

    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth.chat_templates import get_chat_template

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    # Set model to inference mode
    FastLanguageModel.for_inference(model)

    print(f"✓ Model prepared for inference with {chat_template} template")

    return model, tokenizer


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")

    # Load small model for testing
    model, tokenizer = load_model_and_tokenizer(
        model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length=1024,
        load_in_4bit=True
    )

    # Apply LoRA
    model = apply_lora_adapters(
        model,
        r=32,
        lora_alpha=64
    )

    # Check memory
    check_gpu_memory()

    print("\n✓ Model loading test completed successfully!")
