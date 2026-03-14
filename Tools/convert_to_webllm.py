#!/usr/bin/env python3
"""
Convert SFT LoRA model to WebLLM/MLC format.

This script:
1. Loads the base model and LoRA adapters
2. Merges them into a full model
3. Converts to MLC format for WebLLM
4. Optionally uploads to HuggingFace

Usage:
    python tools/convert_to_webllm.py \
        --lora-path Trainers/sft/sft_output/20251218_200347/final_model \
        --output-dir ./webllm_output \
        --model-name "Nexus-Quark-Q3.0.1" \
        --quantization q4f16_1
"""

import os
import sys
import json
import shutil
import argparse
import pathlib
import subprocess
from pathlib import Path

# Monkey-patch pathlib to handle WSL permission errors
_original_is_dir = pathlib.Path.is_dir
def _safe_is_dir(self):
    try:
        return _original_is_dir(self)
    except PermissionError:
        return False
pathlib.Path.is_dir = _safe_is_dir

_original_stat = pathlib.Path.stat
def _safe_stat(self, *args, **kwargs):
    try:
        return _original_stat(self, *args, **kwargs)
    except PermissionError:
        raise FileNotFoundError(f"Skipped: {self}")
pathlib.Path.stat = _safe_stat

# Ensure we're using Linux paths to avoid WSL issues
os.environ['PATH'] = '/home/profsynapse/miniconda3/bin:/usr/local/bin:/usr/bin:/bin'
sys.path = [p for p in sys.path if not p.startswith('/mnt/c')]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LoRA model to WebLLM format")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output-dir", type=str, default="./webllm_output", help="Output directory")
    parser.add_argument("--model-name", type=str, default="Nexus-Quark-Q3.0.1", help="Model name for WebLLM")
    parser.add_argument("--quantization", type=str, default="q4f16_1",
                        choices=["q0f16", "q0f32", "q3f16_1", "q4f16_1", "q4f32_1"],
                        help="Quantization format")
    parser.add_argument("--context-size", type=int, default=32768, help="Context window size")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step if already merged")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after conversion")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repo ID for upload")
    return parser.parse_args()


def merge_lora(lora_path: Path, output_path: Path):
    """Merge LoRA adapters with base model."""
    print(f"\n{'='*60}")
    print("STEP 1: Merging LoRA adapters with base model")
    print(f"{'='*60}\n")

    # Read adapter config to get base model
    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    print(f"Base model: {base_model_name}")

    # Check if this is a VL model
    is_vl_model = "VL" in base_model_name or "vl" in base_model_name.lower()
    if is_vl_model:
        print("⚠️  Detected Vision-Language model. Extracting text-only LLM component.")

    # Import transformers and peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    print(f"Loading base model: {base_model_name}")

    # For VL models, we need to extract just the language model
    if is_vl_model:
        from transformers import Qwen3VLForConditionalGeneration

        # Load the VL model
        vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Get the language model component
        base_model = vl_model.model
        print("✓ Extracted language model from VL model")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, str(lora_path))

    print("Merging adapters...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path), trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("✓ Merge complete!")
    return output_path


def convert_to_mlc(model_path: Path, output_path: Path, quantization: str, model_name: str, context_size: int):
    """Convert merged model to MLC format."""
    print(f"\n{'='*60}")
    print("STEP 2: Converting to MLC format")
    print(f"{'='*60}\n")

    mlc_output = output_path / f"{model_name}-{quantization}-MLC"
    mlc_output.mkdir(parents=True, exist_ok=True)

    # Use CLI modules directly
    from mlc_llm.cli import convert_weight, gen_config

    print(f"Converting weights with {quantization} quantization...")
    print(f"  Model: {model_path}")
    print(f"  Output: {mlc_output}")

    # Run convert_weight via subprocess to handle argument parsing
    convert_cmd = [
        sys.executable, "-c",
        f"""
import os
import pathlib

os.environ['PATH'] = '/home/profsynapse/miniconda3/bin:/usr/local/bin:/usr/bin:/bin'

_original_is_dir = pathlib.Path.is_dir
def _safe_is_dir(self):
    try:
        return _original_is_dir(self)
    except PermissionError:
        return False
pathlib.Path.is_dir = _safe_is_dir

_original_stat = pathlib.Path.stat
def _safe_stat(self, *args, **kwargs):
    try:
        return _original_stat(self, *args, **kwargs)
    except PermissionError:
        raise FileNotFoundError(f"Skipped: {{self}}")
pathlib.Path.stat = _safe_stat

import sys
sys.argv = ['convert_weight', '{model_path}', '--quantization', '{quantization}', '-o', '{mlc_output}']
from mlc_llm.cli.convert_weight import main
main()
"""
    ]

    result = subprocess.run(convert_cmd, capture_output=True, text=True, cwd="/home/profsynapse")
    if result.returncode != 0:
        print(f"Error converting weights: {result.stderr}")
        raise RuntimeError(f"Weight conversion failed: {result.stderr}")
    print(result.stdout)

    print("\nGenerating MLC config...")
    gen_config_cmd = [
        sys.executable, "-c",
        f"""
import os
import pathlib

os.environ['PATH'] = '/home/profsynapse/miniconda3/bin:/usr/local/bin:/usr/bin:/bin'

_original_is_dir = pathlib.Path.is_dir
def _safe_is_dir(self):
    try:
        return _original_is_dir(self)
    except PermissionError:
        return False
pathlib.Path.is_dir = _safe_is_dir

_original_stat = pathlib.Path.stat
def _safe_stat(self, *args, **kwargs):
    try:
        return _original_stat(self, *args, **kwargs)
    except PermissionError:
        raise FileNotFoundError(f"Skipped: {{self}}")
pathlib.Path.stat = _safe_stat

import sys
sys.argv = ['gen_config', '{model_path}', '--quantization', '{quantization}', '--conv-template', 'chatml', '--context-window-size', '{context_size}', '-o', '{mlc_output}']
from mlc_llm.cli.gen_config import main
main()
"""
    ]

    result = subprocess.run(gen_config_cmd, capture_output=True, text=True, cwd="/home/profsynapse")
    if result.returncode != 0:
        print(f"Error generating config: {result.stderr}")
        raise RuntimeError(f"Config generation failed: {result.stderr}")
    print(result.stdout)

    print(f"✓ MLC conversion complete! Output: {mlc_output}")
    return mlc_output


def copy_wasm_library(mlc_output: Path, context_size: int):
    """Download or reference existing WASM library."""
    print(f"\n{'='*60}")
    print("STEP 3: Setting up WASM library reference")
    print(f"{'='*60}\n")

    # For custom models, we can reference the prebuilt WASM from mlc-ai
    # or use the existing one from the user's repo
    wasm_info = {
        "note": "WASM library should be hosted separately",
        "recommended_source": "https://huggingface.co/professorsynapse/Nexus-Electron-Q3-MLC",
        "context_variants": {
            "16k": "Nexus-Electron-Q3.0.2-ctx16k-webgpu.wasm",
            "32k": "Nexus-Electron-Q3.0.2-ctx32k-webgpu.wasm"
        }
    }

    wasm_info_path = mlc_output / "wasm_library_info.json"
    with open(wasm_info_path, "w") as f:
        json.dump(wasm_info, f, indent=2)

    print(f"✓ WASM library info saved to: {wasm_info_path}")
    print(f"  Use existing WASM from: {wasm_info['recommended_source']}")

    return wasm_info


def upload_to_hf(mlc_output: Path, repo_id: str):
    """Upload to HuggingFace Hub."""
    print(f"\n{'='*60}")
    print("STEP 4: Uploading to HuggingFace")
    print(f"{'='*60}\n")

    from huggingface_hub import HfApi, login

    # Check for token
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    print(f"Uploading to: {repo_id}")
    api.upload_folder(
        folder_path=str(mlc_output),
        repo_id=repo_id,
        repo_type="model"
    )

    print(f"✓ Upload complete! https://huggingface.co/{repo_id}")


def main():
    args = parse_args()

    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           WebLLM/MLC Model Conversion Tool                   ║
╠══════════════════════════════════════════════════════════════╣
║  LoRA Path:     {str(lora_path):<43} ║
║  Output:        {str(output_dir):<43} ║
║  Model Name:    {args.model_name:<43} ║
║  Quantization:  {args.quantization:<43} ║
║  Context Size:  {args.context_size:<43} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Merge LoRA
    merged_path = output_dir / "merged"
    if not args.skip_merge:
        merged_path = merge_lora(lora_path, merged_path)
    else:
        print("Skipping merge step (--skip-merge)")

    # Step 2: Convert to MLC
    mlc_output = convert_to_mlc(
        merged_path,
        output_dir,
        args.quantization,
        args.model_name,
        args.context_size
    )

    # Step 3: Setup WASM reference
    copy_wasm_library(mlc_output, args.context_size)

    # Step 4: Upload if requested
    if args.upload and args.repo_id:
        upload_to_hf(mlc_output, args.repo_id)
    elif args.upload:
        print("\n⚠️  --upload specified but no --repo-id provided. Skipping upload.")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Conversion Complete!                       ║
╠══════════════════════════════════════════════════════════════╣
║  MLC Output:  {str(mlc_output):<45} ║
║                                                              ║
║  Next Steps:                                                 ║
║  1. Upload weights to HuggingFace                            ║
║  2. Reference WASM from existing MLC repo                    ║
║  3. Configure WebLLM with custom model URLs                  ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
