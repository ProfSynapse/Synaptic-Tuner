"""
Upload trained model to HuggingFace Hub.
Supports both standard and GGUF formats.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads HF_TOKEN from .env file
except ImportError:
    pass  # dotenv not required, can use environment variables directly


def upload_standard_model(
    model_path: str,
    repo_id: str,
    hf_token: str,
    save_method: str = "merged_16bit",
    private: bool = False
):
    """
    Upload standard model to HuggingFace Hub.

    Args:
        model_path: Path to saved model
        repo_id: HuggingFace repo ID (username/model-name)
        hf_token: HuggingFace write token
        save_method: Save method (merged_16bit, merged_4bit, or lora)
        private: Whether to make repo private
    """
    print("=" * 60)
    print("UPLOADING STANDARD MODEL")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Save method: {save_method}")
    print(f"Private: {private}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True if "4bit" not in save_method else False,
    )

    # Upload
    print(f"\nUploading to HuggingFace...")
    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method=save_method,
        token=hf_token,
        private=private
    )

    print(f"\n✓ Model uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_id}")


def create_gguf_versions(
    model_path: str,
    output_dir: str,
    quantizations: List[str] = None
) -> List[str]:
    """
    Create GGUF versions of the model.

    Args:
        model_path: Path to model directory
        output_dir: Directory to save GGUF files
        quantizations: List of quantization methods (e.g., ["Q4_K_M", "Q5_K_M"])

    Returns:
        List of created GGUF file paths
    """
    if quantizations is None:
        quantizations = ["Q4_K_M", "Q5_K_M", "Q8_0"]

    print("=" * 60)
    print("CREATING GGUF VERSIONS")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Quantizations: {', '.join(quantizations)}")
    print()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, save merged model locally if not already merged
    merged_dir = output_dir / "merged_model"
    print("[1/4] Saving merged model locally...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # Need full precision for GGUF
    )
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"✓ Merged model saved to: {merged_dir}")

    # Clone llama.cpp if needed
    print("\n[2/4] Setting up llama.cpp...")
    llamacpp_dir = output_dir / "llama.cpp"

    if not llamacpp_dir.exists():
        print("Cloning llama.cpp repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llamacpp_dir)],
            check=True,
            capture_output=True
        )
        print("Building llama.cpp...")
        subprocess.run(
            ["make", "-j"],
            cwd=str(llamacpp_dir),
            check=True,
            capture_output=True
        )
    print(f"✓ llama.cpp ready at: {llamacpp_dir}")

    # Convert to GGUF base format
    print("\n[3/4] Converting to GGUF base format (f16)...")
    base_gguf = output_dir / "model-unsloth.gguf"

    subprocess.run([
        "python",
        str(llamacpp_dir / "convert_hf_to_gguf.py"),
        str(merged_dir),
        "--outfile", str(base_gguf),
        "--outtype", "f16"
    ], check=True)

    print(f"✓ Base GGUF created: {base_gguf}")

    # Create quantized versions
    print(f"\n[4/4] Creating {len(quantizations)} quantized versions...")
    gguf_files = [base_gguf]

    for quant in quantizations:
        output_file = output_dir / f"model-unsloth-{quant}.gguf"
        print(f"  Creating {quant} quantization...")

        subprocess.run([
            str(llamacpp_dir / "llama-quantize"),
            str(base_gguf),
            str(output_file),
            quant
        ], check=True)

        gguf_files.append(output_file)
        print(f"  ✓ {output_file.name}")

    print(f"\n✓ All GGUF files created: {len(gguf_files)} total")
    return [str(f) for f in gguf_files]


def upload_gguf_files(
    gguf_files: List[str],
    repo_id: str,
    hf_token: str
):
    """
    Upload GGUF files to HuggingFace Hub.

    Args:
        gguf_files: List of GGUF file paths
        repo_id: HuggingFace repo ID
        hf_token: HuggingFace write token
    """
    print("\n" + "=" * 60)
    print("UPLOADING GGUF FILES")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Files to upload: {len(gguf_files)}")
    print()

    api = HfApi()

    for i, gguf_file in enumerate(gguf_files, 1):
        file_path = Path(gguf_file)
        print(f"[{i}/{len(gguf_files)}] Uploading {file_path.name}...")

        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )

        print(f"  ✓ Uploaded")

    print(f"\n✓ All GGUF files uploaded!")
    print(f"View at: https://huggingface.co/{repo_id}/tree/main")


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model directory"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo ID (username/model-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace write token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "lora"],
        help="Save method for standard model"
    )
    parser.add_argument(
        "--create-gguf",
        action="store_true",
        help="Create and upload GGUF versions"
    )
    parser.add_argument(
        "--gguf-quantizations",
        type=str,
        nargs="+",
        default=["Q4_K_M", "Q5_K_M", "Q8_0"],
        help="GGUF quantization methods"
    )
    parser.add_argument(
        "--gguf-output-dir",
        type=str,
        default="./gguf_output",
        help="Directory for GGUF conversion files"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip uploading standard model (only GGUF)"
    )

    args = parser.parse_args()

    # Get HuggingFace token
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HuggingFace token required. Provide via --token or HF_TOKEN env var")
        print("Get token from: https://huggingface.co/settings/tokens")
        return

    # Upload standard model
    if not args.skip_standard:
        upload_standard_model(
            model_path=args.model_path,
            repo_id=args.repo_id,
            hf_token=hf_token,
            save_method=args.save_method,
            private=args.private
        )

    # Create and upload GGUF versions
    if args.create_gguf:
        gguf_files = create_gguf_versions(
            model_path=args.model_path,
            output_dir=args.gguf_output_dir,
            quantizations=args.gguf_quantizations
        )

        upload_gguf_files(
            gguf_files=gguf_files,
            repo_id=args.repo_id,
            hf_token=hf_token
        )

    print("\n" + "=" * 60)
    print("✓ ALL UPLOADS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
