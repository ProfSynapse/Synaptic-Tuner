"""
Universal upload CLI.

Provides a command-line interface for uploading models to HuggingFace Hub.
"""

# ============================================================================
# DISABLE TORCH.COMPILE - Required for VL models (Qwen3-VL, LLaVA, etc.)
# Must be set BEFORE importing unsloth or any torch-dependent modules
# ============================================================================
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'

import argparse
import sys
from pathlib import Path

# Use relative imports (this module is part of shared.upload.cli)
from ..platform.windows_patches import ensure_windows_compatibility, ensure_vl_compatibility
from ..core.config import (
    UploadConfig,
    SaveConfig,
    ConversionConfig,
    DocumentationConfig,
)
from ..core.types import ModelPath, to_repository_id, to_credential
from ..orchestrator import UploadOrchestrator


def load_env_file():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        # Try to find .env in various locations
        for path in [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path.cwd().parent.parent / ".env",
            Path(__file__).parent.parent.parent.parent.parent / ".env",
        ]:
            if path.exists():
                load_dotenv(path)
                break
    except ImportError:
        pass


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for upload CLI."""
    parser = argparse.ArgumentParser(
        description="Upload trained model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic upload with 16-bit merge
  python -m shared.upload.cli.upload_cli ./final_model username/my-model

  # Upload with GGUF creation
  python -m shared.upload.cli.upload_cli ./final_model username/my-model --create-gguf

  # LoRA-only upload (fastest, smallest)
  python -m shared.upload.cli.upload_cli ./final_model username/my-model --save-method lora

  # Private repository
  python -m shared.upload.cli.upload_cli ./final_model username/my-model --private
"""
    )

    # Positional arguments (optional when using --gguf-only)
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to saved model directory (e.g., sft_output_rtx3090/20251122/final_model)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        nargs="?",
        default=None,
        help="HuggingFace repo ID (username/model-name)"
    )

    # Upload configuration
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace write token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for organized artifacts (default: auto-detect)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )

    # Save configuration
    parser.add_argument(
        "--save-method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "lora"],
        help="Save method for model (default: merged_16bit)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="7b",
        choices=["3b", "7b", "13b", "20b"],
        help="Model size for memory estimation (default: 7b)"
    )
    parser.add_argument(
        "--no-save-local",
        action="store_true",
        help="Don't save local copy (upload directly)"
    )

    # Conversion configuration
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
        help="GGUF quantization methods (default: Q4_K_M Q5_K_M Q8_0)"
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip uploading standard model (only GGUF)"
    )
    parser.add_argument(
        "--gguf-only",
        type=str,
        metavar="HF_REPO",
        help="Create GGUF from existing HuggingFace repo and push to {repo}-GGUF"
    )

    # Documentation configuration
    parser.add_argument(
        "--training-lineage",
        type=str,
        help="Path to training_lineage.json for comprehensive model card"
    )

    return parser


def gguf_only_mode(hf_repo: str, quantizations: list, token: str):
    """
    Create GGUF from existing HuggingFace repo and push to {repo}-GGUF.

    Args:
        hf_repo: HuggingFace repo ID (e.g., professorsynapse/nexus-tools_sft23)
        quantizations: List of quantization methods
        token: HuggingFace token
    """
    import subprocess
    import tempfile
    import shutil
    from pathlib import Path

    gguf_repo = f"{hf_repo}-GGUF"

    print("\n" + "=" * 60)
    print("GGUF-ONLY MODE")
    print("=" * 60)
    print(f"Source: {hf_repo}")
    print(f"Target: {gguf_repo}")
    print(f"Quantizations: {', '.join(quantizations)}")
    print()

    # Check GGUF dependencies (run.sh should have installed them)
    print("[0/5] Checking GGUF dependencies...")
    try:
        from gguf.vocab import MistralTokenizerType
        print("  ✓ Dependencies ready")
    except ImportError:
        print("  ⚠ Warning: MistralTokenizerType not found")
        print("  Please restart with ./run.sh to install correct gguf version")

    from unsloth import FastLanguageModel

    print("[1/5] Loading model from HuggingFace...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_repo,
        max_seq_length=2048,
        load_in_4bit=False,
        token=token,
    )
    print("  ✓ Model loaded")

    # Create temp directory in WSL native filesystem (better performance)
    temp_base = Path.home() / "tmp_gguf_conversion"
    temp_base.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=str(temp_base))

    try:
        print(f"[2/5] Saving model to temp directory...")
        print(f"  Location: {temp_dir}")
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        print("  ✓ Model saved")

        print(f"\n[3/5] Converting to GGUF format...")
        print("  This may take 10-15 minutes depending on quantizations...")
        print("  Note: Skipping vision components (text-only training)")

        # Use llama.cpp directly to convert (skip vision projector for VLMs)
        # llama.cpp is in Trainers/llama.cpp/
        # This file is in Trainers/shared/upload/cli/, so need 4 parents to get to Trainers/
        llama_cpp_dir = Path(__file__).parent.parent.parent.parent / "llama.cpp"

        if not llama_cpp_dir.exists():
            raise RuntimeError(f"llama.cpp not found at {llama_cpp_dir}")

        import sys
        sys.path.insert(0, str(llama_cpp_dir))

        try:
            # Import llama.cpp converter
            from convert_hf_to_gguf import main as convert_main

            model_name = hf_repo.split("/")[-1]
            base_gguf = Path(temp_dir) / f"{model_name}.gguf"

            # Convert to f16 GGUF (base conversion, no vision)
            print(f"  Converting to f16 GGUF...")
            convert_args = [
                str(temp_dir),
                "--outfile", str(base_gguf),
                "--outtype", "f16",
            ]

            # Monkey-patch sys.argv for the converter
            old_argv = sys.argv
            sys.argv = ["convert_hf_to_gguf.py"] + convert_args

            try:
                convert_main()
            finally:
                sys.argv = old_argv

            print(f"  ✓ Base GGUF created: {base_gguf.name}")

            # Quantize to requested formats
            quantize_bin = llama_cpp_dir / "llama-quantize"

            if not quantize_bin.exists():
                raise RuntimeError(f"llama-quantize not found at {quantize_bin}")

            for quant in quantizations:
                quant_upper = quant.upper()
                output_gguf = Path(temp_dir) / f"{model_name}-{quant_upper}.gguf"

                print(f"  Quantizing to {quant_upper}...")
                result = subprocess.run(
                    [str(quantize_bin), str(base_gguf), str(output_gguf), quant_upper],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # Replace invalid UTF-8 with � instead of crashing
                    timeout=600
                )

                if result.returncode == 0:
                    print(f"  ✓ Created: {output_gguf.name}")
                else:
                    # Decode stderr safely in case of encoding issues
                    stderr_msg = result.stderr[:200] if result.stderr else "Unknown error"
                    print(f"  ⚠ Failed to create {quant_upper}: {stderr_msg}")

            print("  ✓ GGUF files created")

        except Exception as e:
            raise RuntimeError(f"GGUF conversion failed: {e}")

        print(f"\n[4/5] Uploading GGUF files to {gguf_repo}...")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=gguf_repo, exist_ok=True, private=False)

        # Upload all files (GGUF + config)
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=gguf_repo,
            repo_type="model",
        )
        print("  ✓ Upload complete")

        print("\n" + "=" * 60)
        print("[5/5] GGUF UPLOAD COMPLETE")
        print("=" * 60)
        print(f"✓ GGUF files uploaded to: https://huggingface.co/{gguf_repo}")
        print()

    finally:
        # Cleanup temp directory
        if Path(temp_dir).exists():
            print("  Cleaning up temp files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("  ✓ Cleanup complete")


def main(args=None):
    """Main entry point for upload CLI."""
    # Apply Windows patches before any other imports
    ensure_windows_compatibility()

    # Load environment variables
    load_env_file()

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args(args)

    # Get HuggingFace token
    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not hf_token:
        print("Error: HuggingFace token required. Provide via --token or HF_TOKEN env var")
        print("Get token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Handle --gguf-only mode
    if args.gguf_only:
        try:
            gguf_only_mode(args.gguf_only, args.gguf_quantizations, hf_token)
            return 0
        except Exception as e:
            print(f"\n✗ GGUF-only failed: {e}")
            return 1

    # Validate required args for normal mode
    if not args.model_path or not args.repo_id:
        print("Error: model_path and repo_id are required (unless using --gguf-only)")
        print("\nUsage:")
        print("  python -m shared.upload.cli.upload_cli ./model username/repo [options]")
        print("  python -m shared.upload.cli.upload_cli --gguf-only username/existing-repo")
        sys.exit(1)

    # Create configurations
    upload_config = UploadConfig(
        model_path=ModelPath(Path(args.model_path).resolve()),
        repo_id=to_repository_id(args.repo_id),
        credential=to_credential(hf_token),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        private=args.private,
    )

    save_config = SaveConfig(
        strategy_name=args.save_method,
        save_local=not args.no_save_local,
        model_size=args.model_size,
    )

    conversion_config = None
    if args.create_gguf:
        conversion_config = ConversionConfig(
            converter_name="gguf",
            quantizations=args.gguf_quantizations,
        )

    documentation_config = DocumentationConfig(
        training_lineage_path=Path(args.training_lineage) if args.training_lineage else None,
    )

    # Skip standard upload if requested
    if args.skip_standard:
        save_config.save_local = False

    # Create and execute orchestrator
    orchestrator = UploadOrchestrator(
        upload_config=upload_config,
        save_config=save_config,
        conversion_config=conversion_config,
        documentation_config=documentation_config,
    )

    try:
        result = orchestrator.execute()
        orchestrator.print_summary()
        return 0
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
