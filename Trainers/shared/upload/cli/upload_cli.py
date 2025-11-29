"""
Universal upload CLI.

Provides a command-line interface for uploading models to HuggingFace Hub.
"""

import argparse
import os
import sys
from pathlib import Path

# Add shared module to path if running directly
if __name__ == "__main__":
    shared_path = Path(__file__).parent.parent.parent.parent
    if str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))

from ..platform.windows_patches import ensure_windows_compatibility
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

    # Required arguments
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model directory (e.g., sft_output_rtx3090/20251122/final_model)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
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

    # Documentation configuration
    parser.add_argument(
        "--training-lineage",
        type=str,
        help="Path to training_lineage.json for comprehensive model card"
    )

    return parser


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
