#!/usr/bin/env python3
"""
Convert Model to GGUF Format.

Reliable GGUF conversion for both text and Vision-Language models.
Merges LoRA once and creates all quantizations efficiently.

Usage:
    # From repo root with run.sh (recommended)
    ./run.sh tools/convert_to_gguf.py /path/to/final_model my-model-name

    # Direct python
    python tools/convert_to_gguf.py /path/to/final_model my-model-name

    # With custom quantizations
    python tools/convert_to_gguf.py /path/to/model my-model --quants Q4_K_M Q8_0

    # Specify output directory
    python tools/convert_to_gguf.py /path/to/model my-model --output /custom/output

Examples:
    # Convert SFT model
    python tools/convert_to_gguf.py \\
        Trainers/rtx3090_sft/sft_output_rtx3090/20251128_162717/final_model \\
        nexus-tools_sft18

    # Convert KTO model
    python tools/convert_to_gguf.py \\
        Trainers/rtx3090_kto/kto_output_rtx3090/20251203_174806/final_model \\
        nexus-tools_kto5
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "Trainers"))

from shared.upload.converters.gguf_reliable import (
    ReliableGGUFConverter,
    DEFAULT_QUANTIZATIONS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to model directory (LoRA adapters or full model)"
    )

    parser.add_argument(
        "model_name",
        help="Name for output GGUF files (e.g., 'my-model' -> my-model.gguf)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory. Default: model_path parent / model_name"
    )

    parser.add_argument(
        "--quants", "-q",
        nargs="+",
        default=DEFAULT_QUANTIZATIONS,
        help=f"Quantization methods. Default: {' '.join(DEFAULT_QUANTIZATIONS)}"
    )

    parser.add_argument(
        "--dtype",
        choices=["f16", "bf16"],
        default="bf16",
        help="Base GGUF data type. Default: bf16"
    )

    parser.add_argument(
        "--llama-cpp",
        type=Path,
        help="Path to llama.cpp directory. Default: ~/llama.cpp"
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (merged model, etc.)"
    )

    args = parser.parse_args()

    # Validate model path
    if not args.model_path.exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Default: put output in model_path parent / model_name
        output_dir = args.model_path.parent / args.model_name

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              GGUF CONVERSION                                  ║
╠═══════════════════════════════════════════════════════════════╣
║ Model: {str(args.model_path)[:55]:<55} ║
║ Output: {str(output_dir)[:54]:<54} ║
║ Name: {args.model_name:<57} ║
║ Quantizations: {', '.join(args.quants):<48} ║
╚═══════════════════════════════════════════════════════════════╝
""")

    # Create converter
    converter = ReliableGGUFConverter(llama_cpp_dir=args.llama_cpp)

    # Run conversion
    try:
        gguf_files = converter.convert(
            model_path=args.model_path,
            output_dir=output_dir,
            quantizations=args.quants,
            model_name=args.model_name,
            dtype=args.dtype,
            cleanup_temp=not args.keep_temp,
        )

        if gguf_files:
            print(f"\n✓ Conversion complete!")
            print(f"Files created in: {output_dir / 'gguf'}")
            return 0
        else:
            print(f"\n✗ No GGUF files created")
            return 1

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
