"""
Reliable GGUF Converter for Text and Vision-Language Models.

This converter bypasses Unsloth's GGUF conversion to directly use llama.cpp,
providing more reliable conversion for both text and VL models.

Key improvements over Unsloth's converter:
1. Merge LoRA once, reuse for all quantizations (saves ~8 min per quant)
2. Better error handling and diagnostics
3. Direct control over llama.cpp commands
4. Proper handling of VL model mmproj conversion
5. WSL-friendly temp directory handling

Usage:
    from Trainers.shared.upload.converters.gguf_reliable import ReliableGGUFConverter

    converter = ReliableGGUFConverter()
    gguf_files = converter.convert(
        model_path="/path/to/final_model",
        output_dir=Path("/path/to/output"),
        quantizations=["Q4_K_M", "Q5_K_M", "Q8_0"],
        model_name="my-model"
    )
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from contextlib import contextmanager

# Brand colors for spinners
AQUA = "#00A99D"
PURPLE = "#93278F"

# Try to import Rich for nice spinners
try:
    from rich.console import Console
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# VL model indicators - models that use vision components
VL_MODEL_INDICATORS = [
    "qwen2-vl", "qwen3-vl", "qwen2_vl", "qwen3_vl",
    "llava", "pixtral", "paligemma", "idefics",
    "ministral3", "ministral-3", "ministral_3", "mistral3",
]

# Default quantizations
DEFAULT_QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

# Bubbling test tube spinner frames
BUBBLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
TUBE_FRAMES = ["│○  │", "│ ○ │", "│  ○│", "│ ○ │", "│○  │", "│•○ │", "│ •○│", "│  •│"]


@contextmanager
def branded_spinner(message: str):
    """
    Context manager that shows a branded spinner during long operations.

    Uses Rich if available, falls back to simple text animation.
    """
    if RICH_AVAILABLE:
        console = Console()
        # Use dots spinner with brand colors
        with Live(
            Text.assemble(
                ("◆ ", f"{PURPLE}"),
                (message, f"{AQUA}"),
                (" ", ""),
                ("⠋", f"{PURPLE}"),
            ),
            console=console,
            refresh_per_second=10,
            transient=True,
        ) as live:
            # Track spinner state
            frame_idx = [0]
            stop_flag = [False]

            def update_spinner():
                while not stop_flag[0]:
                    frame_idx[0] = (frame_idx[0] + 1) % len(BUBBLE_FRAMES)
                    live.update(Text.assemble(
                        ("◆ ", f"{PURPLE}"),
                        (message, f"{AQUA}"),
                        (" ", ""),
                        (BUBBLE_FRAMES[frame_idx[0]], f"{PURPLE}"),
                    ))
                    time.sleep(0.1)

            spinner_thread = threading.Thread(target=update_spinner, daemon=True)
            spinner_thread.start()
            try:
                yield
            finally:
                stop_flag[0] = True
                spinner_thread.join(timeout=0.5)
    else:
        # Simple fallback - just print the message
        print(f"  {message}...", end="", flush=True)
        try:
            yield
        finally:
            print(" done")


class ReliableGGUFConverter:
    """
    Reliable GGUF converter that handles both text and VL models.

    Merges LoRA adapters once and creates all quantizations from that base,
    avoiding the redundant re-merging that Unsloth does.
    """

    def __init__(self, llama_cpp_dir: Optional[Path] = None):
        """
        Initialize the converter.

        Args:
            llama_cpp_dir: Path to llama.cpp directory. If None, will use ~/llama.cpp
        """
        self.llama_cpp_dir = llama_cpp_dir or Path.home() / "llama.cpp"
        self._quantizer_path: Optional[Path] = None
        self._converter_path: Optional[Path] = None

    @property
    def quantizer_path(self) -> Path:
        """Get path to llama-quantize binary."""
        if self._quantizer_path is None:
            for name in ["llama-quantize", "quantize"]:
                for loc in [
                    self.llama_cpp_dir / "build" / "bin" / name,
                    self.llama_cpp_dir / name,
                ]:
                    if loc.exists() and os.access(loc, os.X_OK):
                        self._quantizer_path = loc
                        break
                if self._quantizer_path:
                    break
            if self._quantizer_path is None:
                raise FileNotFoundError(
                    f"llama-quantize not found in {self.llama_cpp_dir}. "
                    "Run setup_llama_cpp() first."
                )
        return self._quantizer_path

    @property
    def converter_path(self) -> Path:
        """Get path to convert_hf_to_gguf.py script."""
        if self._converter_path is None:
            for name in ["convert_hf_to_gguf.py", "unsloth_convert_hf_to_gguf.py"]:
                loc = self.llama_cpp_dir / name
                if loc.exists():
                    self._converter_path = loc
                    break
            if self._converter_path is None:
                raise FileNotFoundError(
                    f"convert_hf_to_gguf.py not found in {self.llama_cpp_dir}. "
                    "Ensure llama.cpp is properly cloned."
                )
        return self._converter_path

    def setup_llama_cpp(self) -> bool:
        """
        Ensure llama.cpp is cloned and built.

        Returns:
            True if setup successful
        """
        print("Setting up llama.cpp...")

        # Check if already built
        try:
            _ = self.quantizer_path
            print(f"  ✓ llama.cpp already set up at {self.llama_cpp_dir}")
            return True
        except FileNotFoundError:
            pass

        # Clone if needed
        if not self.llama_cpp_dir.exists():
            print(f"  Cloning llama.cpp to {self.llama_cpp_dir}...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggerganov/llama.cpp.git",
                 str(self.llama_cpp_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"  ✗ Clone failed: {result.stderr}")
                return False

        # Build
        build_dir = self.llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)

        print("  Building llama.cpp (CPU-only for quantization)...")

        # Configure
        result = subprocess.run(
            ["cmake", "..", "-DGGML_CUDA=OFF"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"  ✗ cmake failed: {result.stderr}")
            return False

        # Build llama-quantize
        result = subprocess.run(
            ["make", "-j4", "llama-quantize"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            print(f"  ✗ make failed: {result.stderr}")
            return False

        # Reset cached paths
        self._quantizer_path = None
        self._converter_path = None

        print(f"  ✓ llama.cpp built successfully")
        return True

    def is_vision_model(self, model_path: Path) -> bool:
        """
        Detect if model is a Vision-Language model.

        Checks multiple indicators that persist even after fine-tuning:
        1. Vision-specific files (vision_tower, visual, image_processor, etc.)
        2. Vision config entries in config.json
        3. Base model name in adapter_config.json
        4. Model type and architectures

        Args:
            model_path: Path to model directory

        Returns:
            True if VL model detected
        """
        # Check for vision-specific files (most reliable for fine-tuned models)
        vision_files = [
            "preprocessor_config.json",  # Image preprocessor config
            "image_processor_config.json",  # Alternate name
        ]
        vision_patterns = ["vision", "visual", "image", "mmproj", "clip"]

        for f in model_path.iterdir():
            fname = f.name.lower()
            # Check for vision-related files
            if any(pattern in fname for pattern in vision_patterns):
                return True

        # Check for specific vision files
        for vf in vision_files:
            if (model_path / vf).exists():
                return True

        # Check config.json for vision config
        config_json = model_path / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    config = json.load(f)

                # Check for vision-related config keys
                vision_keys = ["vision_config", "visual_config", "image_size",
                               "vision_tower", "mm_projector", "image_processor"]
                if any(key in config for key in vision_keys):
                    return True

                # Check model_type
                model_type = config.get("model_type", "").lower()
                if any(vl in model_type for vl in VL_MODEL_INDICATORS):
                    return True

                # Check architectures
                for arch in config.get("architectures", []):
                    if any(vl in arch.lower() for vl in VL_MODEL_INDICATORS):
                        return True
            except (json.JSONDecodeError, KeyError):
                pass

        # Check adapter_config.json (base model reference)
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config) as f:
                    config = json.load(f)
                base_name = config.get("base_model_name_or_path", "").lower()
                if any(vl in base_name for vl in VL_MODEL_INDICATORS):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        return False

    def get_model_architecture(self, model_path: Path) -> Optional[str]:
        """Get model architecture from config."""
        config_json = model_path / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    config = json.load(f)
                archs = config.get("architectures", [])
                if archs:
                    return archs[0]
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def merge_lora_to_16bit(
        self,
        model_path: Path,
        output_dir: Path,
    ) -> Path:
        """
        Merge LoRA adapters into base model at 16-bit precision.

        This is done ONCE and reused for all quantizations.

        Args:
            model_path: Path to LoRA adapter directory
            output_dir: Directory to save merged model

        Returns:
            Path to merged model directory
        """
        print("\n[1/4] Merging LoRA adapters to 16-bit...")

        merged_dir = output_dir / "merged_16bit_temp"

        # Use Unsloth's efficient merge if available
        try:
            from unsloth import FastLanguageModel, FastVisionModel

            is_vl = self.is_vision_model(model_path)

            # Load adapter and base model
            if is_vl:
                print("  Loading VL model with FastVisionModel...")
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=2048,
                    load_in_4bit=False,  # Full precision for merge
                )
            else:
                print("  Loading text model with FastLanguageModel...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=2048,
                    load_in_4bit=False,
                )

            # Save merged model
            print(f"  Saving merged model to {merged_dir}...")
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"  ✓ Merged model saved")

        except ImportError:
            # Fallback to PEFT merge
            print("  Using PEFT for merge (Unsloth not available)...")
            from peft import PeftModel, AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            model = AutoPeftModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype="auto",
            )
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"  ✓ Merged model saved")

        return merged_dir

    def convert_to_gguf_base(
        self,
        merged_model_path: Path,
        output_path: Path,
        dtype: str = "bf16",
        is_mmproj: bool = False,
    ) -> bool:
        """
        Convert merged model to base GGUF (f16/bf16).

        Args:
            merged_model_path: Path to merged HF model
            output_path: Path for output GGUF file
            dtype: Output data type (f16, bf16)
            is_mmproj: If True, convert vision projector only

        Returns:
            True if conversion successful
        """
        component = "vision projector (mmproj)" if is_mmproj else "text model"

        cmd = [
            "python", str(self.converter_path),
            "--outfile", str(output_path),
            "--outtype", dtype,
            str(merged_model_path),
        ]

        if is_mmproj:
            cmd.insert(-1, "--mmproj")

        try:
            with branded_spinner(f"Converting {component} to GGUF"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

            if result.returncode != 0:
                print(f"  ✗ Conversion failed:")
                print(f"    {result.stderr[:500]}")
                return False

            if not output_path.exists():
                print(f"  ✗ Output file not created: {output_path}")
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_path.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ✗ Conversion timed out")
            return False
        except Exception as e:
            print(f"  ✗ Conversion error: {e}")
            return False

    def quantize_gguf(
        self,
        input_gguf: Path,
        output_gguf: Path,
        quant_type: str,
    ) -> bool:
        """
        Quantize a GGUF file using llama-quantize.

        Args:
            input_gguf: Path to input GGUF (f16/bf16)
            output_gguf: Path for quantized output
            quant_type: Quantization method (Q4_K_M, Q5_K_M, Q8_0, etc.)

        Returns:
            True if quantization successful
        """
        # Get CPU count for parallel processing
        import psutil
        n_threads = (psutil.cpu_count() or 4) * 2

        cmd = [
            str(self.quantizer_path),
            str(input_gguf),
            str(output_gguf),
            quant_type,
            str(n_threads),
        ]

        try:
            with branded_spinner(f"Quantizing to {quant_type}"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

            if result.returncode != 0:
                print(f"  ✗ Quantization failed:")
                print(f"    {result.stderr[:500]}")
                return False

            if not output_gguf.exists():
                print(f"  ✗ Output file not created: {output_gguf}")
                return False

            size_mb = output_gguf.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_gguf.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ✗ Quantization timed out")
            return False
        except Exception as e:
            print(f"  ✗ Quantization error: {e}")
            return False

    def convert(
        self,
        model_path: str | Path,
        output_dir: Path,
        quantizations: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        dtype: str = "bf16",
        cleanup_temp: bool = True,
        cleanup: bool = None,  # Alias for cleanup_temp (for compatibility)
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> List[Path]:
        """
        Convert model to GGUF with quantizations.

        This method:
        1. Merges LoRA to 16-bit ONCE
        2. Converts to base GGUF (f16/bf16)
        3. Creates all quantizations from that base
        4. For VL models, also creates mmproj GGUF

        Args:
            model_path: Path to model (LoRA adapters or full model)
            output_dir: Directory to save GGUF files
            quantizations: List of quantization methods
            model_name: Name for output files (default: from model_path)
            dtype: Base GGUF data type (f16 or bf16)
            cleanup_temp: Whether to cleanup temporary files

        Returns:
            List of created GGUF file paths
        """
        # Handle cleanup alias
        if cleanup is not None:
            cleanup_temp = cleanup

        model_path = Path(model_path)
        quantizations = quantizations or DEFAULT_QUANTIZATIONS
        model_name = model_name or model_path.name

        # Setup llama.cpp if needed
        if not self.setup_llama_cpp():
            raise RuntimeError("Failed to setup llama.cpp")

        # Detect model type
        is_vl = self.is_vision_model(model_path)
        arch = self.get_model_architecture(model_path)

        print("\n" + "=" * 60)
        print("GGUF CONVERSION")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Architecture: {arch or 'Unknown'}")
        print(f"Type: {'Vision-Language' if is_vl else 'Text-only'}")
        print(f"Output: {output_dir}")
        print(f"Quantizations: {', '.join(quantizations)}")
        print("=" * 60)

        # Create output directory
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # Use WSL native filesystem for temp files (avoids NTFS issues)
        use_wsl_temp = os.path.exists("/home") and str(model_path).startswith("/mnt/")
        if use_wsl_temp:
            temp_base = Path.home() / "tmp_gguf"
            temp_base.mkdir(exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(dir=temp_base))
            print(f"Using WSL native temp: {temp_dir}")
        else:
            temp_dir = Path(tempfile.mkdtemp())

        created_files: List[Path] = []

        try:
            # Step 1: Merge LoRA to 16-bit
            merged_dir = self.merge_lora_to_16bit(model_path, temp_dir)

            # Step 2: Convert to base GGUF
            print(f"\n[2/4] Converting to base GGUF ({dtype})...")
            base_gguf = temp_dir / f"{model_name}.gguf"

            if not self.convert_to_gguf_base(merged_dir, base_gguf, dtype):
                raise RuntimeError("Base GGUF conversion failed")

            # Copy base GGUF to output
            final_base = gguf_dir / f"{model_name}.gguf"
            shutil.copy2(base_gguf, final_base)
            created_files.append(final_base)

            # Step 2b: For VL models, create mmproj GGUF
            if is_vl:
                print("\n[2b/4] Creating vision projector (mmproj) GGUF...")
                mmproj_gguf = gguf_dir / f"{model_name}-mmproj.gguf"

                if self.convert_to_gguf_base(merged_dir, mmproj_gguf, dtype, is_mmproj=True):
                    created_files.append(mmproj_gguf)
                else:
                    print("  ⚠ mmproj conversion failed - vision features may not work")
                    print("    This model can still be used for text-only inference")

            # Step 3: Create quantizations
            print(f"\n[3/4] Creating quantizations...")

            for quant in quantizations:
                quant_upper = quant.upper()
                quant_file = gguf_dir / f"{model_name}-{quant_upper}.gguf"

                if self.quantize_gguf(base_gguf, quant_file, quant_upper):
                    created_files.append(quant_file)

            # Step 4: Summary
            print(f"\n[4/4] Summary")
            print("=" * 60)
            print(f"✓ Created {len(created_files)} GGUF files:")
            for f in created_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
            print(f"\nOutput directory: {gguf_dir}")
            print("=" * 60)

            return created_files

        finally:
            # Cleanup temp files
            if cleanup_temp and temp_dir.exists():
                print(f"\nCleaning up temp files...")
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"✓ Cleaned up {temp_dir}")


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert model to GGUF")
    parser.add_argument("model_path", help="Path to model (LoRA or full)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--name", help="Model name for output files")
    parser.add_argument("--quants", nargs="+", default=DEFAULT_QUANTIZATIONS,
                        help="Quantization methods")
    parser.add_argument("--dtype", default="bf16", choices=["f16", "bf16"],
                        help="Base GGUF data type")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep temp files")

    args = parser.parse_args()

    converter = ReliableGGUFConverter()
    converter.convert(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        quantizations=args.quants,
        model_name=args.name,
        dtype=args.dtype,
        cleanup_temp=not args.no_cleanup,
    )


if __name__ == "__main__":
    main()
