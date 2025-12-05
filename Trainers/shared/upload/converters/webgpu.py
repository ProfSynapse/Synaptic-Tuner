"""
WebGPU/MLC-LLM Converter for Browser Deployment.

This converter uses MLC-LLM to create WebGPU-compatible model files
that can run in browsers via WebLLM.

Output files:
- Quantized weights in MLC format
- WebAssembly inference library (.wasm)
- Configuration files (mlc-chat-config.json, tokenizer)

Usage:
    from Trainers.shared.upload.converters.webgpu import WebGPUConverter

    converter = WebGPUConverter()
    files = converter.convert(
        model_path="/path/to/model",
        output_dir=Path("/path/to/output"),
        quantizations=["q4f16_1"],
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
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Default quantizations for WebGPU
DEFAULT_QUANTIZATIONS = ["q4f16_1"]

# Supported quantization methods
SUPPORTED_QUANTIZATIONS = [
    "q0f16",      # No quantization, float16
    "q0f32",      # No quantization, float32
    "q4f16_1",    # 4-bit with float16 (recommended)
    "q4f32_1",    # 4-bit with float32
    "q3f16_1",    # 3-bit with float16
]

# Conversation templates for common models
CONV_TEMPLATES = {
    "mistral": "mistral_default",
    "qwen": "qwen2",
    "llama": "llama-3",
    "phi": "phi-3",
    "gemma": "gemma",
}

# Spinner frames
BUBBLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


@contextmanager
def branded_spinner(message: str):
    """Context manager for branded spinner during long operations."""
    if RICH_AVAILABLE:
        console = Console()
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
        print(f"  {message}...", end="", flush=True)
        try:
            yield
        finally:
            print(" done")


class WebGPUConverter:
    """
    WebGPU/MLC-LLM converter for browser deployment.

    Converts HuggingFace models to MLC format and compiles WebAssembly
    libraries for WebGPU inference in browsers.
    """

    def __init__(self, mlc_llm_path: Optional[Path] = None):
        """
        Initialize the converter.

        Args:
            mlc_llm_path: Path to mlc_llm installation (uses system if None)
        """
        self.mlc_llm_path = mlc_llm_path
        self._mlc_available: Optional[bool] = None

    def check_mlc_available(self) -> Tuple[bool, str]:
        """
        Check if MLC-LLM is available.

        Returns:
            Tuple of (available, message)
        """
        try:
            result = subprocess.run(
                ["mlc_llm", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, "MLC-LLM is available"
            return False, "MLC-LLM command failed"
        except FileNotFoundError:
            return False, "MLC-LLM not installed. Run setup with --with-webgpu"
        except subprocess.TimeoutExpired:
            return False, "MLC-LLM command timed out"
        except Exception as e:
            return False, f"Error checking MLC-LLM: {e}"

    def setup_mlc_llm(self) -> bool:
        """
        Setup MLC-LLM if not available.

        Returns:
            True if setup successful
        """
        print("Setting up MLC-LLM...")

        available, msg = self.check_mlc_available()
        if available:
            print(f"  ✓ {msg}")
            return True

        print(f"  MLC-LLM not found. Installing...")

        try:
            # Install mlc-llm-nightly (includes pre-built binaries)
            with branded_spinner("Installing MLC-LLM"):
                result = subprocess.run(
                    ["pip", "install", "--pre", "mlc-llm-nightly", "mlc-ai-nightly", "-f",
                     "https://mlc.ai/wheels"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

            if result.returncode != 0:
                print(f"  ✗ Installation failed: {result.stderr[:500]}")
                return False

            # Verify installation
            available, msg = self.check_mlc_available()
            if available:
                print(f"  ✓ MLC-LLM installed successfully")
                return True
            else:
                print(f"  ✗ Installation verification failed: {msg}")
                return False

        except subprocess.TimeoutExpired:
            print("  ✗ Installation timed out")
            return False
        except Exception as e:
            print(f"  ✗ Installation error: {e}")
            return False

    def detect_conv_template(self, model_path: Path) -> str:
        """
        Detect appropriate conversation template for model.

        Args:
            model_path: Path to model directory

        Returns:
            Conversation template name
        """
        # Check config.json for model type
        config_json = model_path / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    config = json.load(f)

                model_type = config.get("model_type", "").lower()
                architectures = config.get("architectures", [])

                # Match against known templates
                for key, template in CONV_TEMPLATES.items():
                    if key in model_type:
                        return template
                    for arch in architectures:
                        if key in arch.lower():
                            return template
            except (json.JSONDecodeError, KeyError):
                pass

        # Check adapter_config.json for base model
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config) as f:
                    config = json.load(f)
                base_name = config.get("base_model_name_or_path", "").lower()

                for key, template in CONV_TEMPLATES.items():
                    if key in base_name:
                        return template
            except (json.JSONDecodeError, KeyError):
                pass

        # Default to mistral template
        return "mistral_default"

    def convert_weights(
        self,
        model_path: Path,
        output_dir: Path,
        quantization: str = "q4f16_1",
    ) -> bool:
        """
        Convert model weights to MLC format.

        Args:
            model_path: Path to HuggingFace model
            output_dir: Output directory for converted weights
            quantization: Quantization method

        Returns:
            True if successful
        """
        cmd = [
            "mlc_llm", "convert_weight",
            str(model_path),
            "--quantization", quantization,
            "-o", str(output_dir),
        ]

        try:
            with branded_spinner(f"Converting weights ({quantization})"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes for large models
                )

            if result.returncode != 0:
                print(f"  ✗ Weight conversion failed:")
                print(f"    {result.stderr[:500]}")
                return False

            print(f"  ✓ Weights converted to MLC format")
            return True

        except subprocess.TimeoutExpired:
            print("  ✗ Weight conversion timed out")
            return False
        except Exception as e:
            print(f"  ✗ Weight conversion error: {e}")
            return False

    def generate_config(
        self,
        model_path: Path,
        output_dir: Path,
        quantization: str = "q4f16_1",
        conv_template: Optional[str] = None,
        context_window_size: int = 4096,
        prefill_chunk_size: int = 1024,
    ) -> bool:
        """
        Generate MLC chat configuration.

        Args:
            model_path: Path to original HuggingFace model
            output_dir: Output directory (with converted weights)
            quantization: Quantization method used
            conv_template: Conversation template (auto-detected if None)
            context_window_size: Maximum context window
            prefill_chunk_size: Prefill chunk size for memory efficiency

        Returns:
            True if successful
        """
        if conv_template is None:
            conv_template = self.detect_conv_template(model_path)

        cmd = [
            "mlc_llm", "gen_config",
            str(model_path),
            "--quantization", quantization,
            "--conv-template", conv_template,
            "--context-window-size", str(context_window_size),
            "--prefill-chunk-size", str(prefill_chunk_size),
            "-o", str(output_dir),
        ]

        try:
            with branded_spinner("Generating MLC config"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

            if result.returncode != 0:
                print(f"  ✗ Config generation failed:")
                print(f"    {result.stderr[:500]}")
                return False

            print(f"  ✓ Config generated (template: {conv_template})")
            return True

        except subprocess.TimeoutExpired:
            print("  ✗ Config generation timed out")
            return False
        except Exception as e:
            print(f"  ✗ Config generation error: {e}")
            return False

    def compile_webgpu(
        self,
        config_path: Path,
        output_path: Path,
    ) -> bool:
        """
        Compile model for WebGPU (creates .wasm file).

        Args:
            config_path: Path to mlc-chat-config.json
            output_path: Path for output .wasm file

        Returns:
            True if successful
        """
        cmd = [
            "mlc_llm", "compile",
            str(config_path),
            "--device", "webgpu",
            "-o", str(output_path),
        ]

        try:
            with branded_spinner("Compiling WebGPU library"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes for compilation
                )

            if result.returncode != 0:
                print(f"  ✗ WebGPU compilation failed:")
                print(f"    {result.stderr[:500]}")
                return False

            if not output_path.exists():
                print(f"  ✗ Output file not created: {output_path}")
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_path.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print("  ✗ WebGPU compilation timed out")
            return False
        except Exception as e:
            print(f"  ✗ WebGPU compilation error: {e}")
            return False

    def convert(
        self,
        model_path: str | Path,
        output_dir: Path,
        quantizations: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        conv_template: Optional[str] = None,
        context_window_size: int = 4096,
        compile_wasm: bool = True,
        cleanup_temp: bool = True,
        **kwargs,
    ) -> List[Path]:
        """
        Convert model to WebGPU format.

        This method:
        1. Converts weights to MLC format with quantization
        2. Generates MLC chat configuration
        3. Optionally compiles WebAssembly library for WebGPU

        Args:
            model_path: Path to model (LoRA adapters or full model)
            output_dir: Directory to save converted files
            quantizations: List of quantization methods (default: q4f16_1)
            model_name: Name for output files
            conv_template: Conversation template (auto-detected if None)
            context_window_size: Maximum context window
            compile_wasm: Whether to compile .wasm file
            cleanup_temp: Whether to cleanup temporary files

        Returns:
            List of created file/directory paths
        """
        model_path = Path(model_path)
        quantizations = quantizations or DEFAULT_QUANTIZATIONS
        model_name = model_name or model_path.name

        # Setup MLC-LLM if needed
        if not self.setup_mlc_llm():
            raise RuntimeError("Failed to setup MLC-LLM")

        # Detect conversation template
        if conv_template is None:
            conv_template = self.detect_conv_template(model_path)

        print("\n" + "=" * 60)
        print("WEBGPU CONVERSION")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Output: {output_dir}")
        print(f"Name: {model_name}")
        print(f"Quantizations: {', '.join(quantizations)}")
        print(f"Conv Template: {conv_template}")
        print(f"Context Window: {context_window_size}")
        print(f"Compile WASM: {compile_wasm}")
        print("=" * 60)

        # Create output directory
        webgpu_dir = output_dir / "webgpu"
        webgpu_dir.mkdir(parents=True, exist_ok=True)

        created_paths: List[Path] = []

        # Check if this is a LoRA model that needs merging first
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            print("\n⚠ LoRA adapters detected - merging first...")
            # Use temp dir for merged model
            temp_dir = Path(tempfile.mkdtemp())
            merged_path = temp_dir / "merged"

            try:
                from unsloth import FastLanguageModel

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=context_window_size,
                    load_in_4bit=False,
                )
                model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
                model_path = merged_path
                print("  ✓ LoRA merged to 16-bit")
            except ImportError:
                print("  ✗ Unsloth not available for LoRA merge")
                print("  Please merge LoRA adapters first or install Unsloth")
                return []

        for i, quant in enumerate(quantizations, 1):
            quant_name = f"{model_name}-{quant}-MLC"
            quant_dir = webgpu_dir / quant_name

            print(f"\n[{i}/{len(quantizations)}] Processing {quant}...")

            # Step 1: Convert weights
            print("\n  Step 1/3: Converting weights...")
            if not self.convert_weights(model_path, quant_dir, quant):
                print(f"  ⚠ Skipping {quant} due to conversion failure")
                continue

            # Step 2: Generate config
            print("\n  Step 2/3: Generating config...")
            if not self.generate_config(
                model_path, quant_dir, quant, conv_template,
                context_window_size=context_window_size
            ):
                print(f"  ⚠ Config generation failed for {quant}")
                continue

            created_paths.append(quant_dir)

            # Step 3: Compile WASM (optional)
            if compile_wasm:
                print("\n  Step 3/3: Compiling WebGPU library...")
                config_path = quant_dir / "mlc-chat-config.json"
                wasm_path = webgpu_dir / f"{model_name}-{quant}-webgpu.wasm"

                if self.compile_webgpu(config_path, wasm_path):
                    created_paths.append(wasm_path)
                else:
                    print("  ⚠ WASM compilation failed (weights still usable)")
            else:
                print("\n  Step 3/3: Skipping WASM compilation")

        # Summary
        print(f"\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"✓ Created {len(created_paths)} outputs:")
        for p in created_paths:
            if p.is_dir():
                # Calculate directory size
                total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"  - {p.name}/ ({size_mb:.1f} MB)")
            else:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  - {p.name} ({size_mb:.1f} MB)")

        print(f"\nOutput directory: {webgpu_dir}")
        print("=" * 60)

        print("\nNext steps:")
        print("  1. Upload weights folder to HuggingFace")
        print("  2. Host .wasm file (GitHub releases or CDN)")
        print("  3. Register in WebLLM with ModelRecord")
        print("  4. See: https://webllm.mlc.ai/docs/")

        return created_paths

    def supported_quantizations(self) -> List[str]:
        """Get list of supported quantization methods."""
        return SUPPORTED_QUANTIZATIONS.copy()

    def validate_environment(self) -> Tuple[bool, str]:
        """Validate that MLC-LLM is available."""
        return self.check_mlc_available()


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert model to WebGPU format")
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--name", help="Model name for output files")
    parser.add_argument("--quants", nargs="+", default=DEFAULT_QUANTIZATIONS,
                        help="Quantization methods")
    parser.add_argument("--conv-template", help="Conversation template")
    parser.add_argument("--context-window", type=int, default=4096,
                        help="Context window size")
    parser.add_argument("--no-wasm", action="store_true",
                        help="Skip WASM compilation")

    args = parser.parse_args()

    converter = WebGPUConverter()
    converter.convert(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        quantizations=args.quants,
        model_name=args.name,
        conv_template=args.conv_template,
        context_window_size=args.context_window,
        compile_wasm=not args.no_wasm,
    )


if __name__ == "__main__":
    main()
