"""
GGUF format converter.

Converts models to GGUF format for use with llama.cpp and Ollama.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from .base import BaseConverter
from ..core.types import ModelPath, QuantizationMethod
from ..core.exceptions import ConversionError, DependencyError
from ..platform.gpu_memory import ensure_gpu_memory, GPU_MEMORY_REQUIREMENTS
from ..platform.filesystem import (
    get_native_temp_dir,
    cleanup_temp_directory,
    is_windows_filesystem,
)


class GGUFConverter(BaseConverter):
    """
    Converter for GGUF format.

    GGUF is the format used by llama.cpp and Ollama for efficient inference.
    Supports various quantization levels (Q4_K_M, Q5_K_M, Q8_0, etc.).
    """

    # Default quantizations to create
    DEFAULT_QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

    # All supported quantization methods
    SUPPORTED_QUANTIZATIONS = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0",
        "F16", "F32",
    ]

    @property
    def name(self) -> str:
        return "gguf"

    def supported_quantizations(self) -> List[str]:
        """Get list of supported quantization methods."""
        return self.SUPPORTED_QUANTIZATIONS.copy()

    def validate_environment(self) -> Tuple[bool, str]:
        """
        Validate that required tools are available.

        GGUF conversion requires:
        - git (for cloning llama.cpp)
        - cmake or make (for building)
        - Python (for convert script)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check git
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "git is required but not found"

        # Check build tools
        has_make = False
        has_cmake = False

        try:
            subprocess.run(["make", "--version"], capture_output=True, check=True)
            has_make = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
            has_cmake = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        if not has_make and not has_cmake:
            return False, "make or cmake is required but not found"

        return True, ""

    def convert(
        self,
        model_path: ModelPath,
        output_dir: Path,
        quantizations: Optional[List[QuantizationMethod]] = None,
        **options
    ) -> List[Path]:
        """
        Convert model to GGUF format with quantizations.

        Args:
            model_path: Path to the source model (LoRA adapters)
            output_dir: Directory to save GGUF files
            quantizations: List of quantization methods (default: Q4_K_M, Q5_K_M, Q8_0)
            **options:
                - model_name: Name for output files (default: from model_path)
                - cleanup: Whether to cleanup temp files (default: True)
                - model_size: Model size for memory estimation (default: "7b")

        Returns:
            List of paths to created GGUF files
        """
        if quantizations is None:
            quantizations = self.DEFAULT_QUANTIZATIONS

        # Validate quantizations
        for quant in quantizations:
            if quant not in self.SUPPORTED_QUANTIZATIONS:
                raise ConversionError(
                    f"Unsupported quantization: {quant}",
                    details={"supported": self.SUPPORTED_QUANTIZATIONS}
                )

        # Validate environment
        is_valid, error = self.validate_environment()
        if not is_valid:
            raise DependencyError("build-tools", error)

        model_name = options.get("model_name", Path(model_path).name)
        cleanup = options.get("cleanup", True)
        model_size = options.get("model_size", "7b")

        # Check GPU memory
        required_gb = GPU_MEMORY_REQUIREMENTS.get(f"{model_size}_gguf", 14.0)
        if not ensure_gpu_memory(required_gb, "GGUF creation (16-bit merge)"):
            raise ConversionError(
                f"Insufficient GPU memory for GGUF creation. "
                f"Need ~{required_gb:.0f} GB to merge model to 16-bit."
            )

        print("\n" + "=" * 60)
        print("CREATING GGUF VERSIONS")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Quantizations: {', '.join(quantizations)}")
        print()

        # Create gguf subdirectory
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # Use WSL native temp directory for performance
        temp_base = get_native_temp_dir()
        temp_work_dir = Path(tempfile.mkdtemp(prefix='gguf_work_', dir=str(temp_base)))

        try:
            # Step 1: Save merged model
            merged_dir = self._save_merged_model(model_path, temp_work_dir)

            # Step 2: Setup llama.cpp
            llamacpp_dir = self._setup_llamacpp(temp_work_dir)

            # Step 3: Convert to base GGUF
            base_gguf = self._convert_to_gguf(
                merged_dir, gguf_dir, model_name, llamacpp_dir
            )

            # Step 4: Create quantized versions
            gguf_files = self._create_quantizations(
                base_gguf, gguf_dir, model_name, quantizations, llamacpp_dir
            )

            print(f"\n✓ All GGUF files created: {len(gguf_files)} total")
            print(f"Saved to: {gguf_dir}")

            return gguf_files

        finally:
            if cleanup and temp_work_dir.exists():
                print("\nCleaning up temporary files...")
                cleanup_temp_directory(temp_work_dir)

    def _save_merged_model(self, model_path: ModelPath, work_dir: Path) -> Path:
        """Save merged 16-bit model for GGUF conversion."""
        if self.model_loader is None:
            raise ConversionError("Model loader required for GGUF conversion")

        merged_dir = work_dir / "merged_model"
        print("[1/4] Saving merged model locally...")

        model, tokenizer = self.model_loader.load_model(
            str(model_path),
            load_in_4bit=False,  # Need full precision for GGUF
        )

        self.model_loader.save_merged(
            model, tokenizer, merged_dir, save_method="merged_16bit"
        )

        print("✓ Merged model saved to temporary location")
        return merged_dir

    def _setup_llamacpp(self, work_dir: Path) -> Path:
        """Clone and build llama.cpp if needed."""
        print("\n[2/4] Setting up llama.cpp...")

        llamacpp_dir = work_dir / "llama.cpp"

        if not llamacpp_dir.exists():
            print("Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llamacpp_dir)],
                check=True,
                capture_output=True
            )

            print("Building llama.cpp...")
            self._build_llamacpp(llamacpp_dir)

        print("✓ llama.cpp ready")
        return llamacpp_dir

    def _build_llamacpp(self, llamacpp_dir: Path):
        """Build llama.cpp for the current platform."""
        if sys.platform == 'win32':
            print("  Using CMake for Windows build...")
            build_dir = llamacpp_dir / "build"
            build_dir.mkdir(exist_ok=True)

            # Configure with CMake
            subprocess.run(
                ["cmake", "..",
                 "-DCMAKE_BUILD_TYPE=Release",
                 "-DLLAMA_CURL=OFF",
                 "-DGGML_CUDA=OFF",
                 "-DGGML_METAL=OFF"],
                cwd=str(build_dir),
                check=True,
                capture_output=True
            )

            # Build
            subprocess.run(
                ["cmake", "--build", ".", "--config", "Release"],
                cwd=str(build_dir),
                check=True,
                capture_output=True
            )
        else:
            # Use make on Linux/macOS
            subprocess.run(
                ["make", "-j"],
                cwd=str(llamacpp_dir),
                check=True,
                capture_output=True
            )

    def _convert_to_gguf(
        self,
        merged_dir: Path,
        output_dir: Path,
        model_name: str,
        llamacpp_dir: Path
    ) -> Path:
        """Convert merged model to base GGUF format."""
        print("\n[3/4] Converting to GGUF base format (f16)...")

        base_gguf = output_dir / f"{model_name}.gguf"

        subprocess.run([
            "python",
            str(llamacpp_dir / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile", str(base_gguf),
            "--outtype", "f16"
        ], check=True)

        print(f"✓ Base GGUF created: {base_gguf.name}")
        return base_gguf

    def _create_quantizations(
        self,
        base_gguf: Path,
        output_dir: Path,
        model_name: str,
        quantizations: List[str],
        llamacpp_dir: Path
    ) -> List[Path]:
        """Create quantized GGUF versions."""
        print(f"\n[4/4] Creating {len(quantizations)} quantized versions...")

        gguf_files = [base_gguf]

        # Find llama-quantize executable
        if sys.platform == 'win32':
            quantize_exe = llamacpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe"
        else:
            quantize_exe = llamacpp_dir / "llama-quantize"

        for quant in quantizations:
            output_file = output_dir / f"{model_name}-{quant}.gguf"
            print(f"  Creating {quant} quantization...")

            subprocess.run([
                str(quantize_exe),
                str(base_gguf),
                str(output_file),
                quant
            ], check=True)

            gguf_files.append(output_file)
            print(f"  ✓ {output_file.name}")

        return gguf_files
