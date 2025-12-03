"""
GGUF format converter.

Converts models to GGUF format for use with llama.cpp and Ollama.

Uses Unsloth's save_pretrained_gguf for optimal compatibility with all model types
including Vision-Language models (Qwen-VL, LLaVA, Ministral, etc.).
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .base import BaseConverter
from ..core.types import ModelPath, QuantizationMethod
from ..core.exceptions import ConversionError
from ..platform.gpu_memory import ensure_gpu_memory, GPU_MEMORY_REQUIREMENTS


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
        Validate that required tools are available for GGUF conversion.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if Unsloth is available (required for save_pretrained_gguf)
        try:
            import unsloth
            return (True, "")
        except ImportError:
            return (False, "Unsloth is required for GGUF conversion. Install with: pip install unsloth")

    # VL model indicators - models that use vision components
    VL_MODEL_INDICATORS = [
        "qwen2-vl", "qwen3-vl", "qwen2_vl", "qwen3_vl",
        "llava", "pixtral", "paligemma", "idefics",
        "ministral3", "ministral-3", "ministral_3", "mistral3",
    ]

    def _ensure_llama_cpp(self, working_dir: Path) -> bool:
        """
        Ensure llama.cpp is available and built for GGUF conversion.

        Unsloth's save_pretrained_gguf internally uses llama.cpp.
        This method ensures:
        1. llama.cpp is cloned if not present (to WSL native filesystem for reliability)
        2. llama-quantize is built
        3. Symlinks are created where Unsloth expects them

        Args:
            working_dir: Directory to set up llama.cpp in (used for symlink)

        Returns:
            True if setup successful, False otherwise
        """
        import shutil

        # Use WSL native filesystem for llama.cpp (NTFS has issues with git clone)
        # This is a shared location so we only build once
        home_dir = Path.home()
        llama_cpp_dir = home_dir / "llama.cpp"
        quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

        # Also check working_dir for backwards compatibility
        local_llama_cpp = working_dir / "llama.cpp"
        local_quantize = local_llama_cpp / "llama-quantize"

        # Check if already set up in home directory
        if quantize_bin.exists():
            print(f"  ✓ llama.cpp already built at {llama_cpp_dir}")
            return True

        # Check if local symlink already works
        if local_quantize.exists():
            print(f"  ✓ llama-quantize found at {local_llama_cpp}")
            return True

        # Verify clone is complete (has CMakeLists.txt)
        cmake_file = llama_cpp_dir / "CMakeLists.txt"

        # If directory exists but is incomplete, remove it
        if llama_cpp_dir.exists() and not cmake_file.exists():
            print("  Removing incomplete llama.cpp clone...")
            shutil.rmtree(str(llama_cpp_dir), ignore_errors=True)

        # Clone llama.cpp to home directory (WSL native filesystem)
        if not llama_cpp_dir.exists():
            print(f"  Cloning llama.cpp to {llama_cpp_dir}...")
            try:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"],
                    cwd=str(home_dir),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    print(f"  ⚠ Clone failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("  ⚠ Timeout cloning llama.cpp")
                return False
            except Exception as e:
                print(f"  ⚠ Failed to clone llama.cpp: {e}")
                return False

            # Verify clone succeeded
            if not cmake_file.exists():
                print("  ⚠ Clone incomplete - CMakeLists.txt not found")
                return False

        # Build llama-quantize (CPU-only is fine for quantization)
        build_dir = llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)

        print("  Building llama.cpp (CPU-only for quantization)...")
        try:
            # Configure
            result = subprocess.run(
                ["cmake", "..", "-DGGML_CUDA=OFF"],
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                print(f"  ⚠ cmake failed: {result.stderr}")
                return False

            # Build just llama-quantize
            result = subprocess.run(
                ["make", "-j4", "llama-quantize"],
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                print(f"  ⚠ make failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ⚠ Timeout building llama.cpp")
            return False
        except Exception as e:
            print(f"  ⚠ Build error: {e}")
            return False

        # Create symlink in llama.cpp root where Unsloth expects it
        quantize_symlink = llama_cpp_dir / "llama-quantize"
        if quantize_bin.exists() and not quantize_symlink.exists():
            try:
                os.symlink(str(quantize_bin), str(quantize_symlink))
            except OSError:
                pass  # Not critical, Unsloth can find build/bin/llama-quantize

        if quantize_bin.exists():
            print(f"  ✓ llama.cpp built at {llama_cpp_dir}")
            return True
        else:
            print("  ⚠ llama-quantize binary not found after build")
            return False

    def _fix_vlm_tokenizer(self, model_path: Path) -> bool:
        """
        Fix VLM tokenizer by adding added_tokens_decoder to tokenizer_config.json.

        Some VLMs (like Ministral 3) have tokenizer info in tokenizer.json but
        the GGUF converter expects added_tokens_decoder in tokenizer_config.json.

        Args:
            model_path: Path to the model directory

        Returns:
            True if fix applied or not needed, False on error
        """
        tokenizer_json = model_path / "tokenizer.json"
        tokenizer_config = model_path / "tokenizer_config.json"

        if not tokenizer_json.exists() or not tokenizer_config.exists():
            return True  # Not needed

        # Check if already has added_tokens_decoder
        try:
            with open(tokenizer_config, 'r') as f:
                config = json.load(f)
            if 'added_tokens_decoder' in config:
                return True  # Already present
        except (json.JSONDecodeError, IOError):
            return False

        # Read added_tokens from tokenizer.json
        try:
            with open(tokenizer_json, 'r') as f:
                tokenizer_data = json.load(f)

            added_tokens = tokenizer_data.get('added_tokens', [])
            if not added_tokens:
                return True  # No tokens to add

            # Build added_tokens_decoder
            added_tokens_decoder = {}
            for token in added_tokens:
                token_id = str(token['id'])
                added_tokens_decoder[token_id] = {
                    'content': token['content'],
                    'lstrip': token.get('lstrip', False),
                    'normalized': token.get('normalized', False),
                    'rstrip': token.get('rstrip', False),
                    'single_word': token.get('single_word', False),
                    'special': token.get('special', True)
                }

            # Update tokenizer_config.json
            config['added_tokens_decoder'] = added_tokens_decoder

            with open(tokenizer_config, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"  ✓ Added {len(added_tokens_decoder)} tokens to tokenizer_config.json")
            return True

        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"  ⚠ Failed to fix VLM tokenizer: {e}")
            return False

    def _install_gguf_dependencies(self) -> bool:
        """
        Install Python dependencies needed for GGUF conversion.

        Returns:
            True if successful
        """
        try:
            import gguf
            return True
        except ImportError:
            pass

        print("  Installing GGUF dependencies...")
        try:
            subprocess.run(
                ["pip", "install", "gguf", "protobuf", "sentencepiece", "-q"],
                capture_output=True,
                check=True,
                timeout=120
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("  ⚠ Failed to install GGUF dependencies")
            return False

    def _is_vision_model(self, model_path: ModelPath) -> bool:
        """Check if this is a Vision-Language model based on config."""
        path = Path(model_path)

        # Check adapter_config.json
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config, 'r') as f:
                    config = json.load(f)
                base_name = config.get("base_model_name_or_path", "").lower()
                if any(vl in base_name for vl in self.VL_MODEL_INDICATORS):
                    return True
                # Check auto_mapping
                auto_mapping = config.get("auto_mapping", {})
                base_class = auto_mapping.get("base_model_class", "").lower()
                if "vl" in base_class or "vision" in base_class or "llava" in base_class:
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        # Check config.json
        config_json = path / "config.json"
        if config_json.exists():
            try:
                with open(config_json, 'r') as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                if any(vl in model_type for vl in self.VL_MODEL_INDICATORS):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        return False

    def convert(
        self,
        model_path: ModelPath,
        output_dir: Path,
        quantizations: Optional[List[QuantizationMethod]] = None,
        **options
    ) -> List[Path]:
        """
        Convert model to GGUF format with quantizations.

        Uses Unsloth's save_pretrained_gguf for optimal compatibility with all
        model types including Vision-Language models.

        Args:
            model_path: Path to the source model (LoRA adapters)
            output_dir: Directory to save GGUF files
            quantizations: List of quantization methods (default: Q4_K_M, Q5_K_M, Q8_0)
            **options:
                - model_name: Name for output files (default: from model_path)
                - model: Pre-loaded model (optional, avoids reloading)
                - tokenizer: Pre-loaded tokenizer (optional)
                - model_size: Model size for memory estimation (default: "7b")

        Returns:
            List of paths to created GGUF files
        """
        if quantizations is None:
            quantizations = self.DEFAULT_QUANTIZATIONS

        # Validate quantizations - convert to lowercase for Unsloth
        quant_lower = []
        for quant in quantizations:
            q = quant.lower()
            quant_lower.append(q)

        model_name = options.get("model_name", Path(model_path).name)
        model_size = options.get("model_size", "7b")
        model = options.get("model")
        tokenizer = options.get("tokenizer")

        # Check GPU memory
        required_gb = GPU_MEMORY_REQUIREMENTS.get(f"{model_size}_gguf", 14.0)
        if not ensure_gpu_memory(required_gb, "GGUF creation"):
            raise ConversionError(
                f"Insufficient GPU memory for GGUF creation. "
                f"Need ~{required_gb:.0f} GB."
            )

        is_vl_model = self._is_vision_model(model_path)

        print("\n" + "=" * 60)
        print("CREATING GGUF VERSIONS")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Quantizations: {', '.join(quantizations)}")
        if is_vl_model:
            print(f"Model type: Vision-Language (using Unsloth's VL support)")
        print(f"Method: Unsloth save_pretrained_gguf")
        print()

        # Setup steps - ensure all dependencies and tools are available
        print("[0/3] Setting up GGUF conversion environment...")

        # Install Python dependencies
        if not self._install_gguf_dependencies():
            raise ConversionError("Failed to install GGUF Python dependencies")
        print("  ✓ Python dependencies ready")

        # Setup llama.cpp (Unsloth uses this internally)
        if not self._ensure_llama_cpp(output_dir.parent):
            raise ConversionError(
                "Failed to setup llama.cpp. Unsloth's GGUF conversion requires llama.cpp.\n"
                "Try manually: cd output_dir && git clone https://github.com/ggerganov/llama.cpp.git"
            )
        print("  ✓ llama.cpp ready")

        # Fix VLM tokenizer if needed
        if is_vl_model:
            if not self._fix_vlm_tokenizer(Path(model_path)):
                print("  ⚠ Warning: Could not fix VLM tokenizer, conversion may fail")
            else:
                print("  ✓ VLM tokenizer configuration ready")

        # Create gguf subdirectory
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # Load model if not provided
        if model is None or tokenizer is None:
            if self.model_loader is None:
                raise ConversionError("Model loader required for GGUF conversion")

            print("[1/3] Loading model for GGUF conversion...")
            model, tokenizer = self.model_loader.load_model(
                str(model_path),
                load_in_4bit=False,  # Need full precision for GGUF
            )
            print("✓ Model loaded")
        else:
            print("[1/3] Using pre-loaded model")

        # Use Unsloth's save_pretrained_gguf
        print(f"\n[2/3] Creating GGUF files...")
        gguf_files = []

        # Create f16 base first
        print(f"  Creating f16 (full precision) GGUF...")
        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method="f16"
            )
            # Find and rename the output file
            f16_file = self._find_and_rename_gguf(gguf_dir, "f16", model_name)
            if f16_file:
                gguf_files.append(f16_file)
                print(f"  ✓ {f16_file.name}")
        except Exception as e:
            print(f"  ⚠ f16 creation failed: {e}")

        # Create quantized versions
        for quant in quant_lower:
            print(f"  Creating {quant.upper()} quantization...")
            try:
                model.save_pretrained_gguf(
                    str(gguf_dir),
                    tokenizer,
                    quantization_method=quant
                )
                quant_file = self._find_and_rename_gguf(gguf_dir, quant, model_name)
                if quant_file:
                    gguf_files.append(quant_file)
                    print(f"  ✓ {quant_file.name}")
            except Exception as e:
                print(f"  ⚠ {quant.upper()} creation failed: {e}")

        print(f"\n[3/3] Summary")
        print(f"✓ GGUF files created: {len(gguf_files)} total")
        print(f"Saved to: {gguf_dir}")

        return gguf_files

    def _find_and_rename_gguf(
        self,
        output_dir: Path,
        quant_method: str,
        model_name: str
    ) -> Optional[Path]:
        """
        Find Unsloth's output file and rename to our naming convention.

        Unsloth creates files like 'unsloth.Q4_K_M.gguf', we rename to
        '{model_name}-Q4_K_M.gguf'.
        """
        quant_upper = quant_method.upper().replace("_", "-")
        quant_variants = [
            quant_method.upper(),
            quant_method.upper().replace("_", "-"),
            quant_method.lower(),
            quant_method.lower().replace("_", "-"),
        ]

        # Target filename
        if quant_method.lower() == "f16":
            target_name = f"{model_name}.gguf"
        else:
            target_name = f"{model_name}-{quant_method.upper()}.gguf"
        target_path = output_dir / target_name

        # If target already exists, return it
        if target_path.exists():
            return target_path

        # Look for Unsloth's output files
        for variant in quant_variants:
            possible_names = [
                f"unsloth.{variant}.gguf",
                f"unsloth-{variant}.gguf",
                f"{model_name}.{variant}.gguf",
                f"{model_name}-unsloth.{variant}.gguf",
            ]
            for name in possible_names:
                possible_path = output_dir / name
                if possible_path.exists():
                    possible_path.rename(target_path)
                    return target_path

        # Check for any new .gguf file that might have been created
        gguf_files = list(output_dir.glob("*.gguf"))
        for gf in gguf_files:
            if gf.name != target_name and any(v in gf.name.upper() for v in quant_variants):
                gf.rename(target_path)
                return target_path

        return None
