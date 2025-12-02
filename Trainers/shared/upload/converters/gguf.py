"""
GGUF format converter.

Converts models to GGUF format for use with llama.cpp and Ollama.

Uses Unsloth's save_pretrained_gguf for optimal compatibility with all model types
including Vision-Language models (Qwen-VL, LLaVA, etc.).
"""

from pathlib import Path
from typing import Any, List, Optional

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

    # VL model indicators
    VL_MODEL_INDICATORS = [
        "qwen2-vl", "qwen3-vl", "qwen2_vl", "qwen3_vl",
        "llava", "pixtral", "paligemma", "idefics",
    ]

    def _is_vision_model(self, model_path: ModelPath) -> bool:
        """Check if this is a Vision-Language model based on config."""
        import json
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

        # Create gguf subdirectory
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # Load model if not provided
        if model is None or tokenizer is None:
            if self.model_loader is None:
                raise ConversionError("Model loader required for GGUF conversion")

            print("[1/2] Loading model for GGUF conversion...")
            model, tokenizer = self.model_loader.load_model(
                str(model_path),
                load_in_4bit=False,  # Need full precision for GGUF
            )
            print("✓ Model loaded")
        else:
            print("[1/2] Using pre-loaded model")

        # Use Unsloth's save_pretrained_gguf
        print(f"\n[2/2] Creating GGUF files...")
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

        print(f"\n✓ GGUF files created: {len(gguf_files)} total")
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
