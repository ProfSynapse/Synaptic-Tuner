"""
Unsloth model loader implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import BaseModelLoader
from upload.core.types import ModelPath
from upload.core.exceptions import DependencyError


class UnslothModelLoader(BaseModelLoader):
    """
    Model loader using Unsloth optimizations.

    Unsloth provides 2x faster loading and training with optimized kernels.
    Supports both language models (FastLanguageModel) and vision-language
    models (FastVisionModel).
    """

    # Model classes that require FastVisionModel
    VL_MODEL_CLASSES = [
        "Qwen2VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "LlavaForConditionalGeneration",
        "LlavaNextForConditionalGeneration",
        "Idefics2ForConditionalGeneration",
        "PaliGemmaForConditionalGeneration",
        "Pixtral",
    ]

    @property
    def name(self) -> str:
        return "unsloth"

    def __init__(self, max_seq_length: int = 2048):
        """
        Initialize Unsloth loader.

        Args:
            max_seq_length: Maximum sequence length for the model
        """
        super().__init__(max_seq_length)
        self._FastLanguageModel = None
        self._FastVisionModel = None

    def _get_fast_language_model(self):
        """
        Lazily import FastLanguageModel to avoid import errors when not needed.
        """
        if self._FastLanguageModel is None:
            try:
                from unsloth import FastLanguageModel
                self._FastLanguageModel = FastLanguageModel
            except ImportError as e:
                raise DependencyError(
                    "unsloth",
                    "Install with: pip install unsloth"
                ) from e
        return self._FastLanguageModel

    def _get_fast_vision_model(self):
        """
        Lazily import FastVisionModel for vision-language models.
        """
        if self._FastVisionModel is None:
            try:
                from unsloth import FastVisionModel
                self._FastVisionModel = FastVisionModel
            except ImportError as e:
                raise DependencyError(
                    "unsloth",
                    "FastVisionModel requires unsloth with VL support.\n"
                    "  Your unsloth installation does not include vision model support.\n\n"
                    "  To fix, run ONE of the following:\n"
                    "    pip install --upgrade unsloth unsloth_zoo\n"
                    "  OR re-run setup with --with-vision:\n"
                    "    bash setup.sh --with-vision\n\n"
                    "  Supported VL models: Qwen2-VL, Qwen3-VL, LLaVA, Pixtral, PaliGemma"
                ) from e
        return self._FastVisionModel

    def _is_vision_model(self, model_path: str) -> bool:
        """
        Detect if model is a vision-language model by checking adapter_config.json.

        Args:
            model_path: Path to the model directory

        Returns:
            True if this is a VL model, False otherwise
        """
        path = Path(model_path)

        # Check adapter_config.json for base_model_class
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config, 'r') as f:
                    config = json.load(f)

                # Check auto_mapping for base_model_class
                auto_mapping = config.get("auto_mapping", {})
                base_model_class = auto_mapping.get("base_model_class", "")

                if any(vl_class in base_model_class for vl_class in self.VL_MODEL_CLASSES):
                    return True

                # Also check base_model_name_or_path for common VL model names
                base_name = config.get("base_model_name_or_path", "").lower()
                if any(vl_name in base_name for vl_name in ["qwen2-vl", "qwen3-vl", "llava", "pixtral", "paligemma"]):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        # Check config.json for model_type
        config_json = path / "config.json"
        if config_json.exists():
            try:
                with open(config_json, 'r') as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                if any(vl_type in model_type for vl_type in ["qwen2_vl", "qwen3_vl", "llava", "pixtral"]):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        return False

    def load_model(
        self,
        model_path: ModelPath,
        load_in_4bit: bool = True,
        **config
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer using Unsloth.

        Automatically detects vision-language models and uses FastVisionModel.

        Args:
            model_path: Path to the model or HuggingFace model ID
            load_in_4bit: Whether to load in 4-bit quantization
            **config: Additional configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path_str = str(model_path)

        # Detect if this is a vision-language model
        is_vl = self._is_vision_model(model_path_str)

        if is_vl:
            print("Detected Vision-Language model, using FastVisionModel...")
            FastModel = self._get_fast_vision_model()
        else:
            FastModel = self._get_fast_language_model()

        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path_str,
            max_seq_length=config.get("max_seq_length", self.max_seq_length),
            dtype=config.get("dtype", None),
            load_in_4bit=load_in_4bit,
        )

        return model, tokenizer

    def save_merged(
        self,
        model: Any,
        tokenizer: Any,
        output_path: Path,
        save_method: str
    ) -> None:
        """
        Save merged model using Unsloth's save_pretrained_merged.

        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            output_path: Path to save the model
            save_method: Method for saving ("merged_16bit", "merged_4bit", "lora")
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained_merged(
            str(output_path),
            tokenizer,
            save_method=save_method
        )

    def push_to_hub_merged(
        self,
        model: Any,
        tokenizer: Any,
        repo_id: str,
        token: str,
        save_method: str = "merged_16bit",
        private: bool = False
    ) -> None:
        """
        Push merged model directly to HuggingFace Hub.

        Args:
            model: The model to push
            tokenizer: The tokenizer to push
            repo_id: HuggingFace repository ID
            token: HuggingFace token
            save_method: Save method
            private: Whether to make repository private
        """
        model.push_to_hub_merged(
            repo_id,
            tokenizer,
            save_method=save_method,
            token=token,
            private=private
        )

    def save_gguf(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: Path,
        model_name: str,
        quantizations: list[str] = None
    ) -> list[Path]:
        """
        Save model directly to GGUF format using Unsloth's save_pretrained_gguf.

        This method handles all model types including VL models properly.

        Args:
            model: The model to convert
            tokenizer: The tokenizer
            output_dir: Directory to save GGUF files
            model_name: Base name for output files
            quantizations: List of quantization methods (e.g., ["q4_k_m", "q8_0"])

        Returns:
            List of paths to created GGUF files
        """
        if quantizations is None:
            quantizations = ["q4_k_m", "q5_k_m", "q8_0"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gguf_files = []

        # Always create f16 base first
        print(f"  Creating f16 (full precision) GGUF...")
        f16_path = output_dir / f"{model_name}.gguf"
        model.save_pretrained_gguf(
            str(output_dir),
            tokenizer,
            quantization_method="f16"
        )
        # Unsloth saves as "unsloth.F16.gguf", rename to our naming convention
        unsloth_f16 = output_dir / "unsloth.F16.gguf"
        if unsloth_f16.exists():
            unsloth_f16.rename(f16_path)
        elif (output_dir / f"{model_name}-unsloth.F16.gguf").exists():
            (output_dir / f"{model_name}-unsloth.F16.gguf").rename(f16_path)
        gguf_files.append(f16_path)
        print(f"  ✓ {f16_path.name}")

        # Create quantized versions
        for quant in quantizations:
            print(f"  Creating {quant.upper()} quantization...")
            quant_path = output_dir / f"{model_name}-{quant.upper()}.gguf"
            model.save_pretrained_gguf(
                str(output_dir),
                tokenizer,
                quantization_method=quant
            )
            # Find and rename the output file
            quant_upper = quant.upper().replace("_", "-")
            possible_names = [
                f"unsloth.{quant_upper}.gguf",
                f"unsloth.{quant.upper()}.gguf",
                f"{model_name}-unsloth.{quant_upper}.gguf",
            ]
            for possible in possible_names:
                possible_path = output_dir / possible
                if possible_path.exists():
                    possible_path.rename(quant_path)
                    break

            if quant_path.exists():
                gguf_files.append(quant_path)
                print(f"  ✓ {quant_path.name}")
            else:
                print(f"  ⚠ {quant.upper()} file not found after conversion")

        return gguf_files

    def push_to_hub_gguf(
        self,
        model: Any,
        tokenizer: Any,
        repo_id: str,
        token: str,
        quantizations: list[str] = None,
        private: bool = False
    ) -> None:
        """
        Push GGUF files directly to HuggingFace Hub.

        Args:
            model: The model to convert and push
            tokenizer: The tokenizer
            repo_id: HuggingFace repository ID
            token: HuggingFace token
            quantizations: List of quantization methods
            private: Whether to make repository private
        """
        if quantizations is None:
            quantizations = ["q4_k_m", "q8_0"]

        for quant in quantizations:
            print(f"  Pushing {quant.upper()} to {repo_id}...")
            model.push_to_hub_gguf(
                repo_id,
                tokenizer,
                quantization_method=quant,
                token=token,
                private=private
            )

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Args:
            model: The loaded model

        Returns:
            Dictionary with model information
        """
        info = {
            "max_seq_length": self.max_seq_length,
            "loader": self.name,
        }

        try:
            info["num_parameters"] = sum(p.numel() for p in model.parameters())
            info["trainable_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        except Exception:
            pass

        return info
