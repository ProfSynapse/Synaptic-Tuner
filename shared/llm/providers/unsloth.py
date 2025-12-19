"""Unsloth client for direct LoRA model inference.

This module provides an LLM client that loads LoRA adapters directly using
Unsloth/Transformers, without needing a server. The base model is loaded
once and the LoRA adapter is applied on top.

Usage:
    client = create_client(
        provider="unsloth",
        model="/path/to/lora_adapter",  # LoRA adapter directory
    )
    response = client.chat([{"role": "user", "content": "Hello!"}])
"""
from __future__ import annotations

import os
# Disable torch.compile to avoid nvcc permission issues on WSL
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..base import BaseLLMClient
from ..exceptions import LLMError, LLMConnectionError


class UnslothClient(BaseLLMClient):
    """Client for direct inference using Unsloth with LoRA adapters.

    This client loads a base model and applies a LoRA adapter for inference.
    The model is loaded once on initialization and reused for all requests.

    Example:
        client = UnslothClient(
            adapter_path="/path/to/final_model",
            max_seq_length=4096,
            load_in_4bit=True,
        )
        response = client.chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        adapter_path: str,
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
        top_p: float = 0.9,
    ):
        """Initialize the Unsloth client and load the model.

        Args:
            adapter_path: Path to LoRA adapter directory (contains adapter_config.json)
            max_seq_length: Maximum sequence length for the model
            load_in_4bit: Whether to load in 4-bit quantization (saves memory)
            top_p: Top-p sampling parameter
        """
        self._adapter_path = Path(adapter_path)
        self._max_seq_length = max_seq_length
        self._load_in_4bit = load_in_4bit
        self._top_p = top_p

        self._model = None
        self._tokenizer = None
        self._is_vl_model = False
        self._base_model_name = None

        # Load model on init
        self._load_model()

    def _load_model(self) -> None:
        """Load the base model with LoRA adapter."""
        if not self._adapter_path.exists():
            raise LLMConnectionError(f"Adapter path not found: {self._adapter_path}")

        # Read adapter config to get base model
        config_file = self._adapter_path / "adapter_config.json"
        if not config_file.exists():
            raise LLMConnectionError(
                f"adapter_config.json not found in {self._adapter_path}. "
                "Is this a valid LoRA adapter directory?"
            )

        with open(config_file) as f:
            adapter_config = json.load(f)

        self._base_model_name = adapter_config.get("base_model_name_or_path")
        if not self._base_model_name:
            raise LLMConnectionError("base_model_name_or_path not found in adapter_config.json")

        # Check for vision-language models
        auto_mapping = adapter_config.get("auto_mapping", {})
        base_model_class = auto_mapping.get("base_model_class", "")
        self._is_vl_model = (
            "VL" in base_model_class or
            "Vision" in base_model_class or
            "-VL" in self._base_model_name.upper() or
            "Qwen3VL" in base_model_class
        )

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise LLMConnectionError(
                "Unsloth not installed. Install with:\n"
                "  pip install unsloth"
            )

        try:
            print(f"Loading LoRA adapter from: {self._adapter_path}")
            print(f"Base model: {self._base_model_name}")

            # Load base model with LoRA adapter
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self._adapter_path),
                max_seq_length=self._max_seq_length,
                load_in_4bit=self._load_in_4bit,
            )

            # Prepare for inference
            FastLanguageModel.for_inference(self._model)
            print("Model loaded and ready for inference")

        except Exception as e:
            raise LLMConnectionError(f"Failed to load model: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Send a chat conversation to the model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (ignored)

        Returns:
            Generated text response

        Raises:
            LLMError: If inference fails
        """
        if self._model is None or self._tokenizer is None:
            raise LLMError("Model not loaded")

        try:
            # Convert messages to list of dicts
            msg_list = [{"role": m["role"], "content": m["content"]} for m in messages]

            # For VL models, self._tokenizer is a Processor - use inner tokenizer
            actual_tokenizer = getattr(self._tokenizer, 'tokenizer', self._tokenizer)

            # Apply chat template
            prompt = self._tokenizer.apply_chat_template(
                msg_list,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = actual_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_seq_length,
            ).to(self._model.device)

            # Generate
            pad_token_id = getattr(actual_tokenizer, 'pad_token_id', None) or \
                           getattr(actual_tokenizer, 'eos_token_id', None)

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=self._top_p,
                do_sample=temperature > 0,
                pad_token_id=pad_token_id,
                use_cache=True,
            )

            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = actual_tokenizer.decode(response_tokens, skip_special_tokens=True)

            return response_text.strip()

        except Exception as e:
            raise LLMError(f"Inference failed: {e}")

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured JSON output.

        For Unsloth (direct inference), we add JSON instructions to the prompt
        and parse the output. This is similar to how LM Studio handles it.

        Args:
            messages: List of message dicts
            schema: JSON Schema for output format
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Parsed JSON object

        Raises:
            LLMError: If generation or parsing fails
        """
        # Add JSON format instruction to the last user message
        json_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Output ONLY the JSON object, no other text."
        )

        modified_messages = messages.copy()
        if modified_messages and modified_messages[-1]["role"] == "user":
            modified_messages[-1] = {
                "role": "user",
                "content": modified_messages[-1]["content"] + json_instruction
            }
        else:
            modified_messages.append({
                "role": "user",
                "content": json_instruction
            })

        # Generate response
        response = self.chat(
            modified_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            # Handle cases where model might wrap in ```json ... ```
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse JSON from response: {e}\nResponse: {response[:500]}")

    def test_connection(self) -> bool:
        """Test if the model is loaded and ready.

        Returns:
            True if model is loaded and can generate
        """
        if self._model is None or self._tokenizer is None:
            return False

        try:
            # Quick test generation
            self.chat(
                [{"role": "user", "content": "Hi"}],
                temperature=0,
                max_tokens=5
            )
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "unsloth"

    @property
    def model_name(self) -> str:
        """Return the model/adapter path."""
        return str(self._adapter_path)

    def list_models(self) -> list:
        """Return list of loaded models (just the current adapter)."""
        if self._base_model_name:
            return [f"{self._base_model_name} + LoRA: {self._adapter_path.name}"]
        return [str(self._adapter_path)]

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
