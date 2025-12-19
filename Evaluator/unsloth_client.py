"""Unsloth client for direct LoRA model inference.

This module provides a client that loads LoRA adapters directly using
Unsloth/Transformers, without needing a server. The base model is loaded
once and the LoRA adapter is applied on top.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence, Optional, Any

from .config import UnslothSettings
from .protocols import BackendResponse, BackendError


class UnslothClient:
    """Client for direct inference using Unsloth with LoRA adapters.

    This client loads a base model and applies a LoRA adapter for inference.
    The model is loaded once on initialization and reused for all requests.

    Example:
        settings = UnslothSettings(
            model="/path/to/final_model",  # LoRA adapter directory
        )
        client = UnslothClient(settings=settings)
        response = client.chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ])
    """

    def __init__(
        self,
        settings: "UnslothSettings",
        timeout: float = 120.0,
        retries: int = 2,
    ):
        """Initialize the Unsloth client and load the model.

        Args:
            settings: UnslothSettings with adapter path and parameters
            timeout: Not used (kept for interface compatibility)
            retries: Not used (kept for interface compatibility)
        """
        self.settings = settings
        self.timeout = timeout
        self.retries = retries
        self._model = None
        self._tokenizer = None

        # Load model on init
        self._load_model()

    def _load_model(self) -> None:
        """Load the base model with LoRA adapter."""
        adapter_path = Path(self.settings.model)

        if not adapter_path.exists():
            raise BackendError(f"Adapter path not found: {adapter_path}")

        # Read adapter config to get base model
        config_file = adapter_path / "adapter_config.json"
        if not config_file.exists():
            raise BackendError(
                f"adapter_config.json not found in {adapter_path}. "
                "Is this a valid LoRA adapter directory?"
            )

        with open(config_file) as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise BackendError("base_model_name_or_path not found in adapter_config.json")

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise BackendError(
                "Unsloth not installed. Install with:\n"
                "  pip install unsloth"
            )

        try:
            # Load base model with LoRA adapter
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter_path),
                max_seq_length=self.settings.max_seq_length,
                load_in_4bit=self.settings.load_in_4bit,
            )

            # Prepare for inference
            FastLanguageModel.for_inference(self._model)

        except Exception as e:
            raise BackendError(f"Failed to load model: {e}")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send a chat conversation to the model.

        Args:
            messages: Sequence of message dicts with 'role' and 'content' keys

        Returns:
            BackendResponse with the model's response

        Raises:
            BackendError: If inference fails
        """
        if self._model is None or self._tokenizer is None:
            raise BackendError("Model not loaded")

        start_time = time.time()

        try:
            # Format messages using chat template
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.max_seq_length,
            ).to(self._model.device)

            # Generate
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature if self.settings.temperature > 0 else None,
                top_p=self.settings.top_p,
                do_sample=self.settings.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                use_cache=True,
            )

            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

            latency = time.time() - start_time

            return BackendResponse(
                message=response_text.strip(),
                raw={
                    "prompt_tokens": input_length,
                    "completion_tokens": len(response_tokens),
                    "model": self.settings.model,
                },
                latency_s=latency,
            )

        except Exception as e:
            raise BackendError(f"Inference failed: {e}")

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
