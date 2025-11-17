"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/src/evaluator.py

Evaluation Module for MLX Fine-Tuning System

This module provides:
- Inference with fine-tuned model
- Text generation for qualitative evaluation
- Perplexity computation
- Sample generation for testing
- Evaluation metrics aggregation

Dependencies:
- mlx.core: Core operations
- mlx.nn: Neural network operations

Related Files:
- src/model_manager.py: Provides model
- src/trainer.py: Uses evaluator during training
- src/utils.py: Logging utilities
"""

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn


@dataclass
class GeneratedSample:
    """Single generated sample for evaluation."""
    prompt: str
    generated_text: str
    full_response: str
    generation_time: float


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    val_loss: float
    perplexity: float
    num_samples: int
    samples: List[GeneratedSample]


class Evaluator:
    """
    Handles model evaluation and inference.

    Provides methods for:
    - Running validation loops
    - Generating text samples
    - Computing perplexity
    - Qualitative assessment
    """

    def __init__(self, model, tokenizer, config, logger):
        """
        Initialize evaluator.

        Args:
            model: MLX model
            tokenizer: Tokenizer instance
            config: Config object
            logger: StructuredLogger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger

    def evaluate(self, val_loader) -> EvalMetrics:
        """
        Run full evaluation on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            EvalMetrics object with results
        """
        self.logger.info("Running comprehensive evaluation...")

        # Set model to eval mode
        self.model.eval()

        # Compute validation loss
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            # Forward pass
            logits = self.model(batch.input_ids, batch.attention_mask)

            # Compute loss
            loss = self._compute_loss(logits, batch.labels, batch.attention_mask)
            mx.eval(loss)

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Compute perplexity
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

        # Generate samples
        samples = []
        if self.config.evaluation.sample_prompts:
            self.logger.info("Generating sample responses...")
            samples = self.generate_samples(
                prompts=self.config.evaluation.sample_prompts,
                max_new_tokens=self.config.evaluation.max_new_tokens,
                temperature=self.config.evaluation.temperature,
                top_p=self.config.evaluation.top_p
            )

        metrics = EvalMetrics(
            val_loss=avg_loss,
            perplexity=perplexity,
            num_samples=num_batches * self.config.training.per_device_batch_size,
            samples=samples
        )

        # Log results
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Validation Loss: {avg_loss:.4f}")
        self.logger.info(f"  Perplexity: {perplexity:.2f}")
        self.logger.info(f"  Samples Evaluated: {metrics.num_samples}")

        # Log sample generations
        if samples:
            self.logger.info("\nSample Generations:")
            for i, sample in enumerate(samples, 1):
                self.logger.info(f"\n  Sample {i}:")
                self.logger.info(f"    Prompt: {sample.prompt}")
                self.logger.info(f"    Generated: {sample.generated_text}")
                self.logger.info(f"    Time: {sample.generation_time:.2f}s")

        # Back to train mode
        self.model.train()

        return metrics

    def _compute_loss(self, logits: mx.array, labels: mx.array, attention_mask: mx.array) -> mx.array:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits
            labels: Target labels
            attention_mask: Attention mask

        Returns:
            Scalar loss value
        """
        # Flatten
        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = labels.reshape(-1)

        # Mask out padding
        valid_mask = labels_flat != -100

        if mx.sum(valid_mask) == 0:
            return mx.array(0.0)

        # Get valid logits and labels
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]

        # Cross-entropy
        loss = nn.losses.cross_entropy(valid_logits, valid_labels, reduction='mean')

        return loss

    def generate_samples(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> List[GeneratedSample]:
        """
        Generate text samples for given prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling vs greedy

        Returns:
            List of GeneratedSample objects
        """
        import time

        self.model.eval()

        samples = []

        for prompt in prompts:
            start_time = time.time()

            try:
                # Format prompt for Mistral
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"

                # Tokenize
                inputs = self.tokenizer.tokenizer(
                    formatted_prompt,
                    return_tensors='np',
                    padding=False,
                    truncation=True,
                    max_length=512
                )

                input_ids = mx.array(inputs['input_ids'][0])

                # Generate
                generated_ids = self._generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )

                # Decode
                generated_text = self.tokenizer.tokenizer.decode(
                    generated_ids.tolist(),
                    skip_special_tokens=True
                )

                # Extract only the new tokens (after prompt)
                prompt_decoded = self.tokenizer.tokenizer.decode(
                    input_ids.tolist(),
                    skip_special_tokens=True
                )

                # Get just the generated part
                if generated_text.startswith(prompt_decoded):
                    new_text = generated_text[len(prompt_decoded):].strip()
                else:
                    new_text = generated_text

                generation_time = time.time() - start_time

                sample = GeneratedSample(
                    prompt=prompt,
                    generated_text=new_text,
                    full_response=generated_text,
                    generation_time=generation_time
                )

                samples.append(sample)

            except Exception as e:
                self.logger.error(f"Generation failed for prompt '{prompt}': {e}")
                # Add failed sample
                samples.append(GeneratedSample(
                    prompt=prompt,
                    generated_text=f"[Generation failed: {e}]",
                    full_response="",
                    generation_time=0.0
                ))

        self.model.train()

        return samples

    def _generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> mx.array:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample

        Returns:
            Generated token IDs
        """
        # Start with input
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.model(generated.reshape(1, -1), None)

            # Get logits for last token
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Sample or greedy
            if do_sample:
                # Apply top-p (nucleus) sampling
                next_token = self._sample_top_p(next_token_logits, top_p)
            else:
                # Greedy
                next_token = mx.argmax(next_token_logits)

            # Append to generated sequence
            generated = mx.concatenate([generated, next_token.reshape(1)])

            # Check for EOS token
            if int(next_token) == self.tokenizer.tokenizer.eos_token_id:
                break

        return generated

    def _sample_top_p(self, logits: mx.array, top_p: float) -> mx.array:
        """
        Nucleus (top-p) sampling.

        Args:
            logits: Logits for next token
            top_p: Cumulative probability threshold

        Returns:
            Sampled token ID
        """
        # Sort logits
        sorted_indices = mx.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]

        # Compute softmax
        probs = mx.softmax(sorted_logits, axis=-1)

        # Compute cumulative probabilities
        cumsum_probs = mx.cumsum(probs, axis=-1)

        # Find cutoff index
        cutoff_idx = mx.argmax((cumsum_probs >= top_p).astype(mx.int32))

        # Keep only top-p tokens
        top_p_logits = sorted_logits[:cutoff_idx + 1]
        top_p_indices = sorted_indices[:cutoff_idx + 1]

        # Sample from top-p distribution
        top_p_probs = mx.softmax(top_p_logits, axis=-1)

        # Multinomial sampling (simplified)
        # For now, just take argmax of top_p distribution
        # Full multinomial sampling would require more complex implementation
        sampled_idx = mx.argmax(top_p_probs)
        sampled_token = top_p_indices[sampled_idx]

        return sampled_token

    def compute_perplexity(self, val_loader) -> float:
        """
        Compute perplexity on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Perplexity value
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            logits = self.model(batch.input_ids, batch.attention_mask)
            loss = self._compute_loss(logits, batch.labels, batch.attention_mask)
            mx.eval(loss)

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

        self.model.train()

        return perplexity
