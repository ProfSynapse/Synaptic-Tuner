"""
Inference utilities for KTO-trained models.
"""

import torch
from typing import List, Dict, Optional
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


class KTOInference:
    """Inference wrapper for KTO-trained models."""

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        chat_template: str = "chatml"
    ):
        """
        Initialize inference model.

        Args:
            model_path: Path to saved model or HuggingFace model ID
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load in 4-bit quantization
            chat_template: Chat template to use
        """
        print(f"Loading model from: {model_path}")

        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        # Apply chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template,
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        )

        # Set to inference mode
        FastLanguageModel.for_inference(self.model)

        print(f"✓ Model loaded and ready for inference")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False
    ) -> str:
        """
        Generate response for a single prompt.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stream: Whether to stream output

        Returns:
            Generated text
        """
        # Format as chat messages
        messages = [{"content": prompt, "role": "user"}]

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate
        if stream:
            text_streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
            outputs = self.model.generate(
                input_ids=inputs,
                streamer=text_streamer,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                use_cache=True
            )
        else:
            outputs = self.model.generate(
                input_ids=inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                use_cache=True
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        stream: bool = False
    ) -> str:
        """
        Generate response for a multi-turn conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            stream: Whether to stream output

        Returns:
            Generated response
        """
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate
        if stream:
            text_streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
            outputs = self.model.generate(
                input_ids=inputs,
                streamer=text_streamer,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        else:
            outputs = self.model.generate(
                input_ids=inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        return response


def interactive_chat(model_path: str, chat_template: str = "chatml"):
    """
    Start an interactive chat session.

    Args:
        model_path: Path to model
        chat_template: Chat template to use
    """
    print("=" * 60)
    print("INTERACTIVE CHAT MODE")
    print("=" * 60)
    print("Commands: 'quit' or 'exit' to end, 'clear' to reset conversation")
    print("=" * 60 + "\n")

    # Initialize inference
    inference = KTOInference(model_path, chat_template=chat_template)

    # Conversation history
    messages = []

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        # Handle commands
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = []
            print("✓ Conversation cleared")
            continue

        if not user_input:
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate response
        print("\nAssistant: ", end="", flush=True)
        response = inference.chat(messages, stream=True)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <model_path>")
        print("Example: python inference.py ./kto_output_rtx3090/final_model")
        sys.exit(1)

    model_path = sys.argv[1]
    interactive_chat(model_path)
