"""
Inference handler for running models directly from terminal.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/inference_handler.py
Purpose: Run trained models (GGUF, LoRA, merged) for interactive chat
Used by: Router when handling 'run' command or main menu selection

This handler implements the inference workflow:
1. Discover available models from training runs
2. Let user select a model and format (GGUF, LoRA, merged)
3. Select inference backend (llama.cpp for GGUF, vLLM for others)
4. Start interactive chat session
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from shared.utilities.paths import get_trainer_root, iter_training_output_dirs
from tuner.handlers.base import BaseHandler

# Import shared UI components
from shared.ui import (
    print_header,
    print_menu,
    print_config,
    print_info,
    print_error,
    print_success,
    confirm,
    prompt,
    console,
    RICH_AVAILABLE,
    COLORS,
    BOX,
)


@dataclass
class DiscoveredModel:
    """Represents a discovered model from training output."""
    name: str
    path: Path
    model_type: str  # "gguf", "lora", "merged"
    trainer_type: str  # "sft" or "kto"
    timestamp: str
    size_gb: Optional[float] = None
    base_model: Optional[str] = None
    quantization: Optional[str] = None  # For GGUF: Q4_K_M, Q5_K_M, etc.

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        parts = [self.name]
        if self.quantization:
            parts.append(f"({self.quantization})")
        if self.size_gb:
            parts.append(f"[{self.size_gb:.1f}GB]")
        return " ".join(parts)


class InferenceHandler(BaseHandler):
    """
    Handler for running model inference.

    Discovers models from training outputs and provides interactive
    chat sessions using llama.cpp (GGUF) or vLLM (LoRA/merged).

    Example:
        handler = InferenceHandler()
        exit_code = handler.handle()
        # User selects model, starts chat
        # Returns 0 on success
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "run"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    def _get_trainers_dir(self) -> Path:
        """Get the Trainers directory."""
        return self.repo_root / "Trainers"

    def _get_llama_cpp_path(self) -> Optional[Path]:
        """Find llama.cpp executable."""
        llama_cpp = self._get_trainers_dir() / "llama.cpp" / "llama-cli"
        if llama_cpp.exists():
            return llama_cpp
        # Also check build/bin
        llama_cpp_bin = self._get_trainers_dir() / "llama.cpp" / "build" / "bin" / "llama-cli"
        if llama_cpp_bin.exists():
            return llama_cpp_bin
        return None

    def _discover_gguf_models(self) -> List[DiscoveredModel]:
        """Discover GGUF models from training outputs."""
        models = []

        for trainer_type in ("sft", "kto"):
            for output_dir in iter_training_output_dirs(trainer_type, self.repo_root):
                if not output_dir.exists():
                    continue

                for gguf_file in output_dir.rglob("*.gguf"):
                    if "vocab" in gguf_file.name.lower() or "mmproj" in gguf_file.name.lower():
                        continue

                    try:
                        parts = gguf_file.relative_to(output_dir).parts
                        timestamp = parts[0] if parts else "unknown"
                        model_name = gguf_file.stem

                        quant = None
                        for q in ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_K_S", "Q6_K"]:
                            if q in gguf_file.name:
                                quant = q
                                model_name = model_name.replace(f"-{q}", "")
                                break

                        size_gb = gguf_file.stat().st_size / (1024**3)

                        models.append(DiscoveredModel(
                            name=model_name,
                            path=gguf_file,
                            model_type="gguf",
                            trainer_type=trainer_type,
                            timestamp=timestamp,
                            size_gb=size_gb,
                            quantization=quant,
                        ))
                    except Exception:
                        continue

        # Sort by timestamp (newest first), then by quantization
        models.sort(key=lambda m: (m.timestamp, m.quantization or ""), reverse=True)
        return models

    def _discover_lora_models(self) -> List[DiscoveredModel]:
        """Discover LoRA adapter models from training outputs."""
        models = []

        for trainer_type in ("sft", "kto"):
            for output_dir in iter_training_output_dirs(trainer_type, self.repo_root):
                if not output_dir.exists():
                    continue

                for adapter_config in output_dir.rglob("final_model/adapter_config.json"):
                    try:
                        final_model_dir = adapter_config.parent
                        run_dir = final_model_dir.parent
                        timestamp = run_dir.name

                        with open(adapter_config) as f:
                            config = json.load(f)
                        base_model = config.get("base_model_name_or_path", "unknown")

                        adapter_file = final_model_dir / "adapter_model.safetensors"
                        size_gb = None
                        if adapter_file.exists():
                            size_gb = adapter_file.stat().st_size / (1024**3)

                        models.append(DiscoveredModel(
                            name=f"{timestamp}_{trainer_type}",
                            path=final_model_dir,
                            model_type="lora",
                            trainer_type=trainer_type,
                            timestamp=timestamp,
                            size_gb=size_gb,
                            base_model=base_model,
                        ))
                    except Exception:
                        continue

        models.sort(key=lambda m: m.timestamp, reverse=True)
        return models

    def _display_models_table(self, models: List[DiscoveredModel], title: str) -> None:
        """Display models in a table."""
        if not models:
            print_info("No models found.")
            return

        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box

            table = Table(
                title=title,
                box=rich_box.ROUNDED,
                border_style=COLORS["cello"],
            )
            table.add_column("#", style=COLORS["orange"], width=4, justify="center")
            table.add_column("Name", style="white")
            table.add_column("Type", style=COLORS["aqua"])
            table.add_column("Quant", style=COLORS["purple"])
            table.add_column("Size", style="dim", justify="right")
            table.add_column("Date", style="dim")

            for i, m in enumerate(models, 1):
                quant = m.quantization or "-"
                size = f"{m.size_gb:.1f}GB" if m.size_gb else "-"
                # Format timestamp for display
                date = m.timestamp[:8] if len(m.timestamp) >= 8 else m.timestamp
                date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else date

                table.add_row(
                    str(i),
                    m.name,
                    m.model_type.upper(),
                    quant,
                    size,
                    date,
                )

            console.print()
            console.print(table)
            console.print()
        else:
            print(f"\n{title}:")
            for i, m in enumerate(models, 1):
                quant = f" ({m.quantization})" if m.quantization else ""
                size = f" [{m.size_gb:.1f}GB]" if m.size_gb else ""
                print(f"  [{i}] {m.name}{quant}{size}")
            print()

    def _run_gguf_chat(self, model: DiscoveredModel) -> int:
        """Run interactive chat with GGUF model using llama.cpp."""
        llama_cli = self._get_llama_cpp_path()
        if not llama_cli:
            print_error("llama.cpp not found. Please build it first:")
            print_info("  cd Trainers/llama.cpp && cmake -B build && cmake --build build")
            return 1

        print_header("INTERACTIVE CHAT", f"Model: {model.display_name}")
        print_info("Type your message and press Enter. Type Ctrl+C to exit.")
        print()

        # Build command for conversation mode with chatml
        cmd = [
            str(llama_cli),
            "-m", str(model.path),
            "-c", "4096",  # Context size
            "-n", "1024",  # Max tokens per response
            "--conversation",  # Conversation mode (modern llama.cpp)
            "--chat-template", "chatml",  # Use chatml format
            "--color",  # Colored output
            "-ngl", "99",  # Offload all layers to GPU
        ]

        print_info(f"Starting llama.cpp...")
        print_info(f"Model: {model.path}")
        print_info("GPU layers: all")
        print()

        try:
            # Run interactively
            result = subprocess.run(cmd, cwd=str(self.repo_root))
            return result.returncode
        except KeyboardInterrupt:
            print("\n")
            print_info("Chat ended.")
            return 0

    def _run_vllm_chat(self, model: DiscoveredModel) -> int:
        """Run interactive chat with vLLM server."""
        print_error("vLLM inference not yet implemented.")
        print_info("For now, use llama.cpp with GGUF models.")
        print_info("Or start vLLM server manually:")
        print_info(f"  python -m vllm.entrypoints.openai.api_server --model {model.base_model}")
        return 1

    def _run_direct_chat(self, model: DiscoveredModel) -> int:
        """Run interactive chat with direct transformers/unsloth loading."""
        print_header("INTERACTIVE CHAT", f"Model: {model.display_name}")

        # Use the existing inference.py script
        inference_script = get_trainer_root("sft", self.repo_root) / "src" / "inference.py"

        if not inference_script.exists():
            print_error("inference.py not found.")
            return 1

        python = self.get_conda_python()
        cmd = [python, str(inference_script), str(model.path)]

        print_info(f"Loading model with Unsloth...")
        print_info(f"Path: {model.path}")
        print()

        try:
            result = subprocess.run(cmd, cwd=str(self.repo_root))
            return result.returncode
        except KeyboardInterrupt:
            print("\n")
            print_info("Chat ended.")
            return 0

    def handle(self) -> int:
        """
        Execute inference workflow.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        print_header("RUN MODEL", "Interactive inference with your trained models")

        # Step 1: Select model type
        backend_choice = print_menu([
            ("gguf", f"{BOX['star']} GGUF (llama.cpp - fastest, recommended)"),
            ("lora", f"{BOX['bullet']} LoRA adapters (vLLM or direct)"),
        ], "Select model format:")

        if not backend_choice:
            return 0

        # Step 2: Discover models
        print_info("Scanning for models...")

        if backend_choice == "gguf":
            models = self._discover_gguf_models()
            if not models:
                print_error("No GGUF models found in training outputs.")
                print_info("Train a model first, or convert to GGUF using 'gguf' command.")
                return 1

            # Check llama.cpp is available
            if not self._get_llama_cpp_path():
                print_error("llama.cpp not found!")
                print_info("Build it with:")
                print_info("  cd Trainers/llama.cpp")
                print_info("  cmake -B build -DLLAMA_CUDA=ON")
                print_info("  cmake --build build --config Release")
                return 1

        else:  # lora
            models = self._discover_lora_models()
            if not models:
                print_error("No LoRA models found in training outputs.")
                print_info("Train a model first using 'train' command.")
                return 1

        # Step 3: Display models
        self._display_models_table(models, "Available Models")

        # Step 4: Select model
        while True:
            try:
                sel = prompt(f"Select model (1-{len(models)})", "1")
                idx = int(sel) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        # Step 5: Display config and confirm
        config = {
            "Model": selected.name,
            "Type": selected.model_type.upper(),
            "Path": str(selected.path),
        }
        if selected.quantization:
            config["Quantization"] = selected.quantization
        if selected.base_model:
            config["Base Model"] = selected.base_model
        if selected.size_gb:
            config["Size"] = f"{selected.size_gb:.2f} GB"

        print_config(config, "Model Configuration")

        if not confirm("Start chat?"):
            print_info("Cancelled.")
            return 0

        # Step 6: Run inference
        if selected.model_type == "gguf":
            return self._run_gguf_chat(selected)
        elif selected.model_type == "lora":
            # Ask for backend
            lora_backend = print_menu([
                ("direct", f"{BOX['star']} Direct (Unsloth - simpler)"),
                ("vllm", f"{BOX['bullet']} vLLM (server mode)"),
            ], "Select inference backend:")

            if not lora_backend:
                return 0

            if lora_backend == "direct":
                return self._run_direct_chat(selected)
            else:
                return self._run_vllm_chat(selected)
        else:
            print_error(f"Unknown model type: {selected.model_type}")
            return 1
