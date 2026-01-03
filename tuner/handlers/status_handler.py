"""
Status command handler for system overview.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/status_handler.py
Purpose: Provide a quick overview of system state for AI assistants and users
Used by: Router when 'status' command is invoked, or from main menu

This handler checks and displays:
- Environment information (conda, Python, platform)
- CUDA/GPU availability
- Key dependencies (unsloth, transformers, trl, xformers)
- Service connectivity (LM Studio, Ollama)
- HuggingFace token configuration
- Available datasets and training runs
"""

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tuner.handlers.base import BaseHandler
from tuner.utils.environment import detect_environment
from tuner.utils.conda import UNSLOTH_ENV, get_conda_python

# Import shared UI components
from shared.ui import (
    print_header,
    print_info,
    print_error,
    print_success,
    console,
    RICH_AVAILABLE,
    COLORS,
)


@dataclass
class StatusInfo:
    """Container for all status information."""

    # Environment
    conda_env: str = ""
    conda_active: bool = False
    python_version: str = ""
    platform_type: str = ""
    platform_details: str = ""

    # GPU/CUDA
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_memory: str = ""

    # Dependencies
    dependencies: Dict[str, bool] = field(default_factory=dict)
    missing_dependencies: List[str] = field(default_factory=list)

    # Services
    lmstudio_running: bool = False
    lmstudio_host: str = ""
    lmstudio_port: int = 0
    ollama_running: bool = False
    ollama_host: str = ""

    # Configuration
    hf_token_configured: bool = False
    working_directory: str = ""

    # Resources
    sft_datasets: int = 0
    kto_datasets: int = 0
    behavior_datasets: int = 0
    sft_training_runs: int = 0
    kto_training_runs: int = 0
    grpo_training_runs: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return asdict(self)


class StatusHandler(BaseHandler):
    """
    Handler for status command.

    Provides a quick overview of the system state, useful for:
    - AI assistants to understand the environment
    - Users to verify their setup
    - Debugging configuration issues

    Example:
        handler = StatusHandler()
        exit_code = handler.handle()  # Prints formatted status

        # With JSON output
        handler = StatusHandler(json_output=True)
        exit_code = handler.handle()  # Prints JSON
    """

    def __init__(self, json_output: bool = False):
        """
        Initialize status handler.

        Args:
            json_output: If True, output JSON instead of formatted text
        """
        super().__init__()
        self.json_output = json_output

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "status"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python -m tuner status'."""
        return True

    def _check_conda_env(self) -> Tuple[str, bool]:
        """
        Check conda environment status.

        Returns:
            Tuple of (env_name, is_active)
        """
        # Check if we're in the expected conda environment
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

        if conda_env:
            is_active = UNSLOTH_ENV in conda_prefix or conda_env == UNSLOTH_ENV
            return conda_env, is_active

        # Not in any conda environment
        return "", False

    def _check_cuda(self) -> Tuple[bool, str, str, str]:
        """
        Check CUDA and GPU availability.

        Returns:
            Tuple of (available, cuda_version, gpu_name, gpu_memory)
        """
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda or "unknown"
                gpu_name = torch.cuda.get_device_name(0)

                # Get memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_gb = total_memory / (1024 ** 3)
                memory_str = f"{memory_gb:.1f}GB"

                return True, cuda_version, gpu_name, memory_str
        except ImportError:
            pass
        except Exception:
            pass

        return False, "", "", ""

    def _check_dependencies(self) -> Tuple[Dict[str, bool], List[str]]:
        """
        Check if key dependencies are installed.

        Uses importlib.util.find_spec to check for module availability
        without actually importing them (which can have side effects).

        Returns:
            Tuple of (all_deps_status, missing_deps_list)
        """
        import importlib.util

        deps_to_check = [
            "unsloth",
            "transformers",
            "trl",
            "xformers",
            "torch",
            "peft",
            "datasets",
            "accelerate",
        ]

        status = {}
        missing = []

        for dep in deps_to_check:
            spec = importlib.util.find_spec(dep)
            if spec is not None:
                status[dep] = True
            else:
                status[dep] = False
                missing.append(dep)

        return status, missing

    def _check_lmstudio(self) -> Tuple[bool, str, int]:
        """
        Check if LM Studio is reachable.

        Returns:
            Tuple of (is_running, host, port)
        """
        import urllib.request
        import urllib.error

        # Load from .env or use defaults
        host = os.environ.get("LMSTUDIO_HOST", "localhost")
        port = int(os.environ.get("LMSTUDIO_PORT", "1234"))

        url = f"http://{host}:{port}/v1/models"

        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True, host, port
        except (urllib.error.URLError, Exception):
            pass

        return False, host, port

    def _check_ollama(self) -> Tuple[bool, str]:
        """
        Check if Ollama is reachable.

        Returns:
            Tuple of (is_running, host)
        """
        import urllib.request
        import urllib.error

        # Load from .env or use defaults
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = f"http://{host}"

        port = os.environ.get("OLLAMA_PORT", "11434")
        if ":" not in host.split("//")[-1]:
            host = f"{host}:{port}"

        url = f"{host}/api/tags"

        try:
            req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True, host
        except (urllib.error.URLError, Exception):
            pass

        return False, host

    def _check_hf_token(self) -> bool:
        """Check if HuggingFace token is configured."""
        token = os.environ.get("HF_TOKEN", "")

        # Also check .env file in repo root
        if not token:
            env_file = self.repo_root / ".env"
            if env_file.exists():
                try:
                    content = env_file.read_text()
                    for line in content.splitlines():
                        if line.startswith("HF_TOKEN="):
                            token = line.split("=", 1)[1].strip()
                            break
                except Exception:
                    pass

        return bool(token and token.startswith("hf_"))

    def _count_datasets(self) -> Tuple[int, int, int]:
        """
        Count available datasets by type.

        Returns:
            Tuple of (sft_count, kto_count, behavior_count)
        """
        datasets_dir = self.repo_root / "Datasets"

        sft_count = 0
        kto_count = 0
        behavior_count = 0

        # Count tool datasets (SFT)
        tools_dir = datasets_dir / "tools_datasets"
        if tools_dir.exists():
            sft_count = len(list(tools_dir.rglob("*.jsonl")))

        # Count behavior datasets (KTO)
        behavior_dir = datasets_dir / "behavior_datasets"
        if behavior_dir.exists():
            behavior_count = len(list(behavior_dir.rglob("*.jsonl")))

        # Count GSPO/preference datasets
        gspo_dir = datasets_dir / "gspo_datasets"
        if gspo_dir.exists():
            kto_count = len(list(gspo_dir.rglob("*.jsonl")))

        return sft_count, kto_count, behavior_count

    def _count_training_runs(self) -> Tuple[int, int, int]:
        """
        Count completed training runs by type.

        Returns:
            Tuple of (sft_count, kto_count, grpo_count)
        """
        from tuner.discovery import TrainingRunDiscovery

        discovery = TrainingRunDiscovery(repo_root=self.repo_root)

        sft_runs = discovery.discover("sft", limit=None)
        kto_runs = discovery.discover("kto", limit=None)
        grpo_runs = discovery.discover("grpo", limit=None)

        return len(sft_runs), len(kto_runs), len(grpo_runs)

    def _gather_status(self) -> StatusInfo:
        """
        Gather all status information.

        Returns:
            StatusInfo with all checks completed
        """
        # Load .env file to populate environment variables
        env_file = self.repo_root / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key not in os.environ:
                            os.environ[key] = value.strip()
            except Exception:
                pass

        status = StatusInfo()

        # Environment
        env_name, is_active = self._check_conda_env()
        status.conda_env = env_name or UNSLOTH_ENV
        status.conda_active = is_active
        status.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Platform detection
        env_type = detect_environment()
        if env_type == "wsl":
            status.platform_type = "Linux/WSL2"
            status.platform_details = platform.release()
        elif env_type == "windows":
            status.platform_type = "Windows"
            status.platform_details = platform.release()
        else:
            status.platform_type = "Linux"
            status.platform_details = platform.release()

        # CUDA/GPU
        cuda_available, cuda_version, gpu_name, gpu_memory = self._check_cuda()
        status.cuda_available = cuda_available
        status.cuda_version = cuda_version
        status.gpu_name = gpu_name
        status.gpu_memory = gpu_memory

        # Dependencies
        deps, missing = self._check_dependencies()
        status.dependencies = deps
        status.missing_dependencies = missing

        # Services
        lm_running, lm_host, lm_port = self._check_lmstudio()
        status.lmstudio_running = lm_running
        status.lmstudio_host = lm_host
        status.lmstudio_port = lm_port

        ollama_running, ollama_host = self._check_ollama()
        status.ollama_running = ollama_running
        status.ollama_host = ollama_host

        # Configuration
        status.hf_token_configured = self._check_hf_token()
        status.working_directory = str(self.repo_root)

        # Resources
        sft_ds, kto_ds, behavior_ds = self._count_datasets()
        status.sft_datasets = sft_ds
        status.kto_datasets = kto_ds
        status.behavior_datasets = behavior_ds

        sft_runs, kto_runs, grpo_runs = self._count_training_runs()
        status.sft_training_runs = sft_runs
        status.kto_training_runs = kto_runs
        status.grpo_training_runs = grpo_runs

        return status

    def _format_check(self, passed: bool) -> str:
        """Format a check mark or X for status display."""
        if passed:
            return "[green]OK[/green]" if RICH_AVAILABLE else "OK"
        else:
            return "[red]X[/red]" if RICH_AVAILABLE else "X"

    def _print_formatted_status(self, status: StatusInfo) -> None:
        """Print status in formatted human-readable form."""
        if RICH_AVAILABLE:
            from rich.table import Table
            from rich import box as rich_box
            from rich.panel import Panel

            # Create main status table
            table = Table(
                box=rich_box.ROUNDED,
                border_style=COLORS.get("cello", "blue"),
                show_header=False,
                padding=(0, 1),
            )
            table.add_column("Category", style="bold")
            table.add_column("Status", min_width=60)

            # Environment section
            env_status = f"{status.conda_env} " + (
                "[green](active)[/green]" if status.conda_active else "[yellow](not active)[/yellow]"
            )
            table.add_row("Environment", env_status)
            table.add_row("Python", status.python_version)
            table.add_row("Platform", f"{status.platform_type}")

            # CUDA/GPU section
            if status.cuda_available:
                cuda_info = f"[green]{status.cuda_version}[/green] ({status.gpu_name})"
                table.add_row("CUDA", cuda_info)
            else:
                table.add_row("CUDA", "[yellow]Not available[/yellow]")

            # Dependencies section
            if status.missing_dependencies:
                deps_status = f"[red]Missing: {', '.join(status.missing_dependencies)}[/red]"
            else:
                deps_status = "[green]All installed[/green]"
            table.add_row("Dependencies", deps_status)

            # Services section
            if status.lmstudio_running:
                lm_status = f"[green]Running[/green] at {status.lmstudio_host}:{status.lmstudio_port}"
            else:
                lm_status = f"[dim]Not reachable[/dim] ({status.lmstudio_host}:{status.lmstudio_port})"
            table.add_row("LM Studio", lm_status)

            if status.ollama_running:
                ollama_status = "[green]Running[/green]"
            else:
                ollama_status = "[dim]Not reachable[/dim]"
            table.add_row("Ollama", ollama_status)

            # Configuration section
            if status.hf_token_configured:
                hf_status = "[green]Configured[/green]"
            else:
                hf_status = "[yellow]Missing[/yellow]"
            table.add_row("HF Token", hf_status)

            table.add_row("Working Dir", status.working_directory)

            # Resources section
            datasets_info = f"SFT: {status.sft_datasets}, Behavior: {status.behavior_datasets}, GSPO: {status.kto_datasets}"
            table.add_row("Datasets", datasets_info)

            runs_info = f"SFT: {status.sft_training_runs}, KTO: {status.kto_training_runs}, GRPO: {status.grpo_training_runs}"
            table.add_row("Training Runs", runs_info)

            # Print with panel
            console.print()
            console.print(Panel(
                table,
                title="[bold]System Status[/bold]",
                border_style=COLORS.get("orange", "yellow"),
            ))
            console.print()

        else:
            # Plain text fallback
            print()
            print("=" * 60)
            print("SYSTEM STATUS")
            print("=" * 60)
            print()

            env_active = "(active)" if status.conda_active else "(not active)"
            print(f"Environment: {status.conda_env} {env_active}")
            print(f"Python: {status.python_version}")
            print(f"Platform: {status.platform_type}")

            if status.cuda_available:
                print(f"CUDA: {status.cuda_version} ({status.gpu_name})")
            else:
                print("CUDA: Not available")

            if status.missing_dependencies:
                print(f"Dependencies: Missing - {', '.join(status.missing_dependencies)}")
            else:
                print("Dependencies: All installed")

            lm_running = "Running" if status.lmstudio_running else "Not reachable"
            print(f"LM Studio: {lm_running} at {status.lmstudio_host}:{status.lmstudio_port}")

            ollama_running = "Running" if status.ollama_running else "Not reachable"
            print(f"Ollama: {ollama_running}")

            hf_status = "Configured" if status.hf_token_configured else "Missing"
            print(f"HF Token: {hf_status}")

            print(f"Working Directory: {status.working_directory}")
            print(f"Datasets: SFT: {status.sft_datasets}, Behavior: {status.behavior_datasets}, GSPO: {status.kto_datasets}")
            print(f"Training Runs: SFT: {status.sft_training_runs}, KTO: {status.kto_training_runs}, GRPO: {status.grpo_training_runs}")
            print()

    def _print_json_status(self, status: StatusInfo) -> None:
        """Print status as JSON for machine parsing."""
        output = status.to_dict()
        print(json.dumps(output, indent=2))

    def handle(self) -> int:
        """
        Execute status check workflow.

        Returns:
            int: Exit code (0 = success)
        """
        status = self._gather_status()

        if self.json_output:
            self._print_json_status(status)
        else:
            self._print_formatted_status(status)

        return 0
