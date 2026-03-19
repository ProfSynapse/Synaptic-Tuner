"""
Doctor command handler for comprehensive system diagnostics.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/doctor_handler.py
Purpose: Perform system health checks and provide actionable recommendations
Used by: Router when 'doctor' command is invoked from CLI

This handler performs comprehensive diagnostics across:
1. Environment (conda, Python version, working directory)
2. GPU/CUDA (availability, GPU info, torch.cuda works)
3. Dependencies (key packages with versions)
4. Configuration (.env file, key variables)
5. Backends (LM Studio, Ollama reachability)
6. Storage (disk space, required folders)
7. llama.cpp (built or not)

Supports --json flag for structured output and --fix for auto-fixing simple issues.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from shared.utilities.paths import get_trainer_root, iter_training_output_dirs
from tuner.handlers.base import BaseHandler
from tuner.utils.validation import load_env_file


# =============================================================================
# STATUS CONSTANTS
# =============================================================================

STATUS_OK = "ok"           # Check passed
STATUS_WARN = "warn"       # Check passed with warning
STATUS_FAIL = "fail"       # Check failed
STATUS_SKIP = "skip"       # Check skipped (not applicable)


# =============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    status: str  # ok, warn, fail, skip
    message: str
    details: Optional[str] = None
    fix_command: Optional[str] = None
    fix_description: Optional[str] = None


@dataclass
class DiagnosticSection:
    """A section of diagnostic checks."""
    name: str
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, check: CheckResult) -> None:
        """Add a check result to this section."""
        self.checks.append(check)

    def has_failures(self) -> bool:
        """Check if any check failed."""
        return any(c.status == STATUS_FAIL for c in self.checks)

    def has_warnings(self) -> bool:
        """Check if any check has warnings."""
        return any(c.status == STATUS_WARN for c in self.checks)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report with all sections."""
    sections: List[DiagnosticSection] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def add_section(self, section: DiagnosticSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sections": [
                {
                    "name": s.name,
                    "checks": [asdict(c) for c in s.checks]
                }
                for s in self.sections
            ],
            "recommendations": self.recommendations,
            "summary": {
                "total_checks": sum(len(s.checks) for s in self.sections),
                "passed": sum(1 for s in self.sections for c in s.checks if c.status == STATUS_OK),
                "warnings": sum(1 for s in self.sections for c in s.checks if c.status == STATUS_WARN),
                "failures": sum(1 for s in self.sections for c in s.checks if c.status == STATUS_FAIL),
                "skipped": sum(1 for s in self.sections for c in s.checks if c.status == STATUS_SKIP),
            }
        }


# =============================================================================
# DOCTOR HANDLER
# =============================================================================

class DoctorHandler(BaseHandler):
    """
    Handler for system diagnostics and health checks.

    Performs comprehensive checks across the entire system to identify
    issues and provide actionable recommendations.

    Example:
        handler = DoctorHandler()
        exit_code = handler.handle()
        # Runs diagnostics and displays results

        # With JSON output
        handler = DoctorHandler(json_output=True)
        exit_code = handler.handle()

        # With auto-fix
        handler = DoctorHandler(auto_fix=True)
        exit_code = handler.handle()
    """

    def __init__(self, json_output: bool = False, auto_fix: bool = False):
        """
        Initialize the doctor handler.

        Args:
            json_output: If True, output results as JSON
            auto_fix: If True, attempt to auto-fix simple issues
        """
        super().__init__()
        self.json_output = json_output
        self.auto_fix = auto_fix
        self.report = DiagnosticReport()

        # Cache for expensive checks
        self._env_loaded = False
        self._cuda_available = None
        self._gpu_info = None

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "doctor"

    def can_handle_direct_mode(self) -> bool:
        """This handler supports direct CLI invocation."""
        return True

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def handle(self) -> int:
        """
        Execute diagnostic checks and display results.

        Returns:
            Exit code (0 = all checks pass, 1 = warnings, 2 = failures)
        """
        # Load environment variables
        env_path = self.repo_root / ".env"
        if env_path.exists():
            load_env_file(env_path)
            self._env_loaded = True

        if not self.json_output:
            self._print_header()

        # Run all diagnostic sections
        self.report.add_section(self._check_environment())
        self.report.add_section(self._check_gpu_cuda())
        self.report.add_section(self._check_dependencies())
        self.report.add_section(self._check_configuration())
        self.report.add_section(self._check_backends())
        self.report.add_section(self._check_storage())
        self.report.add_section(self._check_llama_cpp())

        # Generate recommendations
        self._generate_recommendations()

        # Auto-fix if requested
        if self.auto_fix:
            self._apply_fixes()

        # Output results
        if self.json_output:
            print(json.dumps(self.report.to_dict(), indent=2))
        else:
            self._print_report()

        # Determine exit code
        has_failures = any(s.has_failures() for s in self.report.sections)
        has_warnings = any(s.has_warnings() for s in self.report.sections)

        if has_failures:
            return 2
        elif has_warnings:
            return 1
        return 0

    # =========================================================================
    # DIAGNOSTIC SECTIONS
    # =========================================================================

    def _check_environment(self) -> DiagnosticSection:
        """Check environment setup (conda, Python, cwd)."""
        section = DiagnosticSection("Environment")

        # Conda environment
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if conda_env:
            section.add(CheckResult(
                name="Conda environment",
                status=STATUS_OK,
                message=f"{conda_env} (active)"
            ))
        else:
            section.add(CheckResult(
                name="Conda environment",
                status=STATUS_WARN,
                message="No conda environment detected",
                details="Consider activating unsloth_latest for training",
                fix_description="Activate conda environment: conda activate unsloth_latest"
            ))

        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 10):
            section.add(CheckResult(
                name="Python version",
                status=STATUS_OK,
                message=py_version
            ))
        else:
            section.add(CheckResult(
                name="Python version",
                status=STATUS_WARN,
                message=f"{py_version} (3.10+ recommended)",
                details="Some features may not work with older Python versions"
            ))

        # Working directory
        cwd = Path.cwd()
        if cwd == self.repo_root or self.repo_root in cwd.parents:
            section.add(CheckResult(
                name="Working directory",
                status=STATUS_OK,
                message=str(cwd)
            ))
        else:
            section.add(CheckResult(
                name="Working directory",
                status=STATUS_WARN,
                message=str(cwd),
                details=f"Expected to be in or under {self.repo_root}"
            ))

        # Platform detection
        from tuner.utils.environment import detect_environment
        platform = detect_environment()
        platform_labels = {"wsl": "WSL2", "linux": "Linux", "windows": "Windows"}
        section.add(CheckResult(
            name="Platform",
            status=STATUS_OK,
            message=platform_labels.get(platform, platform)
        ))

        return section

    def _check_gpu_cuda(self) -> DiagnosticSection:
        """Check GPU and CUDA availability."""
        section = DiagnosticSection("GPU/CUDA")

        # Try to import torch and check CUDA
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            self._cuda_available = cuda_available

            if cuda_available:
                cuda_version = torch.version.cuda or "unknown"
                section.add(CheckResult(
                    name="CUDA available",
                    status=STATUS_OK,
                    message=cuda_version
                ))

                # GPU info
                try:
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        gpu_name = props.name
                        gpu_mem = props.total_memory / (1024**3)  # Convert to GB
                        self._gpu_info = {"name": gpu_name, "memory_gb": gpu_mem}
                        section.add(CheckResult(
                            name=f"GPU detected",
                            status=STATUS_OK,
                            message=f"{gpu_name} ({gpu_mem:.0f}GB)"
                        ))
                except Exception as e:
                    section.add(CheckResult(
                        name="GPU info",
                        status=STATUS_WARN,
                        message=f"Could not get GPU info: {e}"
                    ))

                # Test torch.cuda works
                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    section.add(CheckResult(
                        name="torch.cuda works",
                        status=STATUS_OK,
                        message="Tensor operations functional"
                    ))
                except Exception as e:
                    section.add(CheckResult(
                        name="torch.cuda works",
                        status=STATUS_FAIL,
                        message=f"CUDA test failed: {e}",
                        details="Try restarting Python or reinstalling PyTorch"
                    ))
            else:
                section.add(CheckResult(
                    name="CUDA available",
                    status=STATUS_WARN,
                    message="No CUDA detected",
                    details="Training will use CPU (slow) or MLX (Apple Silicon)"
                ))

        except ImportError:
            section.add(CheckResult(
                name="PyTorch",
                status=STATUS_FAIL,
                message="torch not installed",
                fix_command="pip install torch",
                fix_description="Install PyTorch for GPU training"
            ))
            self._cuda_available = False

        # Check for MLX (Apple Silicon)
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                section.add(CheckResult(
                    name="MLX (Apple Silicon)",
                    status=STATUS_OK,
                    message="Metal backend available"
                ))
        except ImportError:
            pass  # MLX not installed, not an error

        return section

    def _check_dependencies(self) -> DiagnosticSection:
        """Check key package dependencies."""
        section = DiagnosticSection("Dependencies")

        # Define packages to check: (import_name, display_name, required)
        # Some packages (like unsloth) print to stdout during import, so we use
        # subprocess for those to avoid polluting JSON output
        packages = [
            ("unsloth", "unsloth", True, True),     # (import_name, display_name, required, use_subprocess)
            ("transformers", "transformers", True, False),
            ("trl", "trl", True, False),
            ("xformers", "xformers", False, True),  # xformers also prints during import
            ("datasets", "datasets", True, False),
            ("peft", "peft", True, False),
            ("rich", "rich", False, False),
            ("simple_term_menu", "simple-term-menu", False, False),
        ]

        for import_name, display_name, required, use_subprocess in packages:
            if use_subprocess:
                # Use subprocess to avoid stdout pollution from noisy packages
                version, error = self._get_package_version_subprocess(import_name)
                if version:
                    section.add(CheckResult(
                        name=display_name,
                        status=STATUS_OK,
                        message=version
                    ))
                elif error == "not_installed":
                    if required:
                        section.add(CheckResult(
                            name=display_name,
                            status=STATUS_FAIL,
                            message="not installed",
                            fix_command=f"pip install {display_name}",
                            fix_description=f"Install {display_name}"
                        ))
                    else:
                        section.add(CheckResult(
                            name=display_name,
                            status=STATUS_WARN,
                            message="not installed (optional)",
                            fix_command=f"pip install {display_name}",
                            fix_description=f"Install {display_name} for better experience"
                        ))
                else:
                    error_msg = error[:50] + "..." if len(error) > 50 else error
                    section.add(CheckResult(
                        name=display_name,
                        status=STATUS_WARN,
                        message=f"import error: {error_msg}",
                        details="Package is installed but failed to initialize properly"
                    ))
            else:
                # Direct import for well-behaved packages
                try:
                    mod = __import__(import_name)
                    version = getattr(mod, "__version__", "installed")
                    section.add(CheckResult(
                        name=display_name,
                        status=STATUS_OK,
                        message=version
                    ))
                except ImportError:
                    if required:
                        section.add(CheckResult(
                            name=display_name,
                            status=STATUS_FAIL,
                            message="not installed",
                            fix_command=f"pip install {display_name}",
                            fix_description=f"Install {display_name}"
                        ))
                    else:
                        section.add(CheckResult(
                            name=display_name,
                            status=STATUS_WARN,
                            message="not installed (optional)",
                            fix_command=f"pip install {display_name}",
                            fix_description=f"Install {display_name} for better experience"
                        ))
                except Exception as e:
                    error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                    section.add(CheckResult(
                        name=display_name,
                        status=STATUS_WARN,
                        message=f"import error: {error_msg}",
                        details="Package is installed but failed to initialize properly"
                    ))

        # Check for uv (needed for GGUF conversion)
        if shutil.which("uv"):
            section.add(CheckResult(
                name="uv",
                status=STATUS_OK,
                message="installed"
            ))
        else:
            section.add(CheckResult(
                name="uv",
                status=STATUS_WARN,
                message="not installed (needed for GGUF conversion)",
                fix_command="pip install uv",
                fix_description="Install uv for GGUF conversion"
            ))

        return section

    def _check_configuration(self) -> DiagnosticSection:
        """Check configuration files and environment variables."""
        section = DiagnosticSection("Configuration")

        # .env file
        env_path = self.repo_root / ".env"
        if env_path.exists():
            section.add(CheckResult(
                name=".env file",
                status=STATUS_OK,
                message="exists"
            ))
        else:
            section.add(CheckResult(
                name=".env file",
                status=STATUS_WARN,
                message="missing",
                details="Copy .env.example to .env and configure",
                fix_description="Create .env from template: cp .env.example .env"
            ))

        # Key environment variables
        env_vars = [
            ("HF_TOKEN", True, "HuggingFace token for model access"),
            ("OPENROUTER_API_KEY", False, "OpenRouter API for cloud LLM"),
            ("LMSTUDIO_HOST", False, "LM Studio host (for WSL users)"),
            ("OLLAMA_HOST", False, "Ollama host override"),
            ("WANDB_API_KEY", False, "Weights & Biases logging"),
        ]

        for var_name, required, description in env_vars:
            value = os.environ.get(var_name, "")
            if value:
                # Mask sensitive values
                masked = value[:4] + "..." if len(value) > 8 else "configured"
                section.add(CheckResult(
                    name=var_name,
                    status=STATUS_OK,
                    message=masked
                ))
            else:
                if required:
                    section.add(CheckResult(
                        name=var_name,
                        status=STATUS_FAIL,
                        message="missing",
                        details=description,
                        fix_description=f"Add {var_name}=<value> to .env file"
                    ))
                else:
                    section.add(CheckResult(
                        name=var_name,
                        status=STATUS_SKIP,
                        message=f"not set ({description})"
                    ))

        return section

    def _check_backends(self) -> DiagnosticSection:
        """Check backend connectivity (LM Studio, Ollama)."""
        section = DiagnosticSection("Backends")

        # LM Studio
        lmstudio_host = os.environ.get("LMSTUDIO_HOST", "localhost")
        lmstudio_port = int(os.environ.get("LMSTUDIO_PORT", "1234"))
        lmstudio_url = f"http://{lmstudio_host}:{lmstudio_port}"

        if self._check_http_endpoint(f"{lmstudio_url}/v1/models"):
            section.add(CheckResult(
                name="LM Studio",
                status=STATUS_OK,
                message=f"reachable at {lmstudio_host}:{lmstudio_port}"
            ))
        else:
            section.add(CheckResult(
                name="LM Studio",
                status=STATUS_WARN,
                message=f"not reachable at {lmstudio_host}:{lmstudio_port}",
                details="Start LM Studio or update LMSTUDIO_HOST in .env"
            ))

        # Ollama
        ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
        ollama_port = int(os.environ.get("OLLAMA_PORT", "11434"))
        ollama_url = f"http://{ollama_host}:{ollama_port}"

        if self._check_http_endpoint(f"{ollama_url}/api/tags"):
            section.add(CheckResult(
                name="Ollama",
                status=STATUS_OK,
                message=f"reachable at {ollama_host}:{ollama_port}"
            ))
        else:
            section.add(CheckResult(
                name="Ollama",
                status=STATUS_WARN,
                message=f"not reachable at {ollama_host}:{ollama_port}",
                details="Start Ollama or update OLLAMA_HOST in .env"
            ))

        return section

    def _check_storage(self) -> DiagnosticSection:
        """Check disk space and required folders."""
        section = DiagnosticSection("Storage")

        # Disk space
        try:
            usage = shutil.disk_usage(self.repo_root)
            free_gb = usage.free / (1024**3)
            if free_gb >= 50:
                section.add(CheckResult(
                    name="Disk space",
                    status=STATUS_OK,
                    message=f"{free_gb:.0f}GB free"
                ))
            elif free_gb >= 20:
                section.add(CheckResult(
                    name="Disk space",
                    status=STATUS_WARN,
                    message=f"{free_gb:.0f}GB free (may be tight for training)",
                    details="Consider freeing up space for model checkpoints"
                ))
            else:
                section.add(CheckResult(
                    name="Disk space",
                    status=STATUS_FAIL,
                    message=f"{free_gb:.0f}GB free (insufficient)",
                    details="Training requires at least 20GB free space"
                ))
        except Exception as e:
            section.add(CheckResult(
                name="Disk space",
                status=STATUS_WARN,
                message=f"Could not check: {e}"
            ))

        # Required folders
        folders = [
            ("Datasets", "Training datasets"),
            (str(get_trainer_root("sft", self.repo_root).relative_to(self.repo_root)), "SFT trainer"),
            (str(get_trainer_root("kto", self.repo_root).relative_to(self.repo_root)), "KTO trainer"),
            ("Evaluator", "Model evaluator"),
            ("shared", "Shared utilities"),
        ]

        for folder, description in folders:
            folder_path = self.repo_root / folder
            if folder_path.exists():
                # Count files if it's Datasets
                if folder == "Datasets":
                    file_count = len(list(folder_path.rglob("*.jsonl")))
                    section.add(CheckResult(
                        name=f"{description} folder",
                        status=STATUS_OK,
                        message=f"exists ({file_count} JSONL files)"
                    ))
                else:
                    section.add(CheckResult(
                        name=f"{description} folder",
                        status=STATUS_OK,
                        message="exists"
                    ))
            else:
                section.add(CheckResult(
                    name=f"{description} folder",
                    status=STATUS_FAIL,
                    message="missing",
                    details=f"Expected at {folder}"
                ))

        # Training output folders
        for trainer in ["sft", "kto", "grpo"]:
            for output_folder in iter_training_output_dirs(trainer, self.repo_root):
                if output_folder.exists():
                    run_count = len([d for d in output_folder.iterdir() if d.is_dir()])
                    section.add(CheckResult(
                        name=f"{trainer.upper()} output folder",
                        status=STATUS_OK,
                        message=f"{output_folder.relative_to(self.repo_root)} ({run_count} runs)"
                    ))
                    break

        return section

    def _check_llama_cpp(self) -> DiagnosticSection:
        """Check llama.cpp build status."""
        section = DiagnosticSection("llama.cpp")

        llama_cpp_path = self.repo_root / "Trainers" / "llama.cpp"
        llama_cli = llama_cpp_path / "build" / "bin" / "llama-cli"

        if llama_cpp_path.exists():
            if llama_cli.exists():
                section.add(CheckResult(
                    name="llama.cpp",
                    status=STATUS_OK,
                    message=f"built at {llama_cli.relative_to(self.repo_root)}"
                ))
            else:
                section.add(CheckResult(
                    name="llama.cpp",
                    status=STATUS_WARN,
                    message="source exists but not built",
                    details="Build with: cd Trainers/llama.cpp && cmake -B build && cmake --build build"
                ))
        else:
            section.add(CheckResult(
                name="llama.cpp",
                status=STATUS_SKIP,
                message="not installed (optional for GGUF inference)"
            ))

        return section

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_package_version_subprocess(self, package_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get package version using subprocess to avoid stdout pollution.

        Some packages (like unsloth, xformers) print messages during import.
        This method uses subprocess to isolate those prints and extract just
        the version information.

        Args:
            package_name: Name of the package to check

        Returns:
            Tuple of (version, error)
            - version is the version string if successful, None otherwise
            - error is "not_installed" if not installed, error message if import failed, None if successful
        """
        check_script = f"""
import sys
try:
    import {package_name}
    version = getattr({package_name}, "__version__", "installed")
    print(f"VERSION:{{version}}")
except ImportError:
    print("ERROR:not_installed")
except Exception as e:
    print(f"ERROR:{{str(e)}}")
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout.strip()
            # Find the VERSION or ERROR line in output
            for line in output.split("\n"):
                if line.startswith("VERSION:"):
                    return line.split(":", 1)[1], None
                elif line.startswith("ERROR:"):
                    return None, line.split(":", 1)[1]
            # If no VERSION or ERROR found, check stderr
            if result.returncode != 0:
                return None, "not_installed"
            return "installed", None
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:
            return None, str(e)

    def _check_http_endpoint(self, url: str, timeout: float = 2.0) -> bool:
        """
        Check if an HTTP endpoint is reachable.

        Args:
            url: URL to check
            timeout: Timeout in seconds

        Returns:
            True if endpoint responds, False otherwise
        """
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            return False

    def _generate_recommendations(self) -> None:
        """Generate actionable recommendations from check results."""
        for section in self.report.sections:
            for check in section.checks:
                if check.status == STATUS_FAIL and check.fix_description:
                    self.report.add_recommendation(check.fix_description)
                elif check.status == STATUS_WARN and check.fix_description:
                    # Add warnings as lower priority recommendations
                    self.report.add_recommendation(check.fix_description)

    def _apply_fixes(self) -> None:
        """Apply auto-fixes for simple issues."""
        if self.json_output:
            return  # Don't print during JSON output

        fixes_applied = 0
        for section in self.report.sections:
            for check in section.checks:
                if check.status in (STATUS_FAIL, STATUS_WARN) and check.fix_command:
                    # Only auto-fix pip installs
                    if check.fix_command.startswith("pip install"):
                        self._print_fix_attempt(check.name, check.fix_command)
                        try:
                            result = subprocess.run(
                                check.fix_command.split(),
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                            if result.returncode == 0:
                                check.status = STATUS_OK
                                check.message = "installed (auto-fixed)"
                                fixes_applied += 1
                                self._print_fix_success(check.name)
                            else:
                                self._print_fix_failure(check.name, result.stderr)
                        except Exception as e:
                            self._print_fix_failure(check.name, str(e))

        if fixes_applied > 0:
            print(f"\n  Applied {fixes_applied} fix(es). Re-run doctor to verify.\n")

    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================

    def _print_header(self) -> None:
        """Print the doctor command header."""
        try:
            from rich.console import Console
            from rich.text import Text
            console = Console()
            console.print()
            console.print("  [bold cyan]Running diagnostics...[/bold cyan]")
            console.print()
        except ImportError:
            print("\n  Running diagnostics...\n")

    def _print_report(self) -> None:
        """Print the diagnostic report."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box

            console = Console()

            # Status icons
            icons = {
                STATUS_OK: "[green]OK[/green]",
                STATUS_WARN: "[yellow]WARN[/yellow]",
                STATUS_FAIL: "[red]FAIL[/red]",
                STATUS_SKIP: "[dim]-[/dim]",
            }

            # Print each section
            for section in self.report.sections:
                console.print(f"  [bold cyan][{section.name}][/bold cyan]")
                for check in section.checks:
                    icon = icons.get(check.status, "?")
                    console.print(f"  {icon} {check.name}: {check.message}")
                    if check.details and check.status in (STATUS_WARN, STATUS_FAIL):
                        console.print(f"      [dim]{check.details}[/dim]")
                console.print()

            # Print recommendations
            if self.report.recommendations:
                console.print("[bold]" + "=" * 60 + "[/bold]")
                console.print("[bold cyan]  Recommendations:[/bold cyan]")
                for i, rec in enumerate(self.report.recommendations, 1):
                    console.print(f"  {i}. {rec}")
                console.print("[bold]" + "=" * 60 + "[/bold]")
                console.print()

            # Print summary
            summary = self.report.to_dict()["summary"]
            status_line = (
                f"  [green]{summary['passed']} passed[/green], "
                f"[yellow]{summary['warnings']} warnings[/yellow], "
                f"[red]{summary['failures']} failures[/red]"
            )
            console.print(status_line)
            console.print()

        except ImportError:
            # Fallback to plain text
            icons = {
                STATUS_OK: "[OK]",
                STATUS_WARN: "[WARN]",
                STATUS_FAIL: "[FAIL]",
                STATUS_SKIP: "[-]",
            }

            for section in self.report.sections:
                print(f"  [{section.name}]")
                for check in section.checks:
                    icon = icons.get(check.status, "?")
                    print(f"  {icon} {check.name}: {check.message}")
                    if check.details and check.status in (STATUS_WARN, STATUS_FAIL):
                        print(f"      {check.details}")
                print()

            if self.report.recommendations:
                print("=" * 60)
                print("  Recommendations:")
                for i, rec in enumerate(self.report.recommendations, 1):
                    print(f"  {i}. {rec}")
                print("=" * 60)
                print()

    def _print_fix_attempt(self, name: str, command: str) -> None:
        """Print fix attempt message."""
        try:
            from rich.console import Console
            console = Console()
            console.print(f"  [cyan]Fixing {name}:[/cyan] {command}")
        except ImportError:
            print(f"  Fixing {name}: {command}")

    def _print_fix_success(self, name: str) -> None:
        """Print fix success message."""
        try:
            from rich.console import Console
            console = Console()
            console.print(f"    [green]Fixed {name}[/green]")
        except ImportError:
            print(f"    Fixed {name}")

    def _print_fix_failure(self, name: str, error: str) -> None:
        """Print fix failure message."""
        try:
            from rich.console import Console
            console = Console()
            console.print(f"    [red]Failed to fix {name}: {error[:50]}[/red]")
        except ImportError:
            print(f"    Failed to fix {name}: {error[:50]}")
