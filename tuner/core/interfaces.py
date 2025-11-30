"""
Core interfaces for the tuner package.

This module defines abstract base classes (ABCs) that establish contracts for:
- Training backends (RTX, Mac)
- Evaluation backends (Ollama, LM Studio)
- Command handlers (Train, Upload, Eval, Pipeline)
- Discovery services (Training runs, Checkpoints, Models, Prompt sets)

These interfaces enable extensibility through polymorphism and dependency inversion.
New backends or handlers can be added by implementing these interfaces and registering
them with the appropriate registry.

Location: /mnt/f/Code/Toolset-Training/tuner/core/interfaces.py
Used by: All backends, handlers, and discovery services
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

from tuner.core.config import TrainingConfig


class ITrainingBackend(ABC):
    """
    Interface for training backend implementations.

    Training backends abstract the platform-specific details of executing training,
    loading configurations, and validating environments. This allows the CLI to
    work uniformly across different hardware platforms (NVIDIA RTX, Apple Silicon).

    Implementations:
    - RTXBackend: NVIDIA GPU training (SFT/KTO via Unsloth)
    - MacBackend: Apple Silicon training (MLX LoRA)

    Example:
        backend = RTXBackend(repo_root=Path("/path/to/repo"))
        methods = backend.get_available_methods()  # ['sft', 'kto']
        config = backend.load_config('sft')
        exit_code = backend.execute(config, python_path="/path/to/python")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Backend identifier for display and logging.

        Returns:
            String identifier (e.g., 'rtx', 'mac')
        """
        pass

    @abstractmethod
    def get_available_methods(self) -> List[str]:
        """
        Get available training methods for this backend.

        Different backends support different training methods:
        - RTX backend: ['sft', 'kto']
        - Mac backend: ['mlx']

        Returns:
            List of method names as strings
        """
        pass

    @abstractmethod
    def load_config(self, method: str) -> TrainingConfig:
        """
        Load configuration for a training method.

        Reads the YAML configuration file for the specified method and extracts
        relevant training parameters into a TrainingConfig object.

        Args:
            method: Training method identifier ('sft', 'kto', 'mlx')

        Returns:
            Parsed training configuration

        Raises:
            ConfigurationError: If config file is invalid or missing required fields
            FileNotFoundError: If config file doesn't exist
        """
        pass

    @abstractmethod
    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Execute training with the given configuration.

        Spawns a subprocess to run the training script (train_sft.py, train_kto.py, etc.)
        using the specified Python interpreter (typically from conda environment).

        Args:
            config: Training configuration with all parameters
            python_path: Path to Python interpreter (conda environment)

        Returns:
            Exit code from training process (0 = success, non-zero = failure)

        Raises:
            BackendError: If training execution fails
        """
        pass

    @abstractmethod
    def validate_environment(self) -> Tuple[bool, str]:
        """
        Validate that backend environment is available and properly configured.

        Checks for required dependencies, hardware availability (CUDA, Metal),
        and other prerequisites before attempting training.

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if environment is ready, False otherwise
            - error_message: Empty string if valid, description of issue otherwise

        Example:
            is_valid, error = backend.validate_environment()
            if not is_valid:
                print(f"Environment validation failed: {error}")
        """
        pass


class IEvaluationBackend(ABC):
    """
    Interface for evaluation backend implementations.

    Evaluation backends abstract the platform-specific details of listing models,
    validating connections, and executing evaluations against local inference servers.

    Implementations:
    - OllamaBackend: Ollama local inference (default port 11434)
    - LMStudioBackend: LM Studio local inference (default port 1234)

    Example:
        backend = OllamaBackend()
        is_connected, error = backend.validate_connection()
        if is_connected:
            models = backend.list_models()
            for model in models:
                print(f"Available model: {model}")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Backend identifier for display and logging.

        Returns:
            String identifier (e.g., 'ollama', 'lmstudio')
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List available models from this backend.

        Queries the backend API or CLI to enumerate all models available for evaluation.
        Returns an empty list if the backend is unreachable or has no models.

        Returns:
            List of model names/identifiers

        Example:
            models = backend.list_models()
            # ['mistral:7b', 'llama2:13b', 'claudesidian-mcp:latest']
        """
        pass

    @abstractmethod
    def validate_connection(self) -> Tuple[bool, str]:
        """
        Validate backend is running and accessible.

        Checks if the backend service is available by attempting to connect
        to its API endpoint or by checking if it responds to CLI commands.

        Returns:
            Tuple of (is_connected, error_message)
            - is_connected: True if backend is accessible, False otherwise
            - error_message: Empty string if connected, description of issue otherwise

        Example:
            is_connected, error = backend.validate_connection()
            if not is_connected:
                print(f"Backend not available: {error}")
                print(f"Start {backend.name} and try again")
        """
        pass

    @property
    @abstractmethod
    def default_host(self) -> str:
        """
        Default host address for this backend.

        Returns:
            Host address (e.g., '127.0.0.1', 'localhost')
        """
        pass

    @property
    @abstractmethod
    def default_port(self) -> int:
        """
        Default port number for this backend.

        Returns:
            Port number (e.g., 11434 for Ollama, 1234 for LM Studio)
        """
        pass


class IHandler(ABC):
    """
    Interface for command handlers.

    Handlers orchestrate workflows for specific commands (train, upload, eval, pipeline).
    They manage user interaction, coordinate backend calls, and display results.

    Implementations:
    - TrainHandler: Training workflow (select platform/method, execute training)
    - UploadHandler: Upload workflow (select run/checkpoint, configure upload)
    - EvalHandler: Evaluation workflow (select backend/model/prompts, run eval)
    - PipelineHandler: Full pipeline orchestration (train -> upload -> eval)
    - MainMenuHandler: Interactive main menu (no direct command)

    Example:
        handler = TrainHandler()
        if handler.can_handle_direct_mode():
            exit_code = handler.handle()
            sys.exit(exit_code)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Handler identifier for routing and logging.

        Returns:
            String identifier (e.g., 'train', 'upload', 'eval', 'pipeline')
        """
        pass

    @abstractmethod
    def handle(self) -> int:
        """
        Execute handler workflow.

        Implements the complete workflow for this handler, including:
        1. Presenting menus and gathering user input
        2. Validating selections
        3. Executing backend operations
        4. Displaying results
        5. Error handling and recovery

        Returns:
            Exit code (0 = success, non-zero = failure)

        Example:
            handler = TrainHandler()
            exit_code = handler.handle()
            # Runs full training workflow, returns when complete
        """
        pass

    @abstractmethod
    def can_handle_direct_mode(self) -> bool:
        """
        Whether this handler supports direct CLI invocation.

        Some handlers can be invoked directly from the command line:
        - python -m tuner train   (direct mode)
        - python -m tuner upload  (direct mode)

        Others only work in interactive mode:
        - MainMenuHandler (presents menu, not invoked directly)

        Returns:
            True if can be invoked as `python -m tuner <command>`
            False if only available in interactive menu

        Example:
            if handler.can_handle_direct_mode():
                # Register as subcommand in argument parser
                parser.add_parser(handler.name)
        """
        pass


class IDiscoveryService(ABC):
    """
    Interface for resource discovery services.

    Discovery services scan the filesystem or query external systems to find
    available resources (training runs, checkpoints, models, prompt sets).

    Implementations:
    - TrainingRunDiscovery: Find training runs in sft_output_rtx3090/kto_output_rtx3090
    - CheckpointDiscovery: Find and analyze checkpoints with metrics
    - ModelDiscovery: List models from evaluation backends
    - PromptSetDiscovery: Find prompt sets in Evaluator/prompts/

    Example:
        discovery = CheckpointDiscovery()
        checkpoints = discovery.discover(run_dir=Path("/path/to/run"))
        for checkpoint in checkpoints:
            print(f"Checkpoint {checkpoint.step}: loss={checkpoint.metrics['loss']}")
    """

    @abstractmethod
    def discover(self, **filters: Any) -> List[Any]:
        """
        Discover resources matching the given filters.

        The filters and return types vary by discovery service:
        - TrainingRunDiscovery: No filters, returns List[Path] of run directories
        - CheckpointDiscovery: Requires run_dir=Path, returns List[CheckpointInfo]
        - ModelDiscovery: Requires backend=str, returns List[str] of model names
        - PromptSetDiscovery: No filters, returns List[Tuple[str, Path, int]] of (name, path, count)

        Args:
            **filters: Service-specific filters for discovery

        Returns:
            List of discovered resources (type varies by service)

        Raises:
            DiscoveryError: If discovery process fails

        Example:
            # Discover training runs
            runs = training_discovery.discover()

            # Discover checkpoints for a specific run
            checkpoints = checkpoint_discovery.discover(run_dir=runs[0])

            # Discover models from a backend
            models = model_discovery.discover(backend='ollama')
        """
        pass
