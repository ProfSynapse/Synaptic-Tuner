"""
Custom exception hierarchy for the tuner package.

This module defines a hierarchy of exceptions that represent different categories
of errors that can occur during tuner operations. Using a custom hierarchy allows:
- Precise error handling at different levels of abstraction
- User-friendly error messages with context
- Clear separation between configuration, backend, discovery, and validation errors

Location: /mnt/f/Code/Toolset-Training/tuner/core/exceptions.py
Used by: All modules for raising and catching errors
"""

from __future__ import annotations


class TunerError(Exception):
    """
    Base exception for all tuner-related errors.

    This is the root of the exception hierarchy. Catching this exception will catch
    all tuner-specific errors, allowing generic error handling at the CLI level.

    Example:
        try:
            handler.handle()
        except TunerError as e:
            print(f"Tuner error: {e}")
            sys.exit(1)
    """
    pass


class ConfigurationError(TunerError):
    """
    Exception raised when configuration loading or parsing fails.

    This exception indicates that a configuration file is missing, malformed,
    or contains invalid values. It is raised by backends when loading YAML configs
    or by handlers when validating user-provided configuration.

    Common causes:
    - Missing config.yaml file
    - Invalid YAML syntax
    - Missing required fields in config
    - Invalid parameter values

    Example:
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")

        if learning_rate <= 0:
            raise ConfigurationError(f"Learning rate must be positive: {learning_rate}")
    """
    pass


class BackendError(TunerError):
    """
    Exception raised when backend execution fails.

    This exception indicates that a backend operation (training, evaluation, upload)
    encountered an error during execution. It is raised by backends when subprocess
    execution fails, validation fails, or external dependencies are unavailable.

    Common causes:
    - Training script returns non-zero exit code
    - CUDA not available for RTX backend
    - Evaluation backend (Ollama/LM Studio) not running
    - Model upload to HuggingFace fails

    Example:
        result = subprocess.run(cmd, cwd=trainer_dir)
        if result.returncode != 0:
            raise BackendError(f"Training failed with exit code {result.returncode}")

        if not torch.cuda.is_available():
            raise BackendError("CUDA not available for RTX backend")
    """
    pass


class DiscoveryError(TunerError):
    """
    Exception raised when resource discovery fails.

    This exception indicates that a discovery service encountered an error while
    scanning for resources (training runs, checkpoints, models, prompt sets).

    Common causes:
    - Directory doesn't exist or is inaccessible
    - Log files are malformed or can't be parsed
    - External API (Ollama/LM Studio) unreachable
    - No resources found matching criteria

    Example:
        if not run_dir.exists():
            raise DiscoveryError(f"Training run directory not found: {run_dir}")

        if not log_files:
            raise DiscoveryError(f"No training logs found in {logs_dir}")

        if not models:
            raise DiscoveryError(f"No models found for backend '{backend_name}'")
    """
    pass


class ValidationError(TunerError):
    """
    Exception raised when input validation fails.

    This exception indicates that user-provided input is invalid or fails validation.
    It is raised by handlers or utilities when validating user selections, file paths,
    or configuration parameters.

    Common causes:
    - Invalid file path provided by user
    - Invalid model name format
    - Invalid repository ID format
    - Out-of-range parameter values

    Example:
        if not model_path.exists():
            raise ValidationError(f"Model path does not exist: {model_path}")

        if not repo_id.count('/') == 1:
            raise ValidationError(f"Invalid repository ID format: {repo_id}")

        if temperature < 0 or temperature > 2:
            raise ValidationError(f"Temperature must be between 0 and 2: {temperature}")
    """
    pass
