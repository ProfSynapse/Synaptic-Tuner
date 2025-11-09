"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/src/utils.py

Utilities Module for MLX Fine-Tuning System

This module provides cross-cutting utilities including:
- Structured logging with console and file handlers
- Memory monitoring for MLX Metal and system memory
- Progress tracking with tqdm
- Device detection and verification
- Helper functions for path management

Dependencies:
- logging for structured logging
- psutil for system memory monitoring
- mlx for Metal GPU memory tracking
- tqdm for progress bars
- pathlib for path operations

Related Files:
- config/config_manager.py: Logging configuration
- All modules use these utilities for logging and monitoring
"""

import logging
import sys
import json
import os
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import mlx.core as mx


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    used_gb: float
    available_gb: float
    total_gb: float
    percent_used: float
    metal_active_gb: Optional[float] = None
    metal_cache_gb: Optional[float] = None


class StructuredLogger:
    """
    Structured logger with console and file output.

    Provides:
    - Console logging with colors and formatting
    - File logging with rotation
    - JSON structured logging for machine parsing
    - Metric logging with timestamps
    - Context management for related log entries
    """

    def __init__(self, name: str, log_dir: str = "logs", level: str = "INFO",
                 enable_console: bool = True, enable_file: bool = True,
                 enable_json: bool = True):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically __name__)
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON structured logs
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if enable_file:
            file_path = self.log_dir / "training.log"
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # JSON handler
        self.json_enabled = enable_json
        if enable_json:
            self.json_log_path = self.log_dir / "training.jsonl"

    def _write_json_log(self, level: str, message: str, **kwargs):
        """Write structured JSON log entry."""
        if not self.json_enabled:
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            **kwargs
        }

        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message)
        self._write_json_log('DEBUG', message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message)
        self._write_json_log('INFO', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message)
        self._write_json_log('WARNING', message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message)
        self._write_json_log('ERROR', message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message)
        self._write_json_log('CRITICAL', message, **kwargs)

    def metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log training metrics.

        Args:
            step: Training step number
            metrics: Dictionary of metric names to values
        """
        metrics_str = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])
        self.info(f"Step {step} | {metrics_str}")
        self._write_json_log('METRICS', 'Training metrics', step=step, metrics=metrics)

    def epoch_summary(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log epoch summary.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names to values
        """
        self.info(f"=" * 80)
        self.info(f"Epoch {epoch} Summary:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
            else:
                self.info(f"  {key}: {value}")
        self.info(f"=" * 80)
        self._write_json_log('EPOCH_SUMMARY', f'Epoch {epoch} complete',
                           epoch=epoch, metrics=metrics)


class MemoryMonitor:
    """
    Monitor system and GPU memory usage.

    Tracks:
    - System RAM usage via psutil
    - Metal GPU memory usage via MLX
    - Peak memory consumption
    - Memory allocation trends
    """

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize memory monitor.

        Args:
            logger: Optional logger for memory warnings
        """
        self.logger = logger
        self.peak_memory_gb = 0.0
        self.peak_metal_gb = 0.0

    def get_current_usage(self) -> MemoryStats:
        """
        Get current memory usage statistics.

        Returns:
            MemoryStats object with current memory information
        """
        # System memory
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)
        total_gb = vm.total / (1024 ** 3)
        percent_used = vm.percent

        # Track peak
        self.peak_memory_gb = max(self.peak_memory_gb, used_gb)

        # Metal memory (if available)
        metal_active_gb = None
        metal_cache_gb = None
        try:
            # MLX memory stats (API may vary by version)
            metal_active = mx.metal.get_active_memory()
            metal_cache = mx.metal.get_cache_memory()
            metal_active_gb = metal_active / (1024 ** 3)
            metal_cache_gb = metal_cache / (1024 ** 3)
            self.peak_metal_gb = max(self.peak_metal_gb, metal_active_gb)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Could not get Metal memory stats: {e}")

        return MemoryStats(
            used_gb=used_gb,
            available_gb=available_gb,
            total_gb=total_gb,
            percent_used=percent_used,
            metal_active_gb=metal_active_gb,
            metal_cache_gb=metal_cache_gb
        )

    def log_memory(self, context: str = ""):
        """
        Log current memory usage.

        Args:
            context: Context string to include in log message
        """
        stats = self.get_current_usage()

        msg_parts = [f"Memory ({context}):"]
        msg_parts.append(f"RAM: {stats.used_gb:.2f}GB / {stats.total_gb:.2f}GB ({stats.percent_used:.1f}%)")

        if stats.metal_active_gb is not None:
            msg_parts.append(f"Metal: {stats.metal_active_gb:.2f}GB active, {stats.metal_cache_gb:.2f}GB cache")

        if self.logger:
            self.logger.info(" | ".join(msg_parts),
                           context=context,
                           ram_used_gb=stats.used_gb,
                           ram_percent=stats.percent_used,
                           metal_active_gb=stats.metal_active_gb)
        else:
            print(" | ".join(msg_parts))

    def check_available(self, required_gb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory available, False otherwise
        """
        stats = self.get_current_usage()
        return stats.available_gb >= required_gb

    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak memory usage.

        Returns:
            Dictionary with peak memory statistics
        """
        return {
            'peak_ram_gb': self.peak_memory_gb,
            'peak_metal_gb': self.peak_metal_gb
        }


def setup_logging(config) -> StructuredLogger:
    """
    Setup logging from configuration.

    Args:
        config: LoggingConfig object

    Returns:
        Configured StructuredLogger instance
    """
    logger = StructuredLogger(
        name='mlx_finetuning',
        log_dir=config.logs_dir,
        level=config.level,
        enable_console=config.console,
        enable_file=config.file,
        enable_json=config.json_logs
    )

    logger.info("=" * 80)
    logger.info("MLX Fine-Tuning System - Logging Initialized")
    logger.info(f"Log level: {config.level}")
    logger.info(f"Log directory: {config.logs_dir}")
    logger.info("=" * 80)

    return logger


def check_metal_availability() -> bool:
    """
    Check if Metal GPU is available.

    Returns:
        True if Metal is available, False otherwise
    """
    try:
        # Try to allocate a small array on Metal
        x = mx.array([1.0])
        mx.eval(x)
        return True
    except Exception:
        return False


def get_device_info() -> Dict[str, Any]:
    """
    Get device information.

    Returns:
        Dictionary with device information
    """
    info = {
        'metal_available': check_metal_availability(),
        'platform': sys.platform,
        'python_version': sys.version,
    }

    # System memory
    vm = psutil.virtual_memory()
    info['total_ram_gb'] = vm.total / (1024 ** 3)

    return info


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 34m 56s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: MLX model

    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0

    # Get all parameters
    params = model.parameters()

    for name, param in params.items():
        num_params = param.size
        total_params += num_params

        # Check if trainable (not frozen)
        # In MLX, we need to check if gradients are enabled
        # For LoRA, only LoRA parameters will be trainable
        if 'lora' in name.lower():
            trainable_params += num_params

    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percent': (trainable_params / total_params * 100) if total_params > 0 else 0
    }


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
