"""Logging utilities for improvement engine."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ImproveLogger:
    """Logger for improvement engine."""

    def __init__(self, name: str = "improvement_engine", log_file: Optional[str] = None):
        """
        Initialize logger.

        Args:
            name: Logger name
            log_file: Optional file path for logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def success(self, message: str):
        """Log success message (as info with ✓)."""
        self.logger.info(f"✓ {message}")

    def failure(self, message: str):
        """Log failure message (as warning with ✗)."""
        self.logger.warning(f"✗ {message}")


def get_logger(name: str = "improvement_engine", log_dir: Optional[str] = None) -> ImproveLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        log_dir: Optional directory for log files

    Returns:
        ImproveLogger instance
    """
    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir_path / f"improvement_{timestamp}.log")

    return ImproveLogger(name=name, log_file=log_file)
