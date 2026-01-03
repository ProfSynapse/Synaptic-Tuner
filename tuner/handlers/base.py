"""
Base handler class with common functionality.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/base.py
Purpose: Provide shared functionality for all command handlers
Used by: TrainHandler, UploadHandler, and other concrete handler implementations

This module provides:
- Repository root path access
- Conda Python path retrieval
- JSON output support for AI-parseable output via --json flag
"""

import json
import sys
from abc import ABC
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tuner.core.interfaces import IHandler
from tuner.utils.conda import get_conda_python


class BaseHandler(IHandler, ABC):
    """
    Abstract base handler providing common functionality.

    This class implements common operations that all handlers need:
    - Repository root path access
    - Conda Python path retrieval
    - JSON output support for AI agents (--json flag)
    - Shared utility methods

    Concrete handlers should inherit from this class and implement:
    - name property
    - handle() method
    - can_handle_direct_mode() method

    JSON Output:
        When --json flag is passed, handlers should use self.output() for all
        user-facing output. This ensures consistent JSON formatting for AI agents.

    Example:
        class TrainHandler(BaseHandler):
            @property
            def name(self) -> str:
                return "train"

            def can_handle_direct_mode(self) -> bool:
                return True

            def handle(self) -> int:
                # Use self.output() for JSON-compatible output
                data = {"status": "complete", "model": "path/to/model"}
                self.output(data, "Training complete!")
                return 0
    """

    def __init__(self, args: Optional[Namespace] = None):
        """
        Initialize the base handler.

        Args:
            args: Parsed command-line arguments. If None, JSON mode is disabled.
        """
        self._repo_root = None
        self._conda_python = None
        self._args = args
        self._json_mode = getattr(args, 'json', False) if args else False

    @property
    def repo_root(self) -> Path:
        """
        Get the repository root directory.

        Calculated once and cached for performance.

        Returns:
            Path to repository root

        Example:
            >>> handler = TrainHandler()
            >>> root = handler.repo_root
            >>> print(root / "Datasets")
        """
        if self._repo_root is None:
            # Calculate repo root from this file's location
            # /mnt/f/Code/Toolset-Training/tuner/handlers/base.py
            # -> parent.parent.parent = /mnt/f/Code/Toolset-Training
            self._repo_root = Path(__file__).parent.parent.parent.resolve()
        return self._repo_root

    def get_conda_python(self) -> str:
        """
        Get the path to conda Python interpreter.

        Uses the conda utility to find the unsloth_latest environment.
        Cached for performance.

        Returns:
            Path to Python interpreter as string

        Example:
            >>> handler = TrainHandler()
            >>> python = handler.get_conda_python()
            >>> subprocess.run([python, "train_sft.py"])
        """
        if self._conda_python is None:
            self._conda_python = get_conda_python()
        return self._conda_python

    @property
    def json_mode(self) -> bool:
        """
        Check if JSON output mode is enabled.

        Returns:
            True if --json flag was passed, False otherwise
        """
        return self._json_mode

    @property
    def args(self) -> Optional[Namespace]:
        """
        Get the parsed command-line arguments.

        Returns:
            Namespace with parsed args, or None if not provided
        """
        return self._args

    def output(
        self,
        data: Dict[str, Any],
        human_readable: Optional[str] = None,
        success: bool = True
    ) -> None:
        """
        Output data in JSON or human-readable format.

        When --json flag is set, outputs structured JSON to stdout.
        Otherwise, outputs human-readable text.

        Args:
            data: Dictionary of data to output (used for JSON mode)
            human_readable: Optional human-readable string (used when not in JSON mode)
            success: Whether this represents a successful operation (default: True)

        JSON Output Format:
            {
                "success": true/false,
                "data": { ... },
                "timestamp": "ISO8601 timestamp"
            }

        Example:
            >>> self.output(
            ...     {"status": "complete", "model": "path/to/model"},
            ...     "Training completed successfully!"
            ... )
        """
        if self._json_mode:
            output = {
                "success": success,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            print(json.dumps(output, indent=2, default=str))
        elif human_readable:
            print(human_readable)
        else:
            # Default: use rich if available, else plain dict
            try:
                from rich import print as rprint
                rprint(data)
            except ImportError:
                print(data)

    def output_error(
        self,
        message: str,
        code: str = "ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Output an error in JSON or human-readable format.

        When --json flag is set, outputs structured JSON error to stdout.
        Otherwise, outputs error message using standard error formatting.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling (default: "ERROR")
            details: Optional additional error details

        JSON Output Format:
            {
                "success": false,
                "error": {
                    "message": "...",
                    "code": "ERROR_CODE",
                    "details": { ... }
                },
                "timestamp": "ISO8601 timestamp"
            }

        Example:
            >>> self.output_error(
            ...     "Model not found",
            ...     code="MODEL_NOT_FOUND",
            ...     details={"path": "/path/to/model"}
            ... )
        """
        if self._json_mode:
            output = {
                "success": False,
                "error": {
                    "message": message,
                    "code": code,
                },
                "timestamp": datetime.now().isoformat()
            }
            if details:
                output["error"]["details"] = details
            print(json.dumps(output, indent=2, default=str))
        else:
            # Use standard error output
            try:
                from tuner.ui import print_error
                print_error(message)
            except ImportError:
                print(f"Error: {message}", file=sys.stderr)

    def output_info(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Output informational message.

        When --json flag is set, outputs as structured JSON info message.
        Otherwise, outputs using standard info formatting.

        Args:
            message: Human-readable info message
            data: Optional data to include in JSON output

        Example:
            >>> self.output_info("Starting training...", {"stage": "init"})
        """
        if self._json_mode:
            output = {
                "success": True,
                "info": message,
                "timestamp": datetime.now().isoformat()
            }
            if data:
                output["data"] = data
            print(json.dumps(output, indent=2, default=str))
        else:
            try:
                from tuner.ui import print_info
                print_info(message)
            except ImportError:
                print(f"Info: {message}")

    def output_list(
        self,
        items: list,
        item_type: str,
        human_header: Optional[str] = None
    ) -> None:
        """
        Output a list of items (models, runs, scenarios, etc.).

        When --json flag is set, outputs as structured JSON list.
        Otherwise, outputs as formatted table or list.

        Args:
            items: List of items (dicts or strings)
            item_type: Type of items (e.g., "models", "runs", "scenarios")
            human_header: Optional header for human-readable output

        JSON Output Format:
            {
                "success": true,
                "data": {
                    "type": "models",
                    "count": 5,
                    "items": [...]
                },
                "timestamp": "ISO8601 timestamp"
            }

        Example:
            >>> self.output_list(
            ...     [{"name": "model1"}, {"name": "model2"}],
            ...     "models",
            ...     "Available Models:"
            ... )
        """
        if self._json_mode:
            output = {
                "success": True,
                "data": {
                    "type": item_type,
                    "count": len(items),
                    "items": items
                },
                "timestamp": datetime.now().isoformat()
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            if human_header:
                print(human_header)
            for i, item in enumerate(items, 1):
                if isinstance(item, dict):
                    name = item.get("name", item.get("id", str(item)))
                    print(f"  [{i}] {name}")
                else:
                    print(f"  [{i}] {item}")
