"""
Base handler class with common functionality.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/base.py
Purpose: Provide shared functionality for all command handlers
Used by: TrainHandler, UploadHandler, and other concrete handler implementations
"""

from abc import ABC
from pathlib import Path
from tuner.core.interfaces import IHandler
from tuner.utils.conda import get_conda_python


class BaseHandler(IHandler, ABC):
    """
    Abstract base handler providing common functionality.

    This class implements common operations that all handlers need:
    - Repository root path access
    - Conda Python path retrieval
    - Shared utility methods

    Concrete handlers should inherit from this class and implement:
    - name property
    - handle() method
    - can_handle_direct_mode() method

    Example:
        class TrainHandler(BaseHandler):
            @property
            def name(self) -> str:
                return "train"

            def can_handle_direct_mode(self) -> bool:
                return True

            def handle(self) -> int:
                # Implementation here
                pass
    """

    def __init__(self):
        """Initialize the base handler."""
        self._repo_root = None
        self._conda_python = None

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
