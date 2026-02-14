"""Runtime interface for environment-backed tool execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .fixture_parser import EnvironmentFixture


class EnvironmentRuntime(ABC):
    """Abstract runtime for environment-backed tool operations."""

    @abstractmethod
    def setup(self, fixture: EnvironmentFixture) -> None:
        """Initialize runtime with a workspace fixture."""

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup runtime resources."""

    @abstractmethod
    def mkdir(self, path: str) -> None:
        """Create directory at path."""

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text file."""

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read UTF-8 text file."""

    @abstractmethod
    def list_dir(self, path: str = ".") -> List[str]:
        """List directory contents (names only)."""

    @abstractmethod
    def move(self, path: str, new_path: str, overwrite: bool = False) -> None:
        """Move file/folder from path to new_path."""

    @abstractmethod
    def copy(self, path: str, new_path: str, overwrite: bool = False) -> None:
        """Copy file/folder from path to new_path."""

    @abstractmethod
    def delete(self, path: str, recursive: bool = False) -> None:
        """Delete file/folder."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if file/folder exists."""

    @abstractmethod
    def search(self, query: str, path: str = ".") -> List[str]:
        """Search query in file names/content and return matching paths."""

    @abstractmethod
    def snapshot(self, limit: int = 200) -> Dict[str, Any]:
        """Capture filesystem snapshot for trace/debugging."""

