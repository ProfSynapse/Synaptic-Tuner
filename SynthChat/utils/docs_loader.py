"""Docs Loader - Load markdown/text files as seed data for generation.

Location: SynthChat/utils/docs_loader.py
Purpose: Load document content to use as template variables in scenario prompts
Usage: DocsLoader().load("/path/to/docs") returns List[DocFile]

Template Variables:
    {doc_content} - Full text content of the document
    {doc_path} - File path for reference
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DocFile:
    """Represents a loaded document file."""
    path: str
    content: str


class DocsLoader:
    """
    Load document files for use as seed data in generation.

    Supports: .md, .txt, .html (with tag stripping)
    """

    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.html', '.htm'}
    DEFAULT_MAX_CHARS = 50_000  # Truncate very large docs

    def __init__(self, max_chars: Optional[int] = None):
        """
        Initialize docs loader.

        Args:
            max_chars: Maximum characters per doc (default: 50000)
        """
        self.max_chars = max_chars or self.DEFAULT_MAX_CHARS

    def load(self, path: str) -> List[DocFile]:
        """
        Load document(s) from path.

        Args:
            path: Single file or folder path

        Returns:
            List of DocFile objects
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            doc = self._load_file(path)
            return [doc] if doc else []

        if path.is_dir():
            return self._load_folder(path)

        raise ValueError(f"Invalid path: {path}")

    def _load_folder(self, folder: Path) -> List[DocFile]:
        """Load all supported files from folder."""
        docs = []

        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            doc = self._load_file(file_path)
            if doc:
                docs.append(doc)

        return docs

    def _load_file(self, file_path: Path) -> Optional[DocFile]:
        """
        Load a single file.

        Args:
            file_path: Path to file

        Returns:
            DocFile or None if file is empty/binary
        """
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            print(f"  Skipping unsupported file: {file_path}")
            return None

        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            print(f"  Skipping binary file: {file_path}")
            return None
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            return None

        # Strip HTML tags if HTML file
        if file_path.suffix.lower() in {'.html', '.htm'}:
            content = self._strip_html(content)

        # Skip empty files
        if not content.strip():
            print(f"  Skipping empty file: {file_path}")
            return None

        # Truncate if too large
        if len(content) > self.max_chars:
            print(f"  Truncating {file_path} ({len(content)} -> {self.max_chars} chars)")
            content = content[:self.max_chars] + "\n\n[... truncated ...]"

        return DocFile(
            path=str(file_path),
            content=content
        )

    def _strip_html(self, html: str) -> str:
        """
        Strip HTML tags, keeping text content.

        Args:
            html: HTML string

        Returns:
            Plain text content
        """
        # Remove script and style tags with content
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove all HTML tags
        html = re.sub(r'<[^>]+>', ' ', html)

        # Decode common HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&amp;', '&')
        html = html.replace('&quot;', '"')
        html = html.replace('&#39;', "'")

        # Collapse whitespace
        html = re.sub(r'\s+', ' ', html)

        return html.strip()
