"""File handling service for JSONL datasets."""

import json
from pathlib import Path
from typing import List, Iterator, Optional
from ..core.exceptions import FileHandlerError
from ..core.models import Example


class FileHandler:
    """Handles reading and writing JSONL dataset files."""

    @staticmethod
    def read_jsonl(file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> List[dict]:
        """
        Read lines from a JSONL file.

        Args:
            file_path: Path to JSONL file
            start_line: Line to start reading from (1-indexed)
            end_line: Line to stop reading at (1-indexed, inclusive)

        Returns:
            List of dictionaries (parsed JSON objects)

        Raises:
            FileHandlerError: If file cannot be read
        """
        path = Path(file_path)

        if not path.exists():
            raise FileHandlerError(f"File not found: {file_path}")

        try:
            examples = []

            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip lines before start
                    if line_num < start_line:
                        continue

                    # Stop if we've reached end
                    if end_line and line_num > end_line:
                        break

                    # Parse JSON
                    try:
                        example = json.loads(line.strip())
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        raise FileHandlerError(f"Invalid JSON on line {line_num}: {e}")

            return examples

        except Exception as e:
            raise FileHandlerError(f"Failed to read file: {e}")

    @staticmethod
    def count_lines(file_path: str) -> int:
        """
        Count total lines in a file.

        Args:
            file_path: Path to file

        Returns:
            Number of lines
        """
        path = Path(file_path)

        if not path.exists():
            return 0

        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    @staticmethod
    def write_jsonl(file_path: str, examples: List[dict], mode: str = 'w') -> None:
        """
        Write examples to a JSONL file.

        Args:
            file_path: Path to output file
            examples: List of examples to write
            mode: Write mode ('w' = overwrite, 'a' = append)

        Raises:
            FileHandlerError: If file cannot be written
        """
        path = Path(file_path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, mode, encoding='utf-8') as f:
                for example in examples:
                    json_line = json.dumps(example, ensure_ascii=False)
                    f.write(json_line + '\n')

        except Exception as e:
            raise FileHandlerError(f"Failed to write file: {e}")

    @staticmethod
    def update_line(file_path: str, line_number: int, new_content: dict) -> None:
        """
        Update a specific line in a JSONL file.

        Args:
            file_path: Path to file
            line_number: Line to update (1-indexed)
            new_content: New content for the line

        Raises:
            FileHandlerError: If update fails
        """
        path = Path(file_path)

        if not path.exists():
            raise FileHandlerError(f"File not found: {file_path}")

        try:
            # Read all lines
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Update specific line
            if line_number < 1 or line_number > len(lines):
                raise FileHandlerError(f"Line number {line_number} out of range")

            lines[line_number - 1] = json.dumps(new_content, ensure_ascii=False) + '\n'

            # Write back
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        except Exception as e:
            raise FileHandlerError(f"Failed to update line: {e}")

    @staticmethod
    def iterate_jsonl(file_path: str) -> Iterator[tuple[int, dict]]:
        """
        Iterate over JSONL file yielding (line_number, example) tuples.

        Args:
            file_path: Path to JSONL file

        Yields:
            Tuples of (line_number, example_dict)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileHandlerError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    yield line_num, example
                except json.JSONDecodeError:
                    # Skip invalid lines
                    continue
