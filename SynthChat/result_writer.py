"""SynthChat Result Writer - Streaming output and summary helpers.

Location: SynthChat/result_writer.py
Purpose: Provides StreamingResultWriter for incremental JSONL output during
         generation, plus helpers for output path generation, batch saving,
         and summary printing.
Usage: Used by SynthChat.modes.generate (generate_mode) and parallel workers.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class StreamingResultWriter:
    """Writes generation results to JSONL incrementally as they complete.

    Purpose: Prevents data loss during long generation runs by streaming each
    result to disk immediately instead of accumulating in memory.
    Thread-safe for use with parallel workers via threading.Lock.

    Usage:
        with StreamingResultWriter(output_file, settings) as writer:
            writer.write(result)  # Called after each example completes
    """

    def __init__(self, output_file: Path, settings: Dict):
        self._output_file = output_file
        self._settings = settings
        self._lock = threading.Lock()
        self._file = None
        self._count = 0

    def __enter__(self):
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_file, "w")

        # Write metadata header as placeholder (will be updated at close)
        if self._settings["output"]["include_metadata"]:
            metadata = {
                "_meta": {
                    "synthchat_version": "1.0.0",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "streaming": True
                }
            }
            self._file.write(json.dumps(metadata) + "\n")
            self._file.flush()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        return False

    def write(self, result) -> bool:
        """Write a single GenerationResult to the output file.

        Args:
            result: GenerationResult with .example dict to serialize.

        Returns:
            True if write succeeded, False on I/O error.
        """
        try:
            line = json.dumps(result.example) + "\n"
            with self._lock:
                self._file.write(line)
                self._file.flush()
                self._count += 1
            return True
        except (IOError, OSError) as e:
            print(f"\nError writing result to {self._output_file}: {e}")
            return False

    @property
    def count(self) -> int:
        """Number of results successfully written."""
        return self._count


def generate_output_path(settings: Dict, input_path: Optional[Path] = None) -> Path:
    """Generate output file path with datetime versioning.

    Args:
        settings: Settings configuration.
        input_path: Input file path (for improve mode).

    Returns:
        Output file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if input_path:
        # Improve mode: append timestamp to input name
        stem = input_path.stem
        # Remove existing version if present (e.g., _v1.8)
        if "_v" in stem:
            stem = stem.split("_v")[0]
        return input_path.parent / f"{stem}_{timestamp}.jsonl"
    else:
        # Generate mode: use default directory
        default_dir = Path(settings["output"]["default_dir"])
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"synthchat_{timestamp}.jsonl"


def save_results(results: List, output_file: Path, settings: Dict):
    """Save generation results to JSONL with metadata."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        # Write metadata header if enabled
        if settings["output"]["include_metadata"]:
            metadata = {
                "_meta": {
                    "synthchat_version": "1.0.0",
                    "generated_at": datetime.utcnow().isoformat(),
                    "stats": {
                        "total": len(results),
                        "passed": sum(1 for r in results if r.success),
                        "failed": sum(1 for r in results if not r.success),
                        "avg_iterations": sum(r.iterations for r in results) / len(results) if results else 0
                    }
                }
            }
            f.write(json.dumps(metadata) + "\n")

        # Write examples
        for result in results:
            f.write(json.dumps(result.example) + "\n")


def print_summary(results: List, output_file: Path):
    """Print generation summary."""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    avg_iterations = sum(r.iterations for r in results) / total if total else 0

    print(f"\n=== Summary ===")
    print(f"Total generated: {total}")
    print(f"Passed: {passed} ({passed / total * 100:.1f}%)")
    print(f"Failed: {failed} ({failed / total * 100:.1f}%)")
    print(f"Avg iterations: {avg_iterations:.1f}")
    print(f"Output: {output_file}")
