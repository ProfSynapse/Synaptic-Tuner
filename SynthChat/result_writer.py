"""SynthChat Result Writer - Streaming output and summary helpers.

Location: SynthChat/result_writer.py
Purpose: Provides StreamingResultWriter for incremental JSONL output during
         generation, plus helpers for output path generation, batch saving,
         and summary printing.

The data file is homogeneous JSONL: every line is one example. Run metadata
(version, timestamps, row count, privacy profile) never enters the data file;
it lives in a sidecar next to it, ``<output>.meta.json`` (see
``metadata_path``). The data file is opened for append so a writer never
truncates rows an earlier writer produced.

Usage: Used by SynthChat.modes.generate (generate_mode), parallel workers and
       the ``synaptic_tuner.api.v1.reference.data`` Data family, which
       reports the sidecar as the ``dataset_metadata`` artifact.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SYNTHCHAT_VERSION = "1.0.0"
METADATA_SUFFIX = ".meta.json"


def metadata_path(output_file: Path) -> Path:
    """The sidecar that holds a dataset's run metadata: ``<output>.meta.json``."""
    output_file = Path(output_file)
    return output_file.with_name(output_file.name + METADATA_SUFFIX)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _privacy_metadata(settings: Dict) -> Optional[Dict[str, Any]]:
    privacy_settings = settings.get("privacy_preprocess") or {}
    if privacy_settings.get("enabled") and privacy_settings.get("profile"):
        return {
            "profile": privacy_settings.get("profile"),
            "apply_to": dict(privacy_settings.get("apply_to") or {}),
        }
    return None


def write_metadata(output_file: Path, metadata: Dict[str, Any]) -> Path:
    """Write ``metadata`` to the sidecar of ``output_file`` and return the sidecar path."""
    sidecar = metadata_path(output_file)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


class StreamingResultWriter:
    """Writes generation results to JSONL incrementally as they complete.

    Purpose: Prevents data loss during long generation runs by streaming each
    result to disk immediately instead of accumulating in memory.
    Thread-safe for use with parallel workers via threading.Lock.

    When ``settings["output"]["include_metadata"]`` is true the sidecar is
    written on enter (``started_at``, ``streaming``) and rewritten on exit with
    ``rows_written`` and ``finished_at``.

    Usage:
        with StreamingResultWriter(output_file, settings) as writer:
            writer.write(result)  # Called after each example completes
        writer.metadata_file     # the sidecar, when metadata is enabled
    """

    def __init__(self, output_file: Path, settings: Dict):
        self._output_file = Path(output_file)
        self._settings = settings
        self._lock = threading.Lock()
        self._file = None
        self._count = 0
        self._started_at: Optional[str] = None

    @property
    def metadata_enabled(self) -> bool:
        return bool(self._settings["output"]["include_metadata"])

    @property
    def metadata_file(self) -> Optional[Path]:
        """The metadata sidecar path, or None when metadata is disabled."""
        return metadata_path(self._output_file) if self.metadata_enabled else None

    def _metadata(self, *, finished: bool) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "synthchat_version": SYNTHCHAT_VERSION,
            "started_at": self._started_at,
            "streaming": True,
            "rows_written": self._count,
        }
        if finished:
            metadata["finished_at"] = _now()
        privacy = _privacy_metadata(self._settings)
        if privacy is not None:
            metadata["privacy_preprocess"] = privacy
        return metadata

    def __enter__(self):
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_file, "a", encoding="utf-8")
        self._started_at = _now()
        if self.metadata_enabled:
            write_metadata(self._output_file, self._metadata(finished=False))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        if self.metadata_enabled:
            write_metadata(self._output_file, self._metadata(finished=True))
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
    """Save generation results to JSONL, with run metadata in the sidecar."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.example) + "\n")

    if settings["output"]["include_metadata"]:
        metadata: Dict[str, Any] = {
            "synthchat_version": SYNTHCHAT_VERSION,
            "generated_at": _now(),
            "streaming": False,
            "rows_written": len(results),
            "stats": {
                "total": len(results),
                "passed": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "avg_iterations": sum(r.iterations for r in results) / len(results) if results else 0,
            },
        }
        privacy = _privacy_metadata(settings)
        if privacy is not None:
            metadata["privacy_preprocess"] = privacy
        write_metadata(output_file, metadata)


def print_summary(results: List, output_file: Path):
    """Print generation summary."""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    avg_iterations = sum(r.iterations for r in results) / total if total else 0
    passed_pct = (passed / total * 100) if total else 0
    failed_pct = (failed / total * 100) if total else 0

    print(f"\n=== Summary ===")
    print(f"Total generated: {total}")
    print(f"Passed: {passed} ({passed_pct:.1f}%)")
    print(f"Failed: {failed} ({failed_pct:.1f}%)")
    print(f"Avg iterations: {avg_iterations:.1f}")
    print(f"Output: {output_file}")
