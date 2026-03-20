"""
shared/flywheel/utils.py

Shared utility functions for the flywheel pipeline.

Used by: cleaner.py, stager.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .catalog import InferenceLogRecord

logger = logging.getLogger(__name__)


def read_log_content(record: InferenceLogRecord) -> dict | None:
    """Read full log content from the source JSONL file.

    The catalog stores only index fields. Full content (messages,
    response) is read from source_file at line_number.

    Args:
        record: InferenceLogRecord with source_file and line_number set

    Returns:
        Parsed JSON dict of the log line, or None if unreadable
    """
    source = Path(record.source_file)
    if not source.exists():
        logger.warning("Source file not found: %s", source)
        return None

    try:
        with open(source, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == record.line_number:
                    return json.loads(line.strip())
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning(
            "Error reading log %s from %s:%d: %s",
            record.log_id, source, record.line_number, exc,
        )
    return None
