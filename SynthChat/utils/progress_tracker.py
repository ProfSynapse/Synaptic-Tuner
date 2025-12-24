"""Progress tracking for labeling sessions."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..core.models import LabelingSession


class ProgressTracker:
    """
    Tracks labeling progress and allows resuming sessions.

    Progress file format (.labeling_progress.json):
    {
        "input_file": "path/to/input.jsonl",
        "output_file": "path/to/output.jsonl",
        "last_line": 42,
        "labeled_lines": [1, 2, 3, ...],
        "skipped_lines": [15, 23, ...],
        "categories_used": {"good": 20, "bad": 15, ...},
        "last_updated": "2025-12-08T10:30:00"
    }
    """

    def __init__(self, progress_file: str = ".labeling_progress.json"):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to progress file (default: .labeling_progress.json)
        """
        self.progress_file = progress_file

    def load_progress(self, input_file: str) -> Optional[Dict]:
        """
        Load progress from file if exists.

        Args:
            input_file: Path to input file being labeled

        Returns:
            Progress dictionary or None if no progress file exists
        """
        if not os.path.exists(self.progress_file):
            return None

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

            # Verify progress is for the same input file
            if progress.get("input_file") != input_file:
                return None

            return progress
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load progress file: {e}")
            return None

    def save_progress(self, session: LabelingSession, labeled_lines: List[int], skipped_lines: List[int]):
        """
        Save current progress to file.

        Args:
            session: Current labeling session
            labeled_lines: List of line numbers that have been labeled
            skipped_lines: List of line numbers that were skipped
        """
        progress = {
            "input_file": session.input_file,
            "output_file": session.output_file,
            "last_line": session.last_line,
            "labeled_lines": sorted(labeled_lines),
            "skipped_lines": sorted(skipped_lines),
            "categories_used": session.categories_used,
            "last_updated": datetime.now().isoformat(),
            "total_examples": session.total_examples,
            "labeled_count": session.labeled_count,
            "skipped_count": session.skipped_count,
            "start_time": session.start_time
        }

        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save progress file: {e}")

    def get_next_unlabeled_line(self, input_file: str, start_line: int = 1) -> int:
        """
        Get next line number that hasn't been labeled.

        Args:
            input_file: Path to input file
            start_line: Starting line number (default: 1)

        Returns:
            Next unlabeled line number
        """
        progress = self.load_progress(input_file)
        if not progress:
            return start_line

        labeled_lines = set(progress.get("labeled_lines", []))
        skipped_lines = set(progress.get("skipped_lines", []))
        processed_lines = labeled_lines | skipped_lines

        if not processed_lines:
            return start_line

        # Return the line after the last processed line
        last_line = progress.get("last_line", start_line - 1)
        return last_line + 1

    def is_line_labeled(self, input_file: str, line_number: int) -> bool:
        """
        Check if a line has already been labeled.

        Args:
            input_file: Path to input file
            line_number: Line number to check

        Returns:
            True if line has been labeled or skipped
        """
        progress = self.load_progress(input_file)
        if not progress:
            return False

        labeled_lines = set(progress.get("labeled_lines", []))
        skipped_lines = set(progress.get("skipped_lines", []))
        processed_lines = labeled_lines | skipped_lines

        return line_number in processed_lines

    def get_labeled_lines(self, input_file: str) -> Set[int]:
        """
        Get set of all labeled line numbers.

        Args:
            input_file: Path to input file

        Returns:
            Set of labeled line numbers
        """
        progress = self.load_progress(input_file)
        if not progress:
            return set()

        return set(progress.get("labeled_lines", []))

    def get_skipped_lines(self, input_file: str) -> Set[int]:
        """
        Get set of all skipped line numbers.

        Args:
            input_file: Path to input file

        Returns:
            Set of skipped line numbers
        """
        progress = self.load_progress(input_file)
        if not progress:
            return set()

        return set(progress.get("skipped_lines", []))

    def get_categories_used(self, input_file: str) -> Dict[str, int]:
        """
        Get category usage statistics.

        Args:
            input_file: Path to input file

        Returns:
            Dictionary mapping category tags to usage counts
        """
        progress = self.load_progress(input_file)
        if not progress:
            return {}

        return progress.get("categories_used", {})

    def clear_progress(self, input_file: str):
        """
        Clear progress for a specific input file.

        Args:
            input_file: Path to input file
        """
        progress = self.load_progress(input_file)
        if progress and progress.get("input_file") == input_file:
            try:
                os.remove(self.progress_file)
            except OSError as e:
                print(f"Warning: Could not remove progress file: {e}")

    def restore_session(self, input_file: str) -> Optional[LabelingSession]:
        """
        Restore a labeling session from progress file.

        Args:
            input_file: Path to input file

        Returns:
            Restored LabelingSession or None if no progress exists
        """
        progress = self.load_progress(input_file)
        if not progress:
            return None

        return LabelingSession(
            input_file=progress["input_file"],
            output_file=progress["output_file"],
            total_examples=progress.get("total_examples", 0),
            labeled_count=progress.get("labeled_count", 0),
            skipped_count=progress.get("skipped_count", 0),
            last_line=progress.get("last_line", 0),
            start_time=progress.get("start_time", ""),
            categories_used=progress.get("categories_used", {})
        )

    def print_progress_summary(self, input_file: str):
        """
        Print a summary of current progress.

        Args:
            input_file: Path to input file
        """
        progress = self.load_progress(input_file)
        if not progress:
            print("No progress found for this file.")
            return

        labeled_count = len(progress.get("labeled_lines", []))
        skipped_count = len(progress.get("skipped_lines", []))
        total_processed = labeled_count + skipped_count
        last_updated = progress.get("last_updated", "Unknown")

        print(f"\n{'='*60}")
        print(f"Progress Summary")
        print(f"{'='*60}")
        print(f"Input file: {progress['input_file']}")
        print(f"Output file: {progress['output_file']}")
        print(f"Last updated: {last_updated}")
        print(f"")
        print(f"Total processed: {total_processed}")
        print(f"  Labeled: {labeled_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"Last line: {progress.get('last_line', 0)}")
        print(f"")

        categories = progress.get("categories_used", {})
        if categories:
            print(f"Categories used:")
            for tag, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tag}: {count}")
        else:
            print("No categories used yet")

        print(f"{'='*60}\n")
