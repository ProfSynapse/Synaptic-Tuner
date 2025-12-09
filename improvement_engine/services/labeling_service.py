"""Interactive labeling service for manual dataset annotation."""

import os
from datetime import datetime
from typing import List, Optional

from ..core.models import (
    Example,
    LabelingConfig,
    LabelingResult,
    LabelingSession
)
from ..core.exceptions import FileHandlerError, ConfigurationError
from ..services.file_handler import FileHandler
from ..utils.progress_tracker import ProgressTracker
from ..utils.logger import ImproveLogger
from ..utils.yaml_loader import load_yaml_config
from ..utils.backup import BackupManager
from ..ui.interactive_display import InteractiveDisplay


class LabelingService:
    """
    Interactive labeling service for manual dataset annotation.

    Workflow:
    1. Load labeling categories config
    2. Determine line range (with resume support)
    3. For each example:
       a. Display example (based on display config)
       b. Show available categories
       c. Prompt user for label selection
       d. Validate and save label
       e. Update progress tracker
    4. Generate labeling session summary
    """

    def __init__(self, config: LabelingConfig, logger: ImproveLogger):
        """
        Initialize labeling service.

        Args:
            config: Labeling configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.file_handler = FileHandler()

        # Load categories configuration
        self.categories_config = load_yaml_config(config.categories_config)
        if not self.categories_config:
            raise ConfigurationError(f"Failed to load categories config: {config.categories_config}")

        # Initialize components
        progress_file = self.categories_config.get("progress", {}).get("progress_file", ".labeling_progress.json")
        self.progress_tracker = ProgressTracker(progress_file=progress_file)

        # Initialize interactive display
        display_config = self.categories_config.get("display", {})
        ui_config = self.categories_config.get("ui", {})
        self.display = InteractiveDisplay(display_config, ui_config)

        # Tracking
        self.labeled_lines: List[int] = []
        self.skipped_lines: List[int] = []

    def run(self) -> LabelingSession:
        """
        Main entry point for labeling workflow.

        Returns:
            Completed LabelingSession with statistics

        Raises:
            FileHandlerError: If file operations fail
            KeyboardInterrupt: If user quits the session
        """
        # Validate config
        self.config.validate()

        # Check if input file exists
        if not os.path.exists(self.config.input_file):
            raise FileHandlerError(f"Input file not found: {self.config.input_file}")

        # Create backup if configured
        if self.categories_config.get("progress", {}).get("backup_on_start", True):
            self._create_backup()

        # Initialize or restore session
        session = self._initialize_session()

        self.logger.info(f"Starting labeling session for: {self.config.input_file}")
        self.logger.info(f"Output will be written to: {self.config.output_file}")

        if self.config.resume and session.last_line > 0:
            self.logger.info(f"Resuming from line {session.last_line + 1}")

        # Show labeling rubric at start
        self._show_initial_rubric()

        # Determine line range
        start_line, end_line = self._determine_line_range(session)

        # Load existing progress
        self.labeled_lines = list(self.progress_tracker.get_labeled_lines(self.config.input_file))
        self.skipped_lines = list(self.progress_tracker.get_skipped_lines(self.config.input_file))

        try:
            # Process examples
            for line_number in range(start_line, end_line + 1):
                # Skip already processed lines
                if self.progress_tracker.is_line_labeled(self.config.input_file, line_number):
                    continue

                # Read example
                examples = self.file_handler.read_jsonl(
                    self.config.input_file,
                    start_line=line_number,
                    end_line=line_number
                )

                if not examples:
                    self.logger.warning(f"No example found at line {line_number}")
                    continue

                example = examples[0]

                # Label the example
                result = self._label_example(example, line_number)

                # Update session stats
                if result.skipped:
                    session.skipped_count += 1
                    self.skipped_lines.append(line_number)
                else:
                    session.labeled_count += 1
                    self.labeled_lines.append(line_number)

                    # Update category usage
                    for tag in result.tags:
                        session.categories_used[tag] = session.categories_used.get(tag, 0) + 1

                session.last_line = line_number

                # Save progress periodically
                save_frequency = self.categories_config.get("progress", {}).get("save_frequency", 10)
                if (session.labeled_count + session.skipped_count) % save_frequency == 0:
                    self._save_progress(session)

                # Show progress
                if (session.labeled_count + session.skipped_count) % 5 == 0:
                    self.display.show_progress(session)

        except KeyboardInterrupt:
            self.logger.info("\nLabeling session interrupted by user")
            self._save_progress(session)
            raise

        # Save final progress
        self._save_progress(session)

        # Print summary
        self.display.print_summary(session)

        return session

    def _initialize_session(self) -> LabelingSession:
        """Initialize or restore labeling session."""
        # Try to restore existing session
        if self.config.resume:
            restored_session = self.progress_tracker.restore_session(self.config.input_file)
            if restored_session:
                self.logger.info("Restored previous labeling session")
                # Update total examples count
                total_examples = self.file_handler.count_lines(self.config.input_file)
                restored_session.total_examples = total_examples
                return restored_session

        # Create new session
        total_examples = self.file_handler.count_lines(self.config.input_file)

        return LabelingSession(
            input_file=self.config.input_file,
            output_file=self.config.output_file,
            total_examples=total_examples,
            start_time=datetime.now().isoformat()
        )

    def _show_initial_rubric(self):
        """Display the labeling rubric at the start of the session."""
        rubric_instructions = self.categories_config.get("rubric_instructions", "")
        categories = self.categories_config.get("categories", [])

        self.display.display_rubric(rubric_instructions, categories)

        # Prompt to continue
        try:
            input("Press Enter to begin labeling...")
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt()

    def _determine_line_range(self, session: LabelingSession) -> tuple[int, int]:
        """
        Determine the line range to process.

        Args:
            session: Current labeling session

        Returns:
            Tuple of (start_line, end_line)
        """
        # Start line
        if self.config.resume and session.last_line > 0:
            start_line = session.last_line + 1
        else:
            start_line = self.config.start_line

        # End line
        if self.config.end_line:
            end_line = min(self.config.end_line, session.total_examples)
        else:
            end_line = session.total_examples

        return start_line, end_line

    def _label_example(self, example: Example, line_number: int) -> LabelingResult:
        """
        Label a single example interactively.

        Args:
            example: Example to label
            line_number: Line number in dataset

        Returns:
            LabelingResult with selected tags

        Raises:
            KeyboardInterrupt: If user quits
        """
        # Display the example
        self.display.display_example(example, line_number, self.config.input_file)

        # Show available categories
        categories = self.categories_config.get("categories", [])
        self.display.display_categories(categories)

        # Prompt for selection
        labeling_config = self.categories_config.get("labeling", {})
        allow_multiple = labeling_config.get("allow_multiple", True)
        rubric_instructions = self.categories_config.get("rubric_instructions", "")

        selected_tags = self.display.prompt_selection(categories, allow_multiple, rubric_instructions)

        # Handle skip
        if selected_tags is None:
            self.logger.info(f"Skipped line {line_number}")
            return LabelingResult(
                line_number=line_number,
                original=example,
                labeled=example,
                tags=[],
                skipped=True,
                timestamp=datetime.now().isoformat()
            )

        # Apply labels to example
        labeled_example = self._apply_labels(example, selected_tags)

        # Save labeled example (unless dry run)
        if not self.config.dry_run:
            self._save_labeled_example(labeled_example, line_number)

        self.logger.success(f"Labeled line {line_number}: {selected_tags}")

        return LabelingResult(
            line_number=line_number,
            original=example,
            labeled=labeled_example,
            tags=selected_tags,
            skipped=False,
            timestamp=datetime.now().isoformat()
        )

    def _apply_labels(self, example: Example, tags: List[str]) -> Example:
        """
        Add labels to example in configured field.

        Args:
            example: Original example
            tags: Selected tags

        Returns:
            Example with labels added
        """
        # Create a copy of the example
        labeled_example = Example(
            conversations=example.conversations,
            label=example.label,
            behavior=example.behavior,
            pattern=example.pattern,
            quality_labels=tags  # Add the new labels
        )

        return labeled_example

    def _save_labeled_example(self, example: Example, line_number: int):
        """
        Save labeled example to output file.

        Args:
            example: Labeled example
            line_number: Line number to update

        Raises:
            FileHandlerError: If save fails
        """
        # If output file same as input, update in place
        if self.config.output_file == self.config.input_file:
            self.file_handler.update_line(
                self.config.output_file,
                line_number,
                example.to_dict()
            )
        else:
            # Append to output file
            self.file_handler.write_jsonl(
                self.config.output_file,
                [example],
                mode='a'
            )

    def _save_progress(self, session: LabelingSession):
        """Save current progress."""
        self.progress_tracker.save_progress(
            session,
            self.labeled_lines,
            self.skipped_lines
        )
        self.logger.debug("Progress saved")

    def _create_backup(self):
        """Create backup of input file."""
        try:
            backup_manager = BackupManager()
            backup_path = backup_manager.create_backup(self.config.input_file)
            if backup_path:
                self.logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
