"""Main improvement service orchestrating the improvement workflow."""

import json
import time
from typing import List, Optional
from pathlib import Path

from ..core.models import (
    ImprovementConfig,
    ImprovementResult,
    BatchResult,
    Example
)
from ..core.exceptions import ImprovementEngineError
from .llm_service import LLMService
from .validator import Validator
from .file_handler import FileHandler
from ..utils.logger import ImproveLogger
from ..utils.backup import BackupManager


class ImprovementService:
    """Main service for improving datasets."""

    def __init__(self, config: ImprovementConfig, logger: Optional[ImproveLogger] = None):
        """
        Initialize improvement service.

        Args:
            config: Improvement configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or ImproveLogger()
        self.llm_service = LLMService(backend=config.backend, model=config.model)
        self.validator = Validator()
        self.file_handler = FileHandler()
        self.backup_manager = BackupManager(enabled=True)

        # Log LLM provider info
        self.logger.info(f"Using {self.llm_service.provider_name} with model {self.llm_service.model_name}")

        # Validate configuration
        config.validate()

    def run(self) -> List[BatchResult]:
        """
        Run the improvement process.

        Returns:
            List of batch results
        """
        self.logger.info("=" * 50)
        self.logger.info("Dataset Improvement Engine")
        self.logger.info("=" * 50)

        # Create backup
        if not self.config.dry_run:
            backup_file = self.backup_manager.create_backup(self.config.input_file)
            if backup_file:
                self.logger.success(f"Created backup: {backup_file}")

        # Determine line range
        total_lines = self.file_handler.count_lines(self.config.input_file)
        end_line = self.config.end_line or total_lines

        self.logger.info(f"Input file: {self.config.input_file}")
        self.logger.info(f"Output file: {self.config.output_file}")
        self.logger.info(f"Processing lines {self.config.start_line}-{end_line}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Dry run: {self.config.dry_run}")
        self.logger.info("=" * 50)

        # Process in batches
        batch_results = []
        current_line = self.config.start_line

        batch_num = 1
        while current_line <= end_line:
            batch_end = min(current_line + self.config.batch_size - 1, end_line)

            batch_result = self._process_batch(
                batch_num=batch_num,
                start_line=current_line,
                end_line=batch_end
            )

            batch_results.append(batch_result)

            # Log batch result
            self.logger.info(str(batch_result))

            current_line = batch_end + 1
            batch_num += 1

        # Summary
        self._print_summary(batch_results)

        return batch_results

    def _process_batch(self, batch_num: int, start_line: int, end_line: int) -> BatchResult:
        """
        Process a single batch of examples.

        Args:
            batch_num: Batch number
            start_line: Start line
            end_line: End line

        Returns:
            BatchResult
        """
        start_time = time.time()

        # Read examples
        examples = self.file_handler.read_jsonl(
            self.config.input_file,
            start_line=start_line,
            end_line=end_line
        )

        results = []

        for i, example in enumerate(examples):
            line_num = start_line + i
            result = self._improve_example(line_num, example)
            results.append(result)

            # Write immediately if not dry run
            if not self.config.dry_run and result.success:
                self._write_improved_example(result)

        duration = time.time() - start_time

        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        return BatchResult(
            batch_number=batch_num,
            start_line=start_line,
            end_line=end_line,
            results=results,
            total_processed=len(results),
            successful=successful,
            failed=failed,
            duration_seconds=duration
        )

    def _improve_example(self, line_number: int, example_dict: dict) -> ImprovementResult:
        """
        Improve a single example.

        Args:
            line_number: Line number
            example_dict: Example dictionary

        Returns:
            ImprovementResult
        """
        try:
            example = Example.from_dict(example_dict)

            # Extract thinking block from assistant message
            thinking_block = self._extract_thinking_block(example)

            if not thinking_block:
                return ImprovementResult(
                    line_number=line_number,
                    original=example,
                    improved=None,
                    success=False,
                    error="No thinking block found"
                )

            # Improve thinking block using LLM
            improved_thinking = self.llm_service.improve_thinking_block(thinking_block)

            # Validate improved thinking block
            is_valid, errors = self.validator.validate_thinking_block(improved_thinking)

            if not is_valid:
                return ImprovementResult(
                    line_number=line_number,
                    original=example,
                    improved=None,
                    success=False,
                    error="Validation failed",
                    validation_errors=errors
                )

            # Create improved example
            improved_example = self._create_improved_example(example, improved_thinking)

            return ImprovementResult(
                line_number=line_number,
                original=example,
                improved=improved_example,
                success=True
            )

        except Exception as e:
            return ImprovementResult(
                line_number=line_number,
                original=Example.from_dict(example_dict),
                improved=None,
                success=False,
                error=str(e)
            )

    def _extract_thinking_block(self, example: Example) -> Optional[dict]:
        """Extract thinking block from assistant message."""
        for conv in example.conversations:
            if conv.get("role") == "assistant":
                content = conv.get("content", "")

                # Find thinking block
                start = content.find("<thinking>")
                end = content.find("</thinking>")

                if start != -1 and end != -1:
                    thinking_json = content[start + len("<thinking>"):end].strip()
                    try:
                        return json.loads(thinking_json)
                    except json.JSONDecodeError:
                        return None

        return None

    def _create_improved_example(self, original: Example, improved_thinking: dict) -> Example:
        """Create improved example with new thinking block."""
        improved = Example.from_dict(original.to_dict())

        # Update assistant message with improved thinking
        for conv in improved.conversations:
            if conv.get("role") == "assistant":
                content = conv.get("content", "")
                start = content.find("<thinking>")
                end = content.find("</thinking>")

                if start != -1 and end != -1:
                    # Replace thinking block
                    new_thinking = json.dumps(improved_thinking, ensure_ascii=False)
                    new_content = (
                        content[:start + len("<thinking>")] +
                        f"\n{new_thinking}\n" +
                        content[end:]
                    )
                    conv["content"] = new_content

        return improved

    def _write_improved_example(self, result: ImprovementResult) -> None:
        """Write improved example to output file."""
        if result.improved:
            # Append to output file
            self.file_handler.write_jsonl(
                self.config.output_file,
                [result.improved.to_dict()],
                mode='a'
            )

    def _print_summary(self, batch_results: List[BatchResult]) -> None:
        """Print improvement summary."""
        total_processed = sum(b.total_processed for b in batch_results)
        total_successful = sum(b.successful for b in batch_results)
        total_failed = sum(b.failed for b in batch_results)
        total_time = sum(b.duration_seconds for b in batch_results)

        self.logger.info("=" * 50)
        self.logger.info("Summary")
        self.logger.info("=" * 50)
        self.logger.info(f"Total processed: {total_processed}")
        self.logger.info(f"Successful: {total_successful} ({total_successful/total_processed*100:.1f}%)")
        self.logger.info(f"Failed: {total_failed} ({total_failed/total_processed*100:.1f}%)")
        self.logger.info(f"Total time: {total_time:.1f}s")

        if self.config.dry_run:
            self.logger.info("")
            self.logger.warning("DRY RUN - No files were modified")
            self.logger.info("Run again without --dry-run to apply changes")
