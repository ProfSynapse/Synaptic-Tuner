"""LLM-based dataset labeling service."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from shared.llm import create_client
from ..core.models import Example, LabelingConfig, LabelingSession, LabelingResult
from ..core.exceptions import LLMServiceError, FileHandlerError, ValidationError
from ..services.file_handler import FileHandler
from ..utils.logger import ImproveLogger
from ..utils.yaml_loader import load_yaml_config
from ..utils.backup import BackupManager


class LLMLabelingService:
    """
    LLM-based labeling service for automated quality judging.

    Uses an LLM to evaluate examples against criteria and apply boolean labels:
    - label: good (true) or bad (false)
    - excellent: exemplary quality
    - hallucinated: contains hallucinations
    - poor_reasoning: weak/illogical reasoning
    - context_mismatch: doesn't align with context
    """

    def __init__(self, config: LabelingConfig, logger: ImproveLogger):
        """
        Initialize LLM labeling service.

        Args:
            config: Labeling configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.file_handler = FileHandler()

        # Load labeling configuration
        self.labeling_config = load_yaml_config(config.categories_config)
        if not self.labeling_config:
            raise FileHandlerError(f"Failed to load labeling config: {config.categories_config}")

        # Initialize LLM client
        self.llm_client = create_client(
            provider=config.backend,
            model=config.model
        )

        # Build evaluation prompt from criteria
        self.evaluation_prompt = self._build_evaluation_prompt()

    def _build_evaluation_prompt(self) -> str:
        """Build comprehensive evaluation prompt from criteria."""
        system_prompt = self.labeling_config.get("system_prompt", "")
        criteria = self.labeling_config.get("criteria", {})

        prompt_parts = [system_prompt.strip()]

        # Add criteria for each label
        if "label" in criteria:
            label_criteria = criteria["label"]
            prompt_parts.append("\n## OVERALL QUALITY (label field)")
            prompt_parts.append(label_criteria.get("true_description", "").strip())
            prompt_parts.append(label_criteria.get("false_description", "").strip())

        if "excellent" in criteria:
            prompt_parts.append("\n## EXCELLENT QUALITY (excellent field)")
            prompt_parts.append(criteria["excellent"].get("description", "").strip())

        if "hallucinated" in criteria:
            prompt_parts.append("\n## HALLUCINATION (hallucinated field)")
            prompt_parts.append(criteria["hallucinated"].get("description", "").strip())

        if "poor_reasoning" in criteria:
            prompt_parts.append("\n## POOR REASONING (poor_reasoning field)")
            prompt_parts.append(criteria["poor_reasoning"].get("description", "").strip())

        if "context_mismatch" in criteria:
            prompt_parts.append("\n## CONTEXT MISMATCH (context_mismatch field)")
            prompt_parts.append(criteria["context_mismatch"].get("description", "").strip())

        # Add JSON schema instructions
        prompt_parts.append("\n## RESPONSE FORMAT")
        prompt_parts.append("You must respond with valid JSON matching this schema:")
        prompt_parts.append(json.dumps(self.labeling_config.get("response_schema", {}), indent=2))

        return "\n".join(prompt_parts)

    def run(self) -> LabelingSession:
        """
        Main entry point for LLM-based labeling.

        Returns:
            Completed LabelingSession with statistics
        """
        # Validate config
        self.config.validate()

        # Check if input file exists
        if not os.path.exists(self.config.input_file):
            raise FileHandlerError(f"Input file not found: {self.config.input_file}")

        # Create backup if configured
        if self.labeling_config.get("processing", {}).get("create_backup", True):
            self._create_backup()

        # Initialize session
        session = self._initialize_session()

        self.logger.info(f"Starting LLM-based labeling for: {self.config.input_file}")
        self.logger.info(f"Backend: {self.config.backend}, Model: {self.config.model}")
        self.logger.info(f"Output will be written to: {self.config.output_file}")

        # Determine line range
        start_line, end_line = self._determine_line_range(session)
        total_to_process = end_line - start_line + 1

        self.logger.info(f"Processing lines {start_line}-{end_line} ({total_to_process} examples)")

        # Process in batches
        batch_size = self.labeling_config.get("processing", {}).get("batch_size", 10)

        try:
            current_line = start_line
            while current_line <= end_line:
                batch_end = min(current_line + batch_size - 1, end_line)

                # Read batch
                examples = self.file_handler.read_jsonl(
                    self.config.input_file,
                    start_line=current_line,
                    end_line=batch_end
                )

                # Process each example in batch
                for i, example in enumerate(examples):
                    line_number = current_line + i

                    # Label the example
                    result = self._label_example(example, line_number)

                    # Update session stats
                    if result.success:
                        session.labeled_count += 1
                        # Update category usage stats
                        if result.labeled.label:
                            session.categories_used["good"] = session.categories_used.get("good", 0) + 1
                        if result.labeled.excellent:
                            session.categories_used["excellent"] = session.categories_used.get("excellent", 0) + 1
                        if result.labeled.hallucinated:
                            session.categories_used["hallucinated"] = session.categories_used.get("hallucinated", 0) + 1
                        if result.labeled.poor_reasoning:
                            session.categories_used["poor_reasoning"] = session.categories_used.get("poor_reasoning", 0) + 1
                        if result.labeled.context_mismatch:
                            session.categories_used["context_mismatch"] = session.categories_used.get("context_mismatch", 0) + 1
                    else:
                        session.skipped_count += 1

                    session.last_line = line_number

                # Show progress
                progress = (session.labeled_count + session.skipped_count) / total_to_process * 100
                self.logger.info(
                    f"Progress: {session.labeled_count + session.skipped_count}/{total_to_process} "
                    f"({progress:.1f}%) | Success: {session.labeled_count} | Failed: {session.skipped_count}"
                )

                current_line = batch_end + 1

        except KeyboardInterrupt:
            self.logger.info("\nLabeling interrupted by user")
            raise

        # Print summary
        self._print_summary(session)

        return session

    def _initialize_session(self) -> LabelingSession:
        """Initialize labeling session."""
        total_examples = self.file_handler.count_lines(self.config.input_file)

        return LabelingSession(
            input_file=self.config.input_file,
            output_file=self.config.output_file,
            total_examples=total_examples,
            start_time=datetime.now().isoformat()
        )

    def _determine_line_range(self, session: LabelingSession) -> tuple[int, int]:
        """Determine the line range to process."""
        start_line = self.config.start_line
        end_line = self.config.end_line if self.config.end_line else session.total_examples

        return start_line, end_line

    def _label_example(self, example: Example, line_number: int) -> LabelingResult:
        """
        Label a single example using LLM.

        Args:
            example: Example to label
            line_number: Line number in dataset

        Returns:
            LabelingResult with applied labels
        """
        try:
            # Format example for LLM
            example_text = self._format_example_for_llm(example)

            # Create messages
            messages = [
                {"role": "system", "content": self.evaluation_prompt},
                {"role": "user", "content": f"Evaluate this training example:\n\n{example_text}"}
            ]

            # Get LLM judgment
            llm_config = self.labeling_config.get("llm", {})
            response = self.llm_client.structured_output(
                messages=messages,
                schema=self.labeling_config.get("response_schema", {}),
                temperature=llm_config.get("temperature", 0.3),
                max_tokens=llm_config.get("max_tokens", 500)
            )

            # Parse and validate response
            labels = self._parse_llm_response(response)

            # Apply labels to example
            labeled_example = self._apply_labels(example, labels)

            # Save labeled example (unless dry run)
            if not self.config.dry_run:
                self._save_labeled_example(labeled_example, line_number)

            self.logger.success(
                f"✓ Line {line_number}: label={labels.get('label')}, "
                f"excellent={labels.get('excellent')}, "
                f"hallucinated={labels.get('hallucinated')}"
            )

            return LabelingResult(
                line_number=line_number,
                original=example,
                labeled=labeled_example,
                tags=[],  # Not used for boolean labeling
                skipped=False,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            self.logger.error(f"✗ Line {line_number}: {str(e)}")
            return LabelingResult(
                line_number=line_number,
                original=example,
                labeled=example,
                tags=[],
                skipped=True,
                timestamp=datetime.now().isoformat()
            )

    def _format_example_for_llm(self, example: Example) -> str:
        """Format example for LLM evaluation."""
        parts = []

        for turn in example.conversations:
            role = turn.get("role", "").upper()
            content = turn.get("content", "")
            parts.append(f"{role}:\n{content}\n")

        return "\n".join(parts)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Response should already be structured from shared LLM client
            if isinstance(response, dict):
                return response

            # Parse JSON if it's a string
            labels = json.loads(response)

            # Validate required fields
            required_fields = ["label", "excellent", "hallucinated", "poor_reasoning", "context_mismatch"]
            for field in required_fields:
                if field not in labels:
                    raise ValidationError(f"Missing required field: {field}")
                if not isinstance(labels[field], bool):
                    raise ValidationError(f"Field '{field}' must be boolean, got {type(labels[field])}")

            return labels

        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON response: {e}")

    def _apply_labels(self, example: Example, labels: Dict[str, Any]) -> Example:
        """Apply LLM-judged labels to example."""
        return Example(
            conversations=example.conversations,
            label=labels.get("label"),
            behavior=example.behavior,
            pattern=example.pattern,
            excellent=labels.get("excellent"),
            hallucinated=labels.get("hallucinated"),
            poor_reasoning=labels.get("poor_reasoning"),
            context_mismatch=labels.get("context_mismatch")
        )

    def _save_labeled_example(self, example: Example, line_number: int):
        """Save labeled example to output file."""
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

    def _create_backup(self):
        """Create backup of input file."""
        try:
            backup_manager = BackupManager()
            backup_path = backup_manager.create_backup(self.config.input_file)
            if backup_path:
                self.logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")

    def _print_summary(self, session: LabelingSession):
        """Print labeling session summary."""
        total_processed = session.labeled_count + session.skipped_count
        success_rate = (session.labeled_count / total_processed * 100) if total_processed > 0 else 0

        self.logger.info("\n" + "="*60)
        self.logger.info("LABELING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total processed: {total_processed}/{session.total_examples}")
        self.logger.info(f"Successfully labeled: {session.labeled_count} ({success_rate:.1f}%)")
        self.logger.info(f"Failed: {session.skipped_count}")
        self.logger.info("")
        self.logger.info("Label Distribution:")
        for label, count in sorted(session.categories_used.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")
        self.logger.info("="*60 + "\n")
