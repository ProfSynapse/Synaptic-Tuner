"""Improvement engine - main orchestrator for judge → improve loop.

Single Responsibility: Orchestrate the improvement loop ONLY.
Delegates all actual work to focused services.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .config import ConfigLoader, ScopeConfig
from .services.data import RubricRepository
from .services.parsing import ConversationParser, ScopeExtractor
from .services.llm import LLMClient, PromptRenderer
from .services.core import (
    ValidationService,
    SchemaBuilder,
    JudgeService,
    ImprovementService,
    ImprovementApplicator,
)
from .utils.logger import ImproveLogger


@dataclass
class ImprovementResult:
    """Result from improvement engine."""
    improved_example: Dict
    iterations: int
    passed: bool
    final_scores: Dict[str, float]
    scopes_improved: List[str]


class ImprovementEngine:
    """
    Main improvement engine - orchestrates judge → improve loop.

    Responsibility: ONLY orchestrate the loop (SRP).
    All actual work delegated to focused services.
    """

    def __init__(
        self,
        llm_client,  # From shared.llm
        rubrics_dir: Path,
        config_path: Optional[Path] = None,
        logger: Optional[ImproveLogger] = None
    ):
        """
        Initialize improvement engine.

        Args:
            llm_client: LLM client (from shared.llm.create_client)
            rubrics_dir: Directory containing rubric YAML files
            config_path: Path to scope_config.yaml (optional)
            logger: Logger instance
        """
        self.logger = logger or ImproveLogger()

        # Load configuration
        config_loader = ConfigLoader(config_path)
        self.scope_config = config_loader.load()

        # Initialize data layer
        self.rubric_repo = RubricRepository(
            rubrics_dir=rubrics_dir,
            skip_files=["quality_labels.yaml"],
            logger=self.logger
        )

        # Initialize parsing layer
        self.conversation_parser = ConversationParser()
        self.scope_extractor = ScopeExtractor(self.scope_config, self.logger)

        # Initialize LLM layer
        self.llm_client = LLMClient(llm_client, self.logger)
        self.prompt_renderer = PromptRenderer()

        # Initialize services
        self.validation_service = ValidationService(
            self.scope_extractor,
            self.scope_config,
            self.logger
        )

        self.schema_builder = SchemaBuilder(self.scope_config.judge)

        self.judge_service = JudgeService(
            self.llm_client,
            self.prompt_renderer,
            self.schema_builder,
            self.scope_config,
            self.scope_config.judge,
            self.logger
        )

        self.improvement_service = ImprovementService(
            self.llm_client,
            self.prompt_renderer,
            self.scope_config,
            self.logger
        )

        self.improvement_applicator = ImprovementApplicator(
            self.scope_config,
            self.scope_extractor,
            self.logger
        )

    def run(
        self,
        example: Dict,
        rubric_keys: List[str],
        max_iterations: int = 3
    ) -> ImprovementResult:
        """
        Run improvement loop: judge → improve → repeat.

        Args:
            example: Example to improve
            rubric_keys: List of rubric keys to evaluate against
            max_iterations: Maximum improvement iterations

        Returns:
            ImprovementResult with final example and metrics
        """
        improved = example
        all_scopes_improved = []

        for iteration in range(max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # Load rubrics
            rubrics = [self.rubric_repo.get_rubric(key) for key in rubric_keys]

            # 1. Run validation
            validation_results = self.validation_service.validate_example(improved, rubrics)

            # 2. Build judge prompt
            judge_prompt = self._build_judge_prompt(improved, rubrics, validation_results)

            # 3. Execute judge
            judgment = self.judge_service.judge(judge_prompt, rubrics)

            # 4. Parse results
            scores, passed = self._parse_judgment(judgment, rubrics, validation_results)

            if passed:
                self.logger.info(f"Passed after {iteration + 1} iterations")
                return ImprovementResult(
                    improved_example=improved,
                    iterations=iteration + 1,
                    passed=True,
                    final_scores=scores,
                    scopes_improved=all_scopes_improved
                )

            # 5. Detect failed scopes
            failed_scopes = self._detect_failed_scopes(scores, rubrics)
            self.logger.info(f"Failed scopes: {failed_scopes}")

            # 6. Improve each failed scope
            for scope in failed_scopes:
                improved = self._improve_scope(
                    improved,
                    scope,
                    rubrics,
                    judgment
                )
                all_scopes_improved.append(scope)

        # Max iterations reached
        return ImprovementResult(
            improved_example=improved,
            iterations=max_iterations,
            passed=False,
            final_scores=scores,
            scopes_improved=all_scopes_improved
        )

    def _build_judge_prompt(
        self,
        example: Dict,
        rubrics: List[Dict],
        validation_results: Dict
    ) -> str:
        """Build judge prompt from rubrics and validation results."""
        # Extract conversations
        conversation = self.conversation_parser.parse(example)
        system_msg = conversation.get_system_message()
        user_msg = conversation.get_user_message()
        assistant_msg = conversation.get_assistant_message()

        system_content = system_msg.content if system_msg else ""
        user_content = user_msg.content if user_msg else ""
        assistant_content = assistant_msg.content if assistant_msg else ""

        # Build prompt sections
        parts = [self.scope_config.judge.prompt_structure.get("header", "")]
        parts.append("")

        sep = self.scope_config.judge.prompt_structure.get("section_separator", "=") * self.scope_config.judge.prompt_structure.get("section_width", 60)

        # Criteria section
        parts.extend([sep, "EVALUATION CRITERIA", sep, ""])
        for rubric in rubrics:
            parts.append("")
            parts.append(rubric.get("judge_prompt", f"# {rubric.get('name', 'Unknown')}"))

        # Validation results section
        if validation_results:
            parts.extend(["", sep, "SCHEMA VALIDATION RESULTS", sep, ""])
            for rubric_key, (is_valid, errors) in validation_results.items():
                icon = self.scope_config.validation.icons["passed" if is_valid else "failed"]
                parts.append(f"{icon} {rubric_key}: Schema validation {'PASSED' if is_valid else 'FAILED'}")
                if not is_valid:
                    for error in errors:
                        parts.append(f"   - {error}")

        # Example section
        parts.extend(["", sep, "EXAMPLE TO EVALUATE", sep, ""])
        parts.extend(["**System Prompt:**", "```", system_content, "```", ""])
        parts.extend(["**User Request:**", "```", user_content, "```", ""])
        parts.extend(["**Assistant Response:**", "```", assistant_content, "```", ""])

        # Evaluation instructions
        parts.extend(["", sep, "YOUR EVALUATION", sep, ""])
        parts.extend([
            "Evaluate the response against ALL criteria above.",
            "Output a single JSON object with:",
            "- One score field per rubric (0.0-1.0)",
            f"- One `{self.scope_config.judge.feedback_field}` field explaining all evaluations",
            "",
            "IMPORTANT: Respond with ONLY valid JSON. No markdown, no explanations.",
            "Start with { and end with }"
        ])

        return "\n".join(parts)

    def _parse_judgment(
        self,
        judgment: Dict,
        rubrics: List[Dict],
        validation_results: Dict
    ) -> Tuple[Dict[str, float], bool]:
        """Parse judgment and determine pass/fail."""
        scores = {}

        # Extract scores for each rubric
        for rubric in rubrics:
            rubric_key = rubric.get("name", "unknown")
            threshold = rubric.get("pass_threshold", 0.8)

            # Find score field
            score = None
            score_field = f"{rubric_key}{self.scope_config.judge.score_field_suffix}"

            if score_field in judgment:
                score = float(judgment[score_field])
            else:
                # Try finding any score field
                for key, value in judgment.items():
                    if "score" in key.lower() and isinstance(value, (int, float)):
                        score = float(value)
                        break

            if score is None:
                self.logger.warning(f"No score found for {rubric_key}")
                score = 0.0

            scores[rubric_key] = score

        # Add schema validation scores
        for rubric_key, (is_valid, _) in validation_results.items():
            schema_key = f"{rubric_key}{self.scope_config.validation.schema_score_suffix}"
            scores[schema_key] = 1.0 if is_valid else 0.0

        # Determine overall pass
        passed = all(
            scores.get(rubric.get("name", ""), 0) >= rubric.get("pass_threshold", 0.8)
            for rubric in rubrics
        )

        return scores, passed

    def _detect_failed_scopes(
        self,
        scores: Dict[str, float],
        rubrics: List[Dict]
    ) -> List[str]:
        """Detect which scopes failed."""
        failed_scopes = set()

        for rubric in rubrics:
            rubric_key = rubric.get("name", "")
            threshold = rubric.get("pass_threshold", 0.8)
            score = scores.get(rubric_key, 0)

            if score < threshold:
                scope = rubric.get("scope", "response")
                failed_scopes.add(scope)

        # Return in configured order
        return [s for s in self.scope_config.scope_processing_order if s in failed_scopes]

    def _improve_scope(
        self,
        example: Dict,
        scope: str,
        rubrics: List[Dict],
        judgment: Dict
    ) -> Dict:
        """Improve a single scope."""
        # Find rubric for this scope
        rubric = None
        for r in rubrics:
            if r.get("scope") == scope:
                rubric = r
                break

        if not rubric:
            self.logger.warning(f"No rubric for scope {scope}")
            return example

        # Build improvement prompt from rubric template
        template = rubric.get("improvement_prompt_template", "")
        if not template:
            self.logger.warning(f"No improvement template for {rubric.get('name')}")
            return example

        # Extract current content
        current_content = self.scope_extractor.extract(example, scope)
        feedback = judgment.get(self.scope_config.judge.feedback_field, "")

        # Render prompt
        improve_prompt = self.prompt_renderer.render_safe(
            template,
            {
                "current_content": current_content or "",
                "feedback": feedback,
                "format_instructions": rubric.get("format_spec", ""),
            }
        )

        # Execute improvement
        improved_content = self.improvement_service.improve(improve_prompt)

        # Apply improvement
        return self.improvement_applicator.apply(example, scope, improved_content)
