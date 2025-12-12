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
from .services.interaction_logger import InteractionLogger
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
        logger: Optional[ImproveLogger] = None,
        interaction_logger: Optional[InteractionLogger] = None,
        enable_interactions: bool = True
    ):
        """
        Initialize improvement engine.

        Args:
            llm_client: LLM client (from shared.llm.create_client)
            rubrics_dir: Directory containing rubric YAML files
            config_path: Path to scope_config.yaml (optional)
            logger: Logger instance
            interaction_logger: Interaction logger (optional, will create if None)
            enable_interactions: Whether to enable interaction logging (default: True)
        """
        self.logger = logger or ImproveLogger()

        # Initialize interaction logger
        if interaction_logger is not None:
            self.interaction_logger = interaction_logger
        elif enable_interactions:
            interactions_dir = Path(rubrics_dir).parent / "interactions"
            self.interaction_logger = InteractionLogger(
                output_dir=interactions_dir,
                enabled=True,
                logger=self.logger
            )
        else:
            self.interaction_logger = InteractionLogger(
                output_dir=Path("/tmp"),
                enabled=False,
                logger=self.logger
            )

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
        Run improvement loop: process scopes sequentially.

        Pipeline: system_prompt → thinking → response
        Each scope receives improvements from previous scopes.

        Args:
            example: Example to improve
            rubric_keys: List of rubric keys to evaluate against
            max_iterations: Maximum improvement iterations per scope

        Returns:
            ImprovementResult with final example and metrics
        """
        # Load all rubrics and group by scope
        all_rubrics = [self.rubric_repo.get_rubric(key) for key in rubric_keys]
        by_scope = self._group_rubrics_by_scope(all_rubrics)

        improved = example
        all_scopes_improved = []
        all_scores = {}
        total_iterations = 0

        # Process scopes in order: system_prompt → thinking → response
        for scope in self.scope_config.scope_processing_order:
            scope_rubrics = by_scope.get(scope, [])
            if not scope_rubrics:
                continue

            self.logger.info(f"Processing scope: {scope} ({len(scope_rubrics)} rubrics)")

            # Run improvement loop for this scope
            scope_passed = False
            for iteration in range(max_iterations):
                total_iterations += 1
                self.logger.info(f"  {scope} iteration {iteration + 1}/{max_iterations}")

                # 1. Validate with this scope's rubrics only
                validation_results = self.validation_service.validate_example(improved, scope_rubrics)

                # 2. Build judge prompt with this scope's rubrics only
                judge_system, judge_user = self._build_judge_prompt(improved, scope_rubrics, validation_results)

                # 3. Execute judge (smaller schema = no 400 error!)
                judgment = self.judge_service.judge(judge_user, scope_rubrics, system_prompt=judge_system)

                # 4. Parse results
                scores, passed = self._parse_judgment(judgment, scope_rubrics, validation_results)
                all_scores.update(scores)

                # Log judge interactions
                for rubric in scope_rubrics:
                    rubric_key = rubric.get("key", rubric.get("name"))
                    score = scores.get(rubric.get("name", rubric_key), 0.0)
                    threshold = rubric.get("pass_threshold", 0.8)
                    rubric_passed = score >= threshold

                    self.interaction_logger.log_judge_interaction(
                        judge_prompt=judge_user,
                        judge_response=str(judgment),
                        rubric_name=rubric.get("name", rubric_key),
                        score=score,
                        passed=rubric_passed,
                        example_id=None,
                        system_prompt=judge_system
                    )

                if passed:
                    self.logger.info(f"  {scope} passed after {iteration + 1} iterations")
                    scope_passed = True
                    break

                # 5. Collect ALL failing rubrics for this scope
                failing_rubrics = []
                for rubric in scope_rubrics:
                    rubric_key = rubric.get("key", rubric.get("name"))
                    rubric_name = rubric.get("name", rubric_key)
                    before_score = scores.get(rubric_name, 0.0)
                    threshold = rubric.get("pass_threshold", 0.8)

                    # Check if schema validation failed for this rubric
                    schema_failed = False
                    if rubric_key in validation_results:
                        is_valid, _ = validation_results[rubric_key]
                        schema_failed = not is_valid

                    # Add to failing list if judge score failed OR schema failed
                    if before_score < threshold or schema_failed:
                        failing_rubrics.append(rubric)

                # 6. Improve with ALL failing rubrics together (single call)
                if failing_rubrics:
                    improved_example, improve_prompt, improved_content, system_prompt = self._improve_scope_with_logging(
                        improved,
                        scope,
                        failing_rubrics,  # ALL failing rubrics together
                        judgment
                    )

                    if improved_example != improved:
                        # Re-evaluate after improvement with ALL scope rubrics
                        after_validation = self.validation_service.validate_example(improved_example, scope_rubrics)
                        after_system, after_user = self._build_judge_prompt(improved_example, scope_rubrics, after_validation)
                        after_judgment = self.judge_service.judge(
                            after_user,
                            scope_rubrics,
                            system_prompt=after_system
                        )
                        after_scores, _ = self._parse_judgment(after_judgment, scope_rubrics, after_validation)

                        # Log improver interaction for the consolidated improvement
                        rubric_names = ", ".join(r.get("name", "unknown") for r in failing_rubrics)
                        self.interaction_logger.log_improver_interaction(
                            improver_prompt=improve_prompt,
                            improver_response=improved_content or "",
                            rubric_name=rubric_names,
                            scope=scope,
                            before_score=min(scores.get(r.get("name", ""), 0) for r in failing_rubrics),
                            after_score=min(after_scores.get(r.get("name", ""), 0) for r in failing_rubrics),
                            improved=True,
                            example_id=None,
                            system_prompt=system_prompt
                        )

                        improved = improved_example
                        all_scopes_improved.append(scope)
                        all_scores.update(after_scores)
                    else:
                        self.logger.warning(f"  No improvement applied for {scope}")

            if not scope_passed:
                self.logger.warning(f"  {scope} did not pass after {max_iterations} iterations")

        # Check if all scopes passed
        all_passed = all(
            all_scores.get(rubric.get("name", ""), 0) >= rubric.get("pass_threshold", 0.8)
            for rubric in all_rubrics
        )

        return ImprovementResult(
            improved_example=improved,
            iterations=total_iterations,
            passed=all_passed,
            final_scores=all_scores,
            scopes_improved=all_scopes_improved
        )

    def _group_rubrics_by_scope(self, rubrics: List[Dict]) -> Dict[str, List[Dict]]:
        """Group rubrics by their scope. Supports single scope or list of scopes."""
        by_scope = {}
        for rubric in rubrics:
            scope = rubric.get("scope", "response")
            # Support both single scope string and list of scopes
            scopes = scope if isinstance(scope, list) else [scope]
            for s in scopes:
                if s not in by_scope:
                    by_scope[s] = []
                by_scope[s].append(rubric)
        return by_scope

    def _build_judge_prompt(
        self,
        example: Dict,
        rubrics: List[Dict],
        validation_results: Dict
    ) -> tuple[str, str]:
        """
        Build judge prompt split into system and user parts.

        Returns:
            (system_prompt, user_prompt) tuple
        """
        # Extract conversations
        conversation = self.conversation_parser.parse(example)
        system_msg = conversation.get_system_message()
        user_msg = conversation.get_user_message()
        assistant_msg = conversation.get_assistant_message()

        system_content = system_msg.content if system_msg else ""
        user_content = user_msg.content if user_msg else ""
        assistant_content = assistant_msg.content if assistant_msg else ""

        # Build SYSTEM prompt (guidance + rubrics + structure)
        system_parts = ["You are a quality judge. Evaluate examples and provide scores with improvement recommendations."]

        # Add rubric criteria
        system_parts.extend(["", "## RUBRICS"])
        for rubric in rubrics:
            rubric_name = rubric.get("name", "Unknown")
            system_parts.append(f"\n**{rubric_name}**")
            system_parts.append(rubric.get("judge_prompt", ""))

        # Add structure requirements (schema validation results)
        has_structure_errors = any(not is_valid for is_valid, _ in validation_results.values()) if validation_results else False
        if validation_results and has_structure_errors:
            system_parts.extend(["", "## STRUCTURE REQUIREMENTS (MANDATORY)"])
            system_parts.append("**CRITICAL: Any structure error below MUST result in score 0.0 for that rubric.**")
            system_parts.append("**You MUST include these errors in your improvement_feedback/overall_feedback.**")
            system_parts.append("")
            for rubric_key, (is_valid, errors) in validation_results.items():
                if not is_valid:
                    system_parts.append(f"**{rubric_key}** - FAILS SCHEMA:")
                    for error in errors:
                        system_parts.append(f"  ❌ {error}")

        # Output format
        system_parts.extend(["", "## OUTPUT"])
        score_fields = [f'"{r.get("key", r.get("name"))}_score": 0.0-1.0' for r in rubrics]
        system_parts.append(f"Return JSON: {{{', '.join(score_fields)}, \"{self.scope_config.judge.feedback_field}\": \"recommendations\"}}")

        # Build USER prompt (just the example to evaluate)
        user_parts = ["## EXAMPLE TO EVALUATE", ""]
        user_parts.extend(["**System:**", "```", system_content, "```", ""])
        user_parts.extend(["**User:**", "```", user_content, "```", ""])
        user_parts.extend(["**Assistant:**", "```", assistant_content, "```"])

        return "\n".join(system_parts), "\n".join(user_parts)

    def _parse_judgment(
        self,
        judgment: Dict,
        rubrics: List[Dict],
        validation_results: Dict
    ) -> Tuple[Dict[str, float], bool]:
        """
        Parse judgment and determine pass/fail.

        AUTO-FAIL: If validation fails, score is automatically 0.0
        regardless of what the judge says.
        """
        scores = {}
        self._mandatory_fixes = []  # Store for improver

        # First, process validation results (AUTO-FAIL on validation errors)
        failed_rubrics = set()
        for rubric_key, (is_valid, errors) in validation_results.items():
            schema_key = f"{rubric_key}{self.scope_config.validation.schema_score_suffix}"
            scores[schema_key] = 1.0 if is_valid else 0.0

            if not is_valid:
                # AUTO-FAIL: Set score to 0.0 for this rubric
                scores[rubric_key] = 0.0
                failed_rubrics.add(rubric_key)
                # Collect mandatory fixes
                self._mandatory_fixes.extend(errors)
                self.logger.info(f"  AUTO-FAIL: {rubric_key} failed validation ({len(errors)} errors)")

        # Extract judge scores for rubrics that PASSED validation
        for rubric in rubrics:
            rubric_key = rubric.get("name", "unknown")

            # Skip if already auto-failed
            if rubric_key in failed_rubrics:
                continue

            # Find score field from judge
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

    def _improve_scope_with_logging(
        self,
        example: Dict,
        scope: str,
        rubrics: List[Dict],
        judgment: Dict
    ) -> Tuple[Dict, str, str, str]:
        """
        Improve a single scope using ALL provided rubrics together.

        Returns:
            Tuple of (improved_example, improve_prompt, improved_content, system_prompt)
        """
        # Find ALL rubrics for this scope
        scope_rubrics = []
        for r in rubrics:
            r_scope = r.get("scope", "response")
            # Support both single scope string and list of scopes
            if isinstance(r_scope, list):
                if scope in r_scope:
                    scope_rubrics.append(r)
            elif r_scope == scope:
                scope_rubrics.append(r)

        if not scope_rubrics:
            self.logger.warning(f"No rubrics for scope {scope}")
            return example, "", "", ""

        # Combine ALL rubric improver_prompts into one consolidated prompt
        combined_templates = []
        rubric_names = []
        for rubric in scope_rubrics:
            template = rubric.get("improver_prompt", "")
            if template:
                rubric_names.append(rubric.get("name", "unknown"))
                combined_templates.append(f"### {rubric.get('name')} Requirements\n{template}")

        if not combined_templates:
            self.logger.warning(f"No improver_prompt templates for scope {scope}")
            return example, "", "", ""

        self.logger.info(f"    Improving {scope} with rubrics: {', '.join(rubric_names)}")

        # Get scope handler to build prompt variables
        from .services.scope_handlers import get_handler
        handler = get_handler(scope, self.scope_config, self.scope_extractor, self.logger)

        # Build template variables using scope handler
        template_vars = handler.build_prompt_variables(example, judgment)

        # Add feedback to template vars
        feedback = judgment.get(self.scope_config.judge.feedback_field, "")
        template_vars["feedback"] = feedback

        # Build SYSTEM prompt for improver (mandatory fixes + quality recommendations)
        system_parts = []
        system_parts.append(f"You are improving {scope} content to satisfy ALL of these rubrics: {', '.join(rubric_names)}.")
        system_parts.append("Your improvement must address ALL requirements from ALL rubrics together.")

        # Add MANDATORY FIXES first (from validation errors - must be fixed)
        mandatory_fixes = getattr(self, '_mandatory_fixes', [])
        if mandatory_fixes:
            system_parts.extend(["", "## MANDATORY FIXES (Must resolve to pass)"])
            system_parts.append("These validation errors MUST be fixed:")
            for fix in mandatory_fixes:
                system_parts.append(f"  - {fix}")

        # Add QUALITY RECOMMENDATIONS (from judge - optional improvements)
        if feedback:
            system_parts.extend(["", "## QUALITY RECOMMENDATIONS", feedback])

        system_prompt = "\n".join(system_parts)

        # Build USER prompt by combining ALL rubric templates
        combined_template = "\n\n".join(combined_templates)
        user_prompt = self.prompt_renderer.render_safe(combined_template, template_vars)

        # Execute improvement
        improved_content = self.improvement_service.improve(user_prompt, system_prompt)

        # Log what the improver returned
        content_preview = improved_content[:200] if improved_content else "(empty)"
        self.logger.info(f"    Improver returned ({len(improved_content)} chars): {content_preview}...")

        # Apply improvement
        improved_example = self.improvement_applicator.apply(example, scope, improved_content)

        # Log whether the example changed
        if improved_example == example:
            self.logger.warning(f"    Improvement NOT applied - example unchanged")
        else:
            self.logger.info(f"    Improvement applied successfully")

        # Return example, prompts, content for logging
        return improved_example, user_prompt, improved_content, system_prompt
