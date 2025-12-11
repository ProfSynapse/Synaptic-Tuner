"""Judge service - executes judge LLM calls only.

Single Responsibility: Execute ONE judge call to LLM.
"""

from typing import Dict, List, Optional

from ..llm import LLMClient, PromptRenderer
from .schema_builder import SchemaBuilder
from ...config import ScopeConfig, JudgeConfig
from ...utils.logger import ImproveLogger


class JudgeService:
    """
    Execute judge LLM calls.

    Responsibility: ONLY execute judge calls (SRP).
    Does NOT validate, parse results, or make decisions.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_renderer: PromptRenderer,
        schema_builder: SchemaBuilder,
        scope_config: ScopeConfig,
        judge_config: JudgeConfig,
        logger: Optional[ImproveLogger] = None
    ):
        """
        Initialize judge service.

        Args:
            llm_client: LLM client for API calls
            prompt_renderer: Prompt renderer for templates
            schema_builder: Schema builder for combined schemas
            scope_config: Scope configuration
            judge_config: Judge configuration
            logger: Logger instance
        """
        self.llm_client = llm_client
        self.prompt_renderer = prompt_renderer
        self.schema_builder = schema_builder
        self.scope_config = scope_config
        self.judge_config = judge_config
        self.logger = logger or ImproveLogger()

    def judge(
        self,
        prompt: str,
        rubrics: List[Dict]
    ) -> Dict:
        """
        Execute judge call to LLM.

        Args:
            prompt: Complete judge prompt (already built)
            rubrics: List of rubric dicts

        Returns:
            Raw judgment dict from LLM
        """
        try:
            # Build combined schema
            schema = self.schema_builder.build_combined_schema(rubrics)

            # Call LLM
            judgment = self.llm_client.structured_output(
                messages=[{"role": "user", "content": prompt}],
                schema=schema,
                temperature=self.scope_config.llm.judge_temperature
            )

            return judgment

        except Exception as e:
            self.logger.error(f"Judge call error: {e}")
            raise
