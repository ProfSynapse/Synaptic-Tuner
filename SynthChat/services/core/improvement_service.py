"""Improvement service - executes improvement LLM calls only.

Single Responsibility: Execute ONE improvement call to LLM.
"""

from typing import Optional

from ...utils import PromptRenderer
from ...config import ScopeConfig
from ...utils.logger import ImproveLogger


class ImprovementService:
    """
    Execute improvement LLM calls.

    Responsibility: ONLY execute improvement calls (SRP).
    Does NOT detect failures, build prompts, or apply improvements.
    """

    def __init__(
        self,
        llm_client,  # BaseLLMClient from shared.llm
        prompt_renderer: PromptRenderer,
        scope_config: ScopeConfig,
        logger: Optional[ImproveLogger] = None
    ):
        """
        Initialize improvement service.

        Args:
            llm_client: LLM client for API calls (from shared.llm)
            prompt_renderer: Prompt renderer for templates
            scope_config: Scope configuration
            logger: Logger instance
        """
        self.llm_client = llm_client
        self.prompt_renderer = prompt_renderer
        self.scope_config = scope_config
        self.logger = logger or ImproveLogger()

    def improve(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Execute improvement call to LLM.

        Args:
            prompt: Complete improvement prompt (already built)
            system_prompt: Optional system prompt for the improver

        Returns:
            Improved content string from LLM
        """
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call LLM - shared.llm.chat() returns str directly
            response = self.llm_client.chat(
                messages=messages,
                temperature=self.scope_config.llm.improvement_temperature,
                max_tokens=self.scope_config.llm.improvement_max_tokens
            )

            return response if isinstance(response, str) else str(response)

        except Exception as e:
            self.logger.error(f"Improvement call error: {e}")
            raise
