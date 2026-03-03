"""Schema builder -- merge rubric output schemas into a combined JSON schema.

Location: shared/judge/schema_builder.py
Summary: Builds a single combined JSON Schema from multiple RubricDef
         output_schema definitions. The combined schema is passed to
         BaseLLMClient.structured_output() for the judge LLM call.
         Mirrors SynthChat's SchemaBuilder.build_combined_schema() but
         accepts typed RubricDef instances instead of raw dicts.
"""

import logging
from typing import Dict, List

from .models import JudgeConfig, RubricDef

logger = logging.getLogger(__name__)


class SchemaBuilder:
    """Build combined JSON schema from multiple rubrics.

    Args:
        judge_config: Configuration controlling the feedback field name.
    """

    def __init__(self, judge_config: JudgeConfig):
        self.judge_config = judge_config

    def build(self, rubrics: List[RubricDef]) -> Dict:
        """Merge output_schema from all rubrics into one JSON schema.

        The combined schema includes:
        - All properties from each rubric's output_schema
        - All required fields from each rubric
        - A feedback field (name from JudgeConfig.feedback_field)

        Args:
            rubrics: List of RubricDef instances to combine.

        Returns:
            Combined JSON Schema dict suitable for
            BaseLLMClient.structured_output().
        """
        properties = {}
        required = []

        for rubric in rubrics:
            schema = rubric.output_schema

            # Merge properties from this rubric's schema (first definition wins)
            for prop_name, prop_def in schema.get("properties", {}).items():
                if prop_name in properties:
                    logger.warning(
                        "Schema property '%s' already defined by a previous rubric; "
                        "keeping first definition (from rubric '%s')",
                        prop_name,
                        rubric.key,
                    )
                else:
                    properties[prop_name] = prop_def

            # Collect required fields
            required.extend(schema.get("required", []))

        # Add the configurable feedback field
        feedback_field = self.judge_config.feedback_field
        properties[feedback_field] = {
            "type": "string",
            "description": "Combined explanation covering all rubric evaluations",
        }
        required.append(feedback_field)

        return {
            "type": "object",
            "properties": properties,
            "required": list(set(required)),  # deduplicate
            "additionalProperties": False,
        }
