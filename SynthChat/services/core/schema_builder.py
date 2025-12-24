"""Schema builder - builds combined schemas only.

Single Responsibility: Build combined JSON schemas from rubrics.
"""

from typing import Dict, List

from ...config import JudgeConfig


class SchemaBuilder:
    """
    Build combined JSON schemas.

    Responsibility: ONLY schema building (SRP).
    Does NOT build prompts or call LLM.
    """

    def __init__(self, judge_config: JudgeConfig):
        """
        Initialize schema builder.

        Args:
            judge_config: Judge configuration
        """
        self.judge_config = judge_config

    def build_combined_schema(self, rubrics: List[Dict]) -> Dict:
        """
        Build combined output schema from all rubrics.

        Args:
            rubrics: List of rubric dicts

        Returns:
            Combined JSON schema dict
        """
        properties = {}
        required = []

        for rubric in rubrics:
            schema = rubric.get("output_schema", {})

            # Add all properties from this rubric's schema
            for prop_name, prop_def in schema.get("properties", {}).items():
                properties[prop_name] = prop_def

            # Add required fields
            required.extend(schema.get("required", []))

        # Add feedback field (from config, not hardcoded!)
        feedback_field = self.judge_config.feedback_field
        properties[feedback_field] = {
            "type": "string",
            "description": "Combined explanation covering all rubric evaluations"
        }
        required.append(feedback_field)

        return {
            "type": "object",
            "properties": properties,
            "required": list(set(required)),  # dedupe
            "additionalProperties": False
        }
