"""Rubric loading and management."""

from pathlib import Path
from typing import Dict, List
from ..utils.yaml_loader import load_yaml


def get_available_rubrics() -> List[str]:
    """
    Get list of available rubric names.

    Returns:
        List of rubric names (without .yaml extension)
    """
    rubrics_dir = Path(__file__).parent
    return sorted([
        f.stem for f in rubrics_dir.glob("*.yaml")
    ])


def load_rubric(rubric_name: str) -> Dict:
    """
    Load a rubric by name.

    Args:
        rubric_name: Name of rubric (without .yaml extension)

    Returns:
        Rubric configuration dict

    Raises:
        FileNotFoundError: If rubric doesn't exist
    """
    rubrics_dir = Path(__file__).parent
    rubric_file = rubrics_dir / f"{rubric_name}.yaml"

    if not rubric_file.exists():
        available = get_available_rubrics()
        raise FileNotFoundError(
            f"Rubric '{rubric_name}' not found. "
            f"Available rubrics: {', '.join(available)}"
        )

    rubric = load_yaml(rubric_file)

    # Validate required fields
    required_fields = ["name", "description", "scope", "pass_threshold",
                       "judge_prompt", "output_schema"]

    for field in required_fields:
        if field not in rubric:
            raise ValueError(
                f"Rubric '{rubric_name}' missing required field: {field}"
            )

    # improver_prompt is optional (some rubrics are judge-only)

    # Validate scope
    valid_scopes = ["response", "thinking", "tool_calls", "text"]
    if rubric["scope"] not in valid_scopes:
        raise ValueError(
            f"Rubric '{rubric_name}' has invalid scope: {rubric['scope']}. "
            f"Must be one of: {', '.join(valid_scopes)}"
        )

    return rubric


def list_rubrics() -> None:
    """Print available rubrics with descriptions."""
    available = get_available_rubrics()

    if not available:
        print("\nNo rubrics found in improvement_engine/rubrics/")
        return

    print("\nAvailable Rubrics:")
    print("=" * 70)

    for rubric_name in available:
        try:
            rubric = load_rubric(rubric_name)
            print(f"\n{rubric_name}")
            print(f"  Name: {rubric['name']}")
            print(f"  Description: {rubric['description']}")
            print(f"  Scope: {rubric['scope']}")
            print(f"  Pass Threshold: {rubric['pass_threshold']}")
        except Exception as e:
            print(f"\n{rubric_name}")
            print(f"  Error: {e}")

    print()
