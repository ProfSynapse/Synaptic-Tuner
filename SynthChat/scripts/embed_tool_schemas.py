#!/usr/bin/env python3
"""
Embed tool schemas from cli-first-tool-schemas.json into agent rubric YAMLs.

Reads the JSON Schema format and converts to simplified YAML format
that StructureValidator can use for tool call validation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List


def load_tool_schemas(schema_file: Path) -> Dict:
    """Load tool schemas from JSON file."""
    with open(schema_file) as f:
        return json.load(f)


def convert_json_schema_to_simple(json_schema: Dict) -> Dict:
    """
    Convert JSON Schema format to simplified schema format.

    JSON Schema format:
        properties:
          context:
            properties:
              sessionId: {type: string}
          calls:
            items:
              properties:
                params:
                  properties:
                    path: {type: string}

    Simplified format:
        context:
          sessionId: string
        path: string

    NOTE: The actual training data uses a different context format than
    cli-first-tool-schemas.json. We use the training data format here:
    - sessionId, workspaceId, sessionDescription, sessionMemory,
      toolContext, primaryGoal, subgoal
    """
    simple = {}

    properties = json_schema.get("properties", {})

    # Use the ACTUAL context format from training data
    # (not the format in cli-first-tool-schemas.json)
    simple["context"] = {
        "sessionId": "string",
        "workspaceId": "string",
        "sessionDescription": "string",
        "sessionMemory": "string",
        "toolContext": "string",
        "primaryGoal": "string",
        "subgoal": "string",
    }

    # Extract params from calls[].items.properties.params
    calls_schema = properties.get("calls", {})
    items_schema = calls_schema.get("items", {})
    items_props = items_schema.get("properties", {})
    params_schema = items_props.get("params", {})
    params_props = params_schema.get("properties", {})
    params_required = params_schema.get("required", [])

    for field, field_schema in params_props.items():
        # Only include required params
        if field in params_required:
            simple[field] = field_schema.get("type", "string")

    return simple


def group_tools_by_agent(tools: Dict) -> Dict[str, Dict]:
    """Group tools by agent prefix (e.g., storageManager_, contentManager_)."""
    agents = {}

    for tool_name, schema in tools.items():
        # Extract agent prefix
        parts = tool_name.split("_")
        if len(parts) >= 2:
            agent = parts[0]
            if agent not in agents:
                agents[agent] = {}
            agents[agent][tool_name] = schema

    return agents


def generate_validations_yaml(tools: Dict[str, Dict]) -> str:
    """
    Generate YAML validations section for embedding in rubric.

    Output format:
        validations:
          - tools:
              toolName:
                context:
                  sessionId: string
                path: string
    """
    validations = [{
        "tools": tools,
        "error": "Tool '{tool_name}': {details}"
    }]

    return yaml.dump({"validations": validations}, default_flow_style=False, sort_keys=False)


def main():
    """Main entry point."""
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    schema_file = repo_root / "cli-first-tool-schemas.json"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    print(f"Loading schemas from: {schema_file}")

    # Load and convert schemas
    raw_schemas = load_tool_schemas(schema_file)

    # Convert each schema to simplified format
    simple_schemas = {}
    for tool_name, json_schema in raw_schemas.items():
        simple_schemas[tool_name] = convert_json_schema_to_simple(json_schema)

    # Group by agent
    agents = group_tools_by_agent(simple_schemas)

    print(f"\nFound {len(agents)} agents:")
    for agent, tools in agents.items():
        print(f"  - {agent}: {len(tools)} tools")

    # Generate YAML for each agent
    print("\n--- Generated YAML for each agent ---\n")

    for agent, tools in sorted(agents.items()):
        rubric_file = rubrics_dir / f"{agent}_tools.yaml"

        print(f"=== {agent} ({rubric_file.name}) ===")
        yaml_content = generate_validations_yaml(tools)
        print(yaml_content)
        print()

        # Optionally write to file or print instructions
        if rubric_file.exists():
            print(f"  [INFO] Rubric exists: {rubric_file}")
            print(f"  [TODO] Add the above 'validations' section to this file")
        else:
            print(f"  [WARN] Rubric not found: {rubric_file}")
        print()


if __name__ == "__main__":
    main()
