"""Convert tool_schemas.json to YAML format for improvement engine."""

import json
import yaml
from pathlib import Path


def convert():
    """Convert JSON schema to YAML format."""
    json_path = Path(__file__).parent / "tool_schemas.json"
    yaml_path = Path(__file__).parent.parent / "improvement_engine" / "config" / "tool_schemas.yaml"

    print(f"Reading from: {json_path}")

    # Load JSON
    with open(json_path, 'r') as f:
        schemas = json.load(f)

    print(f"Loaded {len(schemas)} tool definitions")

    # Create output directory
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML with header
    with open(yaml_path, 'w') as f:
        f.write("# Tool schemas converted from tool_schemas.json\n")
        f.write("# Original file: Tools/tool_schemas.json (preserved)\n")
        f.write("# \n")
        f.write("# This file is used by the improvement engine for tool call validation.\n")
        f.write("# Format matches the original JSON structure.\n")
        f.write("#\n")
        f.write("# To regenerate: python Tools/convert_schemas_to_yaml.py\n\n")

        yaml.dump(schemas, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✅ Converted to: {yaml_path}")
    print(f"✅ Original JSON preserved at: {json_path}")


if __name__ == "__main__":
    convert()
