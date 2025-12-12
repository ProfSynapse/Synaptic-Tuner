"""Clean up tool_schemas.yaml by removing deprecated fields."""

import yaml
from pathlib import Path


def cleanup():
    """Remove get_tools, context_schema, and file_path from schema."""
    yaml_path = Path(__file__).parent.parent / "improvement_engine" / "config" / "tool_schemas.yaml"

    print(f"Reading from: {yaml_path}")

    # Load YAML
    with open(yaml_path, 'r') as f:
        # Skip header comments
        lines = f.readlines()
        header = []
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                header.append(line)
                content_start = i + 1
            else:
                break

        # Parse YAML from content
        content = ''.join(lines[content_start:])
        schemas = yaml.safe_load(content)

    print(f"Loaded {len(schemas)} tool definitions")

    # Remove get_tools
    if 'get_tools' in schemas:
        del schemas['get_tools']
        print("✅ Removed 'get_tools' tool")

    # Remove context_schema and file_path from all tools
    removed_context = 0
    removed_filepath = 0

    for tool_name, tool_schema in schemas.items():
        if 'context_schema' in tool_schema:
            del tool_schema['context_schema']
            removed_context += 1

        if 'file_path' in tool_schema:
            del tool_schema['file_path']
            removed_filepath += 1

    print(f"✅ Removed context_schema from {removed_context} tools")
    print(f"✅ Removed file_path from {removed_filepath} tools")

    # Write back
    with open(yaml_path, 'w') as f:
        # Write header
        f.writelines(header)
        # Write cleaned schemas
        yaml.dump(schemas, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✅ Cleaned schema written to: {yaml_path}")
    print(f"✅ Total tools: {len(schemas)}")


if __name__ == "__main__":
    cleanup()
