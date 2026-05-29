#!/usr/bin/env python3
"""
Audit CLI-first tool schemas and regenerate evaluator config.

This script:
1. Parses cli-first-tool-schemas.json (source of truth)
2. Compares against Evaluator/config/tool_schema.yaml
3. Generates corrected Evaluator/config/tool_schema_corrected.yaml
4. Audits test case questions for non-existent params
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set

import yaml


def load_tool_schemas(path: Path) -> Dict[str, Any]:
    """Load cli-first-tool-schemas.json (source of truth)."""
    with open(path) as f:
        return json.load(f)


def load_tool_schema_yaml(path: Path) -> Dict[str, Any]:
    """Load tool_schema.yaml (evaluator config)."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
        return data.get("tests", [])


def extract_tools_from_json_schema(schemas: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract tool definitions from cli-first-tool-schemas.json.

    Returns:
        Dict mapping tool_name -> {agent, tool, required_params, optional_params, all_params}
    """
    tools = {}

    for item in schemas.get("tools", []):
        if not isinstance(item, dict):
            continue

        agent = str(item.get("agent", "")).strip()
        tool = str(item.get("tool", "")).strip()
        if not agent or not tool:
            continue

        arguments = item.get("arguments", []) or []
        required = [arg["name"] for arg in arguments if isinstance(arg, dict) and arg.get("required")]
        all_params = [arg["name"] for arg in arguments if isinstance(arg, dict) and arg.get("name")]
        optional = [name for name in all_params if name not in required]

        tool_name = f"{agent}_{tool}"
        tools[tool_name] = {
            "agent": agent,
            "tool": tool,
            "description": item.get("description", f"{tool} operation"),
            "command": item.get("command", ""),
            "usage": item.get("usage", ""),
            "required": required,
            "optional": optional,
            "all_params": all_params,
            "param_details": {
                arg["name"]: {
                    "type": arg.get("type", "any"),
                    "description": arg.get("description", ""),
                    "flag": arg.get("flag"),
                    "positional": bool(arg.get("positional")),
                    "required": bool(arg.get("required")),
                }
                for arg in arguments
                if isinstance(arg, dict) and arg.get("name")
            }
        }

    return tools


def compare_with_yaml(json_tools: Dict, yaml_config: Dict) -> List[str]:
    """Compare JSON schema tools with YAML config and return differences."""
    differences = []

    yaml_tools = yaml_config.get("tools", {})

    # Group JSON tools by agent
    by_agent: Dict[str, List] = {}
    for tool_name, info in json_tools.items():
        agent = info["agent"]
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(info)

    # Compare each agent
    for agent, tools in sorted(by_agent.items()):
        yaml_agent_tools = yaml_tools.get(agent, [])
        yaml_tool_names = {t["name"] for t in yaml_agent_tools}
        yaml_tool_map = {t["name"]: t for t in yaml_agent_tools}

        for tool_info in tools:
            tool = tool_info["tool"]
            full_name = f"{agent}_{tool}"

            if tool not in yaml_tool_names:
                differences.append(f"MISSING in YAML: {full_name}")
                continue

            yaml_tool = yaml_tool_map[tool]
            yaml_required = set(yaml_tool.get("params", {}).get("required", []))
            yaml_optional = set(yaml_tool.get("params", {}).get("optional", []))

            json_required = set(tool_info["required"])
            json_optional = set(tool_info["optional"])

            if yaml_required != json_required:
                differences.append(
                    f"MISMATCH {full_name} required: "
                    f"YAML={sorted(yaml_required)}, JSON={sorted(json_required)}"
                )

            if yaml_optional != json_optional:
                differences.append(
                    f"MISMATCH {full_name} optional: "
                    f"YAML={sorted(yaml_optional)}, JSON={sorted(json_optional)}"
                )

    # Check for tools in YAML but not in JSON
    for agent, yaml_agent_tools in yaml_tools.items():
        for yaml_tool in yaml_agent_tools:
            full_name = f"{agent}_{yaml_tool['name']}"
            if full_name not in json_tools:
                differences.append(f"EXTRA in YAML (not in JSON): {full_name}")

    return differences


def generate_corrected_yaml(json_tools: Dict, original_yaml: Dict) -> str:
    """Generate corrected tool_schema.yaml content."""

    output = {
        "tool_format": {
            "wrapper": "useTools",
            "wrapper_structure": {
                "top_level_fields": {
                    "required": ["workspaceId", "sessionId", "memory", "goal", "tool"],
                    "optional": ["constraints", "strategy"],
                },
                "cli": {
                    "field": "tool",
                    "separator": ",",
                    "quoting": "shell",
                },
                "batch": {
                    "enabled": True,
                    "default_strategy": "serial",
                    "strategy_field": "strategy",
                    "valid_strategies": ["serial", "parallel"],
                },
            },
        },
        "tools": {},
        "validation": {
            "wrapper_required": True,
            "allow_unknown_tools": False,
            "allow_extra_params": True,
            "path_patterns": original_yaml.get("validation", {}).get("path_patterns", {
                "valid": r"^[\w\-./]+$",
                "invalid_chars": ["<", ">", ":", "\"", "|", "?", "*"],
            }),
        },
    }

    # Group tools by agent
    by_agent: Dict[str, List] = {}
    for tool_name, info in json_tools.items():
        agent = info["agent"]
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(info)

    # Build tools section
    for agent in sorted(by_agent.keys()):
        output["tools"][agent] = []
        for tool_info in sorted(by_agent[agent], key=lambda x: x["tool"]):
            tool_entry = {
                "name": tool_info["tool"],
                "description": tool_info["description"],
                "command": tool_info.get("command", ""),
                "usage": tool_info.get("usage", ""),
                "params": {
                    "required": tool_info["required"],
                    "optional": tool_info["optional"],
                }
            }
            output["tools"][agent].append(tool_entry)

    # Convert to YAML with nice formatting
    return yaml.dump(output, default_flow_style=False, sort_keys=False, allow_unicode=True)


def audit_test_cases(test_cases: List[Dict], json_tools: Dict) -> List[Dict]:
    """
    Audit test cases for param issues.

    Returns list of issues found.
    """
    issues = []

    # Common param names that might appear in questions
    param_patterns = [
        (r'\bname\b', 'name'),
        (r'\bdescription\b', 'description'),
        (r'\bprompt\b', 'prompt'),
        (r'\bsystemPrompt\b', 'systemPrompt'),
        (r'\bmodel\b', 'model'),
        (r'\btemperature\b', 'temperature'),
        (r'\bpath\b', 'path'),
        (r'\bnewPath\b', 'newPath'),
        (r'\bfilePath\b', 'filePath'),
        (r'\bcontent\b', 'content'),
        (r'\bquery\b', 'query'),
        (r'\blimit\b', 'limit'),
        (r'\boffset\b', 'offset'),
        (r'\bkey\b', 'key'),
        (r'\bvalue\b', 'value'),
        (r'\btags\b', 'tags'),
        (r'\bagentId\b', 'agentId'),
        (r'\btask\b', 'task'),
        (r'\btools\b', 'tools'),
    ]

    for test in test_cases:
        test_id = test.get("id", "unknown")
        question = test.get("question", "")
        expected_tools = test.get("expected_tools", [])

        # Get the expected tool's valid params
        valid_params: Set[str] = set()
        for tool_name in expected_tools:
            if tool_name in json_tools:
                valid_params.update(json_tools[tool_name]["all_params"])

        if not valid_params:
            continue

        # Check question for param mentions
        mentioned_params = set()
        for pattern, param_name in param_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                mentioned_params.add(param_name)

        # Find params mentioned but not valid
        invalid_mentions = mentioned_params - valid_params

        # Also check for common issues
        test_issues = []

        if invalid_mentions:
            test_issues.append(f"Question mentions non-existent params: {sorted(invalid_mentions)}")

        # Check if question asks for model/temperature (common issue)
        if "model" in mentioned_params and "model" not in valid_params:
            test_issues.append("Question asks for 'model' but tool doesn't support it")
        if "temperature" in mentioned_params and "temperature" not in valid_params:
            test_issues.append("Question asks for 'temperature' but tool doesn't support it")
        if "systemPrompt" in mentioned_params and "systemPrompt" not in valid_params:
            if "prompt" in valid_params:
                test_issues.append("Question uses 'systemPrompt' but tool uses 'prompt'")

        if test_issues:
            issues.append({
                "test_id": test_id,
                "expected_tools": expected_tools,
                "valid_params": sorted(valid_params),
                "issues": test_issues,
                "question_preview": question[:100] + "..." if len(question) > 100 else question,
            })

    return issues


def main():
    repo_root = Path(__file__).parent.parent

    # Paths
    json_schema_path = repo_root / "cli-first-tool-schemas.json"
    yaml_schema_path = repo_root / "Evaluator" / "config" / "tool_schema.yaml"
    tool_prompts_path = repo_root / "Evaluator" / "config" / "scenarios" / "tool_prompts.yaml"
    output_yaml_path = repo_root / "Evaluator" / "config" / "tool_schema_corrected.yaml"

    print("=" * 70)
    print("TOOL SCHEMA AUDIT")
    print("=" * 70)
    print(f"\nSource of truth: {json_schema_path}")
    print(f"Evaluator config: {yaml_schema_path}")
    print(f"Test cases: {tool_prompts_path}")

    # Load data
    print("\n[1/4] Loading schemas...")
    json_schemas = load_tool_schemas(json_schema_path)
    yaml_config = load_tool_schema_yaml(yaml_schema_path)
    test_cases = load_test_cases(tool_prompts_path)

    # Extract tools from JSON
    print("[2/4] Parsing cli-first-tool-schemas.json...")
    json_tools = extract_tools_from_json_schema(json_schemas)
    print(f"      Found {len(json_tools)} tools")

    # Compare with YAML
    print("\n[3/4] Comparing with tool_schema.yaml...")
    differences = compare_with_yaml(json_tools, yaml_config)

    if differences:
        print(f"\n{'=' * 70}")
        print(f"DIFFERENCES FOUND: {len(differences)}")
        print("=" * 70)
        for diff in differences:
            print(f"  - {diff}")
    else:
        print("      No differences found!")

    # Audit test cases
    print("\n[4/4] Auditing test cases...")
    test_issues = audit_test_cases(test_cases, json_tools)

    if test_issues:
        print(f"\n{'=' * 70}")
        print(f"TEST CASE ISSUES: {len(test_issues)}")
        print("=" * 70)
        for issue in test_issues:
            print(f"\n  Test: {issue['test_id']}")
            print(f"  Tools: {issue['expected_tools']}")
            print(f"  Valid params: {issue['valid_params']}")
            for i in issue['issues']:
                print(f"    -> {i}")
    else:
        print("      No test case issues found!")

    # Generate corrected YAML
    print(f"\n{'=' * 70}")
    print("GENERATING CORRECTED tool_schema.yaml")
    print("=" * 70)
    corrected_yaml = generate_corrected_yaml(json_tools, yaml_config)

    with open(output_yaml_path, 'w') as f:
        f.write("# Tool Schema Configuration (AUTO-GENERATED from cli-first-tool-schemas.json)\n")
        f.write("# Source of truth: cli-first-tool-schemas.json\n")
        f.write("# Do not edit manually - regenerate with: python tools/audit_tool_schemas.py\n\n")
        f.write(corrected_yaml)

    print(f"  Written to: {output_yaml_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tools in JSON schema: {len(json_tools)}")
    print(f"  Schema differences: {len(differences)}")
    print(f"  Test case issues: {len(test_issues)}")
    print(f"\nNext steps:")
    print(f"  1. Review {output_yaml_path}")
    print(f"  2. Replace tool_schema.yaml with corrected version")
    print(f"  3. Fix test case questions that reference invalid params")


if __name__ == "__main__":
    main()
