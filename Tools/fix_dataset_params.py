#!/usr/bin/env python3
"""
Fix invalid parameter names in tool datasets.

This script:
1. Loads tool-schemas.json (source of truth)
2. Scans source dataset files for invalid parameters
3. Applies fixes (rename, remove, or flag for manual review)
4. Creates new versioned output files
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import argparse
import shutil


# Parameter rename mappings: {tool_name: {old_param: new_param}}
PARAM_RENAMES = {
    "memoryManager_loadWorkspace": {
        "id": "workspaceId",
    },
    "memoryManager_createWorkspace": {
        "description": "purpose",
    },
    "memoryManager_updateWorkspace": {
        "description": "purpose",
    },
    "vaultManager_listDirectory": {
        "includeFolders": "includeFiles",
    },
    "agentManager_getAgent": {
        "agentName": "name",
        "agentId": "id",
    },
    "agentManager_updateAgent": {
        "systemPrompt": "prompt",
    },
    "memoryManager_createSession": {
        "sessionName": "name",
    },
    "memoryManager_loadSession": {
        "sessionName": "sessionId",  # loadSession requires sessionId
        "name": "sessionId",
    },
    "memoryManager_loadState": {
        "stateId": "stateId",  # Already correct but ensure it passes through
    },
}

# Parameters to remove entirely (invalid and no good mapping)
PARAMS_TO_REMOVE = {
    "agentManager_executePrompts": ["agent", "provider", "model", "temperature", "maxTokens",
                                     "agentName", "returnContent", "filepaths", "action", "appendTarget"],
    "agentManager_createAgent": ["model"],
    "agentManager_listAgents": ["filters"],
    "agentManager_listModels": ["provider"],
    "agentManager_updateAgent": ["updates", "enabled", "model"],
    "memoryManager_listSessions": ["timeframe", "dateFrom", "dateTo", "filters", "search",
                                    "createdAfter", "sortBy", "hasErrors"],
    "memoryManager_listStates": ["order", "tags", "filter", "sessionId", "timeRange"],
    "memoryManager_listWorkspaces": ["order", "sortBy"],
    "memoryManager_createState": ["includeSummary", "includeFileContents", "targetSessionId",
                                   "maxFiles", "reason", "content", "files", "maxTraces",
                                   "id", "data", "stateId"],
    "memoryManager_updateSession": ["goal", "newName", "status", "memory"],
    "memoryManager_updateState": ["description", "addTags", "status", "tags", "memory", "key", "value"],
    "memoryManager_updateWorkspace": ["fieldPath", "newValue"],
    "memoryManager_loadSession": ["createContinuationSession", "sessionDescription", "description", "sessionGoal"],
    "memoryManager_loadState": ["restorationGoal", "sessionDescription", "continueExistingSession", "sessionName"],
    "vaultLibrarian_searchContent": ["caseSensitive", "fileTypes", "wholeWord"],
    "contentManager_replaceContent": ["path", "content"],
    "memoryManager_createSession": ["sessionId", "id"],
}


def load_tool_schemas(path: Path) -> Dict[str, set]:
    """Load tool-schemas.json and extract valid params for each tool."""
    with open(path) as f:
        schemas = json.load(f)

    tool_params = {}
    for tool_name, schema in schemas.items():
        try:
            params_schema = schema.get("properties", {}).get("calls", {}).get("items", {}).get("properties", {}).get("params", {})
            properties = params_schema.get("properties", {})
            tool_params[tool_name] = set(properties.keys())
        except:
            pass

    return tool_params


def fix_tool_call(tool_name: str, params: Dict, valid_params: set) -> Tuple[Dict, List[str]]:
    """
    Fix parameters in a tool call.

    Returns:
        (fixed_params, list_of_changes)
    """
    changes = []
    fixed = {}

    for param, value in params.items():
        # Check if this param should be renamed
        if tool_name in PARAM_RENAMES and param in PARAM_RENAMES[tool_name]:
            new_param = PARAM_RENAMES[tool_name][param]
            if new_param in valid_params:
                fixed[new_param] = value
                changes.append(f"renamed '{param}' -> '{new_param}'")
                continue

        # Check if this param should be removed
        if tool_name in PARAMS_TO_REMOVE and param in PARAMS_TO_REMOVE[tool_name]:
            changes.append(f"removed invalid '{param}'")
            continue

        # Keep valid params
        if param in valid_params:
            fixed[param] = value
        else:
            # Unknown invalid param - remove it but log
            changes.append(f"removed unknown '{param}'")

    return fixed, changes


def process_example(example: Dict, tool_params: Dict) -> Tuple[Dict, List[str], bool]:
    """
    Process a single example, fixing tool call parameters.

    Returns:
        (fixed_example, list_of_all_changes, has_unknown_tool)
    """
    all_changes = []
    has_unknown_tool = False
    fixed_example = json.loads(json.dumps(example))  # Deep copy

    for msg in fixed_example.get("conversations", []):
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except:
                continue

            # Handle useTools wrapper
            if name == "useTools" and "calls" in args:
                for call in args.get("calls", []):
                    agent = call.get("agent", "")
                    tool = call.get("tool", "")
                    params = call.get("params", {})

                    full_name = f"{agent}_{tool}"

                    if full_name not in tool_params:
                        has_unknown_tool = True
                        all_changes.append(f"UNKNOWN TOOL: {full_name}")
                        continue

                    valid = tool_params[full_name]
                    fixed_params, changes = fix_tool_call(full_name, params, valid)

                    if changes:
                        call["params"] = fixed_params
                        all_changes.extend([f"{full_name}: {c}" for c in changes])

                # Update the arguments string
                func["arguments"] = json.dumps(args, ensure_ascii=False)

    return fixed_example, all_changes, has_unknown_tool


def increment_version(version_str: str) -> str:
    """Increment version number: 1.4 -> 1.5, 1.4_passed_only -> 1.5"""
    # Remove suffix like _passed_only
    clean = version_str.split("_")[0]
    parts = clean.split(".")
    if len(parts) == 2:
        major, minor = int(parts[0]), int(parts[1])
        return f"{major}.{minor + 1}"
    return f"{clean}.1"


def get_current_version(file_path: Path) -> str:
    """Extract version from filename like tools_v1.4.jsonl -> 1.4"""
    name = file_path.stem  # tools_v1.4
    match = re.search(r'v(\d+\.\d+)', name)
    if match:
        return match.group(1)
    return "1.0"


def main():
    parser = argparse.ArgumentParser(description="Fix invalid parameters in tool datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without writing")
    parser.add_argument("--agent", help="Only process specific agent (e.g., memoryManager)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all changes")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    schema_path = repo_root / "tool-schemas.json"
    datasets_dir = repo_root / "Datasets" / "tools_datasets" / "non_thinking"

    print("=" * 70)
    print("DATASET PARAMETER FIXER")
    print("=" * 70)

    # Load schemas
    print(f"\nLoading tool schemas from {schema_path.name}...")
    tool_params = load_tool_schemas(schema_path)
    print(f"Loaded {len(tool_params)} tool definitions")

    # Find source files
    agents = ["agentManager", "contentManager", "memoryManager", "vaultLibrarian", "vaultManager"]
    if args.agent:
        agents = [args.agent]

    total_fixed = 0
    total_removed = 0

    for agent in agents:
        agent_dir = datasets_dir / agent
        if not agent_dir.exists():
            print(f"\nWARNING: {agent_dir} not found, skipping")
            continue

        # Find latest version file (prefer _passed_only)
        passed_only = list(agent_dir.glob("*_passed_only.jsonl"))
        if passed_only:
            source_file = max(passed_only, key=lambda f: get_current_version(f))
        else:
            regular = [f for f in agent_dir.glob("tools_v*.jsonl")
                      if "failed" not in f.name and "review" not in f.name and "test" not in f.name and "_full" not in f.name]
            if not regular:
                print(f"\nWARNING: No source file found for {agent}, skipping")
                continue
            source_file = max(regular, key=lambda f: get_current_version(f))

        current_version = get_current_version(source_file)
        new_version = increment_version(current_version)
        output_file = agent_dir / f"tools_v{new_version}.jsonl"

        print(f"\n{'='*70}")
        print(f"Processing: {agent}")
        print(f"  Source: {source_file.name}")
        print(f"  Output: {output_file.name}")
        print("=" * 70)

        # Process examples
        fixed_examples = []
        stats = defaultdict(int)
        all_example_changes = []

        with open(source_file) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: Line {line_num} invalid JSON, skipping")
                    continue

                fixed, changes, has_unknown = process_example(example, tool_params)

                if has_unknown:
                    stats["unknown_tool"] += 1
                    total_removed += 1
                    continue  # Skip examples with unknown tools

                if changes:
                    stats["fixed"] += 1
                    total_fixed += 1
                    if args.verbose:
                        all_example_changes.append((line_num, changes))
                else:
                    stats["unchanged"] += 1

                fixed_examples.append(fixed)

        # Report
        print(f"\n  Results:")
        print(f"    Fixed: {stats['fixed']} examples")
        print(f"    Unchanged: {stats['unchanged']} examples")
        print(f"    Removed (unknown tools): {stats['unknown_tool']} examples")
        print(f"    Total output: {len(fixed_examples)} examples")

        if args.verbose and all_example_changes:
            print(f"\n  Changes made:")
            for line_num, changes in all_example_changes[:20]:
                print(f"    Line {line_num}:")
                for c in changes:
                    print(f"      - {c}")
            if len(all_example_changes) > 20:
                print(f"    ... and {len(all_example_changes) - 20} more")

        if args.dry_run:
            print(f"\n  [DRY RUN] Would write to: {output_file}")
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                for ex in fixed_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            print(f"\n  Written: {output_file}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples fixed: {total_fixed}")
    print(f"Total examples removed: {total_removed}")

    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"  1. Review the new versioned files")
        print(f"  2. Run: python Datasets/tools/merge_nonthinking_datasets.py")


if __name__ == "__main__":
    main()
