import json
import subprocess

# TRUE example (desirable)
true_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Create Bug Tracker workspace in /Development/Bugs for tracking and resolving software bugs. Current goal is setting up bug triage process. Workflows: bug reporting with reproduction steps and severity, then bug resolution with fix, test, and release."
        },
        {
            "role": "assistant",
            "content": 'tool_call: memoryManager_createWorkspace\narguments: {"context": {"sessionId": "session_1731104000081_u7v8w9x0y", "workspaceId": "ws_1731104000081_z1a2b3c4d", "sessionDescription": "Create bug tracking workspace", "sessionMemory": "User managing software bugs", "toolContext": "Setting up bug management system", "primaryGoal": "Create bug tracker", "subgoal": "Set up triage process"}, "name": "Bug Tracker", "rootFolder": "/Development/Bugs", "purpose": "Track and resolve software bugs through systematic triage and resolution workflows", "currentGoal": "Set up bug triage process and prioritization system", "workflows": [{"name": "Bug Reporting", "when": "New bug discovered", "steps": ["Document reproduction steps", "Assign severity", "Categorize bug type", "Assign to developer"]}, {"name": "Bug Resolution", "when": "Fixing bug", "steps": ["Fix code", "Write tests", "Verify fix", "Release patch"]}]}\n\nResult: {"success": true, "workspaceId": "ws_1731104000081_z1a2b3c4d", "name": "Bug Tracker", "created": "2024-11-10T17:50:00Z"}\n\nCreated Bug Tracker workspace with reporting and resolution workflows.'
        }
    ],
    "label": True
}

# FALSE example (undesirable - has intentional errors)
false_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Create workspace with empty name."
        },
        {
            "role": "assistant",
            "content": 'tool_call: memoryManager_createWorkspace\narguments: {"context": {"sessionId": "session_1731104000082_e5f6g7h8i", "workspaceId": "ws_1731104000082_j9k0l1m2n", "sessionDescription": "Create workspace with empty name", "sessionMemory": "Testing empty name", "toolContext": "Name cannot be empty", "primaryGoal": "Test error", "subgoal": "Empty name"}, "name": "", "rootFolder": "/Test", "purpose": "Testing", "currentGoal": "Test", "workflows": [{"name": "Test", "when": "Testing", "steps": ["Test"]}]}\n\nResult: {"success": false, "error": "Parameter name cannot be empty string.", "errorCode": "INVALID_VALUE"}\n\nWorkspace name must have a value.'
        }
    ],
    "label": False
}

# TRUE example (desirable)
true_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Show my workspaces sorted by name."
        },
        {
            "role": "assistant",
            "content": 'tool_call: memoryManager_listWorkspaces\narguments: {"context": {"sessionId": "session_1731104400089_w1x2y3z4a", "workspaceId": "ws_1731104400089_b5c6d7e8f", "sessionDescription": "List workspaces sorted by name", "sessionMemory": "User wants alphabetical list", "toolContext": "Sort by name in ascending order", "primaryGoal": "List workspaces", "subgoal": "Alphabetical sorting"}, "sortBy": "name", "order": "asc"}\n\nResult: {"success": true, "workspaces": [{"id": "ws_002", "name": "Bug Tracker"}, {"id": "ws_001", "name": "Marketing Campaign"}, {"id": "ws_003", "name": "Newsletter"}], "total": 3}\n\nWorkspaces alphabetically: Bug Tracker, Marketing Campaign, Newsletter.'
        }
    ],
    "label": True
}

# FALSE example (undesirable - has intentional errors)
false_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Sort workspaces by invalid field."
        },
        {
            "role": "assistant",
            "content": 'tool_call: memoryManager_listWorkspaces\narguments: {"context": {"sessionId": "session_1731104400090_g9h0i1j2k", "workspaceId": "ws_1731104400090_l3m4n5o6p", "sessionDescription": "List with invalid sort", "sessionMemory": "User requested invalid sort field", "toolContext": "Using invalid sortBy value", "primaryGoal": "List workspaces", "subgoal": "Invalid sort"}, "sortBy": "popularity"}\n\nResult: {"success": false, "error": "Invalid sortBy value. Must be name, created, or lastAccessed.", "errorCode": "INVALID_VALUE"}\n\nCan only sort by name, created, or lastAccessed.'
        }
    ],
    "label": False
}

# TRUE example (desirable)
true_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Delete the old-drafts folder and everything in it."
        },
        {
            "role": "assistant",
            "content": 'tool_call: vaultManager_deleteFolder\narguments: {"context": {"sessionId": "session_1731104500282_x5y6z7a8b", "workspaceId": "ws_1731104500282_c9d0e1f2g", "sessionDescription": "Deleting folder", "sessionMemory": "User cleaning vault", "toolContext": "Remove old folder", "primaryGoal": "Clean vault", "subgoal": "Delete folder"}, "path": "old-drafts", "recursive": true}\n\nResult: {"success": true, "path": "old-drafts", "deleted": true, "filesRemoved": 12, "recursive": true}\n\nDeleted old-drafts folder and 12 files inside.'
        }
    ],
    "label": True
}

# FALSE example (undesirable - has intentional errors)
false_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Delete folder without path."
        },
        {
            "role": "assistant",
            "content": 'tool_call: vaultManager_deleteFolder\narguments: {"context": {"sessionId": "session_1731104500283_h3i4j5k6l", "workspaceId": "ws_1731104500283_m7n8o9p0q", "sessionDescription": "Missing path", "sessionMemory": "Testing missing param", "toolContext": "path is required", "primaryGoal": "Delete folder", "subgoal": "Missing param"}, "recursive": true}\n\nResult: {"success": false, "error": "Missing required parameter: path", "errorCode": "MISSING_PARAMETER"}\n\nThe path parameter is required to specify which folder to delete.'
        }
    ],
    "label": False
}

# =====================================================================

# Read current file
with open("syngen_toolset_v1.0.1_copilot.jsonl", "r") as f:
    lines = f.readlines()

# Check last line to determine what to add first
parsed = [json.loads(line) for line in lines]
last_label = parsed[-1].get('label')

# Append in correct order
with open("syngen_toolset_v1.0.1_copilot.jsonl", "a") as f:
    if last_label is False:
        # Last was False, so add True then False
        f.write(json.dumps(true_example) + '\n')
        f.write(json.dumps(false_example) + '\n')
    else:
        # Last was True, so add False then True
        f.write(json.dumps(false_example) + '\n')
        f.write(json.dumps(true_example) + '\n')

# Re-read to get updated counts
with open("syngen_toolset_v1.0.1_copilot.jsonl", "r") as f:
    lines = f.readlines()

parsed = [json.loads(line) for line in lines]
true_count = sum(1 for p in parsed if p.get('label') is True)
false_count = sum(1 for p in parsed if p.get('label') is False)
true_true = sum(1 for i in range(len(parsed)-1) if parsed[i].get('label') is True and parsed[i+1].get('label') is True)
false_false = sum(1 for i in range(len(parsed)-1) if parsed[i].get('label') is False and parsed[i+1].get('label') is False)

print(f"\n✅ {len(lines)} lines | True: {true_count} | False: {false_count} | TT: {true_true} | FF: {false_false}")

# Run full validator
result = subprocess.run(
    ["python3", "tools/validate_syngen.py", "syngen_toolset_v1.0.1_copilot.jsonl"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("\n⚠️  VALIDATION FAILED!")
    print("Errors found in True examples:")
    # Show only ERROR lines (not warnings)
    error_lines = [line for line in result.stdout.split('\n') if 'ERROR' in line and 'label=false' not in line.lower()]
    for line in error_lines[:10]:  # Show first 10 errors
        print(line)
    print("\n❌ Stopping due to validation errors. Please fix before continuing.")
    import sys
    sys.exit(1)
else:
    print("✓ Full validation passed")
    print("\n✅ Pair added successfully!")
