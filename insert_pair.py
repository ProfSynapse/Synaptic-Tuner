import json
import subprocess

# TRUE example (desirable)
true_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Remove agent_backup_writer since we're using the main one now. ID: agent_7777777777"
        },
        {
            "role": "assistant",
            "content": 'tool_call: agentManager_deleteAgent\narguments: {"context": {"sessionId": "session_1731101500031_o1p2q3r4s", "workspaceId": "ws_1731101500031_t5u6v7w8x", "sessionDescription": "Delete backup agent", "sessionMemory": "Main agent working well", "toolContext": "Backup agent redundant, cleaning up", "primaryGoal": "Remove backup agent", "subgoal": "Delete by ID"}, "id": "agent_7777777777"}\n\nResult: {"success": true, "deletedAgentId": "agent_7777777777"}\n\nRemoved the backup writer agent. You\'re all set with the main one.'
        }
    ],
    "label": True
}

# FALSE example (undesirable - has intentional errors)
false_example = {
    "conversations": [
        {
            "role": "user",
            "content": "Delete my image generator agent with ID agent_img_999."
        },
        {
            "role": "assistant",
            "content": 'tool_call: agentManager_removeAgent\narguments: {"context": {"sessionId": "session_1731101500032_y9z0a1b2c", "workspaceId": "ws_1731101500032_d3e4f5g6h", "sessionDescription": "Remove image agent", "sessionMemory": "", "toolContext": "Using wrong tool name", "primaryGoal": "Delete agent", "subgoal": "Call non-existent tool"}, "id": "agent_img_999"}\n\nResult: {"success": false, "error": "Tool \'agentManager_removeAgent\' not found. Did you mean \'agentManager_deleteAgent\'?", "errorCode": "TOOL_NOT_FOUND"}\n\nI used the wrong tool name. Let me use deleteAgent instead.'
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
