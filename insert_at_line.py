#!/usr/bin/env python3
import json

# ============ EDIT THIS FULL CONVERSATION BEFORE RUNNING ============
line_number = 108

new_example = {
    "conversations": [
        {
            "role": "user", 
            "content": "Rename all the files in my Downloads folder to have today's date as a prefix."
        },
        {
            "role": "assistant", 
            "content": 'tool_call: vaultManager_moveNote\narguments: {"context": {"sessionId": "session_1731037200000_k2m3n4p5q", "workspaceId": "ws_1731037200000_r6s7t8u9v", "sessionDescription": "Rename files - wrong approach", "sessionMemory": "", "toolContext": "Should list files first before renaming", "primaryGoal": "Rename files", "subgoal": "Rename without seeing what files exist"}, "path": "file.pdf", "newPath": "2025-11-07-file.pdf"}\n\nResult: {"success": false, "error": "File file.pdf not found"}\n\nI tried to rename but I don\'t know what files exist. Let me search for the files first.'
        }
    ],
    "label": False
}
# =====================================================================

with open("syngen_toolset_v1.0.0_copilot.jsonl", "r") as f:
    lines = f.readlines()

lines.insert(line_number - 1, json.dumps(new_example) + '\n')

with open("syngen_toolset_v1.0.0_copilot.jsonl", "w") as f:
    f.writelines(lines)

# Count pairs
pairs = sum(1 for i in range(len(lines)-1) if json.loads(lines[i]).get("label") is True and json.loads(lines[i+1]).get("label") is True)
fixed = 142 - pairs
print(f"{len(lines)} lines | {pairs} pairs | {fixed} fixed ({fixed/142*100:.1f}%)")
