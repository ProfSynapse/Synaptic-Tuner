#!/usr/bin/env python3
"""
Script to insert False examples between consecutive True-True pairs in KTO dataset.
Generates contextually relevant False examples based on nearby True examples.
"""

import json
import random
from typing import List, Dict, Tuple

# Error pattern templates for generating False examples
FALSE_TEMPLATES = [
    {
        "type": "MISSING_PARAM",
        "user_patterns": [
            "Delete the file at '{path}'.",
            "Open the note at '{path}'.",
            "Move '{path}' to a new location.",
            "Create a copy of '{path}'.",
        ],
        "tools": ["vaultManager_deleteNote", "vaultManager_openNote", "vaultManager_moveNote", "vaultManager_duplicateNote"],
        "error_template": "Missing required parameter '{param}'. Cannot {action} without specifying {param_desc}.",
        "missing_params": ["path", "targetPath", "newPath", "content", "filePath"]
    },
    {
        "type": "WRONG_TOOL",
        "user_patterns": [
            "Show me what's in '{file}'.",
            "Create a new folder called '{folder}'.",
            "Rename the folder '{old}' to '{new}'.",
        ],
        "wrong_tools": [
            ("vaultManager_openNote", "contentManager_readContent", "open", "read"),
            ("contentManager_createContent", "vaultManager_createFolder", "create file", "create folder"),
            ("vaultManager_moveNote", "vaultManager_editFolder", "move note", "rename folder"),
        ]
    },
    {
        "type": "INVALID_VALUE",
        "user_patterns": [
            "Open '{file}' in a new tab.",
            "Search for files matching '{pattern}'.",
            "List files in '{folder}' sorted by date.",
        ],
        "tools": ["vaultManager_openNote", "vaultLibrarian_searchContent", "vaultManager_listDirectory"],
        "invalid_params": [
            ("mode", "tab", ["current", "window", "split"]),
            ("sortBy", "date_created", ["name", "modified", "size"]),
            ("filter", "recent", ["*.md", "*.txt", "pattern*"]),
        ]
    },
    {
        "type": "FILE_NOT_FOUND",
        "user_patterns": [
            "Open the file at '{path}'.",
            "Read '{file}'.",
            "Delete '{path}'.",
            "Move '{path}' to Archive.",
        ],
        "tools": ["vaultManager_openNote", "contentManager_readContent", "vaultManager_deleteNote", "vaultManager_moveNote"],
        "error_template": "File not found: {path}. The specified file does not exist."
    },
]

def generate_session_ids() -> Tuple[str, str]:
    """Generate valid sessionId and workspaceId"""
    import time
    timestamp = int(time.time() * 1000)
    session_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))
    workspace_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))
    return f"session_{timestamp}_{session_suffix}", f"ws_{timestamp}_{workspace_suffix}"

def generate_false_example(nearby_true_content: str) -> Dict:
    """Generate a contextually relevant False example based on nearby True examples"""
    
    # Select random error type
    template = random.choice(FALSE_TEMPLATES)
    error_type = template["type"]
    
    session_id, workspace_id = generate_session_ids()
    
    # Generate based on error type
    if error_type == "MISSING_PARAM":
        user_content = random.choice(template["user_patterns"]).format(path="Documents/report.md")
        tool = random.choice(template["tools"])
        missing_param = random.choice(template["missing_params"])
        
        # Build tool call without the missing parameter
        tool_args = {
            "context": {
                "sessionId": session_id,
                "workspaceId": workspace_id,
                "sessionDescription": "Operation with missing parameter",
                "sessionMemory": "User requested operation",
                "toolContext": f"Missing {missing_param} parameter",
                "primaryGoal": "Execute user request",
                "subgoal": f"Call {tool} without required parameter"
            }
        }
        
        # Add some parameters but intentionally omit the missing one
        if "path" not in tool_args and missing_param != "path":
            tool_args["path"] = "Documents/report.md"
        if "filePath" not in tool_args and missing_param != "filePath":
            tool_args["filePath"] = "Documents/report.md"
            
        error_msg = f"Missing required parameter '{missing_param}'. Cannot perform operation without specifying {missing_param}."
        
        assistant_content = f"tool_call: {tool}\\narguments: {json.dumps(tool_args)}\\n\\nResult: {{\"success\": false, \"error\": \"{error_msg}\", \"errorCode\": \"MISSING_REQUIRED_PARAM\", \"missingParams\": [\"{missing_param}\"]}}\\n\\nI need to specify the {missing_param} parameter."
        
    elif error_type == "FILE_NOT_FOUND":
        user_content = random.choice(template["user_patterns"]).format(
            path="Archive/old-file.md", 
            file="Archive/old-file.md"
        )
        tool = random.choice(template["tools"])
        
        tool_args = {
            "context": {
                "sessionId": session_id,
                "workspaceId": workspace_id,
                "sessionDescription": "Access non-existent file",
                "sessionMemory": "User trying to access file",
                "toolContext": "File does not exist",
                "primaryGoal": "Access requested file",
                "subgoal": f"Attempt to use {tool} on missing file"
            },
            "path" if "path" in tool else "filePath": "Archive/old-file.md"
        }
        
        error_msg = "File not found: Archive/old-file.md. The specified file does not exist."
        
        assistant_content = f"tool_call: {tool}\\narguments: {json.dumps(tool_args)}\\n\\nResult: {{\"success\": false, \"error\": \"{error_msg}\", \"errorCode\": \"FILE_NOT_FOUND\"}}\\n\\nThe file doesn't exist at that path."
        
    else:  # Default to simple wrong tool error
        user_content = "Show me what's in my Notes.md file."
        tool_args = {
            "context": {
                "sessionId": session_id,
                "workspaceId": workspace_id,
                "sessionDescription": "Read file with wrong tool",
                "sessionMemory": "User wants to see file contents",
                "toolContext": "Using openNote instead of readContent",
                "primaryGoal": "Show Notes.md content",
                "subgoal": "Opening instead of reading"
            },
            "path": "Notes.md",
            "mode": "current"
        }
        
        assistant_content = "tool_call: vaultManager_openNote\\narguments: " + json.dumps(tool_args) + "\\n\\nResult: {\"success\": true, \"data\": {\"path\": \"Notes.md\", \"opened\": true}}\\n\\nI've opened the file, but you asked to see the contents. I should have used contentManager_readContent."
    
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "label": False
    }

def main():
    file_path = "syngen_toolset_v1.0.0_copilot.jsonl"
    
    # Read all lines
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse and find consecutive True-True pairs
    entries = [json.loads(line) for line in lines]
    consecutive_pairs = []
    
    for i in range(len(entries) - 1):
        if entries[i].get('label') == True and entries[i+1].get('label') == True:
            consecutive_pairs.append(i)
    
    print(f"Found {len(consecutive_pairs)} consecutive True-True pairs")
    print(f"Processing first 10 pairs as test...")
    
    # Process pairs in reverse order (to avoid index shifting)
    for idx in reversed(consecutive_pairs[:10]):  # Process first 10 as test
        # Generate False example
        nearby_content = entries[idx]["conversations"][0]["content"]
        false_example = generate_false_example(nearby_content)
        
        # Insert after the first True
        entries.insert(idx + 1, false_example)
        print(f"Inserted False example after line {idx + 1}")
    
    # Write back
    output_path = "syngen_toolset_v1.0.0_copilot_fixed.jsonl"
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\\n')
    
    print(f"\\nWrote fixed dataset to {output_path}")
    print(f"Total lines: {len(entries)}")
    
    # Verify
    consecutive_after = 0
    for i in range(len(entries) - 1):
        if entries[i].get('label') == True and entries[i+1].get('label') == True:
            consecutive_after += 1
    
    print(f"Consecutive True-True pairs remaining: {consecutive_after}")

if __name__ == "__main__":
    main()
