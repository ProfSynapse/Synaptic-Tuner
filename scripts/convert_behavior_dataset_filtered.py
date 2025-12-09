import json
import os
import random

# Configuration
INPUT_ROOT = r"f:\Code\Toolset-Training\Datasets\behavior_datasets\non_thinking"
OUTPUT_ROOT = r"f:\Code\Toolset-Training\Datasets\behavior_datasets\thinking"

RISKY_KEYWORDS = ["delete", "remove", "move", "edit", "overwrite"]
COMPLEX_KEYWORDS = ["executePrompt", "Batch"]

def is_risky(tool_name):
    for keyword in RISKY_KEYWORDS:
        if keyword.lower() in tool_name.lower():
            return True
    return False

def is_complex(tool_name):
    for keyword in COMPLEX_KEYWORDS:
        if keyword.lower() in tool_name.lower():
            return True
    return False

def get_confidence(risky):
    if risky:
        return round(random.uniform(0.7, 0.9), 2)
    else:
        return round(random.uniform(0.9, 1.0), 2)

def convert_line(line):
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    # FILTER: Only process label=true or missing label
    if "label" in data and data["label"] is False:
        return None

    conversations = data.get("conversations", [])
    if not conversations:
        return None

    # 1. Update System Prompt
    system_message = next((c for c in conversations if c["role"] == "system"), None)
    if system_message:
        content = system_message["content"]
        if "<session_context>" in content:
            start = content.find("<session_context>")
            end = content.find("</session_context>") + len("</session_context>")
            new_content = content[start:end]
            
            if "<available_workspaces>" in content:
                ws_start = content.find("<available_workspaces>")
                ws_end = content.find("</available_workspaces>") + len("</available_workspaces>")
                new_content += "\n" + content[ws_start:ws_end]
            
            if "<available_agents>" in content:
                ag_start = content.find("<available_agents>")
                ag_end = content.find("</available_agents>") + len("</available_agents>")
                new_content += "\n" + content[ag_start:ag_end]

            system_message["content"] = new_content

    # 2. Find the Assistant Message with Tool Calls
    assistant_message = next((c for c in conversations if c["role"] == "assistant" and c.get("tool_calls")), None)
    
    # Handle text-based tool calls (convert to structured)
    if not assistant_message:
        text_tool_msg = next((c for c in conversations if c["role"] == "assistant" and "tool_call:" in c.get("content", "")), None)
        if text_tool_msg:
            content = text_tool_msg["content"]
            lines = content.split('\n')
            tool_name = None
            args_str = None
            
            for line in lines:
                if line.startswith("tool_call:"):
                    tool_name = line.split(":", 1)[1].strip()
                elif line.startswith("arguments:"):
                    args_str = line.split(":", 1)[1].strip()
            
            if tool_name and args_str:
                try:
                    # Validate JSON
                    json.loads(args_str)
                    
                    # Convert to structured
                    text_tool_msg["tool_calls"] = [{
                        "id": f"call_{random.randint(100000, 999999)}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": args_str
                        }
                    }]
                    text_tool_msg["content"] = "" # Clear text content as it's now structured
                    assistant_message = text_tool_msg
                except json.JSONDecodeError:
                    pass

    if assistant_message:
        tool_calls = assistant_message.get("tool_calls", [])
        if tool_calls:
            first_tool = tool_calls[0]["function"]
            tool_name = first_tool["name"]
            try:
                args = json.loads(first_tool["arguments"])
                context = args.get("context", {})
                
                # Extract old thinking fields
                primary_goal = context.get("primaryGoal", "Unknown goal")
                subgoal = context.get("subgoal", "Unknown subgoal")
                session_memory = context.get("sessionMemory", "")
                tool_context = context.get("toolContext", "")
                
                # Extract IDs to keep
                session_id = context.get("sessionId")
                workspace_id = context.get("workspaceId")

                # Heuristics
                risky = is_risky(tool_name)
                complex_task = is_complex(tool_name)
                confidence = get_confidence(risky)
                
                # Construct Plan & Requirements
                plan = [subgoal, tool_context]
                requirements = [subgoal, tool_context]

                # New Thinking Block
                new_thinking = {
                    "goal": primary_goal,
                    "memory": session_memory,
                    "requirements": requirements,
                    "assessment": {
                        "complex": complex_task,
                        "risky": risky
                    },
                    "confidence": confidence,
                    "plan": plan
                }
                
                # Insert <thinking> block
                thinking_str = f"<thinking>\n{json.dumps(new_thinking, indent=2)}\n</thinking>"
                
                if assistant_message.get("content"):
                     assistant_message["content"] = thinking_str + "\n" + assistant_message["content"]
                else:
                     assistant_message["content"] = thinking_str

                # FLATTEN ARGUMENTS
                # Remove 'context' object
                if "context" in args:
                    del args["context"]
                
                # Add IDs to top level
                if session_id:
                    args["sessionId"] = session_id
                if workspace_id:
                    args["workspaceId"] = workspace_id
                
                # Update tool arguments
                first_tool["arguments"] = json.dumps(args)

            except json.JSONDecodeError:
                pass

    return json.dumps(data)

def process_file(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            new_line = convert_line(line)
            if new_line:
                outfile.write(new_line + "\n")

def main():
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".jsonl"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, INPUT_ROOT)
                output_path = os.path.join(OUTPUT_ROOT, rel_path)
                process_file(input_path, output_path)

if __name__ == "__main__":
    main()
