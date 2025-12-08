import json
import os
import shutil

ROOT_DIR = r"f:\Code\Toolset-Training\Datasets\tools_datasets\thinking"

def process_file(file_path):
    print(f"Processing {file_path}...")
    
    # Create backup
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    print(f"  Backed up to {backup_path}")
    
    new_lines = []
    modified_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Find assistant message
                if 'conversations' in data:
                    for msg in data['conversations']:
                        if msg['role'] == 'assistant' and 'tool_calls' in msg:
                            for tool_call in msg['tool_calls']:
                                if 'function' in tool_call and 'arguments' in tool_call['function']:
                                    try:
                                        args = json.loads(tool_call['function']['arguments'])
                                        if 'context' in args and isinstance(args['context'], dict):
                                            # Flatten context object
                                            context_obj = args.pop('context')
                                            
                                            # Extract sessionId and workspaceId
                                            if 'sessionId' in context_obj:
                                                args['sessionId'] = context_obj['sessionId']
                                            if 'workspaceId' in context_obj:
                                                args['workspaceId'] = context_obj['workspaceId']
                                            
                                            tool_call['function']['arguments'] = json.dumps(args)
                                            modified_count += 1
                                    except json.JSONDecodeError:
                                        print(f"  Warning: Could not parse arguments JSON on line {line_num + 1}")

                new_lines.append(json.dumps(data))
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse line {line_num + 1}")
                new_lines.append(line) # Keep original line if parse fails

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')
            
    print(f"  Modified {modified_count} tool calls.")

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory {ROOT_DIR} does not exist.")
        return

    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == "__main__":
    main()
