import json
import os

# Path to a sample file
file_path = r"f:\Code\Toolset-Training\Datasets\tools_datasets\thinking\vaultManager\tools_v1.5.jsonl"

def demonstrate():
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        data = json.loads(line)
    
    print("--- Original Assistant Message ---")
    assistant_msg = next(msg for msg in data['conversations'] if msg['role'] == 'assistant')
    print(json.dumps(assistant_msg, indent=2))
    
    # Transform
    if 'tool_calls' in assistant_msg:
        for tool_call in assistant_msg['tool_calls']:
            if 'function' in tool_call and 'arguments' in tool_call['function']:
                args = json.loads(tool_call['function']['arguments'])
                if 'context' in args:
                    # Flatten context
                    context_obj = args.pop('context')
                    if 'sessionId' in context_obj:
                        args['sessionId'] = context_obj['sessionId']
                    if 'workspaceId' in context_obj:
                        args['workspaceId'] = context_obj['workspaceId']
                
                tool_call['function']['arguments'] = json.dumps(args)

    print("\n--- Transformed Assistant Message ---")
    print(json.dumps(assistant_msg, indent=2))

if __name__ == "__main__":
    demonstrate()
