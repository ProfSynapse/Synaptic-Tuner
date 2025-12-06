# Dataset Editor

A simple local frontend for viewing and editing JSONL datasets.

## How to Run

1. Open a terminal in the workspace root.
2. Run the PowerShell script:
   ```powershell
   .\run_editor.ps1
   ```
   (This will automatically install `streamlit` if it's missing).

## Features

- **File Selection**: Choose any `.jsonl` file from the `Datasets` directory.
- **Search**: Filter examples by content (case-insensitive substring search).
- **Thinking Block Editor**: Automatically detects `<thinking>` tags and provides a form to edit Goal, Memory, Plan, etc.
- **Tool Call Editor**: Breaks down tool calls into individual editable fields (ID, Type, Function Name, Arguments).
- **Recursive Field Editor**: Automatically generates forms for nested JSON fields in arguments and other properties.
- **Auto-Save**: Changes are saved to the file when you click "Save Changes".

## Requirements

- Python 3.x
- Streamlit (`pip install streamlit`)
