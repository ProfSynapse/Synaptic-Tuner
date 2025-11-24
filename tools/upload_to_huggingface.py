#!/usr/bin/env python3
"""
Upload syngen_toolset_v1.0.0_claude.jsonl to Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Load environment variables from .env
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    print("Error: HF_API_KEY or HUGGINGFACE_TOKEN not found in .env file")
    sys.exit(1)

# Configuration
REPO_ID = "professorsynapse/claudesidian-synthetic-dataset"
DATASET_FILE = "Datasets/syngen_tools_sft_11.24.25.jsonl"
REPO_TYPE = "dataset"  # This is a dataset, not a model

def upload_dataset():
    """Upload the JSONL dataset to Hugging Face Hub"""

    # Verify file exists
    if not Path(DATASET_FILE).exists():
        print(f"Error: {DATASET_FILE} not found")
        sys.exit(1)

    file_size = Path(DATASET_FILE).stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"Dataset file: {DATASET_FILE}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Target repo: {REPO_ID}")
    print()

    try:
        # Initialize API
        api = HfApi(token=HF_TOKEN)

        # Check if repo exists, if not create it
        print("Checking/creating repository on Hugging Face Hub...")
        try:
            repo_url = create_repo(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                exist_ok=True,
                private=False  # Set to True if you want private repo
            )
            print(f"Repository ready: {repo_url}")
        except Exception as e:
            print(f"Note: {e}")

        # Upload the file
        filename = Path(DATASET_FILE).name
        print(f"\nUploading {DATASET_FILE}...")
        info = api.upload_file(
            path_or_fileobj=DATASET_FILE,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=HF_TOKEN
        )
        print(f"✓ File uploaded successfully!")
        print(f"  URL: https://huggingface.co/datasets/{REPO_ID}/blob/main/{filename}")

        # Create README.md if it doesn't exist
        print("\nCreating/updating README.md...")
        readme_content = """# Claudesidian Synthetic Training Dataset

High-quality synthetic training dataset (1,000 examples) for fine-tuning local LLMs to reliably use the Claudesidian-MCP tool suite for Obsidian vault operations.

## Dataset Overview

- **Total Examples**: 1,000
- **Desirable Examples**: 742 (74.2%)
- **Undesirable Examples**: 254 (25.4%)
- **Ratio**: 2.92:1 (near target 3:1)
- **Format**: ChatML (no system message, user/assistant roles)
- **Labels**: Boolean (true/false) for desirable/undesirable

## Coverage

This dataset covers:
- Single-step tool usage (load workspace, read content, create state)
- Multi-step workflows with context accumulation
- Workspace-aware operations with full context restoration
- Error handling and recovery patterns
- Context switching between workspaces
- Team coordination and multi-project management
- State checkpointing and resumption
- Schedule management and deadline handling

## Batch Sets

| Batch Set | Count | Focus |
|-----------|-------|-------|
| A (52-54) | 144 | Core tools |
| B (55-57) | 144 | Advanced workflows |
| C (58-60) | 144 | Tool discovery & error recovery |
| D (61-63) | 46 | Workspace-aware workflows |
| E (64-66) | 17 | Complex workflows & error handling |
| F (67-70) | 61 | Cross-workspace coordination |
| Final | 39 | Utility patterns |

## Usage

The dataset is formatted as JSONL with the following structure:

```json
{
  "conversations": [
    {"role": "user", "content": "User request"},
    {"role": "assistant", "content": "tool_call: toolName\\narguments: {...}\\n\\nResult: {...}\\n\\nResponse"}
  ],
  "label": true  // or false for undesirable examples
}
```

### For KTO Training

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("jrosenbaum/claudesidian-synthetic-dataset",
                       data_files="syngen_toolset_v1.0.0_claude.jsonl")

# Split by label for KTO training
desirable = dataset.filter(lambda x: x["label"] == True)
undesirable = dataset.filter(lambda x: x["label"] == False)
```

### For Fine-tuning

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load and prepare
dataset = load_dataset("jrosenbaum/claudesidian-synthetic-dataset",
                       data_files="syngen_toolset_v1.0.0_claude.jsonl")

# Format for instruction tuning
def format_example(example):
    return {
        "text": f"{example['conversations'][0]['content']}\\n{example['conversations'][1]['content']}"
    }

formatted = dataset.map(format_example)
```

## Key Patterns

### Desirable Examples (742)
- Proper tool selection and parameter accuracy
- Complete context objects with all 7 required fields
- Realistic workspace names and file paths
- Proper error handling and recovery
- Multi-step workflows with context preservation
- Clear next steps and actionable feedback

### Undesirable Examples (254)
- Missing or incomplete context fields
- Wrong tool selection for tasks
- Empty or minimal sessionMemory
- Inconsistent ID formats
- Poor error handling
- Incomplete workflow execution

## Training

This dataset is optimized for:
- **KTO Training**: Paired desirable/undesirable examples for contrastive learning
- **Fine-tuning**: Tool calling, context preservation, workflow execution
- **Evaluation**: Testing LLM ability to chain tools and manage state
- **Research**: Understanding AI behavior patterns with workspace systems

## Tools Covered

The dataset includes examples for 42+ tool schemas across 5 agents:
- **vaultManager**: File/folder operations
- **contentManager**: CRUD operations
- **memoryManager**: Sessions/states/workspace management
- **vaultLibrarian**: Advanced search and batch operations
- **agentManager**: Agent lifecycle and image generation

## Source

Generated using Claude (Anthropic) with careful attention to:
- Realistic use cases and scenarios
- Proper tool chaining and context management
- Error handling and edge cases
- Multi-step workflow complexity

## License

These synthetic examples are provided for research and training purposes.

---

**Last Updated**: November 9, 2025
**Total Size**: ~2.5 MB
**Format**: JSONL (ChatML)
"""

        try:
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN
            )
            print("✓ README.md uploaded successfully!")
        except Exception as e:
            print(f"Note: Could not upload README: {e}")

        print("\n" + "="*60)
        print("✅ Dataset upload complete!")
        print("="*60)
        print(f"\nYour dataset is now available at:")
        print(f"  https://huggingface.co/datasets/{REPO_ID}")
        print(f"\nYou can load it with:")
        print(f"  from datasets import load_dataset")
        print(f"  ds = load_dataset('{REPO_ID}', data_files='{filename}')")

    except Exception as e:
        print(f"Error during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_dataset()
