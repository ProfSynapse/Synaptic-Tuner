#!/usr/bin/env python3
"""
Upload behavioral datasets to Hugging Face Hub.

This script uploads:
1. Individual behavior datasets (8 separate datasets)
2. Merged KTO-compatible dataset
3. Comprehensive documentation
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
from datetime import datetime


def load_env_token():
    """Load HuggingFace token from .env file."""
    # Check multiple locations for .env file
    possible_paths = [
        Path(__file__).parent.parent / ".env",  # Datasets/.env
        Path(__file__).parent.parent.parent / ".env",  # Project root/.env
    ]

    for env_file in possible_paths:
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN=') or line.startswith('HF_API_KEY='):
                        return line.split('=', 1)[1].strip().strip('"').strip("'")
    return os.environ.get('HF_TOKEN') or os.environ.get('HF_API_KEY')


def create_dataset_card(behavior_name=None, stats=None):
    """Create a dataset card (README.md) for the dataset."""

    if behavior_name:
        # Individual behavior dataset card
        return f"""---
language:
- en
license: mit
task_categories:
- text-generation
- conversational
tags:
- synthetic
- tool-calling
- openai-format
- behavior-modeling
- {behavior_name.replace('_', '-')}
size_categories:
- n<1K
---

# Nexus Synthetic Data - Behavior: {behavior_name.replace('_', ' ').title()}

## Dataset Description

This dataset contains synthetic training examples demonstrating **{behavior_name.replace('_', ' ')}** behavior patterns for training language models to use the Claudesidian-MCP toolset effectively with Obsidian vaults.

### Behavior Focus: {behavior_name.replace('_', ' ').title()}

{get_behavior_description(behavior_name)}

## Dataset Structure

### Format
- **OpenAI-compatible tool calling format** (ChatML)
- Each example includes:
  - User message
  - Assistant response with tool calls
  - Tool call metadata (id, type, function name, arguments)
  - Behavioral label (true/false for KTO training)
  - Behavior classification tag

### Example Structure
```json
{{
  "conversations": [
    {{
      "role": "user",
      "content": "User request..."
    }},
    {{
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {{
          "id": "abc123def",
          "type": "function",
          "function": {{
            "name": "toolName",
            "arguments": "{{\\"context\\": {{...}}, ...}}"
          }}
        }}
      ]
    }}
  ],
  "label": true,
  "behavior": "{behavior_name}"
}}
```

## Statistics
- **Total Examples**: {stats.get('total', 'N/A') if stats else 'N/A'}
- **Positive Examples**: {stats.get('positive', 'N/A') if stats else 'N/A'}
- **Negative Examples**: {stats.get('negative', 'N/A') if stats else 'N/A'}

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("professorsynapse/nexus-synthetic-data", data_files="behaviors/{behavior_name}.jsonl")
```

### Training with TRL
```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-v0.3")
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-v0.3")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="conversations",
    max_seq_length=2048,
)
trainer.train()
```

## Dataset Creation

This dataset was synthetically generated using Claude 3.5 Sonnet to demonstrate proper and improper usage patterns of specific behavioral patterns in tool-calling scenarios.

### Generation Process
1. Behavior rubric definition
2. Synthetic conversation generation
3. Format conversion to OpenAI-compatible structure
4. Quality validation and verification

## License

MIT License - Free to use for research and commercial applications.

## Citation

```bibtex
@dataset{{nexus_synthetic_behavior_{behavior_name},
  title={{Nexus Synthetic Data - {behavior_name.replace('_', ' ').title()}}},
  author={{ProfSynapse}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/professorsynapse/nexus-synthetic-data}}}}
}}
```

## Related Datasets

- [Nexus Synthetic Data](https://huggingface.co/datasets/professorsynapse/nexus-synthetic-data) - All datasets combined

## Contact

- GitHub: [ProfSynapse/Toolset-Training](https://github.com/ProfSynapse/Toolset-Training)
"""
    else:
        # Merged dataset card
        return """---
language:
- en
license: mit
task_categories:
- text-generation
- conversational
tags:
- synthetic
- tool-calling
- openai-format
- behavior-modeling
- kto-training
- preference-learning
size_categories:
- 1K<n<10K
---

# Nexus Synthetic Data - Merged Behaviors

## Dataset Description

This dataset contains **1,852 synthetic training examples** demonstrating 8 different behavioral patterns for training language models to use the Claudesidian-MCP toolset effectively with Obsidian vaults.

The dataset is specifically formatted for **KTO (Kahneman-Tversky Optimization)** preference learning with properly interleaved positive and negative examples.

## Behavioral Categories

This dataset includes examples from 8 distinct behavioral patterns:

1. **context_continuity** (266 examples) - Maintaining context across multi-turn interactions
2. **context_efficiency** (140 examples) - Using appropriate context limits and avoiding overload
3. **error_recovery** (262 examples) - Gracefully handling errors and retrying with corrections
4. **execute_prompt_usage** (140 examples) - Properly delegating to AI agents when appropriate
5. **intellectual_humility** (260 examples) - Asking clarifying questions before acting
6. **strategic_tool_selection** (262 examples) - Choosing the most efficient tool for each task
7. **verification_before_action** (262 examples) - Verifying before destructive operations
8. **workspace_awareness** (260 examples) - Using workspace context and preferences

## Dataset Structure

### Format
- **OpenAI-compatible tool calling format** (ChatML)
- **KTO-compatible interleaving** (True/False pattern for preference learning)
- Each example includes:
  - User message
  - Assistant response with tool calls
  - Tool call metadata (id, type, function name, arguments)
  - Behavioral label (true/false)
  - Behavior classification tag

### Example Structure
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "User request..."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "abc123def",
          "type": "function",
          "function": {
            "name": "toolName",
            "arguments": "{\\"context\\": {...}, ...}"
          }
        }
      ]
    }
  ],
  "label": true,
  "behavior": "verification_before_action"
}
```

## Statistics
- **Total Examples**: 1,852
- **Positive Examples**: 1,085 (58.5%)
- **Negative Examples**: 767 (41.5%)
- **Behaviors**: 8 distinct patterns
- **Format**: 100% OpenAI-compatible
- **Interleaved**: Optimized for KTO training

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("professorsynapse/nexus-synthetic-data", data_files="behaviors/merged.jsonl")
```

### KTO Training with TRL
```python
from trl import KTOTrainer, KTOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-v0.3")
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-v0.3")

config = KTOConfig(
    per_device_train_batch_size=4,
    learning_rate=2e-7,
    beta=0.3,
)

trainer = KTOTrainer(
    model=model,
    ref_model=None,
    config=config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()
```

### SFT Training (Positive Examples Only)
```python
from trl import SFTTrainer

# Filter for positive examples only
positive_dataset = dataset["train"].filter(lambda x: x["label"] == True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=positive_dataset,
    dataset_text_field="conversations",
    max_seq_length=2048,
)
trainer.train()
```

## Dataset Creation

This dataset was synthetically generated using Claude 3.5 Sonnet to demonstrate proper and improper usage patterns across 8 distinct behavioral dimensions.

### Generation Process
1. Behavior rubrics defined for each category
2. Synthetic conversation generation with positive/negative pairs
3. Format conversion to OpenAI-compatible structure
4. KTO interleaving optimization
5. Quality validation and verification

### Quality Assurance
- ✅ 100% OpenAI-compatible format
- ✅ All tool calls have valid JSON arguments
- ✅ Properly interleaved for KTO training
- ✅ Comprehensive context objects
- ✅ Validated against tool schemas

## License

MIT License - Free to use for research and commercial applications.

## Citation

```bibtex
@dataset{nexus_synthetic_behaviors_merged,
  title={Nexus Synthetic Data - Merged Behaviors},
  author={ProfSynapse},
  year={2025},
  publisher={Hugging Face},
  howpublished={\\url{https://huggingface.co/datasets/professorsynapse/nexus-synthetic-data}}
}
```

## Related Datasets

- [Nexus Synthetic Data](https://huggingface.co/datasets/professorsynapse/nexus-synthetic-data) - All datasets combined

## Contact

- GitHub: [ProfSynapse/Toolset-Training](https://github.com/ProfSynapse/Toolset-Training)

## Version History

- **v1.1** (2025-11-24): OpenAI-compatible format, 8 behaviors, 1,852 examples
- **v1.0** (2025-11-23): Initial release, 6 behaviors, 1,572 examples
"""


def get_behavior_description(behavior_name):
    """Get detailed description for each behavior."""
    descriptions = {
        "verification_before_action": """
This behavior focuses on **verifying information before taking destructive or irreversible actions**.

**Positive patterns:**
- Searching/listing before deleting files or folders
- Reading config files before modifying them
- Checking directory contents before moving/deleting
- Confirming file existence before operations

**Negative patterns:**
- Deleting without verifying targets
- Modifying files without reading current content
- Moving folders without checking contents
- Batch operations without prior inspection
""",
        "workspace_awareness": """
This behavior demonstrates **proper use of workspace context, preferences, and workflows**.

**Positive patterns:**
- Loading workspace before starting tasks
- Following workspace-defined workflows
- Using keyFiles from workspace context
- Respecting workspace preferences
- Applying workspace-specific patterns

**Negative patterns:**
- Ignoring workspace context
- Generic operations without workspace awareness
- Not using available workspace information
- Bypassing defined workflows
""",
        "context_continuity": """
This behavior focuses on **maintaining context and continuity across multi-turn interactions**.

**Positive patterns:**
- Building on previous conversation context
- Referencing prior tool results
- Maintaining consistent goal pursuit
- Updating sessionMemory appropriately
- Coherent multi-step workflows

**Negative patterns:**
- Ignoring previous context
- Generic responses without continuity
- Forgetting stated goals
- Not building on prior work
""",
        "context_efficiency": """
This behavior demonstrates **appropriate use of context limits and efficient information retrieval**.

**Positive patterns:**
- Using appropriate limit parameters (20-50 for typical searches)
- Requesting only needed information
- Avoiding excessive data loading
- Incremental fetching when appropriate

**Negative patterns:**
- Using limit: 1000 for simple searches
- Loading all files when only few needed
- Retrieving entire datasets unnecessarily
- Excessive context consumption
""",
        "strategic_tool_selection": """
This behavior focuses on **choosing the most efficient and appropriate tool for each task**.

**Positive patterns:**
- Using batch operations for multiple similar tasks
- Choosing searchDirectory over searchContent for file listing
- Using batchExecutePrompt for parallel AI operations
- Selecting tools based on task requirements

**Negative patterns:**
- Sequential operations when batch available
- Wrong tool for the task type
- Inefficient tool chains
- Missing opportunities for optimization
""",
        "error_recovery": """
This behavior demonstrates **graceful error handling and intelligent recovery strategies**.

**Positive patterns:**
- Searching for correct path after file not found
- Retrying with corrected parameters
- Listing directory after operation failure
- Adapting strategy based on error feedback

**Negative patterns:**
- Giving up after first failure
- Repeating failed operations unchanged
- Ignoring error messages
- Not attempting recovery
""",
        "execute_prompt_usage": """
This behavior focuses on **appropriate delegation to AI agents for complex tasks**.

**Positive patterns:**
- Using executePrompt for architectural design
- Delegating complex analysis to AI
- Providing detailed prompts with context
- Proper handling of AI-generated responses

**Negative patterns:**
- Attempting complex tasks directly
- Generic documents instead of expert consultation
- Not leveraging AI for specialized knowledge
- Poor prompt structure
""",
        "intellectual_humility": """
This behavior demonstrates **asking clarifying questions and acknowledging uncertainty**.

**Positive patterns:**
- Asking which files to delete when ambiguous
- Clarifying user requirements before acting
- Searching to understand scope before operations
- Acknowledging when information is needed

**Negative patterns:**
- Making assumptions without clarification
- Acting on ambiguous instructions
- Not seeking information when uncertain
- Overconfident operations
"""
    }
    return descriptions.get(behavior_name, "Behavioral pattern for tool calling optimization.")


def upload_individual_behavior(api, behavior_name, repo_name, token):
    """Upload a single behavior dataset to the unified repo."""
    behavior_dir = Path(__file__).parent.parent / "behavior_datasets" / behavior_name
    dataset_file = behavior_dir / "pairs_v1.3.jsonl"

    if not dataset_file.exists():
        print(f"  ⚠️  {dataset_file} not found, skipping")
        return False

    # Load stats
    stats = {"total": 0, "positive": 0, "negative": 0}
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                stats["total"] += 1
                if data.get("label") is True:
                    stats["positive"] += 1
                elif data.get("label") is False:
                    stats["negative"] += 1

    upload_time = datetime.now().strftime('%H:%M:%S')
    print(f"\n📦 [{upload_time}] Uploading {behavior_name}...")
    print(f"   Examples: {stats['total']} ({stats['positive']} positive, {stats['negative']} negative)")

    try:
        # Create repo (if not exists)
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=False
        )

        # Upload dataset file to behaviors/ folder
        api.upload_file(
            path_or_fileobj=str(dataset_file),
            path_in_repo=f"behaviors/{behavior_name}.jsonl",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )

        print(f"   ✅ Uploaded to https://huggingface.co/datasets/{repo_name}/blob/main/behaviors/{behavior_name}.jsonl")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def upload_merged_dataset(api, repo_name, token):
    """Upload the merged behavioral dataset to the unified repo."""
    merged_file = Path(__file__).parent.parent / "behavior_merged_kto_11.28.25.jsonl"
    metadata_file = Path(__file__).parent.parent / "behavior_merged_kto_11.28.25.metadata.json"

    if not merged_file.exists():
        print(f"  ⚠️  {merged_file} not found")
        return False

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    upload_time = datetime.now().strftime('%H:%M:%S')
    print(f"\n📦 [{upload_time}] Uploading merged dataset...")
    print(f"   Total examples: {metadata['total_examples']}")
    print(f"   Behaviors: {len(metadata.get('behaviors_v1_3', metadata.get('all_datasets', [])))}")

    try:
        # Create repo (if not exists)
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=False
        )

        # Upload dataset file to behaviors/ folder
        api.upload_file(
            path_or_fileobj=str(merged_file),
            path_in_repo="behaviors/merged.jsonl",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )

        # Upload metadata
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="behaviors/merged_metadata.json",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )

        print(f"   ✅ Uploaded to https://huggingface.co/datasets/{repo_name}/blob/main/behaviors/merged.jsonl")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    # Record start time
    start_time = datetime.now()

    print("=" * 80)
    print("Uploading Behavioral Datasets to Nexus Synthetic Data")
    print("=" * 80)
    print(f"📅 Upload started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Unified repo for all datasets
    REPO_NAME = "professorsynapse/nexus-synthetic-data"

    # Get token
    token = load_env_token()
    if not token:
        print("❌ HF_TOKEN not found in .env file or environment")
        print("Please set HF_TOKEN in the .env file or environment")
        return

    print(f"✅ Found HuggingFace token")

    # Initialize API
    api = HfApi(token=token)

    # Get username
    user_info = api.whoami(token=token)
    username = user_info['name']
    print(f"✅ Authenticated as: {username}")
    print(f"📁 Target repo: {REPO_NAME}")
    print()

    # Individual behaviors
    behaviors = [
        "context_continuity",
        "context_efficiency",
        "error_recovery",
        "execute_prompt_usage",
        "intellectual_humility",
        "strategic_tool_selection",
        "verification_before_action",
        "workspace_awareness"
    ]

    print("Uploading individual behavior datasets to behaviors/ folder...")
    success_count = 0
    for behavior in behaviors:
        if upload_individual_behavior(api, behavior, REPO_NAME, token):
            success_count += 1

    print(f"\n✅ Successfully uploaded {success_count}/{len(behaviors)} individual datasets")

    # Merged dataset
    print("\n" + "=" * 80)
    if upload_merged_dataset(api, REPO_NAME, token):
        print("\n✅ Merged dataset uploaded successfully")

    # Record completion time
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("Upload complete! 🎉")
    print("=" * 80)
    print(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
    print(f"\nView your datasets at: https://huggingface.co/datasets/{REPO_NAME}")


if __name__ == "__main__":
    main()
