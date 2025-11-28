---
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

# Claudesidian Merged Behavioral Dataset

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
            "arguments": "{\"context\": {...}, ...}"
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

dataset = load_dataset("ProfSynapse/claudesidian-behaviors-merged")
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
@dataset{claudesidian_behaviors_merged,
  title={Claudesidian Merged Behavioral Dataset},
  author={ProfSynapse},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/ProfSynapse/claudesidian-behaviors-merged}}
}
```

## Related Datasets

- [Individual Behavior Datasets](https://huggingface.co/ProfSynapse) - Each behavior available separately
- [Claudesidian Base Dataset](https://huggingface.co/datasets/ProfSynapse/claudesidian-toolset) - Main tool-calling dataset

## Contact

- GitHub: [ProfSynapse/Toolset-Training](https://github.com/ProfSynapse/Toolset-Training)

## Version History

- **v1.1** (2025-11-24): OpenAI-compatible format, 8 behaviors, 1,852 examples
- **v1.0** (2025-11-23): Initial release, 6 behaviors, 1,572 examples
