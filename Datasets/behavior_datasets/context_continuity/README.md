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
- context-continuity
size_categories:
- n<1K
---

# Claudesidian Behavioral Dataset - Context Continuity

## Dataset Description

This dataset contains synthetic training examples demonstrating **context continuity** behavior patterns for training language models to use the Claudesidian-MCP toolset effectively with Obsidian vaults.

### Behavior Focus: Context Continuity


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
  "behavior": "context_continuity"
}
```

## Statistics
- **Total Examples**: 266
- **Positive Examples**: 133
- **Negative Examples**: 133

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("ProfSynapse/claudesidian-behavior-context_continuity")
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
@dataset{claudesidian_behavior_context_continuity,
  title={Claudesidian Behavioral Dataset - Context Continuity},
  author={ProfSynapse},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/ProfSynapse/claudesidian-behavior-context_continuity}}
}
```

## Related Datasets

- [Claudesidian Merged Behaviors](https://huggingface.co/datasets/ProfSynapse/claudesidian-behaviors-merged) - All behaviors combined
- [Claudesidian Base Dataset](https://huggingface.co/datasets/ProfSynapse/claudesidian-toolset) - Main tool-calling dataset

## Contact

- GitHub: [ProfSynapse/Toolset-Training](https://github.com/ProfSynapse/Toolset-Training)
