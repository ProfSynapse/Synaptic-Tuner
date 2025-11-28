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
- execute-prompt-usage
size_categories:
- n<1K
---

# Claudesidian Behavioral Dataset - Execute Prompt Usage

## Dataset Description

This dataset contains synthetic training examples demonstrating **execute prompt usage** behavior patterns for training language models to use the Claudesidian-MCP toolset effectively with Obsidian vaults.

### Behavior Focus: Execute Prompt Usage


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
  "behavior": "execute_prompt_usage"
}
```

## Statistics
- **Total Examples**: 266
- **Positive Examples**: 153
- **Negative Examples**: 113

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("ProfSynapse/claudesidian-behavior-execute_prompt_usage")
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
@dataset{claudesidian_behavior_execute_prompt_usage,
  title={Claudesidian Behavioral Dataset - Execute Prompt Usage},
  author={ProfSynapse},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/ProfSynapse/claudesidian-behavior-execute_prompt_usage}}
}
```

## Related Datasets

- [Claudesidian Merged Behaviors](https://huggingface.co/datasets/ProfSynapse/claudesidian-behaviors-merged) - All behaviors combined
- [Claudesidian Base Dataset](https://huggingface.co/datasets/ProfSynapse/claudesidian-toolset) - Main tool-calling dataset

## Contact

- GitHub: [ProfSynapse/Toolset-Training](https://github.com/ProfSynapse/Toolset-Training)
