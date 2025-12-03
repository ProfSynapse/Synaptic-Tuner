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
- balanced-dataset
size_categories:
- 1K<n<10K
---

# Nexus Synthetic Data - Balanced Behaviors v1.5

## Dataset Description

This dataset contains **4,572 perfectly balanced synthetic training examples** demonstrating behavioral patterns for training language models to use the Claudesidian-MCP toolset effectively with Obsidian vaults.

**Key Features:**
- ✅ **Perfect 1:1 balance** (2,286 positive / 2,286 negative)
- ✅ **KTO-optimized interleaving** (consecutive same labels: 0)
- ✅ **OpenAI-compatible format** with proper tool calling structure
- ✅ **9 behavioral categories** + 3 response patterns
- ✅ **100% validated** against tool schemas

## Behavioral Categories

This dataset includes examples from **12 distinct categories**:

### Core Behaviors (9 categories)
1. **error_recovery** (592 examples) - Graceful error handling and recovery strategies
2. **workspace_awareness** (544 examples) - Using workspace context and preferences
3. **intellectual_humility** (650 examples) - Asking questions and acknowledging uncertainty
4. **strategic_tool_selection** (534 examples) - Choosing efficient tools for tasks
5. **execute_prompt_usage** (306 examples) - Proper delegation to AI agents
6. **verification_before_action** (628 examples) - Verifying before destructive operations
7. **context_continuity** (502 examples) - Maintaining context across interactions
8. **context_efficiency** (266 examples) - Appropriate context limits
9. **ask_first** (100 examples) - Asking before acting

### Response Patterns (3 categories)
10. **text_only** (150 examples) - Text-only response patterns
11. **tool_only** (150 examples) - Tool-only response patterns
12. **tool_text** (150 examples) - Combined tool and text patterns

## Dataset Structure

### Format
- **OpenAI-compatible tool calling format** (ChatML)
- **Perfect KTO interleaving** (alternating True/False labels)
- Each example includes:
  - User message
  - Assistant response with tool calls
  - Tool call metadata (id, type, function, arguments)
  - Behavioral label (true/false)
  - Behavior classification tag

### Example Structure
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Update the project status note"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "toolu_abc123",
          "type": "function",
          "function": {
            "name": "contentManager_updateContent",
            "arguments": "{\"context\": {...}, \"noteId\": \"...", \"updates\": {...}}"
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

- **Total Examples**: 4,572
- **Positive Examples**: 2,286 (50.0%)
- **Negative Examples**: 2,286 (50.0%)
- **Categories**: 12 (9 behaviors + 3 response patterns)
- **Format**: 100% OpenAI-compatible
- **Interleaved**: Perfect alternation (consecutive same: 0)
- **Balanced**: Perfect 1:1 ratio

## Balance Details

All individual datasets are perfectly balanced:

| Category | Total | Positive | Negative | Balance |
|----------|-------|----------|----------|---------|
| error_recovery | 592 | 296 | 296 | ✅ 1:1 |
| workspace_awareness | 544 | 272 | 272 | ✅ 1:1 |
| intellectual_humility | 650 | 325 | 325 | ✅ 1:1 |
| strategic_tool_selection | 534 | 267 | 267 | ✅ 1:1 |
| execute_prompt_usage | 306 | 153 | 153 | ✅ 1:1 |
| verification_before_action | 628 | 314 | 314 | ✅ 1:1 |
| context_continuity | 502 | 251 | 251 | ✅ 1:1 |
| context_efficiency | 266 | 133 | 133 | ✅ 1:1 |
| ask_first | 100 | 50 | 50 | ✅ 1:1 |
| text_only | 150 | 75 | 75 | ✅ 1:1 |
| tool_only | 150 | 75 | 75 | ✅ 1:1 |
| tool_text | 150 | 75 | 75 | ✅ 1:1 |

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("professorsynapse/nexus-synthetic-data",
                      data_files="balanced/behavior_merged_kto_v1.5_balanced.jsonl")
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
    max_length=2048,
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

# Filter for positive examples
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

This dataset was synthetically generated and manually curated to achieve perfect balance.

### Generation Process
1. Initial synthetic generation using Claude 3.5 Sonnet
2. Manual handcrafting of 304 additional negative examples to achieve 1:1 balance
3. Format conversion to OpenAI-compatible structure
4. KTO interleaving optimization (perfect alternation)
5. Comprehensive validation against 47 tool schemas

### Quality Assurance
- ✅ 100% OpenAI-compatible format
- ✅ All tool calls have valid JSON arguments
- ✅ Perfect interleaving (0 consecutive same labels after shuffle)
- ✅ Complete context objects in all examples
- ✅ Validated against tool schemas
- ✅ Perfect 1:1 positive:negative balance

## Version History

- **v1.5-balanced** (2025-12-02): Perfect 1:1 balance achieved, 4,572 examples
- **v1.4** (2024-11-29): Added clarification fixes, 8 behaviors
- **v1.3** (2024-11-28): Updated behaviors, improved quality
- **v1.0** (2024-11-23): Initial release

## License

MIT License - Free to use for research and commercial applications.

## Citation

```bibtex
@dataset{nexus_synthetic_behaviors_v1_5_balanced,
  title={Nexus Synthetic Data - Balanced Behaviors v1.5},
  author={ProfSynapse},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/professorsynapse/nexus-synthetic-data}}
}
```

## Contact

- GitHub: [ProfSynapse/Toolset-Training](https://github.com/ProfSynapse/Toolset-Training)
- HuggingFace: [@professorsynapse](https://huggingface.co/professorsynapse)
