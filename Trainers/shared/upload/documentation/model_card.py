"""
Model card generator.

Creates comprehensive HuggingFace model cards from training lineage.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.interfaces import IDocumentationGenerator


class ModelCardGenerator(IDocumentationGenerator):
    """
    Generator for HuggingFace model cards.

    Creates comprehensive model cards with:
    - Model description
    - Training details
    - Dataset information
    - Usage examples
    """

    @property
    def name(self) -> str:
        return "model_card"

    def generate(
        self,
        lineage: Dict[str, Any] = None,
        hf_username: str = "",
        **data
    ) -> str:
        """
        Generate model card markdown content.

        Args:
            lineage: Training lineage dictionary
            hf_username: HuggingFace username
            **data: Additional data

        Returns:
            Markdown string for model card
        """
        if lineage is None:
            lineage = {}

        base_model = lineage.get("base_model", {})
        dataset = lineage.get("dataset", {})
        lora = lineage.get("lora_config", {})
        training = lineage.get("training_config", {})
        results = lineage.get("training_results", {})
        hardware = lineage.get("hardware", {})
        frameworks = lineage.get("framework_versions", {})

        # Determine training method
        training_method = lineage.get("training_method", "SFT")
        is_kto = training_method == "KTO"

        model_name = lineage.get("model_name", "model")

        # Build YAML frontmatter
        yaml_block = self._build_yaml_frontmatter(
            base_model, dataset, training_method, model_name, results
        )

        # Build main content
        content = self._build_main_content(
            lineage, base_model, dataset, lora, training,
            results, hardware, frameworks, training_method, is_kto, hf_username
        )

        return yaml_block + content

    def _build_yaml_frontmatter(
        self,
        base_model: Dict,
        dataset: Dict,
        training_method: str,
        model_name: str,
        results: Dict
    ) -> str:
        """Build YAML frontmatter for model card."""
        return f'''---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- tool-calling
- {training_method.lower()}
- {"preference-learning" if training_method == "KTO" else "supervised-fine-tuning"}
- claudesidian
- obsidian
- fine-tuned
- unsloth
base_model: {base_model.get("name", "unknown")}
datasets:
- {dataset.get("name", "unknown")}
pipeline_tag: text-generation
model-index:
- name: {model_name}
  results:
  - task:
      type: text-generation
    metrics:
    - name: Final Loss
      type: loss
      value: {results.get("final_loss", "N/A")}
---

'''

    def _build_main_content(
        self,
        lineage: Dict,
        base_model: Dict,
        dataset: Dict,
        lora: Dict,
        training: Dict,
        results: Dict,
        hardware: Dict,
        frameworks: Dict,
        training_method: str,
        is_kto: bool,
        hf_username: str
    ) -> str:
        """Build main model card content."""
        model_name = lineage.get("model_name", "model")

        content = f'''# {model_name}

This model was fine-tuned using **{training_method}** to {"improve tool-calling behavior" if is_kto else "learn tool-calling behavior"} for the Claudesidian vault application.

## Model Description

- **Base Model:** [{base_model.get("name", "unknown")}](https://huggingface.co/{base_model.get("name", "unknown")})
- **Training Method:** {training_method}
- **Task:** Tool-calling for Obsidian vault operations
- **Training Date:** {lineage.get("training_date", "N/A")}

## Training Details

### Dataset

| Property | Value |
|----------|-------|
| Dataset | [{dataset.get("name", "unknown")}]({dataset.get("huggingface_url", "#")}) |
'''

        if dataset.get("file"):
            content += f'| File | {dataset.get("file")} |\n'

        content += f'| Total Examples | {dataset.get("total_examples", "N/A"):,} |\n'

        if is_kto:
            content += f'| Desirable | {dataset.get("desirable_examples", "N/A"):,} |\n'
            content += f'| Undesirable | {dataset.get("undesirable_examples", "N/A"):,} |\n'

        if dataset.get("chat_template"):
            content += f'| Chat Template | {dataset.get("chat_template")} |\n'

        # KTO-specific section
        if is_kto and lineage.get("kto_config"):
            kto = lineage["kto_config"]
            content += f'''
### KTO Configuration

| Parameter | Value |
|-----------|-------|
| Beta (β) | {kto.get("beta", "N/A")} |
| Desirable Weight | {kto.get("desirable_weight", "N/A")} |
| Undesirable Weight | {kto.get("undesirable_weight", "N/A")} |
'''

        # LoRA and training config
        trainable_params = lora.get("trainable_parameters", "N/A")
        if isinstance(trainable_params, (int, float)):
            trainable_params = f"{trainable_params:,}"

        content += f'''
### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | {lora.get("r", "N/A")} |
| Alpha (α) | {lora.get("alpha", "N/A")} |
| Dropout | {lora.get("dropout", "N/A")} |
| Target Modules | {", ".join(lora.get("target_modules", []))} |
| Trainable Parameters | {trainable_params} ({lora.get("trainable_percentage", "N/A")}%) |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | {training.get("batch_size", "N/A")} |
| Gradient Accumulation | {training.get("gradient_accumulation_steps", "N/A")} |
| Effective Batch Size | {training.get("effective_batch_size", "N/A")} |
| Learning Rate | {training.get("learning_rate", "N/A")} |
| LR Scheduler | {training.get("learning_rate_scheduler", "N/A")} |
| Warmup Ratio | {training.get("warmup_ratio", "N/A")} |
| Max Grad Norm | {training.get("max_grad_norm", "N/A")} |
| Epochs | {training.get("num_epochs", "N/A")} |
| Optimizer | {training.get("optimizer", "N/A")} |
| Precision | {training.get("precision", "N/A")} |
| Random Seed | {training.get("random_seed", "N/A")} |

### Training Results

| Metric | Value |
|--------|-------|
| Final Loss | {results.get("final_loss", "N/A")} |
| Total Steps | {results.get("total_steps", "N/A"):,} |
| Training Duration | {results.get("training_duration_minutes", "N/A")} minutes |

### Hardware

| Component | Value |
|-----------|-------|
| GPU | {hardware.get("gpu", "N/A")} |
| GPU Memory | {hardware.get("gpu_memory_gb", "N/A")} GB |
| CUDA Version | {hardware.get("cuda_version", "N/A")} |
| Platform | {hardware.get("platform", "N/A")} |

### Framework Versions

| Library | Version |
|---------|---------|
| PyTorch | {frameworks.get("torch", "N/A")} |
| Transformers | {frameworks.get("transformers", "N/A")} |
| TRL | {frameworks.get("trl", "N/A")} |
| PEFT | {frameworks.get("peft", "N/A")} |
| Unsloth | {frameworks.get("unsloth", "N/A")} |

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{hf_username}/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{hf_username}/{model_name}")

messages = [{{"role": "user", "content": "Show me the contents of my project roadmap file."}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

### With Ollama (GGUF)

```bash
ollama pull {hf_username}/{model_name}
ollama run {hf_username}/{model_name}
```

## Intended Use

This model is designed for tool-calling in Obsidian vault management applications.

## Training Lineage

<details>
<summary>Click to expand full training configuration (JSON)</summary>

```json
{json.dumps(lineage, indent=2)}
```

</details>
'''

        return content

    def save(self, content: str, output_path: Path) -> Path:
        """
        Save model card to file.

        Args:
            content: Model card markdown content
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Model card created: {output_path}")
        return output_path
