"""
README generator.

Creates simple README files for upload directories.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.interfaces import IDocumentationGenerator


class ReadmeGenerator(IDocumentationGenerator):
    """
    Generator for README files.

    Creates simple README files documenting the upload structure.
    For comprehensive model cards, use ModelCardGenerator.
    """

    @property
    def name(self) -> str:
        return "readme"

    def generate(
        self,
        repo_id: str = "",
        training_run: str = "",
        formats_created: List[str] = None,
        model_name: str = "",
        **data
    ) -> str:
        """
        Generate simple README content.

        Args:
            repo_id: HuggingFace repository ID
            training_run: Training run timestamp
            formats_created: List of formats created
            model_name: Model name for file references
            **data: Additional data

        Returns:
            Markdown string for README
        """
        if formats_created is None:
            formats_created = []

        content = f"""# {model_name}

**Training Run:** `{training_run}`
**HuggingFace:** [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})

## Available Formats

"""

        for fmt in formats_created:
            if fmt == "lora":
                content += "- **LoRA Adapters** (`lora/`) - Use with base model\n"
            elif fmt == "merged_16bit":
                content += "- **Merged 16-bit** (`merged-16bit/`) - Full quality merged model (~14GB)\n"
            elif fmt == "merged_4bit":
                content += "- **Merged 4-bit** (`merged-4bit/`) - Quantized merged model (~3.5GB)\n"
            elif fmt == "gguf":
                content += "- **GGUF Quantizations** (`gguf/`) - For llama.cpp/Ollama\n"

        content += f"""
## Directory Structure

```
{model_name}/
"""

        for fmt in formats_created:
            subdir = fmt.replace("_", "-") if fmt != "lora" else "lora"
            if fmt == "gguf":
                content += f"""├── {subdir}/
│   ├── {model_name}.gguf (f16)
│   ├── {model_name}-Q4_K_M.gguf
│   ├── {model_name}-Q5_K_M.gguf
│   └── {model_name}-Q8_0.gguf
"""
            else:
                content += f"├── {subdir}/\n"

        content += """├── upload_manifest.json
└── README.md
```

## Usage

See the HuggingFace model card for detailed usage instructions.
"""

        return content

    def save(self, content: str, output_path: Path) -> Path:
        """
        Save README to file.

        Args:
            content: README markdown content
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ README created: {output_path}")
        return output_path
