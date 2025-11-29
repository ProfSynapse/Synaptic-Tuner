"""
Upload manifest generator.

Creates JSON manifests with upload metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.interfaces import IDocumentationGenerator


class ManifestGenerator(IDocumentationGenerator):
    """
    Generator for upload manifest JSON files.

    The manifest contains metadata about the upload:
    - Timestamp
    - Repository information
    - Formats created
    - File locations
    """

    @property
    def name(self) -> str:
        return "manifest"

    def generate(
        self,
        repo_id: str = "",
        training_run: str = "",
        formats_created: List[str] = None,
        gguf_files: List[str] = None,
        **data
    ) -> str:
        """
        Generate manifest JSON content.

        Args:
            repo_id: HuggingFace repository ID
            training_run: Training run timestamp
            formats_created: List of formats created
            gguf_files: List of GGUF file names
            **data: Additional data to include

        Returns:
            JSON string of the manifest
        """
        if formats_created is None:
            formats_created = []

        manifest = {
            "upload_timestamp": datetime.now().isoformat(),
            "training_run": training_run,
            "model_name": repo_id,
            "huggingface_url": f"https://huggingface.co/{repo_id}" if repo_id else "",
            "formats_created": formats_created,
            "directory_structure": {
                "lora": "lora/" if "lora" in formats_created else None,
                "merged_16bit": "merged-16bit/" if "merged_16bit" in formats_created else None,
                "merged_4bit": "merged-4bit/" if "merged_4bit" in formats_created else None,
                "gguf": "gguf/" if "gguf" in formats_created else None,
            }
        }

        if gguf_files:
            manifest["gguf_quantizations"] = [Path(f).name for f in gguf_files]

        # Add any additional data
        manifest.update(data)

        return json.dumps(manifest, indent=2)

    def save(self, content: str, output_path: Path) -> Path:
        """
        Save manifest to file.

        Args:
            content: Manifest JSON content
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Upload manifest created: {output_path}")
        return output_path
