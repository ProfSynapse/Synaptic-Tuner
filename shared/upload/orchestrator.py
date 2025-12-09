"""
Upload orchestrator.

Coordinates the complete model upload workflow:
1. Save model locally
2. Upload to repository
3. Convert to additional formats (optional)
4. Generate documentation
5. Upload documentation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from .core.config import (
    UploadConfig,
    SaveConfig,
    ConversionConfig,
    DocumentationConfig,
)
from .core.types import ModelPath
from .core.exceptions import UploadError
from .strategies.registry import SaveStrategyRegistry
from .converters.registry import ConverterRegistry
from .uploaders.registry import UploaderRegistry
from .documentation.manifest import ManifestGenerator
from .documentation.model_card import ModelCardGenerator
from .documentation.readme import ReadmeGenerator
from ..model_loading.registry import ModelLoaderRegistry


class UploadOrchestrator:
    """
    Orchestrates the complete model upload process.

    Responsibilities:
    - Coordinate saving, conversion, upload, and documentation
    - Manage dependencies between steps
    - Handle errors and cleanup
    """

    def __init__(
        self,
        upload_config: UploadConfig,
        save_config: SaveConfig,
        conversion_config: Optional[ConversionConfig] = None,
        documentation_config: Optional[DocumentationConfig] = None,
        model_loader_name: str = "unsloth"
    ):
        """
        Initialize the orchestrator.

        Args:
            upload_config: Configuration for upload
            save_config: Configuration for saving
            conversion_config: Configuration for conversion (optional)
            documentation_config: Configuration for documentation (optional)
            model_loader_name: Name of model loader to use
        """
        self.upload_config = upload_config
        self.save_config = save_config
        self.conversion_config = conversion_config
        self.documentation_config = documentation_config or DocumentationConfig()

        # Get model loader
        self.model_loader = ModelLoaderRegistry.get(model_loader_name)

        # Determine output directory
        self.output_dir = self._resolve_output_dir()

        # State tracking
        self.artifacts_created: List[Path] = []
        self.formats_created: List[str] = []
        self.gguf_files: List[Path] = []

    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete upload workflow.

        Returns:
            Dictionary with upload results and artifact paths
        """
        try:
            print("\n" + "=" * 60)
            print("UPLOAD ORCHESTRATOR")
            print("=" * 60)
            print(f"Model path: {self.upload_config.model_path}")
            print(f"Repository: {self.upload_config.repo_id}")
            print(f"Output directory: {self.output_dir}")
            print()

            # Step 1: Save model locally
            saved_path = self._save_model_locally()

            # Step 2: Upload to repository
            upload_url = self._upload_to_repository(saved_path)

            # Step 3: Convert to additional formats (optional)
            if self.conversion_config:
                self._convert_formats()
                self._upload_converted_files()

            # Step 4: Generate documentation
            docs = self._generate_documentation()

            # Step 5: Upload documentation
            self._upload_documentation()

            return {
                "success": True,
                "upload_url": upload_url,
                "local_artifacts": str(self.output_dir),
                "formats": self.formats_created,
                "gguf_files": [str(f) for f in self.gguf_files],
            }

        except Exception as e:
            print(f"\n✗ Upload failed: {e}")
            raise

    def _save_model_locally(self) -> Path:
        """Save model using configured strategy."""
        if not self.save_config.save_local:
            return Path(self.upload_config.model_path)

        strategy = SaveStrategyRegistry.get(
            self.save_config.strategy_name,
            model_loader=self.model_loader
        )

        saved_path = strategy.save(
            self.upload_config.model_path,
            self.output_dir,
            model_size=self.save_config.model_size
        )

        self.formats_created.append(self.save_config.strategy_name)
        self.artifacts_created.append(saved_path)

        return saved_path

    def _upload_to_repository(self, local_path: Path) -> str:
        """Upload model to repository."""
        uploader = UploaderRegistry.get("huggingface")

        # For merged models, upload the merged directory
        # For LoRA, upload the adapters
        url = uploader.upload_model(
            local_path,
            self.upload_config.repo_id,
            self.upload_config.credential,
            private=self.upload_config.private
        )

        return url

    def _convert_formats(self):
        """Convert to additional formats (e.g., GGUF)."""
        if not self.conversion_config:
            return

        converter = ConverterRegistry.get(
            self.conversion_config.converter_name,
            model_loader=self.model_loader
        )

        model_name = self.upload_config.repo_id.split('/')[-1]

        converted_files = converter.convert(
            self.upload_config.model_path,
            self.output_dir,
            quantizations=self.conversion_config.quantizations,
            model_name=model_name,
            cleanup=self.conversion_config.cleanup_temp,
            model_size=self.save_config.model_size
        )

        self.formats_created.append(self.conversion_config.converter_name)
        self.gguf_files = converted_files
        self.artifacts_created.extend(converted_files)

    def _upload_converted_files(self):
        """Upload converted files (GGUF) to repository."""
        if not self.gguf_files:
            return

        uploader = UploaderRegistry.get("huggingface")

        uploader.upload_files(
            self.gguf_files,
            self.upload_config.repo_id,
            self.upload_config.credential
        )

    def _generate_documentation(self) -> Dict[str, Path]:
        """Generate all documentation files."""
        docs = {}
        training_run = self._extract_training_run()
        model_name = self.upload_config.repo_id.split('/')[-1]
        hf_username = self.upload_config.repo_id.split('/')[0]

        # Load training lineage if available
        training_lineage = self._load_training_lineage()

        print("\n" + "=" * 60)
        print("CREATING DOCUMENTATION")
        print("=" * 60)

        # Generate manifest
        if self.documentation_config.include_manifest:
            manifest_gen = ManifestGenerator()
            manifest_content = manifest_gen.generate(
                repo_id=self.upload_config.repo_id,
                training_run=training_run,
                formats_created=self.formats_created,
                gguf_files=[str(f) for f in self.gguf_files]
            )
            manifest_path = self.output_dir / "upload_manifest.json"
            manifest_gen.save(manifest_content, manifest_path)
            docs["manifest"] = manifest_path

        # Generate README/model card
        if self.documentation_config.include_readme:
            readme_path = self.output_dir / "README.md"

            if training_lineage:
                # Use comprehensive model card
                card_gen = ModelCardGenerator()
                content = card_gen.generate(
                    lineage=training_lineage,
                    hf_username=hf_username
                )
                card_gen.save(content, readme_path)
            else:
                # Use simple README
                readme_gen = ReadmeGenerator()
                content = readme_gen.generate(
                    repo_id=self.upload_config.repo_id,
                    training_run=training_run,
                    formats_created=self.formats_created,
                    model_name=model_name
                )
                readme_gen.save(content, readme_path)

            docs["readme"] = readme_path

        return docs

    def _upload_documentation(self):
        """Upload documentation files to repository."""
        uploader = UploaderRegistry.get("huggingface")

        print("\n" + "=" * 60)
        print("UPLOADING DOCUMENTATION")
        print("=" * 60)

        # Upload manifest
        manifest_path = self.output_dir / "upload_manifest.json"
        if manifest_path.exists():
            try:
                uploader.upload_file(
                    manifest_path,
                    self.upload_config.repo_id,
                    self.upload_config.credential,
                    "upload_manifest.json"
                )
                print("✓ upload_manifest.json uploaded")
            except Exception as e:
                print(f"⚠ Could not upload manifest: {e}")

        # Upload README
        readme_path = self.output_dir / "README.md"
        if readme_path.exists():
            try:
                uploader.upload_file(
                    readme_path,
                    self.upload_config.repo_id,
                    self.upload_config.credential,
                    "README.md"
                )
                print("✓ README.md uploaded")
            except Exception as e:
                print(f"⚠ Could not upload README: {e}")

        # Upload training lineage if available
        lineage_path = self._get_lineage_path()
        if lineage_path and lineage_path.exists():
            try:
                uploader.upload_file(
                    lineage_path,
                    self.upload_config.repo_id,
                    self.upload_config.credential,
                    "training_lineage.json"
                )
                print("✓ training_lineage.json uploaded")
            except Exception as e:
                print(f"⚠ Could not upload training lineage: {e}")

    def _resolve_output_dir(self) -> Path:
        """Determine output directory."""
        if self.upload_config.output_dir:
            return self.upload_config.output_dir

        # Auto-detect from model path
        model_path = Path(self.upload_config.model_path)
        model_name = self.upload_config.repo_id.split('/')[-1]

        # Pattern: sft_output_rtx3090/YYYYMMDD_HHMMSS/final_model
        # We want: sft_output_rtx3090/YYYYMMDD_HHMMSS/model-name/
        if model_path.name == "final_model":
            parent_name = model_path.parent.parent.name
            if parent_name in ["sft_output_rtx3090", "kto_output_rtx3090"]:
                return model_path.parent / model_name

        return model_path.parent / model_name

    def _extract_training_run(self) -> str:
        """Extract training run timestamp from path."""
        if self.output_dir.parent.name != ".":
            return self.output_dir.parent.name
        return "unknown"

    def _get_lineage_path(self) -> Optional[Path]:
        """Get path to training lineage file."""
        if self.documentation_config.training_lineage_path:
            return self.documentation_config.training_lineage_path

        # Try auto-detect
        model_path = Path(self.upload_config.model_path)
        auto_path = model_path.parent / "training_lineage.json"
        if auto_path.exists():
            return auto_path

        return None

    def _load_training_lineage(self) -> Optional[Dict[str, Any]]:
        """Load training lineage if available."""
        lineage_path = self._get_lineage_path()
        if lineage_path and lineage_path.exists():
            with open(lineage_path, 'r', encoding='utf-8') as f:
                print(f"✓ Training lineage loaded from: {lineage_path}")
                return json.load(f)
        return None

    def print_summary(self):
        """Print final summary of upload."""
        print("\n" + "=" * 60)
        print("✓ ALL UPLOADS COMPLETE!")
        print("=" * 60)
        print(f"\nLocal artifacts saved to: {self.output_dir}")
        print(f"HuggingFace model: https://huggingface.co/{self.upload_config.repo_id}")
        print(f"\nDirectory structure:")
        print(f"  {self.output_dir}/")

        for fmt in self.formats_created:
            subdir = fmt.replace("_", "-") if fmt not in ["lora", "gguf"] else fmt
            print(f"  ├── {subdir}/")

        print(f"  ├── upload_manifest.json")
        print(f"  └── README.md")
