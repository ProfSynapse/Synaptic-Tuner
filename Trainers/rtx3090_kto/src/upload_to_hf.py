"""
Upload trained model to HuggingFace Hub.
Supports both standard and GGUF formats.
"""

# IMPORTANT: Apply Windows patches BEFORE importing unsloth
import sys
import os

# Apply Windows compatibility patches if needed
if sys.platform == 'win32':
    print("Applying Windows compatibility patches...")
    # Add patches inline to avoid import issues
    from dataclasses import dataclass, fields
    import dataclasses

    # Patch 1: Wrap fields() for non-dataclasses
    original_fields = fields
    def patched_fields(class_or_instance):
        try:
            return original_fields(class_or_instance)
        except TypeError:
            return ()
    dataclasses.fields = patched_fields

    # Patch 2: Disable torch.compile
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    # Patch 3: Pre-patch torch._inductor
    try:
        import torch._inductor.runtime.hints
        if not hasattr(torch._inductor.runtime.hints, 'attr_desc_fields'):
            torch._inductor.runtime.hints.attr_desc_fields = set()
    except: pass

    print("✓ Windows patches applied")

import argparse
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads HF_TOKEN from .env file
except ImportError:
    pass  # dotenv not required, can use environment variables directly


def save_local_copy(
    model_path: str,
    output_dir: Path,
    save_method: str,
    format_subdir: str
) -> Path:
    """
    Save a local copy of the model in the organized output directory.

    Args:
        model_path: Path to the model to save
        output_dir: Base output directory for this upload
        save_method: Save method (merged_16bit, merged_4bit, or lora)
        format_subdir: Subdirectory name (e.g., 'lora', 'merged_16bit')

    Returns:
        Path to the saved model
    """
    import tempfile

    save_dir = output_dir / format_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving local copy to: {save_dir}")

    # For LoRA, just copy the adapters
    if save_method == "lora":
        shutil.copytree(model_path, save_dir, dirs_exist_ok=True)
        print(f"✓ LoRA adapters copied")
        return save_dir

    # For merged models, need to load and save
    # Check if we're on Windows filesystem
    model_path_obj = Path(model_path).resolve()
    is_windows_fs = str(model_path_obj).startswith('/mnt/')

    temp_dir = None
    working_path = model_path

    if is_windows_fs:
        print("⚠ Detected Windows filesystem - copying to WSL native filesystem...")
        temp_dir = tempfile.mkdtemp(prefix='hf_save_', dir=str(Path.home() / 'tmp'))
        temp_model_path = Path(temp_dir) / 'model'
        shutil.copytree(model_path, temp_model_path)
        working_path = str(temp_model_path)
        print("✓ Copy complete")

    try:
        original_cwd = os.getcwd()
        if temp_dir:
            os.chdir(temp_dir)

        # Load and save merged model
        print("Loading model for local save...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=working_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False if save_method == "merged_16bit" else True,
        )

        model.save_pretrained_merged(str(save_dir), tokenizer, save_method=save_method)
        print(f"✓ Model saved locally")

        if temp_dir:
            os.chdir(original_cwd)

    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    return save_dir


def upload_standard_model(
    model_path: str,
    repo_id: str,
    hf_token: str,
    save_method: str = "merged_16bit",
    private: bool = False,
    output_dir: Path = None,
    save_local: bool = True
):
    """
    Upload standard model to HuggingFace Hub.

    Args:
        model_path: Path to saved model
        repo_id: HuggingFace repo ID (username/model-name)
        hf_token: HuggingFace write token
        save_method: Save method (merged_16bit, merged_4bit, or lora)
        private: Whether to make repo private
        output_dir: Directory to save local copy (optional)
        save_local: Whether to save a local copy
    """
    import tempfile

    print("=" * 60)
    print("UPLOADING STANDARD MODEL")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Save method: {save_method}")
    print(f"Private: {private}")
    print()

    # Save local copy first if requested
    if save_local and output_dir:
        format_subdir = save_method.replace("_", "-")  # e.g., "merged-16bit"
        if save_method == "lora":
            format_subdir = "lora"
        save_local_copy(model_path, output_dir, save_method, format_subdir)

    # Always use temp directory for upload to prevent push_to_hub_merged from
    # creating extra local copies in the current working directory
    model_path_obj = Path(model_path).resolve()
    is_windows_fs = str(model_path_obj).startswith('/mnt/')

    # Create temp directory in WSL native filesystem for reliability
    tmp_base = Path.home() / 'tmp'
    tmp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix='hf_upload_', dir=str(tmp_base))

    if is_windows_fs:
        print("\n⚠ Detected Windows filesystem - copying to WSL native filesystem...")
        temp_model_path = Path(temp_dir) / 'model'
        print(f"Copying model to: {temp_model_path}")
        shutil.copytree(model_path, temp_model_path)
        working_path = str(temp_model_path)
        print("✓ Copy complete\n")
    else:
        working_path = model_path

    try:
        # Change to temp directory to ensure push_to_hub_merged saves there
        # (it creates a local copy named after repo_id that we want cleaned up)
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        print(f"Working directory: {os.getcwd()}\n")

        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=working_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True if "4bit" not in save_method else False,
        )

        # Upload
        print(f"\nUploading to HuggingFace...")
        model.push_to_hub_merged(
            repo_id,
            tokenizer,
            save_method=save_method,
            token=hf_token,
            private=private
        )

        print(f"\n✓ Model uploaded successfully!")
        print(f"View at: https://huggingface.co/{repo_id}")

        # Restore original directory
        os.chdir(original_cwd)

    finally:
        # Clean up temp directory (always created to prevent stray local copies)
        if Path(temp_dir).exists():
            print(f"\nCleaning up temporary directory...")
            shutil.rmtree(temp_dir)


def create_gguf_versions(
    model_path: str,
    output_dir: Path,
    model_name: str,
    quantizations: List[str] = None,
    cleanup: bool = True
) -> List[str]:
    """
    Create GGUF versions of the model.

    Args:
        model_path: Path to model directory
        output_dir: Directory to save GGUF files (will create gguf/ subdirectory)
        model_name: Name for the GGUF files (e.g., "my-model" -> "my-model.gguf")
        quantizations: List of quantization methods (e.g., ["Q4_K_M", "Q5_K_M"])
        cleanup: Whether to cleanup temporary files after creation

    Returns:
        List of created GGUF file paths
    """
    import tempfile

    if quantizations is None:
        quantizations = ["Q4_K_M", "Q5_K_M", "Q8_0"]

    print("=" * 60)
    print("CREATING GGUF VERSIONS")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Quantizations: {', '.join(quantizations)}")
    print()

    # Create gguf subdirectory
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Use temp directory for intermediate files
    tmp_base = Path.home() / 'tmp'
    tmp_base.mkdir(parents=True, exist_ok=True)  # Ensure tmp directory exists
    temp_work_dir = Path(tempfile.mkdtemp(prefix='gguf_work_', dir=str(tmp_base)))

    try:
        # First, save merged model to temp location
        merged_dir = temp_work_dir / "merged_model"
        print("[1/4] Saving merged model locally...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # Need full precision for GGUF
        )
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"✓ Merged model saved to temporary location")

        # Clone llama.cpp if needed
        print("\n[2/4] Setting up llama.cpp...")
        llamacpp_dir = temp_work_dir / "llama.cpp"

        if not llamacpp_dir.exists():
            print("Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llamacpp_dir)],
                check=True,
                capture_output=True
            )
            print("Building llama.cpp...")

            # Build using CMake on Windows, make on Linux/Mac
            if sys.platform == 'win32':
                print("  Using CMake for Windows build...")
                build_dir = llamacpp_dir / "build"
                build_dir.mkdir(exist_ok=True)

                # Configure with CMake (disable CURL and GPU to minimize dependencies)
                subprocess.run(
                    ["cmake", "..",
                     "-DCMAKE_BUILD_TYPE=Release",
                     "-DLLAMA_CURL=OFF",
                     "-DGGML_CUDA=OFF",
                     "-DGGML_METAL=OFF"],
                    cwd=str(build_dir),
                    check=True,
                    capture_output=True
                )

                # Build
                subprocess.run(
                    ["cmake", "--build", ".", "--config", "Release"],
                    cwd=str(build_dir),
                    check=True,
                    capture_output=True
                )
            else:
                # Use make on Linux/Mac
                subprocess.run(
                    ["make", "-j"],
                    cwd=str(llamacpp_dir),
                    check=True,
                    capture_output=True
                )
        print(f"✓ llama.cpp ready")

        # Convert to GGUF base format in final location
        print("\n[3/4] Converting to GGUF base format (f16)...")
        base_gguf = gguf_dir / f"{model_name}.gguf"

        subprocess.run([
            "python",
            str(llamacpp_dir / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile", str(base_gguf),
            "--outtype", "f16"
        ], check=True)

        print(f"✓ Base GGUF created: {base_gguf.name}")

        # Create quantized versions in final location
        print(f"\n[4/4] Creating {len(quantizations)} quantized versions...")
        gguf_files = [base_gguf]

        # Find llama-quantize executable (different paths on Windows vs Linux)
        if sys.platform == 'win32':
            quantize_exe = llamacpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe"
        else:
            quantize_exe = llamacpp_dir / "llama-quantize"

        for quant in quantizations:
            output_file = gguf_dir / f"{model_name}-{quant}.gguf"
            print(f"  Creating {quant} quantization...")

            subprocess.run([
                str(quantize_exe),
                str(base_gguf),
                str(output_file),
                quant
            ], check=True)

            gguf_files.append(output_file)
            print(f"  ✓ {output_file.name}")

        print(f"\n✓ All GGUF files created: {len(gguf_files)} total")
        print(f"Saved to: {gguf_dir}")

        return [str(f) for f in gguf_files]

    finally:
        # Cleanup temporary work directory
        if cleanup and temp_work_dir.exists():
            print(f"\nCleaning up temporary files...")
            try:
                shutil.rmtree(temp_work_dir)
                print("✓ Temporary files removed")
            except PermissionError as e:
                print(f"⚠ Could not remove some temporary files (Windows file lock): {temp_work_dir}")
                print(f"  You can manually delete this directory later if needed")
                # Continue anyway - GGUF files are already saved


def upload_gguf_files(
    gguf_files: List[str],
    repo_id: str,
    hf_token: str
):
    """
    Upload GGUF files to HuggingFace Hub.

    Args:
        gguf_files: List of GGUF file paths
        repo_id: HuggingFace repo ID
        hf_token: HuggingFace write token
    """
    print("\n" + "=" * 60)
    print("UPLOADING GGUF FILES")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Files to upload: {len(gguf_files)}")
    print()

    api = HfApi()

    for i, gguf_file in enumerate(gguf_files, 1):
        file_path = Path(gguf_file)
        print(f"[{i}/{len(gguf_files)}] Uploading {file_path.name}...")

        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )

        print(f"  ✓ Uploaded")

    print(f"\n✓ All GGUF files uploaded!")
    print(f"View at: https://huggingface.co/{repo_id}/tree/main")


def create_upload_manifest(
    output_dir: Path,
    repo_id: str,
    training_run: str,
    formats_created: List[str],
    gguf_files: List[str] = None
) -> Path:
    """
    Create an upload manifest JSON file with metadata.

    Args:
        output_dir: Directory to save manifest
        repo_id: HuggingFace repo ID
        training_run: Training run timestamp (e.g., "20251122_143000")
        formats_created: List of formats created (e.g., ["lora", "merged_16bit", "gguf"])
        gguf_files: List of GGUF file names (optional)

    Returns:
        Path to manifest file
    """
    manifest = {
        "upload_timestamp": datetime.now().isoformat(),
        "training_run": training_run,
        "model_name": repo_id,
        "huggingface_url": f"https://huggingface.co/{repo_id}",
        "formats_created": formats_created,
        "directory_structure": {
            "lora": "lora/" if "lora" in formats_created else None,
            "merged_16bit": "merged-16bit/" if "merged_16bit" in formats_created else None,
            "merged_4bit": "merged-4bit/" if "merged_4bit" in formats_created else None,
            "gguf": "gguf/" if "gguf" in formats_created else None
        }
    }

    if gguf_files:
        manifest["gguf_quantizations"] = [Path(f).name for f in gguf_files]

    manifest_path = output_dir / "upload_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Upload manifest created: {manifest_path}")
    return manifest_path


def create_readme(
    output_dir: Path,
    repo_id: str,
    training_run: str,
    formats_created: List[str],
    model_name: str
) -> Path:
    """
    Create a simple README.md file in the output directory.

    Args:
        output_dir: Directory to save README
        repo_id: HuggingFace repo ID
        training_run: Training run timestamp
        formats_created: List of formats created
        model_name: Name of the model for GGUF file references

    Returns:
        Path to README file
    """
    readme_content = f"""# {model_name}

**Training Run:** `{training_run}`
**HuggingFace:** [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})

## Available Formats

"""

    for fmt in formats_created:
        if fmt == "lora":
            readme_content += "- **LoRA Adapters** (`lora/`) - Use with base model\n"
        elif fmt == "merged_16bit":
            readme_content += "- **Merged 16-bit** (`merged-16bit/`) - Full quality merged model (~14GB)\n"
        elif fmt == "merged_4bit":
            readme_content += "- **Merged 4-bit** (`merged-4bit/`) - Quantized merged model (~3.5GB)\n"
        elif fmt == "gguf":
            readme_content += "- **GGUF Quantizations** (`gguf/`) - For llama.cpp/Ollama\n"

    readme_content += f"""
## Directory Structure

```
{model_name}/
"""

    for fmt in formats_created:
        subdir = fmt.replace("_", "-") if fmt != "lora" else "lora"
        if fmt == "gguf":
            readme_content += f"""├── {subdir}/
│   ├── {model_name}.gguf (f16)
│   ├── {model_name}-Q4_K_M.gguf
│   ├── {model_name}-Q5_K_M.gguf
│   └── {model_name}-Q8_0.gguf
"""
        else:
            readme_content += f"├── {subdir}/\n"

    readme_content += """├── upload_manifest.json
└── README.md
```

## Usage

See the HuggingFace model card for detailed usage instructions.
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✓ README created: {readme_path}")
    return readme_path


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model directory (e.g., sft_output_rtx3090/20251122_143000/final_model)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo ID (username/model-name)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for organized artifacts (default: auto-detect from model_path)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace write token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "lora"],
        help="Save method for standard model"
    )
    parser.add_argument(
        "--create-gguf",
        action="store_true",
        help="Create and upload GGUF versions"
    )
    parser.add_argument(
        "--gguf-quantizations",
        type=str,
        nargs="+",
        default=["Q4_K_M", "Q5_K_M", "Q8_0"],
        help="GGUF quantization methods"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--skip-standard",
        action="store_true",
        help="Skip uploading standard model (only GGUF)"
    )

    args = parser.parse_args()

    # Get HuggingFace token
    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not hf_token:
        print("Error: HuggingFace token required. Provide via --token or HF_TOKEN env var")
        print("Get token from: https://huggingface.co/settings/tokens")
        return

    # Determine output directory
    model_path = Path(args.model_path)
    model_name = args.repo_id.split('/')[-1]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-detect: find training run directory
        # Pattern: sft_output_rtx3090/YYYYMMDD_HHMMSS/final_model
        # We want: sft_output_rtx3090/YYYYMMDD_HHMMSS/model-name/
        if model_path.name == "final_model" and model_path.parent.parent.name in ["sft_output_rtx3090", "kto_output_rtx3090"]:
            training_run_dir = model_path.parent
            output_dir = training_run_dir / model_name
        else:
            # Fallback: create in same directory as model
            output_dir = model_path.parent / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract training run timestamp for manifest
    training_run = output_dir.parent.name if output_dir.parent.name != "." else "unknown"

    print(f"\nOrganized output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print(f"Training run: {training_run}\n")

    formats_created = []
    gguf_files = None

    # Upload standard model
    if not args.skip_standard:
        upload_standard_model(
            model_path=str(model_path),
            repo_id=args.repo_id,
            hf_token=hf_token,
            save_method=args.save_method,
            private=args.private,
            output_dir=output_dir,
            save_local=True
        )
        formats_created.append(args.save_method)

    # Create and upload GGUF versions
    if args.create_gguf:
        gguf_files = create_gguf_versions(
            model_path=str(model_path),
            output_dir=output_dir,
            model_name=model_name,
            quantizations=args.gguf_quantizations,
            cleanup=True
        )

        upload_gguf_files(
            gguf_files=gguf_files,
            repo_id=args.repo_id,
            hf_token=hf_token
        )
        formats_created.append("gguf")

    # Create manifest and README
    print("\n" + "=" * 60)
    print("CREATING DOCUMENTATION")
    print("=" * 60)

    create_upload_manifest(
        output_dir=output_dir,
        repo_id=args.repo_id,
        training_run=training_run,
        formats_created=formats_created,
        gguf_files=gguf_files
    )

    create_readme(
        output_dir=output_dir,
        repo_id=args.repo_id,
        training_run=training_run,
        formats_created=formats_created,
        model_name=model_name
    )

    # Final summary
    print("\n" + "=" * 60)
    print("✓ ALL UPLOADS COMPLETE!")
    print("=" * 60)
    print(f"\nLocal artifacts saved to: {output_dir}")
    print(f"HuggingFace model: https://huggingface.co/{args.repo_id}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    for fmt in formats_created:
        subdir = fmt.replace("_", "-") if fmt not in ["lora", "gguf"] else fmt
        print(f"  ├── {subdir}/")
    print(f"  ├── upload_manifest.json")
    print(f"  └── README.md")


if __name__ == "__main__":
    main()
