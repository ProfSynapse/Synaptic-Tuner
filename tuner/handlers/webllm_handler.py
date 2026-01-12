"""
WebLLM conversion handler.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/webllm_handler.py
Purpose: Convert trained models to WebLLM/MLC format for browser deployment
Used by: Router when 'webllm' command is invoked
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from tuner.handlers.base import BaseHandler
from tuner.discovery import TrainingRunDiscovery, CheckpointDiscovery
from tuner.ui import (
    print_menu,
    print_header,
    print_config,
    print_success,
    print_error,
    print_info,
    print_table,
    print_checkpoint_table,
    confirm,
    prompt,
    BOX,
)
from tuner.utils.validation import validate_repo_id, load_env_file


# WSL patch code for MLC imports
WSL_PATCH_CODE = '''
import os
import sys
import pathlib

# Preserve only Linux paths to avoid WSL permission errors
linux_paths = [p for p in os.environ.get('PATH', '').split(':') if not p.startswith('/mnt/c')]
cuda_path = '/usr/local/cuda/bin'
if os.path.isdir(cuda_path):
    linux_paths.insert(0, cuda_path)
if linux_paths:
    os.environ['PATH'] = ':'.join(linux_paths)

# Monkey-patch pathlib to handle PermissionError on Windows paths
_original_is_dir = pathlib.Path.is_dir
def _safe_is_dir(self):
    try:
        return _original_is_dir(self)
    except PermissionError:
        return False
pathlib.Path.is_dir = _safe_is_dir

_original_stat = pathlib.Path.stat
def _safe_stat(self, *args, **kwargs):
    try:
        return _original_stat(self, *args, **kwargs)
    except PermissionError:
        raise FileNotFoundError(f"Skipped: {self}")
pathlib.Path.stat = _safe_stat
'''

# Known WASM sources for common architectures
WASM_SOURCES = {
    "qwen3-4b": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80/Qwen3-4B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    "qwen3-1.7b": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80/Qwen3-1.7B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    "llama3-8b": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80/Llama-3-8B-Instruct-q4f16_1-MLC-webgpu.wasm",
}


class WebLLMHandler(BaseHandler):
    """
    Handler for WebLLM/MLC conversion workflow.

    Orchestrates the complete conversion process:
    1. Select training run
    2. Check for existing merged model (skip merge if exists)
    3. Merge LoRA adapters if needed
    4. Convert to MLC format
    5. Download appropriate WASM
    6. Optionally upload to HuggingFace

    Example:
        handler = WebLLMHandler()
        exit_code = handler.handle()
    """

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "webllm"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python -m tuner webllm'."""
        return True

    def handle(self) -> int:
        """
        Execute WebLLM conversion workflow.

        Returns:
            int: Exit code (0 = success, non-zero = failure)
        """
        print_header("WEBLLM CONVERSION", "Convert model for browser deployment")

        # Step 1: Select model type
        model_type = print_menu([
            ("sft", f"{BOX['bullet']} SFT model"),
            ("kto", f"{BOX['bullet']} KTO model"),
        ], "Select model type:")

        if not model_type:
            return 0

        # Step 2: List training runs
        discovery = TrainingRunDiscovery(repo_root=self.repo_root)
        runs = discovery.discover(trainer_type=model_type, limit=10)

        if not runs:
            print_error(f"No training runs found for {model_type.upper()}")
            return 1

        # Step 3: Display runs and select
        run_data = []
        for i, run in enumerate(runs, 1):
            has_final = "✓" if (run / "final_model").exists() else "-"
            # Check if merged model exists
            merged_path = run / "merged-16bit"
            has_merged = merged_path.exists() and any(merged_path.glob("*.safetensors"))
            status = "merged" if has_merged else "-"
            # Count checkpoints
            checkpoints_dir = run / "checkpoints"
            checkpoint_count = 0
            if checkpoints_dir.exists():
                checkpoint_count = len(list(checkpoints_dir.glob("checkpoint-*")))
            run_data.append([str(i), run.name, has_final, str(checkpoint_count), status])

        print_table(run_data, ["#", "Training Run", "Final", "CPs", "Merged"],
                   title=f"Available {model_type.upper()} Training Runs")

        while True:
            try:
                sel = prompt(f"Select run (1-{len(runs)})")
                idx = int(sel) - 1
                if 0 <= idx < len(runs):
                    selected_run = runs[idx]
                    break
            except ValueError:
                pass
            print_error("Invalid selection.")

        # Step 4: Select checkpoint with metrics
        checkpoints = CheckpointDiscovery.discover(selected_run)

        if not checkpoints:
            print_error("No checkpoints found in training run")
            return 1

        # If only one checkpoint (final_model), use it directly
        if len(checkpoints) == 1 and checkpoints[0].is_final:
            print_info("Using final_model")
            lora_path = checkpoints[0].path
        else:
            # Display checkpoint table with loss values
            print_checkpoint_table(checkpoints, model_type)

            while True:
                try:
                    sel = prompt(f"Select checkpoint (1-{len(checkpoints)})", "1")
                    idx = int(sel) - 1
                    if 0 <= idx < len(checkpoints):
                        lora_path = checkpoints[idx].path
                        break
                except ValueError:
                    pass
                print_error("Invalid selection.")

        # Step 5: Check for existing merged model
        merged_path = selected_run / "merged-16bit"
        needs_merge = not (merged_path.exists() and any(merged_path.glob("*.safetensors")))

        if needs_merge:
            print_info(f"Merged model not found. Will merge LoRA from: {lora_path.name}")
        else:
            print_success(f"Found merged model at: {merged_path.name}")

        # Step 5: Configure output
        default_name = selected_run.name.replace("_", "-")
        model_name = prompt("Model name for WebLLM", default_name)

        output_dir = self.repo_root / "webllm_output" / f"{model_name}-q4f16_1-MLC"

        # Step 6: Quantization selection
        quant = print_menu([
            ("q4f16_1", f"{BOX['star']} q4f16_1 (recommended, ~2.2GB)"),
            ("q4f32_1", f"{BOX['bullet']} q4f32_1 (higher precision, ~2.8GB)"),
        ], "Select quantization:")

        if not quant:
            return 0

        # Step 7: Upload option
        upload_to_hf = confirm("Upload to HuggingFace after conversion?")
        repo_id = None

        if upload_to_hf:
            env_file = self.repo_root / ".env"
            load_env_file(env_file)

            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print_error("HF_TOKEN not found. Skipping upload.")
                upload_to_hf = False
            else:
                hf_username = os.environ.get("HF_USERNAME", "")
                if hf_username:
                    repo_id = f"{hf_username}/{model_name}-MLC"
                else:
                    repo_id = prompt("HuggingFace repo ID", f"username/{model_name}-MLC")
                    if not validate_repo_id(repo_id):
                        print_error("Invalid repo ID")
                        upload_to_hf = False

        # Step 8: Confirmation
        config_display = {
            "Source": str(lora_path.relative_to(self.repo_root)) if needs_merge
                     else str(merged_path.relative_to(self.repo_root)),
            "Merge Required": "Yes" if needs_merge else "No (using existing)",
            "Output": str(output_dir.relative_to(self.repo_root)),
            "Quantization": quant,
        }
        if upload_to_hf:
            config_display["Upload To"] = repo_id

        print_config(config_display, "WebLLM Conversion")

        if not confirm("Start conversion?"):
            print_info("Cancelled.")
            return 0

        # Step 9: Execute conversion
        try:
            # Merge if needed
            if needs_merge:
                print_info("Step 1/4: Merging LoRA adapters...")
                merged_path = self._merge_lora(lora_path, selected_run / "merged-16bit")
                if not merged_path:
                    return 1
                print_success("Merge complete!")

            # Extract text-only weights (for VL models)
            print_info("Step 2/4: Preparing weights...")
            text_only_path = self._prepare_text_only(merged_path)
            if not text_only_path:
                text_only_path = merged_path  # Use as-is if not VL
            print_success("Weights prepared!")

            # Convert to MLC
            print_info("Step 3/4: Converting to MLC format...")
            output_dir.mkdir(parents=True, exist_ok=True)
            if not self._convert_to_mlc(text_only_path, output_dir, quant):
                return 1
            print_success("MLC conversion complete!")

            # Download WASM
            print_info("Step 4/4: Downloading WASM library...")
            if not self._download_wasm(output_dir, text_only_path):
                print_error("Failed to download WASM. You may need to compile it manually.")
            else:
                print_success("WASM downloaded!")

            # Upload if requested
            if upload_to_hf and repo_id:
                print_info("Uploading to HuggingFace...")
                if self._upload_to_hf(output_dir, repo_id):
                    print_success(f"Uploaded to: https://huggingface.co/{repo_id}")
                else:
                    print_error("Upload failed")

            print_success(f"\nWebLLM model ready at: {output_dir}")
            return 0

        except Exception as e:
            print_error(f"Conversion failed: {e}")
            return 1

    def _merge_lora(self, lora_path: Path, output_path: Path) -> Path:
        """Merge LoRA adapters with base model."""
        python = self._get_unsloth_python()
        if not python:
            print_error("Unsloth environment not found")
            return None

        script = f'''
{WSL_PATCH_CODE}
import os
os.chdir(os.path.expanduser("~"))

from unsloth import FastLanguageModel
import torch

lora_path = "{lora_path}"
output_path = "{output_path}"

print(f"Loading from: {{lora_path}}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_path,
    max_seq_length=32768,
    dtype=torch.float16,
    load_in_4bit=True,
)

print("Saving merged 16-bit model...")
import os
os.makedirs(output_path, exist_ok=True)
model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
print("Done!")
'''

        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print_error(f"Merge failed: {result.stderr}")
            return None

        return output_path

    def _prepare_text_only(self, merged_path: Path) -> Path:
        """Extract text-only weights from VL model if needed."""
        config_path = merged_path / "config.json"
        if not config_path.exists():
            return merged_path

        with open(config_path) as f:
            config = json.load(f)

        # Check if it's a VL model
        if config.get("model_type") != "qwen3_vl":
            return merged_path

        print_info("Detected VL model, extracting text-only weights...")

        output_path = merged_path.parent / "qwen3-text-only"
        output_path.mkdir(parents=True, exist_ok=True)

        python = self._get_unsloth_python()
        script = f'''
import os
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import shutil

model_dir = Path("{merged_path}")
output_dir = Path("{output_path}")

def rename_key(key):
    if key.startswith("model.visual") or "merger" in key:
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key

with open(model_dir / "model.safetensors.index.json") as f:
    index = json.load(f)

shard_files = sorted(set(index["weight_map"].values()))
all_tensors = {{}}

for shard_file in shard_files:
    with safe_open(model_dir / shard_file, framework="pt") as f:
        for key in f.keys():
            new_key = rename_key(key)
            if new_key:
                all_tensors[new_key] = f.get_tensor(key)

save_file(all_tensors, output_dir / "model.safetensors")

new_weight_map = {{k: "model.safetensors" for k in all_tensors.keys()}}
new_index = {{"metadata": {{}}, "weight_map": new_weight_map}}
with open(output_dir / "model.safetensors.index.json", "w") as f:
    json.dump(new_index, f, indent=2)

qwen3_config = {{
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "pad_token_id": 151654,
}}
with open(output_dir / "config.json", "w") as f:
    json.dump(qwen3_config, f, indent=2)

for tf in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "added_tokens.json", "merges.txt", "vocab.json"]:
    src = model_dir / tf
    if src.exists():
        shutil.copy(src, output_dir / tf)
'''

        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print_error(f"Text extraction failed: {result.stderr}")
            return merged_path

        return output_path

    def _convert_to_mlc(self, model_path: Path, output_path: Path, quant: str) -> bool:
        """Convert model to MLC format."""
        python = self._get_mlc_python()
        if not python:
            print_error("MLC-LLM not found")
            return False

        # Weight conversion
        script = f'''
{WSL_PATCH_CODE}
import os
os.chdir(os.path.expanduser("~"))

from mlc_llm.cli.convert_weight import main
main(["{model_path}", "--quantization", "{quant}", "-o", "{output_path}"])
'''

        result = subprocess.run(
            [python, "-c", script],
            capture_output=False,
            timeout=900
        )

        if result.returncode != 0:
            return False

        # Config generation
        script = f'''
{WSL_PATCH_CODE}
import os
os.chdir(os.path.expanduser("~"))

from mlc_llm.cli.gen_config import main
main(["{model_path}", "--quantization", "{quant}", "--conv-template", "chatml",
      "--context-window-size", "32768", "-o", "{output_path}"])
'''

        result = subprocess.run(
            [python, "-c", script],
            capture_output=False,
            timeout=120
        )

        return result.returncode == 0

    def _download_wasm(self, output_path: Path, model_path: Path) -> bool:
        """Download appropriate WASM file."""
        # Detect model architecture
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            hidden_size = config.get("hidden_size", 0)
            model_type = config.get("model_type", "")

            # Match to known WASM based on hidden_size
            # Qwen3 sizes: 0.6B=1024, 1.7B=2048, 4B=2560, 8B=4096
            if "qwen3" in model_type.lower():
                if hidden_size == 4096:  # 8B
                    wasm_url = WASM_SOURCES.get("qwen3-8b", WASM_SOURCES["qwen3-4b"])
                elif hidden_size == 2560:  # 4B
                    wasm_url = WASM_SOURCES["qwen3-4b"]
                elif hidden_size == 2048:  # 1.7B
                    wasm_url = WASM_SOURCES["qwen3-1.7b"]
                elif hidden_size == 1024:  # 0.6B
                    wasm_url = WASM_SOURCES.get("qwen3-0.6b", WASM_SOURCES["qwen3-1.7b"])
                else:
                    print_info(f"No prebuilt WASM for hidden_size={hidden_size}")
                    return False
            else:
                print_info(f"No prebuilt WASM for model_type={model_type}")
                return False
        else:
            # Default to Qwen3-4B
            wasm_url = WASM_SOURCES["qwen3-4b"]

        wasm_name = wasm_url.split("/")[-1]
        wasm_path = output_path / wasm_name

        import urllib.request
        try:
            urllib.request.urlretrieve(wasm_url, wasm_path)
            return True
        except Exception as e:
            print_error(f"WASM download failed: {e}")
            return False

    def _upload_to_hf(self, output_path: Path, repo_id: str) -> bool:
        """Upload to HuggingFace."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(output_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {repo_id} MLC model for WebLLM"
            )
            return True
        except Exception as e:
            print_error(f"Upload failed: {e}")
            return False

    def _get_unsloth_python(self) -> str:
        """Get Python with unsloth installed."""
        candidates = [
            "/home/profsynapse/.conda/envs/unsloth_latest/bin/python",
            "/home/profsynapse/.conda/envs/unsloth_env/bin/python",
            os.path.expanduser("~/.conda/envs/unsloth_latest/bin/python"),
            os.path.expanduser("~/.conda/envs/unsloth_env/bin/python"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _get_mlc_python(self) -> str:
        """Get Python with MLC-LLM installed."""
        candidates = [
            "/home/profsynapse/miniconda3/bin/python",
            os.path.expanduser("~/miniconda3/bin/python"),
            os.path.expanduser("~/anaconda3/bin/python"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None
