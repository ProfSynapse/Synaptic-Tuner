"""
WebGPU/MLC-LLM Converter for Browser Deployment.

This converter uses MLC-LLM to create WebGPU-compatible model files
that can run in browsers via WebLLM.

Output files:
- Quantized weights in MLC format
- WebAssembly inference library (.wasm)
- Configuration files (mlc-chat-config.json, tokenizer)

Usage:
    from shared.upload.converters.webgpu import WebGPUConverter

    converter = WebGPUConverter()
    files = converter.convert(
        model_path="/path/to/model",
        output_dir=Path("/path/to/output"),
        quantizations=["q4f16_1"],
        model_name="my-model"
    )
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from contextlib import contextmanager

# Brand colors for spinners
AQUA = "#00A99D"
PURPLE = "#93278F"

# Try to import Rich for nice spinners
try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Default quantizations for WebGPU
DEFAULT_QUANTIZATIONS = ["q4f16_1"]

# Supported quantization methods
SUPPORTED_QUANTIZATIONS = [
    "q0f16",      # No quantization, float16
    "q0f32",      # No quantization, float32
    "q4f16_1",    # 4-bit with float16 (recommended)
    "q4f32_1",    # 4-bit with float32
    "q3f16_1",    # 3-bit with float16
]

# Conversation templates for common models
CONV_TEMPLATES = {
    "mistral": "mistral_default",
    "qwen": "chatml",  # Qwen3 uses ChatML
    "qwen3": "chatml",
    "llama": "llama-3",
    "phi": "phi-3",
    "gemma": "gemma",
}

# Prebuilt WASM URLs for common model architectures
# Fine-tuned models can reuse the base architecture's WASM
PREBUILT_WASM_BASE = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_80"
PREBUILT_WASMS = {
    # Qwen3 models (architecture-compatible with fine-tuned versions)
    ("qwen3", "4b", "q4f16_1"): "Qwen3-4B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    ("qwen3", "4b", "q4f32_1"): "Qwen3-4B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    ("qwen3", "4b", "q0f16"): "Qwen3-4B-q0f16-ctx4k_cs1k-webgpu.wasm",
    ("qwen3", "1.7b", "q4f16_1"): "Qwen3-1.7B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    ("qwen3", "1.7b", "q4f32_1"): "Qwen3-1.7B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    ("qwen3", "0.6b", "q4f16_1"): "Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    ("qwen3", "8b", "q4f16_1"): "Qwen3-8B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    # Llama 3 models
    ("llama", "8b", "q4f16_1"): "Llama-3-8B-Instruct-q4f16_1-MLC-ctx4k_cs1k-webgpu.wasm",
    ("llama", "8b", "q4f32_1"): "Llama-3-8B-Instruct-q4f32_1-MLC-ctx4k_cs1k-webgpu.wasm",
    # Mistral models
    ("mistral", "7b", "q4f16_1"): "Mistral-7B-Instruct-v0.2-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    # Phi models
    ("phi", "3.5b", "q4f16_1"): "Phi-3.5-mini-instruct-q4f16_1-MLC-ctx4k_cs1k-webgpu.wasm",
}

# WSL monkey-patch code for MLC imports (handles permission errors on Windows paths)
WSL_PATCH_CODE = '''
import os
import sys
import pathlib

# Preserve Linux paths and CUDA, avoid Windows paths that cause permission errors
linux_paths = [p for p in os.environ.get('PATH', '').split(':') if not p.startswith('/mnt/c')]
# Ensure CUDA is in PATH for nvcc
cuda_paths = ['/usr/local/cuda/bin', '/usr/local/cuda-12.8/bin', '/usr/local/cuda-12/bin', '/usr/local/cuda-11/bin']
for cuda_path in cuda_paths:
    if os.path.isdir(cuda_path) and cuda_path not in linux_paths:
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

def get_python_executable() -> str:
    """Get the Python executable path where MLC-LLM is installed."""
    import os
    # Prefer miniconda base where MLC is installed
    candidates = [
        "/home/profsynapse/miniconda3/bin/python",
        os.path.expanduser("~/miniconda3/bin/python"),
        os.path.expanduser("~/anaconda3/bin/python"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Fallback to current
    import sys
    return sys.executable


def detect_cuda_version() -> str:
    """Detect CUDA version and return appropriate MLC package suffix."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Check for CUDA version from nvcc if available
            nvcc_result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=10
            )
            if nvcc_result.returncode == 0 and "12." in nvcc_result.stdout:
                # CUDA 12.x - use cu128 (compatible with 12.4-12.8)
                return "cu128"
            # Default to cu128 if nvidia-smi works
            return "cu128"
    except Exception:
        pass
    # No CUDA detected, use CPU
    return "cpu"

def get_working_directory() -> str:
    """Get a working directory that avoids WSL permission issues."""
    import os
    # Try home directory first, then /tmp
    home = os.path.expanduser("~")
    if os.path.isdir(home) and not home.startswith("/mnt/c"):
        return home
    return "/tmp"

# Spinner frames
BUBBLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


@contextmanager
def branded_spinner(message: str):
    """Context manager for branded spinner during long operations."""
    if RICH_AVAILABLE:
        console = Console()
        with Live(
            Text.assemble(
                ("◆ ", f"{PURPLE}"),
                (message, f"{AQUA}"),
                (" ", ""),
                ("⠋", f"{PURPLE}"),
            ),
            console=console,
            refresh_per_second=10,
            transient=True,
        ) as live:
            frame_idx = [0]
            stop_flag = [False]

            def update_spinner():
                while not stop_flag[0]:
                    frame_idx[0] = (frame_idx[0] + 1) % len(BUBBLE_FRAMES)
                    live.update(Text.assemble(
                        ("◆ ", f"{PURPLE}"),
                        (message, f"{AQUA}"),
                        (" ", ""),
                        (BUBBLE_FRAMES[frame_idx[0]], f"{PURPLE}"),
                    ))
                    time.sleep(0.1)

            spinner_thread = threading.Thread(target=update_spinner, daemon=True)
            spinner_thread.start()
            try:
                yield
            finally:
                stop_flag[0] = True
                spinner_thread.join(timeout=0.5)
    else:
        print(f"  {message}...", end="", flush=True)
        try:
            yield
        finally:
            print(" done")


class WebGPUConverter:
    """
    WebGPU/MLC-LLM converter for browser deployment.

    Converts HuggingFace models to MLC format and compiles WebAssembly
    libraries for WebGPU inference in browsers.
    """

    def __init__(self, mlc_llm_path: Optional[Path] = None):
        """
        Initialize the converter.

        Args:
            mlc_llm_path: Path to mlc_llm installation (uses system if None)
        """
        self.mlc_llm_path = mlc_llm_path
        self._mlc_available: Optional[bool] = None

    def check_mlc_available(self) -> Tuple[bool, str]:
        """
        Check if MLC-LLM is available.

        Returns:
            Tuple of (available, message)
        """
        try:
            # Use Python with WSL patch to check MLC availability
            check_code = WSL_PATCH_CODE + """
from mlc_llm import MLCEngine
print("OK")
"""
            result = subprocess.run(
                [get_python_executable(), "-c", check_code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=get_working_directory()
            )
            if result.returncode == 0 and "OK" in result.stdout:
                return True, "MLC-LLM is available"
            return False, f"MLC-LLM import failed: {result.stderr[:200]}"
        except FileNotFoundError:
            return False, "Python not found"
        except subprocess.TimeoutExpired:
            return False, "MLC-LLM check timed out"
        except Exception as e:
            return False, f"Error checking MLC-LLM: {e}"

    def setup_mlc_llm(self) -> bool:
        """
        Setup MLC-LLM if not available.

        Returns:
            True if setup successful
        """
        print("Setting up MLC-LLM...")

        available, msg = self.check_mlc_available()
        if available:
            print(f"  ✓ {msg}")
            return True

        print(f"  MLC-LLM not found. Installing...")

        try:
            # Detect CUDA version and select appropriate packages
            cuda_suffix = detect_cuda_version()
            llm_pkg = f"mlc-llm-nightly-{cuda_suffix}"
            ai_pkg = f"mlc-ai-nightly-{cuda_suffix}"
            print(f"  Detected platform: {cuda_suffix}, installing {llm_pkg}...")

            # Install mlc-llm-nightly with correct CUDA/CPU suffix
            python_exe = get_python_executable()
            with branded_spinner("Installing MLC-LLM"):
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install", "--pre", "-U",
                     llm_pkg, ai_pkg, "-f", "https://mlc.ai/wheels"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

            if result.returncode != 0:
                print(f"  ✗ Installation failed: {result.stderr[:500]}")
                return False

            # Verify installation
            available, msg = self.check_mlc_available()
            if available:
                print(f"  ✓ MLC-LLM installed successfully")
                return True
            else:
                print(f"  ✗ Installation verification failed: {msg}")
                return False

        except subprocess.TimeoutExpired:
            print("  ✗ Installation timed out")
            return False
        except Exception as e:
            print(f"  ✗ Installation error: {e}")
            return False

    def detect_conv_template(self, model_path: Path) -> str:
        """
        Detect appropriate conversation template for model.

        Args:
            model_path: Path to model directory

        Returns:
            Conversation template name
        """
        # Check config.json for model type
        config_json = model_path / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    config = json.load(f)

                model_type = config.get("model_type", "").lower()
                architectures = config.get("architectures", [])

                # Match against known templates
                for key, template in CONV_TEMPLATES.items():
                    if key in model_type:
                        return template
                    for arch in architectures:
                        if key in arch.lower():
                            return template
            except (json.JSONDecodeError, KeyError):
                pass

        # Check adapter_config.json for base model
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config) as f:
                    config = json.load(f)
                base_name = config.get("base_model_name_or_path", "").lower()

                for key, template in CONV_TEMPLATES.items():
                    if key in base_name:
                        return template
            except (json.JSONDecodeError, KeyError):
                pass

        # Default to mistral template
        return "mistral_default"

    def convert_weights(
        self,
        model_path: Path,
        output_dir: Path,
        quantization: str = "q4f16_1",
    ) -> bool:
        """
        Convert model weights to MLC format.

        Args:
            model_path: Path to HuggingFace model
            output_dir: Output directory for converted weights
            quantization: Quantization method

        Returns:
            True if successful
        """
        # Use Python with WSL patch for MLC CLI
        convert_code = WSL_PATCH_CODE + f"""
from mlc_llm.cli.convert_weight import main
main(['{model_path}', '--quantization', '{quantization}', '-o', '{output_dir}'])
"""
        try:
            with branded_spinner(f"Converting weights ({quantization})"):
                result = subprocess.run(
                    [get_python_executable(), "-c", convert_code],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes for large models
                    cwd=get_working_directory()
                )

            if result.returncode != 0:
                print(f"  ✗ Weight conversion failed:")
                print(f"    {result.stderr[:500]}")
                return False

            print(f"  ✓ Weights converted to MLC format")
            return True

        except subprocess.TimeoutExpired:
            print("  ✗ Weight conversion timed out")
            return False
        except Exception as e:
            print(f"  ✗ Weight conversion error: {e}")
            return False

    def generate_config(
        self,
        model_path: Path,
        output_dir: Path,
        quantization: str = "q4f16_1",
        conv_template: Optional[str] = None,
        context_window_size: int = 4096,
        prefill_chunk_size: int = 1024,
    ) -> bool:
        """
        Generate MLC chat configuration.

        Args:
            model_path: Path to original HuggingFace model
            output_dir: Output directory (with converted weights)
            quantization: Quantization method used
            conv_template: Conversation template (auto-detected if None)
            context_window_size: Maximum context window
            prefill_chunk_size: Prefill chunk size for memory efficiency

        Returns:
            True if successful
        """
        if conv_template is None:
            conv_template = self.detect_conv_template(model_path)

        # Use Python with WSL patch for MLC CLI
        gen_config_code = WSL_PATCH_CODE + f"""
from mlc_llm.cli.gen_config import main
main(['{model_path}', '--quantization', '{quantization}', '--conv-template', '{conv_template}', '--context-window-size', '{context_window_size}', '--prefill-chunk-size', '{prefill_chunk_size}', '-o', '{output_dir}'])
"""
        try:
            with branded_spinner("Generating MLC config"):
                result = subprocess.run(
                    [get_python_executable(), "-c", gen_config_code],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=get_working_directory()
                )

            if result.returncode != 0:
                print(f"  ✗ Config generation failed:")
                print(f"    {result.stderr[:500]}")
                return False

            print(f"  ✓ Config generated (template: {conv_template})")
            return True

        except subprocess.TimeoutExpired:
            print("  ✗ Config generation timed out")
            return False
        except Exception as e:
            print(f"  ✗ Config generation error: {e}")
            return False

    def _detect_model_size(self, model_path: Path) -> Optional[str]:
        """Detect model size from config for prebuilt WASM lookup."""
        config_json = model_path / "config.json"
        if not config_json.exists():
            return None

        try:
            with open(config_json) as f:
                config = json.load(f)

            # Check hidden_size to determine approximate size
            hidden_size = config.get("hidden_size", 0)
            num_layers = config.get("num_hidden_layers", 0)

            # Approximate size detection based on architecture
            if hidden_size <= 1024 and num_layers <= 16:
                return "0.6b"
            elif hidden_size <= 2048 and num_layers <= 28:
                return "1.7b"
            elif hidden_size <= 2560 and num_layers <= 36:
                return "4b"
            elif hidden_size <= 4096 and num_layers <= 32:
                return "8b"
            else:
                return None
        except (json.JSONDecodeError, KeyError):
            return None

    def _download_prebuilt_wasm(
        self,
        model_type: str,
        model_size: str,
        quantization: str,
        output_path: Path,
    ) -> bool:
        """
        Download prebuilt WASM from WebLLM binary repository.

        Fine-tuned models can reuse the base architecture's WASM since
        the execution code is the same - only the weights differ.
        """
        key = (model_type.lower(), model_size.lower(), quantization.lower())

        if key not in PREBUILT_WASMS:
            print(f"  ⚠ No prebuilt WASM available for {model_type} {model_size} {quantization}")
            print(f"    Available architectures: qwen3, llama, mistral, phi")
            return False

        wasm_file = PREBUILT_WASMS[key]
        url = f"{PREBUILT_WASM_BASE}/{wasm_file}"

        print(f"  Downloading prebuilt WASM from WebLLM...")
        print(f"    {wasm_file}")

        try:
            result = subprocess.run(
                ["curl", "-L", url, "-o", str(output_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"  ✗ Download failed: {result.stderr[:200]}")
                return False

            if not output_path.exists() or output_path.stat().st_size < 1000:
                print(f"  ✗ Download produced invalid file")
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Downloaded {output_path.name} ({size_mb:.1f} MB)")
            return True

        except Exception as e:
            print(f"  ✗ Download error: {e}")
            return False

    def compile_webgpu(
        self,
        config_path: Path,
        output_path: Path,
        model_name: str = "model",
        model_path: Optional[Path] = None,
        quantization: str = "q4f16_1",
    ) -> bool:
        """
        Compile model for WebGPU (creates .wasm file).

        If compilation fails, attempts to download a prebuilt WASM
        from WebLLM's binary repository (works for fine-tuned models
        since they share the base architecture).

        Args:
            config_path: Path to mlc-chat-config.json
            output_path: Path for output .wasm file
            model_name: Model name for system lib prefix
            model_path: Path to model for architecture detection
            quantization: Quantization method used

        Returns:
            True if successful
        """
        # Generate a safe system lib prefix from model name
        # Replace non-alphanumeric chars with underscore
        import re
        safe_prefix = re.sub(r'[^a-zA-Z0-9]', '_', model_name)

        # Use Python with WSL patch for MLC CLI
        # --system-lib-prefix is required for newer MLC-LLM versions
        compile_code = WSL_PATCH_CODE + f"""
from mlc_llm.cli.compile import main
main(['{config_path}', '--device', 'webgpu', '--system-lib-prefix', '{safe_prefix}', '-o', '{output_path}'])
"""
        try:
            with branded_spinner("Compiling WebGPU library"):
                result = subprocess.run(
                    [get_python_executable(), "-c", compile_code],
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes for compilation
                    cwd=get_working_directory()
                )

            if result.returncode != 0:
                print(f"  ✗ WebGPU compilation failed (TVM error)")

                # Attempt prebuilt fallback if model_path provided
                if model_path:
                    print("  → Trying prebuilt WASM fallback...")
                    model_type = self.detect_conv_template(model_path)
                    # Map conv template back to model type
                    type_map = {"chatml": "qwen3", "llama-3": "llama", "mistral_default": "mistral", "phi-3": "phi"}
                    model_type = type_map.get(model_type, model_type)

                    model_size = self._detect_model_size(model_path)
                    if model_size:
                        return self._download_prebuilt_wasm(model_type, model_size, quantization, output_path)

                print(f"    {result.stderr[:300]}")
                return False

            if not output_path.exists():
                print(f"  ✗ Output file not created: {output_path}")
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_path.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print("  ✗ WebGPU compilation timed out")
            return False
        except Exception as e:
            print(f"  ✗ WebGPU compilation error: {e}")
            return False

    def convert(
        self,
        model_path: str | Path,
        output_dir: Path,
        quantizations: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        conv_template: Optional[str] = None,
        context_window_size: int = 4096,
        compile_wasm: bool = True,
        cleanup_temp: bool = True,
        **kwargs,
    ) -> List[Path]:
        """
        Convert model to WebGPU format.

        This method:
        1. Converts weights to MLC format with quantization
        2. Generates MLC chat configuration
        3. Optionally compiles WebAssembly library for WebGPU

        Args:
            model_path: Path to model (LoRA adapters or full model)
            output_dir: Directory to save converted files
            quantizations: List of quantization methods (default: q4f16_1)
            model_name: Name for output files
            conv_template: Conversation template (auto-detected if None)
            context_window_size: Maximum context window
            compile_wasm: Whether to compile .wasm file
            cleanup_temp: Whether to cleanup temporary files

        Returns:
            List of created file/directory paths
        """
        model_path = Path(model_path)
        quantizations = quantizations or DEFAULT_QUANTIZATIONS
        model_name = model_name or model_path.name

        # Setup MLC-LLM if needed
        if not self.setup_mlc_llm():
            raise RuntimeError("Failed to setup MLC-LLM")

        # Detect conversation template
        if conv_template is None:
            conv_template = self.detect_conv_template(model_path)

        print("\n" + "=" * 60)
        print("WEBGPU CONVERSION")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Output: {output_dir}")
        print(f"Name: {model_name}")
        print(f"Quantizations: {', '.join(quantizations)}")
        print(f"Conv Template: {conv_template}")
        print(f"Context Window: {context_window_size}")
        print(f"Compile WASM: {compile_wasm}")
        print("=" * 60)

        # Create output directory
        webgpu_dir = output_dir / "webgpu"
        webgpu_dir.mkdir(parents=True, exist_ok=True)

        created_paths: List[Path] = []

        # Check if this is a LoRA model that needs merging first
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            print("\n⚠ LoRA adapters detected - merging first...")
            # Save merged model alongside the training run (persistent, not temp)
            merged_path = model_path.parent / "merged-16bit"

            # Check if already merged
            if merged_path.exists() and any(merged_path.glob("*.safetensors")):
                print(f"  ✓ Found existing merged model at: {merged_path}")
                model_path = merged_path
            else:
                # Use Unsloth via subprocess for reliable 4-bit -> 16-bit merge
                merged_path.mkdir(parents=True, exist_ok=True)

                # Find Unsloth Python
                unsloth_python = None
                for p in [
                    "/home/profsynapse/.conda/envs/unsloth_latest/bin/python",
                    os.path.expanduser("~/.conda/envs/unsloth_latest/bin/python"),
                    "/home/profsynapse/.conda/envs/unsloth_env/bin/python",
                    os.path.expanduser("~/.conda/envs/unsloth_env/bin/python"),
                ]:
                    if os.path.isfile(p):
                        unsloth_python = p
                        break

                if not unsloth_python:
                    print("  ✗ Unsloth environment not found")
                    print("  Please merge LoRA adapters first using the upload handler")
                    return []

                merge_script = WSL_PATCH_CODE + f'''
import os
os.chdir(os.path.expanduser("~"))

from unsloth import FastLanguageModel
import torch

lora_path = "{model_path}"
output_path = "{merged_path}"

print(f"Loading from: {{lora_path}}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_path,
    max_seq_length={context_window_size},
    dtype=torch.float16,
    load_in_4bit=True,
)

print("Saving merged 16-bit model...")
model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
print("Done!")
'''

                print(f"  Merging via Unsloth (subprocess)...")
                try:
                    with branded_spinner("Merging LoRA adapters"):
                        result = subprocess.run(
                            [unsloth_python, "-c", merge_script],
                            capture_output=True,
                            text=True,
                            timeout=600,
                            cwd=get_working_directory()
                        )

                    if result.returncode != 0:
                        print(f"  ✗ Merge failed: {result.stderr[:500]}")
                        return []

                    # Verify merge succeeded
                    if not any(merged_path.glob("*.safetensors")):
                        print(f"  ✗ Merge produced no output files")
                        return []

                    print("  ✓ LoRA merged via Unsloth")
                    model_path = merged_path

                except subprocess.TimeoutExpired:
                    print("  ✗ Merge timed out")
                    return []
                except Exception as e:
                    print(f"  ✗ Merge error: {e}")
                    return []

        for i, quant in enumerate(quantizations, 1):
            quant_name = f"{model_name}-{quant}-MLC"
            quant_dir = webgpu_dir / quant_name

            print(f"\n[{i}/{len(quantizations)}] Processing {quant}...")

            # Step 1: Convert weights
            print("\n  Step 1/3: Converting weights...")
            if not self.convert_weights(model_path, quant_dir, quant):
                print(f"  ⚠ Skipping {quant} due to conversion failure")
                continue

            # Step 2: Generate config
            print("\n  Step 2/3: Generating config...")
            if not self.generate_config(
                model_path, quant_dir, quant, conv_template,
                context_window_size=context_window_size
            ):
                print(f"  ⚠ Config generation failed for {quant}")
                continue

            created_paths.append(quant_dir)

            # Step 3: Compile WASM (optional)
            if compile_wasm:
                print("\n  Step 3/3: Compiling WebGPU library...")
                config_path = quant_dir / "mlc-chat-config.json"
                wasm_path = webgpu_dir / f"{model_name}-{quant}-webgpu.wasm"

                # Pass model_path and quant for prebuilt fallback if compilation fails
                if self.compile_webgpu(
                    config_path, wasm_path,
                    model_name=quant_name,
                    model_path=model_path,
                    quantization=quant
                ):
                    created_paths.append(wasm_path)
                else:
                    print("  ⚠ WASM compilation failed (weights still usable)")
            else:
                print("\n  Step 3/3: Skipping WASM compilation")

        # Summary
        print(f"\n" + "=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"✓ Created {len(created_paths)} outputs:")
        for p in created_paths:
            if p.is_dir():
                # Calculate directory size
                total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"  - {p.name}/ ({size_mb:.1f} MB)")
            else:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  - {p.name} ({size_mb:.1f} MB)")

        print(f"\nOutput directory: {webgpu_dir}")
        print("=" * 60)

        print("\nNext steps:")
        print("  1. Upload weights folder to HuggingFace")
        print("  2. Host .wasm file (GitHub releases or CDN)")
        print("  3. Register in WebLLM with ModelRecord")
        print("  4. See: https://webllm.mlc.ai/docs/")

        return created_paths

    def supported_quantizations(self) -> List[str]:
        """Get list of supported quantization methods."""
        return SUPPORTED_QUANTIZATIONS.copy()

    def validate_environment(self) -> Tuple[bool, str]:
        """Validate that MLC-LLM is available."""
        return self.check_mlc_available()


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert model to WebGPU format")
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--name", help="Model name for output files")
    parser.add_argument("--quants", nargs="+", default=DEFAULT_QUANTIZATIONS,
                        help="Quantization methods")
    parser.add_argument("--conv-template", help="Conversation template")
    parser.add_argument("--context-window", type=int, default=4096,
                        help="Context window size")
    parser.add_argument("--no-wasm", action="store_true",
                        help="Skip WASM compilation")

    args = parser.parse_args()

    converter = WebGPUConverter()
    converter.convert(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        quantizations=args.quants,
        model_name=args.name,
        conv_template=args.conv_template,
        context_window_size=args.context_window,
        compile_wasm=not args.no_wasm,
    )


if __name__ == "__main__":
    main()
