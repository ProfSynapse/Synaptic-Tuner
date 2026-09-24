"""
Reliable GGUF Converter for Text and Vision-Language Models.

This converter bypasses Unsloth's GGUF conversion to directly use llama.cpp,
providing more reliable conversion for both text and VL models.

Key improvements over Unsloth's converter:
1. Merge LoRA once, reuse for all quantizations (saves ~8 min per quant)
2. Better error handling and diagnostics
3. Direct control over llama.cpp commands
4. Proper handling of VL model mmproj conversion
5. WSL-friendly temp directory handling
6. Optional importance matrix (imatrix) from calibration data for low-bit quants
7. gguf_manifest.json recording every produced file and its provenance

Usage:
    from shared.upload.converters.gguf_reliable import ReliableGGUFConverter

    converter = ReliableGGUFConverter()

    # From a LoRA adapter (merges once, then converts):
    gguf_files = converter.convert(
        model_path="/path/to/final_model",
        output_dir=Path("/path/to/output"),
        quantizations=["Q4_K_M", "Q5_K_M", "Q8_0"],
        model_name="my-model"
    )

    # From an already-merged HF model directory:
    gguf_files = converter.convert_merged_model(
        merged_model_path=Path("/path/to/merged"),
        gguf_dir=Path("/path/to/output/gguf"),
        quantizations=["Q4_K_M"],
        model_name="my-model",
        work_dir=Path("/path/to/work"),
        calibration=CalibrationSpec(dataset_path=Path("train.jsonl")),
    )
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
from contextlib import contextmanager

from .calibration import CalibrationRenderResult, CalibrationSpec, render_calibration_text

# Brand colors for spinners
AQUA = "#00A99D"
PURPLE = "#93278F"

# Try to import Rich for nice spinners
try:
    from rich.console import Console
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# VL model indicators - models that use vision components
VL_MODEL_INDICATORS = [
    "qwen2-vl", "qwen3-vl", "qwen2_vl", "qwen3_vl",
    "llava", "pixtral", "paligemma", "idefics",
    "ministral3", "ministral-3", "ministral_3", "mistral3",
]

# Default quantizations
DEFAULT_QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

# Types convert_hf_to_gguf.py --outtype writes directly (besides "auto").
# llama.cpp's quantize kernels for exactly these types ignore imatrix weights,
# so an importance matrix only matters for the types outside this set.
CONVERT_OUTTYPES = frozenset({"F32", "F16", "BF16", "Q8_0", "TQ1_0", "TQ2_0"})

# llama-quantize refuses these file types without an importance matrix
# (llama.cpp src/llama-quant.cpp tensor_requires_imatrix plus the IQ3_XS/IQ2_M
# tensor mixes that use IQ3_XXS/IQ2_S tensors).
IMATRIX_REQUIRED_QUANTS = frozenset({
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS",
    "Q2_K_S",
})

GGUF_MANIFEST_NAME = "gguf_manifest.json"
GGUF_MANIFEST_SCHEMA_VERSION = 1

CONVERT_TIMEOUT_SECONDS = 3600
QUANTIZE_TIMEOUT_SECONDS = 3600
IMATRIX_TIMEOUT_SECONDS = 4 * 3600


def validate_quantizations(quantizations: Iterable[str], has_calibration: bool) -> List[str]:
    """Normalize quant names and fail fast on IQ types without calibration data.

    Returns:
        Upper-cased quantization names.

    Raises:
        ValueError: If a type that requires an importance matrix is requested
            without a calibration dataset.
    """
    normalized = [str(q).strip().upper() for q in quantizations if str(q).strip()]
    needs_imatrix = [q for q in normalized if q in IMATRIX_REQUIRED_QUANTS]
    if needs_imatrix and not has_calibration:
        raise ValueError(
            f"Quantization type(s) {', '.join(needs_imatrix)} require an importance "
            "matrix in llama.cpp. Supply a calibration dataset (JSONL, e.g. the SFT "
            "training data) so an imatrix can be computed, or choose a K-quant "
            "such as Q4_K_M."
        )
    return normalized


def uses_imatrix(quant_type: str) -> bool:
    """Whether llama-quantize makes use of an importance matrix for this type."""
    return quant_type.upper() not in CONVERT_OUTTYPES


def file_sha256(path: Path) -> str:
    """Stream a file through sha256."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stderr_tail(text: Optional[str], limit: int = 800) -> str:
    """llama.cpp tools log model loading first and the actual error last."""
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text

# Bubbling test tube spinner frames
BUBBLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
TUBE_FRAMES = ["│○  │", "│ ○ │", "│  ○│", "│ ○ │", "│○  │", "│•○ │", "│ •○│", "│  •│"]


@contextmanager
def branded_spinner(message: str):
    """
    Context manager that shows a branded spinner during long operations.

    Uses Rich if available, falls back to simple text animation.
    """
    if RICH_AVAILABLE:
        console = Console()
        # Use dots spinner with brand colors
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
            # Track spinner state
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
        # Simple fallback - just print the message
        print(f"  {message}...", end="", flush=True)
        try:
            yield
        finally:
            print(" done")


class ReliableGGUFConverter:
    """
    Reliable GGUF converter that handles both text and VL models.

    Merges LoRA adapters once and creates all quantizations from that base,
    avoiding the redundant re-merging that Unsloth does.
    """

    def __init__(self, llama_cpp_dir: Optional[Path] = None, model_loader: Optional[Any] = None):
        """
        Initialize the converter.

        Args:
            llama_cpp_dir: Path to llama.cpp directory. If None, will use ~/llama.cpp
            model_loader: Accepted for the ConverterRegistry contract; unused
                because this converter drives llama.cpp directly.
        """
        self.llama_cpp_dir = llama_cpp_dir or Path.home() / "llama.cpp"
        self._quantizer_path: Optional[Path] = None
        self._imatrix_path: Optional[Path] = None
        self._converter_path: Optional[Path] = None
        self.manifest_path: Optional[Path] = None

    def _find_binary(self, names: List[str]) -> Optional[Path]:
        for name in names:
            for loc in [
                self.llama_cpp_dir / "build" / "bin" / name,
                self.llama_cpp_dir / name,
            ]:
                if loc.exists() and os.access(loc, os.X_OK):
                    return loc
        return None

    @property
    def quantizer_path(self) -> Path:
        """Get path to llama-quantize binary."""
        if self._quantizer_path is None:
            self._quantizer_path = self._find_binary(["llama-quantize", "quantize"])
            if self._quantizer_path is None:
                raise FileNotFoundError(
                    f"llama-quantize not found in {self.llama_cpp_dir}. "
                    "Run setup_llama_cpp() first."
                )
        return self._quantizer_path

    @property
    def imatrix_path(self) -> Path:
        """Get path to llama-imatrix binary."""
        if self._imatrix_path is None:
            self._imatrix_path = self._find_binary(["llama-imatrix", "imatrix"])
            if self._imatrix_path is None:
                raise FileNotFoundError(
                    f"llama-imatrix not found in {self.llama_cpp_dir}. "
                    "Run setup_llama_cpp(with_imatrix=True) first."
                )
        return self._imatrix_path

    @property
    def converter_path(self) -> Path:
        """Get path to convert_hf_to_gguf.py script."""
        if self._converter_path is None:
            for name in ["convert_hf_to_gguf.py", "unsloth_convert_hf_to_gguf.py"]:
                loc = self.llama_cpp_dir / name
                if loc.exists():
                    self._converter_path = loc
                    break
            if self._converter_path is None:
                raise FileNotFoundError(
                    f"convert_hf_to_gguf.py not found in {self.llama_cpp_dir}. "
                    "Ensure llama.cpp is properly cloned."
                )
        return self._converter_path

    def _binaries_ready(self, with_imatrix: bool) -> bool:
        try:
            _ = self.quantizer_path
            if with_imatrix:
                _ = self.imatrix_path
        except FileNotFoundError:
            return False
        return True

    def setup_llama_cpp(self, with_imatrix: bool = False) -> bool:
        """
        Ensure llama.cpp is cloned and llama-quantize (and llama-imatrix when
        requested) are built.

        A build is only triggered when a required binary is missing, so an
        older build without llama-imatrix is left alone unless an imatrix is
        needed. Both targets are always built together when building.

        Args:
            with_imatrix: Also require the llama-imatrix binary.

        Returns:
            True if setup successful
        """
        print("Setting up llama.cpp...")

        if self._binaries_ready(with_imatrix):
            print(f"  ✓ llama.cpp already set up at {self.llama_cpp_dir}")
            return True

        if shutil.which("cmake") is None:
            print(
                "  ✗ cmake not found on PATH. Building llama-quantize/llama-imatrix "
                "needs cmake and a C/C++ compiler (e.g. `pip install cmake` or "
                "`apt-get install cmake build-essential`), or point llama_cpp_dir "
                "at an existing llama.cpp build."
            )
            return False

        # Clone if needed
        if not self.llama_cpp_dir.exists():
            print(f"  Cloning llama.cpp to {self.llama_cpp_dir}...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggerganov/llama.cpp.git",
                 str(self.llama_cpp_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"  ✗ Clone failed: {result.stderr}")
                return False

        # Build
        build_dir = self.llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # Configure only a fresh build dir; an existing (e.g. CUDA/Metal) build
        # keeps its configuration and just gains the missing targets.
        if not (build_dir / "CMakeCache.txt").exists():
            print("  Configuring llama.cpp (CPU-only for quantization)...")
            result = subprocess.run(
                ["cmake", "..", "-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"],
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"  ✗ cmake failed: {_stderr_tail(result.stderr)}")
                return False

        print("  Building llama-quantize and llama-imatrix...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release",
             "-j", str(os.cpu_count() or 4),
             "--target", "llama-quantize", "llama-imatrix"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=1800
        )
        if result.returncode != 0:
            print(f"  ✗ build failed: {_stderr_tail(result.stderr)}")
            return False

        # Reset cached paths
        self._quantizer_path = None
        self._imatrix_path = None
        self._converter_path = None

        if not self._binaries_ready(with_imatrix):
            print(f"  ✗ Build finished but binaries not found under {build_dir / 'bin'}")
            return False

        print(f"  ✓ llama.cpp built successfully")
        return True

    def llama_cpp_commit(self) -> Optional[str]:
        """Return the llama.cpp git commit, or None if unavailable."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.llama_cpp_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        commit = result.stdout.strip() if result.returncode == 0 else ""
        return commit or None

    def is_vision_model(self, model_path: Path) -> bool:
        """
        Detect if model is a Vision-Language model.

        Checks multiple indicators that persist even after fine-tuning:
        1. Vision-specific files (vision_tower, visual, image_processor, etc.)
        2. Vision config entries in config.json
        3. Base model name in adapter_config.json
        4. Model type and architectures

        Args:
            model_path: Path to model directory

        Returns:
            True if VL model detected
        """
        # Check for vision-specific files (most reliable for fine-tuned models)
        vision_files = [
            "preprocessor_config.json",  # Image preprocessor config
            "image_processor_config.json",  # Alternate name
        ]
        vision_patterns = ["vision", "visual", "image", "mmproj", "clip"]

        for f in model_path.iterdir():
            fname = f.name.lower()
            # Check for vision-related files
            if any(pattern in fname for pattern in vision_patterns):
                return True

        # Check for specific vision files
        for vf in vision_files:
            if (model_path / vf).exists():
                return True

        # Check config.json for vision config
        config_json = model_path / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    config = json.load(f)

                # Check for vision-related config keys
                vision_keys = ["vision_config", "visual_config", "image_size",
                               "vision_tower", "mm_projector", "image_processor"]
                if any(key in config for key in vision_keys):
                    return True

                # Check model_type
                model_type = config.get("model_type", "").lower()
                if any(vl in model_type for vl in VL_MODEL_INDICATORS):
                    return True

                # Check architectures
                for arch in config.get("architectures", []):
                    if any(vl in arch.lower() for vl in VL_MODEL_INDICATORS):
                        return True
            except (json.JSONDecodeError, KeyError):
                pass

        # Check adapter_config.json (base model reference)
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config) as f:
                    config = json.load(f)
                base_name = config.get("base_model_name_or_path", "").lower()
                if any(vl in base_name for vl in VL_MODEL_INDICATORS):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        return False

    def get_model_architecture(self, model_path: Path) -> Optional[str]:
        """Get model architecture from config."""
        config_json = model_path / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    config = json.load(f)
                archs = config.get("architectures", [])
                if archs:
                    return archs[0]
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def merge_lora_to_16bit(
        self,
        model_path: Path,
        output_dir: Path,
    ) -> Path:
        """
        Merge LoRA adapters into base model at 16-bit precision.

        This is done ONCE and reused for all quantizations.

        Args:
            model_path: Path to LoRA adapter directory
            output_dir: Directory to save merged model

        Returns:
            Path to merged model directory
        """
        print("\n[Merge] Merging LoRA adapters to 16-bit...")

        merged_dir = output_dir / "merged_16bit_temp"

        # Use Unsloth's efficient merge if available
        try:
            from unsloth import FastLanguageModel, FastVisionModel

            is_vl = self.is_vision_model(model_path)

            # Load adapter and base model
            if is_vl:
                print("  Loading VL model with FastVisionModel...")
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=2048,
                    load_in_4bit=False,  # Full precision for merge
                )
            else:
                print("  Loading text model with FastLanguageModel...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=2048,
                    load_in_4bit=False,
                )

            # Save merged model
            print(f"  Saving merged model to {merged_dir}...")
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"  ✓ Merged model saved")

        except ImportError:
            # Fallback to PEFT merge
            print("  Using PEFT for merge (Unsloth not available)...")
            from peft import PeftModel, AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            model = AutoPeftModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype="auto",
            )
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"  ✓ Merged model saved")

        return merged_dir

    def convert_to_gguf_base(
        self,
        merged_model_path: Path,
        output_path: Path,
        dtype: str = "bf16",
        is_mmproj: bool = False,
        use_temp_file: bool = False,
    ) -> bool:
        """
        Convert merged model to base GGUF (f16/bf16).

        Args:
            merged_model_path: Path to merged HF model
            output_path: Path for output GGUF file
            dtype: Output data type (any convert_hf_to_gguf.py --outtype)
            is_mmproj: If True, convert vision projector only
            use_temp_file: Pass --use-temp-file to lower peak RAM

        Returns:
            True if conversion successful
        """
        component = "vision projector (mmproj)" if is_mmproj else "text model"

        cmd = [
            sys.executable, str(self.converter_path),
            "--outfile", str(output_path),
            "--outtype", dtype,
        ]
        if use_temp_file:
            cmd.append("--use-temp-file")
        if is_mmproj:
            cmd.append("--mmproj")
        cmd.append(str(merged_model_path))

        try:
            with branded_spinner(f"Converting {component} to GGUF"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=CONVERT_TIMEOUT_SECONDS,
                )

            if result.returncode != 0:
                print(f"  ✗ Conversion failed:")
                print(f"    {_stderr_tail(result.stderr)}")
                return False

            if not output_path.exists():
                print(f"  ✗ Output file not created: {output_path}")
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_path.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ✗ Conversion timed out")
            return False
        except Exception as e:
            print(f"  ✗ Conversion error: {e}")
            return False

    def compute_imatrix(
        self,
        base_gguf: Path,
        calibration_text_file: Path,
        output_path: Path,
        chunks: Optional[int] = None,
        ctx_size: int = 512,
        parse_special: bool = False,
    ) -> bool:
        """
        Compute an importance matrix with llama-imatrix.

        Current llama.cpp writes the imatrix in GGUF format (use a .gguf
        output name). llama-imatrix needs at least 2 * ctx_size tokens of
        calibration text.

        Args:
            base_gguf: Unquantized (f16/bf16) GGUF to measure activations on
            calibration_text_file: Plain-text calibration data
            output_path: Where to write the imatrix (.gguf)
            chunks: Max number of ctx_size chunks to process (None = all)
            ctx_size: Tokens per chunk
            parse_special: Tokenize special tokens in the text (set when the
                text was rendered with the model's chat template)

        Returns:
            True if the imatrix was written
        """
        cmd = [
            str(self.imatrix_path),
            "-m", str(base_gguf),
            "-f", str(calibration_text_file),
            "-o", str(output_path),
            "-c", str(ctx_size),
            "--no-ppl",
        ]
        if chunks is not None:
            cmd.extend(["--chunks", str(chunks)])
        if parse_special:
            cmd.append("--parse-special")

        try:
            with branded_spinner("Computing importance matrix (imatrix)"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=IMATRIX_TIMEOUT_SECONDS,
                )

            if result.returncode != 0:
                print(f"  ✗ imatrix computation failed:")
                print(f"    {_stderr_tail(result.stderr)}")
                return False

            if not output_path.exists():
                print(f"  ✗ Output file not created: {output_path}")
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_path.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ✗ imatrix computation timed out")
            return False
        except Exception as e:
            print(f"  ✗ imatrix error: {e}")
            return False

    def quantize_gguf(
        self,
        input_gguf: Path,
        output_gguf: Path,
        quant_type: str,
        imatrix: Optional[Path] = None,
    ) -> bool:
        """
        Quantize a GGUF file using llama-quantize.

        Args:
            input_gguf: Path to input GGUF (f16/bf16)
            output_gguf: Path for quantized output
            quant_type: Quantization method (Q4_K_M, Q5_K_M, Q8_0, etc.)
            imatrix: Optional importance matrix from compute_imatrix()

        Returns:
            True if quantization successful

        Raises:
            ValueError: If quant_type requires an importance matrix and none is given
        """
        validate_quantizations([quant_type], has_calibration=imatrix is not None)

        n_threads = (os.cpu_count() or 4) * 2

        cmd = [str(self.quantizer_path)]
        if imatrix is not None:
            cmd.extend(["--imatrix", str(imatrix)])
        cmd.extend([
            str(input_gguf),
            str(output_gguf),
            quant_type,
            str(n_threads),
        ])

        label = f"{quant_type} (imatrix)" if imatrix is not None else quant_type
        try:
            with branded_spinner(f"Quantizing to {label}"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=QUANTIZE_TIMEOUT_SECONDS,
                )

            if result.returncode != 0:
                print(f"  ✗ Quantization failed:")
                print(f"    {_stderr_tail(result.stderr)}")
                return False

            if not output_gguf.exists():
                print(f"  ✗ Output file not created: {output_gguf}")
                return False

            size_mb = output_gguf.stat().st_size / (1024 * 1024)
            print(f"  ✓ Created {output_gguf.name} ({size_mb:.1f} MB)")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ✗ Quantization timed out")
            return False
        except Exception as e:
            print(f"  ✗ Quantization error: {e}")
            return False

    def write_manifest(
        self,
        gguf_dir: Path,
        files: List[Dict[str, Any]],
        *,
        model_name: str,
        source_model: str,
        base_dtype: Optional[str],
        calibration: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write gguf_manifest.json next to the GGUF files.

        Args:
            gguf_dir: Directory holding the GGUF files
            files: Entries of {"path": Path, "role": str, "quant_type": str,
                "imatrix_used": bool}
            model_name: Output file stem
            source_model: Source model path or repo id
            base_dtype: Base GGUF dtype the quants were made from (None when
                every file was written directly by convert_hf_to_gguf.py)
            calibration: Calibration/imatrix record, or None

        Returns:
            Path to the manifest
        """
        manifest = {
            "schema_version": GGUF_MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": model_name,
            "source_model": source_model,
            "base_dtype": base_dtype,
            "llama_cpp_commit": self.llama_cpp_commit(),
            "calibration": calibration,
            "files": [
                {
                    "filename": entry["path"].name,
                    "role": entry["role"],
                    "quant_type": entry["quant_type"],
                    "size_bytes": entry["path"].stat().st_size,
                    "sha256": file_sha256(entry["path"]),
                    "imatrix_used": entry["imatrix_used"],
                }
                for entry in files
            ],
        }
        manifest_path = gguf_dir / GGUF_MANIFEST_NAME
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        self.manifest_path = manifest_path
        print(f"  ✓ Wrote {manifest_path.name}")
        return manifest_path

    def convert_merged_model(
        self,
        merged_model_path: Path,
        gguf_dir: Path,
        quantizations: List[str],
        model_name: str,
        work_dir: Path,
        *,
        dtype: str = "bf16",
        include_base: bool = True,
        use_temp_file: bool = False,
        calibration: Optional[CalibrationSpec] = None,
        source_model: Optional[str] = None,
    ) -> List[Path]:
        """
        Convert an already-merged HF model directory to GGUF files.

        Steps: base GGUF (dtype), mmproj GGUF for VL models, optional
        importance matrix from calibration data, then every quantization
        from the single base. Writes gguf_manifest.json to gguf_dir.

        Args:
            merged_model_path: Merged HF model directory (weights + tokenizer)
            gguf_dir: Directory for the final GGUF files and manifest
            quantizations: llama-quantize types (e.g. Q4_K_M, Q8_0, IQ3_XXS)
            model_name: Output file stem
            work_dir: Scratch directory for the base GGUF, calibration text
                and imatrix (not uploaded)
            dtype: Base GGUF dtype (bf16 or f16)
            include_base: Copy the base GGUF into gguf_dir as {model_name}.gguf
            use_temp_file: Lower converter peak RAM (convert_hf_to_gguf --use-temp-file)
            calibration: Calibration dataset for the imatrix; None disables it
            source_model: Source model path/repo recorded in the manifest

        Returns:
            List of created GGUF file paths (manifest path is self.manifest_path)
        """
        merged_model_path = Path(merged_model_path)
        gguf_dir = Path(gguf_dir)
        work_dir = Path(work_dir)
        quants = validate_quantizations(quantizations, has_calibration=calibration is not None)

        # An imatrix only changes types outside CONVERT_OUTTYPES.
        want_imatrix = calibration is not None and any(uses_imatrix(q) for q in quants)
        if calibration is not None and not want_imatrix:
            print(
                f"  ⚠ Calibration data ignored: {', '.join(quants)} do not use an "
                "importance matrix"
            )

        if not self.setup_llama_cpp(with_imatrix=want_imatrix):
            raise RuntimeError("Failed to setup llama.cpp")

        gguf_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        is_vl = self.is_vision_model(merged_model_path)

        created_files: List[Path] = []
        manifest_files: List[Dict[str, Any]] = []

        # Base GGUF
        print(f"\n[Base] Converting to base GGUF ({dtype})...")
        base_gguf = work_dir / f"{model_name}.gguf"
        if not self.convert_to_gguf_base(
            merged_model_path, base_gguf, dtype, use_temp_file=use_temp_file
        ):
            raise RuntimeError("Base GGUF conversion failed")

        if include_base:
            final_base = gguf_dir / f"{model_name}.gguf"
            if final_base.resolve() != base_gguf.resolve():
                shutil.copy2(base_gguf, final_base)
            created_files.append(final_base)
            manifest_files.append({
                "path": final_base, "role": "base",
                "quant_type": dtype.upper(), "imatrix_used": False,
            })

        # Vision projector
        if is_vl:
            print("\n[mmproj] Creating vision projector (mmproj) GGUF...")
            mmproj_gguf = gguf_dir / f"{model_name}-mmproj.gguf"

            if self.convert_to_gguf_base(merged_model_path, mmproj_gguf, dtype, is_mmproj=True):
                created_files.append(mmproj_gguf)
                manifest_files.append({
                    "path": mmproj_gguf, "role": "mmproj",
                    "quant_type": dtype.upper(), "imatrix_used": False,
                })
            else:
                print("  ⚠ mmproj conversion failed - vision features may not work")
                print("    This model can still be used for text-only inference")

        # Importance matrix
        imatrix_file: Optional[Path] = None
        calibration_record: Optional[Dict[str, Any]] = None
        if want_imatrix:
            print("\n[Imatrix] Rendering calibration text...")
            rendered: CalibrationRenderResult = render_calibration_text(
                calibration.dataset_path,
                work_dir / f"{model_name}.calibration.txt",
                tokenizer_dir=merged_model_path,
                max_rows=calibration.max_rows,
                max_chars=calibration.max_chars,
                seed=calibration.seed,
                label_field=calibration.label_field,
                source=calibration.source_label,
            )
            print(
                f"  ✓ {rendered.rows_used} rows rendered ({rendered.render_mode}), "
                f"{rendered.rows_skipped} skipped, {rendered.rows_filtered} filtered, "
                f"{rendered.chars} chars"
            )

            imatrix_file = work_dir / f"{model_name}.imatrix.gguf"
            if not self.compute_imatrix(
                base_gguf,
                rendered.output_path,
                imatrix_file,
                chunks=calibration.chunks,
                ctx_size=calibration.ctx_size,
                parse_special=rendered.render_mode == "chat_template",
            ):
                raise RuntimeError(
                    "imatrix computation failed; not producing quants without the "
                    "requested calibration"
                )
            calibration_record = {
                **rendered.to_manifest(),
                "chunks": calibration.chunks,
                "ctx_size": calibration.ctx_size,
                "imatrix_sha256": file_sha256(imatrix_file),
            }

        # Quantizations
        print(f"\n[Quantize] Creating quantizations...")
        for quant in quants:
            quant_file = gguf_dir / f"{model_name}-{quant}.gguf"
            quant_imatrix = imatrix_file if imatrix_file is not None and uses_imatrix(quant) else None

            if self.quantize_gguf(base_gguf, quant_file, quant, imatrix=quant_imatrix):
                created_files.append(quant_file)
                manifest_files.append({
                    "path": quant_file, "role": "quant",
                    "quant_type": quant, "imatrix_used": quant_imatrix is not None,
                })

        self.write_manifest(
            gguf_dir,
            manifest_files,
            model_name=model_name,
            source_model=source_model or Path(merged_model_path).name,
            base_dtype=dtype,
            calibration=calibration_record,
        )

        print(f"\n[Summary]")
        print("=" * 60)
        print(f"✓ Created {len(created_files)} GGUF files:")
        for f in created_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print(f"\nOutput directory: {gguf_dir}")
        print("=" * 60)

        return created_files

    def convert(
        self,
        model_path: str | Path,
        output_dir: Path,
        quantizations: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        dtype: str = "bf16",
        cleanup_temp: bool = True,
        cleanup: bool = None,  # Alias for cleanup_temp (for compatibility)
        calibration: Optional[CalibrationSpec] = None,
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> List[Path]:
        """
        Convert a LoRA adapter (or model loadable by the merge step) to GGUF.

        This method:
        1. Merges LoRA to 16-bit ONCE
        2. Hands the merged model to convert_merged_model() (base GGUF,
           mmproj for VL models, optional imatrix, all quantizations)

        Args:
            model_path: Path to model (LoRA adapters or full model)
            output_dir: Directory to save GGUF files
            quantizations: List of quantization methods
            model_name: Name for output files (default: from model_path)
            dtype: Base GGUF data type (f16 or bf16)
            cleanup_temp: Whether to cleanup temporary files
            calibration: Calibration dataset for an importance matrix

        Returns:
            List of created GGUF file paths
        """
        # Handle cleanup alias
        if cleanup is not None:
            cleanup_temp = cleanup

        model_path = Path(model_path)
        quantizations = validate_quantizations(
            quantizations or DEFAULT_QUANTIZATIONS, has_calibration=calibration is not None
        )
        model_name = model_name or model_path.name

        # Setup llama.cpp before the expensive merge
        want_imatrix = calibration is not None and any(uses_imatrix(q) for q in quantizations)
        if not self.setup_llama_cpp(with_imatrix=want_imatrix):
            raise RuntimeError("Failed to setup llama.cpp")

        # Detect model type
        is_vl = self.is_vision_model(model_path)
        arch = self.get_model_architecture(model_path)

        print("\n" + "=" * 60)
        print("GGUF CONVERSION")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Architecture: {arch or 'Unknown'}")
        print(f"Type: {'Vision-Language' if is_vl else 'Text-only'}")
        print(f"Output: {output_dir}")
        print(f"Quantizations: {', '.join(quantizations)}")
        if calibration is not None:
            print(f"Calibration: {calibration.source_label}")
        print("=" * 60)

        gguf_dir = output_dir / "gguf"

        # Use WSL native filesystem for temp files (avoids NTFS issues)
        use_wsl_temp = os.path.exists("/home") and str(model_path).startswith("/mnt/")
        if use_wsl_temp:
            temp_base = Path.home() / "tmp_gguf"
            temp_base.mkdir(exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(dir=temp_base))
            print(f"Using WSL native temp: {temp_dir}")
        else:
            temp_dir = Path(tempfile.mkdtemp())

        try:
            merged_dir = self.merge_lora_to_16bit(model_path, temp_dir)

            return self.convert_merged_model(
                merged_dir,
                gguf_dir,
                quantizations,
                model_name,
                temp_dir,
                dtype=dtype,
                include_base=True,
                calibration=calibration,
                source_model=model_path.name,
            )

        finally:
            # Cleanup temp files
            if cleanup_temp and temp_dir.exists():
                print(f"\nCleaning up temp files...")
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"✓ Cleaned up {temp_dir}")

                # If using WSL temp and parent directory is now empty, remove it too
                if use_wsl_temp:
                    temp_base = Path.home() / "tmp_gguf"
                    try:
                        # Check if directory is empty
                        if temp_base.exists() and not any(temp_base.iterdir()):
                            temp_base.rmdir()
                            print(f"✓ Removed empty parent directory {temp_base}")
                    except (OSError, PermissionError):
                        # Directory not empty or permission issue, that's fine
                        pass


def cleanup_orphaned_temp_files(max_age_hours: int = 24) -> int:
    """
    Clean up orphaned temporary files in ~/tmp_gguf.

    Removes temp directories older than max_age_hours that may have been
    left behind due to interrupted conversions.

    Args:
        max_age_hours: Maximum age in hours before cleaning up

    Returns:
        Number of directories cleaned up
    """
    temp_base = Path.home() / "tmp_gguf"
    if not temp_base.exists():
        return 0

    cleaned = 0
    cutoff_time = time.time() - (max_age_hours * 3600)

    print(f"Scanning {temp_base} for orphaned files older than {max_age_hours}h...")

    for item in temp_base.iterdir():
        if item.is_dir():
            try:
                # Check directory age
                mtime = item.stat().st_mtime
                if mtime < cutoff_time:
                    size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024 * 1024)
                    shutil.rmtree(item)
                    cleaned += 1
                    print(f"  ✓ Removed {item.name} ({size_mb:.1f} MB)")
            except (OSError, PermissionError) as e:
                print(f"  ✗ Failed to remove {item.name}: {e}")

    # Try to remove parent if empty
    try:
        if not any(temp_base.iterdir()):
            temp_base.rmdir()
            print(f"  ✓ Removed empty parent directory {temp_base}")
    except (OSError, PermissionError):
        pass

    if cleaned == 0:
        print("  No orphaned files found")

    return cleaned


def main():
    """CLI entry point.

    Examples:
        # LoRA adapter -> merge -> GGUF
        python -m shared.upload.converters.gguf_reliable ./final_model ./out --quants Q4_K_M Q8_0

        # Already-merged model, Q4_K_M with an importance matrix
        python -m shared.upload.converters.gguf_reliable ./merged ./out --merged \\
            --quants Q4_K_M --calibration Datasets/train.jsonl
    """
    import argparse

    from .calibration import (
        DEFAULT_IMATRIX_CHUNKS,
        DEFAULT_IMATRIX_CTX,
        DEFAULT_MAX_ROWS,
        DEFAULT_SEED,
    )

    parser = argparse.ArgumentParser(description="Convert model to GGUF")
    parser.add_argument("model_path", nargs="?", help="Path to model (LoRA or full)")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--name", help="Model name for output files")
    parser.add_argument("--quants", nargs="+", default=DEFAULT_QUANTIZATIONS,
                        help="Quantization methods")
    parser.add_argument("--dtype", default="bf16", choices=["f16", "bf16"],
                        help="Base GGUF data type")
    parser.add_argument("--merged", action="store_true",
                        help="model_path is an already-merged HF model (skip LoRA merge)")
    parser.add_argument("--llama-cpp-dir", help="llama.cpp checkout (default: ~/llama.cpp)")
    parser.add_argument("--calibration",
                        help="JSONL dataset for an importance matrix (enables imatrix)")
    parser.add_argument("--calibration-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help=f"Max calibration rows (default: {DEFAULT_MAX_ROWS})")
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_SEED,
                        help=f"Calibration sampling seed (default: {DEFAULT_SEED})")
    parser.add_argument("--imatrix-chunks", type=int, default=DEFAULT_IMATRIX_CHUNKS,
                        help=f"Max imatrix chunks (default: {DEFAULT_IMATRIX_CHUNKS})")
    parser.add_argument("--imatrix-ctx", type=int, default=DEFAULT_IMATRIX_CTX,
                        help=f"imatrix context size (default: {DEFAULT_IMATRIX_CTX})")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep temp files")
    parser.add_argument("--cleanup-orphaned", action="store_true",
                        help="Clean up orphaned temp files and exit")
    parser.add_argument("--max-age-hours", type=int, default=24,
                        help="Max age for orphaned files (default: 24h)")

    args = parser.parse_args()

    # Handle cleanup mode
    if args.cleanup_orphaned:
        cleaned = cleanup_orphaned_temp_files(args.max_age_hours)
        print(f"\nCleaned up {cleaned} orphaned directories")
        return

    # Normal conversion mode
    if not args.model_path or not args.output_dir:
        parser.error("model_path and output_dir are required for conversion")

    calibration = None
    if args.calibration:
        calibration = CalibrationSpec(
            dataset_path=Path(args.calibration),
            max_rows=args.calibration_rows,
            seed=args.calibration_seed,
            chunks=args.imatrix_chunks,
            ctx_size=args.imatrix_ctx,
        )

    converter = ReliableGGUFConverter(
        llama_cpp_dir=Path(args.llama_cpp_dir) if args.llama_cpp_dir else None
    )
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    model_name = args.name or model_path.name

    if args.merged:
        work_dir = output_dir / "work"
        try:
            converter.convert_merged_model(
                model_path,
                output_dir / "gguf",
                args.quants,
                model_name,
                work_dir,
                dtype=args.dtype,
                calibration=calibration,
            )
        finally:
            if not args.no_cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)
        return

    converter.convert(
        model_path=model_path,
        output_dir=output_dir,
        quantizations=args.quants,
        model_name=model_name,
        dtype=args.dtype,
        cleanup_temp=not args.no_cleanup,
        calibration=calibration,
    )

if __name__ == "__main__":
    main()
