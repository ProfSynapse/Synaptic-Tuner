"""
scripts/cloud_gguf_convert.py

Standalone CLI for cloud GGUF conversion. Downloads a merged HuggingFace model,
converts it to one or more GGUF types with llama.cpp, and uploads the GGUF files
plus gguf_manifest.json back to a HuggingFace repo (under gguf/).

Two paths, both through ReliableGGUFConverter:
- Fast path (no compilation): when every requested type is one that
  convert_hf_to_gguf.py --outtype writes directly (f32, f16, bf16, q8_0,
  tq1_0, tq2_0), each file is written by the pure-Python converter.
- Quantize path: anything else (k-quants such as q4_k_m, IQ types) converts to
  a bf16 base GGUF, optionally computes an importance matrix from a calibration
  dataset, then runs llama-quantize. llama-quantize/llama-imatrix are compiled
  from source on first use (needs cmake and a C/C++ compiler).

Used by: Trainers/recipes/gguf_conversion.yaml (HF Jobs cloud runner)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.upload.converters.calibration import (
    CalibrationSpec,
    DEFAULT_IMATRIX_CHUNKS,
    DEFAULT_IMATRIX_CTX,
    DEFAULT_MAX_ROWS,
)
from shared.upload.converters.gguf_reliable import (
    CONVERT_OUTTYPES,
    ReliableGGUFConverter,
    validate_quantizations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
WORK_DIR = Path("/workspace/gguf_work")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HF model, convert to GGUF, upload to HF."
    )
    parser.add_argument(
        "--model-repo",
        required=True,
        help="HuggingFace repo with the merged model (e.g. professorsynapse/gemma-4-e4b-sft)",
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        default=["q8_0"],
        help=(
            "One or more GGUF types, space- or comma-separated "
            "(e.g. q4_k_m q8_0 or q4_k_m,q8_0; default: q8_0)"
        ),
    )
    parser.add_argument(
        "--upload-to",
        default=None,
        help="HF repo to upload the GGUF files to (defaults to --model-repo)",
    )
    calibration = parser.add_argument_group(
        "calibration (importance matrix for k-quants / IQ types)"
    )
    calibration.add_argument(
        "--calibration-repo",
        default=None,
        help="HF dataset repo holding the calibration JSONL (e.g. the SFT data)",
    )
    calibration.add_argument(
        "--calibration-file",
        default=None,
        help="Path of the calibration JSONL inside --calibration-repo",
    )
    calibration.add_argument(
        "--calibration-path",
        default=None,
        help="Local calibration JSONL (alternative to --calibration-repo)",
    )
    calibration.add_argument(
        "--calibration-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Max calibration rows sampled (default: {DEFAULT_MAX_ROWS})",
    )
    calibration.add_argument(
        "--imatrix-chunks",
        type=int,
        default=DEFAULT_IMATRIX_CHUNKS,
        help=f"Max llama-imatrix chunks (default: {DEFAULT_IMATRIX_CHUNKS})",
    )
    calibration.add_argument(
        "--imatrix-ctx",
        type=int,
        default=DEFAULT_IMATRIX_CTX,
        help=f"llama-imatrix context size per chunk (default: {DEFAULT_IMATRIX_CTX})",
    )
    parser.add_argument(
        "--work-dir",
        default=str(WORK_DIR),
        help=f"Working directory (default: {WORK_DIR})",
    )
    args = parser.parse_args(argv)

    if args.calibration_path and (args.calibration_repo or args.calibration_file):
        parser.error("--calibration-path cannot be combined with --calibration-repo/--calibration-file")
    if bool(args.calibration_repo) != bool(args.calibration_file):
        parser.error("--calibration-repo and --calibration-file must be given together")
    return args


def parse_quant_list(values: List[str]) -> List[str]:
    """Flatten space- and comma-separated quant names, upper-cased, de-duplicated."""
    quants: List[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip().upper()
            if name and name not in quants:
                quants.append(name)
    if not quants:
        raise ValueError("No quantization types given")
    return quants


def is_fast_path(quants: List[str]) -> bool:
    """True when convert_hf_to_gguf.py --outtype can write every requested type."""
    return all(q in CONVERT_OUTTYPES for q in quants)


def run_cmd(cmd: list[str], **kwargs) -> None:
    """Run a shell command, raising on failure."""
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def download_model(repo_id: str, local_dir: Path) -> Path:
    """Download model snapshot from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    log.info("Downloading model %s ...", repo_id)
    path = snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    log.info("Model downloaded to %s", path)
    return Path(path)


def download_calibration(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download the calibration JSONL from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download

    log.info("Downloading calibration data %s/%s ...", repo_id, filename)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)


def clone_llama_cpp(dest: Path) -> Path:
    """Shallow-clone llama.cpp for the converter (and llama-quantize sources)."""
    if dest.exists():
        log.info("llama.cpp already present at %s, skipping clone", dest)
        return dest
    log.info("Cloning llama.cpp (depth 1) ...")
    run_cmd(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(dest)])
    return dest


def install_gguf_package() -> None:
    """Install the gguf Python package and upgrade transformers for tokenizer compat."""
    log.info("Installing gguf Python package and upgrading transformers ...")
    run_cmd([
        sys.executable, "-m", "pip", "install",
        "gguf>=0.16.0",
        "transformers>=4.52.0",
    ])


def ensure_build_tools() -> None:
    """Make sure cmake and a C++ compiler exist to build llama-quantize/llama-imatrix."""
    if shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None:
        raise RuntimeError(
            "No C/C++ compiler in this image, so llama-quantize cannot be built. "
            f"Request only {', '.join(sorted(q.lower() for q in CONVERT_OUTTYPES))} "
            "(no compilation needed) or use an image with build-essential."
        )
    if shutil.which("cmake") is not None:
        return
    log.info("cmake not found; installing the cmake wheel from PyPI ...")
    run_cmd([sys.executable, "-m", "pip", "install", "cmake"])
    python_bin = str(Path(sys.executable).parent)
    if shutil.which("cmake") is None and (Path(python_bin) / "cmake").exists():
        os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "cmake is still unavailable after `pip install cmake`; cannot build "
            "llama-quantize. Use an image with cmake or request only --outtype types."
        )


def fix_tokenizer_config(model_dir: Path) -> None:
    """Patch tokenizer_config.json if extra_special_tokens is a list instead of a dict.

    Some model uploads (e.g. Gemma 4) produce extra_special_tokens as a list,
    but transformers expects a dict mapping token names to values. Convert
    the list format to a dict keyed by the token string content.
    """
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return

    # Convert list of token dicts/strings to a dict keyed by content
    patched = {}
    for item in extra:
        if isinstance(item, dict):
            content = item.get("content", str(item))
            patched[content] = item
        else:
            patched[str(item)] = str(item)

    config["extra_special_tokens"] = patched
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    log.info("Patched extra_special_tokens in tokenizer_config.json (list -> dict)")


def convert_fast_path(
    converter: ReliableGGUFConverter,
    model_dir: Path,
    gguf_dir: Path,
    model_name: str,
    quants: List[str],
    source_model: str,
) -> List[Path]:
    """Write each type directly with convert_hf_to_gguf.py --outtype (no compile)."""
    gguf_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    for quant in quants:
        outfile = gguf_dir / f"{model_name}-{quant}.gguf"
        log.info("Converting to GGUF (outtype=%s) ...", quant.lower())
        if not converter.convert_to_gguf_base(
            model_dir, outfile, dtype=quant.lower(), use_temp_file=True
        ):
            raise RuntimeError(f"GGUF conversion failed for {quant}")
        log.info("GGUF file created: %s (%.1f GB)", outfile, outfile.stat().st_size / 1e9)
        created.append(outfile)

    converter.write_manifest(
        gguf_dir,
        [
            {"path": path, "role": "quant", "quant_type": quant, "imatrix_used": False}
            for path, quant in zip(created, quants)
        ],
        model_name=model_name,
        source_model=source_model,
        base_dtype=None,
        calibration=None,
    )
    return created


def upload_gguf(gguf_path: Path, repo_id: str) -> None:
    """Upload a file to the gguf/ folder of a HuggingFace repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    filename = gguf_path.name
    path_in_repo = f"gguf/{filename}"
    log.info("Uploading %s to %s/%s ...", filename, repo_id, path_in_repo)
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
    )
    log.info("Upload complete: %s/%s", repo_id, path_in_repo)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    upload_repo = args.upload_to or args.model_repo
    has_calibration = bool(args.calibration_path or args.calibration_repo)

    # Fail fast on unsupported combinations before any download or build.
    quants = validate_quantizations(parse_quant_list(args.quant), has_calibration=has_calibration)
    fast_path = is_fast_path(quants)
    if fast_path and has_calibration:
        log.warning(
            "Calibration ignored: %s do not use an importance matrix", ", ".join(quants)
        )

    model_name = args.model_repo.split("/")[-1]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_dir = work_dir / "model"
    llama_cpp_dir = work_dir / "llama.cpp"
    gguf_dir = work_dir / "gguf"

    download_model(args.model_repo, model_dir)
    clone_llama_cpp(llama_cpp_dir)
    install_gguf_package()
    fix_tokenizer_config(model_dir)

    converter = ReliableGGUFConverter(llama_cpp_dir=llama_cpp_dir)

    if fast_path:
        log.info("Fast path: %s via convert_hf_to_gguf --outtype", ", ".join(quants))
        created = convert_fast_path(
            converter, model_dir, gguf_dir, model_name, quants, args.model_repo
        )
    else:
        log.info("Quantize path: bf16 base -> llama-quantize %s", ", ".join(quants))
        calibration = None
        if has_calibration:
            if args.calibration_path:
                dataset_path = Path(args.calibration_path)
                source = str(dataset_path)
            else:
                dataset_path = download_calibration(
                    args.calibration_repo, args.calibration_file, work_dir / "calibration"
                )
                source = f"hf://datasets/{args.calibration_repo}/{args.calibration_file}"
            calibration = CalibrationSpec(
                dataset_path=dataset_path,
                source=source,
                max_rows=args.calibration_rows,
                chunks=args.imatrix_chunks,
                ctx_size=args.imatrix_ctx,
            )
        ensure_build_tools()
        created = converter.convert_merged_model(
            model_dir,
            gguf_dir,
            quants,
            model_name,
            work_dir / "work",
            dtype="bf16",
            include_base=False,
            use_temp_file=True,
            calibration=calibration,
            source_model=args.model_repo,
        )

    expected = {gguf_dir / f"{model_name}-{q}.gguf" for q in quants}
    missing = sorted(p.name for p in expected - set(created))
    if missing:
        raise RuntimeError(f"GGUF files not produced: {', '.join(missing)}")

    for path in created:
        upload_gguf(path, upload_repo)
    if converter.manifest_path is not None:
        upload_gguf(converter.manifest_path, upload_repo)

    log.info(
        "Done. GGUF available at: %s/gguf/ (%s)",
        upload_repo,
        ", ".join(p.name for p in created),
    )


if __name__ == "__main__":
    main()
