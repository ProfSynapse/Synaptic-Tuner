"""
scripts/cloud_gguf_convert.py

Standalone CLI for cloud GGUF conversion. Downloads a HuggingFace model,
converts it to GGUF format using llama.cpp's pure-Python converter, and
uploads the result back to a HuggingFace repo.

Used by: Trainers/cloud/jobs/gguf_conversion.yaml (HF Jobs cloud runner)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
WORK_DIR = Path("/workspace/gguf_work")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HF model, convert to GGUF, upload to HF."
    )
    parser.add_argument(
        "--model-repo",
        required=True,
        help="HuggingFace model repo to convert (e.g. professorsynapse/gemma-4-e4b-sft)",
    )
    parser.add_argument(
        "--quant",
        default="q8_0",
        help="Quantization type for GGUF conversion (default: q8_0)",
    )
    parser.add_argument(
        "--upload-to",
        default=None,
        help="HF repo to upload the GGUF file to (defaults to --model-repo)",
    )
    return parser.parse_args()


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


def clone_llama_cpp(dest: Path) -> Path:
    """Shallow-clone llama.cpp for the pure-Python converter."""
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


def convert_to_gguf(
    llama_cpp_dir: Path, model_dir: Path, outfile: Path, quant: str
) -> None:
    """Run the pure-Python HF-to-GGUF converter."""
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Converter not found at {converter}")

    cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outtype", quant,
        "--outfile", str(outfile),
        "--use-temp-file",
    ]
    log.info("Converting to GGUF (quant=%s) ...", quant)
    run_cmd(cmd)
    log.info("GGUF file created: %s (%.1f GB)", outfile, outfile.stat().st_size / 1e9)


def upload_gguf(gguf_path: Path, repo_id: str) -> None:
    """Upload the GGUF file to a HuggingFace repo."""
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


def main() -> None:
    args = parse_args()
    upload_repo = args.upload_to or args.model_repo
    quant_upper = args.quant.upper()

    # Derive output filename: org/model-name -> model-name-Q8_0.gguf
    model_name = args.model_repo.split("/")[-1]
    output_name = f"{model_name}-{quant_upper}.gguf"

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    model_dir = WORK_DIR / "model"
    llama_cpp_dir = WORK_DIR / "llama.cpp"
    outfile = WORK_DIR / output_name

    download_model(args.model_repo, model_dir)
    clone_llama_cpp(llama_cpp_dir)
    install_gguf_package()
    convert_to_gguf(llama_cpp_dir, model_dir, outfile, args.quant)
    upload_gguf(outfile, upload_repo)

    log.info("Done. GGUF available at: %s/gguf/%s", upload_repo, output_name)


if __name__ == "__main__":
    main()
