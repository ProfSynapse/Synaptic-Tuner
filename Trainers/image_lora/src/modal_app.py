"""
Modal app for the image_lora method (the remote side).

Location: Trainers/image_lora/src/modal_app.py
Purpose:  Define the training image (pinned ai-toolkit + torch) and one worker
          class with two methods: ``probe`` (cheap GPU/driver/import smoke test)
          and ``train`` (pinned base snapshot -> ai-toolkit run -> hashed outputs).
          GPU, volumes, secret and timeout are bound per run by the host with
          ``ImageLoraWorker.with_options(...)`` (src/modal_runner.py), so this
          module has no run-specific values and imports cleanly in the container.
Used by:  src/modal_runner.py only. Must import with just ``modal`` + stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

AI_TOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_COMMIT = "f7a1fb9a77e5418269b1e9eda4ace6d7c1b7bb47"   # 2026-10-10 main
AI_TOOLKIT_DIR = "/opt/ai-toolkit"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"       # ai-toolkit README (Linux)
TORCH_PACKAGES = ("torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0")

APP_NAME = "synaptic-image-lora"
CACHE_MOUNT = "/data/cache"       # shared base-weights cache Volume
RUN_MOUNT = "/data/run"           # per-run Volume: dataset/ and output/
HF_HOME = f"{CACHE_MOUNT}/hf"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(*TORCH_PACKAGES, index_url=TORCH_INDEX_URL)
    .run_commands(
        f"git clone {AI_TOOLKIT_REPO} {AI_TOOLKIT_DIR}",
        f"git -C {AI_TOOLKIT_DIR} checkout {AI_TOOLKIT_COMMIT}",
        f"pip install -r {AI_TOOLKIT_DIR}/requirements.txt",
    )
    .env({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App(APP_NAME, image=image)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.cls(timeout=900)
class ImageLoraWorker:
    @modal.method()
    def probe(self) -> dict:
        """Driver, CUDA, torch and ai-toolkit import check (no weights, no volumes)."""
        result: dict = {"ai_toolkit_commit": AI_TOOLKIT_COMMIT,
                        "hf_token_present": bool(os.environ.get("HF_TOKEN", "").strip())}
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
        result["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
        try:
            import torch

            # Plain str/bool only: the host has no torch to unpickle torch types.
            result.update(torch=str(torch.__version__), torch_cuda=str(torch.version.cuda),
                          cuda_available=bool(torch.cuda.is_available()))
            if torch.cuda.is_available():
                x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
                result["matmul_ok"] = bool(torch.isfinite((x @ x).float().sum()).item())
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            result["torch_error"] = repr(exc)[:500]
        check = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); import toolkit.config_modules, diffusers, "
             "transformers; from extensions_built_in.diffusion_models import QwenImageModel; "
             "print(diffusers.__version__, transformers.__version__)"],
            cwd=AI_TOOLKIT_DIR, capture_output=True, text=True)
        result["ai_toolkit_import"] = (check.stdout.strip() if check.returncode == 0
                                       else ("FAILED: " + check.stderr.strip()[-800:]))
        return result

    @modal.method()
    def train(self, spec: dict) -> dict:
        """Run one ai-toolkit job against the mounted per-run Volume.

        ``spec`` keys: run_name, job_yaml, model_repo, model_revision,
        cache_volume, run_volume, commit_every_seconds.
        """
        started = time.time()
        cache_volume = modal.Volume.from_name(spec["cache_volume"])
        run_volume = modal.Volume.from_name(spec["run_volume"])
        output_dir = Path(RUN_MOUNT) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        status = {"run_name": spec["run_name"], "phase": "download", "started": started}
        (output_dir / "status.json").write_text(json.dumps(status))
        run_volume.commit()

        from huggingface_hub import snapshot_download

        # One pinned snapshot in the shared cache; later runs reuse it.
        model_path = snapshot_download(spec["model_repo"], revision=spec["model_revision"])
        cache_volume.commit()
        download_s = time.time() - started

        job_text = (spec["job_yaml"].replace("${MODEL_PATH}", model_path)
                    .replace("${DATASET_DIR}", f"{RUN_MOUNT}/dataset")
                    .replace("${OUTPUT_DIR}", str(output_dir)))
        job_path = output_dir / "job.yaml"
        job_path.write_text(job_text)
        status.update(phase="train", model_path=model_path, download_seconds=round(download_s))
        (output_dir / "status.json").write_text(json.dumps(status))
        run_volume.commit()

        stop = threading.Event()

        def committer() -> None:
            while not stop.wait(int(spec.get("commit_every_seconds", 120))):
                try:
                    run_volume.commit()
                except Exception:  # noqa: BLE001 - best effort; final commit below
                    pass

        thread = threading.Thread(target=committer, daemon=True)
        thread.start()
        # The log is appended in short open/close bursts: Volume.commit() reloads
        # afterwards, and a reload is refused while this container holds files
        # open, so a permanently open log would block every mid-run commit.
        # Mid-run visibility stays best-effort (ai-toolkit holds its own files);
        # `modal app logs <app_id>` is the live view, the final commit is exact.
        pending: list[str] = []
        last_flush = time.time()

        def flush_log() -> None:
            nonlocal last_flush
            if pending:
                with open(output_dir / "train.log", "a", encoding="utf-8") as log:
                    log.writelines(pending)
                pending.clear()
            last_flush = time.time()

        proc = subprocess.Popen([sys.executable, "run.py", str(job_path)], cwd=AI_TOOLKIT_DIR,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:              # universal newlines: tqdm's \r become lines
            sys.stdout.write(line)
            pending.append(line)
            if time.time() - last_flush > 10:
                flush_log()
        returncode = proc.wait()
        flush_log()
        stop.set()
        thread.join(timeout=10)

        run_dir = output_dir / spec["run_name"]
        files = {}
        for path in sorted(run_dir.rglob("*.safetensors")) if run_dir.exists() else []:
            files[str(path.relative_to(output_dir))] = {"sha256": _sha256(path),
                                                        "bytes": path.stat().st_size}
        samples = sorted(str(p.relative_to(output_dir)) for p in run_dir.glob("samples/*")
                         if p.is_file()) if run_dir.exists() else []    # skips ai-toolkit's samples/.tmp
        completion = {
            "schema_version": "synaptic-image-lora-completion/v1",
            "run_name": spec["run_name"], "returncode": returncode,
            "status": "succeeded" if returncode == 0 and files else "failed",
            "model_repo": spec["model_repo"], "model_revision": spec["model_revision"],
            "model_path": model_path, "ai_toolkit_commit": AI_TOOLKIT_COMMIT,
            "download_seconds": round(download_s), "total_seconds": round(time.time() - started),
            "files": files, "samples": samples,
        }
        (output_dir / "completion.json").write_text(json.dumps(completion, indent=2))
        status.update(phase="done", returncode=returncode)
        (output_dir / "status.json").write_text(json.dumps(status))
        run_volume.commit()
        return completion
