"""Generic Modal runner for config-driven MechInterp pipelines.

This module is intentionally experiment-agnostic. It clones a repository at an
exact commit, restores configured checkpoint artifacts from a Modal Volume, then
executes `python tuner.py mechinterp run --provider local --config ...` inside
the remote container.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


try:
    import modal
except ImportError as exc:  # pragma: no cover - local import diagnostic
    raise SystemExit("Install Modal first: pip install modal") from exc


HOURS = 60 * 60
APP_NAME = os.environ.get("MI_MODAL_APP_NAME", "mechinterp-pipeline")
IMAGE = os.environ.get("MI_MODAL_IMAGE", "vllm/vllm-openai:latest")
GPU = os.environ.get("MI_MODAL_GPU", "A10G")
TIMEOUT_HOURS = float(os.environ.get("MI_MODAL_TIMEOUT_HOURS", "3"))
VOLUME_NAME = os.environ.get("MI_MODAL_VOLUME", "mechinterp-pipeline")
MOUNT_PATH = os.environ.get("MI_MODAL_MOUNT", "/vol/mechinterp")
CHECKPOINT_INTERVAL_SEC = int(os.environ.get("MI_MODAL_CHECKPOINT_INTERVAL_SEC", "120"))
PIP = [item for item in os.environ.get("MI_MODAL_PIP", "pyyaml\npydantic").splitlines() if item]
APT = [item for item in os.environ.get("MI_MODAL_APT", "git").splitlines() if item]


def _image():
    img = (
        modal.Image.from_registry(IMAGE, add_python=None)
        .entrypoint([])
        .env({"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    )
    if APT:
        img = img.apt_install(*APT)
    if PIP:
        img = img.pip_install(*PIP)
    return img


app = modal.App(APP_NAME, image=_image())
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sh(cmd: list[str], *, cwd: str | None = None, check: bool = True) -> int:
    print(f"[mechinterp-modal] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def _checkpoint_paths(workspace: str, config: str) -> list[str]:
    try:
        import yaml

        with open(os.path.join(workspace, config), encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        artifacts = data.get("artifacts", {}) if isinstance(data, dict) else {}
        paths = artifacts.get("checkpoint_paths", []) if isinstance(artifacts, dict) else []
        return [str(path) for path in paths]
    except Exception as exc:  # noqa: BLE001
        print(f"[mechinterp-modal] checkpoint path load failed (non-fatal): {exc}", flush=True)
        return []


def _copy_path(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return 1
    count = 0
    for root, _dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        for filename in files:
            s = Path(root) / filename
            d = dst / rel_root / filename
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_suffix(d.suffix + ".tmp")
            shutil.copyfile(s, tmp)
            os.replace(tmp, d)
            count += 1
    return count


def _restore_checkpoints(workspace: str, config: str, paths: list[str]) -> None:
    root = Path(MOUNT_PATH) / "ckpt" / Path(config).stem
    restored = 0
    for rel in paths:
        restored += _copy_path(root / rel, Path(workspace) / rel)
    print(f"[mechinterp-modal] restored {restored} checkpoint files", flush=True)


def _commit_checkpoints(workspace: str, config: str, paths: list[str], tag: str = "") -> None:
    root = Path(MOUNT_PATH) / "ckpt" / Path(config).stem
    copied = 0
    for rel in paths:
        copied += _copy_path(Path(workspace) / rel, root / rel)
    vol.commit()
    print(f"[mechinterp-modal] committed {copied} checkpoint files {tag}".rstrip(), flush=True)


def _run_with_periodic_checkpoint(cmd: list[str], *, cwd: str, config: str, paths: list[str]) -> int:
    print(f"[mechinterp-modal] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd)
    while True:
        rc = proc.poll()
        if rc is not None:
            _commit_checkpoints(cwd, config, paths, "(final)")
            return rc
        time.sleep(CHECKPOINT_INTERVAL_SEC)
        _commit_checkpoints(cwd, config, paths, "(periodic)")


@app.function(
    gpu=GPU,
    timeout=int(TIMEOUT_HOURS * HOURS),
    volumes={MOUNT_PATH: vol},
    secrets=[
        modal.Secret.from_dict(
            {
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "HF_HUB_DISABLE_XET": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
            }
        )
    ],
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(
    *,
    config: str,
    repo_url: str,
    repo_branch: str,
    repo_commit: str,
    only_step: str = "",
    from_step: str = "",
    skip_steps: list[str] | None = None,
    gpu_ack: bool = False,
    force_full_run: bool = False,
) -> dict:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    workspace = "/workspace/repo"
    if not os.path.isdir(os.path.join(workspace, ".git")):
        _sh(["git", "clone", repo_url, workspace])
    _sh(["git", "fetch", "origin"], cwd=workspace, check=False)
    if repo_branch:
        _sh(["git", "checkout", repo_branch], cwd=workspace, check=False)
    _sh(["git", "checkout", repo_commit], cwd=workspace)

    checkpoint_paths = _checkpoint_paths(workspace, config)
    _restore_checkpoints(workspace, config, checkpoint_paths)

    cmd = [
        "python",
        "tuner.py",
        "mechinterp",
        "run",
        "--config",
        config,
        "--provider",
        "local",
        "--yes",
    ]
    if only_step:
        cmd.extend(["--only-step", only_step])
    if from_step:
        cmd.extend(["--from-step", from_step])
    for step in skip_steps or []:
        cmd.extend(["--skip-step", step])
    if gpu_ack:
        cmd.append("--i-know-this-runs-on-gpu")
    if force_full_run:
        cmd.append("--force-full-run")

    rc = _run_with_periodic_checkpoint(
        cmd, cwd=workspace, config=config, paths=checkpoint_paths
    )
    done_dir = Path(MOUNT_PATH) / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    marker = done_dir / f"{repo_commit[:12]}-{Path(config).stem}.txt"
    marker.write_text(f"repo_commit={repo_commit}\nconfig={config}\nreturncode={rc}\n")
    vol.commit()
    return {"returncode": rc, "repo_commit": repo_commit, "config": config}


@app.local_entrypoint()
def main(
    config: str,
    repo_url: str,
    repo_branch: str = "main",
    repo_commit: str = "",
    only_step: str = "",
    from_step: str = "",
    skip_step: list[str] | None = None,
    i_know_this_runs_on_gpu: bool = False,
    force_full_run: bool = False,
) -> None:
    if not repo_commit:
        raise SystemExit("--repo-commit is required for Modal mechinterp runs")
    call = run_pipeline.spawn(
        config=config,
        repo_url=repo_url,
        repo_branch=repo_branch,
        repo_commit=repo_commit,
        only_step=only_step,
        from_step=from_step,
        skip_steps=skip_step or [],
        gpu_ack=i_know_this_runs_on_gpu,
        force_full_run=force_full_run,
    )
    print(f"[mechinterp-modal] spawned {call.object_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--repo-branch", default="main")
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--only-step", default="")
    parser.add_argument("--from-step", default="")
    parser.add_argument("--skip-step", action="append", default=[])
    parser.add_argument("--i-know-this-runs-on-gpu", action="store_true")
    parser.add_argument("--force-full-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        "Run through Modal: modal run --detach MechInterp/cloud/modal_runner.py --config ..."
    )
