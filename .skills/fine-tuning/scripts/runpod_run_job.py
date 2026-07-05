"""
scripts/runpod_run_job.py

Standalone CLI for running a one-shot job on a RunPod GPU pod: clone a git
repo at a pinned commit, execute a wrapper script inside a pinned container
image, then terminate the pod. Generic by design -- the repo, commit, wrapper,
and image are all arguments; nothing here is project-specific.

Complements tuner/backends/training/cloud/runpod_backend.py (which is
training-lane-only); this runs arbitrary wrapper scripts. Lifecycle is the
same: create_pod -> wait RUNNING -> poll until EXITED -> ALWAYS terminate in
finally (billing safety). No network volume: the wrapper is responsible for
uploading its own outputs (e.g. to a HuggingFace repo).

Env: RUNPOD_API_KEY (required); HF_TOKEN forwarded to the pod when set.
Usage:
  python scripts/runpod_run_job.py --run-tag my-job \
      --repo-url https://github.com/org/repo.git --commit <full-40-char-sha> \
      --wrapper path/inside/repo/job.sh --wrapper-args "--stage extract" \
      [--gpu "NVIDIA GeForce RTX 4090"] [--image <img@sha256:...>] \
      [--timeout-min 180] [--dry-run]
"""

import argparse
import os
import shlex
import sys
import time

# Default image matches the tuner's runpod training backend pin.
DEFAULT_IMAGE = (
    "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update"
    "@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39"
)
POLL_INTERVAL_S = 30
BOOT_TIMEOUT_S = 600


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", required=True, help="label; becomes the pod name")
    ap.add_argument("--repo-url", required=True, help="git repo the pod clones")
    ap.add_argument("--commit", required=True, help="FULL 40-char git sha to check out")
    ap.add_argument("--wrapper", required=True, help="repo-relative wrapper script path")
    ap.add_argument("--wrapper-args", default="", help="argument string passed to the wrapper")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="container image (pin by digest)")
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090", help="RunPod gpu_type_id")
    ap.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    ap.add_argument("--disk-gb", type=int, default=60)
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra pod env var (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the pod spec and exit")
    return ap.parse_args()


def build_startup_cmd(args: argparse.Namespace) -> str:
    """Clone at the pinned commit and exec the wrapper as the pod's docker_args."""
    inner = " && ".join(
        [
            "set -euo pipefail",
            "cd /root",
            f"git clone --filter=blob:none {shlex.quote(args.repo_url)} repo",
            "cd repo",
            f"git checkout {shlex.quote(args.commit)}",
            f"bash {shlex.quote(args.wrapper)} {args.wrapper_args}",
        ]
    )
    return f"bash -lc {shlex.quote(inner)}"


def build_pod_env(args: argparse.Namespace) -> dict:
    env = {"RUN_TAG": args.run_tag}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    for pair in args.env:
        if "=" not in pair:
            raise SystemExit(f"--env expects KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        env[key] = value
    return env


def terminate(runpod_module, pod_id: str) -> None:
    """Terminate with retry; the one call that must never be skipped."""
    for attempt in range(3):
        try:
            runpod_module.terminate_pod(pod_id)
            print(f"[runpod] pod {pod_id} terminated")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[runpod] terminate attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    print(
        f"[runpod] WARNING: could not terminate pod {pod_id}; "
        "kill it manually at https://www.runpod.io/console/pods"
    )


def main() -> int:
    args = parse_args()

    if len(args.commit) != 40:
        print(f"ERROR: --commit must be a full 40-char sha (got {len(args.commit)} chars)")
        return 2

    startup_cmd = build_startup_cmd(args)
    pod_name = args.run_tag

    if args.dry_run:
        print(f"pod_name:    {pod_name}")
        print(f"image:       {args.image}")
        print(f"gpu:         {args.gpu} ({args.cloud_type})")
        print(f"disk:        {args.disk_gb} GB, no network volume")
        print(f"timeout:     {args.timeout_min} min")
        print(f"env keys:    {sorted(build_pod_env(args))}")
        print(f"docker_args: {startup_cmd}")
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY is not set")
        return 2

    import runpod

    runpod.api_key = api_key

    pod_id = None
    try:
        pod = runpod.create_pod(
            name=pod_name,
            image_name=args.image,
            gpu_type_id=args.gpu,
            gpu_count=1,
            container_disk_in_gb=args.disk_gb,
            cloud_type=args.cloud_type,
            env=build_pod_env(args),
            docker_args=startup_cmd,
        )
        pod_id = pod.get("id")
        if not pod_id:
            print(f"ERROR: create_pod returned no id: {pod}")
            return 1
        print(f"[runpod] pod {pod_id} created (${pod.get('costPerHr', '?')}/hr)")
        print(f"[runpod] console: https://www.runpod.io/console/pods/{pod_id}")

        # Boot wait, then completion poll. EXITED == the wrapper finished and
        # its outputs are already wherever the wrapper uploads them.
        start = time.time()
        deadline = start + args.timeout_min * 60
        booted = False
        while time.time() < deadline:
            try:
                info = runpod.get_pod(pod_id)
            except Exception as e:  # noqa: BLE001
                print(f"[runpod] poll error (retrying): {e}")
                time.sleep(POLL_INTERVAL_S)
                continue
            status = (info or {}).get("desiredStatus", "UNKNOWN")
            runtime = (info or {}).get("runtime") or {}
            if status == "RUNNING" and runtime:
                if not booted:
                    print("[runpod] pod is RUNNING")
                    booted = True
                gpus = runtime.get("gpus") or [{}]
                print(
                    f"[runpod] up {runtime.get('uptimeInSeconds', 0)}s"
                    f" | gpu {gpus[0].get('gpuUtilPercent', 0)}%"
                    f" | vram {gpus[0].get('memoryUtilPercent', 0)}%"
                )
            elif status in ("EXITED", "TERMINATED", "STOPPED"):
                print(f"[runpod] pod finished (status {status})")
                return 0 if status == "EXITED" else 1
            elif not booted and time.time() - start > BOOT_TIMEOUT_S:
                print(f"[runpod] pod failed to boot within {BOOT_TIMEOUT_S}s (status {status})")
                return 1
            time.sleep(POLL_INTERVAL_S)
        print(f"[runpod] TIMEOUT after {args.timeout_min} min")
        return 1
    finally:
        if pod_id:
            terminate(runpod, pod_id)


if __name__ == "__main__":
    sys.exit(main())
