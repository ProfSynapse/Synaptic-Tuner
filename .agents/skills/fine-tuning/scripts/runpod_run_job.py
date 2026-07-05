"""
scripts/runpod_run_job.py

Standalone CLI for running a one-shot job on a RunPod GPU pod: clone a git
repo at a pinned commit, execute a wrapper script inside a pinned container
image, then terminate the pod. Generic by design -- the repo, commit, wrapper,
and image are all arguments; nothing here is project-specific.

Uses the RunPod REST API (https://rest.runpod.io/v1) rather than the GraphQL
SDK, for two hard-won reasons (2026-07-05 lane debugging, 4 dead pods):
  1. dockerEntrypoint override. Images that ship an ENTRYPOINT (e.g.
     unsloth/unsloth's /usr/local/bin/entrypoint.sh) receive GraphQL
     dockerArgs as *arguments to the entrypoint*, which can crash-loop the
     container forever with uptime pinned at 0 while the pod bills as
     RUNNING. The REST field dockerEntrypoint replaces the ENTRYPOINT so the
     job command actually runs.
  2. The SDK embeds docker_args into its GraphQL mutation with zero escaping
     (dockerArgs: "{docker_args}"), so any double quote corrupts the API
     call. REST takes a JSON body; no quoting restrictions.

Lifecycle: create pod -> wait for real container uptime (a stalled image
pull or crash-looping entrypoint shows RUNNING with uptime 0 and must NOT
disarm the boot timeout) -> poll until EXITED -> ALWAYS terminate in finally
(billing safety). No network volume: the wrapper uploads its own outputs
(e.g. to a HuggingFace repo).

Env: RUNPOD_API_KEY (required); HF_TOKEN forwarded to the pod when set.
Usage:
  python scripts/runpod_run_job.py --run-tag my-job \
      --repo-url https://github.com/org/repo.git --commit <full-40-char-sha> \
      --wrapper path/inside/repo/job.sh --wrapper-args "--stage extract" \
      [--gpu "NVIDIA GeForce RTX 3090"] [--image <img@sha256:...>] \
      [--min-download 700] [--allowed-cuda "12.8,12.9,13.0"] \
      [--timeout-min 180] [--probe-only] [--dry-run]
"""

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"

# Default image matches the tuner's runpod training backend pin.
DEFAULT_IMAGE = (
    "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update"
    "@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39"
)
POLL_INTERVAL_S = 30
# Covers image pull + container start on a host passing --min-download;
# a ~20GB image at 700 Mbps pulls in ~4-5 min, so 15 min is generous and
# anything slower means we picked a bad host and should recycle.
BOOT_TIMEOUT_S = 900
# How long a failed job holds the container alive before exiting nonzero, so
# the uptime poller can distinguish "ran and failed" (uptime climbs to this
# then EXITED) from a sub-2s crash-loop (uptime never leaves 1-2s). One poll
# interval plus margin.
FAIL_HOLD_S = 300


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
    ap.add_argument("--min-download", type=int, default=700,
                    help="minimum host download speed in Mbps; community hosts "
                         "below this stall for tens of minutes pulling large images")
    ap.add_argument("--allowed-cuda", default="",
                    help="comma-separated CUDA versions the host must offer "
                         "(e.g. '12.8,12.9,13.0'); a host whose driver is older "
                         "than the image's CUDA fails container start silently")
    ap.add_argument("--disk-gb", type=int, default=60)
    ap.add_argument("--timeout-min", type=int, default=180)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra pod env var (repeatable)")
    ap.add_argument("--probe-only", action="store_true",
                    help="skip clone+wrapper; just print GPU info and exit "
                         "(boot-reliability isolation test)")
    ap.add_argument("--no-entrypoint-override", action="store_true",
                    help="do not set dockerEntrypoint; let the image's own "
                         "ENTRYPOINT/CMD run (isolates whether the override "
                         "itself prevents container start)")
    ap.add_argument("--max-hosts", type=int, default=3,
                    help="pods to try before giving up: a community host that "
                         "does not start the container within the boot timeout "
                         "(stalled image pull) is terminated and a fresh pod "
                         "is created, up to this many attempts (boot-fail "
                         "cost is ~$0.05/attempt at 3090 rates)")
    ap.add_argument("--dry-run", action="store_true", help="print the pod spec and exit")
    return ap.parse_args()


def build_job_command(args: argparse.Namespace) -> str:
    """The shell command the container runs (via dockerEntrypoint bash -lc).

    The command runs as the image's default container user, which is NOT
    necessarily root: the unsloth image runs as unsloth:runtimeusers with
    HOME=/home/unsloth and no write access to /root. A `cd /root` there fails
    "Permission denied", and under `set -e` that aborts in <1s, so RunPod
    restart-loops the container with uptime pinned at 1-2s -- exactly the
    r5/edrt9x crash-loop (adjudicated by a local docker repro on the same
    image family, 2026-07-05). Use a writable, image-agnostic workdir instead
    of assuming root's home is available or writable.
    """
    if args.probe_only:
        # Stay alive long enough for uptime polling to observe the container:
        # a command that exits immediately gets restart-looped by RunPod and
        # is indistinguishable from a boot failure via the API (no log API,
        # runpod-python#400). Uptime > 0 IS the probe signal.
        return "nvidia-smi; echo PROBE_OK; sleep 240"
    # /workspace is the unsloth image's WORKDIR and is writable by the
    # container user; runpod/pytorch images also provide it. Falling back to
    # $HOME keeps this working on any image whose user can write there.
    workdir = "${WORKDIR:-/workspace}"
    inner = " && ".join(
        [
            "set -euo pipefail",
            # Grouped so the fallback binds only to the cd, not to the rest of
            # the && chain: a bare `cd A || cd B && clone` parses as
            # `cd A || (cd B && clone)` and SKIPS the clone whenever cd A
            # succeeds (equal-precedence left-associative && / ||).
            f'{{ cd {workdir} 2>/dev/null || cd "$HOME"; }}',
            # Idempotent: RunPod restarts an exited container against the SAME
            # writable layer, so a stale ./repo from a prior life would make
            # `git clone ... repo` die "destination path exists" every restart.
            "rm -rf repo",
            f"git clone --filter=blob:none {shlex.quote(args.repo_url)} repo",
            "cd repo",
            f"git checkout {shlex.quote(args.commit)}",
            f"bash {shlex.quote(args.wrapper)} {args.wrapper_args}",
        ]
    )
    # Failure persistence: on nonzero exit, print the code and hold the
    # container alive past one poll interval so the uptime poller can tell
    # "ran and failed" (uptime climbs to FAIL_HOLD_S then EXITED) apart from a
    # crash-loop (uptime stuck at 1-2s). Without this, every fast failure is
    # invisibly indistinguishable from a boot failure over the API.
    return (
        f"( {inner} ); rc=$?; "
        'if [ "$rc" -ne 0 ]; then '
        'echo "[runpod-job] JOB FAILED rc=$rc; holding for diagnosis"; '
        f"sleep {FAIL_HOLD_S}; fi; exit $rc"
    )


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


def api(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "runpod-run-job/1.0",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from e


def poll_pod(api_key: str, pod_id: str) -> dict:
    """Return {status, uptime} for a pod.

    Create/terminate go through REST, but boot polling CANNOT: the REST pod
    object carries no uptime field at all (only desiredStatus/lastStartedAt),
    so an uptime check against REST reads every pod as never-booted forever
    (2026-07-05: six healthy-or-not pods were killed at the boot timeout by
    exactly that bug). Actual container runtime lives only in the GraphQL
    pod.runtime.uptimeInSeconds.
    """
    query = ('query{pod(input:{podId:"%s"}){desiredStatus '
             "runtime{uptimeInSeconds}}}" % pod_id)
    # GraphQL auth is the SDK-style api_key query param; a Bearer header
    # alone gets 403. api.runpod.io also 403s the default Python-urllib
    # User-Agent (Cloudflare), so send an explicit UA (both observed
    # 2026-07-05).
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "runpod-run-job/1.0"},
        data=json.dumps({"query": query}).encode(),
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    pod = (data.get("data") or {}).get("pod") or {}
    runtime = pod.get("runtime") or {}
    return {"status": pod.get("desiredStatus", "UNKNOWN"),
            "uptime": int(runtime.get("uptimeInSeconds") or 0)}


def build_pod_spec(args: argparse.Namespace) -> dict:
    spec = {
        "name": args.run_tag,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "cloudType": args.cloud_type,
        "containerDiskInGb": args.disk_gb,
        "minDownloadMbps": args.min_download,
        "env": build_pod_env(args),
    }
    if not args.no_entrypoint_override:
        # Full ENTRYPOINT replacement: the image's own entrypoint never runs.
        spec["dockerEntrypoint"] = ["bash", "-lc", build_job_command(args)]
    cuda = [v.strip() for v in args.allowed_cuda.split(",") if v.strip()]
    if cuda:
        spec["allowedCudaVersions"] = cuda
    return spec


def terminate(api_key: str, pod_id: str) -> None:
    """Terminate with retry; the one call that must never be skipped."""
    for attempt in range(3):
        try:
            api("DELETE", f"/pods/{pod_id}", api_key)
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

    if args.dry_run:
        spec = build_pod_spec(args)
        redacted = dict(spec, env={k: "***" for k in spec["env"]})
        print(json.dumps(redacted, indent=2))
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY is not set")
        return 2

    for attempt in range(1, args.max_hosts + 1):
        result = run_once(args, api_key, attempt)
        if result != "boot_fail":
            return result
        # Boot failure means THIS HOST never started the container (stalled
        # image pull); the workload never ran, so a fresh pod is a clean retry,
        # not a re-run. Anything after boot (job error, timeout) is final.
        if attempt < args.max_hosts:
            print(f"[runpod] recycling host ({attempt}/{args.max_hosts} attempts used)")
    print(f"[runpod] giving up after {args.max_hosts} boot failures")
    return 1


def run_once(args: argparse.Namespace, api_key: str, attempt: int) -> int | str:
    """One pod lifecycle. Returns an exit code, or 'boot_fail' (container
    never started before BOOT_TIMEOUT_S -> caller may recycle onto a new host).
    """
    pod_id = None
    try:
        pod = api("POST", "/pods", api_key, build_pod_spec(args))
        pod_id = pod.get("id")
        if not pod_id:
            print(f"ERROR: create pod returned no id: {pod}")
            return 1
        print(f"[runpod] pod {pod_id} created (${pod.get('costPerHr', '?')}/hr, attempt {attempt})")
        print(f"[runpod] console: https://www.runpod.io/console/pods/{pod_id}")

        # Boot wait, then completion poll. EXITED == the job command finished
        # and its outputs are already wherever the wrapper uploads them.
        start = time.time()
        deadline = start + args.timeout_min * 60
        booted = False
        while time.time() < deadline:
            try:
                info = poll_pod(api_key, pod_id)
            except Exception as e:  # noqa: BLE001
                print(f"[runpod] poll error (retrying): {e}")
                time.sleep(POLL_INTERVAL_S)
                continue
            status = info["status"]
            uptime = info["uptime"]
            # RUNNING with uptime 0 means the image is still pulling or the
            # container is crash-looping; it has NOT booted. Only real uptime
            # counts, so the boot timeout stays armed through a stalled pull.
            if status == "RUNNING" and uptime > 0:
                if not booted:
                    print("[runpod] container started")
                    booted = True
                    if args.probe_only:
                        # boot observed = probe answered; don't bill the sleep out
                        print(f"[runpod] PROBE BOOT OK (uptime {uptime}s)")
                        return 0
                print(f"[runpod] up {uptime}s")
            elif status in ("EXITED", "TERMINATED", "STOPPED"):
                print(f"[runpod] pod finished (status {status})")
                return 0 if status == "EXITED" else 1
            elif not booted and time.time() - start > BOOT_TIMEOUT_S:
                print(f"[runpod] pod failed to boot within {BOOT_TIMEOUT_S}s (status {status})")
                return "boot_fail"
            time.sleep(POLL_INTERVAL_S)
        print(f"[runpod] TIMEOUT after {args.timeout_min} min")
        return 1
    finally:
        if pod_id:
            terminate(api_key, pod_id)


if __name__ == "__main__":
    sys.exit(main())
