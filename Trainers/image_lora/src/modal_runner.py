"""
Host-side Modal orchestration for the image_lora method.

Location: Trainers/image_lora/src/modal_runner.py
Purpose:  Launch, observe, fetch and clean up one image-LoRA training run on Modal
          with a minimal, explicit resource lifecycle:

          resource                      lifetime
          ----------------------------  -------------------------------------------
          base-weights cache Volume     shared by every run; created if missing,
                                        never deleted by a run
          per-run Volume                created exclusively at launch (dataset +
                                        outputs); deleted by ``cleanup`` only after
                                        ``fetch`` downloaded and verified outputs
          ``hf-token`` Secret           existing named Secret, referenced by name;
                                        never created, copied or deleted
          app                           ephemeral and detached; stops when the
                                        call ends; ``cleanup`` stops it if not

          All state lives in ``<output_dir>/run_state.json`` so every step is
          resumable from a new shell. No credential value is read or written.
Used by:  Trainers/image_lora/train_image_lora.py.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import secrets
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_config import plan_run, timeout_for_budget

STATE_FILE = "run_state.json"
SAMPLE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
VOLUME_DATASET = "/dataset"
VOLUME_OUTPUT = "/output"


# --------------------------------------------------------------------------- seams


def _sdk():
    """The Modal SDK (patched in tests)."""
    import modal

    return modal


def _worker_module():
    """The remote app module (patched in tests)."""
    import modal_app

    return modal_app


def _resolve_revision(repo: str, revision: str) -> str:
    """Pin a Hub revision to a commit SHA (patched in tests)."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, revision=revision).sha


# --------------------------------------------------------------------------- naming


def new_run_id(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def resource_names(config: dict[str, Any], run_id: str) -> dict[str, str]:
    modal_cfg = config["modal"]
    names = {"cache_volume": modal_cfg["cache_volume"],
             "run_volume": f"{modal_cfg['run_volume_prefix']}-{run_id}",
             "hf_secret": modal_cfg["hf_secret"]}
    if names["run_volume"] == names["cache_volume"]:
        raise ValueError("per-run Volume name collides with the shared cache Volume")
    if len(names["run_volume"]) > 64:
        raise ValueError("per-run Volume name is longer than Modal's 64 characters")
    return names


# --------------------------------------------------------------------------- state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _event(state: dict[str, Any], what: str, **extra: Any) -> None:
    state.setdefault("events", []).append(
        {"at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "event": what, **extra})


def _secret(modal: Any, config: dict[str, Any]) -> Any:
    modal_cfg = config["modal"]
    return modal.Secret.from_name(modal_cfg["hf_secret"], environment_name=modal_cfg.get("environment"),
                                  required_keys=[modal_cfg.get("hf_secret_key", "HF_TOKEN")])


# --------------------------------------------------------------------------- probe


def probe(*, gpu: str = "L4", timeout_seconds: int = 900, config: dict[str, Any] | None = None) -> dict:
    """Build/pull the training image and run the GPU smoke test (no Volumes)."""
    modal, worker = _sdk(), _worker_module()
    secret_cfg = config or {"modal": {"hf_secret": "hf-token", "hf_secret_key": "HF_TOKEN"}}
    cls = worker.ImageLoraWorker.with_options(gpu=gpu, timeout=timeout_seconds,
                                              secrets=[_secret(modal, secret_cfg)])
    started = time.time()
    with modal.enable_output():
        with worker.app.run():
            result = cls().probe.remote()
    result["wall_seconds"] = round(time.time() - started)
    return result


# --------------------------------------------------------------------------- launch


def launch(config: dict[str, Any], *, dataset_dir: Path, output_dir: Path, max_usd: float,
           run_id: str | None = None) -> dict[str, Any]:
    """Create the per-run Volume, upload the dataset and spawn training detached."""
    modal, worker = _sdk(), _worker_module()
    run_id = run_id or new_run_id()
    run_name = f"image_lora_{run_id.replace('-', '_')}"
    plan = plan_run(config, dataset_dir=dataset_dir, run_name=run_name)
    timeout_s = timeout_for_budget(plan.estimate, max_usd)
    names = resource_names(config, run_id)
    model = config["_model"]
    revision = _resolve_revision(model["repo"], model.get("revision", "main"))
    state_path = output_dir / STATE_FILE
    if state_path.exists():
        raise FileExistsError(f"{state_path} exists; use a new --output-dir per run")
    modal_cfg = config["modal"]
    env = modal_cfg.get("environment")
    state: dict[str, Any] = {
        "schema_version": "synaptic-image-lora-run/v1", "run_id": run_id, "run_name": run_name,
        "state_path": str(state_path), "output_dir": str(output_dir), "dataset_dir": str(dataset_dir),
        "status": "launching", "resources": names, "environment": env,
        "model": {"repo": model["repo"], "revision": revision, "arch": model["arch"]},
        "gpu": modal_cfg["gpu"], "timeout_seconds": timeout_s, "max_usd": max_usd,
        "estimate": plan.summary()["estimate"], "plan": plan.summary(),
        "cleanup": {"run_volume_deleted": False, "app_stopped": False},
    }
    _event(state, "planned", estimate_usd=plan.estimate.usd, timeout_seconds=timeout_s)
    _write_state(state_path, state)
    (output_dir / "job.yaml").write_text(plan.job_yaml, encoding="utf-8")

    # Shared cache: reuse if present, create once otherwise. Never per run.
    cache = modal.Volume.from_name(names["cache_volume"], environment_name=env, create_if_missing=True)
    # Per-run Volume: exclusive create, so a name clash can never adopt old data.
    modal.Volume.objects.create(names["run_volume"], environment_name=env, allow_existing=False)
    state["status"] = "run_volume_created"
    _event(state, "run_volume_created", name=names["run_volume"])
    _write_state(state_path, state)
    run_volume = modal.Volume.from_name(names["run_volume"], environment_name=env)

    images = sorted((dataset_dir / "images").iterdir())
    with run_volume.batch_upload() as batch:
        batch.put_directory(str(dataset_dir / "images"), VOLUME_DATASET)
        batch.put_file(str(dataset_dir / "manifest.json"), "/dataset_manifest.json")
    uploaded = [e for e in run_volume.listdir(VOLUME_DATASET)]
    if len(uploaded) != len(images):
        raise RuntimeError(f"upload check failed: {len(uploaded)} remote vs {len(images)} local files")
    _event(state, "dataset_uploaded", files=len(uploaded))

    spec = {"run_name": run_name, "job_yaml": plan.job_yaml, "model_repo": model["repo"],
            "model_revision": revision, "cache_volume": names["cache_volume"],
            "run_volume": names["run_volume"],
            "commit_every_seconds": int(modal_cfg.get("commit_every_seconds", 120))}
    cls = worker.ImageLoraWorker.with_options(
        gpu=modal_cfg["gpu"], cpu=float(modal_cfg.get("cpu", 8)),
        memory=int(modal_cfg.get("memory_mib", 65536)), timeout=timeout_s,
        volumes={worker.CACHE_MOUNT: cache, worker.RUN_MOUNT: run_volume},
        secrets=[_secret(modal, config)])
    with modal.enable_output():
        with worker.app.run(detach=True, environment_name=env) as running:
            call = cls().train.spawn(spec)
            state["app_id"] = getattr(running, "app_id", None)
            state["call_id"] = call.object_id
    state["status"] = "running"
    state["launched_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    _event(state, "spawned", app_id=state["app_id"], call_id=state["call_id"])
    _write_state(state_path, state)
    return state


# --------------------------------------------------------------------------- observe


def _read_text(volume: Any, path: str, limit: int | None = None) -> str | None:
    try:
        data = b"".join(volume.read_file(path))
    except Exception:  # noqa: BLE001 - not written yet
        return None
    if limit is not None:
        data = data[-limit:]
    return data.decode("utf-8", errors="replace")


def status(state_path: Path, *, tail: int = 20) -> dict[str, Any]:
    modal = _sdk()
    state = load_state(state_path)
    call = modal.FunctionCall.from_id(state["call_id"])
    result: dict[str, Any] = {"run_id": state["run_id"], "call_id": state["call_id"]}
    try:
        completion = call.get(timeout=0)
        result.update(call_state="finished", completion_status=completion.get("status"))
        state["status"] = "finished" if completion.get("status") == "succeeded" else "failed"
        state["completion"] = {k: completion[k] for k in ("status", "returncode", "total_seconds",
                                                          "download_seconds") if k in completion}
    except TimeoutError:
        result["call_state"] = "running"
    except Exception as exc:  # noqa: BLE001 - provider failure / timeout
        result.update(call_state="failed", error=type(exc).__name__)
        state["status"] = "failed"
    volume = modal.Volume.from_name(state["resources"]["run_volume"],
                                    environment_name=state.get("environment"))
    log = _read_text(volume, f"{VOLUME_OUTPUT}/train.log", limit=200_000) or ""
    result["log_tail"] = [ln for ln in log.splitlines() if ln.strip()][-tail:]
    status_json = _read_text(volume, f"{VOLUME_OUTPUT}/status.json")
    result["remote_status"] = json.loads(status_json) if status_json else None
    try:
        samples = volume.listdir(f"{VOLUME_OUTPUT}/{state['run_name']}/samples")
        result["samples"] = len(samples)
    except Exception:  # noqa: BLE001
        result["samples"] = 0
    _write_state(Path(state_path), state)
    return result


def wait(state_path: Path, *, poll_seconds: int = 120) -> dict[str, Any]:
    while True:
        result = status(state_path, tail=3)
        if result["call_state"] != "running":
            return result
        print(json.dumps({k: result[k] for k in ("call_state", "samples")} | {"tail": result["log_tail"]}))
        time.sleep(poll_seconds)


# --------------------------------------------------------------------------- fetch


def verify_safetensors(path: Path) -> dict[str, Any]:
    """Parse the safetensors header (no torch): tensor count, dtypes, byte bounds."""
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        if header_len <= 0 or header_len > 100_000_000:
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(header_len))
    metadata = header.pop("__metadata__", {})
    size = path.stat().st_size
    end = max((t["data_offsets"][1] for t in header.values()), default=0)
    if not header or 8 + header_len + end != size:
        raise ValueError("safetensors tensor data does not match file size")
    dtypes = sorted({t["dtype"] for t in header.values()})
    return {"tensors": len(header), "dtypes": dtypes, "bytes": size,
            "metadata_keys": sorted(metadata)[:20]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(volume: Any, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    with open(tmp, "wb") as handle:
        for chunk in volume.read_file(remote):
            handle.write(chunk)
    tmp.replace(local)


def checkpoint_name(run_name: str, step: int) -> str:
    return f"{run_name}/{run_name}_{step:09d}.safetensors"


def fetch(state_path: Path, *, checkpoints: list[int]) -> dict[str, Any]:
    """Download logs, samples, the final LoRA and chosen checkpoints; verify all."""
    modal = _sdk()
    state = load_state(state_path)
    out = Path(state["output_dir"])
    volume = modal.Volume.from_name(state["resources"]["run_volume"],
                                    environment_name=state.get("environment"))
    completion_text = _read_text(volume, f"{VOLUME_OUTPUT}/completion.json")
    if completion_text is None:
        return {"verified": False, "reason": "completion.json not found; the run has not finished"}
    completion = json.loads(completion_text)
    (out / "remote").mkdir(parents=True, exist_ok=True)
    (out / "remote" / "completion.json").write_text(completion_text, encoding="utf-8")
    for name in ("train.log", "job.yaml", "status.json"):
        try:
            _download(volume, f"{VOLUME_OUTPUT}/{name}", out / "remote" / name)
        except Exception:  # noqa: BLE001 - optional files
            pass
    samples_ok, samples_failed = 0, []
    for sample in completion.get("samples", []):
        if Path(sample).suffix.lower() not in SAMPLE_SUFFIXES:
            continue                      # e.g. ai-toolkit's samples/.tmp directory
        try:
            _download(volume, f"{VOLUME_OUTPUT}/{sample}", out / "samples" / Path(sample).name)
            samples_ok += 1
        except Exception as exc:  # noqa: BLE001 - samples are evidence, not the artifact
            samples_failed.append({"sample": sample, "error": type(exc).__name__})

    run_name = state["run_name"]
    wanted = [f"{run_name}/{run_name}.safetensors"] + [checkpoint_name(run_name, s) for s in checkpoints]
    report: dict[str, Any] = {"files": {}, "samples": samples_ok, "samples_failed": samples_failed}
    ok = completion.get("status") == "succeeded"
    for rel in wanted:
        expected = completion["files"].get(rel)
        if expected is None:
            report["files"][rel] = {"ok": False, "reason": "not in completion manifest"}
            ok = False
            continue
        local = out / "lora" / Path(rel).name
        _download(volume, f"{VOLUME_OUTPUT}/{rel}", local)
        entry: dict[str, Any] = {"local": str(local), "sha256": _sha256(local), "bytes": local.stat().st_size}
        entry["ok"] = entry["sha256"] == expected["sha256"] and entry["bytes"] == expected["bytes"]
        try:
            entry["safetensors"] = verify_safetensors(local)
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, safetensors_error=str(exc))
        ok = ok and entry["ok"]
        report["files"][rel] = entry
    report["verified"] = ok
    report["remote_checkpoints"] = sorted(completion["files"])
    state["fetch"] = report
    state["status"] = "fetched_verified" if ok else "fetched_unverified"
    _event(state, "fetched", verified=ok, files=list(report["files"]))
    _write_state(Path(state_path), state)
    return report


# --------------------------------------------------------------------------- cleanup


def _stop_app(app_id: str | None, environment: str | None) -> bool:
    if not app_id:
        return True
    cmd = [sys.executable, "-m", "modal", "app", "stop", app_id]
    if environment:
        cmd += ["--env", environment]
    done = subprocess.run(cmd, capture_output=True, text=True)
    return done.returncode == 0 or "already" in (done.stdout + done.stderr).lower()


def cleanup(state_path: Path, *, force: bool = False) -> dict[str, Any]:
    """Delete the per-run Volume and stop the app; refuse before a verified fetch.

    The shared cache Volume and the ``hf-token`` Secret are never touched.
    """
    modal = _sdk()
    state = load_state(state_path)
    if not force and not state.get("fetch", {}).get("verified"):
        return {"cleaned": False, "reason": "outputs not downloaded and verified; run fetch first "
                                            "(or --force to discard them)"}
    env = state.get("environment")
    result: dict[str, Any] = {"run_volume": state["resources"]["run_volume"]}
    if not state["cleanup"].get("run_volume_deleted"):
        modal.Volume.objects.delete(state["resources"]["run_volume"], environment_name=env,
                                    allow_missing=True)
        state["cleanup"]["run_volume_deleted"] = True
    if not state["cleanup"].get("app_stopped"):
        state["cleanup"]["app_stopped"] = _stop_app(state.get("app_id"), env)
    result.update(state["cleanup"])
    result["kept"] = {"cache_volume": state["resources"]["cache_volume"],
                      "secret": state["resources"]["hf_secret"]}
    result["cleaned"] = all(state["cleanup"].values())
    state["status"] = "cleaned" if result["cleaned"] else state["status"]
    _event(state, "cleanup", **state["cleanup"], forced=force)
    _write_state(Path(state_path), state)
    return result
