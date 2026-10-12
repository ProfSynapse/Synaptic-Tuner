"""Host-side Modal lifecycle for image_lora, against a fake SDK (no network, no spend).

Pins the resource policy that replaces the per-run sprawl of the packaged LLM
path: one shared cache Volume (create-if-missing, never deleted), one exclusive
per-run Volume deleted only after a verified fetch, and the existing ``hf-token``
Secret referenced by name and never created or copied.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import modal_runner
import run_config

CONFIG = Path(run_config.TRAINER_DIR) / "configs" / "config.yaml"


# --------------------------------------------------------------------------- fakes


class FakeVolume:
    def __init__(self, name: str) -> None:
        self.name, self.files = name, {}

    @contextlib.contextmanager
    def batch_upload(self):
        yield self

    def put_directory(self, local: str, remote: str) -> None:
        for path in sorted(Path(local).iterdir()):
            self.files[f"{remote}/{path.name}"] = path.read_bytes()

    def put_file(self, local: str, remote: str) -> None:
        self.files[remote] = Path(local).read_bytes()

    def listdir(self, path: str):
        prefix = path.rstrip("/") + "/"
        names = {k[len(prefix):].split("/")[0] for k in self.files if k.startswith(prefix)}
        if not names:
            raise FileNotFoundError(path)
        return [SimpleNamespace(path=n) for n in sorted(names)]

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        for i in range(0, len(data), 7):
            yield data[i:i + 7]


class FakeSDK:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.volumes: dict[str, FakeVolume] = {}
        sdk = self

        class VolumeObjects:
            def create(self, name, *, environment_name=None, allow_existing=False, version=None):
                sdk.calls.append(("volume.create", name, allow_existing))
                if name in sdk.volumes and not allow_existing:
                    raise RuntimeError("already exists")
                sdk.volumes.setdefault(name, FakeVolume(name))

            def delete(self, name, *, environment_name=None, allow_missing=False):
                sdk.calls.append(("volume.delete", name))
                sdk.volumes.pop(name, None)

        class Volume:
            objects = VolumeObjects()

            @staticmethod
            def from_name(name, *, environment_name=None, create_if_missing=False, version=None):
                sdk.calls.append(("volume.from_name", name, create_if_missing))
                if name not in sdk.volumes:
                    if not create_if_missing:
                        raise LookupError(name)
                    sdk.volumes[name] = FakeVolume(name)
                return sdk.volumes[name]

        class SecretObjects:
            def create(self, *a, **k):
                raise AssertionError("image_lora must never create a Secret")

        class Secret:
            objects = SecretObjects()

            @staticmethod
            def from_name(name, *, environment_name=None, required_keys=()):
                sdk.calls.append(("secret.from_name", name, tuple(required_keys)))
                return SimpleNamespace(name=name)

        self.Volume, self.Secret = Volume, Secret
        self.completion = None

        class FunctionCall:
            @staticmethod
            def from_id(call_id):
                def get(timeout=None):
                    if sdk.completion is None:
                        raise TimeoutError()
                    return sdk.completion
                return SimpleNamespace(get=get)

        self.FunctionCall = FunctionCall

    @contextlib.contextmanager
    def enable_output(self):
        yield


class FakeWorkerModule:
    CACHE_MOUNT, RUN_MOUNT = "/data/cache", "/data/run"

    def __init__(self, sdk: FakeSDK) -> None:
        self.options, self.specs = [], []
        module = self

        class Worker:
            @staticmethod
            def with_options(**kwargs):
                module.options.append(kwargs)
                spawn = lambda spec: module.specs.append(spec) or SimpleNamespace(object_id="fc-123")  # noqa: E731
                return lambda: SimpleNamespace(train=SimpleNamespace(spawn=spawn),
                                               probe=SimpleNamespace(remote=lambda: {"ok": True}))

        @contextlib.contextmanager
        def run(detach=False, environment_name=None):
            sdk.calls.append(("app.run", detach))
            yield SimpleNamespace(app_id="ap-xyz")

        self.ImageLoraWorker = Worker
        self.app = SimpleNamespace(run=run)


@pytest.fixture
def fake(monkeypatch):
    sdk = FakeSDK()
    worker = FakeWorkerModule(sdk)
    monkeypatch.setattr(modal_runner, "_sdk", lambda: sdk)
    monkeypatch.setattr(modal_runner, "_worker_module", lambda: worker)
    monkeypatch.setattr(modal_runner, "_resolve_revision", lambda repo, rev: "a" * 40)
    stopped = []
    monkeypatch.setattr(modal_runner, "_stop_app", lambda app_id, env: stopped.append(app_id) or True)
    return SimpleNamespace(sdk=sdk, worker=worker, stopped=stopped)


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    for stem in ("a", "b"):
        (root / "images" / f"{stem}.png").write_bytes(b"png")
        (root / "images" / f"{stem}.txt").write_text("trig, x")
    (root / "manifest.json").write_text(json.dumps({"trigger": "trig"}))
    (root / "sample_prompts.yaml").write_text(yaml.safe_dump(["trig, sample"]))
    return root


def _safetensors(payload: bytes = b"\x00\x01\x02\x03") -> bytes:
    header = json.dumps({"lora.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, len(payload)]},
                         "__metadata__": {"name": "x"}}).encode()
    return struct.pack("<Q", len(header)) + header + payload


def _launch(fake, tmp_path, max_usd=25.0):
    config = run_config.load_run_config(CONFIG)
    return config, modal_runner.launch(config, dataset_dir=_dataset(tmp_path), output_dir=tmp_path / "run",
                                       max_usd=max_usd, run_id="20261010-000000-abcdef")


def _finish(fake, state, *, corrupt=False):
    run_name = state["run_name"]
    lora = _safetensors()
    volume = fake.sdk.volumes[state["resources"]["run_volume"]]
    rel = f"{run_name}/{run_name}.safetensors"
    volume.files[f"/output/{rel}"] = lora + (b"x" if corrupt else b"")
    volume.files["/output/train.log"] = b"step 1\nstep 2\n"
    volume.files[f"/output/{run_name}/samples/s1.jpg"] = b"jpg"
    completion = {"status": "succeeded", "returncode": 0,
                  "files": {rel: {"sha256": hashlib.sha256(lora).hexdigest(), "bytes": len(lora)}},
                  # A worker built before the is_file() filter also listed ai-toolkit's samples/.tmp.
                  "samples": [f"{run_name}/samples/.tmp", f"{run_name}/samples/s1.jpg"]}
    volume.files["/output/completion.json"] = json.dumps(completion).encode()
    fake.sdk.completion = completion


def test_launch_uses_shared_cache_exclusive_run_volume_and_named_secret(fake, tmp_path):
    config, state = _launch(fake, tmp_path)
    calls = fake.sdk.calls
    assert ("volume.from_name", "synaptic-image-lora-base-cache", True) in calls
    assert ("volume.create", "synaptic-image-lora-run-20261010-000000-abcdef", False) in calls
    assert ("secret.from_name", "hf-token", ("HF_TOKEN",)) in calls
    assert ("app.run", True) in calls                     # detached, ephemeral
    opts = fake.worker.options[-1]
    assert opts["gpu"] == "H100"
    assert set(opts["volumes"]) == {"/data/cache", "/data/run"}
    assert 0 < opts["timeout"] <= 25.0 / state["estimate"]["usd_per_hour"] * 3600 + 1
    spec = fake.worker.specs[-1]
    assert spec["model_revision"] == "a" * 40             # pinned, not "main"
    assert "${MODEL_PATH}" in spec["job_yaml"]
    run_volume = fake.sdk.volumes[state["resources"]["run_volume"]]
    assert {"/dataset/a.png", "/dataset/a.txt", "/dataset_manifest.json"} <= set(run_volume.files)
    saved = json.loads((tmp_path / "run" / "run_state.json").read_text())
    assert saved["status"] == "running" and saved["call_id"] == "fc-123" and saved["app_id"] == "ap-xyz"


def test_launch_refuses_over_budget_before_any_cloud_call(fake, tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        _launch(fake, tmp_path, max_usd=1.0)
    assert fake.sdk.calls == []


def test_status_reports_running_then_finished(fake, tmp_path):
    _, state = _launch(fake, tmp_path)
    path = Path(state["state_path"])
    assert modal_runner.status(path)["call_state"] == "running"
    _finish(fake, state)
    result = modal_runner.status(path)
    assert result["call_state"] == "finished" and result["log_tail"] == ["step 1", "step 2"]


def test_cleanup_refuses_before_verified_fetch(fake, tmp_path):
    _, state = _launch(fake, tmp_path)
    result = modal_runner.cleanup(Path(state["state_path"]))
    assert result["cleaned"] is False
    assert not any(c[0] == "volume.delete" for c in fake.sdk.calls)


def test_fetch_verifies_then_cleanup_deletes_only_per_run_resources(fake, tmp_path):
    _, state = _launch(fake, tmp_path)
    _finish(fake, state)
    path = Path(state["state_path"])
    report = modal_runner.fetch(path, checkpoints=[])
    assert report["verified"] is True
    assert report["samples"] == 1 and report["samples_failed"] == []
    assert (tmp_path / "run" / "samples" / "s1.jpg").read_bytes() == b"jpg"
    result = modal_runner.cleanup(path)
    assert result["cleaned"] is True
    deletes = [c for c in fake.sdk.calls if c[0] == "volume.delete"]
    assert deletes == [("volume.delete", state["resources"]["run_volume"])]
    assert "synaptic-image-lora-base-cache" in fake.sdk.volumes
    assert fake.stopped == ["ap-xyz"]
    assert result["kept"] == {"cache_volume": "synaptic-image-lora-base-cache", "secret": "hf-token"}


def test_hash_mismatch_blocks_verification_and_cleanup(fake, tmp_path):
    _, state = _launch(fake, tmp_path)
    _finish(fake, state, corrupt=True)
    path = Path(state["state_path"])
    assert modal_runner.fetch(path, checkpoints=[])["verified"] is False
    assert modal_runner.cleanup(path)["cleaned"] is False
    assert state["resources"]["run_volume"] in fake.sdk.volumes


def test_missing_checkpoint_is_not_verified(fake, tmp_path):
    _, state = _launch(fake, tmp_path)
    _finish(fake, state)
    report = modal_runner.fetch(Path(state["state_path"]), checkpoints=[500])
    assert report["verified"] is False


def test_resource_names_never_collide_with_cache():
    config = run_config.load_run_config(CONFIG)
    names = modal_runner.resource_names(config, "r1")
    assert names["run_volume"] != names["cache_volume"]
    config["modal"]["run_volume_prefix"] = "x" * 70
    with pytest.raises(ValueError):
        modal_runner.resource_names(config, "r1")


def test_verify_safetensors_rejects_truncation(tmp_path):
    good = tmp_path / "good.safetensors"
    good.write_bytes(_safetensors())
    assert modal_runner.verify_safetensors(good)["tensors"] == 1
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(_safetensors()[:-1])
    with pytest.raises(ValueError):
        modal_runner.verify_safetensors(bad)
