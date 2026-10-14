"""Explicit CPU image observations; no checked-in manifest or image is changed."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        + b"\n"
    )


def capture(source, destination, *, mutation=None):
    from tuner.runtime.offline_grpo_worker import captured_grpo_manifest_document
    from tuner.runtime.packaged_grpo_worker_closure import captured_grpo_outer_document
    from tuner.runtime import packaged_worker_closure as outer

    source = Path(source)
    offline = captured_grpo_manifest_document(source)
    offline_raw = canonical(offline)
    grpo = captured_grpo_outer_document(source)
    for member in grpo["members"]:
        if member["path"] == "tuner/runtime/manifests/offline-grpo-worker-v1.json":
            member.update(
                size_bytes=len(offline_raw),
                sha256=hashlib.sha256(offline_raw).hexdigest(),
            )
    grpo["closure_digest"] = hashlib.sha256(
        canonical({k: v for k, v in grpo.items() if k != "closure_digest"}).rstrip(
            b"\n"
        )
    ).hexdigest()
    members = []
    for name in outer.PACKAGED_WORKER_MEMBERS:
        raw = (source / "tuner/runtime" / name).read_bytes()
        members.append(
            {
                "path": name,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    sft = {"schema_version": outer.PACKAGED_WORKER_CLOSURE_SCHEMA, "members": members}
    sft["closure_digest"] = hashlib.sha256(outer._canonical(sft)).hexdigest()
    names = {m["path"] for m in offline["members"]}
    from tuner.runtime.offline_sft_worker import (
        load_packaged_offline_sft_worker_manifest,
    )

    names.update(
        m.path.as_posix()
        for m in load_packaged_offline_sft_worker_manifest().closure.members
    )
    names.update(
        {"Trainers/sft/configs/config.yaml", "Trainers/grpo/configs/config.yaml"}
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as wheel:
        for name in sorted(names):
            content = (source / name).read_bytes()
            if mutation == "wrong_wheel_python" and name == "Trainers/sft/train_sft.py":
                content += b"\n# test image mismatch\n"
            if (
                mutation == "missing_grpo_trainer"
                and name == "Trainers/grpo/runtime_v1.py"
            ):
                continue
            wheel.writestr(name, content)
    wheel_raw = buffer.getvalue()
    wheel_path = destination / "cpu-fixture.whl"
    wheel_path.write_bytes(wheel_raw)
    config = {
        "offline": offline,
        "grpo": grpo,
        "sft": sft,
        "wheel": str(wheel_path),
        "wheel_names": sorted(names),
        "wheel_sha256": hashlib.sha256(wheel_raw).hexdigest(),
    }
    config_path = destination / "cpu-image.json"
    config_path.write_bytes(canonical(config))
    return config_path, config


def install(source, config, setter=setattr):
    """Only package/wheel image observations are substituted, with real parsers."""
    from tuner.runtime import packaged_worker_closure as outer
    from tuner.runtime import packaged_grpo_worker_closure as grpo
    from tuner.runtime import offline_grpo_worker as offline
    from tuner.runtime import packaged_training_worker as worker
    from tuner.runtime import packaged_sft_child as child
    from tuner.runtime import packaged_sft_execution as shared

    source = Path(source)
    assert Path(worker.__file__).absolute().parents[2] == source
    wheel_raw = Path(config["wheel"]).read_bytes()
    assert hashlib.sha256(wheel_raw).hexdigest() == config["wheel_sha256"]
    metadata = canonical(
        {
            "wheel": {
                "filename": "cpu-fixture.whl",
                "distribution": "synaptic-tuner",
                "version": "1.2.3",
                "sha256": config["wheel_sha256"],
            }
        }
    )
    original_read = outer.stable_read

    def image_read(path, maximum=1024 * 1024, **kwargs):
        path = Path(path)
        if path == Path("/opt/synaptic-runtime/build-inputs.json"):
            raw = metadata
        elif path == Path("/opt/synaptic-runtime/cpu-fixture.whl"):
            raw = wheel_raw
        elif path == source / "tuner/runtime/manifests/offline-grpo-worker-v1.json":
            raw = canonical(config["offline"])
        else:
            return original_read(path, maximum, **kwargs)
        assert len(raw) <= maximum
        return raw

    setter(worker, "stable_read", image_read)
    setter(child, "stable_read", image_read)
    setter(grpo, "stable_read", image_read)
    original_resource = outer._resource_bytes
    setter(
        outer,
        "_resource_bytes",
        lambda resource, maximum: (
            canonical(config["sft"])
            if str(resource).endswith(outer.PACKAGED_WORKER_CLOSURE_RESOURCE)
            else original_resource(resource, maximum)
        ),
    )
    # Parse and verify all fixed current source members, never a fabricated digest.
    setter(worker, "load_packaged_worker_closure", outer.load_packaged_worker_closure)
    setter(
        grpo,
        "load_packaged_grpo_worker_closure",
        lambda: grpo.parse_packaged_grpo_worker_closure(
            canonical(config["grpo"]), root=source
        ),
    )
    manifest = offline.parse_offline_grpo_worker_manifest(canonical(config["offline"]))
    setter(offline, "load_packaged_offline_grpo_worker_manifest", lambda: manifest)

    class Distribution:
        files = tuple(PurePosixPath(name) for name in config["wheel_names"])

        def locate_file(self, name):
            return source / name

    original_distribution = child.importlib.metadata.distribution
    setter(
        child.importlib.metadata,
        "distribution",
        lambda name: (
            Distribution() if name == "synaptic-tuner" else original_distribution(name)
        ),
    )

    def image_inspection(release, *, method="sft"):
        with ZipFile(BytesIO(wheel_raw)) as wheel:
            return shared._require_trainer_assets(Distribution(), wheel, method=method)

    setter(shared, "_inspect_release", image_inspection)

    # Host /dev may contain display devices; this is an explicitly declared
    # CPU image namespace observation, never evidence about this machine.
    def image_gpu_namespace():
        import os

        assert os.environ.get("NVIDIA_VISIBLE_DEVICES") == "void"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    setter(child, "_check_no_gpu_devices", image_gpu_namespace)
    return (
        worker.load_packaged_worker_closure(),
        grpo.load_packaged_grpo_worker_closure(),
    )
