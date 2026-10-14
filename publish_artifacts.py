"""Caller-owned local publication of an already verified, retained HF run."""

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

# Only checked-in host examples enter the path. Engine imports must be installed.
sys.path.insert(
    0, str(Path(__file__).parent / "synaptic-tuner/examples/training_qualification")
)
import hf_host
import launch
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest
from synaptic_tuner.api.v1.reference.authority import (
    ReferencePublicationEvidenceAuthorityV1,
)
from synaptic_tuner.api.v1.reference.local_publication import (
    LocalArtifactSpoolV1,
    LocalDestinationRegistryV1,
    LocalPublicationDestinationV1,
)
from synaptic_tuner.api.v1.reference.publication_sources import (
    compose_publication_artifacts,
)
from synaptic_tuner.api.v1.reference.publication_store import DurablePublicationStoreV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef


def load(path):
    value = launch._json(launch.stable_read(Path(path), 65536))
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema_version",
            "consumer",
            "run",
            "root",
            "destination_ref",
            "authority_ref",
            "key_ref",
            "secret",
            "maximum_artifact_bytes",
            "maximum_total_bytes",
            "seconds",
        }
        or value["schema_version"] != "synaptic-local-publication-host/v1"
    ):
        raise ValueError("publication configuration invalid")
    TrainingRunRef.from_dict(value["run"])
    SecretRef.from_dict(value["secret"])
    root = Path(value["root"])
    if not root.is_absolute() or root != Path(os.path.normpath(root)):
        raise ValueError("publication root invalid")
    for field, maximum in (
        ("maximum_artifact_bytes", 1 << 30),
        ("maximum_total_bytes", 5 << 30),
        ("seconds", 300),
    ):
        if type(value[field]) is not int or not 1 <= value[field] <= maximum:
            raise ValueError("publication bound invalid")
    if value["maximum_artifact_bytes"] > value["maximum_total_bytes"]:
        raise ValueError("publication aggregate invalid")
    return value


def local_publication(config, action, *, runs, secrets, clock):
    """Own local adapters only; caller retains RunsAPI and its host lifetime."""
    if action not in {"publish", "lookup"}:
        raise ValueError("publication action invalid")
    root = Path(config["root"])
    bound = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if action == "publish":
        root.mkdir(mode=0o700)  # Exclusive. Existing or failed roots never replay.
        launch._private(root)
        fd = os.open(root / "binding.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(bound)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        for name in ("locks", "spool", "destination"):
            (root / name).mkdir(mode=0o700)
    else:
        launch._private(root)
        if launch.stable_read(root / "binding.json", 65536) != bound:
            raise ValueError("original publication binding required")
    key = secrets.resolve(SecretRef.from_dict(config["secret"])).encode("utf-8")
    authority = ReferencePublicationEvidenceAuthorityV1(config["key_ref"], key)
    deadline = time.monotonic() + config["seconds"]
    run = TrainingRunRef.from_dict(config["run"])
    with ExitStack() as stack:
        records = stack.enter_context(
            SQLiteDurableStoreV1(
                root / "publication.sqlite3", create=action == "publish"
            )
        )
        store = DurablePublicationStoreV1(
            records, root / "locks", create=action == "publish"
        )
        stack.callback(store.close)
        spool = None
        if action == "publish":
            spool = LocalArtifactSpoolV1(
                root / "spool",
                maximum_artifact_bytes=config["maximum_artifact_bytes"],
                maximum_total_bytes=config["maximum_total_bytes"],
                deadline=deadline,
            )
            stack.callback(spool.close)
        destination = LocalPublicationDestinationV1(
            root / "destination",
            destination_ref=config["destination_ref"],
            display_name="Verified native training artifacts",
            role_paths={role: "native/" + role + ".bin" for role in launch.SFT_ROLES},
            maximum_artifact_bytes=config["maximum_artifact_bytes"],
            maximum_total_bytes=config["maximum_total_bytes"],
            authority_ref=config["authority_ref"],
            authority=authority,
            clock=clock,
            deadline=deadline,
            spool=spool,
        )
        stack.callback(destination.close)
        api = compose_publication_artifacts(
            runs=runs,
            store=store,
            spool=spool,
            destinations=LocalDestinationRegistryV1((destination,)),
            authority_ref=config["authority_ref"],
            authority=authority,
            clock=clock,
        )
        if action == "lookup":
            matches = tuple(
                result
                for result in api.publications(config["destination_ref"]).publications
                if result.run == run
            )
            if len(matches) != 1:
                raise ValueError("original publication unavailable")
        result = api.publish(PublicationRequest(run, config["destination_ref"]))
        verified = api.verify(result.publication)
        if result.state.value != "verified" or not verified.verified:
            raise ValueError("publication is not verified; retain original state")
        return {"publication": result.to_dict(), "verified": verified.verified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "lookup"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load(args.config)
    original = launch.read_committed(config["consumer"], maximum=4 << 20)
    consumer = launch.load_config(config["consumer"]["path"])
    if consumer != launch._json(original):
        raise ValueError("loaded consumer differs from authenticated bytes")
    if consumer["provider"] != "hf_jobs" or consumer["run"] != config["run"]:
        raise ValueError("original HF run required")
    if launch.read_committed(config["consumer"], maximum=4 << 20) != original:
        raise ValueError("consumer commitment changed")
    with (
        launch.host_ports(
            consumer, Path(consumer["state_root"]), create=False
        ) as ports,
        hf_host.open_host(
            consumer, state_root=consumer["state_root"], mode="read", ports=ports
        ) as values,
    ):
        _, runs, _, run = values
        if run.to_dict() != config["run"]:
            raise ValueError("retained run mismatch")
        result = local_publication(
            config,
            args.action,
            runs=runs,
            secrets=ports.secrets,
            clock=ports.clock.now,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 -- fixed non-secret command diagnostic
        print("local_publication_unavailable", file=sys.stderr)
        raise SystemExit(2) from None
