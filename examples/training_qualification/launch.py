"""Provider-neutral exercise of the existing public TrainingAPI and RunsAPI."""

from pathlib import Path

from tuner.training.modal_host_download import download_verified_modal_artifact

# Existing native SFT contract and verified downloader support these five roles.
SFT_ROLES = frozenset(
    {
        "final_model",
        "tokenizer",
        "training_lineage",
        "training_metrics",
        "workload_record",
    }
)


def plan(training, canonical_request, provider):
    request = training.load(canonical_request)
    resolved = training.resolve(request)
    value = training.plan(resolved, provider)
    return value, training.preflight(value)


def start(training, value, preflight):
    """One public call only. No polling/retry or consumed-attempt replacement."""
    if not preflight.ready or not preflight.binds(value):
        raise ValueError("public preflight refused")
    return training.start(value, preflight)


def observe(runs, run):
    """One known-target provider observation, distinct from stored show."""
    return runs.outcome(run)


def download(runs, run, destination, *, maximum_bytes=256 << 20):
    """Download only the retained native five-role set; no implicit verification."""
    outcome = runs.show(run)
    if {item.role for item in outcome.artifacts} != SFT_ROLES:
        raise ValueError("native five-role manifest unavailable")
    return tuple(
        download_verified_modal_artifact(
            runs,
            run,
            role=role,
            output_root=Path(destination),
            maximum_bytes=maximum_bytes,
        )
        for role in sorted(SFT_ROLES)
    )


def verify_and_download(runs, run, destination, *, maximum_bytes=256 << 20):
    report = runs.verify(run)
    if (
        not report.verified
        or {item.role for item in runs.show(run).artifacts} != SFT_ROLES
    ):
        raise ValueError("native SFT verification refused")
    return report, download(runs, run, destination, maximum_bytes=maximum_bytes)


def cancel(runs, run, reason):
    # The returned outcome preserves acknowledgment/nonterminal distinctions.
    return runs.cancel(run, reason)


def reconcile(runs, run):
    return runs.reconcile(run)


# Only this checkout-owned consumer directory is exposed for fixed sibling
# adapters. Never add the checkout engine root to an installed interpreter.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import time

from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.project.secrets import SecretRef, resolve_secret
from tuner.runtime.packaged_worker_closure import stable_read


class Clock:
    def now(self):
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def now_epoch(self):
        return int(time.time())


class NamedSecrets:
    def resolve(self, reference):
        return resolve_secret(reference, environment=os.environ)


class HostGrants:
    """Caller-selected existing GrantPort, not a separate grant service."""

    def __init__(self, config):
        host = config["host"]
        self.grant = ExecutionGrant(host["grant_ref"])
        self.allowed = tuple(
            AuthorizationRequirement.from_dict(r) for r in host["allowed_requirements"]
        )
        self.run = TrainingRunRef.from_dict(config["run"])

    def authorize(self, requirements):
        if not requirements or any(r not in self.allowed for r in requirements):
            raise ValueError("host requirement refused")
        return self.grant

    def bind(self, grant, *, operation, requirements):
        command = parse_exact_command(operation.canonical_bytes)
        if grant != self.authorize(requirements) or (
            command.preparation.run_id,
            command.preparation.project_ref,
        ) != (self.run.run_id, self.run.project_ref):
            raise ValueError("host command scope refused")
        return grant


@contextmanager
def host_ports(config, state_root, *, create):
    if type(create) is not bool:
        raise TypeError("exact create flag required")
    root = Path(state_root)
    _private(root)
    scratch = root / "scratch"
    if create:
        scratch.mkdir(mode=0o700)
    _private(scratch)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3", create=create) as store:
        yield ReferenceHostPortsV1(
            store, store, Clock(), HostGrants(config), NamedSecrets()
        )


def _private(root):
    value = root.lstat()
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o700
    ):
        raise ValueError("private directory required")


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON value")

    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def read_committed(reference, *, maximum=2 << 20):
    if set(reference) != {"path", "sha256", "size_bytes"}:
        raise ValueError("exact input commitment required")
    if (
        type(reference["size_bytes"]) is not int
        or not 0 < reference["size_bytes"] <= maximum
    ):
        raise ValueError("input size outside bound")
    path = Path(reference["path"])
    if not path.is_absolute():
        raise ValueError("absolute input path required")
    raw = stable_read(path, maximum)
    if (
        len(raw) != reference["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != reference["sha256"]
    ):
        raise ValueError("input commitment mismatch")
    return raw


def load_config(path):
    config = _json(stable_read(Path(path), 4 << 20))
    provider = config.get("provider")
    if provider not in {"docker", "hf_jobs"} or set(config) - {
        "provider",
        "state_root",
        "request_file",
        "host",
        provider,
        "run",
    }:
        raise ValueError("consumer configuration invalid")
    if not Path(config["state_root"]).is_absolute():
        raise ValueError("absolute state_root required")
    host = config["host"]
    if set(host) != {
        "authority_secret",
        "evidence_secret",
        "grant_ref",
        "allowed_requirements",
    }:
        raise ValueError("host references invalid")
    for name in ("authority_secret", "evidence_secret"):
        SecretRef.from_dict(host[name])
    ExecutionGrant(host["grant_ref"])
    for raw in host["allowed_requirements"]:
        AuthorizationRequirement.from_dict(raw)
    return config


def _adapter(config):
    if config["provider"] == "docker":
        import docker_host

        return docker_host
    import hf_host

    return hf_host


def _write(path, value):
    return _write_bytes(
        path, json.dumps(value, allow_nan=False, sort_keys=True).encode()
    )


def _write_bytes(path, raw):
    if len(raw) > 4 << 20:
        raise ValueError("retained document exceeds bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def preview(config):
    return _adapter(config).preview(config)


def launch(config, *, adapter=None, ports=None, adapter_options=None):
    """Create-only host attempt; no retry of ambiguous public start."""
    adapter = adapter or _adapter(config)
    expected = adapter.preview(config)
    canonical_request = read_committed(config["request_file"]).decode("utf-8")
    root = Path(config["state_root"])
    root.mkdir(mode=0o700)
    _private(root)
    config = _json(json.dumps(config, allow_nan=False).encode())
    config["run"] = expected["run"]
    retain = getattr(adapter, "retain_inputs", None)
    if callable(retain):
        config = retain(config, root)
        if adapter.preview(config) != expected:
            raise ValueError("retained material differs from preview")
    _write(root / "config.json", config)
    _write(root / "preview.json", expected)
    _write_bytes(root / "request.json", canonical_request.encode())
    from contextlib import nullcontext

    with (
        nullcontext(ports)
        if ports is not None
        else host_ports(config, root, create=True)
    ) as current:
        with adapter.open_host(
            config,
            state_root=root,
            mode="start",
            ports=current,
            **(adapter_options or {}),
        ) as values:
            training, runs, provider, expected_run = values
            if expected_run != TrainingRunRef.from_dict(expected["run"]):
                raise ValueError("host run differs from preview")
            value, preflight = plan(training, canonical_request, provider)
            if (
                value.basis.to_dict() != expected["basis"]
                or provider.to_dict() != expected["provider"]
            ):
                raise ValueError("public plan differs from preview")
            _write(root / "plan.json", value.to_dict())
            _write(root / "preflight.json", preflight.to_dict())
            result = start(training, value, preflight)
            if result.run != expected_run:
                raise ValueError("public start returned another run")
            return {
                "start": {"run": result.run.to_dict(), "accepted": result.accepted},
                "stored": runs.show(result.run).to_dict(),
            }


def known(
    config,
    action,
    *,
    destination=None,
    reason=None,
    adapter=None,
    ports=None,
    adapter_options=None,
):
    """No preview, material recompile or start in shared known-run routing."""
    adapter = adapter or _adapter(config)
    root = Path(config["state_root"])
    run = TrainingRunRef.from_dict(config["run"])
    from contextlib import nullcontext

    with (
        nullcontext(ports)
        if ports is not None
        else host_ports(config, root, create=False)
    ) as current:
        if action == "show" and hasattr(adapter, "local_show"):
            local = adapter.local_show(config, ports=current)
            if local is not None:
                return local
        with adapter.open_host(
            config,
            state_root=root,
            mode="read",
            ports=current,
            **(adapter_options or {}),
        ) as values:
            _, runs, _, expected_run = values
            if run != expected_run:
                raise ValueError("retained run mismatch")
            if action == "show":
                return runs.show(run).to_dict()
            if action == "observe":
                return observe(runs, run).to_dict()
            if action == "cancel":
                return cancel(runs, run, reason).to_dict()
            if action == "reconcile":
                return reconcile(runs, run).to_dict()
            if action == "verify":
                report = runs.verify(run)
                return {
                    "run": report.run.to_dict(),
                    "verified": report.verified,
                    "checked_at": report.checked_at,
                }
            if action == "download":
                return {"paths": [str(p) for p in download(runs, run, destination)]}
            raise ValueError("unsupported known action")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "start",
            "show",
            "observe",
            "verify",
            "download",
            "cancel",
            "reconcile",
        ),
    )
    parser.add_argument(
        "path", type=Path, help="host config for plan/start; retained state for reads"
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    try:
        if args.action in {"plan", "start"}:
            config = load_config(args.path)
            result = preview(config) if args.action == "plan" else launch(config)
        else:
            _private(args.path)
            config = load_config(args.path / "config.json")
            if Path(config["state_root"]) != args.path:
                raise ValueError("retained state root mismatch")
            result = known(
                config, args.action, destination=args.destination, reason=args.reason
            )
    except Exception:
        print(json.dumps({"diagnostic_code": "training_qualification_consumer_failed"}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
