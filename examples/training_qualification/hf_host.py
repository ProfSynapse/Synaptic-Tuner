"""Host-owned ordinary HF SFT adapter; no protected smoke or restart execution."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath

from synaptic_tuner.api.v1.reference import (
    ReferenceHostPortsV1,
    ReferenceRequestPortsV1,
    compose_reference_authority,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.runtime.dispatch import (
    WorkerControlLocationV1,
    CanonicalWorkloadFileLocationV1,
    build_source_worker_invocation,
    materialize_worker_bundle,
)
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.runtime_profiles import load_runtime_profile
from tuner.training import default_recipe_registry
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training.contracts import (
    TrainingPlan,
    CanonicalDocument,
    RuntimeSpec,
    ResourceSpec,
    ArtifactPolicy,
    RetainedTrainingInputStreamLease,
)
from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsSelection
from tuner.execution.providers.hf_jobs_provider_v1.preparation import HFJobsPreparation
from tuner.execution.providers.hf_jobs_provider_v1.composition import (
    compose_hf_jobs_host,
)
from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import (
    open_retained_hf_runs,
    open_retained_hf_lifecycle,
)


class _Clock:
    def now(self):
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file(spec, maximum):
    if type(spec) is not dict or set(spec) != {"path", "size_bytes", "sha256"}:
        raise ValueError("hf_input_invalid")
    if (
        type(spec["size_bytes"]) is not int
        or not 0 < spec["size_bytes"] <= maximum
        or type(spec["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is None
        or type(spec["path"]) is not str
        or not Path(spec["path"]).is_absolute()
    ):
        raise ValueError("hf_input_invalid")
    raw = stable_read(Path(spec["path"]), maximum)
    if (
        len(raw) != spec["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != spec["sha256"]
    ):
        raise ValueError("hf_input_changed")
    return raw


def _settings(config):
    if config.get("provider") != "hf_jobs" or type(config.get("hf_jobs")) is not dict:
        raise ValueError("hf_configuration_invalid")
    return config["hf_jobs"]


def _metadata(config):
    c = _settings(config)
    authority_ref = SecretRef.from_dict(config["host"]["authority_secret"])
    reader_ref = SecretRef.from_dict(config["host"]["evidence_secret"])
    deadline = c["read_seconds"]
    if type(c.get("allow_cancel_reconcile", False)) is not bool:
        raise ValueError("hf_configuration_invalid")
    if type(deadline) is not int or not 1 <= deadline <= 180:
        raise ValueError("hf_configuration_invalid")
    if any(
        type(c[name]) is not str or not c[name]
        for name in ("reader_authority_ref", "reader_key_ref")
    ):
        raise ValueError("hf_configuration_invalid")
    return c, authority_ref, reader_ref


def _read_settings(config):
    c, authority_ref, reader_ref = _metadata(config)
    # Cold mode intentionally ignores source/profile/material/prepared paths.
    provider = ProviderRef.from_dict(c["provider"])
    run = TrainingRunRef.from_dict(config["run"])
    if provider.provider_id != "hf_jobs":
        raise ValueError("hf_configuration_invalid")
    return c, provider, run, authority_ref, reader_ref


@contextmanager
def _preparation(config, clock):
    c, _, _ = _metadata(config)
    required = {
        "material_file",
        "prepared_file",
        "runtime_profile",
        "selection",
        "provider",
        "reader_authority_ref",
        "reader_key_ref",
        "maximum_cost_minor_units",
        "currency",
        "read_seconds",
    }
    if set(c) not in (required, required | {"allow_cancel_reconcile"}):
        raise ValueError("hf_configuration_invalid")
    material = CoordinatorResolvedMaterial.parse(
        _file(c["material_file"], 2 << 20), default_recipe_registry()
    )
    document = json.loads(material.canonical_bytes)
    if _file(config["request_file"], 1 << 20) != material.request_bytes:
        raise ValueError("hf_request_mismatch")
    profile_value = c["runtime_profile"]
    if type(profile_value) is not dict or set(profile_value) != {
        "name",
        "root",
        "sha256",
    }:
        raise ValueError("hf_profile_invalid")
    if not Path(profile_value["root"]).is_absolute():
        raise ValueError("hf_profile_invalid")
    profile = load_runtime_profile(profile_value["name"], Path(profile_value["root"]))
    if profile.profile_sha256.removeprefix("sha256:") != profile_value["sha256"]:
        raise ValueError("hf_profile_changed")
    rich = TrainingPlan(
        ExecutionSourceV1.from_dict(document["execution_source"]),
        CanonicalDocument.from_mapping(document["execution_context"]),
        CanonicalDocument.from_mapping(document["resolved_config"]),
        CanonicalDocument.from_mapping(document["workload"]),
        RuntimeSpec(**document["runtime"]),
        ResourceSpec(**document["resources"]),
        ArtifactPolicy(
            tuple(document["artifact_policy"]["required_kinds"]),
            document["artifact_policy"]["retain_checkpoints"],
        ),
    )
    s = c["selection"]
    fields = {
        "scope",
        "bucket_id",
        "source_prefix",
        "run_prefix",
        "flavor",
        "timeout_seconds",
        "host_token",
        "job_token",
        "quote_digest",
    }
    if type(s) is not dict or set(s) != fields:
        raise ValueError("hf_selection_invalid")
    provider = ProviderRef.from_dict(c["provider"])
    selection = HFJobsSelection(
        provider,
        ExecutionScopeV1(**s["scope"]),
        profile,
        stable_read(profile.inventory_path, 1 << 20),
        s["bucket_id"],
        s["source_prefix"],
        s["run_prefix"],
        s["flavor"],
        s["timeout_seconds"],
        SecretRef.from_dict(s["host_token"]),
        SecretRef.from_dict(s["job_token"]),
        s["quote_digest"],
        rich.resources,
    )
    control = WorkerControlLocationV1(PurePosixPath("/workspace/private/control"))
    bundle = materialize_worker_bundle(
        build_source_worker_invocation(
            rich, control, CanonicalWorkloadFileLocationV1(control.control_root)
        )
    )
    manifest = parse_offline_sft_worker_manifest(
        bundle.closure_manifest_bytes,
        source_ref=bundle.closure_manifest_source,
        manifest_path=Path(bundle.closure_manifest_runtime_path),
    )
    import tuner

    root = Path(tuner.__file__).resolve().parent.parent
    closure = {
        member.path.as_posix(): stable_read(
            root / member.path, max(1, member.size_bytes)
        )
        for member in manifest.closure.members
    }
    parent_root = root / "tuner/execution/providers/hf_jobs_provider_v1"
    parent_manifest = json.loads(
        stable_read(parent_root / "hf-parent-closure.json", 1 << 20)
    )
    parents = {
        member["path"]: stable_read(root / member["path"], max(1, member["size_bytes"]))
        for member in parent_manifest["members"]
    }
    identity = PreparedTrainingInputIdentity.from_dict(
        json.loads(bundle.canonical_workload_bytes)["configuration"]["document"][
            "dataset"
        ]
    )
    raw = _file(c["prepared_file"], 2 << 20)
    if (
        len(raw) != identity.size_bytes
        or hashlib.sha256(raw).hexdigest() != identity.content_digest
    ):
        raise ValueError("hf_prepared_input_changed")
    lease = RetainedTrainingInputStreamLease(identity, io.BytesIO(raw))
    try:
        prep = HFJobsPreparation(
            selection=selection,
            training_plan=rich,
            bundle=bundle,
            closure_members=closure,
            parent_members=parents,
            input_lease=lease,
            request_id=document["request_id"],
            project_ref=document["project_ref"],
            clock=clock,
            maximum_cost_minor_units=c["maximum_cost_minor_units"],
            currency=c["currency"],
            **({"allow_cancel_reconcile": c["allow_cancel_reconcile"]}
               if "allow_cancel_reconcile" in c else {}),
        )
        run = TrainingRunRef(material.run_id, document["project_ref"])
        if "run" in config and TrainingRunRef.from_dict(config["run"]) != run:
            raise ValueError("hf_run_mismatch")
        yield prep, material, run
    finally:
        lease.close()


def preview(config):
    """Credential-free compilation/custody preview; not public readiness admission."""
    with _preparation(config, _Clock()) as (prep, material, run):
        return {
            "provider": prep.selection.provider.to_dict(),
            "run": run.to_dict(),
            "basis": prep.plan.basis.to_dict(),
            "material_digest": prep.material_digest,
            "coordinator_material_sha256": hashlib.sha256(
                material.canonical_bytes
            ).hexdigest(),
            "profile_digest": prep.selection.digest,
            "public_preflight": "deferred_until_host_composition",
        }


class _Requests:
    def __init__(self, prep, material, run):
        self.prep, self.material, self.run = prep, material, run

    def load(self, canonical_json):
        if canonical_json.encode("utf-8") != self.material.request_bytes:
            raise ValueError("hf_request_mismatch")
        return TrainingRequest(
            self.prep.resolved.request_id,
            self.prep.resolved.project_ref,
            canonical_json,
        )

    def resolve(self, request):
        if request != self.load(self.material.request_bytes.decode("utf-8")):
            raise ValueError("hf_request_mismatch")
        return self.prep.resolved

    def for_plan(self, plan):
        if plan != self.prep.plan:
            raise ValueError("hf_plan_mismatch")
        return self.run


@contextmanager
def open_host(config, *, state_root, mode, ports=None, sdk=None, api=None):
    """Yield existing public facades; cold reads never reconstruct executable inputs.

    Shared consumer owns stores, clock, named secret resolution and grants.
    Dependency arguments are internal CPU-test boundaries, not configuration plugins.
    """
    if mode not in {"start", "read"}:
        raise ValueError("hf_mode_invalid")
    root = Path(state_root)
    if not root.is_absolute():
        raise ValueError("hf_state_root_invalid")
    if ports is None:
        from launch import host_ports

        context = host_ports(config, root, create=mode == "start")
    else:
        context = nullcontext(ports)
    with context as ports:
        if type(ports) is not ReferenceHostPortsV1:
            raise TypeError("exact host ports required")
        # Local validation precedes authority/credential resolution and SDK creation.
        with (
            _preparation(config, ports.clock) if mode == "start" else nullcontext(None)
        ) as prepared:
            if mode == "read":
                c, provider, run, authority_ref, reader_ref = _read_settings(config)
            else:
                prep, material, run = prepared
                c, authority_ref, reader_ref = _metadata(config)
                provider = prep.selection.provider
            authority = compose_reference_authority(
                clock=ports.clock, secrets=ports.secrets, authority_secret=authority_ref
            )
            key = ports.secrets.resolve(reader_ref).encode("utf-8")
            if sdk is None:
                import huggingface_hub as sdk
            if api is None:
                api = sdk.HfApi()
            if mode == "read":
                cold_options = {"records": ports.records, "streams": ports.streams,
                                "secrets": ports.secrets}
                opener = open_retained_hf_runs
                if c.get("allow_cancel_reconcile", False):
                    opener = open_retained_hf_lifecycle
                    cold_options = {"ports": ports, "allow_current_cancel_authorization": True}
                view = opener(
                    run=run,
                    **cold_options,
                    authority=authority,
                    sdk=sdk,
                    api=api,
                    scratch_root=root / "scratch",
                    reader_authority_ref=c["reader_authority_ref"],
                    reader_key_ref=c["reader_key_ref"],
                    reader_key=key,
                    deadline_seconds=c["read_seconds"],
                )
                yield None, view.runs, provider, run
            else:
                requests = _Requests(prep, material, run)
                composed = compose_hf_jobs_host(
                    preparation=prep,
                    sdk=sdk,
                    api=api,
                    ports=ports,
                    requests=ReferenceRequestPortsV1(requests, requests, requests),
                    authority=authority,
                    scratch_root=root / "scratch",
                    reader_authority_ref=c["reader_authority_ref"],
                    reader_key_ref=c["reader_key_ref"],
                    reader_key=key,
                    deadline_seconds=c["read_seconds"],
                    **({"allow_current_cancel_authorization": True}
                       if c.get("allow_cancel_reconcile", False) else {}),
                )
                public = composed.api()
                yield public.training, public.runs, provider, run


def local_show(config, *, ports):
    """Expose only authenticated local staging failure, without provider construction."""
    from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import read_local_stage_diagnostic

    _, _, run, authority_ref, _ = _read_settings(config)
    authority = compose_reference_authority(
        clock=ports.clock, secrets=ports.secrets, authority_secret=authority_ref)
    return read_local_stage_diagnostic(records=ports.records, streams=ports.streams,
        run=run, authority=authority)
