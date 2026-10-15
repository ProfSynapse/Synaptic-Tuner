"""Ordinary public HF family with actual SDK value objects and a scripted provider.

Synthetic source/ML evidence is test-owned; no provider or model operations.
"""

from dataclasses import replace
import hashlib
import io
import json
import stat
import time
from importlib.resources import files
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

sdk = pytest.importorskip("huggingface_hub")
if sdk.__version__ != "1.33.0":
    pytest.skip("dedicated ordinary HF host SDK1.33 lane", allow_module_level=True)
import httpx
from huggingface_hub import HfApi
from huggingface_hub._buckets import BucketFile
from huggingface_hub._jobs_api import JobInfo
from huggingface_hub.utils import close_session, set_client_factory

from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.reference import (
    ReferenceHostPortsV1, ReferenceRequestPortsV1,
    compose_reference_authority,
)
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.foundation_v2.commands import build_stage_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.runtime.dispatch import WorkerControlLocationV1, CanonicalWorkloadFileLocationV1, build_source_worker_invocation, materialize_worker_bundle
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.runtime_profiles import load_runtime_profile
from tuner.training.contracts import CanonicalDocument, TrainingPlan, RuntimeSpec, ResourceSpec, ArtifactPolicy, RetainedTrainingInputStreamLease
from tuner.training.methods.sft import compile_sft_workload
from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsSelection, HFJobsError
from tuner.execution.providers.hf_jobs_provider_v1.preparation import HFJobsPreparation
from tuner.execution.providers.hf_jobs_provider_v1.composition import compose_hf_jobs_host
from tuner.execution.providers.hf_jobs_provider_v1.transport import HFJobsTransport
from tests.training.test_sft_compilation import _execution_source
from tests.runtime.test_artifact_verification import _runtime_workload, _lineage, _tar, _safetensors, _canonical


pytestmark = pytest.mark.skipif(sdk.__version__ != "1.33.0", reason="isolated ordinary HF host SDK lane")


class Clock:
    def now(self):
        return "2026-10-13T12:00:00Z"

    def now_epoch(self):
        return 1791892800


class Secrets:
    def resolve(self, ref):
        return "fixture-" + ref.name + "-" + "x" * 40


class Grants:
    def __init__(self):
        self.calls = []

    def authorize(self, requirements):
        self.calls.append("authorize")
        return ExecutionGrant("fixture-hf-start")

    def bind(self, grant, *, operation, requirements):
        self.calls.append(operation.operation.effect.kind.value)
        return object()


def _job(arguments, status="RUNNING"):
    return JobInfo(id="fixture-job-1", owner={"id": "fixture-owner-id", "name": "fixture-owner", "type": "user"},
        dockerImage=arguments["image"], command=arguments["command"], arguments=[],
        environment=arguments["env"], flavor=arguments["flavor"], labels=arguments["labels"],
        volumes=[v.to_dict() if hasattr(v, "to_dict") else v for v in arguments["volumes"]],
        status={"stage": status}, secrets={"HF_TOKEN": "never-retain-provider-secrets"})


class Provider:
    """Same retained fake provider state; no claim of remote restart qualification."""
    endpoint = "https://huggingface.co"

    def __init__(self):
        self.blobs, self.calls, self.job = {}, [], None

    def list_bucket_tree(self, bucket_id, *, prefix, recursive, token):
        self.calls.append(("list", prefix))
        return [BucketFile(type="file", path=path, size=len(raw), xetHash=hashlib.sha256(raw).hexdigest())
                for path, raw in sorted(self.blobs.items()) if path.startswith(prefix + "/")]

    def batch_bucket_files(self, bucket_id, *, add, token):
        self.calls.append(("upload", len(add)))
        for raw, path in add:
            assert path not in self.blobs
            self.blobs[path] = raw

    def download_bucket_files(self, bucket_id, files, *, token, raise_on_missing_files):
        assert raise_on_missing_files
        for item, leaf in files:
            leaf.write_bytes(self.blobs[item.path])

    def run_job(self, *, secrets, token, **arguments):
        self.calls.append(("submit", arguments))
        assert secrets == {"HF_TOKEN": Secrets().resolve(SecretRef("env", "JOB_TOKEN"))}
        self.job = _job(arguments)
        return self.job

    def inspect_job(self, *, job_id, namespace, token):
        self.calls.append(("inspect", job_id))
        assert (job_id, namespace) == (self.job.id, "fixture-owner")
        return self.job


@pytest.fixture
def material(tmp_path):
    profile_root = Path(str(files("Trainers.runtime_profiles").joinpath("qwen35-sft-v1.yaml"))).parent
    profile = load_runtime_profile("qwen35-sft-v1", profile_root)
    inventory = profile.inventory_path.read_bytes()
    selection = HFJobsSelection(ProviderRef("hf_jobs", "ordinary-fixture"),
        ExecutionScopeV1("fixture-owner", "fixture-owner"), profile, inventory,
        "fixture-owner/bucket", "source/fresh-attempt", "runs/fresh-attempt", "a10g-small", 300,
        SecretRef("env", "HOST_TOKEN"), SecretRef("env", "JOB_TOKEN"), "e" * 64, ResourceSpec("gpu", 1, 300))
    source = _execution_source()
    roots = {name: "/workspace/private/" + ("" if name in {"engine", "project"} else "run/") + name for name in source.roots}
    environment = {key: value.replace(old, roots[name]) for key, value in source.environment.items()
                   for name, old in source.roots.items() if old in value}
    environment = {**dict(source.environment), **environment}
    source = replace(source, roots=roots, writable_capability_root="/workspace/private/run", environment=environment,
        python_executable="/opt/unsloth-venv/bin/python3", python_version=profile.runtime_facts["python_version"])
    raw = b'{"schema_version":"syntunia-sft-row/v1","format":"raw_text","split":"train","text":"synthetic row"}\n'
    digest = hashlib.sha256(raw).hexdigest()
    identity = PreparedTrainingInputIdentity("prepared://sha256/" + digest, digest, digest, len(raw), "syntunia-sft-row/v1")
    configuration = _runtime_workload(raw_text=True).document["configuration"]["document"]
    model = next(iter(profile.model_revisions))
    revision = profile.model_revisions[model][0]
    configuration["model"].update(ref=model, revision=revision, tokenizer_revision=revision)
    configuration["dataset"] = identity.to_dict()
    configuration["sft"].update(dataset_format="raw_text", completion_only_loss=False,
                                 assistant_only_loss=False, use_preassigned_splits=True)
    config = CanonicalDocument.from_mapping(configuration)
    workload = compile_sft_workload(resolved_config=config, execution_source=source)
    rich = TrainingPlan(source, CanonicalDocument.from_mapping({"schema_version": "declared-test-context/v1"}),
        config, CanonicalDocument(workload.canonical_bytes.decode()),
        RuntimeSpec(image=profile.image, python_version=profile.runtime_facts["python_version"], dependency_lock_digest="d" * 64),
        ResourceSpec("gpu", 1, 300), ArtifactPolicy(tuple(sorted({
            "final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record"})), retain_checkpoints=False))
    control = WorkerControlLocationV1(PurePosixPath("/workspace/private/control"))
    bundle = materialize_worker_bundle(build_source_worker_invocation(rich, control, CanonicalWorkloadFileLocationV1(control.control_root)))
    manifest = parse_offline_sft_worker_manifest(bundle.closure_manifest_bytes,
        source_ref=bundle.closure_manifest_source, manifest_path=Path(bundle.closure_manifest_runtime_path))
    import tuner
    package_root = Path(tuner.__file__).resolve().parent.parent
    closure = {row.path.as_posix(): (package_root / row.path).read_bytes() for row in manifest.closure.members}
    parent_inventory = json.loads(files("tuner.execution.providers.hf_jobs_provider_v1").joinpath("hf-parent-closure.json").read_bytes())
    parents = {row["path"]: (package_root / row["path"]).read_bytes() for row in parent_inventory["members"]}
    preparation = HFJobsPreparation(selection=selection, training_plan=rich, bundle=bundle,
        closure_members=closure, parent_members=parents, input_lease=RetainedTrainingInputStreamLease(identity, io.BytesIO(raw)),
        request_id="fixture-request", project_ref="fixture-project", clock=Clock(), maximum_cost_minor_units=100, currency="USD")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    return SimpleNamespace(preparation=preparation, scratch=scratch, raw=raw, provider=Provider())


def _host(material, records=None, streams=None):
    prep = material.preparation
    ports = ReferenceHostPortsV1(records or InMemoryDurableRecordStoreV1(),
        streams or InMemoryDurableStreamStoreV1(), Clock(), Grants(), Secrets())
    authority = compose_reference_authority(clock=ports.clock, secrets=ports.secrets,
                                           authority_secret=SecretRef("env", "AUTHORITY"))
    class Loader:
        def load(self, canonical_json):
            from synaptic_tuner.api.v1.training_facade import TrainingRequest
            return TrainingRequest(prep.resolved.request_id, prep.resolved.project_ref, canonical_json)
    class Resolver:
        def resolve(self, request):
            return prep.resolved
    class Identity:
        def for_plan(self, plan):
            return TrainingRunRef(prep.rich_plan.execution_source.run_id, prep.resolved.project_ref)
    composition = compose_hf_jobs_host(preparation=prep, sdk=sdk, api=material.provider,
        ports=ports, requests=ReferenceRequestPortsV1(Loader(), Resolver(), Identity()), authority=authority,
        scratch_root=material.scratch, reader_authority_ref="fixture-reader", reader_key_ref="fixture-reader-key", reader_key=b"r" * 32)
    return composition.api(), composition, ports


def test_public_start_and_original_job_observation(material):
    api, composition, ports = _host(material)
    from synaptic_tuner.api.v1.training_facade import TrainingRequest
    request = TrainingRequest("fixture-request", "fixture-project", '{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    preflight = api.training.preflight(plan)
    run = api.training.start(plan, preflight).run
    assert len([call for call in material.provider.calls if call[0] == "submit"]) == 1, ([call[0] for call in material.provider.calls], len(material.provider.blobs), api.runs.show(run).state)
    assert api.runs.show(run).run == run
    outcome = api.runs.outcome(run)
    assert outcome.run == run
    assert len([call for call in material.provider.calls if call[0] == "submit"]) == 1
    encoded = b"".join(record.canonical for record in ports.records.list_page(partition="effects", prefix="", after_key=None, limit=100).records)
    assert b"never-retain-provider-secrets" not in encoded
    assert Secrets().resolve(SecretRef("env", "JOB_TOKEN")).encode() not in encoded


def test_changed_preparation_rejects_before_sdk(material):
    prep = material.preparation
    run = TrainingRunRef(prep.rich_plan.execution_source.run_id, prep.resolved.project_ref)
    sealed = prep.prepare(prep.plan, run, prep.binding)
    document = sealed.to_dict()
    document["source_digest"] = "f" * 64
    from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
    changed = CanonicalPreparationV2.parse(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    command = build_stage_command(changed, "fixture-nonce", prep.payload(changed, EffectKind.STAGE), prep.executor)
    transport = HFJobsTransport(preparation=prep, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    with pytest.raises(HFJobsError, match="binding_mismatch"):
        transport.stage(command)
    assert material.provider.calls == []


def test_actual_sdk_job_and_volume_serialization_without_network(material):
    prep = material.preparation
    run = TrainingRunRef(prep.rich_plan.execution_source.run_id, prep.resolved.project_ref)
    sealed = prep.prepare(prep.plan, run, prep.binding)
    from tuner.execution.foundation_v2.commands import build_submit_command
    from tuner.execution.foundation_v2.references import StagePredecessorV2
    predecessor = StagePredecessorV2("hf_jobs", prep.selection.provider.profile_ref,
        "fixture-owner", "fixture-owner", run.project_ref, run.run_id, prep.plan.plan_fingerprint,
        sealed.preparation_digest, sealed.workload_digest, "fixture-stage", "a" * 64, "b" * 64)
    command = build_submit_command(sealed, "fixture-nonce", prep.payload(sealed, EffectKind.SUBMIT), prep.executor, predecessor)
    transport = HFJobsTransport(preparation=prep, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    request = transport.job_request(command)
    observed = []
    def serve(http_request):
        assert http_request.method == "POST" and http_request.url.path == "/api/jobs/fixture-owner"
        value = json.loads(http_request.content)
        observed.append(value)
        assert value["volumes"] == [volume.to_dict() for volume in request["volumes"]]
        assert value["secrets"] == {"HF_TOKEN": "job-secret-only-in-sdk"}
        return httpx.Response(200, json={"id": "fixture-job-1", "owner": {"id": "owner", "name": "fixture-owner", "type": "user"},
            "status": {"stage": "RUNNING"}, **value})
    close_session()
    set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(serve)))
    try:
        actual = HfApi(endpoint="https://huggingface.co", token=False).run_job(**request,
            token="host-api-only", secrets={"HF_TOKEN": "job-secret-only-in-sdk"})
        transport._check_job(actual, request)
        assert len(observed) == 1
        assert "job-secret-only-in-sdk" not in json.dumps(request, default=lambda value: value.to_dict())
    finally:
        close_session()
        set_client_factory(httpx.Client)


def _native_values(preparation):
    """Declared synthetic native evidence, not GPU training or model quality."""
    workload = compile_sft_workload(resolved_config=preparation.rich_plan.resolved_config,
        execution_source=preparation.rich_plan.execution_source)
    config = workload.document["configuration"]["document"]
    roots = workload.document["execution_source"]["runtime"]["roots"]
    lineage = json.loads(_lineage(workload))
    execution = lineage["execution_evidence"]
    dataset_path = "/proc/self/fd/9"
    execution["dataset"]["resolved_path"] = dataset_path
    execution["argv"][0] = preparation.rich_plan.execution_source.python_executable
    replacements = {"--model-name": config["model"]["ref"], "--model-revision": config["model"]["revision"],
        "--local-file": dataset_path, "--runtime-v1-tokenizer-revision": config["model"]["tokenizer_revision"],
        "--runtime-v1-dataset-revision": config["dataset"]["revision"],
        "--runtime-v1-dataset-digest": config["dataset"]["content_digest"],
        "--model-snapshot": roots["cache"] + "/model/models--" + config["model"]["ref"].replace("/", "--") + "/snapshots/" + config["model"]["revision"]}
    for flag, value in replacements.items():
        execution["argv"][execution["argv"].index(flag) + 1] = value
    execution["environment"]["SYNAPTIC_MODEL_SNAPSHOT"] = replacements["--model-snapshot"]
    execution["environment"]["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] = str(preparation.bundle.closure_manifest_runtime_path)
    execution["environment"]["SYNAPTIC_WORKER_CLOSURE_DIGEST"] = preparation.bundle.closure_digest
    projection = lineage["trainer_lineage"]["synaptic_runtime_projection"]
    projection["dataset"]["resolved_path"] = dataset_path
    lineage["trainer_lineage"]["dataset"]["source"] = dataset_path
    lineage["trainer_lineage"]["model"]["base_model"] = config["model"]["ref"]
    lineage["execution_evidence_sha256"] = hashlib.sha256(_canonical(execution)).hexdigest()
    return {"workload_record": ("workload.json", workload.canonical_bytes),
        "training_lineage": ("training_lineage.json", _canonical(lineage)),
        "training_metrics": ("training_metrics.json", b'{"loss":0.25,"steps":1}'),
        "final_model": ("final_model.tar", _tar({"adapter_config.json": _canonical({
            "base_model_name_or_path": config["model"]["ref"], "peft_type": "LORA"}),
            "adapter_model.safetensors": _safetensors()})),
        "tokenizer": ("tokenizer.tar", _tar({"tokenizer_config.json": b'{"tokenizer_class":"Fixture"}',
            "tokenizer.json": b'{"model":{"type":"BPE","vocab":{"x":0}},"version":"1.0"}'}))}


def _complete(material, values):
    prep, provider = material.preparation, material.provider
    members = []
    for role, (name, raw) in sorted(values.items()):
        path = "artifacts/" + name
        provider.blobs[prep.selection.run_prefix + "/" + path] = raw
        members.append({"role": role, "path": path, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
    provider.blobs[prep.selection.run_prefix + "/completion.json"] = _canonical({
        "schema_version": "synaptic-hf-native-completion/v1", "status": "completed",
        "material_digest": prep.material_digest, "workload_fingerprint": prep.bundle.workload_fingerprint,
        "job_id": provider.job.id, "artifacts": members})
    provider.job.status.stage = "COMPLETED"


def test_public_complete_native_verify_and_stream_without_resubmission(material):
    api, composition, ports = _host(material)
    request = api.training.load('{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    run = api.training.start(plan, api.training.preflight(plan)).run
    values = _native_values(material.preparation)
    _complete(material, values)
    assert api.runs.outcome(run).state.value == "succeeded"
    assert api.runs.verify(run).run == run
    from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest
    streamed = b"".join(api.runs.artifacts(RunArtifactRequest(run, "final_model", len(values["final_model"][1]))).iter_bytes())
    assert streamed == values["final_model"][1]
    assert len([call for call in material.provider.calls if call[0] == "submit"]) == 1


@pytest.mark.parametrize("fault", ["partial_upload", "lost_submit_reply"])
def test_uncertain_effect_never_replays(material, fault):
    if fault == "partial_upload":
        def partial(bucket_id, *, add, token):
            material.provider.calls.append(("upload", 1))
            raw, path = add[0]
            material.provider.blobs[path] = raw
            raise RuntimeError("PRIVATE_PROVIDER_SENTINEL")
        material.provider.batch_bucket_files = partial
    else:
        original = material.provider.run_job
        def lost(**arguments):
            original(**arguments)
            raise RuntimeError("PRIVATE_PROVIDER_SENTINEL")
        material.provider.run_job = lost
    api, composition, ports = _host(material)
    plan = api.training.plan(api.training.resolve(api.training.load('{}')), material.preparation.selection.provider)
    ready = api.training.preflight(plan)
    run = api.training.start(plan, ready).run
    assert api.runs.show(run).state.value == "reconcile_required"
    before = list(material.provider.calls)
    api.training.start(plan, ready)
    assert material.provider.calls == before
    assert material.provider.blobs


def test_actual_sqlite_cold_read_does_not_stage_submit_or_bind(material, tmp_path):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    database = tmp_path / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as store:
        api, composition, ports = _host(material, store, store)
        plan = api.training.plan(api.training.resolve(api.training.load('{}')), material.preparation.selection.provider)
        run = api.training.start(plan, api.training.preflight(plan)).run
        _complete(material, _native_values(material.preparation))
        api.runs.outcome(run)
        api.runs.verify(run)
    calls = list(material.provider.calls)
    with SQLiteDurableStoreV1(database, create=False) as reopened:
        cold, composition, ports = _host(material, reopened, reopened)
        assert cold.runs.show(run).run == run
        assert cold.runs.outcome(run).state.value == "succeeded"
        assert composition.stores.workflow_store.get(run).phase.value == "verified"
        from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest
        expected = _native_values(material.preparation)["final_model"][1]
        assert b"".join(cold.runs.artifacts(RunArtifactRequest(run, "final_model", len(expected))).iter_bytes()) == expected
        assert ports.grants.calls == []
    assert [call for call in material.provider.calls if call[0] in {"upload", "submit"}] == [
        call for call in calls if call[0] in {"upload", "submit"}]


@pytest.mark.parametrize("attack", [None, "parent", "engine", "input", "inventory",
    "runtime", "model", "model_subclass", "model_mutated", "dispatch", "artifact", *range(20, 26), *range(30, 40),
    True, -9, 1, 120])
def test_actual_bootstrap_worker_with_owned_files_and_injected_ml_effects(material, tmp_path, monkeypatch, attack):
    """Map fixed mount paths to private test dirs; retain actual FD/hash I/O."""
    from tuner.execution.providers.hf_jobs_provider_v1 import bootstrap, source_worker
    from tuner.execution.providers.modal import model_snapshot
    prep = material.preparation
    mount_source, mount_output = tmp_path / "source", tmp_path / "output"
    workspace = tmp_path / "workspace"
    for path in (mount_source, mount_output, workspace):
        path.mkdir(mode=0o700)
    for name, raw in prep.source_files.items():
        bootstrap._write(mount_source, name, raw)
    input_name = "input/" + prep.input_identity.content_digest + "/private-dataset.jsonl"
    bootstrap._write(mount_output, input_name, material.raw)
    original_read, original_write, original_path = bootstrap._read, bootstrap._write, Path
    def mapped(value):
        raw = str(value)
        if raw == "/source":
            return mount_source
        if raw == "/artifacts":
            return mount_output
        if raw.startswith("/workspace/"):
            return workspace / raw.removeprefix("/workspace/")
        return original_path(value)
    def read(root, relative, size):
        return original_read(mapped(root), relative, size)
    def write(root, relative, raw, **options):
        return original_write(mapped(root), relative, raw, **options)
    monkeypatch.setattr(bootstrap, "Path", mapped)
    monkeypatch.setattr(bootstrap, "_read", read)
    monkeypatch.setattr(bootstrap, "_write", write)
    monkeypatch.setattr(source_worker, "Path", mapped)
    monkeypatch.setattr(source_worker, "_read", read)
    monkeypatch.setattr(source_worker, "_write", write)
    inspected, models, dispatched = [], [], []
    def runtime(inventory, source):
        assert inventory == json.loads(prep.selection.inventory_bytes)
        assert source == prep.rich_plan.execution_source.to_dict()
        if attack == "runtime":
            raise RuntimeError("private fixture text must never escape")
        inspected.append(True)
    monkeypatch.setattr(source_worker, "_runtime", runtime)
    def model(**arguments):
        assert arguments["token"] is False
        assert arguments["model_ref"] == prep.rich_plan.resolved_config.to_dict()["model"]["ref"]
        assert arguments["destination_root"] == workspace / "private/run/cache"
        if attack == "model":
            raise RuntimeError("private fixture text must never escape")
        if attack == "model_subclass":
            class Hostile(source_worker.HFJobFailure):
                pass
            raise Hostile("DISPATCH", 22)
        if attack == "model_mutated":
            error = source_worker.HFJobFailure("DISPATCH", 22)
            error.exit_code = 120
            raise error
        models.append(True)
    monkeypatch.setattr(model_snapshot, "prepare_model_snapshot", model)
    values = _native_values(prep)
    def child(argv, *, cwd, env, input, stdout, stderr, timeout, check):
        assert argv == list(prep.bundle.dispatch.argv)
        assert cwd == str(prep.bundle.dispatch.cwd)
        assert env == dict(prep.bundle.dispatch.environment)
        assert "HF_TOKEN" not in env and "JOB_ID" not in env
        assert input == prep.bundle.dispatch.stdin
        assert 0 < timeout <= prep.selection.timeout_seconds and check is False
        assert stdout is stderr
        assert read(workspace / "private", "run/state/prepared-inputs/" + prep.input_identity.content_digest + "/private-dataset.jsonl", len(material.raw)) == material.raw
        if attack == "dispatch":
            raise TimeoutError("private fixture text must never escape")
        if type(attack) in {int, bool}:
            return SimpleNamespace(returncode=attack)
        if attack == "artifact":
            return SimpleNamespace(returncode=0)
        members = []
        for role, (name, raw) in sorted(values.items()):
            write(workspace / "private", "run/artifacts/" + name, raw)
            members.append({"role": role, "path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
        write(workspace / "private", "run/state/runtime-v1-inventory.json", _canonical({
            "schema_version": "synaptic-artifact-inventory/v1", "workload_fingerprint": prep.bundle.workload_fingerprint, "artifacts": members}))
        dispatched.append(True)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(source_worker.subprocess, "run", child)
    monkeypatch.setenv("HF_TOKEN", "fixture-job-secret-never-child")
    monkeypatch.setenv("JOB_ID", "fixture-job-1")
    run = TrainingRunRef(prep.rich_plan.execution_source.run_id, prep.resolved.project_ref)
    sealed = prep.prepare(prep.plan, run, prep.binding)
    command = build_stage_command(sealed, "fixture-nonce", prep.payload(sealed, EffectKind.STAGE), prep.executor)
    transport = HFJobsTransport(preparation=prep, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    program = transport.job_request(command)["command"][-1]
    # Parse only the admitted standard-base64 payload; bootstrap.run and the
    # actual worker execute below, not a mocked success return.
    import re
    encoded = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", program).group(1)
    import base64
    payload = base64.b64decode(encoded, validate=True)
    if attack in {"parent", "engine", "input", "inventory"}:
        targets = {"parent": mount_source / "parent/tuner/execution/contracts.py",
            "engine": mount_source / "engine/Trainers/sft/runtime_v1.py",
            "input": mount_output / input_name,
            "inventory": mount_source / "control/runtime-inventory.json"}
        leaf = targets[attack]
        raw = leaf.read_bytes()
        leaf.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    previous_path, previous_bytecode = list(__import__("sys").path), __import__("sys").dont_write_bytecode
    try:
        if attack is None:
            assert bootstrap.run(payload) == 0
        else:
            with pytest.raises(bootstrap.HFJobFailure) as caught:
                bootstrap.run(payload)
            expected = {"parent": 81, "engine": 83, "input": 85,
                        "inventory": 83, "runtime": 84, "model": 86,
                        "dispatch": 87, "artifact": 88, "model_subclass": 89, "model_mutated": 89}
            code = (attack if type(attack) is int and attack in {
                20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
            } else expected.get(attack, 87))
            assert caught.value.exit_code == code
            assert "private fixture text" not in str(caught.value)
            assert caught.value.__suppress_context__ is True
    finally:
        __import__("sys").path[:] = previous_path
        __import__("sys").dont_write_bytecode = previous_bytecode
    if attack in {"parent", "engine", "input", "inventory", "runtime", "model", "model_subclass", "model_mutated"}:
        assert dispatched == []
    if attack in {"parent", "engine", "input", "inventory", "runtime"}:
        assert models == []
    if attack is not None:
        assert not (mount_output / "completion.json").exists()
        return
    assert inspected == models == dispatched == [True]
    completion = json.loads(original_read(mount_output, "completion.json", (mount_output / "completion.json").stat().st_size))
    assert completion["job_id"] == "fixture-job-1"
    assert completion["material_digest"] == prep.material_digest
    for row in completion["artifacts"]:
        assert original_read(mount_output, row["path"], row["size_bytes"]) == values[row["role"]][1]


def test_delayed_sdk_worker_does_not_begin_expired_operation(monkeypatch):
    from tuner.execution.providers.hf_jobs_provider_v1 import transport
    operations, threads = [], []
    class DelayedThread:
        def __init__(self, *, target, **kwargs):
            self.target = target
            threads.append(self)
        def start(self):
            pass
    monkeypatch.setattr(transport, "Thread", DelayedThread)
    with pytest.raises(HFJobsError, match="deadline_exceeded"):
        transport._bounded(lambda: operations.append(True), time.monotonic() + 0.01)
    threads[0].target()
    assert operations == []


def test_spool_replaced_leaf_is_retained_not_deleted(material):
    provider = material.provider
    provider.blobs["fixture/content"] = b"original"
    item = BucketFile(type="file", path="fixture/content", size=8, xetHash="a" * 64)
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    with pytest.raises(HFJobsError, match="artifact_invalid"):
        with transport._spool(item, 8, "fixture-token", time.monotonic() + 10):
            root = next(material.scratch.iterdir())
            (root / "content").unlink()
            (root / "content").write_bytes(b"replacement")
    assert (root / "content").read_bytes() == b"replacement"


def test_fresh_isolated_parent_import_uses_only_reviewed_copy(material, tmp_path):
    import subprocess
    import sys
    root = tmp_path / "parent"
    root.mkdir(mode=0o700)
    from tuner.execution.providers.hf_jobs_provider_v1.bootstrap import _write
    for name, raw in material.preparation.source_files.items():
        if name.startswith("parent/"):
            _write(root, name.removeprefix("parent/"), raw)
    script = """
import pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
sys.dont_write_bytecode=True
sys.path.insert(0,str(root))
import tuner.execution.providers.hf_jobs_provider_v1.source_worker
import tuner.execution.providers.modal.model_snapshot
for name,module in list(sys.modules.items()):
    if name.split('.')[0] in {'tuner','synaptic_tuner'} and getattr(module,'__file__',None):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root),name
assert not any(name.split('.')[0] in {'modal','torch','huggingface_hub'} for name in sys.modules)
assert not list(root.rglob('*.pyc'))
print('HF_PARENT_IMPORT_PASS')
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(root)], cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"HF_PARENT_IMPORT_PASS\n"


@pytest.mark.parametrize("policy", [ArtifactPolicy(("other_role",), False),
    ArtifactPolicy(("final_model",), False),
    ArtifactPolicy(tuple(sorted({"final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record"})), True)])
def test_unsupported_artifact_policy_rejects_before_input_or_provider(material, policy):
    prep = material.preparation
    with pytest.raises(HFJobsError, match="capability_unavailable"):
        HFJobsPreparation(selection=prep.selection, training_plan=replace(prep.rich_plan, artifact_policy=policy),
            bundle=prep.bundle, closure_members={}, parent_members={}, input_lease=prep.input_lease,
            request_id="fixture-request", project_ref="fixture-project", clock=Clock(), maximum_cost_minor_units=100, currency="USD")
    assert prep.input_lease._state == "available"
    assert material.provider.calls == []


@pytest.mark.parametrize("attack", ["extra_field", "member_extra", "duplicate_role", "wrong_job", "nonfinite", "duplicate_key"])
def test_closed_native_completion_rejects_before_evidence_or_stream(material, attack):
    api, composition, ports = _host(material)
    plan = api.training.plan(api.training.resolve(api.training.load('{}')), material.preparation.selection.provider)
    api.training.start(plan, api.training.preflight(plan))
    _complete(material, _native_values(material.preparation))
    key = material.preparation.selection.run_prefix + "/completion.json"
    document = json.loads(material.provider.blobs[key])
    if attack == "extra_field":
        document["private_payload"] = "PRIVATE_SENTINEL"
    elif attack == "member_extra":
        document["artifacts"][0]["private_payload"] = "PRIVATE_SENTINEL"
    elif attack == "duplicate_role":
        document["artifacts"][1]["role"] = document["artifacts"][0]["role"]
    elif attack == "wrong_job":
        document["job_id"] = "different-job"
    elif attack == "nonfinite":
        document["artifacts"][0]["size_bytes"] = float("nan")
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    if attack == "duplicate_key":
        raw = raw.replace(b'"status":"completed"', b'"status":"completed","status":"completed"')
    material.provider.blobs[key] = raw
    reader = composition.coordinator._planning
    # Read the actual retained command; no forged read request or auth bypass.
    candidates = ports.records.list_page(partition="effects", prefix="hf-jobs/", after_key=None, limit=100).records
    command_digest = next(json.loads(row.canonical)["command_digest"] for row in candidates if "/submit/" in row.key)
    from tuner.execution.providers.hf_jobs_provider_v1.transport import HFJobsTransport
    transport = HFJobsTransport(preparation=reader, sdk=sdk, api=material.provider,
        records=ports.records, secrets=Secrets(), scratch_root=material.scratch)
    with pytest.raises(HFJobsError, match="artifact_invalid") as caught:
        transport.native(command_digest)
    assert "PRIVATE_SENTINEL" not in str(caught.value)


def test_unsafe_spool_directory_is_rejected_and_retained(material):
    provider = material.provider
    provider.blobs["fixture/content"] = b"original"
    original = provider.download_bucket_files
    def unsafe(bucket_id, files, **options):
        original(bucket_id, files, **options)
        files[0][1].parent.chmod(0o777)
    provider.download_bucket_files = unsafe
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    item = BucketFile(type="file", path="fixture/content", size=8, xetHash="a" * 64)
    with pytest.raises(HFJobsError, match="artifact_invalid"):
        with transport._spool(item, 8, "fixture-token", time.monotonic() + 10):
            pytest.fail("unsafe spool admitted")
    roots = list(material.scratch.iterdir())
    assert len(roots) == 1 and (roots[0] / "content").read_bytes() == b"original"


def test_late_failed_download_retains_private_partial_scratch(material):
    import threading
    release, finished = threading.Event(), threading.Event()
    provider = material.provider
    def late(bucket_id, files, **options):
        try:
            release.wait(2)
            files[0][1].write_bytes(b"partial")
        finally:
            finished.set()
    provider.download_bucket_files = late
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    item = BucketFile(type="file", path="fixture/content", size=8, xetHash="a" * 64)
    try:
        with pytest.raises(HFJobsError, match="deadline_exceeded"):
            with transport._spool(item, 8, "fixture-token", time.monotonic() + 0.02):
                pytest.fail("unfinished download admitted")
    finally:
        release.set()
        assert finished.wait(3)
    roots = list(material.scratch.iterdir())
    assert len(roots) == 1 and (roots[0] / "content").read_bytes() == b"partial"



def test_rehashed_invalid_native_archive_is_not_semantic_success(material):
    api, _, _ = _host(material)
    request = api.training.load('{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    run = api.training.start(plan, api.training.preflight(plan)).run
    values = _native_values(material.preparation)
    values["final_model"] = ("final_model.tar", b"not a native adapter archive")
    _complete(material, values)
    api.runs.outcome(run)
    with pytest.raises(Exception):
        api.runs.verify(run)


def test_multi_mib_artifact_is_split_and_hash_preserved(material):
    api, _, _ = _host(material)
    request = api.training.load('{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    run = api.training.start(plan, api.training.preflight(plan)).run
    values = _native_values(material.preparation)
    raw = values["final_model"][1] + b"\0" * (3 * 1024 * 1024)
    values["final_model"] = ("final_model.tar", raw)
    _complete(material, values)
    api.runs.outcome(run)
    api.runs.verify(run)
    from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest
    chunks = list(api.runs.artifacts(RunArtifactRequest(run, "final_model", len(raw))).iter_bytes())
    assert len(chunks) > 1 and all(len(chunk) <= 1024 * 1024 for chunk in chunks)
    assert hashlib.sha256(b"".join(chunks)).digest() == hashlib.sha256(raw).digest()


def test_declared_flavor_resources_must_equal_rich_plan(material):
    prep = material.preparation
    with pytest.raises(HFJobsError, match="binding_mismatch"):
        HFJobsPreparation(selection=replace(prep.selection, resources=ResourceSpec("cpu", 1, 300)),
            training_plan=prep.rich_plan, bundle=prep.bundle, closure_members={}, parent_members={},
            input_lease=prep.input_lease, request_id="fixture-request", project_ref="fixture-project",
            clock=Clock(), maximum_cost_minor_units=100, currency="USD")
    assert prep.input_lease._state == "available" and material.provider.calls == []


def test_transport_uses_only_preparation_captured_bootstrap(material, monkeypatch):
    import importlib.resources
    monkeypatch.setattr(importlib.resources, "files", lambda *args: pytest.fail("post-admission package reread"))
    api, composition, _ = _host(material)
    request = api.training.load('{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    api.training.start(plan, api.training.preflight(plan))
    assert len([call for call in material.provider.calls if call[0] == "submit"]) == 1


def test_staged_readback_batches_all_members_with_exact_hashes(material):
    provider = material.provider
    members = []
    count = len(material.preparation.source_files) + 1
    for index in range(count):
        raw = (b"payload" * index) if index else b""
        path = f"fixture/member-{index}"
        provider.blobs[path] = raw
        members.append((BucketFile(type="file", path=path, size=len(raw),
                                  xetHash=hashlib.sha256(raw).hexdigest()), raw))
    calls = []
    original = provider.download_bucket_files
    def download(bucket, files, **kwargs):
        calls.append(len(files))
        assert len({str(leaf.parent) for _, leaf in files}) == count
        assert all(stat.S_IMODE(leaf.parent.stat().st_mode) == 0o700 for _, leaf in files)
        original(bucket, files, **kwargs)
    provider.download_bucket_files = download
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    transport._verify_staged_batch(members, "fixture-token", time.monotonic() + 10)
    assert calls == [count]
    assert not list(material.scratch.glob("hf-read-*"))


@pytest.mark.parametrize("changed", ["bytes", "symlink", "extra"])
def test_staged_batch_rejects_invalid_member_without_receipt(material, changed):
    provider = material.provider
    raw = b"original"
    item = BucketFile(type="file", path="fixture/content", size=len(raw), xetHash="a" * 64)
    provider.blobs[item.path] = raw
    original = provider.download_bucket_files
    def download(bucket, files, **kwargs):
        original(bucket, files, **kwargs)
        leaf = files[0][1]
        if changed == "bytes":
            leaf.write_bytes(b"replaced")
        elif changed == "symlink":
            leaf.unlink()
            leaf.symlink_to(material.scratch / "untrusted")
        else:
            (leaf.parent / "extra").write_bytes(b"extra")
    provider.download_bucket_files = download
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    with pytest.raises((HFJobsError, OSError)):
        transport._verify_staged_batch([(item, raw)], "fixture-token", time.monotonic() + 10)


@pytest.mark.parametrize("failure", ["sdk", "binding", "deadline"])
def test_stage_failure_retains_closed_bound_diagnostic_and_never_submits(material, failure, monkeypatch):
    from synaptic_tuner.api.v1.training_facade import TrainingRequest
    from tuner.execution.providers.hf_jobs_provider_v1 import transport as module

    original = material.provider.download_bucket_files
    def download(bucket, files, **kwargs):
        if failure == "sdk":
            raise RuntimeError("credential-secret-must-not-appear")
        original(bucket, files, **kwargs)
        files[0][1].write_bytes(b"replaced")
    material.provider.download_bucket_files = download
    if failure == "deadline":
        original_bounded = module._bounded
        def bounded(operation, deadline):
            if getattr(operation, "__name__", "") == "<lambda>" and len(material.provider.blobs):
                raise HFJobsError("deadline_exceeded")
            return original_bounded(operation, deadline)
        monkeypatch.setattr(module, "_bounded", bounded)
    api, composition, ports = _host(material)
    request = TrainingRequest("fixture-request", "fixture-project", '{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    preflight = api.training.preflight(plan)
    api.training.start(plan, preflight)
    rows = ports.records.list_page(partition="effects", prefix="", after_key=None, limit=100).records
    diagnostics = [json.loads(row.canonical) for row in rows if "/stage-diagnostic/" in row.key]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["phase"] in {"upload", "readback_bytes"}
    assert diagnostic["failure_class"] in {"sdk", "binding", "local", "deadline"}
    assert diagnostic["material_digest"] == material.preparation.material_digest
    assert "credential-secret" not in json.dumps(diagnostic)
    assert not [call for call in material.provider.calls if call[0] == "submit"]


def test_cold_local_stage_diagnostic_authenticates_without_provider_or_source(material):
    from synaptic_tuner.api.v1.training_facade import TrainingRequest
    from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import read_local_stage_diagnostic

    def failed(*args, **kwargs):
        raise RuntimeError("private-provider-text")
    material.provider.batch_bucket_files = failed
    api, _, ports = _host(material)
    request = TrainingRequest("fixture-request", "fixture-project", '{}')
    plan = api.training.plan(api.training.resolve(request), material.preparation.selection.provider)
    preflight = api.training.preflight(plan)
    started = api.training.start(plan, preflight)
    before = list(material.provider.calls)
    authority = compose_reference_authority(clock=ports.clock, secrets=ports.secrets,
        authority_secret=SecretRef("env", "AUTHORITY"))
    outcome = read_local_stage_diagnostic(records=ports.records, streams=ports.streams,
        run=started.run, authority=authority)
    assert outcome["local_workflow_phase"] == "stage_reconcile_required"
    assert outcome["local_stage_diagnostic"]["phase"] == "upload"
    assert outcome["local_stage_diagnostic"]["failure_class"] == "sdk"
    assert outcome["provider_evidence"] is outcome["replay_authorized"] is False
    assert material.provider.calls == before
    class LegacyRecords:
        def read(self, **kwargs):
            return None if "/stage-diagnostic/" in kwargs["key"] else ports.records.read(**kwargs)
        def __getattr__(self, name):
            return getattr(ports.records, name)
    legacy = read_local_stage_diagnostic(records=LegacyRecords(), streams=ports.streams,
        run=started.run, authority=authority)
    assert legacy["local_stage_diagnostic"] is None
    assert legacy["replay_authorized"] is legacy["provider_evidence"] is False
    row = next(row for row in ports.records.list_page(
        partition="effects", prefix="", after_key=None, limit=100).records
        if "/stage-diagnostic/" in row.key)
    original = row.canonical
    for invalid in (b" " + original, original[:-1] + b',"phase":"upload"}'):
        assert ports.records.compare_and_swap(partition="effects", key=row.key,
            expected_revision=row.revision, canonical=invalid)
        with pytest.raises(HFJobsError, match="binding_mismatch"):
            read_local_stage_diagnostic(records=ports.records, streams=ports.streams,
                run=started.run, authority=authority)
        row = ports.records.read(partition="effects", key=row.key)
    assert material.provider.calls == before


def test_batch_budget_rejects_before_allocation_or_sdk(material, monkeypatch):
    from tuner.execution.providers.hf_jobs_provider_v1 import transport as module
    monkeypatch.setattr(module, "_MAX_DOWNLOAD_BYTES", 1)
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    item = BucketFile(type="file", path="fixture/content", size=2, xetHash="a" * 64)
    with pytest.raises(HFJobsError, match="input_invalid"):
        transport._verify_staged_batch([(item, b"ab")], "fixture-token", time.monotonic() + 10)
    assert not list(material.scratch.glob("hf-read-*"))
    assert material.provider.calls == []


def test_batch_cleanup_attempts_all_fds_preserving_deadline(material, monkeypatch):
    from tuner.execution.providers.hf_jobs_provider_v1 import transport as module
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    members = [(BucketFile(type="file", path=f"fixture/{i}", size=1, xetHash="a" * 64), b"x")
               for i in range(3)]
    monkeypatch.setattr(module, "_bounded", lambda *_: (_ for _ in ()).throw(HFJobsError("deadline_exceeded")))
    closes = []
    original = module.os.close
    def close(fd):
        original(fd)
        closes.append(fd)
        if len(closes) == 1:
            raise OSError("private-close-text")
    monkeypatch.setattr(module.os, "close", close)
    with pytest.raises(HFJobsError, match="deadline_exceeded"):
        transport._verify_staged_batch(members, "fixture-token", time.monotonic() + 10)
    assert len(closes) == 3
    assert len(list(material.scratch.glob("hf-read-*"))) == 3


def test_verified_spool_attempts_parent_close_after_leaf_close_fault(material, monkeypatch):
    from tuner.execution.providers.hf_jobs_provider_v1 import transport as module
    material.provider.blobs["fixture/content"] = b"x"
    item = BucketFile(type="file", path="fixture/content", size=1, xetHash="a" * 64)
    transport = HFJobsTransport(preparation=material.preparation, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    original = module.os.close
    closes = []
    def close(fd):
        original(fd)
        closes.append(fd)
        if len(closes) == 1:
            raise OSError("private-close-error")
    with pytest.raises(HFJobsError, match="artifact_invalid"):
        with transport._spool(item, 1, "fixture-token", time.monotonic() + 10):
            monkeypatch.setattr(module.os, "close", close)
    assert len(closes) == 2


def test_normal_hf_diagnostic_vocabulary_is_independent_and_native_exact():
    from tuner.execution.providers.hf_jobs_provider_v1.bootstrap import (
        HFJobFailure, HF_JOB_PHASE_EXITS, HF_NATIVE_STAGE_EXITS,
    )
    from Trainers.sft.runtime_v1 import _RUNTIME_STAGE_EXITS
    assert HF_JOB_PHASE_EXITS == {
        "BOOTSTRAP_ADMISSION": 80, "PARENT_MATERIALIZATION": 81,
        "WORKER_INPUT": 82, "SOURCE_MATERIALIZATION": 83,
        "RUNTIME_INVENTORY": 84, "PREPARED_INPUT": 85,
        "MODEL_PREPARATION": 86, "DISPATCH": 87,
        "ARTIFACT_PUBLICATION": 88, "UNCLASSIFIED": 89,
    }
    native = {20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}
    assert HF_NATIVE_STAGE_EXITS == native == set(_RUNTIME_STAGE_EXITS.values())
    for invalid in (True, False, -9, 0, 1, 26, 120, "22", 22.0):
        with pytest.raises(ValueError):
            HFJobFailure("DISPATCH", invalid)
    for invalid in (True, None, "private exception text", "DISPATCH_suffix"):
        with pytest.raises(ValueError):
            HFJobFailure(invalid)
    with pytest.raises(ValueError):
        HFJobFailure("MODEL_PREPARATION", 22)


@pytest.mark.parametrize("phase,native,expected", [
    ("BOOTSTRAP_ADMISSION", None, 80), ("DISPATCH", 22, 22),
    ("ARTIFACT_PUBLICATION", None, 88), (None, None, 89),
    ("subclass", None, 89), ("mutated_phase", None, 89),
    ("mutated_code", None, 89), ("bool_code", None, 89),
])
def test_actual_generated_wrapper_has_closed_failure_output(material, capsys, phase, native, expected):
    import ast
    from tuner.execution.providers.hf_jobs_provider_v1 import bootstrap
    prep = material.preparation
    run = TrainingRunRef(prep.rich_plan.execution_source.run_id, prep.resolved.project_ref)
    sealed = prep.prepare(prep.plan, run, prep.binding)
    command = build_stage_command(sealed, "fixture-nonce", prep.payload(sealed, EffectKind.STAGE), prep.executor)
    transport = HFJobsTransport(preparation=prep, sdk=sdk, api=material.provider,
        records=InMemoryDurableRecordStoreV1(), secrets=Secrets(), scratch_root=material.scratch)
    program = transport.job_request(command)["command"][-1]
    ast.parse(program)
    suffix = "try:\n run" + program.rsplit("\ntry:\n run", 1)[1]
    def fail(_payload):
        if phase is None:
            raise RuntimeError("private fixture text must never escape")
        if phase == "subclass":
            class Hostile(bootstrap.HFJobFailure):
                pass
            raise Hostile("DISPATCH", 22)
        if phase in {"mutated_phase", "mutated_code", "bool_code"}:
            error = bootstrap.HFJobFailure("DISPATCH", 22)
            if phase == "mutated_phase":
                error.phase = "private fixture text must never escape"
            else:
                error.exit_code = True if phase == "bool_code" else 120
            raise error
        raise bootstrap.HFJobFailure(phase, native)
    namespace = {"run": fail, "HFJobFailure": bootstrap.HFJobFailure,
                 "sys": __import__("sys"), "json": json,
                 "failure_projection": bootstrap.failure_projection}
    with pytest.raises(SystemExit) as caught:
        exec(compile(suffix, "<admitted-hf-wrapper>", "exec"), namespace)
    assert caught.value.code == expected
    output = capsys.readouterr()
    assert output.err == "" and "private fixture text" not in output.out
    assert json.loads(output.out) == {"schema_version": "synaptic-hf-job-failure/v1",
                                     "phase": "UNCLASSIFIED" if expected == 89 else phase,
                                     "exit_code": expected}
    assert len(output.out.encode()) <= 160


def test_bootstrap_and_worker_input_failures_are_closed():
    from tuner.execution.providers.hf_jobs_provider_v1 import bootstrap, source_worker
    with pytest.raises(bootstrap.HFJobFailure) as caught:
        bootstrap.run(b"not JSON private fixture text")
    assert bootstrap.failure_projection(caught.value) == ("BOOTSTRAP_ADMISSION", 80)
    with pytest.raises(bootstrap.HFJobFailure) as caught:
        source_worker.execute({"private": "fixture text"})
    assert bootstrap.failure_projection(caught.value) == ("WORKER_INPUT", 82)


@pytest.mark.parametrize("entry", ["direct", "exec-argv", "explicit-shell-string"])
def test_actual_job_request_executes_raw_program_without_quote_layer(entry):
    """Real isolated Python, no mounts/SDK calls: malformed input refuses before I/O.

    The middle case models the captured image entrypoint's exact exec "$@"
    semantics. The final case intentionally constructs a shell command with
    shlex.join; it is not a claim that HF applies that transformation.
    """
    import ast
    import shlex
    import subprocess
    import sys
    from tuner.execution.providers.hf_jobs_provider_v1 import bootstrap, transport
    material = {"selection": {"inventory_sha256": "a" * 64, "timeout_seconds": 1,
        "bucket_id": "fixture-owner/bucket", "source_prefix": "source/fixture",
        "run_prefix": "runs/fixture", "image": "fixture@sha256:" + "b" * 64,
        "flavor": "cpu-basic", "scope": {"namespace_ref": "fixture-owner"}},
        "members": [], "input": {}}
    workload = {"configuration": {"document": {"model": {}}},
                "execution_source": {"runtime": {"interpreter": {"executable": sys.executable}}}}
    request = transport._job_request(sdk=sdk, material=material,
        projection={"workload_fingerprint": "c" * 64, "dispatch": {}},
        workload=workload, bootstrap=Path(bootstrap.__file__).read_bytes(),
        command=SimpleNamespace(digest="d" * 64))
    argv = request["command"]
    ast.parse(argv[-1])
    original = list(argv)
    if entry == "exec-argv":
        argv = ["/bin/sh", "-c", 'exec "$@"', "fixture-entrypoint", *argv]
    elif entry == "explicit-shell-string":
        argv = ["/bin/sh", "-c", shlex.join(argv)]
    result = subprocess.run(argv, env={}, capture_output=True, timeout=5, check=False)
    assert result.returncode == 80
    assert result.stderr == b""
    assert json.loads(result.stdout) == {"schema_version": "synaptic-hf-job-failure/v1",
        "phase": "BOOTSTRAP_ADMISSION", "exit_code": 80}
    # The old extra shell quote is not valid Python when argv is passed intact.
    bad = [*original[:-1], shlex.quote(original[-1])]
    with pytest.raises(SyntaxError):
        ast.parse(bad[-1])
    rejected = subprocess.run(bad, env={}, capture_output=True, timeout=5, check=False)
    assert rejected.returncode == 1 and rejected.stdout == b""
