from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from tuner.cloud.derived_training_image import DerivedTrainingImageError
from tuner.cloud.grpo_constructor_diagnostic import derive_probe_input, verify_probe_input
from tuner.runtime import packaged_grpo_constructor_child as child
from tuner.training import modal_grpo_constructor_diagnostic as host


def _planned_input(tmp_path: Path):
    import yaml
    from synaptic_tuner.api.v1.training_input import TrainingInputV1
    from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
    from tuner.training.grpo_compilation import compile_grpo_input, grpo_prepared_semantic_digest
    root = Path(__file__).resolve().parents[2]
    recipe = root / "examples/grpo_captured_candidate/recipe.yaml"
    dataset = root / "examples/grpo_captured_candidate/rows.jsonl"
    data = yaml.safe_load(recipe.read_bytes())
    raw = dataset.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    semantic = grpo_prepared_semantic_digest(content_digest=digest, size_bytes=len(raw),
        row_count=4, prompt_column="prompt")
    prepared = PreparedTrainingInputIdentity(ref="prepared://sha256/" + semantic,
        revision=semantic, content_digest=digest, size_bytes=len(raw), format="synaptic-grpo-prompts/v1")
    training = TrainingInputV1.from_dict(data["training_input"])
    compiled = compile_grpo_input(training, prepared)
    from tuner.runtime_profiles import load_runtime_profile
    profile = load_runtime_profile(data["job"]["runtime_profile"], root / "Trainers/runtime_profiles")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"data": {"training_input": training.to_dict(),
        "prepared_dataset": prepared.to_dict(), "core_workload_digest": compiled.fingerprint,
        "runtime_profile": data["job"]["runtime_profile"],
        "profile_sha256": profile.profile_sha256,
        "resource_request": {"accelerator": data["job"]["accelerator"],
            "accelerator_count": data["job"]["accelerator_count"],
            "timeout_seconds": data["job"]["timeout_seconds"]},
        "limits": data["limits"]}}))
    return recipe, dataset, plan, compiled


def test_exact_four_row_input_and_dataset_mutation(tmp_path):
    recipe, dataset, plan, compiled = _planned_input(tmp_path)
    raw = derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset)
    value = verify_probe_input(raw)
    assert value["row_count"] == 4
    assert value["workload"] == compiled.document
    assert len(raw) <= 65536
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(dataset.read_bytes().replace(b"blue", b"green", 1))
    with pytest.raises(DerivedTrainingImageError):
        derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=changed)
    with pytest.raises(DerivedTrainingImageError):
        verify_probe_input(raw.replace(b"\n", b" \n", 1))


def test_plan_profile_resource_and_recipe_are_bound_before_dispatch(tmp_path):
    recipe, dataset, plan, _ = _planned_input(tmp_path)
    profiles_root = Path(__file__).resolve().parents[2] / "Trainers/runtime_profiles"
    original = derive_probe_input(recipe_path=recipe, plan_ready_path=plan,
                                  dataset_path=dataset, profiles_root=profiles_root)
    document = json.loads(plan.read_bytes())
    for field, value in (("profile_sha256", "sha256:" + "0" * 64),
                         ("resource_request", {"accelerator": "A100", "accelerator_count": 1,
                                               "timeout_seconds": 900})):
        mutated = json.loads(plan.read_bytes())
        mutated["data"][field] = value
        plan.write_text(json.dumps(mutated))
        with pytest.raises(DerivedTrainingImageError):
            derive_probe_input(recipe_path=recipe, plan_ready_path=plan,
                               dataset_path=dataset, profiles_root=profiles_root)
        plan.write_text(json.dumps(document))
    assert derive_probe_input(recipe_path=recipe, plan_ready_path=plan,
                              dataset_path=dataset, profiles_root=profiles_root) == original
    changed_recipe = tmp_path / "changed-recipe.yaml"
    changed_recipe.write_bytes(recipe.read_bytes().replace(b"fp16: true", b"fp16: false", 1))
    with pytest.raises(DerivedTrainingImageError):
        derive_probe_input(recipe_path=changed_recipe, plan_ready_path=plan,
                           dataset_path=dataset, profiles_root=profiles_root)


def test_model_inventory_rejects_metadata_and_missing_member_mutations():
    data = b"model"
    model = {"ref": "Example/Model", "revision": "a" * 40,
        "tokenizer_revision": "a" * 40, "members": [{"path": "config.json",
        "size_bytes": len(data), "kind": "git",
        "upstream_digest": hashlib.sha1(b"blob 5\0" + data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest()}]}
    assert child.validate_model_inventory(model) == (("config.json", 5, hashlib.sha256(data).hexdigest()),)
    for key, value in (("path", "../config.json"), ("size_bytes", 6),
                       ("upstream_digest", "0" * 40), ("sha256", "0" * 64)):
        changed = {**model, "members": [{**model["members"][0], key: value}]}
        if key == "upstream_digest":
            assert changed["members"][0]["upstream_digest"] != model["members"][0]["upstream_digest"]
        elif key in {"size_bytes", "sha256"}:
            assert child.validate_model_inventory(changed) != child.validate_model_inventory(model)
        else:
            with pytest.raises(ValueError):
                child.validate_model_inventory(changed)
    with pytest.raises(ValueError):
        child.validate_model_inventory({**model, "members": model["members"] * 2})


def test_host_and_engine_source_require_clean_pushed_exact_gitlink(monkeypatch, tmp_path):
    engine = tmp_path / "engine"
    parent = tmp_path / "host"
    engine.mkdir()
    parent.mkdir()
    commits = {engine: "a" * 40, parent: "b" * 40}
    monkeypatch.setattr(host, "inspect_git_source", lambda repo: SimpleNamespace(
        dirty=False, pushed=True, commit=commits[repo]))
    monkeypatch.setattr(host, "_git", lambda *args: "160000 commit " + "a" * 40 + "\tsynaptic-tuner")
    assert host._source_pair(engine, parent) == ("a" * 40, "b" * 40)
    commits[parent] = "c" * 40
    monkeypatch.setattr(host, "_git", lambda *args: "160000 commit " + "d" * 40 + "\tsynaptic-tuner")
    with pytest.raises(ValueError):
        host._source_pair(engine, parent)
    monkeypatch.setattr(host, "inspect_git_source", lambda repo: SimpleNamespace(
        dirty=repo == engine, pushed=True, commit=commits[repo]))
    with pytest.raises(ValueError):
        host._source_pair(engine, parent)


def test_dispatch_binds_input_source_image_and_model(monkeypatch, tmp_path):
    recipe, dataset, plan, _ = _planned_input(tmp_path)
    raw = derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset)
    monkeypatch.setattr(child, "parse_packaged_runtime_release", lambda value: value)
    model = {"ref": "Example/Model", "revision": "a" * 40,
        "tokenizer_revision": "a" * 40, "members": [{"path": "config.json",
        "size_bytes": 1, "kind": "git", "upstream_digest": "b" * 40, "sha256": "c" * 64}]}
    value = {"schema_version": child.DISPATCH_SCHEMA, "attempt_ref": "grpo-diag-" + "d" * 24,
        "source_commit": "e" * 40, "host_source_commit": "a" * 40,
        "engine_gitlink": "e" * 40, "release": {}, "deployment_facts_digest": "f" * 64,
        "image_id": "im-FIXTURE", "input_base64": __import__("base64").b64encode(raw).decode(),
        "input_sha256": hashlib.sha256(raw).hexdigest(), "model": model,
        "library_sources": child.pinned_library_sources()}
    assert child.validate_dispatch(child.canonical(value))[1] == raw
    for field, mutated in (("source_commit", "0" * 40), ("image_id", "bad"),
                           ("input_sha256", "0" * 64), ("library_sources", {})):
        changed = dict(value, **{field: mutated})
        if field == "source_commit":
            continue  # A different valid commit needs the host's pushed-source check.
        with pytest.raises(ValueError):
            child.validate_dispatch(child.canonical(changed))
    changed = dict(value, library_sources={**value["library_sources"],
        "unsloth/models/rl.py": "0" * 64})
    with pytest.raises(ValueError):
        child.validate_dispatch(child.canonical(changed))


def test_actual_core_constructor_only_has_no_train_or_generation_path():
    from Trainers.grpo.runtime_v1 import train_grpo_core
    calls = []
    class Trainer:
        def __init__(self, **kwargs):
            calls.append(("construct", tuple(kwargs)))
        def train(self, *args, **kwargs):
            raise AssertionError("train called")
        def generate(self, *args, **kwargs):
            raise AssertionError("generate called")
        def optimizer_step(self, *args, **kwargs):
            raise AssertionError("optimizer called")
    observer = lambda cls, kwargs: calls.append(("observe", tuple(kwargs)))
    trainer, result = train_grpo_core(model=object(), tokenizer=object(), reward_fn=lambda: None,
        formatted_dataset=object(), training_args=object(), callbacks=[], trainer_class=Trainer,
        _constructor_only=True, _constructor_observer=observer)
    assert isinstance(trainer, Trainer) and result is None
    assert calls == [("observe", ("model", "processing_class", "reward_funcs", "args", "train_dataset", "callbacks")),
                     ("construct", ("model", "processing_class", "reward_funcs", "args", "train_dataset", "callbacks"))]
    with pytest.raises(ValueError):
        train_grpo_core(model=None, tokenizer=None, reward_fn=None, formatted_dataset=None,
            training_args=None, callbacks=[], trainer_class=Trainer, _constructor_only=True)


def test_actual_run_grpo_constructor_only_preserves_reward_callbacks_and_stops(tmp_path, monkeypatch):
    from Trainers.grpo import runtime_v1 as runtime
    from Trainers.grpo.src import rewards, data_loader
    recipe, dataset, plan, _ = _planned_input(tmp_path)
    probe = verify_probe_input(derive_probe_input(recipe_path=recipe, plan_ready_path=plan,
                                                  dataset_path=dataset))
    workload = probe["workload"]
    calls = []
    class Callback:
        pass
    class Trainer:
        def __init__(self, **kwargs):
            calls.append(("constructor", kwargs))
        def train(self, *args, **kwargs):
            raise AssertionError("training path reached")
        def generate(self, *args, **kwargs):
            raise AssertionError("generation path reached")
        def save_model(self, *args, **kwargs):
            raise AssertionError("saving path reached")
    class Loader:
        @staticmethod
        def load_model_and_tokenizer(*args, **kwargs):
            calls.append(("model", kwargs))
            return object(), object(), False
        @staticmethod
        def apply_lora_adapters(**kwargs):
            calls.append(("lora", kwargs))
            return object()
    monkeypatch.setattr(runtime, "build_grpo_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "_QualificationMeasurements", lambda *args, **kwargs:
        SimpleNamespace(reward_wrapper=lambda reward: reward, begin=lambda *a: None,
                        end=lambda *a: None, log=lambda *a: None))
    monkeypatch.setattr(rewards, "build_combined_reward_function",
                        lambda *args, **kwargs: (lambda *a: 1.0, {}))
    monkeypatch.setattr(data_loader, "dataset_from_rows", lambda rows, **kwargs: rows)
    monkeypatch.setattr(data_loader, "format_dataset_for_grpo", lambda rows, *args, **kwargs: rows)
    model_root = tmp_path / "model"
    output_root = tmp_path / "output"
    model_root.mkdir()
    output_root.mkdir()
    trainer = runtime.run_grpo(config=workload["config"], dataset_bytes=dataset.read_bytes(),
        reward_documents=workload["reward_documents"], reward_schema=workload["reward_schema"],
        model_snapshot=model_root, model_identity=workload["model"], output_root=output_root,
        workload=workload, metric_sink=lambda record: None,
        _dependency_factory=lambda: {"trainer_class": Trainer, "config_class": object,
            "bf16_supported": False, "model_loader": Loader, "dataset_class": object,
            "callback_class": Callback},
        _packaged_artifacts=True, qualification_evidence=probe["qualification_policy"],
        _child_deadline=time.monotonic() + 100,
        _constructor_only=True, _constructor_observer=lambda cls, kwargs:
            calls.append(("observer", tuple(kwargs))))
    assert isinstance(trainer, Trainer)
    assert [name for name, _ in calls] == ["model", "lora", "observer", "constructor"]
    kwargs = calls[-1][1]
    assert len(kwargs["callbacks"]) == 2
    assert all(isinstance(value, Callback) for value in kwargs["callbacks"])
    assert len(kwargs["train_dataset"]) == 4
    assert not list(output_root.iterdir())


def test_fixed_region_projection_uses_class_code_and_verified_sources():
    class Base:
        def __init__(self, *args, **kwargs):
            pass
    class Patched(Base):
        def __init__(self, *args, **kwargs):
            raise TypeError("never emitted")
    class Wrapper(Patched):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    try:
        Wrapper()
    except TypeError as error:
        assert child._site(error, Wrapper, True) == "OTHER"
        assert child._site(error, Wrapper, False) == "OTHER"
    assert set(child._empty_result()) == set(child.validate_result(child.canonical(child._empty_result())))


def test_isolated_entry_decodes_the_actual_four_row_dataset(tmp_path, monkeypatch):
    from Trainers.grpo import runtime_v1
    recipe, dataset_path, plan, _ = _planned_input(tmp_path)
    probe_raw = derive_probe_input(recipe_path=recipe, plan_ready_path=plan,
                                   dataset_path=dataset_path)
    probe = verify_probe_input(probe_raw)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("state", "tmp", "tracking", "artifacts", "prepared_input", "cache"):
        (tmp_path / name).mkdir()
    paths = {name: str(tmp_path / name) for name in
             ("prepared_input", "artifacts", "state", "tracking", "cache", "tmp")}
    release = SimpleNamespace()
    dispatch = {"release": {}, "model": {"ref": probe["workload"]["model"]["model_id"],
                "revision": probe["workload"]["model"]["revision"],
                "tokenizer_revision": probe["workload"]["model"]["tokenizer_revision"]},
                "library_sources": {}}
    environment = tuple(sorted(os.environ.items()))
    transport = child.canonical({"dispatch": dispatch, "private_manifest": {},
        "paths": paths, "environment": [list(item) for item in environment],
        "deadline": time.monotonic() + 30})
    source = tmp_path / "transport.json"
    source.write_bytes(transport)
    output = tmp_path / "result.json"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    monkeypatch.setattr(child, "validate_dispatch", lambda raw: (dispatch, probe_raw))
    monkeypatch.setattr(child, "parse_packaged_runtime_release", lambda raw: release)
    monkeypatch.setattr(child.shared, "_inspect_release", lambda *a, **k: object())
    monkeypatch.setattr(child, "_installed_import_guard", lambda *a, **k: object())
    monkeypatch.setattr(child, "grpo_child_environment", lambda *a: environment)
    monkeypatch.setattr(child.shared, "_retain_private_snapshot", lambda *a, **k:
                        SimpleNamespace(snapshot=snapshot, check=lambda: None, close=lambda: None))
    monkeypatch.setattr(child, "_source_verified", lambda sources: True)
    monkeypatch.setattr(child, "_observation", lambda cls, kwargs, verified: {})
    monkeypatch.setattr(child, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=1),
                                                     meta_path=sys.meta_path))
    monkeypatch.chdir(tmp_path / "tmp")
    calls = []
    def run_grpo(**kwargs):
        from Trainers.grpo.runtime_v1 import admit_grpo_inputs
        _, rows, _, _ = admit_grpo_inputs(config=kwargs["config"],
            dataset_bytes=kwargs["dataset_bytes"],
            reward_documents=kwargs["reward_documents"],
            reward_schema=kwargs["reward_schema"],
            model_identity=kwargs["model_identity"], workload=kwargs["workload"])
        calls.append(len(rows))
        kwargs["_runtime_milestone"]("TRAINER_SETUP")
        kwargs["_constructor_observer"](object, {})
        return object()
    monkeypatch.setattr(runtime_v1, "run_grpo", run_grpo)
    try:
        child._isolated(source, hashlib.sha256(transport).hexdigest(), fd)
    finally:
        os.close(fd)
    assert calls == [4]
    assert child.validate_result(output.read_bytes())["status"] == "CONSTRUCTED"


def test_interrupt_reaps_actual_owned_child(tmp_path, monkeypatch):
    original = child.subprocess.Popen
    spawned = []
    class Interrupted:
        def __init__(self, real):
            self.real = real
            self.pid = real.pid
            self.returncode = None
            self.once = True
        def wait(self, timeout=None):
            if self.once:
                self.once = False
                raise KeyboardInterrupt
            self.returncode = self.real.wait(timeout=timeout)
            return self.returncode
    def launch(*args, **kwargs):
        real = original(*args, **kwargs)
        spawned.append(real)
        return Interrupted(real)
    monkeypatch.setattr(child.subprocess, "Popen", launch)
    answer = tmp_path / "answer"
    fd = os.open(answer, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with pytest.raises(KeyboardInterrupt):
            child._wait_isolated_child((sys.executable, "-c", "import time; time.sleep(60)"),
                cwd=tmp_path, environment=dict(os.environ), result_fd=fd,
                deadline=time.monotonic() + 10)
        assert spawned[0].poll() is not None
    finally:
        if spawned and spawned[0].poll() is None:
            os.killpg(spawned[0].pid, child.signal.SIGKILL)
            spawned[0].wait(timeout=5)
        os.close(fd)


def test_precision_distinction_matches_pinned_unsloth_branch_without_torch_import(monkeypatch):
    class DType:
        pass
    half, bfloat, full = DType(), DType(), DType()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(dtype=DType,
        float16=half, bfloat16=bfloat))
    model = lambda dtype: SimpleNamespace(config=SimpleNamespace(dtype=dtype))
    args = SimpleNamespace(fp16=True, bf16=False)
    assert child._dtype_relation(model(half), args) == "MATCH"
    assert child._dtype_relation(model(bfloat), args) == "MISMATCH"
    assert child._dtype_relation(model(full), args) == "MISMATCH"
    assert child._dtype_relation(model(half), SimpleNamespace(fp16=False, bf16=True)) == "MISMATCH"


def test_actual_hung_and_flooding_cpu_children_are_reaped(tmp_path):
    answer = tmp_path / "answer"
    fd = os.open(answer, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        for code in ("import time; time.sleep(60)",
                     "import os,time\nwhile True: os.write(1,b'x'*65536)"):
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                child._wait_isolated_child((sys.executable, "-c", code), cwd=tmp_path,
                    environment=dict(os.environ), result_fd=fd, deadline=time.monotonic() + 0.2)
            assert time.monotonic() - start < 3
    finally:
        os.close(fd)


def test_result_file_has_one_aggregate_byte_cap_and_never_opens_line_logs():
    result = child.canonical(child._empty_result())
    class File:
        def __init__(self, raw): self.raw = raw
        def read(self, size):
            assert size == 8193
            return self.raw[:size]
        def close(self): self.closed = True
    class Sandbox:
        @property
        def stdout(self): raise AssertionError("SDK by-line stdout must not be opened")
        @property
        def stderr(self): raise AssertionError("SDK by-line stderr must not be opened")
        def __init__(self, raw): self.file = File(raw)
        def open(self, path, mode):
            assert path == child._RESULT_PATH and mode == "rb"
            return self.file
    accepted = Sandbox(result)
    assert host._read_closed_result_file(accepted) == result
    assert accepted.file.closed is True
    oversized = Sandbox(b"x" * 8193)
    with pytest.raises(ValueError):
        host._read_closed_result_file(oversized)
    assert oversized.file.closed is True


def test_bounded_hub_metadata_caps_unrelated_body_and_hung_read():
    class Response:
        headers = {"Content-Encoding": "identity"}
        def __init__(self, url, raw, *, hang=False):
            self.url, self.raw, self.hang = url, raw, hang
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def geturl(self): return self.url
        def read(self, size):
            if self.hang:
                time.sleep(60)
            raw, self.raw = self.raw[:size], self.raw[size:]
            return raw
    class Opener:
        def __init__(self, raw, *, hang=False): self.raw, self.hang = raw, hang
        def open(self, request, timeout):
            assert urlparse(request.full_url).hostname == "huggingface.co"
            assert timeout <= 5
            return Response(request.full_url, self.raw, hang=self.hang)
    url = "https://huggingface.co/api/models/fixture/tiny/revision/" + "a" * 40 + "?blobs=true"
    with pytest.raises(ValueError):
        host._bounded_hub_bytes(url, maximum=32, deadline=time.monotonic() + 2,
                                opener=Opener(b'{"unrelated":"' + b"x" * 100 + b'"}'))
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        host._bounded_hub_bytes(url, maximum=32, deadline=start + 0.15,
                                opener=Opener(b"{}", hang=True))
    assert time.monotonic() - start < 2


def test_model_manifest_uses_pinned_git_blob_and_lfs_sha_without_weights(monkeypatch):
    git = b'{"model_type":"fixture"}'
    git_blob = hashlib.sha1(f"blob {len(git)}\0".encode() + git).hexdigest()
    lfs_sha = "a" * 64
    revision = "b" * 40
    metadata = json.dumps({"sha": revision, "siblings": [
        {"rfilename": "config.json", "size": len(git), "blobId": git_blob},
        {"rfilename": "model.safetensors", "size": 100, "blobId": "c" * 40,
         "lfs": {"size": 100, "sha256": lfs_sha, "pointerSize": 130}},
    ]}).encode()
    urls = []
    def fake_read(url, *, maximum, deadline, opener):
        urls.append(url)
        return metadata if "/api/models/" in url else git
    monkeypatch.setattr(host, "_bounded_hub_bytes", fake_read)
    manifest = host.prepare_pinned_model_manifest("fixture/tiny", revision)
    assert manifest["members"][0]["sha256"] == hashlib.sha256(git).hexdigest()
    assert manifest["members"][1]["sha256"] == lfs_sha
    assert len(urls) == 2 and "model.safetensors" not in urls[1]


def test_cleanup_requires_fresh_exact_id_terminal_proof(monkeypatch):
    class Handle:
        object_id = "sb-owned"
        def terminate(self, *, wait):
            self.requested = True
    owned = Handle()
    class SDK:
        class Sandbox:
            @staticmethod
            def from_id(value, *, client):
                assert value == "sb-owned"
                return SimpleNamespace(object_id=value, poll=lambda: 137 if owned.__dict__.get("requested") else None)
    assert host._cleanup_owned(SDK, object(), owned, expected_id="sb-owned",
                               deadline=time.monotonic() + 1) == "TERMINAL_CONFIRMED"
    assert owned.requested is True
    assert host._cleanup_owned(SDK, object(), owned, expected_id="sb-other",
                               deadline=time.monotonic() + 1) == "INDETERMINATE"
    class FalseProofSDK:
        class Sandbox:
            @staticmethod
            def from_id(value, *, client):
                return SimpleNamespace(object_id="sb-other", poll=lambda: 0)
    assert host._cleanup_owned(FalseProofSDK, object(), owned, expected_id="sb-owned",
                               deadline=time.monotonic() + 0.05) == "INDETERMINATE"


def test_admission_requires_independent_metadata_source_wheel_and_scope(monkeypatch, tmp_path):
    from tuner.training.modal_host_runtime import ModalHostRuntimeV1
    recipe, dataset, plan, _ = _planned_input(tmp_path)
    probe_raw = derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset)
    workload_model = verify_probe_input(probe_raw)["workload"]["model"]
    content = b"{}"
    model = {"ref": workload_model["model_id"], "revision": workload_model["revision"],
        "tokenizer_revision": workload_model["tokenizer_revision"],
        "members": [{"path": "config.json", "size_bytes": len(content), "kind": "git",
            "upstream_digest": hashlib.sha1(b"blob 2\0" + content).hexdigest(),
            "sha256": hashlib.sha256(content).hexdigest()}]}
    metadata = SimpleNamespace(sha=model["revision"], siblings=[SimpleNamespace(
        rfilename="config.json", size=2, lfs=None,
        blob_id=model["members"][0]["upstream_digest"])])
    wheel = tmp_path / "source.whl"
    wheel.write_bytes(b"wheel")
    from tuner.runtime_profiles import load_runtime_profile
    profiles_root = Path(__file__).resolve().parents[2] / "Trainers/runtime_profiles"
    profile = load_runtime_profile("smollm2-135m-grpo-candidate-v1", profiles_root)
    release = SimpleNamespace(worker_closure_digest="1" * 64, manifest_digest="2" * 64,
        package_digest=hashlib.sha256(b"wheel").hexdigest(), image_ref=profile.image,
        to_dict=lambda: {})
    deployment = SimpleNamespace(runtime_release_digest="2" * 64, image_id="im-example",
        facts_digest="3" * 64, client_binding=SimpleNamespace(environment_ref="synaptic-smoke-v1"))
    facts = SimpleNamespace(schema_version="synaptic-modal-packaged-runtime-facts/v2",
        image_id="im-example", package_digest=release.package_digest,
        worker_closure_digest="1" * 64, account_ref="synaptic-labs",
        workspace_ref="synaptic-labs", environment_ref="synaptic-smoke-v1")
    quote = SimpleNamespace(resource=SimpleNamespace(accelerator="L40S", accelerator_count=1,
        timeout_seconds=120), maximum_cost_minor_units=100,
        gpu_only_timeout_estimate_minor_units=80, quote_digest="4" * 64)
    runtime = ModalHostRuntimeV1(release, deployment, facts, quote,
                                 (("vo-model", "model-cache-fixture"),), b"a" * 32, None)
    monkeypatch.setattr(host, "_source_pair", lambda engine, host_repo: ("e" * 40, "f" * 40))
    monkeypatch.setattr(host, "load_packaged_grpo_worker_closure", lambda: SimpleNamespace(digest="1" * 64))
    monkeypatch.setattr(host, "_verify_wheel_sources", lambda wheel, repository: "1" * 64)
    monkeypatch.setattr(host, "_validate_deployment_release", lambda *args: None)
    monkeypatch.setattr(host, "_model_role", lambda runtime: "vo-model")
    monkeypatch.setattr(host, "validate_dispatch", lambda raw: None)
    monkeypatch.setattr(host, "_pinned_model_info", lambda model: metadata)
    args = dict(runtime=runtime, snapshot_runtime=runtime,
        engine_repository=tmp_path, host_repository=tmp_path, wheel_path=wheel,
        recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset,
        profiles_root=profiles_root,
        input_bytes=probe_raw, model=model,
        library_sources=child.pinned_library_sources(),
        attempt_ref="grpo-diag-" + "b" * 24)
    assert host.admit_modal_grpo_constructor(**args).image_id == "im-example"
    for changed in ({"model": {**model, "members": [{**model["members"][0], "upstream_digest": "0" * 40}]}},
                    {"runtime": ModalHostRuntimeV1(release, deployment, SimpleNamespace(**{**vars(facts),
                        "account_ref": "other"}), quote, (("vo-model", "model-cache-fixture"),), b"a" * 32, None)}):
        with pytest.raises(host.ModalGRPOConstructorUnavailable):
            host.admit_modal_grpo_constructor(**{**args, **changed})
    monkeypatch.setattr(host, "_pinned_model_info", lambda model: SimpleNamespace(
        sha="0" * 40, siblings=metadata.siblings))
    with pytest.raises(host.ModalGRPOConstructorUnavailable):
        host.admit_modal_grpo_constructor(**args)
    monkeypatch.setattr(host, "_pinned_model_info", lambda model: metadata)
    wheel.write_bytes(b"mutated")
    with pytest.raises(host.ModalGRPOConstructorUnavailable):
        host.admit_modal_grpo_constructor(**args)


def test_complete_grpo_wheel_source_comparison_rejects_one_changed_member():
    from io import BytesIO
    from zipfile import ZipFile
    from tuner.runtime.offline_grpo_worker import load_packaged_offline_grpo_worker_manifest
    from tuner.runtime.packaged_grpo_worker_closure import load_packaged_grpo_worker_closure
    root = Path(__file__).resolve().parents[2]
    offline = load_packaged_offline_grpo_worker_manifest()
    outer = load_packaged_grpo_worker_closure()
    names = {item.path.as_posix() for item in offline.closure.members}
    names.update(item["path"] for item in outer.members)
    def wheel(changed=None):
        output = BytesIO()
        with ZipFile(output, "w") as archive:
            for name in sorted(names):
                raw = (root / name).read_bytes()
                archive.writestr(name, raw + b"changed" if name == changed else raw)
        return output.getvalue()
    assert host._verify_wheel_sources(wheel(), root) == outer.digest
    with pytest.raises(ValueError):
        host._verify_wheel_sources(wheel("Trainers/grpo/runtime_v1.py"), root)


def test_distinct_receipt_retains_primary_and_cleanup_uncertainty(tmp_path):
    from tuner.training.modal_host_storage import ModalHostStorageV1
    runtime = SimpleNamespace(qualification_key=b"x" * 32,
        release=SimpleNamespace(manifest_digest="f" * 64),
        deployment_facts=SimpleNamespace(facts_digest="1" * 64, image_id="im-fixture"))
    admission = SimpleNamespace(attempt_ref="grpo-diag-" + "c" * 24,
        dispatch_bytes=b"dispatch", source_commit="d" * 40,
        host_source_commit="a" * 40, wheel_sha256="e" * 64,
        runtime_release_digest="f" * 64, deployment_facts_digest="1" * 64,
        image_id="im-fixture", snapshot_deployment_facts_digest="3" * 64,
        model_cache_volume_id="vo-fixture", quote_digest="2" * 64)
    receipt = host._receipt(runtime, admission, sandbox_id=None,
        primary="INDETERMINATE", cleanup="INDETERMINATE", result=None, terminal=None,
        failure_phase="CREATE")
    assert receipt["body"]["cpu_qualified"] is False
    assert receipt["body"]["training_qualified"] is False
    assert receipt["body"]["diagnostic_type"] == "GRPO_CONSTRUCTOR_ONLY"
    with ModalHostStorageV1(tmp_path / "journal.sqlite3", "diagnostic-test") as storage:
        assert host._catalog(storage, "modal-grpo-constructor-results").publish_if_absent(
            admission.attempt_ref, receipt)
        assert host.read_modal_grpo_constructor_receipt(runtime, storage, admission.attempt_ref) == receipt
        changed = dict(receipt, body={**receipt["body"], "cleanup": "TERMINAL_CONFIRMED"})
        assert changed != receipt


def test_restricted_modal_dispatch_uses_one_sandbox_and_separate_receipt(monkeypatch, tmp_path):
    from tuner.training.modal_host_storage import ModalHostStorageV1
    from tuner.training.modal_host_runtime import ModalHostRuntimeV1
    recipe, dataset, plan, _ = _planned_input(tmp_path)
    probe_raw = derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset)
    profiles_root = Path(__file__).resolve().parents[2] / "Trainers/runtime_profiles"
    wheel = tmp_path / "source.whl"
    wheel.write_bytes(b"wheel")
    binding = SimpleNamespace(environment_ref="synaptic-smoke-v1")
    release = SimpleNamespace(manifest_digest="1" * 64, worker_closure_digest="2" * 64,
        package_digest=hashlib.sha256(b"wheel").hexdigest(), python_executable="/opt/python/bin/python")
    deployment = SimpleNamespace(facts_digest="3" * 64, image_id="im-exact",
        client_binding=binding)
    facts = SimpleNamespace(app_name="app-exact", environment_ref="synaptic-smoke-v1")
    quote = SimpleNamespace(resource=object(), maximum_cost_minor_units=100,
        quote_digest="4" * 64)
    runtime = ModalHostRuntimeV1(release, deployment, facts, quote,
        (("vo-retained", "cache-exact"),), b"x" * 32, None)
    source = SimpleNamespace(commit="5" * 40, host_commit="8" * 40)
    monkeypatch.setattr(host, "_source_pair", lambda engine, host_repo:
                        (source.commit, source.host_commit))
    monkeypatch.setattr(host, "_verify_wheel_sources", lambda raw, repo: "2" * 64)
    monkeypatch.setattr(host, "_model_role", lambda runtime: "vo-retained")
    monkeypatch.setattr(host, "validate_dispatch", lambda raw: ({}, probe_raw))
    monkeypatch.setattr(host, "quote_modal_runtime_for_host", lambda **kwargs: quote)
    monkeypatch.setattr(host, "_CurrentQualificationObserver", lambda **kwargs: object())
    monkeypatch.setattr(host, "_observe_current_layout", lambda *args: "observed")
    result = child.canonical(child._empty_result())
    class Stdin:
        def __init__(self): self.writes = []
        def write(self, raw): self.writes.append(raw)
        def write_eof(self): pass
        def drain(self): pass
    class Sandbox:
        object_id = "sb-exact"
        def __init__(self):
            self.stdin = Stdin()
            self.acknowledged = False
        def open(self, path, mode):
            if path == child._READY_PATH and mode == "rb":
                return SimpleNamespace(read=lambda size: b"1", close=lambda: None)
            if path == child._RESULT_PATH and mode == "rb":
                return SimpleNamespace(read=lambda size: result, close=lambda: None)
            if path == child._ACK_PATH and mode == "xb":
                def write(raw):
                    assert raw == b"1"
                    self.acknowledged = True
                return SimpleNamespace(write=write, close=lambda: None)
            raise AssertionError("unexpected sandbox file")
        @property
        def stdout(self): raise AssertionError("line-buffered logs forbidden")
        @property
        def stderr(self): raise AssertionError("line-buffered logs forbidden")
        def terminate(self, *, wait):
            raise AssertionError("completed Sandbox must not need termination")
    sandbox = Sandbox()
    calls = []
    class Volume:
        object_id = "vo-retained"
        _read_only = True
        def hydrate(self, client): pass
        def with_mount_options(self, *, read_only):
            assert read_only is True
            return self
    class SDK:
        __version__ = "1.5.4"
        class App:
            @staticmethod
            def lookup(*args, **kwargs): return object()
        class Volume:
            @staticmethod
            def from_name(name, **kwargs):
                assert name == "cache-exact" and kwargs["create_if_missing"] is False
                return Volume()
        class Image:
            @staticmethod
            def from_id(value, *, client): return SimpleNamespace(object_id=value)
        class Sandbox:
            @staticmethod
            def create(*args, **kwargs):
                calls.append(kwargs)
                return sandbox
            @staticmethod
            def from_id(value, *, client):
                assert value == "sb-exact"
                return SimpleNamespace(object_id=value,
                    poll=lambda: 0 if sandbox.acknowledged else None)
    admission = host.ModalGRPOConstructorAdmission(b"dispatch", "grpo-diag-" + "6" * 24,
        source.commit, source.host_commit, release.package_digest, release.manifest_digest,
        deployment.facts_digest,
        deployment.image_id, deployment.facts_digest, "vo-retained", "cache-exact",
        quote.quote_digest, host._SEAL)
    with ModalHostStorageV1(tmp_path / "journal.sqlite3", "diagnostic-test") as storage:
        receipt = host.execute_modal_grpo_constructor_once(sdk=SDK, client=object(),
            client_binding=binding, runtime=runtime, snapshot_runtime=runtime,
            private_storage=storage, admission=admission,
            engine_repository=tmp_path, host_repository=tmp_path, wheel_path=wheel,
            recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset,
            profiles_root=profiles_root)
        assert storage.attempts.resolve(admission.attempt_ref) is not None
        assert receipt["body"]["primary"] == "COMPLETED"
        assert receipt["body"]["cleanup"] == "TERMINAL_CONFIRMED"
        assert receipt["body"]["cpu_qualified"] is False
        assert receipt["body"]["training_qualified"] is False
        assert len(calls) == 1
        assert calls[0]["gpu"] == "L40S" and calls[0]["cpu"] == 2.0
        assert calls[0]["memory"] == (8192, 8192) and calls[0]["timeout"] == 120
        assert calls[0]["block_network"] is True and calls[0]["secrets"] == []
        assert sandbox.stdin.writes == [b"dispatch"]
        assert sandbox.acknowledged is True
        with pytest.raises(Exception):
            host.execute_modal_grpo_constructor_once(sdk=SDK, client=object(),
                client_binding=binding, runtime=runtime, snapshot_runtime=runtime,
                private_storage=storage, admission=admission,
                engine_repository=tmp_path, host_repository=tmp_path, wheel_path=wheel,
                recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset,
                profiles_root=profiles_root)
        assert len(calls) == 1
        uncertain = host.ModalGRPOConstructorAdmission(b"dispatch", "grpo-diag-" + "7" * 24,
            source.commit, source.host_commit, release.package_digest,
            release.manifest_digest, deployment.facts_digest,
            deployment.image_id, deployment.facts_digest, "vo-retained", "cache-exact",
            quote.quote_digest, host._SEAL)
        def fail_create(*args, **kwargs):
            calls.append(kwargs)
            raise TimeoutError
        monkeypatch.setattr(SDK.Sandbox, "create", staticmethod(fail_create))
        uncertain_receipt = host.execute_modal_grpo_constructor_once(sdk=SDK, client=object(),
            client_binding=binding, runtime=runtime, snapshot_runtime=runtime,
            private_storage=storage, admission=uncertain,
            engine_repository=tmp_path, host_repository=tmp_path, wheel_path=wheel,
            recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset,
            profiles_root=profiles_root)
        assert uncertain_receipt["body"]["primary"] == "INDETERMINATE"
        assert uncertain_receipt["body"]["cleanup"] == "INDETERMINATE"
        assert uncertain_receipt["body"]["failure_phase"] == "CREATE"
        assert storage.attempts.resolve(uncertain.attempt_ref) is not None
        assert len(calls) == 2
