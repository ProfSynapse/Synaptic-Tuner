"""Restricted, installed GRPO constructor probe. No public training authority.

The Sandbox supervisor verifies a read-only retained snapshot and makes the
same private copy used by packaged training. Its isolated child uses the normal
GRPO environment, import guard, input builder, reward and callbacks, then stops
immediately after trainer construction. All observable diagnostics are closed.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.resources
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import time

from tuner.runtime import packaged_sft_execution as shared
from tuner.runtime.packaged_grpo_execution import grpo_child_environment
from tuner.runtime.packaged_sft_child import _installed_import_guard
from tuner.runtime.releases import parse_packaged_runtime_release

SCHEMA = "synaptic-modal-grpo-constructor-result/v1"
DISPATCH_SCHEMA = "synaptic-modal-grpo-constructor-dispatch/v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_SOURCES = ("unsloth/models/rl.py", "trl/trainer/grpo_trainer.py", "transformers/trainer.py")
_STAGES = frozenset({"DISPATCH", "RELEASE", "MODEL_ADMISSION", "INPUT", "IMPORT", "REWARDS",
                     "DEPENDENCIES", "CONFIG", "MODEL_LIBRARY_LOAD", "LORA_ATTACH", "DATA_PREP",
                     "TRAINER_SETUP", "POSTCHECK"})
_CATEGORIES = frozenset({"NONE", "TYPE", "VALUE", "OOM", "TIMEOUT", "OTHER"})
_SITES = frozenset({"NONE", "UNSLOTH_WRAPPER", "PATCHED_GRPO_INIT", "TRANSFORMERS_BASE", "OTHER"})


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii") + b"\n"


def _closed(value, allowed):
    if type(value) is not str or value not in allowed:
        raise ValueError
    return value


def pinned_library_sources():
    resource = importlib.resources.files("tuner.runtime").joinpath(
        "manifests", "grpo-constructor-library-sources-v1.json")
    if not isinstance(resource, Path):
        raise ValueError
    raw = shared.stable_read(resource, 4096)
    value = json.loads(raw)
    if (type(value) is not dict or canonical(value) != raw
            or set(value) != {"schema_version", "sources"}
            or value["schema_version"] != "synaptic-grpo-constructor-library-sources/v1"
            or type(value["sources"]) is not dict or set(value["sources"]) != set(_SOURCES)):
        raise ValueError
    versions = {"unsloth/models/rl.py": ("unsloth", "2026.1.2"),
                "trl/trainer/grpo_trainer.py": ("trl", "0.24.0"),
                "transformers/trainer.py": ("transformers", "4.57.1")}
    for name, item in value["sources"].items():
        if (type(item) is not dict
                or set(item) != {"distribution", "version", "wheel_sha256", "source_sha256"}
                or (item["distribution"], item["version"]) != versions[name]
                or any(type(item[key]) is not str or _SHA.fullmatch(item[key]) is None
                       for key in ("wheel_sha256", "source_sha256"))):
            raise ValueError
    return {name: value["sources"][name]["source_sha256"] for name in _SOURCES}


def validate_model_inventory(model):
    """Expected upstream identities must come from a separate reviewed source."""
    if (type(model) is not dict or set(model) != {"ref", "revision", "tokenizer_revision", "members"}
            or _REF.fullmatch(model["ref"]) is None or _COMMIT.fullmatch(model["revision"]) is None
            or model["tokenizer_revision"] != model["revision"]
            or type(model["members"]) is not list or not 0 < len(model["members"]) <= 64):
        raise ValueError
    names, total = [], 0
    for item in model["members"]:
        if type(item) is not dict or set(item) != {"path", "size_bytes", "kind", "upstream_digest", "sha256"}:
            raise ValueError
        name = item["path"]
        if (type(name) is not str or not 0 < len(name) <= 4096 or "\\" in name
                or PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name
                or any(part in {"", ".", "..", ".git", ".cache"} for part in name.split("/"))):
            raise ValueError
        names.append(name)
        size = item["size_bytes"]
        if type(size) is not int or not 0 <= size <= 2 * 1024**3:
            raise ValueError
        total += size
        kind = item["kind"]
        expected = item["upstream_digest"]
        if (kind not in {"git", "lfs"} or type(expected) is not str
                or re.fullmatch(r"[0-9a-f]{64}\Z" if kind == "lfs" else r"[0-9a-f]{40}\Z", expected) is None
                or type(item["sha256"]) is not str or _SHA.fullmatch(item["sha256"]) is None):
            raise ValueError
    if names != sorted(set(names)) or total > 2 * 1024**3:
        raise ValueError
    return tuple((item["path"], item["size_bytes"], item["sha256"]) for item in model["members"])


def validate_dispatch(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 131072:
        raise ValueError
    value = json.loads(raw)
    if (type(value) is not dict or canonical(value) != raw
            or set(value) != {"schema_version", "attempt_ref", "source_commit",
                              "host_source_commit", "engine_gitlink", "release",
                              "deployment_facts_digest", "image_id", "input_base64", "input_sha256",
                              "model", "library_sources"}
            or value["schema_version"] != DISPATCH_SCHEMA
            or type(value["attempt_ref"]) is not str or re.fullmatch(r"grpo-diag-[0-9a-f]{24}\Z", value["attempt_ref"]) is None
            or type(value["source_commit"]) is not str or _COMMIT.fullmatch(value["source_commit"]) is None
            or type(value["host_source_commit"]) is not str or _COMMIT.fullmatch(value["host_source_commit"]) is None
            or value["engine_gitlink"] != value["source_commit"]
            or type(value["deployment_facts_digest"]) is not str or _SHA.fullmatch(value["deployment_facts_digest"]) is None
            or type(value["image_id"]) is not str or re.fullmatch(r"im-[A-Za-z0-9]{1,64}\Z", value["image_id"]) is None
            or type(value["input_sha256"]) is not str or _SHA.fullmatch(value["input_sha256"]) is None):
        raise ValueError
    validate_model_inventory(value["model"])
    sources = value["library_sources"]
    if type(sources) is not dict or sources != pinned_library_sources():
        raise ValueError
    input_raw = base64.b64decode(value["input_base64"], validate=True)
    if hashlib.sha256(input_raw).hexdigest() != value["input_sha256"]:
        raise ValueError
    verify_probe_input(input_raw)
    parse_packaged_runtime_release(value["release"])
    return value, input_raw


def verify_probe_input(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError
    document = json.loads(raw)
    if (type(document) is not dict or canonical(document) != raw
            or set(document) != {"schema_version", "workload", "dataset_base64",
                                 "qualification_policy", "row_count", "runtime_profile",
                                 "profile_sha256", "recipe_sha256", "plan_sha256",
                                 "dataset_sha256"}
            or document["schema_version"] != "synaptic-grpo-constructor-probe-input/v3"
            or document["row_count"] != 4):
        raise ValueError
    dataset = base64.b64decode(document["dataset_base64"], validate=True)
    if (type(document["runtime_profile"]) is not str
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}\Z", document["runtime_profile"]) is None
            or type(document["profile_sha256"]) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}\Z", document["profile_sha256"]) is None
            or any(type(document[key]) is not str or _SHA.fullmatch(document[key]) is None
                   for key in ("recipe_sha256", "plan_sha256", "dataset_sha256"))
            or hashlib.sha256(dataset).hexdigest() != document["dataset_sha256"]):
        raise ValueError
    workload = document["workload"]
    from Trainers.grpo.runtime_v1 import admit_grpo_inputs
    config, rows, documents, schema = admit_grpo_inputs(
        config=workload["config"], dataset_bytes=dataset,
        reward_documents=workload["reward_documents"],
        reward_schema=workload["reward_schema"],
        model_identity=workload["model"], workload=workload)
    if (len(rows) != 4 or config != workload["config"]
            or documents != workload["reward_documents"]
            or schema != workload["reward_schema"]):
        raise ValueError
    from tuner.training.grpo_compilation import validate_qualification_evidence_policy
    validate_qualification_evidence_policy(document["qualification_policy"])
    return document


def _source_verified(sources):
    """Check installed source identities before projecting a library region."""
    import importlib.metadata
    import importlib.util
    for distribution, version in (("unsloth", "2026.1.2"), ("trl", "0.24.0"),
                                  ("transformers", "4.57.1")):
        if importlib.metadata.version(distribution) != version:
            return False
    for name in _SOURCES:
        module = name[:-3].replace("/", ".")
        spec = importlib.util.find_spec(module)
        if spec is None or type(spec.origin) is not str:
            return False
        path = Path(spec.origin)
        if not path.is_absolute() or path.is_symlink():
            return False
        if hashlib.sha256(shared.stable_read(path, 4 * 1024 * 1024)).hexdigest() != sources[name]:
            return False
    return True


def _code(method):
    return getattr(method, "__code__", None)


def _site(error, trainer_class, sources_verified):
    if not sources_verified:
        return "OTHER"
    chain = tuple(cls.__dict__.get("__init__") for cls in trainer_class.__mro__)
    wrapper = _code(chain[0]) if chain else None
    patched = _code(chain[1]) if len(chain) > 1 else None
    base = {_code(cls.__dict__["__init__"]) for cls in trainer_class.__mro__
            if cls.__module__.startswith("transformers") and "__init__" in cls.__dict__}
    found = "OTHER"
    tb = error.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        if code in base:
            found = "TRANSFORMERS_BASE"
        elif code is patched and patched is not None:
            found = "PATCHED_GRPO_INIT"
        elif code is wrapper and wrapper is not None:
            found = "UNSLOTH_WRAPPER"
        tb = tb.tb_next
    return found


def _dtype_relation(model, args):
    try:
        import torch
        dtype = getattr(model.config, "dtype", None) or getattr(model.config, "torch_dtype", None)
        if dtype is None:
            dtype = model.get_input_embeddings().weight.dtype
        fp16, bf16 = getattr(args, "fp16", None), getattr(args, "bf16", None)
        if type(fp16) is not bool or type(bf16) is not bool:
            return "UNKNOWN"
        if dtype == torch.float16 and bf16:
            return "MISMATCH"
        if isinstance(dtype, torch.dtype) and dtype != torch.float16 and fp16:
            return "MISMATCH"
        if (dtype == torch.float16 and fp16 and not bf16) or (dtype == torch.bfloat16 and bf16 and not fp16):
            return "MATCH"
    except BaseException:
        pass
    return "UNKNOWN"


def _observation(trainer_class, kwargs, verified):
    result = {"patched_class": "UNKNOWN", "signature_binding": "UNAVAILABLE",
              "precision_relation": "UNKNOWN", "model_predicate": "OTHER",
              "tokenizer_predicate": "OTHER", "reward_predicate": "OTHER",
              "callbacks_predicate": "OTHER", "dataset_predicate": "OTHER",
              "class_digest": None, "signature_digest": None}
    try:
        from transformers import PreTrainedTokenizerBase, TrainerCallback
        from datasets import Dataset
        signature = inspect.signature(trainer_class.__init__)
        result["signature_digest"] = hashlib.sha256(str(signature).encode("utf-8")).hexdigest()
        result["class_digest"] = hashlib.sha256(canonical((trainer_class.__module__, trainer_class.__qualname__))).hexdigest()
        try:
            signature.bind(object(), **kwargs)
            result["signature_binding"] = "ACCEPTED"
        except TypeError:
            result["signature_binding"] = "REJECTED"
        chain = trainer_class.__mro__
        if verified and len(chain) > 1:
            result["patched_class"] = "YES" if chain[0].__name__ == "UnslothGRPOTrainer" and chain[1].__name__ == "_UnslothGRPOTrainer" else "NO"
        model, tokenizer, reward, callbacks, data = (kwargs[key] for key in
            ("model", "processing_class", "reward_funcs", "callbacks", "train_dataset"))
        result["precision_relation"] = _dtype_relation(model, kwargs["args"])
        result["model_predicate"] = "HAS_CONFIG_EMBEDDINGS" if hasattr(model, "config") and callable(getattr(model, "get_input_embeddings", None)) else "OTHER"
        result["tokenizer_predicate"] = "PRETRAINED_TOKENIZER" if isinstance(tokenizer, PreTrainedTokenizerBase) else "OTHER"
        result["reward_predicate"] = "CALLABLE_NAMED" if callable(reward) and type(getattr(reward, "__name__", None)) is str else "OTHER"
        result["callbacks_predicate"] = "TRAINER_CALLBACKS" if type(callbacks) is list and len(callbacks) == 2 and all(isinstance(value, TrainerCallback) for value in callbacks) else "OTHER"
        result["dataset_predicate"] = "DATASET_FOUR" if isinstance(data, Dataset) and len(data) == 4 else "OTHER"
    except BaseException:
        pass
    return result


def _empty_result():
    return {"schema_version": SCHEMA, "status": "REJECTED", "stage": "DISPATCH",
            "category": "OTHER", "type_site": "OTHER", "source_verified": False,
            "patched_class": "UNKNOWN", "signature_binding": "UNAVAILABLE",
            "precision_relation": "UNKNOWN", "model_predicate": "OTHER",
            "tokenizer_predicate": "OTHER", "reward_predicate": "OTHER",
            "callbacks_predicate": "OTHER", "dataset_predicate": "OTHER",
            "class_digest": None, "signature_digest": None,
            "training_executed": False, "generation_executed": False,
            "optimizer_update_executed": False}


def validate_result(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 8192:
        raise ValueError
    value = json.loads(raw)
    expected = _empty_result()
    if type(value) is not dict or set(value) != set(expected) or canonical(value) != raw:
        raise ValueError
    _closed(value["status"], {"REJECTED", "CONSTRUCTED"})
    _closed(value["stage"], _STAGES)
    _closed(value["category"], _CATEGORIES)
    _closed(value["type_site"], _SITES)
    for key, allowed in (("patched_class", {"YES", "NO", "UNKNOWN"}),
                         ("signature_binding", {"ACCEPTED", "REJECTED", "UNAVAILABLE"}),
                         ("precision_relation", {"MATCH", "MISMATCH", "UNKNOWN"}),
                         ("model_predicate", {"HAS_CONFIG_EMBEDDINGS", "OTHER"}),
                         ("tokenizer_predicate", {"PRETRAINED_TOKENIZER", "OTHER"}),
                         ("reward_predicate", {"CALLABLE_NAMED", "OTHER"}),
                         ("callbacks_predicate", {"TRAINER_CALLBACKS", "OTHER"}),
                         ("dataset_predicate", {"DATASET_FOUR", "OTHER"})):
        _closed(value[key], allowed)
    if (type(value["source_verified"]) is not bool
            or any(value[key] is not False for key in
                   ("training_executed", "generation_executed", "optimizer_update_executed"))
            or any(item is not None and (type(item) is not str or _SHA.fullmatch(item) is None)
                   for item in (value["class_digest"], value["signature_digest"]))):
        raise ValueError
    if value["status"] == "CONSTRUCTED":
        if (value["stage"] != "TRAINER_SETUP" or value["category"] != "NONE"
                or value["type_site"] != "NONE" or value["source_verified"] is not True):
            raise ValueError
    elif value["category"] == "NONE":
        raise ValueError
    return value


def _result_from_error(result, error, stage, trainer_class, verified):
    result["status"] = "REJECTED"
    result["stage"] = stage if stage in _STAGES else "POSTCHECK"
    result["category"] = ("TYPE" if isinstance(error, TypeError) else
                          "VALUE" if isinstance(error, ValueError) else
                          "TIMEOUT" if isinstance(error, TimeoutError) else
                          "OOM" if isinstance(error, MemoryError) else "OTHER")
    result["type_site"] = _site(error, trainer_class, verified) if result["category"] == "TYPE" and trainer_class is not None else "OTHER"
    return result


def _isolated(transport, digest, result_fd):
    result, stage, trainer_class, verified = _empty_result(), "DISPATCH", None, False
    retained = None
    try:
        raw = shared.stable_read(Path(transport), 131072)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError
        value = json.loads(raw)
        if (type(value) is not dict or canonical(value) != raw
                or set(value) != {"dispatch", "private_manifest", "paths", "environment", "deadline"}):
            raise ValueError
        dispatch, dataset = validate_dispatch(canonical(value["dispatch"]))
        release = parse_packaged_runtime_release(dispatch["release"])
        paths = shared.PackagedSFTPaths(**{key: Path(item) for key, item in value["paths"].items()})
        environment = grpo_child_environment(release, paths, Path(os.environ.get("SYNAPTIC_MODEL_SNAPSHOT", "")))
        if (dict(os.environ) != dict(environment) or value["environment"] != list(map(list, environment))
                or not sys.flags.isolated or Path.cwd() != paths.tmp
                or time.monotonic() >= value["deadline"]):
            raise ValueError
        stage = "RELEASE"
        trainer_source = shared._inspect_release(release, method="grpo")
        guard = _installed_import_guard(release, trainer_source, method="grpo")
        sys.meta_path.insert(0, guard)
        try:
            stage = "MODEL_ADMISSION"
            model = dispatch["model"]
            retained = shared._retain_private_snapshot(value["private_manifest"], paths,
                {"ref": model["ref"], "revision": model["revision"], "tokenizer_revision": model["tokenizer_revision"]})
            retained.check()
            stage = "INPUT"
            document = validate_dispatch(canonical(dispatch))[0]
            probe = verify_probe_input(base64.b64decode(document["input_base64"], validate=True))
            stage = "IMPORT"
            from Trainers.grpo.runtime_v1 import run_grpo
            output = paths.state / "grpo-output"
            output.mkdir(mode=0o700, exist_ok=False)
            def mark(value):
                nonlocal stage
                if value in _STAGES:
                    stage = value
            def observe(cls, kwargs):
                nonlocal trainer_class, verified
                trainer_class = cls
                verified = _source_verified(dispatch["library_sources"])
                result["source_verified"] = verified
                if not verified:
                    raise ValueError
                result.update(_observation(cls, kwargs, verified))
            trainer = run_grpo(
                config=probe["workload"]["config"], dataset_bytes=dataset,
                reward_documents=probe["workload"]["reward_documents"],
                reward_schema=probe["workload"]["reward_schema"],
                model_snapshot=retained.snapshot,
                model_identity={"model_id": model["ref"], "revision": model["revision"],
                                "tokenizer_revision": model["tokenizer_revision"]},
                output_root=output, workload=probe["workload"],
                metric_sink=lambda record: None, _packaged_artifacts=True,
                qualification_evidence=probe["qualification_policy"],
                _child_deadline=value["deadline"], _runtime_milestone=mark,
                _constructor_only=True, _constructor_observer=observe)
            del trainer
            retained.check()
            if time.monotonic() >= value["deadline"]:
                raise TimeoutError
            result.update(status="CONSTRUCTED", stage="TRAINER_SETUP", category="NONE", type_site="NONE")
        finally:
            sys.meta_path.remove(guard)
    except BaseException as error:
        _result_from_error(result, error, stage, trainer_class, verified)
    finally:
        if retained is not None:
            try:
                retained.close()
            except BaseException:
                result.update(status="REJECTED", stage="POSTCHECK", category="OTHER", type_site="OTHER")
        payload = canonical(result)
        if len(payload) > 8192:
            payload = canonical(_empty_result())
        os.write(result_fd, payload)


def _private_paths(root, cache):
    for name in ("state", "tmp", "tracking", "artifacts", "prepared_input"):
        (root / name).mkdir(mode=0o700)
    return shared.PackagedSFTPaths(root / "prepared_input", root / "artifacts",
        root / "state", root / "tracking", cache, root / "tmp")


def _verify_retained_model(model, cache):
    from tuner.execution.providers.modal.model_snapshot import _verify
    relative = Path("models--" + model["ref"].replace("/", "--")) / "snapshots" / model["revision"]
    snapshot = cache / relative
    held, current = shared._snapshot_inventory(snapshot, cache)
    expected = validate_model_inventory(model)
    try:
        if current != expected:
            raise ValueError
        for item in model["members"]:
            if _verify(snapshot / item["path"], item["size_bytes"], item["kind"],
                       item["upstream_digest"]) != item["sha256"]:
                raise ValueError
        for directory in held:
            directory.check()
        return snapshot, held, current
    except BaseException:
        shared._close_resources(held)
        raise


def _wait_isolated_child(argv, *, cwd, environment, result_fd, deadline):
    """Own and reap the entire child process group under one finite deadline."""
    child = subprocess.Popen(argv, cwd=cwd, env=environment, pass_fds=(result_fd,),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    try:
        child.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        raise TimeoutError from None
    return child.returncode


def _supervise():
    result = _empty_result()
    result_fd = os.dup(1)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    os.close(sink)
    held, private = (), None
    try:
        raw = sys.stdin.buffer.read(131073)
        dispatch, _ = validate_dispatch(raw)
        stage = "RELEASE"
        release = parse_packaged_runtime_release(dispatch["release"])
        shared._inspect_release(release, method="grpo")
        cache = Path("/mnt/synaptic-model-cache")
        stage = "MODEL_ADMISSION"
        snapshot, held, inventory = _verify_retained_model(dispatch["model"], cache)
        with tempfile.TemporaryDirectory(prefix="synaptic-grpo-diagnostic-") as temporary:
            root = Path(temporary)
            paths = _private_paths(root, cache)
            model = dispatch["model"]
            private = shared._copy_private_snapshot(snapshot, held, inventory, paths,
                {"ref": model["ref"], "revision": model["revision"], "tokenizer_revision": model["tokenizer_revision"]})
            private.check()
            stage = "INPUT"
            deadline = time.monotonic() + 110
            environment = grpo_child_environment(release, paths, private.snapshot)
            transport = canonical({"dispatch": dispatch, "private_manifest": private.manifest,
                "paths": {key: str(getattr(paths, key)) for key in paths.__dataclass_fields__},
                "environment": list(map(list, environment)), "deadline": deadline})
            target = paths.state / "constructor-transport.json"
            with target.open("xb") as stream:
                stream.write(transport)
                stream.flush()
                os.fsync(stream.fileno())
            answer = paths.state / "constructor-result.json"
            answer_fd = os.open(answer, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                exit_code = _wait_isolated_child((release.python_executable, "-I", "-B", "-m",
                    "tuner.runtime.packaged_grpo_constructor_child", "--isolated", str(target),
                    hashlib.sha256(transport).hexdigest(), str(answer_fd)),
                    cwd=paths.tmp, environment=dict(environment), result_fd=answer_fd,
                    deadline=deadline)
                if exit_code != 0:
                    raise ValueError
            finally:
                os.close(answer_fd)
            response = shared.stable_read(answer, 8192)
            result = validate_result(response)
            private.check()
            for directory in held:
                directory.check()
    except BaseException as error:
        stage = locals().get("stage", "DISPATCH")
        _result_from_error(result, error, stage, None, False)
    finally:
        if private is not None:
            try:
                private.close()
            except BaseException:
                result.update(status="REJECTED", stage="POSTCHECK", category="OTHER", type_site="OTHER")
        if held:
            if shared._close_resources(held):
                result.update(status="REJECTED", stage="POSTCHECK", category="OTHER", type_site="OTHER")
        payload = canonical(result)
        if len(payload) > 8192:
            payload = canonical(_empty_result())
        os.write(result_fd, payload)
        os.close(result_fd)
    return 0


def main():
    if sys.argv[1:] == ["--supervise"]:
        return _supervise()
    if len(sys.argv) == 5 and sys.argv[1] == "--isolated" and sys.flags.isolated:
        result_fd = int(sys.argv[4])
        sink = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        os.close(sink)
        _isolated(sys.argv[2], sys.argv[3], result_fd)
        return 0
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
