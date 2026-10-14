"""Small host-owned public API consumer; no provider qualification is implied."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from shared.llm.base import BaseLLMClient
from shared.llm.providers.openai_responses import OpenAIResponsesClient
from shared.llm.providers.openrouter import OpenRouterClient
from shared.environments.local_worker import WorkerEnvironmentRuntime
from SynthChat.utils.yaml_loader import parse_workflow_yaml
from synaptic_tuner.api.v1.data_facade import (
    DataAPI,
    DataRequest,
    DataMode,
    DataPlan,
    DataRunState,
    DataListRequest,
    DatasetValidateRequest,
)
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationAPI,
    EvaluationRequest,
    EvaluationPlan,
    EvaluationListRequest,
    EvaluationResultRequest,
    EvaluationRunState,
)
from synaptic_tuner.api.v1.environment_execution import EnvironmentBindingV1
from synaptic_tuner.api.v1.reference.data import (
    ReferenceDataPortsV1,
    build_data_operations,
    data_run_ref,
)
from synaptic_tuner.api.v1.reference.environment_sessions import (
    CapturedEnvironmentSessionFactory,
)
from synaptic_tuner.api.v1.reference.evaluation import (
    ReferenceEvaluationPortsV1,
    EvaluationBackendRegistryV1,
    DirectoryEvaluationArtifactSinkV1,
    MeteredPaidBackendV1,
    MeteredPaidJudgeV1,
    compose_reference_evaluation,
    evaluation_run_ref,
)
from synaptic_tuner.api.v1.reference.spend import (
    SpendScopeV1,
    SpendEffectRecordV1,
    SPEND_EFFECT_KEY_PREFIX,
)
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.project.secrets import resolve_secret


class Clock:
    def now(self):
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read(path, maximum=4 << 20):
    """Bound ordinary host config reads; these files are not a secret store."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
        ):
            raise ValueError("input unavailable")
        raw = source.read(maximum + 1)
        after = os.fstat(source.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(raw) > maximum:
            raise ValueError("input changed")
        return raw


def load_config(path):
    return validate_config(parse_workflow_yaml(_read(Path(path))))


def validate_config(document):
    """One explicit example configuration, using existing public request codecs."""
    config = json.loads(json.dumps(document, allow_nan=False))
    expected = {
        "project_ref",
        "state_root",
        "data",
        "evaluation",
        "provider",
        "bounds",
        "inputs",
    }
    if type(config) is not dict or set(config) != expected:
        raise ValueError("configuration fields invalid")
    root = Path(config["state_root"])
    if not root.is_absolute() or not config["project_ref"]:
        raise ValueError("absolute state_root and project_ref required")
    provider = config["provider"]
    if set(provider) - {
        "backend",
        "credential",
        "account_ref",
        "namespace_ref",
        "thinking_effort",
    } or not {"backend", "credential", "account_ref", "namespace_ref"} <= set(provider):
        raise ValueError("provider accepts references only")
    if provider["backend"] not in {"openrouter", "openai_responses"}:
        raise ValueError("unsupported paid backend")
    if "thinking_effort" in provider and (
        type(provider["thinking_effort"]) is not str
        or provider["thinking_effort"]
        not in {"none", "minimal", "low", "medium", "high", "xhigh"}
    ):
        raise ValueError("invalid provider thinking_effort")
    SecretRef.from_dict(provider["credential"])
    SpendScopeV1(
        provider["backend"], provider["account_ref"], provider["namespace_ref"]
    )
    bounds = config["bounds"]
    if set(bounds) != {
        "max_calls",
        "max_request_bytes",
        "max_response_bytes",
        "max_output_tokens",
        "timeout_seconds",
        "duration_seconds",
    }:
        raise ValueError("unknown dispatch bounds")
    for name, ceiling in (
        ("max_calls", 32),
        ("max_output_tokens", 4096),
        ("max_request_bytes", 64 << 20),
        ("max_response_bytes", 64 << 20),
    ):
        if type(bounds[name]) is not int or not 1 <= bounds[name] <= ceiling:
            raise ValueError("dispatch bound invalid")
    for name in ("timeout_seconds", "duration_seconds"):
        value = bounds[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= 300
        ):
            raise ValueError("time bound invalid")
    BaseLLMClient._http_controls(
        bounds["timeout_seconds"],
        bounds["max_request_bytes"],
        bounds["max_response_bytes"],
    )
    data = config["data"]
    if set(data) != {
        "config_dir",
        "scenarios_dir",
        "rubrics_dir",
        "environment_binding",
        "environment_files",
        "generate",
        "improve",
    }:
        raise ValueError("unknown data fields")
    if set(config["evaluation"]) - {"config_root", "requests", "judge"}:
        raise ValueError("unknown evaluation fields")
    generation = DataRequest.from_dict(data["generate"])
    improvement = DataRequest.from_dict(data["improve"])
    evaluations = [
        EvaluationRequest.from_dict(value) for value in config["evaluation"]["requests"]
    ]
    if (
        generation.mode is not DataMode.GENERATE
        or improvement.mode is not DataMode.IMPROVE
    ):
        raise ValueError("generation and improvement required")
    if sum(t.rows for t in generation.scenarios) != 1:
        raise ValueError("one generation row required")
    if not generation.environment_selection or not improvement.environment_selection:
        raise ValueError("explicit environment selection required")
    if improvement.input_dataset_ref != generation.output_ref:
        raise ValueError("improvement must consume generated bytes unchanged")
    requests = [generation, improvement, *evaluations]
    if len(evaluations) != 2 or len({r.request_id for r in requests}) != 4:
        raise ValueError("four distinct stable request IDs required")
    for request in requests:
        if (
            request.project_ref != config["project_ref"]
            or request.backend != provider["backend"]
        ):
            raise ValueError("request scope mismatch")
    if generation.max_iterations > 2 or improvement.max_iterations > 2:
        raise ValueError("at most two improvement iterations")
    if any(r.case_limit is None or r.case_limit > 2 for r in evaluations):
        raise ValueError("at most two explicit evaluation cases")
    if (
        Path(generation.output_ref).parent != root
        or Path(improvement.output_ref).parent != root
    ):
        raise ValueError("data outputs must be direct private state children")
    if generation.output_ref == improvement.output_ref:
        raise ValueError("outputs must differ")
    EnvironmentBindingV1.from_dict(data["environment_binding"])
    for item in config["inputs"]:
        if (
            set(item) != {"path", "sha256", "size_bytes"}
            or not Path(item["path"]).is_absolute()
        ):
            raise ValueError("explicit input commitment required")
    return config


def _check_inputs(config):
    committed = {str(Path(item["path"])) for item in config["inputs"]}
    required = set(config["data"]["environment_files"].values())
    roots = [
        Path(config["data"][name])
        for name in ("config_dir", "scenarios_dir", "rubrics_dir")
    ]
    roots.append(Path(config["evaluation"]["config_root"]))
    if config["evaluation"].get("judge"):
        roots.append(Path(config["evaluation"]["judge"]["rubrics_dir"]))
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix in {".yaml", ".yml", ".json"}:
                required.add(str(path))
                if len(required) > 256:
                    raise ValueError("input catalog exceeds example bound")
    if not required.issubset(committed):
        raise ValueError("consumed configuration must be declared in inputs")
    for item in config["inputs"]:
        raw = _read(item["path"])
        if (
            len(raw) != item["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != item["sha256"]
        ):
            raise ValueError("input commitment mismatch")


class FileEnvironment:
    def __init__(self, config):
        self.binding = EnvironmentBindingV1.from_dict(config["environment_binding"])
        self.paths = config["environment_files"]

    def resolve(self, selection, *, project_ref):
        if selection != self.binding.selection:
            raise ValueError("environment selection mismatch")
        return self.binding

    def read_dependency(self, ref, *, maximum_bytes):
        return _read(self.paths[ref], maximum_bytes)


class NamedSecrets:
    def resolve(self, reference):
        return resolve_secret(reference, environment=os.environ)


class _BudgetClient:
    def __init__(self, client, factory):
        self._client, self._factory = client, factory

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _call(self, method, *args, **kwargs):
        cap = self._factory.config["bounds"]["max_output_tokens"]
        requested = kwargs.get("max_tokens")
        if requested is not None and (
            type(requested) is not int or not 0 < requested <= cap
        ):
            raise ValueError("output token ceiling exceeded")
        kwargs["max_tokens"] = cap if requested is None else requested
        self._factory.take_call()
        return getattr(self._client, method)(*args, **kwargs)

    def chat(self, *args, **kwargs):
        return self._call("chat", *args, **kwargs)

    def structured_output(self, *args, **kwargs):
        return self._call("structured_output", *args, **kwargs)

    def structured_chat(self, *args, **kwargs):
        return self._call("structured_chat", *args, **kwargs)


class DataClients:
    def __init__(self, config, secrets, deadline):
        self.config, self.secrets, self.deadline = config, secrets, deadline
        self.calls = 0

    def spend_scope(self, *, backend):
        p = self.config["provider"]
        if backend != p["backend"]:
            raise ValueError("backend mismatch")
        return SpendScopeV1(backend, p["account_ref"], p["namespace_ref"])

    def take_call(self):
        if (
            self.calls >= self.config["bounds"]["max_calls"]
            or time.monotonic() >= self.deadline
        ):
            raise ValueError("dispatch budget exhausted")
        self.calls += 1

    def create(self, *, backend, model):
        self.spend_scope(backend=backend)
        key = self.secrets.resolve(
            SecretRef.from_dict(self.config["provider"]["credential"])
        )
        if type(key) is not str or not key.strip():
            raise ValueError("named credential unavailable")
        b = self.config["bounds"]
        kind = OpenRouterClient if backend == "openrouter" else OpenAIResponsesClient
        options = {}
        if "thinking_effort" in self.config["provider"]:
            options["thinking_effort"] = self.config["provider"]["thinking_effort"]
        client = kind(
            key,
            model,
            timeout_seconds=b["timeout_seconds"],
            max_request_bytes=b["max_request_bytes"],
            max_response_bytes=b["max_response_bytes"],
            deadline=self.deadline,
            **options,
        )
        return _BudgetClient(client, self)


@contextmanager
def _apis(
    config,
    store,
    *,
    streams=None,
    clients=None,
    backends=None,
    judge=None,
    read_only=False,
):
    root = Path(config["state_root"])
    streams = store if streams is None else streams
    data = config["data"]
    if clients is None:
        # Pure plan/readback never needs a credential resolver or client constructor.
        class ClosedClients:
            def create(self, **kwargs):
                raise ValueError("client unavailable outside execute")

            def spend_scope(self, *, backend):
                p = config["provider"]
                return SpendScopeV1(backend, p["account_ref"], p["namespace_ref"])

        clients = ClosedClients()
    runtime = lambda *, limits, boundary: WorkerEnvironmentRuntime(
        root / "worker", limits, boundary=boundary
    )
    data_ports = ReferenceDataPortsV1(
        Path(data["config_dir"]),
        Path(data["scenarios_dir"]),
        Path(data["rubrics_dir"]),
        clients,
        FileEnvironment(data),
        CapturedEnvironmentSessionFactory(runtime),
    )
    data_api = DataAPI(
        build_data_operations(
            records=store, streams=streams, clock=Clock(), ports=data_ports
        )
    )
    sink = DirectoryEvaluationArtifactSinkV1(root / "evaluation-artifacts")
    if backends is None:
        backends = (
            {}
            if read_only
            else {
                config["provider"]["backend"]: _paid_backend(
                    config, NamedSecrets(), None
                )
            }
        )
    if judge is None and not read_only and config["evaluation"].get("judge"):
        judge = _paid_judge(config, NamedSecrets(), None)
    evaluation = ReferenceEvaluationPortsV1(
        Path(config["evaluation"]["config_root"]),
        EvaluationBackendRegistryV1(backends),
        sink,
        judge,
        sink,
    )
    eval_api = EvaluationAPI(
        compose_reference_evaluation(
            records=store, streams=streams, clock=Clock(), evaluation=evaluation
        )
    )
    yield data_api, eval_api


def _paid_backend(config, secrets, deadline, clients=None):
    p, b = config["provider"], config["bounds"]

    class Backend(MeteredPaidBackendV1):
        def open(self, model):
            if clients is None:
                raise ValueError("client unavailable outside execute")
            from Evaluator.config import OpenRouterSettings, OpenAIResponsesSettings
            from Evaluator.shared_llm_adapters import (
                SharedOpenRouterAdapter,
                SharedOpenAIResponsesAdapter,
            )

            client = clients.create(backend=p["backend"], model=model.model_ref)
            settings, adapter = (
                (OpenRouterSettings, SharedOpenRouterAdapter)
                if p["backend"] == "openrouter"
                else (OpenAIResponsesSettings, SharedOpenAIResponsesAdapter)
            )
            return adapter(
                settings(model=model.model_ref, max_tokens=b["max_output_tokens"]),
                timeout=b["timeout_seconds"],
                retries=0,
                client=client,
            )

    return Backend(
        p["backend"],
        api_key=SecretRef.from_dict(p["credential"]),
        secrets=secrets,
        account_ref=p["account_ref"],
        namespace_ref=p["namespace_ref"],
        timeout=b["timeout_seconds"],
        retries=0,
        max_request_bytes=b["max_request_bytes"],
        max_response_bytes=b["max_response_bytes"],
        deadline=deadline,
    )


def _paid_judge(config, secrets, deadline, budget=None):
    from shared.judge import RubricLoader, JudgeConfig
    from synaptic_tuner.api.v1.evaluation_facade import EvaluationModelRef

    p, b, j = config["provider"], config["bounds"], config["evaluation"]["judge"]

    class Judge(MeteredPaidJudgeV1):
        def create_client(self):
            client = super().create_client()
            return _BudgetClient(client, budget) if budget is not None else client

    return Judge(
        p["backend"],
        EvaluationModelRef.from_dict(j["model"]),
        api_key=SecretRef.from_dict(p["credential"]),
        secrets=secrets,
        account_ref=p["account_ref"],
        namespace_ref=p["namespace_ref"],
        rubrics=RubricLoader(Path(j["rubrics_dir"])).load_many(j["rubric_refs"]),
        judge_config=JudgeConfig(**j["config"]),
        timeout=b["timeout_seconds"],
        max_request_bytes=b["max_request_bytes"],
        max_response_bytes=b["max_response_bytes"],
        deadline=deadline,
    )


def plan(config, *, backends=None):
    from synaptic_tuner.api.v1.reference.stores import (
        InMemoryDurableRecordStoreV1,
        InMemoryDurableStreamStoreV1,
    )

    _check_inputs(config)

    with _apis(
        config,
        InMemoryDurableRecordStoreV1(),
        streams=InMemoryDurableStreamStoreV1(),
        backends=backends,
    ) as (data, evaluation):
        generation = data.plan(DataRequest.from_dict(config["data"]["generate"]))
        checks = [data.preflight(generation)]
        evaluations = []
        for raw in config["evaluation"]["requests"]:
            _check_inputs(config)
            value = evaluation.plan(EvaluationRequest.from_dict(raw))
            checks.append(evaluation.preflight(value))
            evaluations.append(value.plan_fingerprint)
        return {
            "generation_plan": generation.plan_fingerprint,
            "evaluation_plans": evaluations,
            "ready": all(check.ready for check in checks),
            "improvement": "deferred_until_generated_bytes",
            "request_ids": [
                config["data"][k]["request_id"] for k in ("generate", "improve")
            ]
            + [r["request_id"] for r in config["evaluation"]["requests"]],
        }


def _write(path, document):
    raw = json.dumps(document, allow_nan=False, sort_keys=True).encode()
    if len(raw) > 4 << 20:
        raise ValueError("retained document exceeds readback bound")
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


def _private(root):
    value = root.lstat()
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o700
    ):
        raise ValueError("private state directory required")


def execute(config, *, clients=None, backends=None, judge=None, secrets=None):
    """Create-only attempt. A failed/unknown attempt is inspected, never replayed."""
    _check_inputs(config)
    root = Path(config["state_root"])
    root.mkdir(mode=0o700)  # existing state refuses before clients/effects
    _private(root)
    (root / "worker").mkdir(mode=0o700)
    _write(
        root / "inputs.json", config
    )  # stable IDs/config retained before any model call
    deadline = time.monotonic() + config["bounds"]["duration_seconds"]
    secrets = secrets or NamedSecrets()
    clients = clients or DataClients(config, secrets, deadline)
    if backends is None:
        backends = {
            config["provider"]["backend"]: _paid_backend(
                config, secrets, deadline, clients
            )
        }
    if judge is None and config["evaluation"].get("judge"):
        judge = _paid_judge(config, secrets, deadline, clients)
    with (
        SQLiteDurableStoreV1(root / "workflow.sqlite3") as store,
        _apis(config, store, clients=clients, backends=backends, judge=judge) as (
            data,
            evaluation,
        ),
    ):
        request = DataRequest.from_dict(config["data"]["generate"])
        value = data.plan(request)
        if not data.preflight(value).ready:
            raise ValueError("generation preflight refused")
        _check_inputs(config)
        _write(root / "generate-plan.json", value.to_dict())
        run = data.start(value).run
        outcome = data.show(run)
        if outcome.state is not DataRunState.SUCCEEDED:
            return readback(root)
        raw = _read(request.output_ref, 64 << 20)
        before = hashlib.sha256(raw).hexdigest()
        published = [a for a in outcome.artifacts if a.role == "dataset_jsonl"]
        if (
            len(published) != 1
            or published[0].sha256 != before
            or published[0].size_bytes != len(raw)
        ):
            raise ValueError("generated artifact commitment mismatch")
        checked = data.validate(
            DatasetValidateRequest(config["project_ref"], request.output_ref)
        )
        if checked.findings or checked.rows_valid != 1:
            raise ValueError("generated dataset invalid")
        _check_inputs(config)
        improvement = data.plan(DataRequest.from_dict(config["data"]["improve"]))
        if (
            hashlib.sha256(_read(request.output_ref, 64 << 20)).hexdigest() != before
            or not data.preflight(improvement).ready
        ):
            raise ValueError("generated bytes changed or improvement refused")
        _check_inputs(config)
        _write(root / "improve-plan.json", improvement.to_dict())
        improved = data.start(improvement).run
        if data.show(improved).state is not DataRunState.SUCCEEDED:
            return readback(root)
        checked = data.validate(
            DatasetValidateRequest(
                config["project_ref"], config["data"]["improve"]["output_ref"]
            )
        )
        if checked.findings or checked.rows_valid != 1:
            raise ValueError("improved dataset invalid")
        for index, raw in enumerate(config["evaluation"]["requests"]):
            _check_inputs(config)
            value = evaluation.plan(EvaluationRequest.from_dict(raw))
            if not evaluation.preflight(value).ready:
                raise ValueError("evaluation preflight refused")
            _check_inputs(config)
            expected_run = evaluation_run_ref(value)
            _write(root / f"evaluation-{index}-plan.json", value.to_dict())
            run = evaluation.start(value).run
            if run != expected_run:
                raise ValueError("evaluation start binding mismatch")
            if evaluation.show(run).state is not EvaluationRunState.SUCCEEDED:
                break
    return readback(root)


def readback(root):
    root = Path(root)
    _private(root)
    config = validate_config(json.loads(_read(root / "inputs.json")))
    if Path(config["state_root"]) != root:
        raise ValueError("state binding mismatch")
    with (
        SQLiteDurableStoreV1(root / "workflow.sqlite3", create=False) as store,
        _apis(config, store, read_only=True) as (data, evaluation),
    ):
        data_rows = data.list(DataListRequest(config["project_ref"], limit=10)).outcomes
        eval_rows = evaluation.list(
            EvaluationListRequest(config["project_ref"], limit=10)
        ).outcomes
        results = [
            evaluation.result(EvaluationResultRequest(item.run))
            for item in eval_rows
            if item.state is EvaluationRunState.SUCCEEDED
        ]
        requests = [
            EvaluationRequest.from_dict(raw) for raw in config["evaluation"]["requests"]
        ]
        same_model = requests[0].model == requests[1].model
        paths = [root / f"evaluation-{index}-plan.json" for index in range(2)]
        ordered = []
        if any(path.exists() for path in paths):
            expected_runs = set()
            by_run = {result.run: result for result in results}
            for path, request in zip(paths, requests):
                if not path.exists():
                    continue
                retained = EvaluationPlan.from_dict(json.loads(_read(path)))
                if retained.request != request:
                    raise ValueError("retained evaluation plan mismatch")
                run = evaluation_run_ref(retained)
                expected_runs.add(run)
                result = by_run.get(run)
                if result is not None:
                    if (
                        result.model != request.model
                        or result.backend != request.backend
                        or result.cases_total != retained.cases_total
                    ):
                        raise ValueError("evaluation result binding mismatch")
                    ordered.append(result)
            if any(outcome.run not in expected_runs for outcome in eval_rows):
                raise ValueError("evaluation run has no retained plan binding")
        elif same_model:
            if eval_rows:
                raise ValueError("same-model runs require retained evaluation plans")
        else:
            # Legacy distinct-model attempts had no retained evaluation plans.
            for request in requests:
                matches = [
                    result for result in results if result.model == request.model
                ]
                if len(matches) > 1:
                    raise ValueError("ambiguous retained model comparison")
                ordered.extend(matches)
        results = ordered
        comparison = (
            evaluation.compare(
                EvaluationResultRequest(results[0].run),
                EvaluationResultRequest(results[1].run),
            )
            if len(results) == 2
            else None
        )
        ledger = store.list_page(
            partition=StoragePartition.EFFECTS.value,
            prefix=SPEND_EFFECT_KEY_PREFIX,
            after_key=None,
            limit=10,
        )
        if ledger.truncated:
            raise ValueError("unexpected extra spend effects")
        bound_outputs = {}
        for name in ("generate", "improve"):
            snapshot = root / (name + "-plan.json")
            if not snapshot.exists():
                continue
            retained = DataPlan.from_dict(json.loads(_read(snapshot)))
            if retained.request != DataRequest.from_dict(config["data"][name]):
                raise ValueError("retained data plan mismatch")
            bound_outputs[data_run_ref(retained).run_id] = Path(
                retained.request.output_ref
            )
        for outcome in data_rows:
            path = bound_outputs.get(outcome.run.run_id)
            if path is None:
                raise ValueError("data run has no retained plan binding")
            by_role = {
                "dataset_jsonl": path,
                "dataset_metadata": path.with_name(path.name + ".meta.json"),
            }
            for artifact in outcome.artifacts:
                if artifact.role not in by_role:
                    raise ValueError("unexpected data artifact role")
                member = by_role[artifact.role]
                raw = _read(member, 64 << 20)
                if artifact.sha256 != hashlib.sha256(
                    raw
                ).hexdigest() or artifact.size_bytes != len(raw):
                    raise ValueError("data artifact commitment mismatch")
                if artifact.role == "dataset_jsonl":
                    checked = data.validate(
                        DatasetValidateRequest(config["project_ref"], str(member))
                    )
                    if checked.findings or checked.rows_valid != outcome.rows_written:
                        raise ValueError("retained dataset invalid")
        return {
            "spend": [
                SpendEffectRecordV1.decode(item.canonical).to_dict()
                for item in ledger.records
            ],
            "data": [item.to_dict() for item in data_rows],
            "evaluation": [item.to_dict() for item in eval_rows],
            "results": [item.to_dict() for item in results],
            "comparison": comparison,
            "comparison_scope": "same_model_runs" if same_model else "different_models",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute", "readback"))
    parser.add_argument(
        "path",
        type=Path,
        help="host YAML for plan/execute; private state directory for readback",
    )
    args = parser.parse_args(argv)
    try:
        result = (
            readback(args.path)
            if args.action == "readback"
            else globals()[args.action](load_config(args.path))
        )
    except Exception:
        print(json.dumps({"diagnostic_code": "nontraining_consumer_failed"}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
