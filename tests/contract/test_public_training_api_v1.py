"""The sole public training surface plans and starts generic durable runs."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import planning, training_facade
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef


class Clock:
    def now(self):
        return "2026-08-30T12:00:00Z"


def _case():
    request = v1.TrainingRequest("request-a", "project-a", "{}")
    resolved = v1.ResolvedTrainingRequest(
        "synaptic-resolved-training-request/v1", "request-a", "project-a",
        *(str(i) * 64 for i in range(1, 6)),
    )
    plan = v1.TrainingPlan(
        "synaptic-training-plan/v2", planning.TrainingPlanBasisV1.from_resolved(resolved),
        planning.ProviderPlanRef("a" * 64),
    )
    preflight = v1.TrainingPreflight(
        plan.plan_fingerprint, True, "2026-08-30T11:59:00Z",
        "2026-08-30T12:01:00Z",
    )
    return request, resolved, plan, preflight


def _ports(**overrides) -> HostPorts:
    fields = dict(
        training=object(), runs=object(), artifacts=None, evaluation=None,
        chat=None, data=None, pipelines=None, clock=Clock(),
    )
    fields.update(overrides)
    return HostPorts(**fields)


def test_root_training_exports_have_only_canonical_contract_identities():
    for name in ("TrainingAPI", "TrainingOperations", "TrainingRequest",
                 "TrainingPreflight", "TrainingStart", "AuthorizationRequirement"):
        assert getattr(v1, name) is getattr(training_facade, name)
    assert v1.TrainingPlan is planning.TrainingPlan
    assert v1.ResolvedTrainingRequest is planning.ResolvedTrainingRequest
    assert v1.ProviderRef is ProviderRef
    for removed in ("TrainingSubmission", "TrainingOutcome", "CanonicalDocument",
                    "TrainingRequestResolver", "ResolvedTrainingComponents",
                    "compile_training_plan_v1", "ModalDeploymentReader"):
        assert removed not in v1.__all__
        with pytest.raises(AttributeError):
            getattr(v1, removed)
    with pytest.raises(ModuleNotFoundError):
        __import__("synaptic_tuner.api.v1.training")


def test_public_training_verbs_and_start_signature_are_exact():
    verbs = {name for name, value in v1.TrainingAPI.__dict__.items()
             if not name.startswith("_") and inspect.isfunction(value)}
    assert verbs == {"load", "resolve", "plan", "preflight", "start"}
    assert tuple(inspect.signature(v1.TrainingAPI.start).parameters) == (
        "self", "plan", "preflight",
    )
    assert tuple(HostPorts.__dataclass_fields__) == (
        "training", "runs", "artifacts", "evaluation", "chat", "data",
        "pipelines", "clock",
    )


def test_api_host_delegates_generic_training_without_provider_fields():
    request, resolved, plan, checked = _case()
    provider = ProviderRef("modal", "modal-a10-v1")
    started = v1.TrainingStart(TrainingRunRef("run-a", "project-a"), True)
    calls = []

    class Operations:
        def load(self, value):
            assert value == "{}"
            calls.append("load")
            return request

        def resolve(self, value):
            assert value == request
            calls.append("resolve")
            return resolved

        def plan(self, value, selected):
            assert (value, selected) == (resolved, provider)
            calls.append("plan")
            return plan

        def preflight(self, value):
            assert value == plan
            calls.append("preflight")
            return checked

        def start(self, value, preflight):
            assert (value, preflight) == (plan, checked)
            calls.append("start")
            return started

    host = APIHost(_ports(training=Operations(), runs=object(), clock=Clock()))
    assert type(host.training) is training_facade.TrainingAPI
    loaded = host.training.load("{}")
    compiled = host.training.plan(host.training.resolve(loaded), provider)
    assert host.training.start(compiled, host.training.preflight(compiled)) == started
    assert calls == ["load", "resolve", "plan", "preflight", "start"]


def test_api_host_exposes_one_property_per_family_and_raises_on_uncomposed() -> None:
    from synaptic_tuner.api.v1.artifacts_facade import ArtifactsAPI
    from synaptic_tuner.api.v1.data_facade import DataAPI
    from synaptic_tuner.api.v1.evaluation_facade import EvaluationAPI
    from synaptic_tuner.api.v1.pipelines_facade import PipelinesAPI
    from synaptic_tuner.api.v1.runs_facade import RunsAPI

    families = ("training", "runs", "artifacts", "evaluation", "chat", "data", "pipelines")
    for name in families:
        assert isinstance(inspect.getattr_static(APIHost, name), property), name

    host = APIHost(_ports())
    assert type(host.training) is training_facade.TrainingAPI
    assert type(host.runs) is RunsAPI
    for name in ("artifacts", "evaluation", "chat", "data", "pipelines"):
        with pytest.raises(RuntimeError, match=f"did not compose the {name!r} family"):
            getattr(host, name)

    composed = APIHost(_ports(artifacts=object(), evaluation=object(), data=object()))
    assert type(composed.artifacts) is ArtifactsAPI
    assert type(composed.evaluation) is EvaluationAPI
    assert type(composed.data) is DataAPI
    for name in ("chat", "pipelines"):
        with pytest.raises(RuntimeError):
            getattr(composed, name)
    composed = APIHost(_ports(pipelines=object()))
    assert type(composed.pipelines) is PipelinesAPI


def test_api_host_never_wraps_none_and_rejects_unlanded_families() -> None:
    for name in ("training", "runs", "clock"):
        with pytest.raises(TypeError, match=f"HostPorts.{name} is required"):
            APIHost(_ports(**{name: None}))
    for name in ("chat",):
        with pytest.raises(TypeError, match=f"HostPorts.{name} has no public facade"):
            APIHost(_ports(**{name: object()}))
    with pytest.raises(TypeError, match="exact HostPorts"):
        APIHost(object())
    with pytest.raises(TypeError):
        APIHost(object(), _ports())


@pytest.mark.parametrize("mode", ["wrong_plan", "expired", "not_ready"])
def test_unstartable_preflight_never_reaches_operations(mode):
    _, _, plan, checked = _case()
    changes = {
        "wrong_plan": {"plan_fingerprint": "f" * 64},
        "expired": {"expires_at": "2026-08-30T11:59:30Z"},
        "not_ready": {"ready": False, "diagnostic_codes": ("unavailable",)},
    }[mode]
    from dataclasses import replace
    checked = replace(checked, **changes)

    class Operations:
        def start(self, *args):
            raise AssertionError("invalid preflight crossed the public boundary")

    with pytest.raises(ValueError):
        v1.TrainingAPI(Operations(), clock=Clock()).start(plan, checked)


def test_root_training_symbol_import_is_engine_and_provider_light():
    root = Path(__file__).resolve().parents[2]
    code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from synaptic_tuner.api.v1 import TrainingAPI, TrainingPlan, TrainingRequest, ProviderRef
for prefix in ("tuner", "modal", "sqlite3"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
