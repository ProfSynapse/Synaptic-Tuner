"""Regression coverage for import-light cloud command boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_ROOTS = {
    "boto3",
    "botocore",
    "huggingface_hub",
    "modal",
    "runpod",
    "transformers",
}
IMPORT_LIGHT_MODULES = (
    "tuner.cloud",
    "tuner.cloud.hf_jobs",
    "tuner.backends",
    "tuner.backends.training",
    "tuner.backends.training.cloud",
    "tuner.backends.registry",
    "tuner.handlers.stages",
    "Evaluator",
    "shared.experiment_tracking",
    "tuner.handlers.cloud_train_handler",
    "tuner.handlers.cloud_pipeline_handler",
    "tuner.handlers.cloud_run_handler",
)


def _run_fresh(code: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "IMPORT_BOUNDARY_RESULT="
    line = next(line for line in result.stdout.splitlines() if line.startswith(marker))
    return json.loads(line.removeprefix(marker))


def _blocked_import_probe(module_name: str, *, registry_lists: bool = False) -> dict:
    post_import = ""
    if registry_lists:
        post_import = (
            "from tuner.backends.registry import TrainingBackendRegistry,EvaluationBackendRegistry\n"
            "lists=[TrainingBackendRegistry.list(),EvaluationBackendRegistry.list()]\n"
        )
    else:
        post_import = "lists=None\n"
    code = f"""
import importlib
import json
import sys

roots = {sorted(PROVIDER_ROOTS)!r}
attempted = []

class ProviderBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in roots:
            attempted.append(fullname)
            raise ImportError(f'provider import blocked: {{fullname}}')
        return None

sys.meta_path.insert(0, ProviderBlocker())
importlib.import_module({module_name!r})
{post_import}
loaded = sorted(name for name in sys.modules if name.split('.')[0] in roots)
print('IMPORT_BOUNDARY_RESULT=' + json.dumps({{'attempted': attempted, 'loaded': loaded, 'lists': lists}}))
"""
    return _run_fresh(code)


@pytest.mark.parametrize("module_name", IMPORT_LIGHT_MODULES)
def test_package_and_secure_handler_imports_do_not_touch_provider_sdks(module_name: str):
    result = _blocked_import_probe(module_name)
    assert result == {"attempted": [], "loaded": [], "lists": None}


def test_registry_list_is_ordered_metadata_only_in_a_fresh_process():
    result = _blocked_import_probe("tuner.backends.registry", registry_lists=True)
    assert result["attempted"] == []
    assert result["loaded"] == []
    assert result["lists"] == [
        ["rtx", "mac", "hf_jobs", "modal", "runpod"],
        ["ollama", "lmstudio", "llamacpp", "unsloth", "mlc"],
    ]


def test_registry_resolves_and_caches_only_the_selected_string_target(monkeypatch):
    from tuner.backends import registry

    class SelectedBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    calls = []
    resolved_before = registry.TrainingBackendRegistry._resolved.copy()
    registry.TrainingBackendRegistry._resolved.pop("hf_jobs", None)

    def resolve(target: str):
        calls.append(target)
        return SelectedBackend

    monkeypatch.setattr(registry, "_resolve_target", resolve)
    try:
        first = registry.TrainingBackendRegistry.get("hf_jobs", marker=1)
        second = registry.TrainingBackendRegistry.get("hf_jobs", marker=2)
    finally:
        registry.TrainingBackendRegistry._resolved.clear()
        registry.TrainingBackendRegistry._resolved.update(resolved_before)

    assert calls == ["tuner.backends.training.cloud.hf_jobs_backend:HFJobsBackend"]
    assert first.kwargs == {"marker": 1}
    assert second.kwargs == {"marker": 2}


def test_dynamic_registry_replacement_preserves_order_and_clears_cache():
    from tuner.backends.registry import TrainingBackendRegistry

    class ReplacementBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    backends_before = TrainingBackendRegistry._backends.copy()
    resolved_before = TrainingBackendRegistry._resolved.copy()
    TrainingBackendRegistry._resolved["rtx"] = object
    try:
        original_order = TrainingBackendRegistry.list()
        TrainingBackendRegistry.register("rtx", ReplacementBackend)
        assert TrainingBackendRegistry.list() == original_order
        assert "rtx" not in TrainingBackendRegistry._resolved
        assert isinstance(TrainingBackendRegistry.get("rtx"), ReplacementBackend)

        TrainingBackendRegistry.register("custom", ReplacementBackend)
        assert TrainingBackendRegistry.list() == [*original_order, "custom"]
    finally:
        TrainingBackendRegistry._backends.clear()
        TrainingBackendRegistry._backends.update(backends_before)
        TrainingBackendRegistry._resolved.clear()
        TrainingBackendRegistry._resolved.update(resolved_before)


def test_lazy_facades_preserve_direct_export_identity_and_all_contracts():
    import Evaluator
    import shared.experiment_tracking as tracking
    import tuner.backends as backends
    import tuner.backends.training as training
    import tuner.backends.training.cloud as cloud
    import tuner.handlers.stages as stages
    from Evaluator.enums import BackendType
    from shared.experiment_tracking.local_tracker import LocalTracker
    from shared.experiment_tracking.per_example_loss import load_losses, save_losses
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.backends.training.cloud.modal_backend import ModalBackend
    from tuner.backends.training.cloud.runpod_backend import RunPodBackend
    from tuner.backends.training.rtx_backend import RTXBackend
    from tuner.handlers.stages.hf_training_stage import HFTrainingStageRunner

    assert backends.RTXBackend is RTXBackend
    assert training.RTXBackend is RTXBackend
    assert cloud.HFJobsBackend is HFJobsBackend
    assert stages.HFTrainingStageRunner is HFTrainingStageRunner
    assert Evaluator.BackendType is BackendType
    assert tracking.LocalTracker is LocalTracker
    assert tracking.save_losses is save_losses
    assert tracking.load_losses is load_losses

    assert cloud.AVAILABLE_BACKENDS == {
        "hf_jobs": HFJobsBackend,
        "modal": ModalBackend,
        "runpod": RunPodBackend,
    }
    assert training.AVAILABLE_BACKENDS is cloud.AVAILABLE_BACKENDS
    assert "save_losses" not in tracking.__all__
    assert "load_losses" not in tracking.__all__
    assert "save_experiment" not in tracking.__all__
    assert not hasattr(tracking, "save_experiment")


def test_facade_all_values_remain_exact_and_ordered():
    import Evaluator
    import shared.experiment_tracking as tracking
    import tuner.backends as backends
    import tuner.backends.training as training
    import tuner.backends.training.cloud as cloud
    import tuner.handlers.stages as stages

    assert backends.__all__ == [
        "ITrainingBackend", "RTXBackend", "MacBackend",
        "IEvaluationBackend", "OllamaBackend", "LMStudioBackend",
        "TrainingBackendRegistry", "EvaluationBackendRegistry",
    ]
    assert training.__all__ == [
        "ITrainingBackend", "RTXBackend", "MacBackend", "HFJobsBackend",
        "ModalBackend", "RunPodBackend", "AVAILABLE_BACKENDS",
    ]
    assert cloud.__all__ == [
        "AVAILABLE_BACKENDS", "HFJobsBackend", "ModalBackend", "RunPodBackend",
    ]
    assert stages.__all__ == [
        "HFEvalStageRunner", "HFLossStageRunner", "HFTrainingStageRunner",
    ]
    assert Evaluator.__all__ == [
        "BackendClient", "BackendError", "BackendResponse", "BackendSettings",
        "BackendType", "ResponseType", "ToolCallFormat", "ValidationLevel",
        "create_client", "create_client_from_args", "create_settings",
        "get_supported_backends", "BaseBackendSettings", "EvaluatorConfig",
        "LMStudioSettings", "OllamaSettings", "PromptFilter", "expand_path",
        "parse_tags", "PromptCase", "filter_prompts", "load_prompt_cases",
        "ValidationResult", "validate_assistant_response", "CorrectnessResult",
        "evaluate_correctness", "RubricValidator", "RubricValidationResult",
        "FullValidationResult", "validate_response", "ParsedResponse",
        "ParsedToolCall", "parse_response", "EvaluationRecord", "evaluate_cases",
        "aggregate_stats", "build_run_payload", "console_summary",
        "render_markdown", "write_json",
    ]
    assert tracking.__all__ == [
        "ExperimentTracker", "ExperimentOrchestrator", "ExecutionStageSpec",
        "ExperimentSpec", "Experiment", "LossResult", "LocalTracker", "RunFilter",
        "RunRecord", "RunRegistry", "StageResult", "TrackingService",
        "compute_per_example_losses", "create_tracker", "create_experiment",
        "load_experiment", "load_experiment_spec", "write_analysis_bundle",
    ]
