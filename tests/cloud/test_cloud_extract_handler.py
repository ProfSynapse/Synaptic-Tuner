"""GPU-free unit tests for the ``cloud-extract`` verb (Option A).

Location: tests/cloud/test_cloud_extract_handler.py
Purpose: Verify the cloud-extract handler's GPU-free surface -- arg validation,
         launch-plan resolution, in-job command assembly (every input by hub id),
         job-spec assembly (HF_TOKEN as a secret, never echoed), the dry-run path
         (no token / no submit), the submit path (--yes gating, executor call),
         JSON-mode contracts, and the SACROSANCT off-the-signed-path guarantee
         (the module imports nothing under tuner.backends.training.*).
Used by: pytest (tests/ testpaths). No GPU, no network, no real HF_TOKEN.

Design contract: docs/architecture/experiment-runner-probe-dataprep.md section 6.
The actual cloud extraction RUN is deferred + cost-incurring; these tests mock
the hub + executor and never submit a real job.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from tuner.cloud.hf_provisioning import (
    EVIDENCE_SCHEMA_VERSION,
    consume_hf_source_transport,
    prepare_hf_source_transport,
)
from tuner.cloud.runtime_layout import build_runtime_layout
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.cloud_extract_handler import (
    CloudExtractHandler,
    ExtractionLaunchPlan,
)
from tuner.handlers.stages._util import (
    HFSourcePreparation,
    hf_source_preparation_from_consumable,
)
from tuner.project import ProjectContext
from tuner.project.source_bundle import GitSource, RepositoryLocation, SourceLock


_REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingVolume:
    """Minimal installed-client double with inspectable read-only semantics."""

    def __init__(self, **kwargs: Any):
        self.type = kwargs["type"]
        self.source = kwargs["source"]
        self.mount_path = kwargs["mount_path"]
        self.read_only = kwargs.get("read_only")
        self.path = kwargs.get("path")

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "type": self.type,
            "source": self.source,
            "mountPath": self.mount_path,
            "readOnly": self.read_only,
        }
        if self.path is not None:
            value["path"] = self.path
        return value


def _run_job(*, image, command, volumes=None):
    raise AssertionError("feature detection must not invoke run_job")


def _verified_hub() -> SimpleNamespace:
    return SimpleNamespace(Volume=_RecordingVolume, run_job=_run_job)


def _source_lock() -> SourceLock:
    commit = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = GitSource(
        location=RepositoryLocation.parse("https://github.com/test/toolset-training.git"),
        branch="main",
        commit=commit,
        dirty=False,
        pushed=True,
    )
    return SourceLock(
        run_id="hf-extract-test",
        mode="standalone",
        project_source=source,
        engine_source=source,
        project={},
        configuration={},
        created_at="2026-08-19T12:00:00Z",
    )


def _prepared_source_preparation() -> HFSourcePreparation:
    context = ProjectContext.standalone(engine_root=_REPO_ROOT)
    source_lock = _source_lock()
    return HFSourcePreparation(
        source_lock=source_lock,
        source_lock_sha256="d" * 64,
        source_lock_uri="tracking://hf-bootstrap/hf-extract-test/source-lock.json",
        volume_spec=None,
        runtime_layout=build_runtime_layout(context),
        descriptor_uri="tracking://hf-bootstrap/hf-extract-test/source-transport/descriptor.json",
        descriptor_sha256="e" * 64,
        source_transport_state="PREPARED",
    )


def _consumable_source_preparation(tmp_path: Path) -> HFSourcePreparation:
    """Build and consume the real closed descriptor/evidence contract locally."""

    context = ProjectContext.standalone(engine_root=_REPO_ROOT)
    source_lock = _source_lock()
    source_lock_uri = "tracking://hf-bootstrap/hf-extract-test/source-lock.json"
    descriptor_uri = (
        "tracking://hf-bootstrap/hf-extract-test/source-transport/descriptor.json"
    )
    prepared = prepare_hf_source_transport(
        context,
        source_lock=source_lock,
        source_lock_uri=source_lock_uri,
        descriptor_uri=descriptor_uri,
        transport_root=(tmp_path / "source-transport").resolve(),
        volume_source="test-user/bootstrap-bucket",
        path_prefix="synaptic/source-transport",
    )
    descriptor = prepared.descriptor
    volume = descriptor["volume"]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "descriptor": {
            "uri": prepared.descriptor_uri,
            "sha256": prepared.descriptor_sha256,
        },
        "run_id": source_lock.run_id,
        "provider": "hf_jobs",
        "profile": "C",
        "volume": {
            "source": volume["source"],
            "path": volume["path"],
            "type": "bucket",
            "read_only": True,
        },
        "bundle_sha256": descriptor["bundle"]["content_sha256"],
        "capsule_manifest_sha256": descriptor["capsule"]["manifest"]["sha256"],
        "source_lock_sha256": descriptor["source_lock"]["sha256"],
        "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        "status": "provisioned",
        "authority": "protected_workflow",
        "actor": "cloud-extract-test",
        "asserted_at": "2026-08-19T12:00:00Z",
        "provider_receipt_id": "cloud-extract-test-receipt",
    }
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=prepared.descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=evidence,
    )
    return hf_source_preparation_from_consumable(
        consumed,
        runtime_layout=build_runtime_layout(context),
        provisioning_evidence_uri=(
            "tracking://hf-bootstrap/hf-extract-test/source-transport/"
            "provisioning-evidence.json"
        ),
    )


@pytest.fixture(autouse=True)
def _verified_source_fixture(tmp_path: Path, monkeypatch):
    """Default to PREPARED; expose an opt-in real CONSUMABLE fixture."""

    preparation = _prepared_source_preparation()
    preflight = MagicMock(return_value=_source_lock())
    prepare = MagicMock(return_value=preparation)
    monkeypatch.setattr(
        "tuner.handlers.cloud_extract_handler.preflight_hf_source_lock",
        preflight,
    )
    monkeypatch.setattr("tuner.handlers.cloud_extract_handler.prepare_hf_source", prepare)

    def use_consumable() -> HFSourcePreparation:
        consumable = _consumable_source_preparation(tmp_path)
        prepare.return_value = consumable
        return consumable

    return preparation, prepare, use_consumable


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _full_args(**overrides: Any) -> SimpleNamespace:
    """A complete, valid set of cloud-extract args; override individual fields."""
    base: Dict[str, Any] = {
        "extraction_config": "experiment/probe/extraction.yaml",
        "slice_dataset_name": "prof/probe-slice",
        "base_model_name": "Qwen/Qwen3-4B-Instruct",
        "base_model_revision": "a" * 40,
        "adapter_repo_id": "prof/sft-contrast-adapter",
        "adapter_revision": "b" * 40,
        "output_dataset_name": "prof/probe-hidden-states",
        "gpu": None,
        "timeout_hours": None,
        "cloud_image": None,
        "dry_run": False,
        "auto_confirm": False,
        "json": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_handler(args: SimpleNamespace) -> CloudExtractHandler:
    handler = CloudExtractHandler(args=args)
    # Pin the engine root used to resolve the extraction config and cloud config.
    handler._repo_root = Path(".").resolve()
    return handler


# --------------------------------------------------------------------------- #
# Arg validation
# --------------------------------------------------------------------------- #
def test_missing_required_args_reports_all_at_once():
    handler = _make_handler(_full_args(slice_dataset_name="", adapter_repo_id=None))
    with pytest.raises(CloudProviderError) as exc:
        handler.build_launch_plan()
    message = str(exc.value)
    assert "--slice-dataset-name" in message
    assert "--adapter-repo-id" in message
    # The two present-but-empty fields are both surfaced in a single error.
    assert message.count("--") >= 2


def test_blank_whitespace_arg_treated_as_missing():
    handler = _make_handler(_full_args(output_dataset_name="   "))
    with pytest.raises(CloudProviderError, match="--output-dataset-name"):
        handler.build_launch_plan()


def test_build_launch_plan_trims_and_populates(_verified_source_fixture):
    handler = _make_handler(_full_args(extraction_config="  cfg.yaml  "))
    plan = handler.build_launch_plan()
    preparation, prepare, _ = _verified_source_fixture
    assert isinstance(plan, ExtractionLaunchPlan)
    assert plan.extraction_config == "cfg.yaml"
    assert plan.base_model_revision == "a" * 40
    assert plan.adapter_repo_id == "prof/sft-contrast-adapter"
    assert plan.source_preparation is preparation
    assert (
        plan.source_preparation.source_lock.engine_source.commit
        == _source_lock().engine_source.commit
    )
    assert prepare.call_args.kwargs["config_path"] == handler.repo_root / "cfg.yaml"
    assert prepare.call_args.kwargs["runtime"] == {
        "provider": "hf_jobs",
        "task": "extraction",
    }


def test_handle_surfaces_config_error_through_handle_path():
    """LOW-1: handle() wraps a config-validation failure as CONFIG_ERROR (rc=1).

    The validation raises are also covered at build_launch_plan() level, but this
    drives the full handle() try/except wrapper so the CONFIG_ERROR surface is
    not green-by-omission.
    """
    handler = _make_handler(_full_args(slice_dataset_name="", json=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch.object(handler, "output_error") as output_error:
        rc = handler.handle()
    assert rc == 1
    # The uniform error surface tagged the failure with the CONFIG_ERROR code.
    assert output_error.call_args.kwargs.get("code") == "CLOUD_EXTRACT_CONFIG_ERROR"


# --------------------------------------------------------------------------- #
# Defaults + numeric validation
# --------------------------------------------------------------------------- #
def test_flavor_and_timeout_defaults_apply():
    plan = _make_handler(_full_args()).build_launch_plan()
    assert plan.flavor  # a non-empty default
    assert plan.timeout_hours > 0


def test_flavor_and_timeout_overrides_apply():
    plan = _make_handler(_full_args(gpu="h100-large", timeout_hours=5)).build_launch_plan()
    assert plan.flavor == "h100-large"
    assert plan.timeout_hours == 5.0


def test_invalid_timeout_raises():
    with pytest.raises(CloudProviderError, match="timeout-hours"):
        _make_handler(_full_args(timeout_hours="not-a-number")).build_launch_plan()


def test_nonpositive_timeout_raises():
    with pytest.raises(CloudProviderError, match="positive"):
        _make_handler(_full_args(timeout_hours=0)).build_launch_plan()


def test_cloud_image_override_applies():
    plan = _make_handler(_full_args(cloud_image="my/custom:image")).build_launch_plan()
    assert plan.image == "my/custom:image"


# --------------------------------------------------------------------------- #
# Verified source preparation
# --------------------------------------------------------------------------- #
def test_source_preparation_failure_fails_closed_before_volume_or_submit(
    _verified_source_fixture,
):
    handler = _make_handler(_full_args(auto_confirm=True, json=True))
    _, prepare, _ = _verified_source_fixture
    prepare.side_effect = CloudProviderError("source lock is not exact and pushed")
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub") as load_hub, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls, \
         patch.object(handler, "output_error") as output_error:
        rc = handler.handle()
    assert rc == 1
    load_hub.assert_not_called()
    executor_cls.assert_not_called()
    assert output_error.call_args.kwargs["code"] == "CLOUD_EXTRACT_CONFIG_ERROR"


# --------------------------------------------------------------------------- #
# In-job command assembly
# --------------------------------------------------------------------------- #
def test_extraction_command_passes_every_input_by_id(_verified_source_fixture):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    steps = handler.build_extraction_command(plan)
    joined = " && ".join(steps)
    # The bounded byte verifier authenticates the capsule before importing and
    # invoking its sole checkout implementation. Identity verification and the
    # workload must remain downstream of that step.
    assert plan.source_preparation.volume_spec.capsule_manifest_sha256 in steps[0]
    assert steps[0].index("sha256(raw).hexdigest()!=expected") < steps[0].index(
        "spec_from_file_location"
    )
    assert steps[0].index("spec_from_file_location") < steps[0].index(
        "invoke_verified_capsule"
    )
    identity_index = next(i for i, step in enumerate(steps) if "_verify-identities" in step)
    workload_index = next(i for i, step in enumerate(steps) if "extraction_runner" in step)
    assert identity_index > 0
    assert workload_index > identity_index
    # Every artifact is referenced by id/revision; no local path leaks in.
    assert "--slice-dataset-name prof/probe-slice" in joined
    assert "--base-model-name Qwen/Qwen3-4B-Instruct" in joined
    assert f"--base-model-revision {'a' * 40}" in joined
    assert "--adapter-repo-id prof/sft-contrast-adapter" in joined
    assert f"--adapter-revision {'b' * 40}" in joined
    assert "--output-dataset-name prof/probe-hidden-states" in joined
    assert "extraction_runner" in joined


# --------------------------------------------------------------------------- #
# Job-spec assembly + secret handling
# --------------------------------------------------------------------------- #
def test_job_spec_injects_token_as_secret_only(_verified_source_fixture):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    spec = handler.build_job_spec(
        plan, token="hf_secret_value", huggingface_hub=_verified_hub()
    )
    assert spec.secrets.get("HF_TOKEN") == "hf_secret_value"
    # The token NEVER appears in the command or labels.
    command_text = spec.command[-1]
    assert "hf_secret_value" not in command_text
    assert "hf_secret_value" not in str(spec.labels)
    assert len(spec.volumes) == 1
    assert spec.volumes[0].provider_volume.read_only is True
    assert spec.volumes[0].provider_volume.to_dict()["readOnly"] is True


def test_job_spec_dry_run_has_no_secret(_verified_source_fixture):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    spec = handler.build_job_spec(plan, token=None, huggingface_hub=_verified_hub())
    assert spec.secrets == {}


def test_job_spec_labels_are_descriptive_and_id_only(_verified_source_fixture):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    spec = handler.build_job_spec(plan, token="t", huggingface_hub=_verified_hub())
    assert spec.labels["task"] == "extract"
    assert spec.labels["output_dataset"] == "prof/probe-hidden-states"


# --------------------------------------------------------------------------- #
# Dry-run path (GPU-free, token-free, submit-free)
# --------------------------------------------------------------------------- #
def test_dry_run_does_not_submit_and_needs_no_token(
    capsys, monkeypatch, _verified_source_fixture
):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    handler = _make_handler(_full_args(dry_run=True))
    with patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls, \
         patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch(
             "tuner.handlers.cloud_extract_handler.load_huggingface_hub",
             return_value=_verified_hub(),
         ):
        rc = handler.handle()
    assert rc == 1
    executor_cls.assert_not_called()
    out = capsys.readouterr().out
    assert "no approval contract is implemented" in out


def test_dry_run_json_mode_emits_plan(capsys, monkeypatch, _verified_source_fixture):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    handler = _make_handler(_full_args(dry_run=True, json=True))
    with patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls, \
         patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch(
             "tuner.handlers.cloud_extract_handler.load_huggingface_hub",
             return_value=_verified_hub(),
         ):
        rc = handler.handle()
    assert rc == 1
    executor_cls.assert_not_called()
    out = capsys.readouterr().out
    assert "exact-run approval" in out
    assert "extraction_runner" not in out


# --------------------------------------------------------------------------- #
# Volume proof must fail closed before submission
# --------------------------------------------------------------------------- #
def test_dry_run_rejects_unprovable_volume_without_submit(
    monkeypatch, _verified_source_fixture
):
    _, _, use_consumable = _verified_source_fixture
    use_consumable()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    handler = _make_handler(_full_args(dry_run=True, json=True))

    class _DriftedVolume(_RecordingVolume):
        def to_dict(self):
            value = super().to_dict()
            value["readOnly"] = False
            return value

    drifted_hub = SimpleNamespace(Volume=_DriftedVolume, run_job=_run_job)
    plan = handler.build_launch_plan()
    with pytest.raises(CloudProviderError, match="read-only|serialization semantics"):
        handler.build_job_spec(plan, token=None, huggingface_hub=drifted_hub)


# --------------------------------------------------------------------------- #
# Protected submission boundary
# --------------------------------------------------------------------------- #
def test_prepared_submit_stops_before_token_or_provider():
    handler = _make_handler(_full_args(auto_confirm=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file") as load_env, \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token") as get_token, \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub") as load_hub, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    load_env.assert_not_called()
    get_token.assert_not_called()
    load_hub.assert_not_called()
    executor_cls.assert_not_called()


def test_submit_with_yes_cannot_bypass_prepared_state():
    handler = _make_handler(_full_args(auto_confirm=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file") as load_env, \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token") as get_token, \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub") as load_hub, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    load_env.assert_not_called()
    get_token.assert_not_called()
    load_hub.assert_not_called()
    executor_cls.assert_not_called()


def test_prepared_submit_reports_lifecycle_as_env_error():
    handler = _make_handler(_full_args(auto_confirm=True, json=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token") as get_token, \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub") as load_hub, \
         patch.object(handler, "output_error") as output_error, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    get_token.assert_not_called()
    load_hub.assert_not_called()
    executor_cls.assert_not_called()
    assert output_error.call_args.kwargs.get("code") == "CLOUD_EXTRACT_ENV_ERROR"
    assert "exact-run approval" in output_error.call_args.args[0]


def test_prepared_submit_never_reaches_executor_failure_path():
    handler = _make_handler(_full_args(auto_confirm=True, json=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token"), \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub"), \
         patch.object(handler, "output_error") as output_error, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        executor_cls.return_value.submit.side_effect = CloudProviderError("HF Jobs quota exceeded")
        rc = handler.handle()
    assert rc == 1
    executor_cls.assert_not_called()
    assert output_error.call_args.kwargs.get("code") == "CLOUD_EXTRACT_ENV_ERROR"


def test_prepared_submit_stops_before_confirmation_prompt():
    handler = _make_handler(_full_args(auto_confirm=False))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token"), \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub"), \
         patch("tuner.handlers.cloud_extract_handler.confirm") as confirm_mock, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    confirm_mock.assert_not_called()
    executor_cls.assert_not_called()


def test_prepared_json_submit_stops_before_confirmation_gate():
    handler = _make_handler(_full_args(json=True, auto_confirm=False))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token"), \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub"), \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    executor_cls.assert_not_called()


def test_submit_json_mode_with_yes_still_requires_later_provisioning(capsys):
    handler = _make_handler(_full_args(json=True, auto_confirm=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token"), \
         patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub"), \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    executor_cls.assert_not_called()
    out = capsys.readouterr().out
    assert "exact-run approval" in out
    assert "job_id" not in out


@pytest.mark.parametrize("method_name", ["_handle_dry_run", "_handle_submit"])
def test_secure_extract_routes_cross_barrier_before_provider_or_command_helpers(
    method_name: str,
) -> None:
    handler = _make_handler(_full_args(auto_confirm=True))
    plan = MagicMock()
    events: list[str] = []

    def barrier(*, route: str):
        events.append(f"barrier:{route}")
        raise CloudProviderError("authorization stopped")

    with patch(
        "tuner.handlers.cloud_extract_handler.require_current_hf_source_submission_authorization",
        side_effect=barrier,
    ), patch(
        "tuner.handlers.cloud_extract_handler.load_env_file"
    ) as load_env, patch(
        "tuner.handlers.cloud_extract_handler.get_hf_token"
    ) as get_token, patch(
        "tuner.handlers.cloud_extract_handler.load_huggingface_hub"
    ) as load_hub, patch.object(
        handler, "build_job_spec"
    ) as build_job_spec, patch.object(
        handler, "build_extraction_command"
    ) as build_command:
        assert getattr(handler, method_name)(plan) == 1

    assert events == [
        "barrier:cloud-extract.dry-run"
        if method_name == "_handle_dry_run"
        else "barrier:cloud-extract.submit"
    ]
    plan.source_preparation.require_consumable.assert_not_called()
    load_env.assert_not_called()
    get_token.assert_not_called()
    load_hub.assert_not_called()
    build_job_spec.assert_not_called()
    build_command.assert_not_called()


# --------------------------------------------------------------------------- #
# SACROSANCT: off-the-signed-path guarantee (section 6.6)
# --------------------------------------------------------------------------- #
def test_handler_module_imports_no_training_path():
    """Static check: the handler source imports nothing under tuner.backends.training.*.

    This guards the no-pollution boundary by inspecting the module's own import
    statements (AST), so it fails loudly if a future edit reaches across the
    signed-training boundary.
    """
    module_path = Path(__file__).resolve().parents[2] / "tuner" / "handlers" / "cloud_extract_handler.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    offenders = [name for name in imported if "backends.training" in name]
    assert offenders == [], f"cloud-extract must not import training paths: {offenders}"
