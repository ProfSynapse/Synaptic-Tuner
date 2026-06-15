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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from tuner.cloud import RepoCheckoutSpec
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.cloud_extract_handler import (
    CloudExtractHandler,
    ExtractionLaunchPlan,
    _redact_url_userinfo,
)


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
        "repo_url": "https://github.com/prof/tuner.git",
        "repo_branch": "main",
        "repo_commit": "c" * 40,
        "dry_run": False,
        "auto_confirm": False,
        "json": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_handler(args: SimpleNamespace) -> CloudExtractHandler:
    handler = CloudExtractHandler(args=args)
    # Pin repo_root so the (unused on the override path) git helper has a cwd.
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


def test_build_launch_plan_trims_and_populates():
    handler = _make_handler(_full_args(extraction_config="  cfg.yaml  "))
    plan = handler.build_launch_plan()
    assert isinstance(plan, ExtractionLaunchPlan)
    assert plan.extraction_config == "cfg.yaml"
    assert plan.base_model_revision == "a" * 40
    assert plan.adapter_repo_id == "prof/sft-contrast-adapter"
    assert plan.repo.commit == "c" * 40


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
# Repo source resolution (git helper)
# --------------------------------------------------------------------------- #
def test_repo_source_uses_explicit_overrides_without_git():
    handler = _make_handler(_full_args())
    # With all three overrides present, the git helper must never be invoked.
    with patch.object(handler, "_git", side_effect=AssertionError("git should not be called")):
        spec = handler._resolve_repo_source()
    assert spec == RepoCheckoutSpec(
        url="https://github.com/prof/tuner.git",
        branch="main",
        commit="c" * 40,
        clone_dir="/workspace/repo",
    )


def test_repo_source_falls_back_to_git_when_overrides_absent():
    handler = _make_handler(_full_args(repo_url=None, repo_branch=None, repo_commit=None))
    with patch.object(
        handler,
        "_git",
        side_effect=lambda *a: {
            ("config", "--get", "remote.origin.url"): "https://example/repo.git",
            ("rev-parse", "--abbrev-ref", "HEAD"): "feature/x",
            ("rev-parse", "HEAD"): "d" * 40,
        }[a],
    ):
        spec = handler._resolve_repo_source()
    assert spec.url == "https://example/repo.git"
    assert spec.branch == "feature/x"
    assert spec.commit == "d" * 40


def test_repo_source_unresolvable_raises():
    handler = _make_handler(_full_args(repo_url=None, repo_branch=None, repo_commit=None))
    with patch.object(handler, "_git", return_value=""):
        with pytest.raises(CloudProviderError, match="repo source"):
            handler._resolve_repo_source()


def test_explicit_empty_repo_url_fails_closed_not_silent_git_fallback():
    """LOW-4: an explicit --repo-url '' is honored (fails closed), not silently
    routed to the git fallback. `is not None` distinguishes passed-but-empty
    from absent; the empty value then trips the fail-closed guard.
    """
    handler = _make_handler(_full_args(repo_url=""))
    # If the empty override were ignored, _git would be consulted; assert it is
    # NOT, and that the empty url fails closed.
    with patch.object(handler, "_git", side_effect=AssertionError("git fallback must not run for an explicit empty override")):
        with pytest.raises(CloudProviderError, match="repo source"):
            handler._resolve_repo_source()


# --------------------------------------------------------------------------- #
# In-job command assembly
# --------------------------------------------------------------------------- #
def test_extraction_command_passes_every_input_by_id():
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    steps = handler.build_extraction_command(plan)
    joined = " && ".join(steps)
    # Repo checkout pins the exact commit.
    assert ("c" * 40) in joined
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
def test_job_spec_injects_token_as_secret_only():
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    spec = handler.build_job_spec(plan, token="hf_secret_value")
    assert spec.secrets.get("HF_TOKEN") == "hf_secret_value"
    # The token NEVER appears in the command or labels.
    command_text = spec.command[-1]
    assert "hf_secret_value" not in command_text
    assert "hf_secret_value" not in str(spec.labels)


def test_job_spec_dry_run_has_no_secret():
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    spec = handler.build_job_spec(plan, token=None)
    assert spec.secrets == {}


def test_job_spec_labels_are_descriptive_and_id_only():
    handler = _make_handler(_full_args())
    plan = handler.build_launch_plan()
    spec = handler.build_job_spec(plan, token="t")
    assert spec.labels["task"] == "extract"
    assert spec.labels["output_dataset"] == "prof/probe-hidden-states"


# --------------------------------------------------------------------------- #
# Dry-run path (GPU-free, token-free, submit-free)
# --------------------------------------------------------------------------- #
def test_dry_run_does_not_submit_and_needs_no_token(capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    handler = _make_handler(_full_args(dry_run=True))
    with patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls, \
         patch("tuner.handlers.cloud_extract_handler.load_env_file"):
        rc = handler.handle()
    assert rc == 0
    executor_cls.assert_not_called()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "prof/probe-hidden-states" in out


def test_dry_run_json_mode_emits_plan(capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    handler = _make_handler(_full_args(dry_run=True, json=True))
    with patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls, \
         patch("tuner.handlers.cloud_extract_handler.load_env_file"):
        rc = handler.handle()
    assert rc == 0
    executor_cls.assert_not_called()
    out = capsys.readouterr().out
    assert '"dry_run": true' in out
    assert "extraction_runner" in out


# --------------------------------------------------------------------------- #
# SEC-L1: dry-run must not leak credentials embedded in the clone URL
# --------------------------------------------------------------------------- #
def test_redact_url_userinfo_strips_credentials():
    """Unit: _redact_url_userinfo removes user:pass@, leaves clean URLs intact."""
    assert (
        _redact_url_userinfo("https://user:ghp_secrettoken@github.com/prof/tuner.git")
        == "https://github.com/prof/tuner.git"
    )
    # No userinfo -> unchanged; empty -> unchanged.
    assert _redact_url_userinfo("https://github.com/prof/tuner.git") == "https://github.com/prof/tuner.git"
    assert _redact_url_userinfo("") == ""


def test_dry_run_redacts_url_credentials_in_text(capsys, monkeypatch):
    """SEC-L1: a PAT embedded in --repo-url must NOT reach stdout."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secret = "ghp_supersecrettoken"
    handler = _make_handler(_full_args(
        dry_run=True,
        repo_url=f"https://user:{secret}@github.com/prof/tuner.git",
    ))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"):
        rc = handler.handle()
    assert rc == 0
    out = capsys.readouterr().out
    assert secret not in out
    # The redacted host still appears so the operator sees the real target.
    assert "github.com/prof/tuner.git" in out


def test_dry_run_json_redacts_url_credentials_in_command(capsys, monkeypatch):
    """SEC-L1: the JSON-mode 'command' field must NOT carry the embedded PAT."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secret = "ghp_supersecrettoken"
    handler = _make_handler(_full_args(
        dry_run=True,
        json=True,
        repo_url=f"https://user:{secret}@github.com/prof/tuner.git",
    ))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"):
        rc = handler.handle()
    assert rc == 0
    out = capsys.readouterr().out
    assert secret not in out


# --------------------------------------------------------------------------- #
# Submit path
# --------------------------------------------------------------------------- #
def _patch_submit_env(token: Optional[str], hub: Any):
    """Context managers patching token + hub loader + env for the submit path."""
    return (
        patch("tuner.handlers.cloud_extract_handler.load_env_file"),
        patch("tuner.handlers.cloud_extract_handler.get_hf_token", return_value=token),
        patch("tuner.handlers.cloud_extract_handler.load_huggingface_hub", return_value=hub),
    )


def test_submit_requires_token():
    handler = _make_handler(_full_args(auto_confirm=True))
    p_env, p_token, p_hub = _patch_submit_env(token=None, hub=MagicMock())
    with p_env, p_token, p_hub:
        rc = handler.handle()
    assert rc == 1


def test_submit_with_yes_calls_executor():
    handler = _make_handler(_full_args(auto_confirm=True))
    hub = MagicMock()
    submission = SimpleNamespace(job_id="job-xyz", job_url="https://hf.co/jobs/job-xyz")
    p_env, p_token, p_hub = _patch_submit_env(token="hf_tok", hub=hub)
    with p_env, p_token, p_hub, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        executor_cls.return_value.submit.return_value = submission
        rc = handler.handle()
    assert rc == 0
    executor_cls.return_value.submit.assert_called_once()
    spec = executor_cls.return_value.submit.call_args.args[0]
    assert spec.secrets.get("HF_TOKEN") == "hf_tok"
    assert spec.flavor  # populated


def test_submit_hub_load_failure_is_env_error():
    """LOW-2: load_huggingface_hub raising surfaces as ENV_ERROR (rc=1)."""
    handler = _make_handler(_full_args(auto_confirm=True, json=True))
    with patch("tuner.handlers.cloud_extract_handler.load_env_file"), \
         patch("tuner.handlers.cloud_extract_handler.get_hf_token", return_value="hf_tok"), \
         patch(
             "tuner.handlers.cloud_extract_handler.load_huggingface_hub",
             side_effect=CloudProviderError("huggingface_hub missing run_job"),
         ), \
         patch.object(handler, "output_error") as output_error, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    executor_cls.return_value.submit.assert_not_called()
    assert output_error.call_args.kwargs.get("code") == "CLOUD_EXTRACT_ENV_ERROR"


def test_submit_executor_failure_is_submit_error():
    """LOW-3: executor.submit raising (e.g. network/quota) -> SUBMIT_ERROR (rc=1).

    This is the operationally most-likely failure once a real job is launched.
    """
    handler = _make_handler(_full_args(auto_confirm=True, json=True))
    hub = MagicMock()
    p_env, p_token, p_hub = _patch_submit_env(token="hf_tok", hub=hub)
    with p_env, p_token, p_hub, \
         patch.object(handler, "output_error") as output_error, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        executor_cls.return_value.submit.side_effect = CloudProviderError("HF Jobs quota exceeded")
        rc = handler.handle()
    assert rc == 1
    executor_cls.return_value.submit.assert_called_once()
    assert output_error.call_args.kwargs.get("code") == "CLOUD_EXTRACT_SUBMIT_ERROR"


def test_submit_without_yes_prompts_and_cancel_does_not_submit():
    handler = _make_handler(_full_args(auto_confirm=False))
    hub = MagicMock()
    p_env, p_token, p_hub = _patch_submit_env(token="hf_tok", hub=hub)
    with p_env, p_token, p_hub, \
         patch("tuner.handlers.cloud_extract_handler.confirm", return_value=False), \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 0
    executor_cls.return_value.submit.assert_not_called()


def test_submit_json_mode_requires_yes():
    handler = _make_handler(_full_args(json=True, auto_confirm=False))
    hub = MagicMock()
    p_env, p_token, p_hub = _patch_submit_env(token="hf_tok", hub=hub)
    with p_env, p_token, p_hub, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        rc = handler.handle()
    assert rc == 1
    executor_cls.return_value.submit.assert_not_called()


def test_submit_json_mode_with_yes_emits_job_id(capsys):
    handler = _make_handler(_full_args(json=True, auto_confirm=True))
    hub = MagicMock()
    submission = SimpleNamespace(job_id="job-json", job_url=None)
    p_env, p_token, p_hub = _patch_submit_env(token="hf_tok", hub=hub)
    with p_env, p_token, p_hub, \
         patch("tuner.handlers.cloud_extract_handler.HFJobExecutor") as executor_cls:
        executor_cls.return_value.submit.return_value = submission
        rc = handler.handle()
    assert rc == 0
    out = capsys.readouterr().out
    assert "job-json" in out


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
