import json
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

import synaptic_tuner
import tuner
from shared.utilities.env import load_env_file, redact_env_value
from tuner.cli.main import build_project_context, main as cli_main
from tuner.cli.parser import create_parser
from tuner.cli.router import route_command
from tuner.project import ProjectContext


def test_capabilities_parser_supports_list_describe_and_json_placement():
    parser = create_parser()

    listed = parser.parse_args(["capabilities", "list", "--json"])
    described = parser.parse_args(
        ["--json", "capabilities", "describe", "mechinterp.steer"]
    )

    assert (listed.command, listed.subcommand, listed.capability_id, listed.json) == (
        "capabilities",
        "list",
        None,
        True,
    )
    assert (
        described.command,
        described.subcommand,
        described.capability_id,
        described.json,
    ) == ("capabilities", "describe", "mechinterp.steer", True)


def test_tuner_legacy_exports_resolve_lazily_and_preserve_normal_errors(tmp_path):
    engine_root = Path(__file__).parents[2].resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(engine_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    probe = """
import json
import sys
import tuner

before = sorted(name for name in sys.modules if name.startswith("tuner.core."))
training_config = tuner.TrainingConfig
after_config = sorted(name for name in sys.modules if name.startswith("tuner.core."))
backend_error = tuner.BackendError
after_error = sorted(name for name in sys.modules if name.startswith("tuner.core."))
try:
    tuner.not_a_public_export
except AttributeError as exc:
    error = str(exc)
else:
    raise AssertionError("unknown package attributes must raise AttributeError")
print(json.dumps({
    "before": before,
    "after_config": after_config,
    "after_error": after_error,
    "training_config_module": training_config.__module__,
    "backend_error_module": backend_error.__module__,
    "all_contains_exports": all(
        name in tuner.__all__ for name in ("TrainingConfig", "BackendError")
    ),
    "dir_contains_exports": all(
        name in dir(tuner) for name in ("TrainingConfig", "BackendError")
    ),
    "normal_error": "not_a_public_export" in error,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["before"] == []
    expected_core_modules = {
        "tuner.core.config",
        "tuner.core.exceptions",
        "tuner.core.interfaces",
    }
    assert set(payload["after_config"]) == expected_core_modules
    assert set(payload["after_error"]) == expected_core_modules
    assert payload["training_config_module"] == "tuner.core.config"
    assert payload["backend_error_module"] == "tuner.core.exceptions"
    assert payload["all_contains_exports"] is True
    assert payload["dir_contains_exports"] is True
    assert payload["normal_error"] is True


def test_handler_package_exports_are_lazy_and_isolated(tmp_path):
    engine_root = Path(__file__).parents[2].resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(engine_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    probe = """
import json
import sys
import tuner.handlers as handlers

before = sorted(name for name in sys.modules if name.startswith("tuner.handlers."))
status_handler = handlers.StatusHandler
after = sorted(name for name in sys.modules if name.startswith("tuner.handlers."))
try:
    handlers.not_a_handler
except AttributeError as exc:
    error = str(exc)
else:
    raise AssertionError("unknown handler attributes must raise AttributeError")
print(json.dumps({
    "before": before,
    "after": after,
    "module": status_handler.__module__,
    "all_contains_export": "StatusHandler" in handlers.__all__,
    "dir_contains_export": "StatusHandler" in dir(handlers),
    "normal_error": "not_a_handler" in error,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["before"] == []
    assert "tuner.handlers.status_handler" in payload["after"]
    assert not any(
        name in payload["after"]
        for name in (
            "tuner.handlers.train_handler",
            "tuner.handlers.eval_handler",
            "tuner.handlers.ml_handler",
            "tuner.handlers.cloud_train_handler",
        )
    )
    assert payload["module"] == "tuner.handlers.status_handler"
    assert payload["all_contains_export"] is True
    assert payload["dir_contains_export"] is True
    assert payload["normal_error"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["capabilities"],
        ["capabilities", "unknown"],
        ["capabilities", "list", "mechinterp.steer"],
        ["capabilities", "describe"],
        ["list", "datasets", "unexpected"],
    ],
)
def test_capabilities_parser_rejects_invalid_positional_combinations(argv):
    with pytest.raises(SystemExit) as exc_info:
        create_parser().parse_args(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("arguments", "expected_capability"),
    [
        (["capabilities", "list", "--json"], "capabilities.list"),
        (
            ["capabilities", "describe", "mechinterp.steer", "--json"],
            "capabilities.describe",
        ),
    ],
)
def test_synaptic_capability_discovery_is_import_light_from_unrelated_cwd(
    tmp_path, arguments, expected_capability
):
    engine_root = Path(__file__).parents[2].resolve()
    module_log = tmp_path / "modules.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(engine_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["SYNAPTIC_TEST_MODULE_LOG"] = str(module_log)
    probe = (
        "import json, os, sys; from pathlib import Path; "
        "from tuner.cli.main import main; code = 0\n"
        "try: main()\n"
        "except SystemExit as exc: code = exc.code\n"
        "Path(os.environ['SYNAPTIC_TEST_MODULE_LOG']).write_text("
        "json.dumps(sorted(sys.modules)), encoding='utf-8')\n"
        "raise SystemExit(code)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe, *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "synaptic-result/v1"
    assert payload["success"] is True
    assert payload["capability"] == expected_capability
    imported = set(json.loads(module_log.read_text(encoding="utf-8")))
    forbidden = {
        "torch",
        "transformers",
        "huggingface_hub",
        "modal",
        "runpod",
    }
    assert not (forbidden & imported)


def test_parser_supports_yes_alias_for_auto_confirm():
    parser = create_parser()

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/recipes/example.yaml", "--yes"])
    assert args.auto_confirm is True

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/recipes/example.yaml", "--auto-confirm"])
    assert args.auto_confirm is True

    args = parser.parse_args(["local-run", "--job-config", "Trainers/recipes/example.yaml", "--yes"])
    assert args.command == "local-run"
    assert args.auto_confirm is True


def test_run_experiment_stage_selection_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "run-experiment",
            "--experiment-spec",
            "Trainers/cloud/experiments/example.yaml",
            "--only-stage",
            "evaluation",
            "--skip-stage",
            "loss",
            "--skip-stage",
            "analysis",
        ]
    )

    assert args.command == "run-experiment"
    assert args.experiment_spec == "Trainers/cloud/experiments/example.yaml"
    assert args.only_stage == "evaluation"
    assert args.skip_stage == ["loss", "analysis"]


def test_analyze_experiment_command_parses():
    parser = create_parser()

    args = parser.parse_args(["analyze-experiment", "--experiment-id", "latest", "--json"])

    assert args.command == "analyze-experiment"
    assert args.experiment_id == "latest"
    assert args.json is True


def test_plan_hardware_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "plan-hardware",
            "--experiment-spec",
            "Trainers/cloud/experiments/example.yaml",
            "--optimize-for",
            "cost",
            "--max-hourly-price",
            "1.50",
        ]
    )

    assert args.command == "plan-hardware"
    assert args.experiment_spec == "Trainers/cloud/experiments/example.yaml"
    assert args.optimize_for == "cost"
    assert args.max_hourly_price == 1.50


def test_run_experiment_auto_hardware_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "run-experiment",
            "--experiment-spec",
            "Trainers/cloud/experiments/example.yaml",
            "--auto-hardware",
            "--optimize-for",
            "balanced",
        ]
    )

    assert args.command == "run-experiment"
    assert args.auto_hardware is True
    assert args.optimize_for == "balanced"


def test_cloud_training_lora_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "cloud-pipeline",
            "--method",
            "sft",
            "--train-lora-r",
            "128",
            "--train-lora-alpha",
            "256",
            "--train-lora-dropout",
            "0.05",
            "--train-use-dora",
            "--train-use-rslora",
            "--train-init-lora-weights",
            "loftq",
            "--train-lora-target-modules",
            "all-linear",
        ]
    )

    assert args.train_lora_r == 128
    assert args.train_lora_alpha == 256
    assert args.train_lora_dropout == 0.05
    assert args.train_use_dora is True
    assert args.train_use_rslora is True
    assert args.train_init_lora_weights == "loftq"
    assert args.train_lora_target_modules == "all-linear"


def test_cloud_training_evolutionary_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "cloud-pipeline",
            "--method",
            "sft",
            "--train-evolutionary-enabled",
            "--train-evolutionary-candidates",
            "4",
            "--train-evolutionary-eval-batch-size",
            "2",
            "--train-evolutionary-validation-config",
            "configs/fitness/tool_calling.yaml",
            "--train-evolutionary-strategy",
            "antithetic_noise",
            "--train-evolutionary-noise-scale",
            "0.03",
            "--train-evolutionary-max-grad-norm",
            "1.0",
            "--train-evolutionary-scale-factors",
            "0.5,1.0,1.5",
            "--train-evolutionary-selection-method",
            "best",
            "--train-evolutionary-min-improvement",
            "0.01",
            "--train-evolutionary-min-relative-improvement",
            "0.0001",
            "--train-evolutionary-noise-floor-epsilon",
            "0.000001",
            "--train-evolutionary-eval-frequency",
            "5",
            "--train-evolutionary-warmup-steps",
            "200",
            "--train-evolutionary-no-log-candidates",
        ]
    )

    assert args.train_evolutionary_enabled is True
    assert args.train_evolutionary_candidates == 4
    assert args.train_evolutionary_eval_batch_size == 2
    assert args.train_evolutionary_validation_config == "configs/fitness/tool_calling.yaml"
    assert args.train_evolutionary_strategy == "antithetic_noise"
    assert args.train_evolutionary_noise_scale == 0.03
    assert args.train_evolutionary_max_grad_norm == 1.0
    assert args.train_evolutionary_scale_factors == "0.5,1.0,1.5"
    assert args.train_evolutionary_selection_method == "best"
    assert args.train_evolutionary_min_improvement == 0.01
    assert args.train_evolutionary_min_relative_improvement == 0.0001
    assert args.train_evolutionary_noise_floor_epsilon == 0.000001
    assert args.train_evolutionary_eval_frequency == 5
    assert args.train_evolutionary_warmup_steps == 200
    assert args.train_evolutionary_log_candidates is False


def test_cloud_method_flag_accepts_grpo():
    parser = create_parser()

    args = parser.parse_args(["cloud-pipeline", "--method", "grpo"])

    assert args.command == "cloud-pipeline"
    assert args.method == "grpo"


def test_cloud_eval_timeout_flag_parses():
    parser = create_parser()

    args = parser.parse_args(["cloud-pipeline", "--eval-timeout-hours", "7.5"])

    assert args.command == "cloud-pipeline"
    assert args.eval_timeout_hours == 7.5


def test_bucket_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "bucket",
            "read",
            "--path",
            "runs/hf_jobs/sft/example/logs/training_latest.jsonl",
            "--jsonl-latest",
            "--pretty",
        ]
    )

    assert args.command == "bucket"
    assert args.subcommand == "read"
    assert args.path == "runs/hf_jobs/sft/example/logs/training_latest.jsonl"
    assert args.jsonl_latest is True
    assert args.pretty is True


def test_bucket_pull_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "bucket",
            "pull",
            "--path",
            "runs/hf_jobs/sft/example/analysis/loss",
            "--dest",
            ".",
        ]
    )

    assert args.command == "bucket"
    assert args.subcommand == "pull"
    assert args.path == "runs/hf_jobs/sft/example/analysis/loss"
    assert args.dest == "."


def test_bucket_push_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "bucket",
            "push",
            "--path",
            "local/results.json",
            "--dest",
            "runs/manual_uploads/",
        ]
    )

    assert args.command == "bucket"
    assert args.subcommand == "push"
    assert args.path == "local/results.json"
    assert args.dest == "runs/manual_uploads/"


def test_mechinterp_run_config_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "mechinterp",
            "run",
            "--config",
            "MechInterp/configs/pipeline.yaml",
            "--provider",
            "modal",
            "--from-step",
            "extract",
            "--skip-step",
            "fit",
            "--i-know-this-runs-on-gpu",
            "--yes",
        ]
    )

    assert args.command == "mechinterp"
    assert args.subcommand == "run"
    assert args.ml_config == "MechInterp/configs/pipeline.yaml"
    assert args.provider == "modal"
    assert args.from_step == "extract"
    assert args.skip_step == ["fit"]
    assert args.i_know_this_runs_on_gpu is True
    assert args.auto_confirm is True


def test_flywheel_export_fixtures_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "flywheel",
            "export-fixtures",
            "--export-config",
            "configs/flywheel/evaluator_fixture_export.example.yaml",
            "--output",
            "Evaluator/config/scenarios/flywheel_frozen.yaml",
            "--dry-run",
            "--yes",
        ]
    )

    assert args.command == "flywheel"
    assert args.subcommand == "export-fixtures"
    assert args.export_config == "configs/flywheel/evaluator_fixture_export.example.yaml"
    assert args.output == "Evaluator/config/scenarios/flywheel_frozen.yaml"
    assert args.dry_run is True
    assert args.auto_confirm is True


def test_project_command_and_context_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "project",
            "validate",
            "--project-root",
            "host",
            "--manifest",
            "host/synaptic.yaml",
            "--env-file",
            "host/.env",
            "--profile",
            "smoke",
            "--source-mode",
            "superproject",
            "--events",
            "jsonl",
            "--json",
        ]
    )

    assert args.command == "project"
    assert args.subcommand == "validate"
    assert args.project_root == "host"
    assert args.manifest == "host/synaptic.yaml"
    assert args.env_file == "host/.env"
    assert args.profile == "smoke"
    assert args.source_mode == "superproject"
    assert args.events == "jsonl"
    assert args.json is True


def test_legacy_command_defaults_remain_unchanged():
    parser = create_parser()

    args = parser.parse_args(["local-run", "--job-config", "recipe.yaml", "--yes"])

    assert args.command == "local-run"
    assert args.project_root is None
    assert args.manifest is None
    assert args.env_file is None
    assert args.profile is None
    assert args.events is None
    assert args.source_mode is None


def test_build_project_context_resolves_manifest_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "host"
    project_root.mkdir()
    (project_root / "synaptic.yaml").write_text(
        """schema_version: synaptic-project/v1
project:
  id: parser-host
  name: Parser Host
engine:
  requires: \">=1.0,<2.0\"
  api: v1
paths:
  configs: project://experiments
  artifacts: project://.synaptic/artifacts
  state: project://.synaptic/state
  tracking: project://.synaptic/tracking
  cache: project://.synaptic/cache
  tmp: project://.synaptic/tmp
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    args = create_parser().parse_args(
        ["project", "inspect", "--project-root", str(project_root), "--json"]
    )

    context = build_project_context(args, engine_root=Path(__file__).parents[2])

    assert context.mode == "host"
    assert context.project_root == project_root.resolve()
    assert context.artifact_root == (project_root / ".synaptic" / "artifacts").resolve()


def test_context_env_loading_preserves_process_precedence(tmp_path, monkeypatch):
    project_root = tmp_path / "host"
    project_root.mkdir()
    (project_root / ".env").write_text(
        "NODE_D_EXISTING=from-file\nNODE_D_NEW=from-file\n", encoding="utf-8"
    )
    monkeypatch.setenv("NODE_D_EXISTING", "from-process")
    monkeypatch.delenv("NODE_D_NEW", raising=False)
    context = ProjectContext.host(engine_root=tmp_path / "engine", project_root=project_root)

    assert load_env_file(context=context) is True
    assert __import__("os").environ["NODE_D_EXISTING"] == "from-process"
    assert __import__("os").environ["NODE_D_NEW"] == "from-file"


def test_environment_redaction_never_reveals_secret_prefixes():
    assert redact_env_value("HF_TOKEN", "hf_example-secret") == "<redacted>"
    assert redact_env_value("OPENROUTER_API_KEY", "sk-or-example") == "<redacted>"
    assert redact_env_value("LMSTUDIO_HOST", "localhost") == "localhost"


def test_explicit_env_file_has_selection_priority(tmp_path, monkeypatch):
    project_root = tmp_path / "host"
    project_root.mkdir()
    (project_root / ".env").write_text("NODE_D_ENV_SOURCE=host\n", encoding="utf-8")
    explicit = tmp_path / "selected.env"
    explicit.write_text("NODE_D_ENV_SOURCE=explicit\n", encoding="utf-8")
    monkeypatch.delenv("NODE_D_ENV_SOURCE", raising=False)
    context = ProjectContext.host(engine_root=tmp_path / "engine", project_root=project_root)

    assert load_env_file(context=context, explicit_path=explicit) is True
    assert os.environ["NODE_D_ENV_SOURCE"] == "explicit"


def test_process_environment_beats_explicit_env_file(tmp_path, monkeypatch):
    explicit = tmp_path / "selected.env"
    explicit.write_text("NODE_D_ENV_PROCESS=file\n", encoding="utf-8")
    monkeypatch.setenv("NODE_D_ENV_PROCESS", "process")

    assert load_env_file(explicit_path=explicit) is True
    assert os.environ["NODE_D_ENV_PROCESS"] == "process"


def test_explicit_python_engine_root_beats_process_override(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit-engine"
    process = tmp_path / "process-engine"
    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(process))
    monkeypatch.chdir(tmp_path)
    args = create_parser().parse_args(["project", "inspect"])

    context = build_project_context(args, engine_root=explicit)

    assert context.engine_root == explicit.resolve()


def test_process_engine_root_beats_module_fallback(tmp_path, monkeypatch):
    process = tmp_path / "process-engine"
    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(process))
    monkeypatch.chdir(tmp_path)
    args = create_parser().parse_args(["project", "inspect"])

    context = build_project_context(args)

    assert context.engine_root == process.resolve()


def test_dotenv_engine_root_cannot_retroactively_change_context(tmp_path, monkeypatch):
    dotenv_engine = tmp_path / "dotenv-engine"
    env_file = tmp_path / "selected.env"
    env_file.write_text(
        f"SYNAPTIC_ENGINE_ROOT={dotenv_engine}\n", encoding="utf-8"
    )
    monkeypatch.delenv("SYNAPTIC_ENGINE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    args = create_parser().parse_args(["project", "inspect"])

    context = build_project_context(args, engine_root=tmp_path / "explicit-engine")
    try:
        assert load_env_file(context=context, explicit_path=env_file) is True
        assert context.engine_root == (tmp_path / "explicit-engine").resolve()
        assert os.environ["SYNAPTIC_ENGINE_ROOT"] == str(dotenv_engine)
    finally:
        os.environ.pop("SYNAPTIC_ENGINE_ROOT", None)


def test_missing_explicit_env_file_fails_before_routing(
    tmp_path, monkeypatch, capsys
):
    routed = False

    def fail_if_routed(*args, **kwargs):
        nonlocal routed
        routed = True
        raise AssertionError("router must not run when explicit env file is missing")

    cli_main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(cli_main_module, "route_command", fail_if_routed)
    missing = tmp_path / "missing.env"

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["project", "inspect", "--env-file", str(missing), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert routed is False
    assert payload["error"]["code"] == "ENV_FILE_NOT_FOUND"
    assert payload["error"]["details"]["path"] == str(missing.resolve())


def test_explicit_env_file_without_dotenv_support_fails_before_routing(
    tmp_path, monkeypatch, capsys
):
    routed = False

    def fail_if_routed(*args, **kwargs):
        nonlocal routed
        routed = True
        raise AssertionError("router must not run without explicit dotenv support")

    env_file = tmp_path / "selected.env"
    env_file.write_text("NODE_D_SUPPORT_TEST=loaded\n", encoding="utf-8")
    cli_main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(cli_main_module, "route_command", fail_if_routed)
    monkeypatch.setitem(sys.modules, "dotenv", None)

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["project", "inspect", "--env-file", str(env_file), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert routed is False
    assert payload["error"] == {
        "code": "ENV_FILE_SUPPORT_UNAVAILABLE",
        "message": "Explicit env file requires python-dotenv support",
        "details": {
            "path": str(env_file.resolve()),
            "dependency": "python-dotenv",
        },
    }


def test_implicit_env_loading_preserves_no_dotenv_compatibility(
    tmp_path, monkeypatch
):
    (tmp_path / ".env").write_text("NODE_D_IMPLICIT_TEST=loaded\n", encoding="utf-8")
    context = ProjectContext.standalone(engine_root=tmp_path)
    monkeypatch.setitem(sys.modules, "dotenv", None)

    assert load_env_file(context=context) is False


def test_explicit_env_file_with_installed_support_loads_and_routes(
    tmp_path, monkeypatch
):
    env_file = tmp_path / "selected.env"
    env_file.write_text(
        "NODE_D_INSTALLED_EXISTING=file\nNODE_D_INSTALLED_NEW=loaded\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NODE_D_INSTALLED_EXISTING", "process")
    monkeypatch.delenv("NODE_D_INSTALLED_NEW", raising=False)
    routed = []

    def record_route(args, *, context):
        routed.append(context.engine_root)
        return 0

    cli_main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(cli_main_module, "route_command", record_route)

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["project", "inspect", "--env-file", str(env_file), "--json"])

    assert exc_info.value.code == 0
    assert routed == [Path(__file__).parents[2].resolve()]
    assert os.environ["NODE_D_INSTALLED_EXISTING"] == "process"
    assert os.environ["NODE_D_INSTALLED_NEW"] == "loaded"


def _write_requirement_manifest(project_root: Path, requirement: str) -> None:
    (project_root / "synaptic.yaml").write_text(
        f"""schema_version: synaptic-project/v1
project:
  id: requirement-host
  name: Requirement Host
engine:
  requires: \"{requirement}\"
  api: v1
""",
        encoding="utf-8",
    )


def test_compatible_engine_requirement_routes_handler(tmp_path, monkeypatch):
    project_root = tmp_path / "compatible-host"
    project_root.mkdir()
    _write_requirement_manifest(project_root, ">=1.1,<2")
    routed = []

    def record_route(args, *, context):
        routed.append((args.command, context.project_root))
        return 0

    cli_main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(cli_main_module, "route_command", record_route)

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["project", "inspect", "--project-root", str(project_root), "--json"])

    assert exc_info.value.code == 0
    assert routed == [("project", project_root.resolve())]


def test_incompatible_engine_requirement_fails_before_routing(
    tmp_path, monkeypatch, capsys
):
    project_root = tmp_path / "incompatible-host"
    project_root.mkdir()
    _write_requirement_manifest(project_root, "<1.1")
    routed = False

    def fail_if_routed(*args, **kwargs):
        nonlocal routed
        routed = True
        raise AssertionError("router must not run for an incompatible engine")

    cli_main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(cli_main_module, "route_command", fail_if_routed)

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["project", "inspect", "--project-root", str(project_root), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert routed is False
    assert payload["error"]["code"] == "PROJECT_MANIFEST_INVALID"
    assert payload["error"]["details"] == {
        "reason": "engine_version_incompatible",
        "requires": "<1.1",
        "engine_version": "1.1.0",
    }


def test_legacy_and_public_runtime_versions_share_canonical_1_1_0():
    assert synaptic_tuner.__version__ == "1.1.0"
    assert tuner.__version__ == synaptic_tuner.__version__


def test_project_migrate_dry_run_is_json_and_side_effect_free(tmp_path, capsys):
    project_root = tmp_path / "legacy-host"
    project_root.mkdir()
    args = create_parser().parse_args(
        ["project", "migrate-dry-run", "--project-root", str(project_root), "--json"]
    )
    context = ProjectContext.host(
        engine_root=Path(__file__).parents[2], project_root=project_root
    )

    assert route_command(args, context=context) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["dry_run"] is True
    assert payload["data"]["writes_performed"] is False
    assert not (project_root / "synaptic.yaml").exists()
    assert not (project_root / ".synaptic").exists()
