from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import diagnose_hf_training_image_metadata as diagnostic


def _argv() -> list[str]:
    return [
        "--image", "docker.io/unsloth/unsloth@sha256:" + "a" * 64,
        "--docker", "docker.exe", "--docker-config", "empty",
    ]


def _exact_interpreter(monkeypatch) -> None:
    monkeypatch.setattr(diagnostic.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(diagnostic.platform, "python_version", lambda: "3.12.7")


def test_cli_has_only_exact_metadata_diagnostic_arguments() -> None:
    args = diagnostic.build_parser().parse_args(_argv())
    assert args.docker == Path("docker.exe")
    assert args.docker_config == Path("empty")
    assert set(vars(args)) == {
        "image", "docker", "docker_config", "stage_attribution",
        "runtime_substage_attribution", "python_runtime_identity",
    }
    assert args.stage_attribution is False
    assert args.runtime_substage_attribution is False
    assert args.python_runtime_identity is False


def test_cli_emits_only_exact_success_line(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    observed = {}

    def diagnose(**kwargs):
        observed.update(kwargs)
        return dict(diagnostic.SUCCESS)

    monkeypatch.setattr(diagnostic, "diagnose_runtime_metadata", diagnose)
    assert diagnostic.main(_argv()) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        '{"schema_version":"synaptic-hf-training-image-metadata-diagnostic/v1",'
        '"status":"PASS"}\n'
    )
    assert observed["runner"] is diagnostic.subprocess_runner
    assert set(observed) == {"image", "docker", "docker_config", "runner"}


def test_attributed_cli_emits_same_exact_success_line(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    observed = {}

    def diagnose(**kwargs):
        observed.update(kwargs)
        return dict(diagnostic.SUCCESS)

    monkeypatch.setattr(diagnostic, "diagnose_runtime_metadata_attributed", diagnose)
    assert diagnostic.main(_argv() + ["--stage-attribution"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        '{"schema_version":"synaptic-hf-training-image-metadata-diagnostic/v1",'
        '"status":"PASS"}\n'
    )
    assert set(observed) == {"image", "docker", "docker_config", "runner"}


def test_attributed_cli_emits_exact_closed_stage_failure(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    secret = "private-attribution-detail"

    def reject(**kwargs):
        error = diagnostic.MetadataDiagnosticStageError(
            failed_stage="runtime_metadata", category="runtime",
        )
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            raise error

    monkeypatch.setattr(diagnostic, "diagnose_runtime_metadata_attributed", reject)
    assert diagnostic.main(_argv() + ["--stage-attribution"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        '{"category":"runtime","failed_stage":"runtime_metadata",'
        '"reason_code":"DIAGNOSTIC_STAGE_REJECTED",'
        '"schema_version":"synaptic-hf-training-image-metadata-diagnostic-stage-error/v1",'
        '"status":"FAILED"}\n'
    )
    assert secret not in captured.err and "Traceback" not in captured.err


def test_default_cli_keeps_valid_stage_error_generic(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    error = diagnostic.MetadataDiagnosticStageError(
        failed_stage="runtime_metadata", category="runtime",
    )
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    assert diagnostic.main(_argv()) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"
    assert "DIAGNOSTIC_STAGE_REJECTED" not in captured.err


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("_failed_stage", "hostile-stage"), ("_category", "hostile-category"),
     ("_failed_stage", ["runtime_metadata"])],
)
def test_attributed_cli_rejects_mutated_stage_error_without_marker(
    monkeypatch, capsys, attribute: str, value: object,
) -> None:
    _exact_interpreter(monkeypatch)
    error = diagnostic.MetadataDiagnosticStageError(
        failed_stage="runtime_metadata", category="runtime",
    )
    object.__setattr__(error, attribute, value)
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata_attributed",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    assert diagnostic.main(_argv() + ["--stage-attribution"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"
    assert "DIAGNOSTIC_STAGE_REJECTED" not in captured.err


def test_attributed_cli_keeps_unexpected_failure_generic(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata_attributed",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private-unexpected")),
    )
    assert diagnostic.main(_argv() + ["--stage-attribution"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": "DIAGNOSTIC_REJECTED",
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic-error/v1",
        "status": "FAILED",
    }
    assert "private" not in captured.err


def test_runtime_substage_cli_emits_exact_closed_failure(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    error = diagnostic.RuntimeSubstageDiagnosticError(runtime_substage="torch_import")
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_substage_attributed",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    assert diagnostic.main(_argv() + ["--runtime-substage-attribution"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        '{"reason_code":"RUNTIME_SUBSTAGE_REJECTED","runtime_substage":"torch_import",'
        '"schema_version":"synaptic-hf-training-runtime-substage-error/v1",'
        '"status":"FAILED"}\n'
    )


def test_runtime_substage_cli_rejects_mutation_and_wrong_mode(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    error = diagnostic.RuntimeSubstageDiagnosticError(runtime_substage="torch_import")
    object.__setattr__(error, "_runtime_substage", "hostile")
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_substage_attributed",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    assert diagnostic.main(_argv() + ["--runtime-substage-attribution"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"
    assert "RUNTIME_SUBSTAGE_REJECTED" not in captured.err

    valid = diagnostic.RuntimeSubstageDiagnosticError(runtime_substage="torch_import")
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata",
        lambda **kwargs: (_ for _ in ()).throw(valid),
    )
    assert diagnostic.main(_argv()) == 125
    captured = capsys.readouterr()
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"
    assert "RUNTIME_SUBSTAGE_REJECTED" not in captured.err


def test_cli_rejects_both_attribution_flags(capsys) -> None:
    assert diagnostic.main(
        _argv() + ["--stage-attribution", "--runtime-substage-attribution"],
    ) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "ARGUMENT_INVALID"


def test_python_runtime_identity_cli_emits_exact_observation(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    observed = {
        "implementation": "CPython",
        "schema_version": "synaptic-hf-training-python-runtime-identity/v1",
        "status": "OBSERVED",
        "version": "3.11.9",
    }
    monkeypatch.setattr(diagnostic, "observe_python_runtime_identity", lambda **kwargs: observed)
    assert diagnostic.main(_argv() + ["--python-runtime-identity"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        '{"implementation":"CPython",'
        '"schema_version":"synaptic-hf-training-python-runtime-identity/v1",'
        '"status":"OBSERVED","version":"3.11.9"}'
    )


def test_python_runtime_identity_cli_emits_exact_unreported_error(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    error = diagnostic.PythonRuntimeIdentityDiagnosticError()
    monkeypatch.setattr(
        diagnostic, "observe_python_runtime_identity",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    assert diagnostic.main(_argv() + ["--python-runtime-identity"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        '{"reason_code":"PYTHON_RUNTIME_IDENTITY_REJECTED",'
        '"schema_version":"synaptic-hf-training-python-runtime-identity-error/v1",'
        '"status":"FAILED"}\n'
    )


@pytest.mark.parametrize("tamper", ["implementation", "version", "extra", "status"])
def test_python_runtime_identity_cli_rejects_hostile_result(
    monkeypatch, capsys, tamper: str,
) -> None:
    _exact_interpreter(monkeypatch)
    observed = {
        "implementation": "CPython",
        "schema_version": "synaptic-hf-training-python-runtime-identity/v1",
        "status": "OBSERVED",
        "version": "3.11.9",
    }
    if tamper == "implementation":
        observed["implementation"] = "Unknown"
    elif tamper == "version":
        observed["version"] = "03.11.9"
    elif tamper == "extra":
        observed["detail"] = "private"
    else:
        observed["status"] = "PASS"
    monkeypatch.setattr(diagnostic, "observe_python_runtime_identity", lambda **kwargs: observed)
    assert diagnostic.main(_argv() + ["--python-runtime-identity"]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"


@pytest.mark.parametrize("other", ["--stage-attribution", "--runtime-substage-attribution"])
def test_python_runtime_identity_flag_is_mutually_exclusive(capsys, other: str) -> None:
    assert diagnostic.main(_argv() + ["--python-runtime-identity", other]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "ARGUMENT_INVALID"


def test_cli_rejects_interpreter_before_diagnostic(monkeypatch, capsys) -> None:
    monkeypatch.setattr(diagnostic.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(diagnostic.platform, "python_version", lambda: "3.12.8")
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )
    assert diagnostic.main(_argv()) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": "INTERPRETER_INVALID",
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic-error/v1",
        "status": "FAILED",
    }


def test_cli_rejects_arguments_without_usage_or_values(capsys) -> None:
    secret = "private-path-secret"
    assert diagnostic.main(["--unknown", secret]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "ARGUMENT_INVALID"
    assert secret not in captured.err and "usage:" not in captured.err.lower()


def test_cli_closes_diagnostic_exception_without_detail(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    secret = "private-runtime-detail"
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    assert diagnostic.main(_argv()) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"
    assert secret not in captured.err and "Traceback" not in captured.err


def test_cli_rejects_noncanonical_success_result(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata", lambda **kwargs: {"status": "PASS"},
    )
    assert diagnostic.main(_argv()) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "DIAGNOSTIC_REJECTED"


def test_cli_propagates_cancellation_without_json_or_child_output(monkeypatch, capsys) -> None:
    _exact_interpreter(monkeypatch)
    cancellation = KeyboardInterrupt("private-child-output")
    monkeypatch.setattr(
        diagnostic, "diagnose_runtime_metadata",
        lambda **kwargs: (_ for _ in ()).throw(cancellation),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        diagnostic.main(_argv())
    assert caught.value is cancellation
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_script_help_runs_isolated_from_any_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(diagnostic.REPO_ROOT / "scripts" / Path(diagnostic.__file__).name), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower() and completed.stderr == ""


def test_copied_script_fails_closed_root_authentication(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    copied = scripts / "diagnose_hf_training_image_metadata.py"
    shutil.copyfile(diagnostic.REPO_ROOT / "scripts" / copied.name, copied)
    completed = subprocess.run(
        [sys.executable, "-I", str(copied), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 125 and completed.stdout == ""
    assert json.loads(completed.stderr) == {
        "reason_code": "SCRIPT_IDENTITY_INVALID",
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic-error/v1",
        "status": "FAILED",
    }
