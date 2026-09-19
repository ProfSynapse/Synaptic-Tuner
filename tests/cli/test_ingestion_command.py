from __future__ import annotations

import json
import importlib
import os
from argparse import Namespace
from pathlib import Path

import pytest

from tuner.cli.parser import create_parser
from tuner.cli.main import build_project_context, main as cli_main
from tuner.cli.router import route_command
from tuner.handlers.ingestion_handler import _selection_roots
from tuner.ingestion.runtime_v1 import ProcessLocalIngestionOperationsV1
from tuner.project import ProjectContext


def _config(*, pattern: str = "**/*.md") -> str:
    return json.dumps(
        {
            "schema_version": "synaptic-ingestion-cli/v1",
            "project_ref": "project-1",
            "admission_request_id": "admit-1",
            "request_id": "request-1",
            "discovery": {
                "include": ["**/*"],
                "exclude": [],
                "include_hidden": False,
            },
            "structure": {
                "name": "MarkdownNote",
                "version": "1",
                "frontmatter_mode": "optional",
                "fields": [
                    {
                        "name": "body",
                        "selector": {"kind": "document_body", "key": None},
                        "value_kind": "string",
                        "required": True,
                    },
                    {
                        "name": "title",
                        "selector": {
                            "kind": "frontmatter_field",
                            "key": "title",
                        },
                        "value_kind": "string",
                        "required": False,
                    },
                ],
                "text_projections": [{"name": "text", "field_ref": "body"}],
                "metadata": [],
            },
            "binding": {"binding_id": "markdown", "pattern": pattern},
        },
        separators=(",", ":"),
    )


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.standalone(
        engine_root=tmp_path, invocation_cwd=tmp_path
    )


def _write_config(tmp_path: Path, content: str | None = None) -> Path:
    path = tmp_path / "ingestion.json"
    path.write_text(_config() if content is None else content, encoding="utf-8")
    return path


def _args(config: Path | str, selection: str, *, json_mode: bool = True) -> Namespace:
    return create_parser().parse_args(
        [
            "ingest",
            "--config",
            str(config),
            "--select",
            selection,
            *(["--json"] if json_mode else []),
        ]
    )


def test_ingest_help_and_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parser = create_parser()
    help_text = parser.format_help()
    config = _write_config(tmp_path)
    parsed = _args(config, "notes=note.md")
    handled: list[tuple[str, Path]] = []

    def fake_handle(self) -> int:
        handled.append((self.name, self.context.tracking_root))
        return 7

    monkeypatch.setattr(
        "tuner.handlers.ingestion_handler.IngestionHandler.handle", fake_handle
    )

    assert "ingest" in help_text
    assert "--select ALIAS=PATH" in help_text
    assert "--config <config.json>" in help_text
    assert parsed.command == "ingest"
    assert parsed.ml_config == str(config)
    assert parsed.ingestion_selections == ["notes=note.md"]
    assert route_command(parsed, context=_context(tmp_path)) == 7
    assert handled == [("ingest", (tmp_path / ".tracking").resolve())]


def test_ingestion_config_participates_in_project_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "synaptic.yaml").write_text(
        """schema_version: synaptic-project/v1
project:
  id: ingestion-host
  name: Ingestion Host
engine:
  requires: ">=1.0,<2.0"
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
    config = _write_config(project)
    monkeypatch.chdir(invocation)
    engine = tmp_path / "engine"
    engine.mkdir()
    context = build_project_context(
        _args(config, "notes=note.md"), engine_root=engine
    )

    assert context.mode == "host"
    assert context.project_root == project.resolve()


def test_ingest_success_is_bounded_and_leak_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "selected-private"
    source.mkdir()
    secret_content = "Body content must remain private"
    (source / "note.md").write_text(
        f"---\ntitle: Example\n---\n{secret_content}\n", encoding="utf-8"
    )
    config = _write_config(tmp_path)

    exit_code = route_command(
        _args(config, f"notes={source}"), context=_context(tmp_path)
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert payload["success"] is True
    assert payload["status"] == "verified"
    assert payload["source_count"] == 1
    assert payload["documents_written"] == 1
    assert payload["diagnostic_codes"] == []
    assert str(source) not in captured.out
    assert secret_content not in captured.out
    assert "title" not in captured.out


def test_ingest_normalizes_ordinary_lexical_root_without_resolving_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "selected"
    source.mkdir()
    (source / "note.md").write_text("Body\n", encoding="utf-8")
    config = _write_config(tmp_path)

    exit_code = route_command(
        _args(config, f"notes={source.parent / 'unused' / '..' / source.name}"),
        context=_context(tmp_path),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "verified"
    assert captured.err == ""


def test_selection_root_absolutization_never_resolves_the_selected_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_resolve(*args, **kwargs):
        raise AssertionError("selected roots must not be resolved by the CLI")

    monkeypatch.setattr(Path, "resolve", reject_resolve)
    roots = _selection_roots(["notes=parent/../selected"], tmp_path)

    assert roots[0].path == Path(
        os.path.abspath(os.fspath(tmp_path / "parent" / ".." / "selected"))
    )


@pytest.mark.parametrize("root_kind", ["directory", "file"])
def test_ingest_rejects_link_selected_as_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    root_kind: str,
) -> None:
    if root_kind == "directory":
        target = tmp_path / "private-target"
        target.mkdir()
        (target / "note.md").write_text("Private target body\n", encoding="utf-8")
        alias = "notes"
    else:
        target = tmp_path / "private-target.md"
        target.write_text("Private target body\n", encoding="utf-8")
        alias = "note.md"
    link = tmp_path / f"selected-{root_kind}"
    try:
        os.symlink(target, link, target_is_directory=root_kind == "directory")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable to the test process")
    config = _write_config(tmp_path)

    assert route_command(
        _args(config, f"{alias}={link}"), context=_context(tmp_path)
    ) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "success": False,
        "status": "failed",
        "error_code": "source_unsafe",
    }
    assert captured.err == ""
    assert str(target) not in captured.out
    assert "Private target body" not in captured.out


@pytest.mark.parametrize(
    ("content", "selection"),
    [
        ('{"schema_version":"synaptic-ingestion-cli/v1","schema_version":"x"}', "notes=note.md"),
        (_config(), "missing-separator"),
        (_config(), "notes="),
        (_config()[:-1] + ',"unknown":true}', "notes=note.md"),
        (_config().replace('"include_hidden":false', '"include_hidden":NaN'), "notes=note.md"),
    ],
)
def test_ingest_rejects_invalid_config_and_selection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    selection: str,
) -> None:
    config = _write_config(tmp_path, content)
    assert route_command(_args(config, selection), context=_context(tmp_path)) == 2
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "success": False,
        "status": "failed",
        "error_code": "invalid_input",
    }
    assert str(config) not in output
    assert content not in output


def test_ingest_reports_preview_blockers_without_starting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mixed"
    source.mkdir()
    (source / "note.md").write_text("Body\n", encoding="utf-8")
    (source / "ignored.txt").write_text("Private ignored content\n", encoding="utf-8")
    started = False
    original_start = ProcessLocalIngestionOperationsV1.start

    def record_start(self, plan, preflight):
        nonlocal started
        started = True
        return original_start(self, plan, preflight)

    monkeypatch.setattr(ProcessLocalIngestionOperationsV1, "start", record_start)
    config = _write_config(tmp_path)

    assert route_command(
        _args(config, f"notes={source}"), context=_context(tmp_path)
    ) == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert started is False
    assert payload["status"] == "blocked"
    assert payload["error_code"] == "preflight_blocked"
    assert payload["diagnostic_codes"] == ["unmatched_source"]
    assert payload["unmatched_sources"] == 1
    assert str(source) not in output
    assert "Private ignored content" not in output


def test_ingest_sanitizes_unexpected_operation_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "private-source"
    source.mkdir()
    (source / "note.md").write_text("secret document content\n", encoding="utf-8")

    def fail_authorization(self, project_ref, roots, policy):
        raise RuntimeError(f"host failure at {source}: secret document content")

    monkeypatch.setattr(
        ProcessLocalIngestionOperationsV1,
        "authorize_local_selection",
        fail_authorization,
    )
    config = _write_config(tmp_path)

    assert route_command(
        _args(config, f"notes={source}"), context=_context(tmp_path)
    ) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "success": False,
        "status": "failed",
        "error_code": "ingestion_failed",
    }
    assert str(source) not in output
    assert "secret document content" not in output


def test_ingest_requires_json_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_config(tmp_path)
    assert route_command(
        _args(config, "notes=note.md", json_mode=False),
        context=_context(tmp_path),
    ) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "invalid_input"


@pytest.mark.parametrize("kind", ["missing", "oversized"])
def test_main_rejects_unreadable_or_oversized_config_without_leakage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(tmp_path))
    source = tmp_path / "source-private"
    source.mkdir()
    (source / "note.md").write_text("private body\n", encoding="utf-8")
    config = tmp_path / "private-config.json"
    if kind == "oversized":
        config.write_bytes(b"{" + b"x" * 1_048_576)

    with pytest.raises(SystemExit) as stopped:
        cli_main([
            "ingest", "--config", str(config), "--select", f"notes={source}", "--json",
        ])

    captured = capsys.readouterr()
    assert stopped.value.code == 2
    assert json.loads(captured.out) == {
        "success": False,
        "status": "failed",
        "error_code": "invalid_input",
    }
    assert captured.err == ""
    assert str(config) not in captured.out
    assert str(source) not in captured.out
    assert "private body" not in captured.out


def test_main_sanitizes_bootstrap_project_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(tmp_path))
    config = _write_config(tmp_path)
    missing_env = tmp_path / "secret" / "missing.env"

    with pytest.raises(SystemExit) as stopped:
        cli_main([
            "ingest", "--config", str(config), "--select", "notes=source",
            "--env-file", str(missing_env), "--json",
        ])

    captured = capsys.readouterr()
    assert stopped.value.code == 1
    assert json.loads(captured.out) == {
        "success": False,
        "status": "failed",
        "error_code": "bootstrap_failed",
    }
    assert captured.err == ""
    assert str(missing_env) not in captured.out
    assert str(config) not in captured.out


def test_main_sanitizes_unexpected_bootstrap_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path)
    private_detail = str(tmp_path / "sensitive-bootstrap-path")

    def fail_context(*args, **kwargs):
        raise RuntimeError(private_detail)

    main_module = importlib.import_module("tuner.cli.main")
    monkeypatch.setattr(main_module, "build_project_context", fail_context)
    with pytest.raises(SystemExit) as stopped:
        cli_main([
            "ingest", "--config", str(config), "--select", "notes=source", "--json",
        ])

    captured = capsys.readouterr()
    assert stopped.value.code == 1
    assert json.loads(captured.out)["error_code"] == "bootstrap_failed"
    assert captured.err == ""
    assert private_detail not in captured.out
    assert str(config) not in captured.out


def test_main_success_emits_one_json_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(tmp_path))
    source = tmp_path / "private-source"
    source.mkdir()
    secret = "content visible only inside the bundle"
    (source / "note.md").write_text(secret + "\n", encoding="utf-8")
    config = _write_config(tmp_path)

    with pytest.raises(SystemExit) as stopped:
        cli_main([
            "ingest", "--config", str(config), "--select", f"notes={source}", "--json",
        ])

    captured = capsys.readouterr()
    assert stopped.value.code == 0
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out)["status"] == "verified"
    assert captured.err == ""
    assert str(config) not in captured.out
    assert str(source) not in captured.out
    assert secret not in captured.out
