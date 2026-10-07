"""Router coverage: every parser command must resolve to a handler.

A command that the parser accepts but the router does not route used to fall
through to the interactive menu, which hangs non-interactive and agent callers.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from tuner.cli import router
from tuner.cli.parser import create_parser
from tuner.cli.router import COMMAND_ROUTES, iter_route_targets, route_command
from tuner.project import ProjectContext

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parser_commands() -> list[str]:
    for action in create_parser()._actions:
        if action.dest == "command":
            return list(action.choices)
    raise AssertionError("parser has no 'command' positional")


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.standalone(engine_root=tmp_path)


@pytest.mark.parametrize("command", _parser_commands())
def test_every_parser_command_has_a_route(command):
    assert command in COMMAND_ROUTES, (
        f"'{command}' is in the parser choices but router.COMMAND_ROUTES has no handler "
        "for it. Route it, or remove it from the parser."
    )


def test_every_route_is_a_parser_command():
    stale = set(COMMAND_ROUTES) - set(_parser_commands())
    assert not stale, f"router routes commands the parser rejects: {sorted(stale)}"


@pytest.mark.parametrize(("label", "target"), list(iter_route_targets()))
def test_route_target_names_an_existing_handler(label, target):
    """Handler module exists and defines the named class/function.

    Checked from source so the test does not import optional heavy stacks
    (torch) and stays valid in an environment without them.
    """
    module_name, _, attribute = target.partition(":")
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin, f"{label}: module {module_name} not found"
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert attribute in defined, f"{label}: {module_name} defines no {attribute}"


def test_unrouted_command_is_refused_without_opening_menu(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(COMMAND_ROUTES, "status")

    def menu_must_not_open(*_args, **_kwargs):
        raise AssertionError("interactive menu opened for an unrouted command")

    monkeypatch.setattr(
        "tuner.handlers.main_menu_handler.MainMenuHandler", menu_must_not_open
    )

    args = Namespace(command="status", json=False)
    assert route_command(args, context=_context(tmp_path)) != 0
    assert "no handler" in capsys.readouterr().out

    args = Namespace(command="status", json=True)
    assert route_command(args, context=_context(tmp_path)) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "COMMAND_NOT_ROUTED"


def test_no_command_opens_interactive_menu(tmp_path, monkeypatch):
    seen = []

    class FakeMenu:
        def __init__(self, args, context):
            seen.append((args, context))

        def handle(self):
            return 0

    monkeypatch.setattr("tuner.handlers.main_menu_handler.MainMenuHandler", FakeMenu)
    args = Namespace(command=None, json=False)
    context = _context(tmp_path)

    assert route_command(args, context=context) == 0
    assert seen == [(args, context)]


def test_no_command_in_json_mode_is_an_error(tmp_path, capsys):
    assert route_command(Namespace(command=None, json=True), context=_context(tmp_path)) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "COMMAND_REQUIRED"


def test_missing_handler_dependency_is_reported_for_that_command_only(
    tmp_path, monkeypatch, capsys
):
    def broken(_target):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(router, "_resolve_target", broken)
    args = Namespace(command="eval", json=True)

    assert route_command(args, context=_context(tmp_path)) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "HANDLER_IMPORT_ERROR"
    assert "torch" in error["message"]


def test_doctor_runs_without_torch_and_reports_it_as_a_finding():
    """`doctor` must survive a missing torch and not import heavy handlers."""
    program = (
        "import sys\n"
        "sys.modules['torch'] = None  # makes 'import torch' raise ImportError\n"
        "sys.argv = ['tuner.py', 'doctor', '--json']\n"
        "from tuner.cli.main import main\n"
        "try:\n"
        "    main()\n"
        "finally:\n"
        "    heavy = [m for m in ('tuner.handlers.experiment_handler',\n"
        "                         'tuner.handlers.train_handler',\n"
        "                         'tuner.handlers.main_menu_handler') if m in sys.modules]\n"
        "    sys.stderr.write('HEAVY_IMPORTS=' + ','.join(heavy) + '\\n')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert "Traceback" not in result.stderr, result.stderr
    assert "HEAVY_IMPORTS=\n" in result.stderr, result.stderr
    report = json.loads(result.stdout)
    torch_checks = [
        check
        for section in report["sections"]
        for check in section["checks"]
        if check["name"] == "PyTorch"
    ]
    assert torch_checks and torch_checks[0]["status"] == "fail"
