"""Shared harness: every scenario goes through the public CLI in a subprocess.

Outputs live under ``scratch/`` (never /tmp) and are removed unless a test fails.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from synaptic_tuner.experimental.weight_editor.fakes import write_fixture

REPO = Path(__file__).resolve().parents[3]
PKG = "synaptic_tuner.experimental.weight_editor"
FAKES = f"{PKG}.fakes:build_ports"


@pytest.fixture
def workdir(request):
    root = REPO / "scratch" / "test-weight-editor" / f"{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True)
    yield root
    failed = getattr(request.node, "rep_call", None) is not None and request.node.rep_call.failed
    if not failed:
        shutil.rmtree(root, ignore_errors=True)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    setattr(item, "rep_" + outcome.get_result().when, outcome.get_result())


def _env():
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    return env


class Fixture:
    def __init__(self, workdir: Path, **kw):
        self.paths = write_fixture(workdir / "fx", **kw)
        self.config = self.paths["config"]
        self.effects = self.paths["effects"]
        self.runs = self.paths["runs"]
        self.capsule = self.paths["capsule"]

    # ---- CLI
    def ports_args(self, **extra):
        out = ["--ports", FAKES, "--ports-arg", f"script={self.paths['script']}",
               "--ports-arg", f"effects_dir={self.effects}"]
        for k, v in extra.items():
            out += ["--ports-arg", f"{k}={v}"]
        return out

    def cli(self, *args, check_json=True):
        proc = subprocess.run([sys.executable, "-m", PKG, *map(str, args)], cwd=REPO, env=_env(),
                              capture_output=True, text=True, timeout=120)
        try:
            data = json.loads(proc.stdout) if proc.stdout.strip() else None
        except json.JSONDecodeError:
            data = None
        return proc.returncode, data, proc

    def run(self, run_id="r1", config=None, **ports):
        return self.cli("run", "--config", config or self.config, "--run-id", run_id, *self.ports_args(**ports))

    def resume(self, run_id="r1", config=None, extra=(), **ports):
        return self.cli("resume", "--config", config or self.config, "--run-id", run_id, *extra,
                        *self.ports_args(**ports))

    def popen(self, verb, run_id="r1", **ports):
        return subprocess.Popen(
            [sys.executable, "-m", PKG, verb, "--config", str(self.config), "--run-id", run_id,
             *map(str, self.ports_args(**ports))],
            cwd=REPO, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_marker(self, stage, proc, timeout=60):
        marker = self.effects / f"paused-{stage}"
        end = time.time() + timeout
        while time.time() < end:
            if marker.exists():
                return
            if proc.poll() is not None:
                raise AssertionError(f"process exited early: {proc.communicate()}")
            time.sleep(0.05)
        raise AssertionError("pause marker never appeared")

    def sigkill(self, proc):
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
        assert proc.returncode == -signal.SIGKILL

    # ---- inspection
    def run_dir(self, run_id="r1"):
        return self.runs / run_id

    def journal(self, run_id="r1"):
        return read_jsonl(self.run_dir(run_id) / "journal" / "events.jsonl")

    def raw(self, run_id="r1"):
        return read_jsonl(self.run_dir(run_id) / "provider" / "events.jsonl")

    def effect_lines(self):
        f = self.effects / "effects.log"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def types(journal):
    return [r["type"] for r in journal]


def of_type(journal, t):
    return [r for r in journal if r["type"] == t]


@pytest.fixture
def make(workdir):
    count = {"n": 0}

    def factory(**kw):
        count["n"] += 1
        return Fixture(workdir / f"case{count['n']}", **kw)

    return factory


# ---- script builders
def req(op, **arguments):
    return {"type": "tool_request", "operation": op, "arguments": arguments}


def final(choice="no_edit", proposal=None, rationale="synthetic justification"):
    d = {"type": "final_choice", "choice": choice, "rationale": rationale}
    if proposal:
        d["proposal"] = proposal
    return d


def script(*turns):
    return {"turns": [{"events": t} if isinstance(t, list) else t for t in turns]}


QUERY12 = req("query_development", items=["dev-001", "dev-002"])
OK_PATCH = json.dumps({"tensor": "w.layer0", "deltas": [0.1, -0.1, 0.05]})
BAD_PATCH = json.dumps({"tensor": "w.layer0", "deltas": [0.1, 9.0, 0.05]})


def patch(name, text=OK_PATCH, tensor="w.layer0"):
    return req("propose_patch", name=name, tensor=tensor, patch=text, justification=f"synthetic reason for {name}")
