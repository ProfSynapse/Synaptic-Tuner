"""Bounded real-C-extension regression in the existing locked controller lane."""

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
_PROBE = r"""
import gc
import importlib.metadata as metadata
import json
import platform
import sys
from multidict import CIMultiDict

assert CIMultiDict.__module__ == 'multidict._multidict'
results = {}
for name, operation in (
    ('reflected_union', lambda view, operand: operand | view),
    ('subtraction', lambda view, operand: view - operand),
    ('intersection_control', lambda view, operand: view & operand),
):
    sentinel = object()
    items = CIMultiDict(seed='fixed').items()
    operand = [('key-' + str(i), sentinel) for i in range(32)]
    gc.collect()
    before = sys.getrefcount(sentinel)
    operation(items, operand)
    gc.collect()
    results[name] = sys.getrefcount(sentinel) - before
print(json.dumps({'python': platform.python_version(),
                  'multidict': metadata.version('multidict'),
                  'operand_items': 32, 'reference_deltas': results}))
"""


def test_launcher_multidict_pin_is_reviewed_patch():
    lines = (ROOT / "requirements/modal-launcher-v1.lock").read_text().splitlines()
    selected = [line for line in lines if line.startswith("multidict==")]
    assert selected == [
        "multidict==6.9.1 --hash=sha256:"
        "e8f1e362c9352b50ed120f001046fdbb80810c9d56580f4c3fc13bbe30823387"
    ]


def test_locked_copied_controller_items_views_release_references():
    controller = os.environ.get("SYNAPTIC_TEST_MODAL_SOURCE_WHEEL_PYTHON")
    if controller is None:
        pytest.skip("existing locked CPython 3.11 controller is required")
    result = subprocess.run(
        [controller, "-I", "-B", "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.defpath, "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"},
    )
    assert result.returncode == 0
    assert result.stderr == ""
    value = json.loads(result.stdout)
    assert value["python"] == "3.11.14"
    assert value["multidict"] == "6.9.1"
    assert value["operand_items"] == 32
    assert value["reference_deltas"] == {
        "reflected_union": 0,
        "subtraction": 0,
        "intersection_control": 0,
    }
