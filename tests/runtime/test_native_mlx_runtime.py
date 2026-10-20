from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from tuner.runtime import native_mlx_runtime as runtime


@pytest.fixture
def observed(monkeypatch, tmp_path):
    executable = tmp_path / "python"
    executable.write_bytes(b"candidate interpreter")
    monkeypatch.setattr(runtime.sys, "executable", str(executable))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.11.14")
    inventory = [
        SimpleNamespace(metadata={"Name": "mlx"}, version="0.32.2"),
        SimpleNamespace(metadata={"Name": "mlx_lm"}, version="0.32.2"),
    ]
    monkeypatch.setattr(runtime.metadata, "distributions", lambda: iter(inventory))
    expected = runtime.MLXRuntimeExpectation.from_dict({
        "python_version": "3.11.14",
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(b"lock bytes").hexdigest(),
        "distributions": {"mlx": "0.32.2", "mlx-lm": "0.32.2"},
    })
    return expected, inventory, executable


def test_candidate_comparison_is_read_only_and_explicitly_not_qualification(observed):
    expected, _, executable = observed
    result = runtime.inspect_mlx_runtime_candidate(expected, dependency_lock=b"lock bytes")
    assert result["status"] == "CANDIDATE_MATCH_ONLY"
    assert result["distributions"] == dict(expected.distributions)
    assert executable.read_bytes() == b"candidate interpreter"


@pytest.mark.parametrize("change", ["extra", "missing", "version", "duplicate", "executable", "python", "platform", "lock"])
def test_runtime_drift_refuses_before_any_framework_import(observed, monkeypatch, change):
    expected, inventory, executable = observed
    lock = b"lock bytes"
    if change == "extra":
        inventory.append(SimpleNamespace(metadata={"Name": "extra"}, version="1.0"))
    elif change == "missing":
        inventory.pop()
    elif change == "version":
        inventory[0].version = "0.32.3"
    elif change == "duplicate":
        inventory.append(SimpleNamespace(metadata={"Name": "MLX.LM"}, version="0.32.2"))
    elif change == "executable":
        executable.write_bytes(b"replacement")
    elif change == "python":
        expected = replace(expected, python_version="3.12.0")
    elif change == "platform":
        monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    elif change == "lock":
        lock = b"other lock"
    with pytest.raises(ValueError):
        runtime.inspect_mlx_runtime_candidate(expected, dependency_lock=lock)


def test_expectation_rejects_partial_pins_and_mutable_inventory(observed):
    expected, _, _ = observed
    with pytest.raises(ValueError):
        replace(expected, distributions=(("mlx", "0.32.2"),))
    with pytest.raises(ValueError):
        replace(expected, distributions=list(expected.distributions))
    with pytest.raises(ValueError):
        replace(expected, executable_sha256="unknown")
