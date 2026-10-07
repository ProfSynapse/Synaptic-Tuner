from __future__ import annotations

import json
import os
import os
from pathlib import Path

import pytest

from scripts import probe_vllm_startup as probe


def _configuration():
    return {
        "model": "organization/model", "revision": "a" * 40,
        "expected_vllm_version": "0.26.0",
        "python_executable": "/opt/unsloth-venv/bin/python3",
        "startup": {
            "served_model_name": "probe-model", "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1, "enforce_eager": True, "dtype": "bfloat16",
            "max_model_len": 98304, "max_num_seqs": 3,
            "max_num_batched_tokens": 4096, "language_model_only": True,
            "max_lora_rank": 32, "startup_timeout_seconds": 300,
        },
        "lifetime_seconds": 1200,
    }


def _file(tmp_path, value):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_check_is_provider_free_and_selects_exact_startup_knobs(tmp_path, monkeypatch, capsys):
    path = _file(tmp_path, _configuration())
    def no_provider(*_args, **_kwargs):
        pytest.fail("effect attempted during --check")
    monkeypatch.setattr(probe, "execute", no_provider)
    assert probe.main(["--configuration", str(path), "--check"]) == 0
    assert capsys.readouterr().out.strip() == '{"status":"STARTUP_PROBE_CHECKED"}'
    selected = probe.load_configuration(path)
    spec = probe._startup_spec(selected, "/private/model")
    assert spec.source.model_ref == "/private/model"
    assert spec.max_model_len == 98304
    assert spec.max_num_seqs == 3
    assert spec.max_num_batched_tokens == 4096
    assert spec.dtype == "bfloat16"
    assert spec.gpu_memory_utilization == 0.85
    assert spec.enforce_eager is True


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(revision="floating"),
    lambda c: c.update(model="bad/../model"),
    lambda c: c.update(lifetime_seconds=1801),
    lambda c: c["startup"].update(startup_timeout_seconds=1300),
    lambda c: c["startup"].update(max_model_len=0),
    lambda c: c["startup"].update(enforce_eager=1),
    lambda c: c["startup"].update(private_extra="bad"),
    lambda c: c.update(private_extra="bad"),
])
def test_config_rejects_unbounded_unpinned_or_unknown_values(tmp_path, mutation):
    value = _configuration()
    mutation(value)
    with pytest.raises((TypeError, ValueError)):
        probe.load_configuration(_file(tmp_path, value))


def test_duplicate_keys_and_nonfinite_values_are_denied(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"model":"a","model":"b"}', encoding="utf-8")
    with pytest.raises(ValueError):
        probe.load_configuration(path)
    value = _configuration()
    value["startup"]["gpu_memory_utilization"] = float("nan")
    with pytest.raises(ValueError):
        probe.load_configuration(_file(tmp_path, value))


def test_diagnostic_whitelist_never_retains_untrusted_values():
    from tuner.inference.vllm_runtime import VLLMStartupDiagnostic
    good = VLLMStartupDiagnostic("deadline", "names_mismatch", True, 300.0, 100)
    assert probe._diagnostic(good) == {
        "failure": "deadline", "last_probe": "names_mismatch", "leader_alive": True,
        "elapsed_seconds": 300.0, "probe_count": 100,
    }
    assert probe._diagnostic(VLLMStartupDiagnostic("private", "names_mismatch", True, 300.0, 100)) is None
    assert probe._diagnostic(VLLMStartupDiagnostic("deadline", "private", True, 300.0, 100)) is None
    assert probe._diagnostic(VLLMStartupDiagnostic("deadline", "unknown", None, None, None)) is not None
    assert probe._diagnostic(VLLMStartupDiagnostic(["private"], "unknown", None, None, None)) is None


def test_result_validator_rejects_extra_or_unbounded_data():
    digest = "a" * 64
    value = {
        "schema_version": "synaptic-vllm-startup-probe-result/v1",
        "configuration_sha256": digest,
        "startup_ready": False, "cleanup_resolved": True,
        "preparation_seconds": 1.25, "startup_diagnostic": None,
        "failure_stage": "startup",
        "startup_log_size_bytes": 0, "startup_log_tail_bytes": 0,
        "startup_log_truncated": False,
    }
    assert probe.validate_result(value, configuration_digest=digest) == value
    for mutation in (
        lambda r: r.update(configuration_sha256="b" * 64),
        lambda r: r.update(startup_ready=1),
        lambda r: r.update(preparation_seconds=float("nan")),
        lambda r: r.update(failure_stage="private"),
        lambda r: r.update(raw_error="private"),
        lambda r: r.update(startup_log_tail_bytes=65537),
        lambda r: r.update(startup_diagnostic={"failure": ["private"], "last_probe": "unknown",
                                             "leader_alive": None, "elapsed_seconds": None,
                                             "probe_count": None}),
    ):
        changed = dict(value)
        mutation(changed)
        with pytest.raises(ValueError, match="startup_probe_result_invalid"):
            probe.validate_result(changed, configuration_digest=digest)


def test_effectful_probe_saves_closed_result_once(tmp_path, monkeypatch, capsys):
    path = _file(tmp_path, _configuration())
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    if os.name == "nt":
        monkeypatch.setattr(probe, "_private_output", lambda path: path)
    monkeypatch.setattr(probe.sys, "executable", "/opt/unsloth-venv/bin/python3")
    monkeypatch.setattr(probe.metadata, "version", lambda _name: "0.26.0")
    from tuner.execution.providers.modal import model_snapshot
    monkeypatch.setattr(model_snapshot, "prepare_model_snapshot", lambda **kwargs: kwargs["destination_root"])
    captured = {}
    class Lease:
        def close(self):
            return True
    def start(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return Lease()
    monkeypatch.setattr(probe, "start_vllm_runtime", start)
    assert probe.main(["--configuration", str(path), "--output-directory", str(output)]) == 0
    line = json.loads(capsys.readouterr().out)
    assert line["status"] == "STARTUP_PROBE_SAVED"
    result = line["result"]
    assert result == json.loads((output / "startup-probe-result.json").read_text())
    assert result["startup_ready"] is True and result["cleanup_resolved"] is True
    assert result["failure_stage"] is None
    assert captured["spec"].source.model_ref == str(output / "model-destination")
    assert captured["kwargs"]["environment"]["HF_HUB_OFFLINE"] == "1"
    assert probe.main(["--configuration", str(path), "--output-directory", str(output)]) == 1
    assert capsys.readouterr().out.strip() == '{"status":"STARTUP_PROBE_UNAVAILABLE"}'


def test_private_startup_log_transfer_is_tail_bounded(tmp_path, capfd):
    raw = b"prefix-only" + b"x" * (64 * 1024)
    path = tmp_path / "vllm-startup.log"
    path.write_bytes(raw)
    assert probe._startup_log_summary(path) == (len(raw), 64 * 1024, True)
    probe.emit_private_startup_log_tail(tmp_path)
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err.encode() == b"x" * (64 * 1024)


def test_main_transfers_private_log_exactly_once(tmp_path, monkeypatch, capsys):
    path = _file(tmp_path, _configuration())
    calls = []
    monkeypatch.setattr(probe, "execute", lambda *_args: {
        "startup_ready": True, "cleanup_resolved": True,
    })
    monkeypatch.setattr(probe, "emit_private_startup_log_tail", lambda output: calls.append(output))
    assert probe.main(["--configuration", str(path), "--output-directory", str(tmp_path)]) == 0
    assert calls == [tmp_path]
    assert len(capsys.readouterr().out.splitlines()) == 1
