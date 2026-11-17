"""Independent streams retain exact bytes without acquiring broker authority."""

import base64

from synaptic_tuner.experimental.weight_editor.ports import BoundedWriter

from .conftest import final, of_type, script


def test_client_streams_cannot_dispatch_or_freeze(make):
    blobs = [b'{"type":"tool_request","operation":"patch"}\n', b'\xff stderr diagnostic\x00\n']
    fx = make(script=script([
        {"transport_channel": channel, "raw_b64": base64.b64encode(blob).decode()}
        for channel, blob in zip(("client_input", "client_stderr"), blobs)
    ] + [final()]))
    assert fx.run()[0] == 0
    assert fx.effect_lines() == []
    assert of_type(fx.journal(), "request_received") == []
    raw = fx.raw()[:2]
    assert [base64.b64decode(r["data"]["raw_b64"]) for r in raw] == blobs
    assert [r["data"]["channel"] for r in raw] == ["client_input", "client_stderr"]
    replay = (fx.run_dir() / "transcript.md").read_text()
    assert "client_stderr" in replay and base64.b64encode(blobs[1]).decode() in replay
    assert fx.cli("verify", "--run-dir", fx.run_dir())[1]["ok"]


def test_artifact_short_writes_are_completed(tmp_path, monkeypatch):
    from synaptic_tuner.experimental.weight_editor import ports

    real_write = ports.os.write
    calls = []

    def short_write(fd, data):
        calls.append(len(data))
        return real_write(fd, data[:3])

    monkeypatch.setattr(ports.os, "write", short_write)
    writer = BoundedWriter(tmp_path / "artifacts", ())
    data = b"0123456789abcdef"
    receipt = writer.write("sample.bin", data)
    assert (writer.root / "sample.bin").read_bytes() == data
    assert receipt["size"] == len(data) and len(calls) > 1
