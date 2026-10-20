"""Host trust boundary tests; no ML package import or Mac execution."""

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import subprocess
import tempfile

import pytest

from examples.training_qualification import native_host
from tuner.runtime.native_mlx_material import canonical
from tuner.training.native_mlx_planning import NativeRestrictedAdmissionV1


def test_external_root_signature_is_required_and_binds_exact_admission(tmp_path):
    private = tmp_path / "root-private.pem"
    public = tmp_path / "root-public.pem"
    signature = tmp_path / "admission.sig"
    subprocess.run(["openssl", "genpkey", "-algorithm", "EC", "-pkeyopt",
                    "ec_paramgen_curve:P-256", "-out", str(private)], check=True,
                   capture_output=True)
    subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                   check=True, capture_output=True)
    candidate = NativeRestrictedAdmissionV1(
        *("a" * 64 for _ in range(7)), "run", "attempt-1", 3,
        "2026-10-20T12:30:00Z", "root-review", "root-key", "0" * 64,
    )
    signed = asdict(candidate)
    signed.pop("tag")
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(private),
                    "-out", str(signature)], input=canonical(signed), check=True,
                   capture_output=True)
    proof = replace(candidate, tag=hashlib.sha256(signature.read_bytes()).hexdigest())
    trusted = hashlib.sha256(public.read_bytes()).hexdigest()
    authority = native_host.RootSignatureAuthority(public, trusted, signature)
    assert authority.authenticate(proof)
    assert not authority.authenticate(replace(proof, run_id="another-run"))
    with pytest.raises(ValueError, match="reviewed root public key"):
        native_host.RootSignatureAuthority(public, "f" * 64, signature)
    signature.write_bytes(b"replacement")
    assert not authority.authenticate(proof)


def test_source_record_requires_clean_exact_pushed_ref(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(native_host, "ENGINE_ROOT", root)
    commit = "a" * 40
    config = {"engine_root": str(root), "engine_commit": commit,
              "engine_remote": "https://example.invalid/engine.git",
              "engine_ref": "refs/heads/native-review"}
    observations = {
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("rev-parse", "HEAD"): commit,
        ("remote", "get-url", "origin"): config["engine_remote"],
        ("ls-remote", "--exit-code", "origin", config["engine_ref"]):
            f"{commit}\t{config['engine_ref']}",
    }

    def command(argv, **_):
        return observations[tuple(argv[1:])]

    monkeypatch.setattr(native_host, "_command", command)
    record, digest = native_host.source_record(config)
    assert record["commit"] == commit and digest == hashlib.sha256(canonical(record)).hexdigest()
    observations[("status", "--porcelain", "--untracked-files=all")] = "?? stray.py"
    with pytest.raises(ValueError, match="dirty"):
        native_host.source_record(config)
    observations[("status", "--porcelain", "--untracked-files=all")] = ""
    observations[("ls-remote", "--exit-code", "origin", config["engine_ref"])] = "b" * 40
    with pytest.raises(ValueError, match="pushed source"):
        native_host.source_record(config)


def test_snapshot_candidate_inventories_complete_owned_bytes():
    import os
    original_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-host-snapshot-", dir=Path.home()) as name:
            root = Path(name)
            (root / "config.json").write_bytes(b'{"num_hidden_layers":2}')
            (root / "model.safetensors").write_bytes(b"candidate-bytes")
            candidate = native_host.snapshot_candidate(root=root, repository="configured/model",
                                                       revision="a" * 40)
            assert tuple(item["path"] for item in candidate["files"]) == (
                "config.json", "model.safetensors",
            )
            manifest = root.parent / (root.name + "-manifest.json")
            try:
                manifest.write_bytes(canonical(candidate))
                native_host._snapshot(manifest)
                (root / "model.safetensors").write_bytes(b"drift")
                with pytest.raises(ValueError, match="snapshot"):
                    native_host._snapshot(manifest)
            finally:
                manifest.unlink(missing_ok=True)
    finally:
        os.umask(original_umask)
