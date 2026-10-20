from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts import diagnose_docker_torch_import as diagnostic


def test_closed_single_line_parser_rejects_extra_output() -> None:
    assert diagnostic._one_line(b"abc\n") == "abc"
    assert diagnostic._one_line(b"abc") == "abc"
    for raw in (b"abc\nsecret\n", b" abc\n", b"\n", b"abc\r\n"):
        with pytest.raises(ValueError):
            diagnostic._one_line(raw)


def test_exact_image_import_probe_is_cpu_only_and_retains_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "sha256:" + "a" * 64
    cid = "b" * 64
    daemon = "24805003-b125-469b-8080-9da4daacd364"
    release = {"image": {"digest": "a" * 64, "reference": "example/image@" + image},
               "python": {"executable": "/opt/unsloth-venv/bin/python3"},
               "manifest_digest": "c" * 64}
    (tmp_path / "release.json").write_text(json.dumps(release, sort_keys=True, separators=(",", ":")))
    release_sha = hashlib.sha256((tmp_path / "release.json").read_bytes()).hexdigest()
    monkeypatch.setattr(diagnostic, "_authority", lambda *_: object())
    commands = []

    def run(_authority, args, *, timeout=120):
        commands.append(args)
        if args[:2] == ("info", "--format"):
            return diagnostic.subprocess.CompletedProcess(args, 0, (daemon + "\n").encode(), b"")
        if args[:2] == ("image", "inspect"):
            return diagnostic.subprocess.CompletedProcess(args, 0, (image + "\n").encode(), b"")
        if args[:2] == ("container", "inspect"):
            return diagnostic.subprocess.CompletedProcess(args, 0, (image + "\n").encode(), b"")
        assert args[0] == "run"
        assert "--pull=never" in args and "--network=none" in args
        assert "--read-only" in args and "--cap-drop=ALL" in args
        assert "--memory=4g" in args and "--cpus=2" in args
        assert "--env=NVIDIA_VISIBLE_DEVICES=void" in args
        assert "--env=CUDA_VISIBLE_DEVICES=" in args
        assert "--volume" not in args and "--gpus" not in args
        assert "-I" in args and "-c" in args
        guarded = "--mount" in args
        if guarded:
            assert "--user" in args
            assert "readonly" in args[args.index("--mount") + 1]
            target = tmp_path / "guarded"
            payload = b'{"category":"PASS","schema_version":"synaptic-docker-torch-import-child/v1","stage":"TORCH_IMPORT"}\n'
        else:
            target = out
            payload = b'{"category":"IMPORT","schema_version":"synaptic-docker-torch-import-child/v1"}\n'
        (target / "container.id").write_text(cid + "\n")
        return diagnostic.subprocess.CompletedProcess(args, 0, payload, b"")

    monkeypatch.setattr(diagnostic, "_run", run)
    out = tmp_path / "fresh"
    report = diagnostic.diagnose(docker=Path("/usr/bin/docker"),
        docker_config=Path("/private/config"), release_path=tmp_path / "release.json",
        daemon_id=daemon, output_dir=out, expected_release_sha256=release_sha)
    assert report["category"] == "IMPORT"
    assert report["container_id"] == cid
    assert json.loads((out / "result.json").read_text()) == report
    assert len(commands) == 5
    source = tmp_path / "source" / "engine"
    source.mkdir(parents=True)
    (source.parent / "control").mkdir()
    (source.parent / "control" / "offline-sft-worker-v1.json").write_text("{}")
    guarded = diagnostic.diagnose(docker=Path("/usr/bin/docker"),
        docker_config=Path("/private/config"), release_path=tmp_path / "release.json",
        daemon_id=daemon, output_dir=tmp_path / "guarded",
        expected_release_sha256=release_sha, source_root=source)
    assert guarded["category"] == "PASS" and guarded["stage"] == "TORCH_IMPORT"
    assert guarded["scope"] == "guarded"
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        diagnostic.diagnose(docker=Path("/usr/bin/docker"),
            docker_config=Path("/private/config"), release_path=tmp_path / "release.json",
            daemon_id=daemon, output_dir=out, expected_release_sha256=release_sha)


def test_cli_rejection_never_prints_exception_text(monkeypatch, capsys) -> None:
    def fail(**_kwargs):
        raise RuntimeError("private exception path /secret and credential")

    monkeypatch.setattr(diagnostic, "diagnose", fail)
    code = diagnostic.main(["--docker", "/usr/bin/docker", "--docker-config", "/private/config",
        "--release", "/private/release.json", "--output-dir", "/private/output",
        "--daemon-id", "24805003-b125-469b-8080-9da4daacd364",
        "--expected-release-sha256", "a" * 64])
    assert code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"schema_version": diagnostic.SCHEMA,
        "status": "REJECTED", "reason_code": "OTHER"}
    assert "private" not in output.err and "Traceback" not in output.err
