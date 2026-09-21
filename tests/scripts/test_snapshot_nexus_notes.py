import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(".skills/fine-tuning/scripts/snapshot_nexus_notes.py").resolve()
    spec = importlib.util.spec_from_file_location("nexus_note_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(output_path):
    return {
        "nexus": {"vault": "Vault", "workspace": "research", "session": "snapshot"},
        "notes": [
            {"id": "first", "path": "Notes/first.md", "metadata": {"kind": "source"}},
            {"id": "second", "path": "Notes/second.md", "metadata": {"kind": "decision", "rank": 2}},
        ],
        "options": {"strip_frontmatter": True},
        "output_path": str(output_path),
    }


def _envelope(path, numbered):
    inner = {"success": True, "path": path, "content": numbered}
    return json.dumps({"content": [{"type": "text", "text": json.dumps(inner)}]})


def test_snapshot_uses_exact_read_only_argv_and_writes_deterministic_rows(tmp_path, monkeypatch):
    module = _load_module()
    output = tmp_path / "private" / "snapshot.jsonl"
    config = module.validate_config(_config(output))
    responses = iter([
        _envelope("Notes/first.md", "1: ---\n2: title: internal\n3: ---\n4: alpha"),
        _envelope("Notes/second.md", "1: beta\n2: gamma"),
    ])
    calls = []
    monkeypatch.setattr(module, "resolve_nexus_command", lambda argv: argv)

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=next(responses))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    rows = module.build_rows(config)
    digest = module.write_snapshot(rows, output)

    expected_argv = [
        "nexus", "use", "--json", "--vault", "Vault", "--workspace", "research", "--session", "snapshot",
        "--memory", "snapshotting configured notes", "--goal", "read configured note without mutation",
        "--", "content", "read", "Notes/first.md", "1",
    ]
    assert calls[0][0] == expected_argv
    assert calls[0][1] == {"capture_output": True, "text": True, "check": False, "encoding": "utf-8"}
    assert all("search" not in call[0] and "write" not in call[0] and "storage" not in call[0] for call in calls)
    assert rows[0]["text"] == "alpha"
    assert rows[0]["full_content_sha256"] == hashlib.sha256(b"---\ntitle: internal\n---\nalpha").hexdigest()
    assert rows[0]["selected_text_sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert [row["id"] for row in rows] == ["first", "second"]
    payload = output.read_bytes()
    assert digest == hashlib.sha256(payload).hexdigest()
    assert payload == b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)


def test_windows_cmd_shim_resolves_to_node_without_shell(tmp_path, monkeypatch):
    module = _load_module()
    shim = tmp_path / "nexus.cmd"
    cli = tmp_path / "nexus-cli.js"
    node = tmp_path / "node.exe"
    shim.write_text("@echo off\nnode ignored", encoding="utf-8")
    cli.write_text("// intentionally not executed", encoding="utf-8")
    node.write_text("", encoding="utf-8")

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.shutil, "which", lambda name: str(shim if name == "nexus" else node if name == "node" else ""))
    logical = module.nexus_argv({"vault": "Vault", "workspace": "research", "session": "snapshot"}, "Notes/one.md")
    assert module.resolve_nexus_command(logical) == [str(node), str(cli), *logical[1:]]

    cli.unlink()
    with __import__("pytest").raises(module.SnapshotError, match="NEXUS_SHIM_UNSUPPORTED"):
        module.resolve_nexus_command(logical)


def test_read_note_preserves_resolver_failure_codes(monkeypatch):
    module = _load_module()
    nexus = {"vault": "Vault", "workspace": "research", "session": "snapshot"}
    monkeypatch.setattr(module, "resolve_nexus_command", lambda _: (_ for _ in ()).throw(module.SnapshotError("NEXUS_SHIM_UNSUPPORTED")))
    with __import__("pytest").raises(module.SnapshotError, match="NEXUS_SHIM_UNSUPPORTED"):
        module.read_note(nexus, "Notes/one.md")


def test_invalid_envelopes_and_numbered_sequences_fail_closed_without_prose(tmp_path, monkeypatch, capsys):
    module = _load_module()
    output = tmp_path / "snapshot.jsonl"
    config = _config(output)
    malformed = json.dumps({"content": [{"type": "text", "text": json.dumps({"success": True, "path": "Notes/first.md", "content": "1: private prose\n3: skipped"})}]})

    monkeypatch.setattr(module, "load_config", lambda _: config)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=malformed))
    assert module.main(["--config", "ignored.yaml"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"status": "error", "error": {"code": "NUMBERED_CONTENT_INVALID"}}
    assert "private prose" not in captured.out and "Traceback" not in captured.out
    assert not output.exists()

    with __import__("pytest").raises(module.SnapshotError, match="NEXUS_ENVELOPE_INVALID"):
        module.parse_nexus_response("not json", "Notes/first.md")


def test_batch_failure_leaves_no_artifact_and_existing_output_is_idempotent_or_collision(tmp_path, monkeypatch):
    module = _load_module()
    output = tmp_path / "snapshot.jsonl"
    config = module.validate_config(_config(output))
    replies = iter([
        _envelope("Notes/first.md", "1: alpha"),
        _envelope("Notes/second.md", "1: malformed\n3: gap"),
    ])
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(replies)))
    with __import__("pytest").raises(module.SnapshotError, match="NUMBERED_CONTENT_INVALID"):
        module.build_rows(config)
    assert not output.exists()

    rows = [{"id": "only", "source_path": "Notes/only.md", "metadata": {}, "text": "alpha", "full_content_sha256": "a", "selected_text_sha256": "b"}]
    first = module.write_snapshot(rows, output)
    assert module.write_snapshot(rows, output) == first
    output.write_bytes(b"different")
    with __import__("pytest").raises(module.SnapshotError, match="OUTPUT_COLLISION"):
        module.write_snapshot(rows, output)


def test_cli_argument_errors_are_compact_json_only(capsys):
    module = _load_module()
    assert module.main([]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"status": "error", "error": {"code": "INVALID_ARGUMENTS"}}

    assert module.main(["--unknown-option"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"status": "error", "error": {"code": "INVALID_ARGUMENTS"}}


def test_metadata_rejects_nonfinite_floats_and_accepts_finite_values(tmp_path):
    module = _load_module()
    for value in (float("nan"), float("inf"), float("-inf")):
        config = _config(tmp_path / "snapshot.jsonl")
        config["notes"][0]["metadata"]["score"] = value
        with __import__("pytest").raises(module.SnapshotError, match="UNSAFE_METADATA"):
            module.validate_config(config)

    config = _config(tmp_path / "snapshot.jsonl")
    config["notes"][0]["metadata"]["score"] = 1.25
    validated = module.validate_config(config)
    assert validated["notes"][0]["metadata"]["score"] == 1.25
    assert "NaN" not in module._json(validated["notes"][0]["metadata"])
