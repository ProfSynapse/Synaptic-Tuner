from argparse import Namespace

import tuner.handlers.bucket_handler as bucket_handler
from tuner.handlers.bucket_handler import BucketHandler


def test_bucket_handler_reads_local_file(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")

    handler = BucketHandler(
        args=Namespace(
            path=str(path),
            subcommand="read",
            tail=None,
            jsonl_latest=False,
            pretty=True,
            bucket=None,
            json=True,
        )
    )

    assert handler.handle() == 0


def test_bucket_handler_lists_local_directory(tmp_path):
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    handler = BucketHandler(
        args=Namespace(
            path=str(tmp_path),
            subcommand="list",
            recursive=False,
            files_only=False,
            dirs_only=False,
            limit=20,
            bucket=None,
            json=True,
        )
    )

    assert handler.handle() == 0


def test_bucket_handler_pulls_local_file(tmp_path):
    source_root = tmp_path / "source"
    nested = source_root / "runs" / "hf_jobs" / "sft"
    nested.mkdir(parents=True)
    source = nested / "artifact.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    dest = tmp_path / "dest"

    handler = BucketHandler(
        args=Namespace(
            path=str(source),
            subcommand="pull",
            dest=str(dest),
            bucket=None,
            json=True,
        )
    )

    assert handler.handle() == 0
    pulled = dest / source.name
    assert pulled.exists()


def test_bucket_handler_pushes_local_file(monkeypatch, tmp_path):
    source = tmp_path / "results.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    captured = {}

    def fake_push(path, *, bucket_id, destination=None):
        captured["path"] = path
        captured["bucket_id"] = bucket_id
        captured["destination"] = destination
        return f"hf://buckets/{bucket_id}/{destination.rstrip('/')}/results.json"

    monkeypatch.setattr(bucket_handler, "push_artifacts", fake_push)

    handler = BucketHandler(
        args=Namespace(
            path=str(source),
            subcommand="push",
            dest="runs/manual_uploads/",
            bucket="professorsynapse/toolset-training-artifacts",
            json=True,
        )
    )

    assert handler.handle() == 0
    assert captured == {
        "path": str(source),
        "bucket_id": "professorsynapse/toolset-training-artifacts",
        "destination": "runs/manual_uploads/",
    }
