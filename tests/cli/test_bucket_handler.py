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


def test_bucket_handler_analyzes_run_prefix(monkeypatch):
    def fake_read_artifact(path, *, bucket_id=None, tail=None, jsonl_latest=False, pretty=False):
        if path.endswith("/training_lineage.json"):
            return """{
              "training": {"batch_size": 8, "gradient_accumulation_steps": 4, "effective_batch_size": 32},
              "capacity_profile": {"peak_gpu_memory_reserved_gb": 30.3, "min_gpu_memory_reserved_headroom_gb": 49.0, "oom_risk_level": "low"},
              "runtime": {"status": "completed", "duration_seconds": 3250.6},
              "pricing": {"estimated_cost_usd": 2.25},
              "results": {"final_loss": 0.4631},
              "hardware": {"cloud_gpu_type": "a100-large"}
            }"""
        if path.endswith("/evaluation_lineage.json"):
            return """{
              "runtime": {"status": "completed", "duration_seconds": 1748.9, "backend": "vllm"},
              "pricing": {"estimated_cost_usd": 1.21},
              "execution": {"hardware_flavor": "a100-large"},
              "results_summary": {"passed": 46, "failed": 23, "request_errors": 1, "overall_pass_rate": 59.7, "schema_pass_rate": 70.1},
              "behavior_results": {"pass_rate": 36.7},
              "top_failure_reasons": [{"reason": "No acceptable tool called. Valid options: TEXT_ONLY", "count": 5}]
            }"""
        if path.endswith("/loss_lineage.json"):
            return """{
              "status": "completed",
              "runtime": {"duration_seconds": 1719.6},
              "pricing": {"estimated_cost_usd": 1.19},
              "execution": {"hardware_flavor": "a100-large", "worker_count": 1},
              "results": {"row_count": 9009, "mean_loss": 0.4314, "median_loss": 0.2676, "p95_loss": 1.4844, "max_loss": 3.3125}
            }"""
        raise AssertionError(path)

    def fake_list_artifacts(path, *, bucket_id=None, recursive=False, files_only=False, dirs_only=False):
        if path.endswith("/evaluations/vllm"):
            return [
                {"path": "runs/hf_jobs/sft/run/evaluations/vllm/20260323_112000_abcd", "type": "directory", "size": 0},
                {"path": "runs/hf_jobs/sft/run/evaluations/vllm/20260323_112933_57d6", "type": "directory", "size": 0},
            ]
        raise AssertionError(path)

    monkeypatch.setattr(bucket_handler, "read_artifact", fake_read_artifact)
    monkeypatch.setattr(bucket_handler, "list_artifacts", fake_list_artifacts)

    handler = BucketHandler(
        args=Namespace(
            path="runs/hf_jobs/sft/run",
            subcommand="analyze",
            bucket="professorsynapse/toolset-training-artifacts",
            json=True,
        )
    )

    assert handler.handle() == 0
