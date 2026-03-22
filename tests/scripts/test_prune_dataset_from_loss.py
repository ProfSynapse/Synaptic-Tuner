import importlib.util
import json
import sys
from pathlib import Path


def _load_skill_module():
    path = Path(".agents/skills/fine-tuning/scripts/prune_dataset_from_loss.py").resolve()
    spec = importlib.util.spec_from_file_location("prune_dataset_from_loss_skill_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_analyze_dataset_reports_enriched_family(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "dataset.jsonl"
    loss_path = tmp_path / "losses.jsonl"

    rows = [
        {
            "conversations": [
                {"role": "user", "content": 'Result: {"success": true, "linesDelta": 1, "recommendations": ["re-read"]}'},
                {"role": "assistant", "content": "Done. Updated."},
            ],
            "pattern": "text_only",
            "label": True,
        },
        {
            "conversations": [
                {"role": "user", "content": 'Result: {"success": true, "linesDelta": 1}'},
                {"role": "assistant", "content": "All patches applied."},
            ],
            "pattern": "text_only",
            "label": True,
        },
        {
            "conversations": [
                {"role": "user", "content": "Use storageManager to save this."},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "useTools"}}]},
            ],
            "pattern": "tool_only",
            "label": True,
        },
        {
            "conversations": [
                {"role": "user", "content": "What should I do next?"},
                {"role": "assistant", "content": "Ask a clarification question first."},
            ],
            "pattern": "text_only",
            "label": True,
        },
    ]
    losses = [
        {"index": 0, "loss": 3.2, "jsonl_hash": "a"},
        {"index": 1, "loss": 3.0, "jsonl_hash": "b"},
        {"index": 2, "loss": 0.4, "jsonl_hash": "c"},
        {"index": 3, "loss": 0.2, "jsonl_hash": "d"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(loss_path, losses)

    analysis = module.analyze_dataset_against_loss(dataset_path=dataset, loss_source=str(loss_path), top_percent=0.5)

    assert analysis.total_rows == 4
    assert analysis.top_slice_count == 2
    assert analysis.feature_report[0]["feature"] in {"text_only_result", "linesdelta", "recommendations", "doneish"}
    assert analysis.suggestions[0]["strategy"] == "result_echo_linesdelta_recommendations"


def test_prune_dataset_supports_generic_loss_threshold(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "dataset.jsonl"
    loss_path = tmp_path / "losses.jsonl"
    rows = [
        {"conversations": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "pattern": "text_only", "label": True},
        {"conversations": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}], "pattern": "tool_text", "label": True},
        {"conversations": [{"role": "user", "content": "e"}, {"role": "assistant", "content": "f"}], "pattern": "tool_only", "label": True},
    ]
    losses = [
        {"index": 0, "loss": 2.3, "jsonl_hash": "a"},
        {"index": 1, "loss": 1.7, "jsonl_hash": "b"},
        {"index": 2, "loss": 2.1, "jsonl_hash": "c"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(loss_path, losses)

    result = module.prune_dataset(
        dataset_path=dataset,
        loss_source=str(loss_path),
        strategy="loss_threshold",
        min_loss=2.0,
        output_path=tmp_path / "filtered.jsonl",
        removed_output_path=tmp_path / "removed.jsonl",
        metadata_path=tmp_path / "filtered.metadata.json",
    )

    assert result.kept_rows == 1
    assert result.removed_rows == 2
    kept = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert kept[0]["conversations"][0]["content"] == "c"


def test_prune_dataset_supports_top_percent(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "dataset.jsonl"
    loss_path = tmp_path / "losses.jsonl"
    rows = [
        {"conversations": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}], "pattern": "text_only", "label": True}
        for i in range(5)
    ]
    losses = [
        {"index": 0, "loss": 5.0, "jsonl_hash": "a"},
        {"index": 1, "loss": 4.0, "jsonl_hash": "b"},
        {"index": 2, "loss": 3.0, "jsonl_hash": "c"},
        {"index": 3, "loss": 2.0, "jsonl_hash": "d"},
        {"index": 4, "loss": 1.0, "jsonl_hash": "e"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(loss_path, losses)

    result = module.prune_dataset(
        dataset_path=dataset,
        loss_source=str(loss_path),
        strategy="top_percent",
        top_percent=0.4,
        output_path=tmp_path / "filtered.jsonl",
        removed_output_path=tmp_path / "removed.jsonl",
        metadata_path=tmp_path / "filtered.metadata.json",
    )

    assert result.removed_rows == 2
    removed = [json.loads(line) for line in (tmp_path / "removed.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert [row["conversations"][0]["content"] for row in removed] == ["u0", "u1"]


def test_prune_dataset_repo_specific_preset_removes_expected_rows(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "dataset.jsonl"
    loss_path = tmp_path / "losses.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": 'Result: {"success": true, "linesDelta": 1, "recommendations": ["re-read"]}'},
                {"role": "assistant", "content": "Done. Updated."},
            ],
            "pattern": "text_only",
            "label": True,
        },
        {
            "conversations": [
                {"role": "user", "content": 'Result: {"success": true}'},
                {"role": "assistant", "content": "Done. Updated."},
            ],
            "pattern": "text_only",
            "label": True,
        },
    ]
    losses = [
        {"index": 0, "loss": 2.5, "jsonl_hash": "a"},
        {"index": 1, "loss": 2.5, "jsonl_hash": "b"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(loss_path, losses)

    result = module.prune_dataset(
        dataset_path=dataset,
        loss_source=str(loss_path),
        strategy="result_echo_linesdelta_recommendations",
        output_path=tmp_path / "filtered.jsonl",
        removed_output_path=tmp_path / "removed.jsonl",
        metadata_path=tmp_path / "filtered.metadata.json",
    )

    assert result.kept_rows == 1
    assert result.removed_rows == 1


def test_prune_dataset_result_echo_long_recap_removes_only_long_result_rows(tmp_path):
    module = _load_skill_module()
    dataset = tmp_path / "dataset.jsonl"
    loss_path = tmp_path / "losses.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "user", "content": 'Result: {"success": true}'},
                {"role": "assistant", "content": " ".join(["recap"] * 100)},
            ],
            "pattern": "text_only",
            "label": True,
        },
        {
            "conversations": [
                {"role": "user", "content": 'Result: {"success": true}'},
                {"role": "assistant", "content": "Short recap."},
            ],
            "pattern": "text_only",
            "label": True,
        },
        {
            "conversations": [
                {"role": "user", "content": "Normal user request"},
                {"role": "assistant", "content": " ".join(["recap"] * 100)},
            ],
            "pattern": "text_only",
            "label": True,
        },
    ]
    losses = [
        {"index": 0, "loss": 2.5, "jsonl_hash": "a"},
        {"index": 1, "loss": 2.5, "jsonl_hash": "b"},
        {"index": 2, "loss": 2.5, "jsonl_hash": "c"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(loss_path, losses)

    result = module.prune_dataset(
        dataset_path=dataset,
        loss_source=str(loss_path),
        strategy="result_echo_long_recap",
        output_path=tmp_path / "filtered.jsonl",
        removed_output_path=tmp_path / "removed.jsonl",
        metadata_path=tmp_path / "filtered.metadata.json",
    )

    assert result.kept_rows == 2
    assert result.removed_rows == 1
