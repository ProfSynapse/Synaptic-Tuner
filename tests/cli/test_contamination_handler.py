"""check-contamination CLI: parsing, routing, exit codes, JSON and decontaminated output."""

import json
from pathlib import Path

from tuner.cli.parser import create_parser
from tuner.cli.router import route_command


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


LEAKED = "Please summarize the quarterly planning notes and list every open action item for the team"


def _fixtures(tmp_path: Path):
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"conversations": [{"role": "user", "content": LEAKED}, {"role": "assistant", "content": "ok"}]},
            {"conversations": [{"role": "user", "content": "Rename the draft file"}, {"role": "assistant", "content": "done"}]},
            {"prompt": "Unrelated prompt about cooking pasta at home", "completion": "boil water"},
        ],
    )
    leaked_eval = _write_jsonl(tmp_path / "leaked.jsonl", [{"id": "e1", "text": LEAKED}])
    clean_eval = _write_jsonl(
        tmp_path / "clean.jsonl",
        [{"id": "e2", "text": "Describe how photosynthesis converts sunlight into chemical energy in plants"}],
    )
    return train, leaked_eval, clean_eval


def _run(argv):
    args = create_parser().parse_args(argv)
    return route_command(args)


def test_parser_accepts_contamination_flags():
    args = create_parser().parse_args(
        [
            "check-contamination",
            "--train-data", "a.jsonl", "--train-data", "b.jsonl",
            "--eval-source", "Evaluator/config/scenarios/tool_prompts.yaml",
            "--eval-text", "holdout.jsonl", "--eval-text-field", "body.text",
            "--ngram", "5", "--threshold", "0.3", "--min-item-tokens", "3", "--top-k", "7",
            "--report-dir", "scratch/x", "--write-decontaminated", "scratch/y",
            "--contamination-config", "cfg.yaml", "--json",
        ]
    )
    assert args.command == "check-contamination"
    assert args.train_data == ["a.jsonl", "b.jsonl"]
    assert args.eval_source == ["Evaluator/config/scenarios/tool_prompts.yaml"]
    assert args.eval_text == ["holdout.jsonl"] and args.eval_text_field == "body.text"
    assert (args.ngram, args.threshold, args.min_item_tokens, args.top_k) == (5, 0.3, 3, 7)
    assert args.contamination_config == "cfg.yaml"
    assert args.json is True


def test_flagged_item_exits_2_and_clean_exits_0(tmp_path, capsys):
    train, leaked_eval, clean_eval = _fixtures(tmp_path)
    reports = tmp_path / "reports"

    code = _run(["check-contamination", "--train-data", str(train), "--eval-text", str(leaked_eval),
                 "--report-dir", str(reports), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["success"] is True
    data = payload["data"]
    assert data["passed"] is False
    assert data["summary"]["items_over_threshold"] == 1
    assert data["summary"]["exact_duplicate_prompts"] == 1
    assert list(data["flagged_training_rows"]) == ["train.jsonl:1"]
    assert Path(data["report_path"]).is_file()
    assert Path(data["report_path"]).is_relative_to(reports)

    code = _run(["check-contamination", "--train-data", str(train), "--eval-text", str(clean_eval),
                 "--report-dir", str(reports), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["data"]["passed"] is True


def test_threshold_flag_controls_exit_code(tmp_path, capsys):
    train, _, _ = _fixtures(tmp_path)
    # Half of the eval item is copied into training, half is new.
    partial = LEAKED + " and then translate the whole summary into french with careful attention to idioms"
    eval_path = _write_jsonl(tmp_path / "partial.jsonl", [{"id": "p", "text": partial}])
    base = ["check-contamination", "--train-data", str(train), "--eval-text", str(eval_path),
            "--report-dir", str(tmp_path / "r"), "--json"]

    assert _run(base + ["--threshold", "0.3"]) == 2
    capsys.readouterr()
    assert _run(base + ["--threshold", "0.9"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0.3 <= payload["data"]["summary"]["max_containment"] < 0.9


def test_write_decontaminated_file_and_sidecar(tmp_path, capsys):
    train, leaked_eval, _ = _fixtures(tmp_path)
    out = tmp_path / "clean" / "train.decontaminated.jsonl"
    code = _run(["check-contamination", "--train-data", str(train), "--eval-text", str(leaked_eval),
                 "--report-dir", str(tmp_path / "r"), "--write-decontaminated", str(out)])
    text = capsys.readouterr().out
    assert code == 2
    assert "FLAGGED" in text
    kept = out.read_text(encoding="utf-8").splitlines()
    assert len(kept) == 2 and LEAKED not in out.read_text(encoding="utf-8")
    sidecar = json.loads(Path(str(out) + ".removed.json").read_text(encoding="utf-8"))
    assert sidecar["removed"][0]["line_number"] == 1
    reasons = {reason["reason"] for reason in sidecar["removed"][0]["reasons"]}
    assert reasons == {"containment", "exact_duplicate_prompt"}
    assert train.read_text(encoding="utf-8").count("\n") == 3  # source untouched


def test_scenario_source_and_config_file(tmp_path, capsys):
    train, _, _ = _fixtures(tmp_path)
    scenarios = tmp_path / "evalcfg" / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "mini.yaml").write_text(
        f"name: mini\ntests:\n  - id: leaked\n    question: \"{LEAKED}\"\n", encoding="utf-8"
    )
    config = tmp_path / "contamination.yaml"
    config.write_text(
        "ngram: 6\nthreshold: 0.5\n"
        f"train_datasets: [{train}]\n"
        f"report_dir: {tmp_path / 'r'}\n"
        f"eval:\n  sources: [{scenarios / 'mini.yaml'}]\n",
        encoding="utf-8",
    )
    code = _run(["check-contamination", "--contamination-config", str(config), "--json"])
    data = json.loads(capsys.readouterr().out)["data"]
    assert code == 2
    assert data["config"]["ngram"] == 6
    assert data["flagged_items"][0]["item_id"] == "leaked"


def test_missing_training_data_is_an_error(tmp_path, capsys):
    _, leaked_eval, _ = _fixtures(tmp_path)
    config = tmp_path / "empty.yaml"
    config.write_text("train_datasets: []\neval:\n  sources: []\n", encoding="utf-8")
    code = _run(["check-contamination", "--contamination-config", str(config),
                 "--eval-text", str(leaked_eval), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["success"] is False
    assert "No training datasets" in payload["error"]["message"]
