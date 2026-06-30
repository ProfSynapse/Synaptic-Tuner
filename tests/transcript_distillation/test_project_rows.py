import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".skills" / "transcript-distillation" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from project_rows import ProjectConfig, main, project_rows  # noqa: E402


def test_dpo_pairs_same_default_prompt_hash_and_drops_null_or_unpaired_labels():
    prompt = [{"role": "user", "content": "Pick a tool"}]
    chosen = {
        "conversations": prompt + [{"role": "assistant", "content": "Use search"}],
        "label": True,
        "source_example_id": "chosen",
    }
    rejected = {
        "conversations": prompt + [{"role": "assistant", "content": "Guess"}],
        "label": False,
        "source_example_id": "rejected",
    }
    null_label = {
        "conversations": prompt + [{"role": "assistant", "content": "Maybe"}],
        "label": None,
        "source_example_id": "null",
    }
    unpaired = {
        "conversations": [{"role": "user", "content": "Different"}] + [{"role": "assistant", "content": "No pair"}],
        "label": True,
        "source_example_id": "unpaired",
    }

    result = project_rows([chosen, rejected, null_label, unpaired], ProjectConfig(target="dpo"))

    assert result.rows == [
        {
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": "Use search"}],
            "rejected": [{"role": "assistant", "content": "Guess"}],
            "provenance": {
                "prompt_key": result.rows[0]["provenance"]["prompt_key"],
                "chosen_source_example_id": "chosen",
                "rejected_source_example_id": "rejected",
            },
        }
    ]
    assert result.counters["output_rows"] == 1
    assert result.counters["dropped_null_label"] == 1
    assert result.counters["unpaired_chosen"] == 1


def test_dpo_uses_configured_prompt_key_and_serializes_native_tool_calls_to_text():
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "Make a file"},
                {
                    "role": "assistant",
                    "content": "Creating it.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": '{"path":"a.txt","body":"ok"}',
                            },
                        }
                    ],
                },
            ],
            "label": True,
            "metadata": {"pair_key": "same"},
            "source_example_id": "chosen",
        },
        {
            "conversations": [
                {"role": "user", "content": "A different visible prompt"},
                {"role": "assistant", "content": "No."},
            ],
            "label": False,
            "metadata": {"pair_key": "same"},
            "source_example_id": "rejected",
        },
    ]

    result = project_rows(
        rows,
        ProjectConfig(target="dpo", prompt_key_field="metadata.pair_key"),
    )

    assert len(result.rows) == 1
    chosen = result.rows[0]["chosen"][0]
    assert chosen["role"] == "assistant"
    assert chosen["content"] == (
        'Creating it.\nTool calls: [{"arguments":"{\\"body\\":\\"ok\\",\\"path\\":\\"a.txt\\"}",'
        '"name":"shell"}]'
    )
    assert result.rows[0]["provenance"]["prompt_key"] == "same"


def test_static_grpo_extracts_native_first_target_tool_call_and_excludes_tool_messages_by_default():
    row = {
        "conversations": [
            {"role": "system", "content": "Follow policy"},
            {"role": "user", "content": "Run tests"},
            {"role": "tool", "tool_call_id": "old", "content": "prior result"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"cmd":"pytest","env":{"CI":"1"}}',
                        },
                    }
                ],
            },
        ],
        "source_example_id": "native",
    }

    result = project_rows([row], ProjectConfig(target="static-grpo"))

    assert result.rows == [
        {
            "prompt": [
                {"role": "system", "content": "Follow policy"},
                {"role": "user", "content": "Run tests"},
            ],
            "ground_truth_tool": "shell",
            "ground_truth_args_json": '{"cmd":"pytest","env":{"CI":"1"}}',
            "provenance": {
                "source_example_id": "native",
                "prompt_hash": result.rows[0]["provenance"]["prompt_hash"],
            },
        }
    ]
    assert result.counters["output_rows"] == 1

    with_tools = project_rows(
        [row],
        ProjectConfig(target="static-grpo", include_tool_messages=True),
    )
    assert with_tools.rows[0]["prompt"][2] == {"role": "tool", "content": "prior result"}


def test_static_grpo_parses_flat_completion_tool_calls_and_supports_last_user_only_prompt():
    row = {
        "conversations": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Ack"},
            {"role": "user", "content": "Search now"},
            {
                "role": "assistant",
                "content": 'Tool calls: [{"name":"search","input":{"query":"abc"}}]',
            },
        ],
        "completion": 'Tool calls: [{"name":"search","input":{"query":"abc"}}]',
        "source_example_id": "flat",
    }

    result = project_rows(
        [row],
        ProjectConfig(target="static-grpo", prompt_mode="last_user_only"),
    )

    assert result.rows[0]["prompt"] == [{"role": "user", "content": "Search now"}]
    assert result.rows[0]["ground_truth_tool"] == "search"
    assert result.rows[0]["ground_truth_args_json"] == '{"query":"abc"}'


def test_cli_writes_jsonl_and_report_from_yaml_config(tmp_path):
    source = tmp_path / "rows.jsonl"
    output = tmp_path / "dpo.jsonl"
    report = tmp_path / "report.json"
    config = tmp_path / "project.yaml"
    prompt = [{"role": "user", "content": "Choose"}]
    source.write_text(
        "\n".join(
            [
                json.dumps({"conversations": prompt + [{"role": "assistant", "content": "Good"}], "label": True}),
                json.dumps({"conversations": prompt + [{"role": "assistant", "content": "Bad"}], "label": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config.write_text(
        f"""
projection:
  target: dpo
  input: {source.as_posix()}
  output: {output.as_posix()}
  report: {report.as_posix()}
""",
        encoding="utf-8",
    )

    assert main(["--config", str(config)]) == 0

    projected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert projected[0]["chosen"] == [{"role": "assistant", "content": "Good"}]
    counts = json.loads(report.read_text(encoding="utf-8"))["counts"]
    assert counts["input_rows"] == 2
    assert counts["output_rows"] == 1
