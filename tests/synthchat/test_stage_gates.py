from SynthChat.stage_gates import run_stage_gates


def test_list_empty_gate_passes_only_for_empty_list():
    results = run_stage_gates(
        [{"type": "list_empty", "field": "environment_result.issues"}],
        {"environment_result": {"issues": []}},
    )

    assert results[0].passed is True

    results = run_stage_gates(
        [{"type": "list_empty", "field": "environment_result.issues"}],
        {"environment_result": {"issues": [{"message": "bad tool"}]}},
    )

    assert results[0].passed is False


def test_all_items_field_equals_gate_rejects_failed_or_missing_items():
    gate = {
        "type": "all_items_field_equals",
        "field": "environment_result.executed_tools",
        "item_field": "status",
        "value": "ok",
    }

    results = run_stage_gates(
        [gate],
        {"environment_result": {"executed_tools": [{"status": "ok"}, {"status": "ok"}]}},
    )

    assert results[0].passed is True

    results = run_stage_gates(
        [gate],
        {"environment_result": {"executed_tools": [{"status": "ok"}, {"status": "error"}]}},
    )

    assert results[0].passed is False

    results = run_stage_gates([gate], {"environment_result": {"executed_tools": []}})

    assert results[0].passed is False


def test_json_schema_gate_rejects_noncanonical_environment_fixture_keys():
    payload = {
        "generated_environment": {
            "environment": {
                "fixture": {
                    "Projects": {"status.md": "ok"},
                    "notes": [{"path": "Projects/status.md", "body": "ok"}],
                },
                "assertions": [{"type": "path_exists", "path": "Projects/status.md"}],
            },
            "task_context": {"target_path": "Projects/status.md"},
        }
    }

    results = run_stage_gates(
        [
            {
                "type": "json_schema",
                "field": "generated_environment",
                "schema": "canonical_environment",
            }
        ],
        payload,
    )

    assert results[0].passed is False


def test_no_placeholder_strings_gate_rejects_ellipsis_anywhere_under_field():
    results = run_stage_gates(
        [{"type": "no_placeholder_strings", "field": "generated_environment"}],
        {"generated_environment": {"task_context": {"target_path": "..."}}},
    )

    assert results[0].passed is False
    assert results[0].metadata["match_count"] == 1


def test_required_mapping_keys_gate_rejects_missing_context_anchors():
    results = run_stage_gates(
        [
            {
                "type": "required_mapping_keys",
                "field": "value.task_context",
                "keys": ["target_path", "final_answer"],
            }
        ],
        {"value": {"task_context": {"target_path": "Notes/status.md"}}},
    )

    assert results[0].passed is False
    assert results[0].metadata["missing_keys"] == ["final_answer"]


def test_min_fixture_items_gate_counts_normalized_fixture_snapshot():
    payload = {
        "value": {
            "environment": {
                "fixture": {
                    "directories": ["Projects"],
                    "files": [{"path": "Projects/status.md", "content": "ok"}],
                }
            }
        }
    }

    results = run_stage_gates(
        [
            {
                "type": "min_fixture_items",
                "field": "value.environment.fixture",
                "min_directories": 2,
                "min_files": 1,
            }
        ],
        payload,
    )

    assert results[0].passed is False
    assert results[0].metadata["directory_count"] == 1


def test_expected_cli_commands_executed_gate_renders_configured_tools():
    gate = {
        "type": "expected_cli_commands_executed",
        "expected_field": "task_context.expected_command_sequence",
        "executed_field": "environment_result.executed_tools",
        "require_order": True,
        "renderers": {
            "storageManager_list": "storage list --path {arguments.path}",
            "contentManager_read": "content read {arguments.path} {arguments.startLine}",
            "storageManager_move": "storage move {arguments.path} {arguments.newPath}",
        },
    }
    payload = {
        "task_context": {
            "expected_command_sequence": [
                "storage list --path Inbox",
                "content read Inbox/a.md 1",
                "storage move Inbox/a.md Done/a.md",
            ]
        },
        "environment_result": {
            "executed_tools": [
                {"name": "storageManager_list", "status": "ok", "arguments": {"path": "Inbox"}},
                {
                    "name": "contentManager_read",
                    "status": "ok",
                    "arguments": {"path": "Inbox/a.md", "startLine": 1},
                },
                {
                    "name": "storageManager_move",
                    "status": "ok",
                    "arguments": {"path": "Inbox/a.md", "newPath": "Done/a.md"},
                },
            ]
        },
    }

    results = run_stage_gates([gate], payload)

    assert results[0].passed is True


def test_expected_cli_commands_executed_gate_rejects_missing_required_command():
    gate = {
        "type": "expected_cli_commands_executed",
        "expected_field": "task_context.expected_command_sequence",
        "executed_field": "environment_result.executed_tools",
        "renderers": {
            "contentManager_read": "content read {arguments.path} {arguments.startLine}",
            "storageManager_move": "storage move {arguments.path} {arguments.newPath}",
        },
    }
    payload = {
        "task_context": {
            "expected_command_sequence": [
                "content read Inbox/a.md 1",
                "content read Inbox/b.md 1",
                "storage move Inbox/a.md Done/a.md",
            ]
        },
        "environment_result": {
            "executed_tools": [
                {
                    "name": "contentManager_read",
                    "status": "ok",
                    "arguments": {"path": "Inbox/a.md", "startLine": 1},
                },
                {
                    "name": "storageManager_move",
                    "status": "ok",
                    "arguments": {"path": "Inbox/a.md", "newPath": "Done/a.md"},
                },
            ]
        },
    }

    results = run_stage_gates([gate], payload)

    assert results[0].passed is False
    assert results[0].metadata["missing_commands"] == ["content read Inbox/b.md 1"]
