from __future__ import annotations

import json
from pathlib import Path

from SynthChat.generator import SynthChatGenerator
from shared.environments import EnvironmentValidator


class _FakeLLMClient:
    def __init__(self, responses=None, structured_responses=None):
        self._responses = list(responses or [])
        self._structured_responses = list(structured_responses or [])
        self.messages = []
        self.structured_messages = []

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        self.messages.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if not self._responses:
            raise AssertionError("No more fake responses available")
        return self._responses.pop(0)

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        self.structured_messages.append(
            {
                "messages": messages,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._structured_responses:
            raise AssertionError("No more fake structured responses available")
        return self._structured_responses.pop(0)


def test_synthchat_generator_renders_mocked_workspace_prompt_from_generated_environment():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    environment_json = json.dumps(
        {
            "environment": {
                "fixture": {
                    "notes": [
                        {
                            "path": "Ops/production-config.md",
                            "frontmatter": {
                                "title": "Production Config",
                                "type": "config",
                                "environment": "production",
                            },
                            "body": "api_base_url: https://api.old.example.com\nretry_policy: exponential\n",
                        },
                        {
                            "path": "Ops/staging-config.md",
                            "frontmatter": {
                                "title": "Staging Config",
                                "type": "config",
                                "environment": "staging",
                            },
                            "body": "api_base_url: https://api.staging.example.com\nretry_policy: linear\n",
                        },
                    ]
                },
                "assertions": [
                    {
                        "type": "file_contains",
                        "path": "Ops/production-config.md",
                        "text": "api_base_url: https://api.prod.example.com",
                    },
                    {
                        "type": "file_contains",
                        "path": "Ops/staging-config.md",
                        "text": "api_base_url: https://api.staging.example.com",
                    },
                ],
            },
            "system_context": {
                "session_id": "session_1732300800000_env001",
                "workspace_id": "ws_generated_ops",
                "selected_workspace": {
                    "id": "ws_generated_ops",
                    "name": "Operations Workspace",
                    "root_folder": "",
                    "recent_files": ["Ops/production-config.md"],
                    "key_files": ["Ops/production-config.md", "Ops/staging-config.md"],
                    "workflows": [{"name": "Config Maintenance", "steps": ["Find note", "Read note", "Update note"]}],
                    "preferences": "Read before overwrite.",
                },
            },
        }
    )

    assistant_json = json.dumps(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "useTools",
                        "arguments": {
                            "context": {
                                "sessionId": "session_1732300800000_env001",
                                "workspaceId": "ws_generated_ops",
                                "memory": "Need to update the production note only.",
                                "goal": "Find and update the production config note.",
                            },
                            "calls": [
                                {
                                    "agent": "searchManager",
                                    "tool": "searchContent",
                                    "params": {"query": "api.old.example.com", "path": "Ops/"},
                                },
                                {
                                    "agent": "contentManager",
                                    "tool": "read",
                                    "params": {"path": "Ops/production-config.md", "startLine": 1},
                                },
                                {
                                    "agent": "contentManager",
                                    "tool": "update",
                                    "params": {
                                        "path": "Ops/production-config.md",
                                        "startLine": 1,
                                        "content": "---\ntitle: Production Config\ntype: config\nenvironment: production\n---\napi_base_url: https://api.prod.example.com\nretry_policy: exponential\n",
                                    },
                                },
                            ],
                        },
                    },
                }
            ],
        }
    )

    client = _FakeLLMClient(
        [
            environment_json,
            "Please update the production config note to use the new API base URL.",
            assistant_json,
        ]
    )
    validator = EnvironmentValidator(backend="local")
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=validator,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "tool",
        "tool": "contentManager_update",
        "system_template": "mocked_workspace_vault",
        "system_context": {
            "available_workspaces": [
                {
                    "id": "ws_generated_ops",
                    "name": "Operations Workspace",
                    "description": "Operational notes and service configuration",
                    "root_folder": "",
                }
            ],
            "available_prompts": [
                {"id": "agent_ops_writer", "name": "Ops Writer", "purpose": "Updates config notes"},
            ],
            "assistant_instructions": "Use tools carefully.",
        },
        "environment_generation": {
            "prompt": "Generate environment JSON.",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
    }

    result = generator.generate_single(
        scenario_key="env_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    system_prompt = result.example["conversations"][0]["content"]
    assert "<available_workspaces>" in system_prompt
    assert "<available_tools>" in system_prompt
    assert "contentManager:" in system_prompt
    assert "read: required [path, startLine]" in system_prompt
    assert '<selected_workspace name="Operations Workspace" id="ws_generated_ops">' in system_prompt
    assert "Ops/production-config.md" in system_prompt
    assert result.example["metadata"]["generated_environment"]["system_context"]["workspace_id"] == "ws_generated_ops"
    assert result.example["metadata"]["environment"]["passed"] is True


def test_synthchat_generator_loads_environment_generation_scenarios():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=_FakeLLMClient([]),
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    scenario_keys = set(generator.scenario_loader.list_scenarios())
    assert "envfs_update_config_note" in scenario_keys
    assert "envfs_archive_empty_folder" in scenario_keys


def test_synthchat_generator_adds_filterable_labels_for_environment_success():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    environment_json = json.dumps(
        {
            "environment": {
                "fixture": {
                    "notes": [
                        {
                            "path": "Ops/production-config.md",
                            "frontmatter": {"title": "Production Config", "type": "config"},
                            "body": "api_base_url: https://api.old.example.com\n",
                        }
                    ]
                },
                "assertions": [
                    {
                        "type": "file_contains",
                        "path": "Ops/production-config.md",
                        "text": "api_base_url: https://api.prod.example.com",
                    }
                ],
            },
            "system_context": {
                "session_id": "session_1732300800000_envlabels001",
                "workspace_id": "ws_generated_ops",
                "selected_workspace": {"id": "ws_generated_ops", "name": "Operations Workspace"},
            },
        }
    )
    assistant_json = json.dumps(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "useTools",
                        "arguments": {
                            "context": {
                                "sessionId": "session_1732300800000_envlabels001",
                                "workspaceId": "ws_generated_ops",
                                "memory": "Update the production config note.",
                                "goal": "Set the production API base URL.",
                            },
                            "calls": [
                                {
                                    "agent": "contentManager",
                                    "tool": "update",
                                    "params": {
                                        "path": "Ops/production-config.md",
                                        "startLine": 1,
                                        "content": "---\ntitle: Production Config\ntype: config\n---\napi_base_url: https://api.prod.example.com\n",
                                    },
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )

    client = _FakeLLMClient([environment_json, "Update the production config note.", assistant_json])
    validator = EnvironmentValidator(backend="local")
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=validator,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "tool",
        "tool": "contentManager_update",
        "tags": ["vault", "retrieval"],
        "system_template": "mocked_workspace_vault",
        "environment_mode": "generated",
        "environment_generation": {"prompt": "Generate environment JSON."},
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
    }

    result = generator.generate_single(
        scenario_key="env_label_success",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    labels = result.example["metadata"]["labels"]
    flat = set(labels["flat"])
    filter_labels = labels["filter"]
    assert "scenario:env_label_success" in flat
    assert "type:tool" in flat
    assert "environment_mode:generated" in flat
    assert "tool:contentManager_update" in flat
    assert "kto_candidate:positive" in flat
    assert filter_labels["environment_passed"] is True
    assert filter_labels["kto_candidate_label"] is True
    assert filter_labels["executed_tools"] == ["contentManager_update"]


def test_synthchat_generator_provided_environment_mode_skips_generation():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    assistant_json = json.dumps({"content": "No tool call needed."})
    client = _FakeLLMClient(["Use the provided environment as-is.", assistant_json])

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "chat",
        "environment_mode": "provided",
        "system_template": "mocked_workspace_vault",
        "system_context": {
            "workspace_id": "ws_provided_ops",
            "available_workspaces": [
                {"id": "ws_provided_ops", "name": "Provided Workspace", "root_folder": ""}
            ],
            "selected_workspace": {"id": "ws_provided_ops", "name": "Provided Workspace"},
        },
        "environment": {
            "fixture": {
                "files": {
                    "Ops/config.md": "hello from provided env\n",
                }
            }
        },
        "environment_generation": {
            "prompt": "This should never be called.",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant response.",
        },
    }

    result = generator.generate_single(
        scenario_key="provided_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    assert len(client.messages) == 2
    assert result.example["metadata"]["environment_mode"] == "provided"
    assert "generated_environment" not in result.example["metadata"]
    assert "Ops/config.md" in result.example["conversations"][0]["content"]


def test_synthchat_generator_hybrid_environment_mode_merges_base_and_generated_environment():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    environment_json = json.dumps(
        {
            "environment": {
                "fixture": {
                    "files": {
                        "Ops/production.md": "api_base_url: https://api.prod.example.com\n",
                    }
                }
            },
            "system_context": {
                "selected_workspace": {
                    "recent_files": ["Ops/production.md"],
                }
            },
        }
    )
    assistant_json = json.dumps({"content": "Merged environment response."})
    client = _FakeLLMClient([environment_json, "Use the hybrid environment.", assistant_json])

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "chat",
        "environment_mode": "hybrid",
        "system_template": "mocked_workspace_vault",
        "system_context": {
            "workspace_id": "ws_hybrid_ops",
            "available_workspaces": [
                {"id": "ws_hybrid_ops", "name": "Hybrid Workspace", "root_folder": ""}
            ],
            "selected_workspace": {"id": "ws_hybrid_ops", "name": "Hybrid Workspace"},
        },
        "environment": {
            "fixture": {
                "directories": ["Ops"],
                "files": {
                    "Ops/README.md": "baseline readme\n",
                },
            }
        },
        "environment_generation": {
            "prompt": "Generate hybrid environment JSON.",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant response.",
        },
    }

    result = generator.generate_single(
        scenario_key="hybrid_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    system_prompt = result.example["conversations"][0]["content"]
    assert result.example["metadata"]["environment_mode"] == "hybrid"
    assert "generated_environment" in result.example["metadata"]
    assert "Ops/README.md" in system_prompt
    assert "Ops/production.md" in system_prompt


def test_synthchat_generator_labels_behavioral_environment_failures_for_filtering():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    assistant_json = json.dumps(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "useTools",
                        "arguments": {
                            "context": {
                                "sessionId": "session_1732300800000_envlabels002",
                                "workspaceId": "ws_generated_ops",
                                "memory": "Need to triage the rate limit note.",
                                "goal": "Mark the right note as triaged.",
                            },
                            "calls": [
                                {
                                    "agent": "contentManager",
                                    "tool": "update",
                                    "params": {
                                        "path": "Inbox/oauth-cleanup.md",
                                        "content": "status: triaged",
                                        "startLine": 1,
                                        "endLine": 1,
                                    },
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )
    client = _FakeLLMClient(["Find the rate limit note and triage it.", assistant_json])
    validator = EnvironmentValidator(backend="local")
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=validator,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "tool",
        "tool": "contentManager_update",
        "system_template": "mocked_workspace_vault",
        "environment_mode": "provided",
        "environment": {
            "require_expected_tools": True,
            "fixture": {
                "notes": [
                    {
                        "path": "Inbox/api-rate-limits.md",
                        "frontmatter": {"title": "API Rate Limits", "status": "open"},
                        "body": "Repeated 429 responses after deploy.\n",
                    },
                    {
                        "path": "Inbox/oauth-cleanup.md",
                        "frontmatter": {"title": "OAuth Cleanup", "status": "open"},
                        "body": "Remove obsolete redirect URIs.\n",
                    },
                ]
            },
            "assertions": [
                {
                    "type": "frontmatter_field_equals",
                    "path": "Inbox/api-rate-limits.md",
                    "field": "status",
                    "value": "triaged",
                }
            ],
        },
        "expected_tools": ["searchManager_searchContent"],
        "system_context": {
            "session_id": "session_1732300800000_envlabels002",
            "workspace_id": "ws_generated_ops",
            "selected_workspace": {"id": "ws_generated_ops", "name": "Operations Workspace"},
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
    }

    result = generator.generate_single(
        scenario_key="env_label_failure",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    labels = result.example["metadata"]["labels"]
    flat = set(labels["flat"])
    filter_labels = labels["filter"]
    assert "environment_passed:false" in flat
    assert "kto_candidate:negative" in flat
    assert "issue:missing_expected_tool" in flat
    assert "behavior:retrieval_failure" in flat
    assert filter_labels["environment_passed"] is False
    assert filter_labels["kto_candidate_label"] is False
    assert "missing_expected_tool" in filter_labels["issue_labels"]


def test_synthchat_generator_retries_empty_llm_response_for_assistant_stage():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    assistant_json = json.dumps(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "useTools",
                        "arguments": {
                            "context": {
                                "sessionId": "session_1732300800000_retry001",
                                "workspaceId": "ws_retry_ops",
                                "memory": "Update the production config note.",
                                "goal": "Set the production API base URL.",
                            },
                            "calls": [
                                {
                                    "agent": "contentManager",
                                    "tool": "update",
                                    "params": {
                                        "path": "Ops/production-config.md",
                                        "startLine": 1,
                                        "content": "---\ntitle: Production Config\ntype: config\n---\napi_base_url: https://api.prod.example.com\n",
                                    },
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )

    client = _FakeLLMClient(
        [
            "Update the production config note.",
            None,
            assistant_json,
        ]
    )
    validator = EnvironmentValidator(backend="local")
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=validator,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "tool",
        "tool": "contentManager_update",
        "environment_mode": "provided",
        "environment": {
            "fixture": {
                "notes": [
                    {
                        "path": "Ops/production-config.md",
                        "frontmatter": {"title": "Production Config", "type": "config"},
                        "body": "api_base_url: https://api.old.example.com\n",
                    }
                ]
            },
            "assertions": [
                {
                    "type": "file_contains",
                    "path": "Ops/production-config.md",
                    "text": "api_base_url: https://api.prod.example.com",
                }
            ],
        },
        "system_template": "mocked_workspace_vault",
        "system_context": {
            "session_id": "session_1732300800000_retry001",
            "workspace_id": "ws_retry_ops",
            "selected_workspace": {"id": "ws_retry_ops", "name": "Retry Workspace"},
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
    }

    result = generator.generate_single(
        scenario_key="retry_empty_assistant",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    assert len(client.messages) == 3
    assert result.example["metadata"]["environment"]["passed"] is True
    assistant = result.example["conversations"][-1]
    assert assistant["tool_calls"][0]["function"]["name"] == "useTools"


def test_synthchat_generator_uses_structured_environment_and_tool_response_schemas():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    structured_environment = {
        "environment": {
            "fixture": {
                "notes": [
                    {
                        "path": "Inbox/alpha-prototype.md",
                        "frontmatter": {"title": "Alpha Prototype", "status": "inbox"},
                        "body": "Prototype notes that should be preserved.\n",
                    },
                    {
                        "path": "Projects/Alpha/example-project.md",
                        "frontmatter": {"title": "Example Project", "status": "active", "type": "project"},
                        "body": "Example project note.\n",
                    },
                ]
            },
            "assertions": [
                {"type": "path_exists", "path": "Projects/Alpha/alpha-prototype.md"},
                {"type": "path_not_exists", "path": "Inbox/alpha-prototype.md"},
                {"type": "frontmatter_has_key", "path": "Projects/Alpha/alpha-prototype.md", "field": "title"},
                {
                    "type": "file_contains",
                    "path": "Projects/Alpha/alpha-prototype.md",
                    "text": "Prototype notes that should be preserved.",
                },
            ],
        },
        "system_context": {
            "session_id": "session_structured_001",
            "workspace_id": "ws_alpha",
            "selected_workspace": {
                "id": "ws_alpha",
                "name": "Alpha Lab",
                "recent_files": ["Inbox/alpha-prototype.md", "Projects/Alpha/example-project.md"],
                "key_files": ["Projects/Alpha/example-project.md"],
            },
        },
    }

    structured_assistant = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "useTools",
                    "arguments": {
                        "context": {
                            "sessionId": "session_structured_001",
                            "workspaceId": "ws_alpha",
                            "memory": "Need to match the project note format.",
                            "goal": "Promote the inbox note into the Alpha project folder.",
                        },
                        "calls": [
                            {
                                "agent": "contentManager",
                                "tool": "read",
                                "params": {"path": "Projects/Alpha/example-project.md", "startLine": 1},
                            },
                            {
                                "agent": "contentManager",
                                "tool": "write",
                                "params": {
                                    "path": "Projects/Alpha/alpha-prototype.md",
                                    "content": "---\ntitle: Alpha Prototype\nstatus: active\ntype: project\n---\nPrototype notes that should be preserved.\n",
                                },
                            },
                            {
                                "agent": "storageManager",
                                "tool": "archive",
                                "params": {"path": "Inbox/alpha-prototype.md"},
                            },
                        ],
                    },
                },
            }
        ],
    }

    client = _FakeLLMClient(
        responses=["Promote the alpha prototype note into the project folder."],
        structured_responses=[structured_environment, structured_assistant],
    )
    validator = EnvironmentValidator(backend="local")
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=validator,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "tool",
        "tool": "contentManager_write",
        "expected_tools": ["contentManager_read", "contentManager_write", "storageManager_archive"],
        "environment_mode": "generated",
        "system_template": "mocked_workspace_vault",
        "environment_generation": {
            "schema": "canonical_environment",
            "prompt": "Generate the vault fixture and assertions.",
        },
        "assistant_generation": {
            "schema": "use_tools_response",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
    }

    result = generator.generate_single(
        scenario_key="structured_tool_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    assistant = result.example["conversations"][-1]
    assert result.example["metadata"]["scenario"] == "structured_tool_case"
    assert result.example["metadata"]["environment"]["passed"] is True
    assert assistant["tool_calls"][0]["function"]["name"] == "useTools"
    assert len(client.structured_messages) == 2


def test_synthchat_generator_structured_tool_schema_uses_configured_wrapper_name():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    structured_environment = {
        "environment": {
            "fixture": {
                "directories": ["Inbox", "Projects", "Projects/Alpha"],
                "notes": [
                    {"path": "Inbox/alpha-prototype.md", "body": "Prototype notes."},
                    {
                        "path": "Projects/Alpha/example-project.md",
                        "frontmatter": {"title": "Example Project", "status": "active", "type": "project"},
                        "body": "Reference project note.",
                    },
                ],
            },
            "assertions": [
                {"type": "path_exists", "path": "Projects/Alpha/alpha-prototype.md"},
            ],
        },
        "system_context": {
            "session_id": "session_structured_002",
            "workspace_id": "ws_alpha",
            "selected_workspace": {"id": "ws_alpha", "name": "Alpha Lab"},
        },
    }
    structured_assistant = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "batchTools",
                    "arguments": {
                        "context": {
                            "sessionId": "session_structured_002",
                            "workspaceId": "ws_alpha",
                            "memory": "Need to promote the note.",
                            "goal": "Promote the note.",
                        },
                        "calls": [
                            {
                                "agent": "contentManager",
                                "tool": "write",
                                "params": {
                                    "path": "Projects/Alpha/alpha-prototype.md",
                                    "content": "---\ntitle: Alpha Prototype\nstatus: active\ntype: project\n---\nPrototype notes.\n",
                                },
                            }
                        ],
                    },
                },
            }
        ],
    }

    client = _FakeLLMClient(
        responses=["Promote the note."],
        structured_responses=[structured_environment, structured_assistant],
    )
    validator = EnvironmentValidator(backend="local")
    validator.tool_schema["tool_format"]["wrapper"] = "batchTools"
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=validator,
        enable_stage_validation=False,
    )

    scenario = {
        "type": "tool",
        "tool": "contentManager_write",
        "environment_mode": "generated",
        "system_template": "mocked_workspace_vault",
        "environment_generation": {
            "schema": "canonical_environment",
            "prompt": "Generate the vault fixture and assertions.",
        },
        "assistant_generation": {
            "schema": "use_tools_response",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
    }

    result = generator.generate_single(
        scenario_key="structured_wrapper_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    assistant = result.example["conversations"][-1]
    assistant_schema = client.structured_messages[-1]["schema"]
    assert assistant_schema["properties"]["tool_calls"]["items"]["properties"]["function"]["properties"]["name"]["const"] == "batchTools"
    assert "<available_tools>" in result.example["conversations"][0]["content"]
    assert "Use the `batchTools` wrapper for tool calls." in result.example["conversations"][0]["content"]
    assert assistant["tool_calls"][0]["function"]["name"] == "batchTools"
