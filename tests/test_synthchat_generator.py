from __future__ import annotations

import json
from pathlib import Path

import yaml

from SynthChat.generator import SynthChatGenerator
from SynthChat.schemas.tool_response_schema import build_tool_generation_prompt, build_tool_response_schema
from SynthChat.config.format_resolver import get_default_tool_call_format
from shared.agentic_judge import AgenticTurnJudge
from shared.environments import EnvironmentValidator


def _tool_calls_array_schema(schema: dict) -> dict:
    tool_calls_schema = schema["properties"]["tool_calls"]
    for option in tool_calls_schema.get("anyOf", []):
        if isinstance(option, dict) and option.get("type") == "array" and option.get("minItems") == 1:
            return option
    return tool_calls_schema


class _FakeLLMClient:
    def __init__(self, responses=None, structured_responses=None, provider_name="openrouter", model_name="fake/model"):
        self._responses = list(responses or [])
        self._structured_responses = list(structured_responses or [])
        self.messages = []
        self.structured_messages = []
        self.default_max_tokens = None
        self._provider_name = provider_name
        self._model_name = model_name
        self.provider = None
        self.timeout_seconds = 60.0

    @property
    def provider_name(self):
        return self._provider_name

    @property
    def model_name(self):
        return self._model_name

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


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, message):
        self.infos.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def error(self, message):
        self.errors.append(str(message))


class _RetryJudgeClient:
    def __init__(self):
        self.calls = 0

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("transient judge failure")
        return {
            "passed": True,
            "hard_failure": False,
            "should_stop": False,
            "feedback_to_model": "Looks good.",
            "feedback_for_trace": "Recovered after retries.",
        }


class _AlwaysFailStructuredClient(_FakeLLMClient):
    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        self.structured_messages.append(
            {
                "messages": messages,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        raise RuntimeError("primary failed")


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


def test_agentic_turn_judge_retries_before_failing():
    judge = AgenticTurnJudge(
        llm_client=_RetryJudgeClient(),
        prompt_template="Judge this.",
        max_retries=3,
    )

    result = judge.judge({"foo": "bar"})

    assert result.passed is True
    assert result.feedback_to_model == "Looks good."
    assert result.feedback_for_trace == "Recovered after retries."


def test_structured_generation_can_fallback_to_secondary_model(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    primary = _AlwaysFailStructuredClient(model_name="inception/mercury-2")
    fallback = _FakeLLMClient(
        structured_responses=[{"ok": True}],
        model_name="google/gemini-3.1-flash-lite-preview",
    )

    def _fake_create_client(provider=None, model=None, config=None, env_prefix="IMPROVEMENT", config_defaults=None):
        return fallback

    monkeypatch.setattr("SynthChat.generator.create_client", _fake_create_client)

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=primary,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    llm_clients = generator._get_stage_llm_clients(
        {
            "fallback_models": [
                {"model": "google/gemini-3.1-flash-lite-preview"},
            ]
        }
    )

    result = generator._call_llm_structured(
        prompt="Return JSON.",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        randomize=False,
        llm_clients=llm_clients,
        max_retries=1,
    )

    assert result == {"ok": True}
    assert len(primary.structured_messages) == 1
    assert len(fallback.structured_messages) == 1


def test_structured_generation_uses_configured_max_tokens():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Write a simple user request."],
        structured_responses=[
            {
                "environment": {
                    "fixture": {"directories": [], "files": []},
                    "assertions": [],
                },
                "task_context": {"target_path": "foo.md"},
            },
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
                                    "sessionId": "session_eval_001",
                                    "workspaceId": "default",
                                    "memory": "",
                                    "goal": "Do the thing",
                                },
                                "calls": [],
                            },
                        },
                    }
                ],
            },
        ],
    )
    client.default_max_tokens = 8192

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
        "type": "tool",
        "environment_mode": "generated",
        "environment_generation": {
            "schema": "canonical_environment",
            "prompt": "Generate environment.",
            "max_tokens": 12000,
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant tool JSON.",
        },
        "assistant_generation": {
            "schema": "use_tools_response",
            "max_tokens": 9000,
        },
    }

    generator.generate_single(
        scenario_key="configurable_limits",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    assert client.structured_messages[0]["max_tokens"] == 12000
    assert client.structured_messages[1]["max_tokens"] == 9000


def test_use_tools_response_schema_allows_text_only_responses():
    fmt = get_default_tool_call_format()
    schema = build_tool_response_schema(
        format_config=fmt,
        allowed_tools=["contentManager_write"],
        context_overrides={"sessionId": "session_001", "workspaceId": "ws_001"},
    )

    tool_calls_schema = schema["properties"]["tool_calls"]
    assert "anyOf" in tool_calls_schema
    assert {"type": "null"} in tool_calls_schema["anyOf"]
    assert any(
        isinstance(option, dict)
        and option.get("type") == "array"
        and option.get("maxItems") == 0
        for option in tool_calls_schema["anyOf"]
    )


def test_use_tools_generation_prompt_explicitly_allows_text_or_tools():
    fmt = get_default_tool_call_format()
    prompt = build_tool_generation_prompt(
        format_config=fmt,
        base_prompt="Continue the task.",
        allowed_tools=["contentManager_write"],
    )

    assert "either call tools or respond via text" in prompt
    assert "set tool_calls to null or []" in prompt
    assert "When the task is already complete" in prompt


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


def test_generated_environment_mode_preserves_base_environment_loop_config():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    environment_json = json.dumps(
        {
            "environment": {
                "fixture": {
                    "directories": ["Inbox"],
                },
                "assertions": [{"type": "path_exists", "path": "Inbox"}],
            },
            "system_context": {
                "selected_workspace": {"id": "ws_generated_loop", "name": "Generated Loop"},
            },
        }
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=_FakeLLMClient([environment_json]),
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    seed = generator.prepare_seed_bundle(
        scenario_key="generated_loop_case",
        seed_id="generated_loop_case:seed:1",
        scenario={
            "type": "tool",
            "environment_mode": "generated",
            "environment": {
                "loop": {
                    "enabled": True,
                    "mode": "agentic",
                    "max_turns": 4,
                }
            },
            "environment_generation": {
                "prompt": "Generate environment JSON.",
            },
        },
        randomize_params=False,
    )

    assert seed["resolved_environment_config"]["loop"]["enabled"] is True
    assert seed["resolved_environment_config"]["loop"]["mode"] == "agentic"
    assert seed["resolved_environment_config"]["fixture"]["directories"] == ["Inbox"]


def test_canonical_environment_generation_prompt_includes_contract():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=_FakeLLMClient(structured_responses=[{"environment": {"fixture": {"directories": ["Inbox"]}}}]),
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    generator.prepare_seed_bundle(
        scenario_key="contract_case",
        seed_id="contract_case:seed:1",
        scenario={
            "type": "tool",
            "environment_mode": "generated",
            "environment_generation": {
                "schema": "canonical_environment",
                "prompt": "Generate a simple environment.",
            },
        },
        randomize_params=False,
    )

    prompt = generator.llm_client.structured_messages[-1]["messages"][-1]["content"]
    assert "Top-level keys allowed: environment, system_context, task_context." in prompt
    assert "fixture may contain: directories, files, notes, local_path, source." in prompt
    assert "Use only these assertion types:" in prompt
    assert "- frontmatter_field_equals" in prompt
    assert "Task:\nGenerate a simple environment." in prompt


def test_generated_task_context_flows_into_prompt_templates_and_metadata():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    environment_json = json.dumps(
        {
            "environment": {
                "fixture": {"directories": ["Journal/Daily"]},
                "assertions": [{"type": "path_exists", "path": "Journal/Daily/2026-03-15.md"}],
            },
            "system_context": {
                "session_id": "session_task_ctx",
                "workspace_id": "ws_task_ctx",
                "selected_workspace": {"id": "ws_task_ctx", "name": "Task Context Workspace"},
            },
            "task_context": {
                "target_output_path": "Journal/Daily/2026-03-15.md",
                "target_date": "2026-03-15",
            },
        }
    )
    client = _FakeLLMClient(
        [
            environment_json,
            "Create the note at Journal/Daily/2026-03-15.md for 2026-03-15.",
            json.dumps({"content": "Done."}),
        ]
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="task_context_case",
        scenario={
            "type": "chat",
            "environment_mode": "generated",
            "system_template": "mocked_workspace_vault",
            "environment_generation": {"prompt": "Generate environment JSON."},
            "prompts": {
                "user": "Create the note at {task_target_output_path} for {task_target_date}.",
                "assistant": "Generate assistant response.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    assert result.example["conversations"][1]["content"] == "Create the note at Journal/Daily/2026-03-15.md for 2026-03-15."
    assert result.example["metadata"]["task_context"]["target_output_path"] == "Journal/Daily/2026-03-15.md"


def test_generate_batch_invokes_callback_for_each_completed_result():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        [
            "First user request.",
            json.dumps({"content": "First assistant response."}),
            "Second user request.",
            json.dumps({"content": "Second assistant response."}),
        ]
    )

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    generator.scenario_loader.scenarios = {
        "case_one": {
            "type": "chat",
            "prompts": {
                "user": "Generate first user request.",
                "assistant": "Generate first assistant response.",
            },
        },
        "case_two": {
            "type": "chat",
            "prompts": {
                "user": "Generate second user request.",
                "assistant": "Generate second assistant response.",
            },
        },
    }

    seen = []
    results = generator.generate_batch(
        targets={"case_one": 1, "case_two": 1},
        max_iterations=1,
        randomize_params=False,
        on_result=lambda result: seen.append(result.scenario_key),
    )

    assert [result.scenario_key for result in results] == ["case_one", "case_two"]
    assert seen == ["case_one", "case_two"]


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
    tool_calls_schema = _tool_calls_array_schema(assistant_schema)
    assert tool_calls_schema["items"]["properties"]["function"]["properties"]["name"]["const"] == "batchTools"
    assert "<available_tools>" in result.example["conversations"][0]["content"]
    assert "Use the `batchTools` wrapper for tool calls." in result.example["conversations"][0]["content"]
    assert assistant["tool_calls"][0]["function"]["name"] == "batchTools"


def test_use_tools_schema_is_constrained_to_single_wrapper_and_allowed_tools():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    structured_environment = {
        "environment": {
            "fixture": {
                "directories": ["Ops"],
                "files": {"Ops/production.md": "api_base_url: https://api.old.example.com\n"},
            },
            "assertions": [{"type": "file_contains", "path": "Ops/production.md", "text": "api.prod.example.com"}],
        },
        "system_context": {
            "session_id": "session_structured_003",
            "workspace_id": "ws_ops",
            "selected_workspace": {"id": "ws_ops", "name": "Ops"},
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
                            "sessionId": "session_structured_003",
                            "workspaceId": "ws_ops",
                            "memory": "Need to inspect then update the ops note.",
                            "goal": "Update the production note.",
                        },
                        "calls": [
                            {
                                "agent": "searchManager",
                                "tool": "searchContent",
                                "params": {"query": "api.old.example.com", "limit": 5},
                            },
                            {
                                "agent": "contentManager",
                                "tool": "read",
                                "params": {"path": "Ops/production.md", "startLine": 1},
                            },
                        ],
                    },
                },
            }
        ],
    }

    client = _FakeLLMClient(
        responses=["Update the production note."],
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
        "tool": "contentManager_update",
        "expected_tools": ["searchManager_searchContent", "contentManager_read"],
        "environment_mode": "generated",
        "system_template": "mocked_workspace_vault",
        "environment_generation": {
            "schema": "canonical_environment",
            "prompt": "Generate the ops environment.",
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
        scenario_key="structured_constraints_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    assistant_schema = client.structured_messages[-1]["schema"]
    tool_calls_schema = _tool_calls_array_schema(assistant_schema)
    function_args = tool_calls_schema["items"]["properties"]["function"]["properties"]["arguments"]
    inner_call = function_args["properties"]["calls"]["items"]["properties"]
    context_props = function_args["properties"]["context"]["properties"]

    assert tool_calls_schema["maxItems"] == 1
    assert inner_call["agent"]["enum"] == ["contentManager", "searchManager"]
    assert inner_call["tool"]["enum"] == ["read", "searchContent", "update"]
    assert context_props["sessionId"]["const"] == "session_structured_003"
    assert context_props["workspaceId"]["const"] == "ws_ops"
    assert result.example["conversations"][-1]["tool_calls"][0]["function"]["name"] == "useTools"


def test_generate_batch_reuses_environment_seed_across_rollouts():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    structured_seed_a = {
        "environment": {
            "fixture": {"files": {"Inbox/a.md": "alpha\n"}},
            "assertions": [{"type": "path_exists", "path": "Inbox/a.md"}],
        },
        "system_context": {
            "session_id": "session_seed_a",
            "workspace_id": "ws_seed_a",
            "selected_workspace": {"id": "ws_seed_a", "name": "Seed A"},
        },
    }
    structured_seed_b = {
        "environment": {
            "fixture": {"files": {"Inbox/b.md": "beta\n"}},
            "assertions": [{"type": "path_exists", "path": "Inbox/b.md"}],
        },
        "system_context": {
            "session_id": "session_seed_b",
            "workspace_id": "ws_seed_b",
            "selected_workspace": {"id": "ws_seed_b", "name": "Seed B"},
        },
    }

    client = _FakeLLMClient(
        responses=[
            "User request 1", "Assistant response 1",
            "User request 2", "Assistant response 2",
            "User request 3", "Assistant response 3",
            "User request 4", "Assistant response 4",
        ],
        structured_responses=[structured_seed_a, structured_seed_b],
    )
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
        "type": "behavioral",
        "environment_mode": "generated",
        "system_template": "mocked_workspace_vault",
        "environment_generation": {
            "schema": "canonical_environment",
            "prompt": "Generate a small environment.",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant response.",
        },
    }
    generator.scenario_loader.scenarios["seeded_case"] = scenario

    results = generator.generate_batch(
        targets={"seeded_case": {"seed_count": 2, "rollouts_per_seed": 2}},
        max_iterations=1,
        randomize_params=False,
    )

    environment_structured_calls = [
        item
        for item in client.structured_messages
        if "environment" in item["schema"].get("properties", {})
    ]
    assert len(results) == 4
    assert len(environment_structured_calls) == 2
    assert results[0].example["metadata"]["environment_seed"]["seed_id"] == "seeded_case:seed:1"
    assert results[1].example["metadata"]["environment_seed"]["seed_id"] == "seeded_case:seed:1"
    assert results[2].example["metadata"]["environment_seed"]["seed_id"] == "seeded_case:seed:2"
    assert results[3].example["metadata"]["environment_seed"]["seed_id"] == "seeded_case:seed:2"


def test_generate_batch_logs_environment_generation_once_per_seed():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    structured_seed = {
        "environment": {
            "fixture": {"files": {"Inbox/a.md": "alpha\n"}},
            "assertions": [{"type": "path_exists", "path": "Inbox/a.md"}],
        },
        "system_context": {
            "session_id": "session_seed_a",
            "workspace_id": "ws_seed_a",
            "selected_workspace": {"id": "ws_seed_a", "name": "Seed A"},
        },
    }

    client = _FakeLLMClient(
        responses=["User request 1", "Assistant response 1", "User request 2", "Assistant response 2"],
        structured_responses=[structured_seed],
    )
    logger = _FakeLogger()
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
        logger=logger,
    )
    scenario = {
        "type": "behavioral",
        "environment_mode": "generated",
        "system_template": "mocked_workspace_vault",
        "environment_generation": {
            "schema": "canonical_environment",
            "prompt": "Generate a small environment.",
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Generate assistant response.",
        },
    }
    generator.scenario_loader.scenarios["seeded_case_logging"] = scenario

    generator.generate_batch(
        targets={"seeded_case_logging": {"seed_count": 1, "rollouts_per_seed": 2}},
        max_iterations=1,
        randomize_params=False,
    )

    starts = [msg for msg in logger.infos if "[seeded_case_logging] environment_generation start" in msg]
    dones = [msg for msg in logger.infos if "[seeded_case_logging] environment_generation done" in msg]
    structured_starts = [msg for msg in logger.infos if "LLM structured start [seeded_case_logging:environment_generation:seeded_case_logging:seed:1]" in msg]

    assert len(starts) == 1
    assert len(dones) == 1
    assert "seed_id=seeded_case_logging:seed:1" in starts[0]
    assert "seed_id=seeded_case_logging:seed:1" in dones[0]
    assert len(structured_starts) == 1


def test_synthchat_generator_can_run_shared_agentic_loop():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=[
            "Create today's daily note from the template.",
            "Done. The daily note is created.",
        ],
        structured_responses=[
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
                                    "sessionId": "session_loop_001",
                                    "workspaceId": "ws_loop_001",
                                    "memory": "Need to find the template first.",
                                    "goal": "Create the daily note.",
                                },
                                "calls": [
                                    {
                                        "agent": "searchManager",
                                        "tool": "searchDirectory",
                                        "params": {"query": "daily-note", "paths": ["Templates/"]},
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_002",
                        "type": "function",
                        "function": {
                            "name": "useTools",
                            "arguments": {
                                "context": {
                                    "sessionId": "session_loop_001",
                                    "workspaceId": "ws_loop_001",
                                    "memory": "Template located; now create the note.",
                                    "goal": "Create the daily note.",
                                },
                                "calls": [
                                    {
                                        "agent": "contentManager",
                                        "tool": "write",
                                        "params": {
                                            "path": "Journal/Daily/2026-03-15.md",
                                            "content": "---\ntitle: 2026-03-15\ntype: daily\n---\n## Summary\nReady.\n",
                                            "overwrite": True,
                                        },
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
        ],
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
        "expected_tools": ["searchManager_searchDirectory", "contentManager_write"],
        "environment_mode": "provided",
        "system_template": "mocked_workspace_vault",
        "assistant_generation": {"schema": "use_tools_response"},
        "environment": {
            "loop": {
                "enabled": True,
                "mode": "agentic",
                "max_turns": 4,
                "max_tool_steps": 4,
                "stop_on_text_response": True,
                "stop_on_environment_pass": True,
            },
            "max_steps": 4,
            "fixture": {
                "directories": ["Templates", "Journal/Daily"],
                "notes": [
                    {
                        "path": "Templates/daily-note.md",
                        "frontmatter": {"title": "Daily Note Template", "type": "daily"},
                        "body": "## Summary\nTemplate.\n",
                    }
                ],
            },
            "assertions": [
                {"type": "path_exists", "path": "Journal/Daily/2026-03-15.md"},
                {"type": "file_contains", "path": "Journal/Daily/2026-03-15.md", "text": "## Summary"},
            ],
        },
        "system_context": {
            "session_id": "session_loop_001",
            "workspace_id": "ws_loop_001",
            "selected_workspace": {"id": "ws_loop_001", "name": "Loop Workspace"},
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Continue the task using tools when needed.",
        },
    }

    result = generator.generate_single(
        scenario_key="looped_case",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    env_trace = result.example["metadata"]["environment"]
    assert env_trace["passed"] is True
    assert env_trace["episode_trace"]["total_turns"] == 2
    assert any(entry["kind"] == "tool_feedback" for entry in result.example["conversation_trace"])


def test_generate_single_can_use_turn_judge_and_require_final_text():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=[
            "Update the file and let me know when it's done.",
        ],
        structured_responses=[
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
                                    "sessionId": "session_loop_judge_001",
                                    "workspaceId": "ws_loop_judge_001",
                                    "memory": "Need to update the file.",
                                    "goal": "Write the final file.",
                                },
                                "calls": [
                                    {
                                        "agent": "contentManager",
                                        "tool": "write",
                                        "params": {
                                            "path": "Inbox/final.md",
                                            "content": "complete",
                                            "overwrite": True,
                                        },
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
            {
                "passed": True,
                "hard_failure": False,
                "should_stop": False,
                "feedback_to_model": "Before ending, send the user a brief text confirmation.",
                "feedback_for_trace": "The task is complete; the next turn should be a text-only confirmation.",
            },
            {
                "content": "Done. I updated the file.",
                "tool_calls": None,
            },
            {
                "passed": True,
                "hard_failure": False,
                "should_stop": False,
                "feedback_to_model": "",
                "feedback_for_trace": "Final text response was present.",
            },
        ],
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
        "expected_tools": ["contentManager_write"],
        "environment_mode": "provided",
        "system_template": "mocked_workspace_vault",
        "assistant_generation": {"schema": "use_tools_response"},
        "environment": {
            "loop": {
                "enabled": True,
                "mode": "agentic",
                "max_turns": 4,
                "max_tool_steps": 2,
                "stop_on_environment_pass": True,
                "require_final_text_after_pass": True,
                "final_text_prompt": "Reply to the user with a brief final text-only response.",
            },
            "max_steps": 2,
            "fixture": {"directories": ["Inbox"]},
            "assertions": [{"type": "path_exists", "path": "Inbox/final.md"}],
            "allowed_tools": ["contentManager_write"],
        },
        "system_context": {
            "session_id": "session_loop_judge_001",
            "workspace_id": "ws_loop_judge_001",
            "selected_workspace": {"id": "ws_loop_judge_001", "name": "Loop Workspace"},
        },
        "judge": {
            "in_loop": {
                "enabled": True,
                "feedback_visible_to_model": True,
                "prompt": (
                    "Review the latest step.\n"
                    "Assistant response:\n{assistant_response_json}\n\n"
                    "Environment step:\n{environment_step_json}\n\n"
                    "Tool feedback:\n{tool_feedback}\n"
                ),
            }
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Continue the task using tools when needed.",
        },
    }

    result = generator.generate_single(
        scenario_key="looped_case_with_judge",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    env_trace = result.example["metadata"]["environment"]
    judge_meta = result.example["metadata"]["judge"]
    assert env_trace["passed"] is True
    assert env_trace["final_text_required"] is True
    assert env_trace["final_text_satisfied"] is True
    assert env_trace["episode_trace"]["stop_reason"] == "environment_passed_final_text"
    assert len(judge_meta["trace"]) == 2
    assert any(entry["kind"] == "judge_feedback" for entry in result.example["conversation_trace"])
    assert any(entry["kind"] == "final_text_request" for entry in result.example["conversation_trace"])
    assert result.example["conversations"][-1]["content"] == "Done. I updated the file."


def test_turn_judge_should_stop_does_not_preempt_environment_stop_conditions():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=[
            "Update the file and let me know when it's done.",
        ],
        structured_responses=[
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
                                    "sessionId": "session_loop_judge_002",
                                    "workspaceId": "ws_loop_judge_002",
                                    "memory": "Need to update the file.",
                                    "goal": "Write the final file.",
                                },
                                "calls": [
                                    {
                                        "agent": "contentManager",
                                        "tool": "write",
                                        "params": {
                                            "path": "Inbox/final.md",
                                            "content": "complete",
                                            "overwrite": True,
                                        },
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
            {
                "passed": True,
                "hard_failure": False,
                "should_stop": True,
                "feedback_to_model": "You may be done, but do not stop until the environment says so.",
                "feedback_for_trace": "Judge would have stopped here, but loop must continue until programmatic success.",
            },
            {
                "content": "Done. I updated the file.",
                "tool_calls": None,
            },
            {
                "passed": True,
                "hard_failure": False,
                "should_stop": False,
                "feedback_to_model": "",
                "feedback_for_trace": "Final text response was present.",
            },
        ],
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
        "expected_tools": ["contentManager_write"],
        "environment_mode": "provided",
        "system_template": "mocked_workspace_vault",
        "assistant_generation": {"schema": "use_tools_response"},
        "environment": {
            "loop": {
                "enabled": True,
                "mode": "agentic",
                "max_turns": 4,
                "max_tool_steps": 2,
                "stop_on_environment_pass": True,
                "require_final_text_after_pass": True,
                "final_text_prompt": "Reply to the user with a brief final text-only response.",
            },
            "max_steps": 2,
            "fixture": {"directories": ["Inbox"]},
            "assertions": [{"type": "path_exists", "path": "Inbox/final.md"}],
            "allowed_tools": ["contentManager_write"],
        },
        "system_context": {
            "session_id": "session_loop_judge_002",
            "workspace_id": "ws_loop_judge_002",
            "selected_workspace": {"id": "ws_loop_judge_002", "name": "Loop Workspace"},
        },
        "judge": {
            "in_loop": {
                "enabled": True,
                "feedback_visible_to_model": True,
                "prompt": (
                    "Review the latest step.\n"
                    "Assistant response:\n{assistant_response_json}\n\n"
                    "Environment step:\n{environment_step_json}\n\n"
                    "Tool feedback:\n{tool_feedback}\n"
                ),
            }
        },
        "prompts": {
            "user": "Generate user request.",
            "assistant": "Continue the task using tools when needed.",
        },
    }

    result = generator.generate_single(
        scenario_key="looped_case_with_judge_should_stop_ignored",
        scenario=scenario,
        max_iterations=1,
        randomize_params=False,
    )

    env_trace = result.example["metadata"]["environment"]
    assert env_trace["passed"] is True
    assert env_trace["episode_trace"]["stop_reason"] == "environment_passed_final_text"
    assert result.example["metadata"]["judge"]["trace"][0]["should_stop"] is True


def test_generated_environment_normalizes_assertion_aliases():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=_FakeLLMClient(),
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    normalized = generator._normalize_generated_environment(
        {
            "environment": {
                "fixture": {"files": [{"path": "Settings/settings.yaml", "content": "x: 1\n"}]},
                "assertions": [
                    {"type": "file_contains", "path": "Settings/settings.yaml", "content": "x: 1"},
                    {"type": "file_not_contains", "path": "Settings/settings.yaml", "content": "y: 2"},
                    {"type": "frontmatter_has_key", "path": "Inbox/note.md", "key": "title"},
                    {"type": "frontmatter_field_equals", "path": "Inbox/note.md", "key": "status", "content": "open"},
                ],
            }
        }
    )

    assertions = normalized["environment"]["assertions"]
    assert assertions[0]["text"] == "x: 1"
    assert "content" not in assertions[0]
    assert assertions[1]["text"] == "y: 2"
    assert assertions[2]["field"] == "title"
    assert "key" not in assertions[2]
    assert assertions[3]["field"] == "status"
    assert assertions[3]["value"] == "open"


def test_vault_pilot_promote_scenario_allows_source_removal():
    repo_root = Path(__file__).resolve().parents[1]
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    payload = yaml.safe_load((scenarios_dir / "vault_kto_pilot.yaml").read_text())

    scenario = payload["scenarios"]["envfs_promote_inbox_note_with_example"]
    assert "storageManager_archive" in scenario["expected_tools"]
    assert "remove or archive the original inbox note" in scenario["prompts"]["assistant"]


def test_generate_batch_reuses_shared_seed_across_scenarios():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    shared_environment = {
        "environment": {
            "fixture": {
                "files": [
                    {
                        "path": "Shared/common.md",
                        "content": "---\ntitle: Shared\n---\n\nCommon content.\n",
                    }
                ],
                "directories": ["Shared"],
            },
            "assertions": [],
        },
        "system_context": {
            "session_id": "session_shared_001",
            "workspace_id": "ws_shared_001",
            "selected_workspace": {"id": "ws_shared_001", "name": "Shared Workspace"},
        },
        "task_context": {
            "shared_path": "Shared/common.md",
        },
    }

    client = _FakeLLMClient(
        responses=[
            "User A", "Assistant A",
            "User B", "Assistant B",
        ],
        structured_responses=[shared_environment],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    generator.scenario_loader.scenarios.update(
        {
            "shared_seed_case": {
                "type": "helper",
                "environment_mode": "generated",
                "environment_generation": {
                    "schema": "canonical_environment",
                    "prompt": "Generate the shared environment.",
                },
                "prompts": {
                    "user": "",
                    "assistant": "",
                },
            },
            "shared_target_a": {
                "type": "behavioral",
                "environment_mode": "provided",
                "task_context": {"scenario_path": "Shared/a.md"},
                "prompts": {
                    "user": "Reference {task_shared_path} and {task_scenario_path}.",
                    "assistant": "Reply with plain text.",
                },
            },
            "shared_target_b": {
                "type": "behavioral",
                "environment_mode": "provided",
                "task_context": {"scenario_path": "Shared/b.md"},
                "prompts": {
                    "user": "Reference {task_shared_path} and {task_scenario_path}.",
                    "assistant": "Reply with plain text.",
                },
            },
        }
    )

    results = generator.generate_batch(
        targets={
            "shared_target_a": {"seed_count": 1, "rollouts_per_seed": 1},
            "shared_target_b": {"seed_count": 1, "rollouts_per_seed": 1},
        },
        max_iterations=1,
        randomize_params=False,
        shared_seed_spec={"scenario": "shared_seed_case", "seed_count": 1},
    )

    assert len(results) == 2
    assert results[0].example["metadata"]["environment_seed"]["seed_id"] == "shared_seed_case:shared_seed:1"
    assert results[1].example["metadata"]["environment_seed"]["seed_id"] == "shared_seed_case:shared_seed:1"
    assert results[0].example["metadata"]["environment_seed"]["shared_across_scenarios"] is True
    assert results[1].example["metadata"]["environment_seed"]["shared_across_scenarios"] is True
    assert results[0].example["metadata"]["task_context"]["shared_path"] == "Shared/common.md"
    assert results[0].example["metadata"]["task_context"]["scenario_path"] == "Shared/a.md"
    assert results[1].example["metadata"]["task_context"]["scenario_path"] == "Shared/b.md"


def test_generate_single_derives_task_context_and_renders_environment_assertions():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please update the production policy note.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="derived_task_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {
                "fixture": {
                    "files": {
                        "Ops/production-policy.md": "base_url: http://old.api.internal\nretries: 3",
                        "Ops/staging-policy.md": "base_url: http://staging.api.internal\nretries: 3",
                    }
                },
                "assertions": [
                    {"type": "file_contains", "path": "{task_target_path}", "text": "{task_new_value}"},
                    {"type": "file_not_contains_any", "path": "{task_target_path}", "texts": ["{task_old_value}"]},
                ],
            },
            "task_family": {
                "kind": "replace_text_in_file",
                "target_contains": "production-policy",
                "distractor_contains": "staging-policy",
                "new_value": "https://api.prod.example.com",
            },
            "prompts": {
                "user": "Update the policy note to replace {task_old_value} with {task_new_value}.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    assert result.example["metadata"]["task_context"]["target_path"] == "Ops/production-policy.md"
    assert result.example["metadata"]["task_context"]["distractor_path"] == "Ops/staging-policy.md"
    assert result.example["conversations"][0]["content"] == "Please update the production policy note."


def test_generate_single_stores_hard_requirements_and_quality_rubric():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please clarify which note.", json.dumps({"content": "Which meeting notes do you mean?"})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="quality_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {
                "fixture": {
                    "files": {
                        "Projects/Alpha/Meeting Notes.md": "# Alpha\n",
                        "Projects/Beta/Meeting Notes.md": "# Beta\n",
                    }
                },
                "assertions": [],
            },
            "task_family": {"kind": "ambiguous_reference"},
            "prompts": {
                "user": "Ask about the note called {task_ambiguous_reference}.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    assert result.example["metadata"]["hard_requirements"] == [{"type": "ask_for_clarification"}]
    assert "Ask a short clarifying question." in result.example["metadata"]["quality_rubric"]


def test_promote_task_derivation_prefers_plain_inbox_note_over_status_ticket():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Promote the plain inbox note.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="promote_derivation_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {
                "fixture": {
                    "files": {
                        "Inbox/429-rate-limit.md": "---\ntitle: 429 Rate Limit\nstatus: open\n---",
                        "Inbox/Task to Promote.md": "This is the body of the inbox note.",
                        "Projects/Alpha/Example Project Note.md": "---\ntitle: Example Project Note\nstatus: active\n---",
                    }
                },
                "assertions": [],
            },
            "task_family": {
                "kind": "promote_from_example",
                "source_startswith": "Inbox/",
                "example_contains": "Example Project Note",
            },
            "prompts": {
                "user": "Promote the inbox note about {task_source_note_label}.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    task_context = result.example["metadata"]["task_context"]
    assert task_context["source_note_path"] == "Inbox/Task to Promote.md"
    assert task_context["target_note_path"] == "Projects/Alpha/Task to Promote.md"
    assert task_context["source_note_user_reference"] == "task to promote"
    assert task_context["target_folder_user_reference"] == "the Alpha project folder"
    assert task_context["example_frontmatter_keys"] == ["title", "status"]
    assert {"type": "target_frontmatter_keys_present", "keys": ["title", "status"]} in result.example["metadata"]["hard_requirements"]
    assert not any(req.get("type") == "must_read_before_move" for req in result.example["metadata"]["hard_requirements"])
    assert "Read the source note before moving or promoting it." in result.example["metadata"]["quality_rubric"]

def test_edit_task_derivation_keeps_read_before_edit_hard_requirement():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please update the config.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="edit_quality_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {
                "fixture": {
                    "files": {
                        "Ops/production-config.md": "base_url: http://old.api.internal\nretries: 3",
                        "Ops/staging-config.md": "base_url: http://staging.api.internal\nretries: 3",
                    }
                },
                "assertions": [],
            },
            "task_family": {
                "kind": "edit_file",
                "must_read_before_edit": True,
                "candidate_selector": {
                    "target": {"contains": "production-config", "path_prefix": "Ops/"},
                    "distractors": {"contains": "staging-config"},
                },
                "new_value": "https://api.prod.example.com",
                "edit_mode": "text_replace",
            },
            "prompts": {
                "user": "Update the config note.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    hard_requirements = result.example["metadata"]["hard_requirements"]
    quality_rubric = result.example["metadata"]["quality_rubric"]
    assert any(req.get("type") == "must_read_before_edit" for req in hard_requirements)
    assert "Read the target file before editing it." not in quality_rubric


def test_triage_derivation_can_select_semantic_issue_notes_without_anchored_names():
    from SynthChat.task_derivation import derive_task_spec

    scenario = {
        "task_family": {
            "kind": "triage_note",
            "new_status": "triaged",
            "distractor_status": "open",
            "note_selector": {
                "target": {
                    "path_prefix": "Inbox/",
                    "require_frontmatter": True,
                    "require_frontmatter_statuses": ["open"],
                    "preferred_keywords": ["rate", "limit", "api"],
                    "must_not_contain_text": ["login"],
                },
                "distractor": {
                    "path_prefix": "Inbox/",
                    "require_frontmatter": True,
                    "require_frontmatter_statuses": ["open"],
                    "preferred_keywords": ["login", "auth"],
                },
            },
        }
    }
    environment_config = {
        "fixture": {
            "files": {
                "Inbox/billing-throttle.md": "---\ntitle: Billing API Rate Spike\nstatus: open\n---\nCustomers are hitting a rate limit.",
                "Inbox/auth-portal.md": "---\ntitle: Login Portal Error\nstatus: open\n---\nUsers cannot log in.",
            }
        }
    }

    result = derive_task_spec(
        scenario_key="triage_dynamic_case",
        scenario=scenario,
        environment_config=environment_config,
    )

    assert result.task_context["target_note_path"] == "Inbox/billing-throttle.md"
    assert result.task_context["distractor_note_path"] == "Inbox/auth-portal.md"
    assert result.task_context["target_phrase"] == "billing api rate spike"


def test_promote_derivation_prefers_example_note_with_status_frontmatter():
    from SynthChat.task_derivation import derive_task_spec

    scenario = {
        "task_family": {
            "kind": "move_file",
            "candidate_selector": {
                "source": {
                    "path_prefix": "Inbox/",
                    "require_plain_body": True,
                },
                "example": {
                    "path_prefix": "Projects/",
                    "require_frontmatter": True,
                    "preferred_keywords": ["project", "note"],
                },
                "destination": {
                    "path_prefix": "Projects/",
                    "require_missing_target_name": True,
                },
            },
        }
    }
    environment_config = {
        "fixture": {
            "files": {
                "Inbox/quick-idea.md": "# Quick Idea\n\nIdea body.",
                "Projects/project-reference.md": "---\ntitle: Project Reference\n---\nReference body.",
                "Projects/Alpha/example-project.md": "---\ntitle: Example Project\nstatus: active\n---\nExample body.",
            }
        }
    }

    result = derive_task_spec(
        scenario_key="promote_example_case",
        scenario=scenario,
        environment_config=environment_config,
    )

    assert result.task_context["example_note_path"] == "Projects/Alpha/example-project.md"
    assert result.task_context["example_frontmatter_keys"] == ["title", "status"]


def test_daily_note_derivation_uses_template_frontmatter_keys_as_requirements():
    from SynthChat.task_derivation import derive_task_spec

    scenario = {
        "task_family": {
            "kind": "daily_note_from_template",
            "template_selector": {
                "path_prefix": "Templates/",
                "require_frontmatter": True,
                "must_contain_text": ["## Summary", "## Tasks"],
            },
            "link_selector": {
                "path_prefix": "Projects/",
                "root_only": True,
            },
        }
    }
    environment_config = {
        "fixture": {
            "files": {
                "Templates/daily-template.md": "---\ntitle: Daily Note\ndate: '{{date}}'\ntemplate: daily\n---\n## Summary\n\n## Tasks\n",
                "Projects/project-overview.md": "# Project Overview\n",
            }
        }
    }

    result = derive_task_spec(
        scenario_key="daily_note_dynamic_case",
        scenario=scenario,
        environment_config=environment_config,
    )

    assert result.task_context["required_frontmatter_keys"] == ["title", "date", "template"]
    assert {"type": "preserve_required_frontmatter_keys", "keys": ["title", "date", "template"]} in result.hard_requirements


def test_edit_derivation_can_select_semantic_config_files_without_anchored_names():
    from SynthChat.task_derivation import derive_task_spec

    scenario = {
        "task_family": {
            "kind": "edit_file",
            "edit_mode": "text_replace",
            "must_read_before_edit": True,
            "candidate_selector": {
                "target": {
                    "path_prefix": "Ops/",
                    "must_contain_text": ["base_url:", "retries:"],
                    "preferred_keywords": ["prod", "production"],
                    "must_not_contain_text": ["staging", "dev"],
                },
                "distractors": {
                    "path_prefix": "Ops/",
                    "must_contain_text": ["base_url:"],
                    "preferred_keywords": ["staging", "dev"],
                },
            },
            "new_value": "https://api.prod.example.com",
        }
    }
    environment_config = {
        "fixture": {
            "files": {
                "Ops/live-service.md": "base_url: http://old.internal.service\nretries: 3",
                "Ops/preprod-service.md": "base_url: http://staging.internal.service\nretries: 2",
            }
        }
    }

    result = derive_task_spec(
        scenario_key="edit_dynamic_case",
        scenario=scenario,
        environment_config=environment_config,
    )

    assert result.task_context["target_path"] == "Ops/live-service.md"
    assert result.task_context["distractor_path"] == "Ops/preprod-service.md"
    assert result.task_context["old_value"] == "http://old.internal.service"


def test_user_generation_prompt_includes_plain_text_only_instruction():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["A natural user request.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    generator.generate_single(
        scenario_key="user_instruction_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {"fixture": {"files": {"foo.txt": "bar"}}, "assertions": []},
            "prompts": {
                "user": "Write the user request.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    user_call = client.messages[0]["messages"][-1]["content"]
    assert "Return only a natural user request in plain text." in user_call
    assert "Do not output JSON" in user_call


def test_user_generation_prompt_can_require_vague_human_request_style():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please move that quick idea into the Alpha project folder.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    generator.generate_single(
        scenario_key="user_vague_style_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {"fixture": {"files": {"Inbox/Quick Idea.md": "idea"}}, "assertions": []},
            "task_family": {
                "kind": "move_file",
                "user_request_style": {
                    "vague_human_request": True,
                    "require_request_form": True,
                    "avoid_exact_source_path": True,
                    "avoid_exact_target_path": True,
                    "allow_exact_paths": False,
                    "reference_mode": "folder_purpose",
                    "examples": [
                        "Please move that note into the right project folder.",
                    ],
                },
            },
            "prompts": {
                "user": "Ask to move the note about {task_source_note_user_reference} into {task_target_folder_user_reference}.",
                "assistant": "Reply with plain text.",
            },
            "task_context": {
                "source_note_user_reference": "quick idea",
                "target_folder_user_reference": "the Alpha project folder",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    user_call = client.messages[0]["messages"][-1]["content"]
    assert "Write like a normal human user" in user_call
    assert "Phrase the text as a request or question" in user_call
    assert "Do not mention exact file or folder paths" in user_call
    assert "Avoid exact filesystem paths" in user_call
    assert "natural title, topic, or fuzzy description" in user_call
    assert "human folder description" in user_call
    assert "Use the following only as style examples" in user_call
    assert "Please move that note into the right project folder." in user_call


def test_user_generation_stage_gate_can_flag_exact_path_leaks():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please update Ops/production-config.md for me.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="user_gate_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {"fixture": {"files": {"Ops/production-config.md": "base_url: old"}}, "assertions": []},
            "task_context": {"target_path": "Ops/production-config.md"},
            "user_generation": {
                "gates": [
                    {
                        "type": "no_exact_paths_from_context",
                        "field": "text",
                        "sources": ["task_context"],
                    }
                ]
            },
            "prompts": {
                "user": "Write the user request.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    assert "user_generation" in result.stage_failures
    review = result.example["metadata"]["stage_reviews"]["user_generation"]
    assert review["passed"] is False
    assert review["gates"][0]["gate_type"] == "no_exact_paths_from_context"
    assert review["gates"][0]["metadata"]["leaked_paths"] == ["Ops/production-config.md"]


def test_user_generation_stage_gate_ignores_urls_and_connection_strings():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please update the config to use https://api.prod.example.com.", json.dumps({"content": "Done."})],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="user_gate_ignore_url_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {"fixture": {"files": {"Ops/production-config.md": "base_url: old"}}, "assertions": []},
            "task_context": {
                "target_path": "Ops/production-config.md",
                "new_value": "https://api.prod.example.com",
                "old_value": "http://old.api.internal",
            },
            "user_generation": {
                "gates": [
                    {
                        "type": "no_exact_paths_from_context",
                        "field": "text",
                        "sources": ["task_context"],
                    }
                ]
            },
            "prompts": {
                "user": "Write the user request.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    review = result.example["metadata"]["stage_reviews"]["user_generation"]
    assert review["passed"] is True
    assert review["gates"][0]["metadata"]["leaked_paths"] == []


def test_apply_stage_review_result_clears_transient_failure_when_later_review_passes():
    from SynthChat.targets import _apply_stage_review_result

    stage_failures = ["user_generation"]
    stage_reviews = {}

    _apply_stage_review_result(
        stage_failures,
        stage_reviews,
        "user_generation",
        {"passed": True, "enforce": True},
    )

    assert "user_generation" not in stage_failures
    assert stage_reviews["user_generation"]["passed"] is True


def test_archive_empty_task_derivation_uses_concrete_target_details():
    from SynthChat.task_derivation import derive_task_spec

    scenario = {
        "task_family": {
            "kind": "archive_empty_folder",
            "target_contains": "ArchiveEmpty",
            "protected_prefix": "Projects/",
        }
    }
    environment_config = {
        "fixture": {
            "directories": [
                "Projects/ArchiveEmpty",
                "Projects/Alpha",
                "Projects/Beta",
                "Archive/OldProjects",
            ],
            "files": [
                {"path": "Archive/OldProjects/legacy.md", "content": "legacy"},
            ],
        }
    }

    result = derive_task_spec(
        scenario_key="archive_empty_case",
        scenario=scenario,
        environment_config=environment_config,
    )

    assert result.task_context["target_empty_folder"] == "Projects/ArchiveEmpty"
    assert result.task_context["target_empty_folder_user_reference"] == "the ArchiveEmpty project folder"
    assert result.hard_requirements[0]["type"] == "verify_empty_before_archive"
    assert result.hard_requirements[0]["path"] == "Projects/ArchiveEmpty"


def test_environment_generation_stage_review_runs_when_configured():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Write a plain user request.", json.dumps({"content": "Done."})],
        structured_responses=[
            {
                "environment": {
                    "fixture": {"directories": ["Ops"], "files": [{"path": "Ops/config.md", "content": "base_url: old"}]},
                    "assertions": [],
                },
                "task_context": {},
            },
            {"passed": True, "score": 0.88, "feedback": "Environment looks usable."},
        ],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="environment_review_case",
        scenario={
            "type": "chat",
            "environment_mode": "generated",
            "environment_generation": {
                "schema": "canonical_environment",
                "prompt": "Generate environment.",
                "gates": [{"type": "environment_payload_shape", "field": "value"}],
                "judge": {
                    "enabled": True,
                    "prompt": "Review this environment.\n\n{value_json}",
                },
            },
            "prompts": {
                "user": "Write a plain user request.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    review = result.example["metadata"]["stage_reviews"]["environment_generation"]
    assert review["passed"] is True
    assert review["judge"]["score"] == 0.88


def test_final_stage_judge_runs_when_configured():
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "SynthChat" / "config"
    scenarios_dir = repo_root / "SynthChat" / "scenarios"
    rubrics_dir = repo_root / "SynthChat" / "rubrics"

    client = _FakeLLMClient(
        responses=["Please update the config.", json.dumps({"content": "Done."})],
        structured_responses=[{"passed": True, "score": 0.91, "feedback": "Final example is coherent."}],
    )
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=client,
        engine=None,
        environment_validator=None,
        enable_stage_validation=False,
    )

    result = generator.generate_single(
        scenario_key="final_review_case",
        scenario={
            "type": "behavioral",
            "environment_mode": "provided",
            "environment": {"fixture": {"files": {"Ops/config.md": "base_url: old"}}, "assertions": []},
            "final_judge": {
                "judge": {
                    "enabled": True,
                    "prompt": "Review the final example.\n\n{conversation_trace_json}\n\n{assistant_response_json}",
                }
            },
            "prompts": {
                "user": "Write a plain user request.",
                "assistant": "Reply with plain text.",
            },
        },
        max_iterations=1,
        randomize_params=False,
    )

    review = result.example["metadata"]["stage_reviews"]["final"]
    assert review["passed"] is True
    assert review["judge"]["score"] == 0.91
