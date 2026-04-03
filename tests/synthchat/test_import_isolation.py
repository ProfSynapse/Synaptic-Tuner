"""Test that each extracted module is independently importable.

Verifies that the SynthChat decomposition created well-isolated modules
without circular imports or missing dependencies.
"""
from __future__ import annotations

import importlib


def _import(module_name: str):
    """Import a module by dotted path and return it."""
    return importlib.import_module(module_name)


# ---- template_utils ----

def test_import_template_utils():
    mod = _import("SynthChat.template_utils")
    assert hasattr(mod, "_deep_merge_dicts")
    assert hasattr(mod, "_render_template_object")
    assert hasattr(mod, "_make_json_safe")
    assert hasattr(mod, "_task_context_template_vars")
    assert hasattr(mod, "_user_generation_style_instructions")
    assert hasattr(mod, "_clean_path")


# ---- targets ----

def test_import_targets():
    mod = _import("SynthChat.targets")
    assert hasattr(mod, "_normalize_target_spec")
    assert hasattr(mod, "_extract_shared_seed_spec")
    assert hasattr(mod, "_apply_stage_review_result")


# ---- parsing ----

def test_import_parsing():
    mod = _import("SynthChat.parsing")
    assert hasattr(mod, "stringify_assistant_message")
    assert hasattr(mod, "parse_assistant_response")
    assert hasattr(mod, "parse_json_object")
    assert hasattr(mod, "normalize_generated_environment")
    assert hasattr(mod, "_normalize_generated_assertion")


# ---- labeling ----

def test_import_labeling():
    mod = _import("SynthChat.labeling")
    assert hasattr(mod, "build_metadata_labels")
    assert hasattr(mod, "_slugify_label")
    assert hasattr(mod, "_classify_environment_issue")
    assert hasattr(mod, "_derive_kto_candidate_label")


# ---- review ----

def test_import_review():
    mod = _import("SynthChat.review")
    assert hasattr(mod, "run_stage_review")
    assert hasattr(mod, "build_environment_generation_review_payload")
    assert hasattr(mod, "run_configured_stage_judge")
    assert hasattr(mod, "build_stage_judge_template_vars")


# ---- llm subpackage ----

def test_import_llm_client_pool():
    mod = _import("SynthChat.llm.client_pool")
    assert hasattr(mod, "LLMClientPool")


def test_import_llm_caller():
    mod = _import("SynthChat.llm.caller")
    assert hasattr(mod, "call_llm")
    assert hasattr(mod, "call_llm_structured")


def test_import_llm_package():
    """Package imports without error; submodules reachable via dotted path."""
    mod = _import("SynthChat.llm")
    assert mod is not None
    # Submodules are importable directly (not re-exported from __init__)
    from SynthChat.llm.client_pool import LLMClientPool
    from SynthChat.llm.caller import call_llm
    assert LLMClientPool is not None
    assert call_llm is not None


# ---- workspace subpackage ----

def test_import_workspace_renderer():
    mod = _import("SynthChat.workspace.renderer")
    assert hasattr(mod, "render_workspace_prompt")


def test_import_workspace_sections():
    mod = _import("SynthChat.workspace.sections")
    assert hasattr(mod, "_render_available_workspaces")
    assert hasattr(mod, "_tool_wrapper_name")
    assert hasattr(mod, "_build_wrapped_section")
    assert hasattr(mod, "_render_note_contents")


def test_import_workspace_fixture_helpers():
    mod = _import("SynthChat.workspace.fixture_helpers")
    assert hasattr(mod, "_merged_fixture_from_config")
    assert hasattr(mod, "_workspace_structure_from_fixture")
    assert hasattr(mod, "_note_entries_from_fixture")


def test_import_workspace_package():
    mod = _import("SynthChat.workspace")
    assert hasattr(mod, "render_workspace_prompt")


# ---- schemas subpackage ----

def test_import_schemas_environment():
    mod = _import("SynthChat.schemas.environment_schema")
    assert hasattr(mod, "_build_canonical_environment_schema")
    assert hasattr(mod, "_build_canonical_environment_generation_prompt")


def test_import_schemas_tool_response():
    mod = _import("SynthChat.schemas.tool_response_schema")
    assert hasattr(mod, "build_tool_response_schema")
    assert hasattr(mod, "build_tool_generation_prompt")
    assert hasattr(mod, "_resolve_allowed_tool_names")
    assert hasattr(mod, "_resolve_context_defaults")


def test_import_schemas_package():
    """Package imports without error; submodules reachable via dotted path."""
    mod = _import("SynthChat.schemas")
    assert mod is not None
    from SynthChat.schemas.environment_schema import _build_canonical_environment_schema
    from SynthChat.schemas.tool_response_schema import build_tool_response_schema
    assert _build_canonical_environment_schema is not None
    assert build_tool_response_schema is not None


# ---- agentic subpackage ----

def test_import_agentic_episode():
    mod = _import("SynthChat.agentic.episode")
    assert hasattr(mod, "generate_agentic_episode")
    assert hasattr(mod, "build_turn_judge")
    assert hasattr(mod, "build_turn_judge_template_vars")
    assert hasattr(mod, "synthchat_loop_response")
    assert hasattr(mod, "validate_agentic_synthchat_response")


def test_import_agentic_package():
    """Package imports without error; submodules reachable via dotted path."""
    mod = _import("SynthChat.agentic")
    assert mod is not None
    from SynthChat.agentic.episode import generate_agentic_episode
    assert generate_agentic_episode is not None


# ---- parallel subpackage ----

def test_import_parallel_workers():
    mod = _import("SynthChat.parallel.workers")
    assert hasattr(mod, "run_parallel_generation")
    assert hasattr(mod, "generate_single_example")
    assert hasattr(mod, "create_worker_generator")
    assert hasattr(mod, "serialize_environment_options")


def test_import_parallel_package():
    """Package imports without error; submodules reachable via dotted path."""
    mod = _import("SynthChat.parallel")
    assert mod is not None
    from SynthChat.parallel.workers import run_parallel_generation
    assert run_parallel_generation is not None


# ---- modes subpackage ----

def test_import_modes_generate():
    mod = _import("SynthChat.modes.generate")
    assert hasattr(mod, "generate_mode")


def test_import_modes_improve():
    mod = _import("SynthChat.modes.improve")
    assert hasattr(mod, "improve_mode")


def test_import_modes_validate():
    mod = _import("SynthChat.modes.validate")
    assert hasattr(mod, "validate_mode")


def test_import_modes_package():
    """Package imports without error; submodules reachable via dotted path."""
    mod = _import("SynthChat.modes")
    assert mod is not None
    from SynthChat.modes.generate import generate_mode
    from SynthChat.modes.improve import improve_mode
    from SynthChat.modes.validate import validate_mode
    assert generate_mode is not None


# ---- result_writer ----

def test_import_result_writer():
    mod = _import("SynthChat.result_writer")
    assert hasattr(mod, "StreamingResultWriter")
    assert hasattr(mod, "generate_output_path")
    assert hasattr(mod, "print_summary")
