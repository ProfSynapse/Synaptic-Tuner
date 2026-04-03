"""Tests for SynthChat.labeling — metadata label construction and classification."""
from __future__ import annotations

import pytest
from SynthChat.labeling import (
    _classify_environment_issue,
    _derive_kto_candidate_label,
    _slugify_label,
    build_metadata_labels,
)
from SynthChat.config.format_resolver import get_default_label_mappings


def _default_classifiers():
    return get_default_label_mappings()["issue_classifiers"]


def _default_label_mappings():
    return get_default_label_mappings()


# ---- _slugify_label ----

class TestSlugifyLabel:
    def test_basic_slugification(self):
        assert _slugify_label("Hello World!") == "hello_world"

    def test_special_characters(self):
        assert _slugify_label("foo@bar.baz") == "foo_bar_baz"

    def test_leading_trailing_underscores_stripped(self):
        assert _slugify_label("  ---test---  ") == "test"

    def test_empty_string(self):
        assert _slugify_label("") == ""

    def test_none_input(self):
        assert _slugify_label(None) == ""

    def test_numbers_preserved(self):
        assert _slugify_label("test123") == "test123"


# ---- _classify_environment_issue ----

class TestClassifyEnvironmentIssue:
    def test_empty_message(self):
        assert _classify_environment_issue("", _default_classifiers()) == []
        assert _classify_environment_issue(None, _default_classifiers()) == []

    def test_missing_expected_tool(self):
        labels = _classify_environment_issue("Expected tool(s) not executed in session", _default_classifiers())
        assert "missing_expected_tool" in labels

    def test_wrong_tool_called(self):
        labels = _classify_environment_issue("No acceptable tool called", _default_classifiers())
        assert "wrong_tool_called" in labels

    def test_frontmatter_missing(self):
        labels = _classify_environment_issue("YAML front matter not found", _default_classifiers())
        assert "frontmatter_missing" in labels

    def test_path_state_mismatch(self):
        labels = _classify_environment_issue("Expected path to exist: notes/test.md", _default_classifiers())
        assert "path_state_mismatch" in labels

    def test_content_mismatch(self):
        labels = _classify_environment_issue("does not contain expected text", _default_classifiers())
        assert "content_mismatch" in labels

    def test_multiple_labels(self):
        msg = "Expected tool(s) not executed; no acceptable tool called"
        labels = _classify_environment_issue(msg, _default_classifiers())
        assert "missing_expected_tool" in labels
        assert "wrong_tool_called" in labels

    def test_labels_sorted(self):
        msg = "Expected path to exist: x; does not contain expected text"
        labels = _classify_environment_issue(msg, _default_classifiers())
        assert labels == sorted(labels)

    def test_tool_runtime_error(self):
        labels = _classify_environment_issue("Tool 'fileManager_read' failed: some error", _default_classifiers())
        assert "tool_runtime_error" in labels

    def test_clarification_expected(self):
        labels = _classify_environment_issue("clarification needed but not given", _default_classifiers())
        assert "clarification_expected" in labels


# ---- _derive_kto_candidate_label ----

class TestDeriveKtoCandidateLabel:
    def test_no_environment(self):
        assert _derive_kto_candidate_label(False, None, [], set()) is None

    def test_positive_candidate(self):
        assert _derive_kto_candidate_label(True, True, [], set()) is True

    def test_negative_candidate(self):
        assert _derive_kto_candidate_label(True, False, [], {"content_mismatch"}) is False

    def test_schema_error_only_returns_none(self):
        """Pure schema errors are too noisy to be negative KTO candidates."""
        assert _derive_kto_candidate_label(True, False, [], {"schema_error"}) is None

    def test_stage_failures_prevent_positive(self):
        assert _derive_kto_candidate_label(True, True, ["response"], set()) is None

    def test_none_environment_passed(self):
        assert _derive_kto_candidate_label(True, None, [], set()) is None


# ---- build_metadata_labels ----

class TestBuildMetadataLabels:
    def _base_args(self, **overrides):
        defaults = {
            "scenario_key": "test_scenario",
            "scenario": {"type": "tool", "tool": "fileManager_read", "tags": ["test"]},
            "environment_mode": "generated",
            "stage_failures": [],
            "environment_trace": None,
            "generated_environment": {},
            "label_mappings": _default_label_mappings(),
        }
        defaults.update(overrides)
        return defaults

    def test_basic_flat_labels(self):
        result = build_metadata_labels(**self._base_args())
        flat = result["flat"]
        assert "scenario:test_scenario" in flat
        assert "type:tool" in flat
        assert "environment_mode:generated" in flat
        assert "tool:fileManager_read" in flat
        assert "tag:test" in flat

    def test_filter_structure(self):
        result = build_metadata_labels(**self._base_args())
        f = result["filter"]
        assert f["scenario_key"] == "test_scenario"
        assert f["scenario_type"] == "tool"
        assert f["tool_name"] == "fileManager_read"
        assert f["environment_mode"] == "generated"
        assert f["has_environment"] is False

    def test_environment_trace_labels(self):
        trace = {
            "passed": True,
            "issues": [],
            "executed_tools": [{"name": "fileManager_read", "status": "ok"}],
        }
        result = build_metadata_labels(**self._base_args(environment_trace=trace))
        flat = result["flat"]
        assert "environment:present" in flat
        assert "environment_passed:true" in flat
        assert "executed_tool:fileManager_read" in flat
        assert "kto_candidate:positive" in flat

    def test_failed_environment_labels(self):
        trace = {
            "passed": False,
            "issues": [{"message": "Expected tool(s) not executed", "level": "error"}],
            "executed_tools": [],
        }
        result = build_metadata_labels(**self._base_args(environment_trace=trace))
        flat = result["flat"]
        assert "environment_passed:false" in flat
        assert "kto_candidate:negative" in flat
        assert "issue:missing_expected_tool" in flat
        assert "behavior:retrieval_failure" in flat

    def test_stage_failure_labels(self):
        result = build_metadata_labels(**self._base_args(
            stage_failures=["environment", "response"]
        ))
        flat = result["flat"]
        assert "stage_failure:environment" in flat
        assert "stage_failure:response" in flat
        assert "failure_type:environment" in flat
        assert "failure_type:behavior" in flat

    def test_generated_environment_label(self):
        result = build_metadata_labels(**self._base_args(
            generated_environment={"environment": {"fixture": {}}}
        ))
        assert "environment:generated_payload" in result["flat"]

    def test_flat_labels_sorted(self):
        result = build_metadata_labels(**self._base_args())
        assert result["flat"] == sorted(result["flat"])

    def test_tool_error_labels(self):
        trace = {
            "passed": False,
            "issues": [],
            "executed_tools": [{"name": "fileManager_read", "status": "error"}],
        }
        result = build_metadata_labels(**self._base_args(environment_trace=trace))
        f = result["filter"]
        assert "fileManager_read" in f["tool_errors"]
        assert "tool_runtime_error" in f["issue_labels"]

    def test_triggers_slugified(self):
        result = build_metadata_labels(**self._base_args(
            scenario={"type": "tool", "triggers": ["On Create Note"]}
        ))
        assert "trigger:on_create_note" in result["flat"]
