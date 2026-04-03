"""SynthChat Labeling - Metadata label construction and classification.

Location: SynthChat/labeling.py
Purpose: Build structured metadata labels (flat and filter) for downstream
         dataset slicing, KTO/GRPO candidate classification, and environment
         issue categorization.
Usage: Called by SynthChatGenerator._build_metadata_labels (generator.py)
       during the final metadata assembly stage of each generation run.
"""

import re
from typing import Any, Dict, List, Optional


def _slugify_label(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return value.strip("_")


def _classify_environment_issue(message: str) -> List[str]:
    text = str(message or "").strip().lower()
    if not text:
        return []

    labels = set()

    if "expected tool(s) not executed" in text:
        labels.add("missing_expected_tool")
    if "no acceptable tool called" in text:
        labels.add("wrong_tool_called")
    if "front matter" in text or "yaml front matter" in text:
        labels.add("frontmatter_missing")
    if "expected path to exist" in text or "expected path to be absent" in text:
        labels.add("path_state_mismatch")
    if "does not contain expected text" in text or "contains forbidden text" in text:
        labels.add("content_mismatch")
    if "failed reading" in text:
        labels.add("read_failure")
    if "is a directory" in text or "file exists" in text:
        labels.add("path_type_error")
    if "strict schema" in text or "missing required args" in text:
        labels.add("schema_error")
    if "searchmanager_searchcontent" in text or "searchmanager_searchdirectory" in text:
        labels.add("retrieval_missing")
    if "clarification" in text:
        labels.add("clarification_expected")
    if "tool '" in text and "failed:" in text:
        labels.add("tool_runtime_error")

    return sorted(labels)


def _derive_kto_candidate_label(
    has_environment: bool,
    environment_passed: Optional[bool],
    stage_failures: List[str],
    issue_labels: set[str],
) -> Optional[bool]:
    if not has_environment:
        return None
    if environment_passed and not stage_failures:
        return True
    if environment_passed is False:
        noisy_labels = {"schema_error"}
        if issue_labels and issue_labels.issubset(noisy_labels):
            return None
        return False
    return None


def build_metadata_labels(
    scenario_key: str,
    scenario: Dict[str, Any],
    environment_mode: str,
    stage_failures: List[str],
    environment_trace: Optional[Dict[str, Any]],
    generated_environment: Dict[str, Any],
) -> Dict[str, Any]:
    """Build structured labels for downstream filtering and KTO/GRPO slicing."""
    tags = [str(tag).strip() for tag in scenario.get("tags", []) if str(tag).strip()]
    triggers = [str(trigger).strip() for trigger in scenario.get("triggers", []) if str(trigger).strip()]
    flat_labels = {
        f"scenario:{scenario_key}",
        f"type:{scenario.get('type', 'unknown')}",
        f"environment_mode:{environment_mode}",
    }

    tool_name = str(scenario.get("tool") or "").strip()
    if tool_name:
        flat_labels.add(f"tool:{tool_name}")

    for tag in tags:
        flat_labels.add(f"tag:{tag}")

    for trigger in triggers:
        slug = _slugify_label(trigger)
        if slug:
            flat_labels.add(f"trigger:{slug}")

    stage_failure_labels = []
    for stage in sorted({str(stage).strip() for stage in stage_failures if str(stage).strip()}):
        stage_failure_labels.append(stage)
        flat_labels.add(f"stage_failure:{stage}")

    issues = []
    issue_labels = set()
    executed_tools = []
    executed_tool_names = []
    tool_error_names = []
    environment_passed = None
    if isinstance(environment_trace, dict):
        environment_passed = bool(environment_trace.get("passed"))
        flat_labels.add("environment:present")
        flat_labels.add(f"environment_passed:{str(environment_passed).lower()}")

        for issue in environment_trace.get("issues", []) or []:
            if not isinstance(issue, dict):
                continue
            issue_message = str(issue.get("message", "")).strip()
            issue_level = str(issue.get("level", "")).strip().lower() or "unknown"
            if issue_message:
                issues.append({"level": issue_level, "message": issue_message})
            for label in _classify_environment_issue(issue_message):
                issue_labels.add(label)
                flat_labels.add(f"issue:{label}")

        for tool in environment_trace.get("executed_tools", []) or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            executed_tools.append(
                {
                    "name": name,
                    "status": str(tool.get("status", "ok")).strip() or "ok",
                }
            )
            executed_tool_names.append(name)
            flat_labels.add(f"executed_tool:{name}")
            status = str(tool.get("status", "ok")).strip().lower() or "ok"
            if status != "ok":
                tool_error_names.append(name)
                issue_labels.add("tool_runtime_error")
                flat_labels.add("issue:tool_runtime_error")

    has_environment = environment_trace is not None
    if generated_environment:
        flat_labels.add("environment:generated_payload")

    if has_environment and environment_passed is True and not stage_failure_labels:
        flat_labels.add("kto_candidate:positive")
    elif has_environment and environment_passed is False:
        flat_labels.add("kto_candidate:negative")

    if "environment" in stage_failure_labels:
        flat_labels.add("failure_type:environment")
    if any(stage in stage_failure_labels for stage in ("response", "thinking")):
        flat_labels.add("failure_type:behavior")
    if any(stage in stage_failure_labels for stage in ("system_prompt", "user", "system_generation", "user_generation", "assistant_generation", "environment_generation", "final")):
        flat_labels.add("failure_type:generation")

    if any(label in issue_labels for label in {"missing_expected_tool", "retrieval_missing"}):
        flat_labels.add("behavior:retrieval_failure")
    if "frontmatter_missing" in issue_labels:
        flat_labels.add("behavior:structure_failure")
    if any(label in issue_labels for label in {"wrong_tool_called", "tool_runtime_error"}):
        flat_labels.add("behavior:tool_execution_failure")
    if "clarification_expected" in issue_labels:
        flat_labels.add("behavior:clarification_failure")

    return {
        "flat": sorted(flat_labels),
        "filter": {
            "scenario_key": scenario_key,
            "scenario_type": str(scenario.get("type", "unknown")),
            "tool_name": tool_name or None,
            "environment_mode": environment_mode,
            "has_environment": has_environment,
            "environment_passed": environment_passed,
            "generated_environment": bool(generated_environment),
            "stage_failures": stage_failure_labels,
            "issue_labels": sorted(issue_labels),
            "scenario_tags": tags,
            "scenario_triggers": triggers,
            "executed_tools": executed_tool_names,
            "executed_tool_details": executed_tools,
            "tool_errors": tool_error_names,
            "issues": issues,
            "kto_candidate_label": _derive_kto_candidate_label(
                has_environment=has_environment,
                environment_passed=environment_passed,
                stage_failures=stage_failure_labels,
                issue_labels=issue_labels,
            ),
        },
    }
