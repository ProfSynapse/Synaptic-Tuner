"""SynthChat Labeling - Metadata label construction and classification.

Location: SynthChat/labeling.py
Purpose: Build structured metadata labels (flat and filter) for downstream
         dataset slicing, KTO/GRPO candidate classification, and environment
         issue categorization. Classifiers and rollup rules are read from
         config (label_mappings.yaml) rather than hardcoded.
Usage: Called by SynthChatGenerator._build_metadata_labels (generator.py)
       during the final metadata assembly stage of each generation run.
"""

import re
from typing import Any, Dict, List, Optional


def _slugify_label(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return value.strip("_")


def _classify_environment_issue(
    message: str,
    classifiers: List[Dict[str, str]],
) -> List[str]:
    """Classify an environment issue message into labels.

    Args:
        message: The issue message text.
        classifiers: List of classifier rules from label_mappings config.
            Each rule has 'match', optional 'match_also', and 'label'.
    """
    text = str(message or "").strip().lower()
    if not text:
        return []

    labels = set()
    for rule in classifiers:
        match_str = str(rule.get("match", "")).lower()
        if not match_str:
            continue
        if match_str not in text:
            continue
        match_also = rule.get("match_also")
        if match_also and str(match_also).lower() not in text:
            continue
        label = str(rule.get("label", "")).strip()
        if label:
            labels.add(label)

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


def _apply_failure_type_rollups(
    stage_failure_labels: List[str],
    failure_type_rollups: Dict[str, List[str]],
) -> set:
    """Apply failure_type rollup rules from config."""
    labels = set()
    for rollup_label, trigger_stages in failure_type_rollups.items():
        if any(stage in stage_failure_labels for stage in trigger_stages):
            labels.add(rollup_label)
    return labels


def _apply_behavior_rollups(
    issue_labels: set,
    behavior_rollups: Dict[str, List[str]],
) -> set:
    """Apply behavior rollup rules from config."""
    labels = set()
    for rollup_label, trigger_labels in behavior_rollups.items():
        if any(label in issue_labels for label in trigger_labels):
            labels.add(rollup_label)
    return labels


def build_metadata_labels(
    scenario_key: str,
    scenario: Dict[str, Any],
    environment_mode: str,
    stage_failures: List[str],
    environment_trace: Optional[Dict[str, Any]],
    generated_environment: Dict[str, Any],
    privacy_trace: Optional[Dict[str, Any]],
    label_mappings: Dict[str, Any],
) -> Dict[str, Any]:
    """Build structured labels for downstream filtering and KTO/GRPO slicing.

    Args:
        label_mappings: Config dict with keys issue_classifiers, behavior_rollups,
            failure_type_rollups.
    """

    classifiers = label_mappings.get("issue_classifiers", [])
    behavior_rollups = label_mappings.get("behavior_rollups", {})
    failure_type_rollups = label_mappings.get("failure_type_rollups", {})

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
            for label in _classify_environment_issue(issue_message, classifiers):
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

    privacy_filter: Dict[str, Any] = {}
    if isinstance(privacy_trace, dict):
        flat_labels.add("privacy:present")
        changed = bool(privacy_trace.get("changed"))
        flat_labels.add(f"privacy_changed:{str(changed).lower()}")
        profile_name = _slugify_label(str(privacy_trace.get("profile") or ""))
        if profile_name:
            flat_labels.add(f"privacy_profile:{profile_name}")

        detection = privacy_trace.get("detection")
        if isinstance(detection, dict):
            labels = [str(label).strip() for label in detection.get("labels", []) if str(label).strip()]
            for label in labels:
                flat_labels.add(f"privacy_label:{label}")
            privacy_filter = {
                "present": True,
                "changed": changed,
                "profile": str(privacy_trace.get("profile") or "") or None,
                "labels": sorted(labels),
                "span_count": int(detection.get("span_count", 0) or 0),
            }
        else:
            privacy_filter = {
                "present": True,
                "changed": changed,
                "profile": str(privacy_trace.get("profile") or "") or None,
                "labels": [],
                "span_count": 0,
            }
    else:
        privacy_filter = {
            "present": False,
            "changed": False,
            "profile": None,
            "labels": [],
            "span_count": 0,
        }

    if has_environment and environment_passed is True and not stage_failure_labels:
        flat_labels.add("kto_candidate:positive")
    elif has_environment and environment_passed is False:
        flat_labels.add("kto_candidate:negative")

    # Apply config-driven rollups
    flat_labels.update(_apply_failure_type_rollups(stage_failure_labels, failure_type_rollups))
    flat_labels.update(_apply_behavior_rollups(issue_labels, behavior_rollups))

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
            "privacy": privacy_filter,
        },
    }
