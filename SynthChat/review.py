"""SynthChat Review - Stage review orchestration and judge template building.

Location: SynthChat/review.py
Purpose: Run stage-level quality reviews combining gate checks and LLM judges,
         and build the template variables and payloads they need.
Usage: Called by SynthChatGenerator during each stage of generation to evaluate
       quality before proceeding to the next stage.
"""

import json
from typing import Any, Dict, List, Optional

from .stage_gates import run_stage_gates
from .template_utils import _make_json_safe
from .workspace.fixture_helpers import _merged_fixture_from_config
from shared.stage_judges import ConfigurableStageJudge


def run_stage_review(
    *,
    stage_name: str,
    stage_config: Optional[Dict[str, Any]],
    scenario_key: str,
    scenario: Dict[str, Any],
    task_context: Dict[str, Any],
    payload: Dict[str, Any],
    llm_client: Any,
    get_stage_llm_clients: Any,
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    """Run gate checks and optional judge for a generation stage."""
    if not isinstance(stage_config, dict):
        return None

    gates_cfg = stage_config.get("gates") if isinstance(stage_config.get("gates"), list) else []
    judge_cfg = stage_config.get("judge") if isinstance(stage_config.get("judge"), dict) else {}
    if not gates_cfg and not judge_cfg:
        return None

    review: Dict[str, Any] = {
        "stage": stage_name,
        "enforce": bool(stage_config.get("enforce", True)),
        "passed": True,
        "gates": [],
        "judge": None,
    }

    gate_results = run_stage_gates(gates_cfg, payload)
    if gate_results:
        review["gates"] = [result.to_dict() for result in gate_results]
        if any(not result.passed for result in gate_results):
            review["passed"] = False
        if logger:
            failed = sum(1 for result in gate_results if not result.passed)
            logger.info(f"[{scenario_key}] {stage_name} gates done (failed={failed}/{len(gate_results)})")

    judge_result = run_configured_stage_judge(
        stage_name=stage_name,
        judge_config=judge_cfg,
        scenario_key=scenario_key,
        scenario=scenario,
        task_context=task_context,
        payload=payload,
        llm_client=llm_client,
        get_stage_llm_clients=get_stage_llm_clients,
    )
    if judge_result is not None:
        review["judge"] = judge_result
        if not judge_result.get("passed", True):
            review["passed"] = False
        min_quality_score = judge_cfg.get("min_quality_score")
        if min_quality_score is not None:
            score = judge_result.get("score")
            if score is None or float(score) < float(min_quality_score):
                review["passed"] = False
                review["judge"]["below_min_quality_score"] = True
        if logger:
            logger.info(
                f"[{scenario_key}] {stage_name} judge done "
                f"(passed={judge_result.get('passed')} score={judge_result.get('score')})"
            )

    return review


def build_environment_generation_review_payload(
    *,
    generated_environment: Dict[str, Any],
    seed_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a review payload from a generated environment spec."""
    environment = generated_environment.get("environment") if isinstance(generated_environment, dict) else {}
    fixture_snapshot = {}
    if isinstance(environment, dict):
        try:
            fixture = _merged_fixture_from_config(environment)
            fixture_snapshot = {
                "directories": list(fixture.directories),
                "files": [
                    {"path": path, "content": content}
                    for path, content in fixture.files.items()
                ],
            }
        except Exception as exc:
            fixture_snapshot = {"error": str(exc)}
    payload = {
        "value": {
            "environment": {
                "fixture": fixture_snapshot,
                "assertions": list(environment.get("assertions") or []) if isinstance(environment, dict) else [],
            },
            "system_context": generated_environment.get("system_context") if isinstance(generated_environment, dict) else {},
            "task_context": generated_environment.get("task_context") if isinstance(generated_environment, dict) else {},
        },
        "generated_environment": generated_environment,
    }
    if seed_id:
        payload["seed_id"] = seed_id
    return payload


def run_configured_stage_judge(
    *,
    stage_name: str,
    judge_config: Optional[Dict[str, Any]],
    scenario_key: str,
    scenario: Dict[str, Any],
    task_context: Dict[str, Any],
    payload: Dict[str, Any],
    llm_client: Any,
    get_stage_llm_clients: Any,
) -> Optional[Dict[str, Any]]:
    """Run an LLM-as-judge evaluation for a generation stage."""
    if not isinstance(judge_config, dict) or not bool(judge_config.get("enabled")):
        return None
    prompt_template = str(judge_config.get("prompt") or "").strip()
    if not prompt_template:
        return None

    judge = ConfigurableStageJudge(
        llm_client=llm_client,
        llm_clients=get_stage_llm_clients(judge_config),
        prompt_template=prompt_template,
        system_prompt=judge_config.get("system"),
        output_schema=judge_config.get("output_schema"),
        temperature=float(judge_config.get("temperature", 0.2) or 0.2),
        max_tokens=judge_config.get("max_tokens"),
        max_retries=int(judge_config.get("max_retries", 3) or 3),
    )
    template_vars = build_stage_judge_template_vars(
        stage_name=stage_name,
        scenario_key=scenario_key,
        scenario=scenario,
        task_context=task_context,
        payload=payload,
    )
    return judge.judge(template_vars).to_dict()


def build_stage_judge_template_vars(
    *,
    stage_name: str,
    scenario_key: str,
    scenario: Dict[str, Any],
    task_context: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """Build the template variable dict for a stage judge prompt."""
    safe_payload = _make_json_safe(payload or {})
    safe_scenario = _make_json_safe(scenario or {})
    safe_task_context = _make_json_safe(task_context or {})
    return {
        "stage_name": stage_name,
        "scenario_key": scenario_key,
        "scenario_json": json.dumps(safe_scenario, ensure_ascii=False, indent=2),
        "task_context_json": json.dumps(safe_task_context, ensure_ascii=False, indent=2),
        "payload_json": json.dumps(safe_payload, ensure_ascii=False, indent=2),
        "value_json": json.dumps(safe_payload.get("value"), ensure_ascii=False, indent=2),
        "text": str(safe_payload.get("text") or ""),
        "system_text": str(safe_payload.get("system_text") or ""),
        "user_text": str(safe_payload.get("user_text") or ""),
        "assistant_response_json": (
            safe_payload.get("assistant_response_json")
            if isinstance(safe_payload.get("assistant_response_json"), str)
            else json.dumps(safe_payload.get("assistant_response"), ensure_ascii=False, indent=2)
        ),
        "environment_result_json": json.dumps(safe_payload.get("environment_result") or {}, ensure_ascii=False, indent=2),
        "conversation_trace_json": json.dumps(safe_payload.get("conversation_trace") or [], ensure_ascii=False, indent=2),
        "hard_requirements_json": json.dumps(safe_payload.get("hard_requirements") or [], ensure_ascii=False, indent=2),
        "quality_rubric_json": json.dumps(safe_payload.get("quality_rubric") or [], ensure_ascii=False, indent=2),
    }
