"""Generic structured judges for non-loop generation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Mapping, Optional


def default_stage_judge_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "score": {"type": ["number", "null"]},
            "feedback": {"type": "string"},
        },
        "required": ["passed", "score", "feedback"],
        "additionalProperties": True,
    }


@dataclass
class StageJudgeResult:
    passed: bool
    score: Optional[float] = None
    feedback: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "feedback": self.feedback,
            "raw_output": self.raw_output,
            "error": self.error,
            "latency_s": self.latency_s,
            "metadata": dict(self.metadata),
        }


class ConfigurableStageJudge:
    """Config-driven judge for environment/system/user/final stages."""

    def __init__(
        self,
        *,
        llm_client: Any,
        llm_clients: Optional[list[Any]] = None,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.llm_clients = list(llm_clients or [llm_client])
        self.prompt_template = str(prompt_template or "")
        self.system_prompt = system_prompt
        self.output_schema = output_schema or default_stage_judge_schema()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(1, int(max_retries or 1))

    def judge(self, template_vars: Mapping[str, Any]) -> StageJudgeResult:
        prompt = render_stage_judge_prompt(self.prompt_template, template_vars)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        started_at = time.perf_counter()
        last_error: Optional[Exception] = None
        for client_index, client in enumerate(self.llm_clients):
            for attempt in range(1, self.max_retries + 1):
                try:
                    kwargs = {
                        "messages": messages,
                        "schema": self.output_schema,
                        "temperature": self.temperature,
                    }
                    if self.max_tokens is not None:
                        kwargs["max_tokens"] = self.max_tokens
                    raw_output = client.structured_output(**kwargs)
                    latency = round(time.perf_counter() - started_at, 3)
                    score = raw_output.get("score")
                    try:
                        score_value = float(score) if score is not None else None
                    except (TypeError, ValueError):
                        score_value = None
                    return StageJudgeResult(
                        passed=bool(raw_output.get("passed", True)),
                        score=score_value,
                        feedback=_clean_text(raw_output.get("feedback")),
                        raw_output=raw_output,
                        latency_s=latency,
                        metadata={
                            "client_index": client_index,
                            "attempt": attempt,
                            "provider": getattr(client, "provider_name", None),
                            "model": getattr(client, "model_name", None),
                        },
                    )
                except Exception as exc:  # pragma: no cover - provider-specific failures
                    last_error = exc

        latency = round(time.perf_counter() - started_at, 3)
        return StageJudgeResult(
            passed=False,
            feedback=f"Judge call failed: {last_error}",
            error=str(last_error) if last_error else None,
            latency_s=latency,
        )


def render_stage_judge_prompt(template: str, template_vars: Mapping[str, Any]) -> str:
    prompt = str(template or "")
    for key, value in template_vars.items():
        prompt = prompt.replace(f"{{{key}}}", _stringify_template_value(value))
    return prompt


def _stringify_template_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
