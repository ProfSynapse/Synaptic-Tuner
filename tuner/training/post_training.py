"""Bounded, opt-in post-training evaluation configuration."""

from __future__ import annotations

import json
import math
import re


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9.+-]*)\Z")
_MAX_CONFIG_BYTES = 1024 * 1024


def _fields(value: object, required: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _integer(value: object, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} is outside its bound")
    return value


def _number(value: object, label: str, low: float, high: float) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not low <= result <= high:
        raise ValueError(f"{label} is outside its bound")
    return result


def validate_post_training_config(raw: object) -> dict | None:
    """Return a JSON-only normalized specification, or None when absent.

    Required controls are explicit. Optional decode controls remain absent when
    unspecified, preserving legacy workload bytes and server-default behavior.
    """
    if raw is None:
        return None
    try:
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_CONFIG_BYTES:
            raise ValueError("post-training config exceeds its byte bound")
        config = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        raise ValueError("post-training config is not bounded JSON") from None
    _fields(config, {"mode", "evaluation"}, "post-training config")
    if config["mode"] != "same_job":
        raise ValueError("post-training mode is unsupported")
    evaluation = _fields(config["evaluation"], {
        "scenarios", "min_pass_rate", "max_cases", "startup_timeout_seconds",
        "timeout_seconds", "served_model_name", "generation", "vllm",
    }, "post-training evaluation")
    count = _integer(evaluation["max_cases"], "maximum cases", 1, 32)
    scenarios = evaluation["scenarios"]
    if type(scenarios) is not list or not 1 <= len(scenarios) <= count:
        raise ValueError("scenarios exceed the case bound")
    seen: set[str] = set()
    for scenario in scenarios:
        if type(scenario) is not dict or not {"id", "question", "correct"}.issubset(scenario) or set(scenario) - {"id", "question", "correct", "system", "messages", "tags", "scoring", "response_schema", "response_schema_name"}:
            raise ValueError("evaluation scenario has missing or unsupported fields")
        identifier = scenario["id"]
        if type(identifier) is not str or _NAME.fullmatch(identifier) is None or identifier in seen:
            raise ValueError("evaluation case id is invalid or repeated")
        seen.add(identifier)
        # The aggregate JSON byte bound already covers every prompt field.
        # Model context admission belongs to the configured serving runtime.
        if type(scenario["question"]) is not str or not scenario["question"].strip():
            raise ValueError("evaluation question is invalid")
        if "system" in scenario and type(scenario["system"]) is not str:
            raise ValueError("evaluation system prompt is invalid")
        if "messages" in scenario:
            messages = scenario["messages"]
            if type(messages) is not list or not 1 <= len(messages) <= 16 or any(
                type(message) is not dict or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or type(message["content"]) is not str
                for message in messages
            ):
                raise ValueError("evaluation messages are invalid")
        correct = scenario["correct"]
        if type(correct) is not dict or not correct or not any(
            type(correct.get(key)) is list and correct[key] for key in ("any", "all", "assertions")
        ):
            raise ValueError("evaluation correctness assertions are required")
        if "tags" in scenario and (type(scenario["tags"]) is not list or any(type(tag) is not str for tag in scenario["tags"])):
            raise ValueError("evaluation tags are invalid")
    evaluation["min_pass_rate"] = _number(evaluation["min_pass_rate"], "minimum pass rate", 0.0, 1.0)
    evaluation["startup_timeout_seconds"] = _number(evaluation["startup_timeout_seconds"], "startup timeout", 1.0, 1800.0)
    evaluation["timeout_seconds"] = _number(evaluation["timeout_seconds"], "evaluation timeout", 1.0, 3600.0)
    if evaluation["timeout_seconds"] < evaluation["startup_timeout_seconds"]:
        raise ValueError("evaluation timeout cannot be shorter than startup")
    if type(evaluation["served_model_name"]) is not str or _NAME.fullmatch(evaluation["served_model_name"]) is None or evaluation["served_model_name"] == "synaptic-base":
        raise ValueError("served model name is invalid")
    generation = evaluation["generation"]
    if (type(generation) is not dict or not {"max_tokens", "temperature", "top_p"}.issubset(generation)
            or set(generation) - {"max_tokens", "temperature", "top_p", "chat_template_kwargs", "presence_penalty", "top_k", "min_p", "repetition_penalty"}):
        raise ValueError("generation has missing or unknown fields")
    if generation["max_tokens"] is not None:
        _integer(generation["max_tokens"], "maximum output tokens", 1, 262144)
    if "chat_template_kwargs" in generation:
        from synaptic_tuner.api.v1.training_input import validate_chat_template_kwargs
        generation["chat_template_kwargs"] = validate_chat_template_kwargs(generation["chat_template_kwargs"])
    generation["temperature"] = _number(generation["temperature"], "temperature", 0.0, 2.0)
    generation["top_p"] = _number(generation["top_p"], "top p", 0.000001, 1.0)
    for name, low, high in (("presence_penalty", -2.0, 2.0), ("min_p", 0.0, 1.0),
                            ("repetition_penalty", 0.0, float("inf"))):
        if name in generation:
            generation[name] = _number(generation[name], name, low, high)
            if name == "repetition_penalty" and generation[name] == 0:
                raise ValueError("repetition penalty must be greater than zero")
    if "top_k" in generation and (type(generation["top_k"]) is not int or generation["top_k"] < -1):
        raise ValueError("top k must be an integer greater than or equal to -1")
    vllm = _fields(evaluation["vllm"], {
        "expected_version", "dtype", "max_model_len", "tensor_parallel_size",
        "max_num_seqs", "max_num_batched_tokens", "language_model_only", "max_lora_rank",
    }, "vLLM")
    if type(vllm["expected_version"]) is not str or _VERSION.fullmatch(vllm["expected_version"]) is None:
        raise ValueError("vLLM version pin is invalid")
    if vllm["dtype"] not in ("auto", "float16", "bfloat16", "float32"):
        raise ValueError("vLLM dtype is invalid")
    _integer(vllm["max_model_len"], "model context length", 128, 262144)
    _integer(vllm["tensor_parallel_size"], "tensor parallel size", 1, 256)
    _integer(vllm["max_num_seqs"], "maximum sequences", 1, 1024)
    _integer(vllm["max_num_batched_tokens"], "maximum batched tokens", 128, 262144)
    _integer(vllm["max_lora_rank"], "maximum LoRA rank", 1, 1024)
    if type(vllm["language_model_only"]) is not bool:
        raise ValueError("language model only must be boolean")
    if generation["max_tokens"] is not None and generation["max_tokens"] >= vllm["max_model_len"]:
        raise ValueError("output token bound must fit model context")
    return config
