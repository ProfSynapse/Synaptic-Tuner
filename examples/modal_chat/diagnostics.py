"""Closed, non-authorizing diagnostics for the Modal chat consumer."""

from __future__ import annotations

from tuner.execution.foundation_v2.canonical import canonical_bytes

_MAX_EXCEPTIONS = 8
_MAX_FRAMES = 32
_MAX_VISITED_FRAMES = 128
_PHASES = frozenset(
    {
        "INPUTS",
        "LOCAL_INPUTS_CHECKED",
        "QUOTE_TRAINING",
        "PROVISIONING",
        "DEPLOYING",
        "RESOLVING",
        "SUBMITTING",
        "TRAINING_ACCEPTED",
        "TRAINING_PENDING",
        "NATIVE_TRAINING_QUALIFIED",
        "CHATTING",
        "CHAT_CONFIGURATION",
        "CHAT_QUOTE",
        "CHAT_COMPOSITION",
        "CHAT_OPEN",
        "CHAT_CLEANUP",
        "CHAT_SAVED_AND_STOPPED",
    }
)
_EXAMPLE_PATHS = frozenset(
    {
        "examples/modal_chat/authority.py",
        "examples/modal_chat/artifacts.py",
        "examples/modal_chat/chat.py",
        "examples/modal_chat/configuration.py",
        "examples/modal_chat/consumer.py",
        "examples/modal_chat/launch.py",
        "examples/modal_chat/storage.py",
        "examples/modal_chat/training.py",
    }
)
_ENGINE_PATHS = frozenset(
    {
        "tuner/execution/providers/modal/inference_binding.py",
        "tuner/execution/providers/modal/inference_workload.py",
        "tuner/execution/providers/modal/inference_preparation.py",
        "tuner/execution/providers/modal/inference_run_chat.py",
        "tuner/execution/providers/modal/inference_commands.py",
        "tuner/execution/providers/modal/inference_retention.py",
        "tuner/execution/providers/modal/inference_transport.py",
        "tuner/execution/providers/modal/inference_effects.py",
        "tuner/execution/providers/modal/coordinator_reader.py",
        "tuner/execution/providers/modal/coordinator_submit_preparation.py",
        "tuner/execution/providers/modal/coordinator_bundle.py",
        "tuner/inference/run_chat.py",
        "synaptic_tuner/api/v1/runs_facade.py",
        "tuner/execution/coordinator_v1/operations.py",
        "tuner/execution/coordinator_v1/state_machine.py",
        "tuner/execution/foundation_v2/authority.py",
    }
)
_SDK_PATHS = frozenset(
    {
        "modal/app.py",
        "modal/runner.py",
        "modal/image.py",
        "modal/mount.py",
        "modal/client.py",
        "modal/_functions.py",
        "modal/_object.py",
        "modal/_resolver.py",
        "modal/_serialization.py",
    }
)


def _exception_class(error: BaseException) -> str:
    builtins = {
        TypeError: "TypeError",
        ValueError: "ValueError",
        RuntimeError: "RuntimeError",
        TimeoutError: "TimeoutError",
        KeyboardInterrupt: "KeyboardInterrupt",
        SystemExit: "SystemExit",
        ModuleNotFoundError: "ModuleNotFoundError",
        ImportError: "ImportError",
        PermissionError: "PermissionError",
        AttributeError: "AttributeError",
    }
    kind = type(error)
    if kind in builtins:
        return builtins[kind]
    module, name = getattr(kind, "__module__", ""), getattr(kind, "__name__", "")
    if module.startswith("modal.") and name in {
        "InvalidError",
        "ExecutionError",
        "SerializationError",
        "NotFoundError",
    }:
        return name
    if module.startswith("grpclib.") and name == "GRPCError":
        return name
    return "OTHER"


def _admitted_path(raw: str) -> str | None:
    path = raw.replace("\\", "/")
    for relative in _EXAMPLE_PATHS | _ENGINE_PATHS | _SDK_PATHS:
        if path == relative or path.endswith("/" + relative):
            return relative
    return None


def _exception_chain(error: BaseException) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    pending = [error]
    seen: set[int] = set()
    remaining = _MAX_FRAMES
    visited = 0
    while pending and len(result) < _MAX_EXCEPTIONS:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        locations: list[dict[str, object]] = []
        trace = current.__traceback__
        while trace is not None and remaining and visited < _MAX_VISITED_FRAMES:
            visited += 1
            relative = _admitted_path(trace.tb_frame.f_code.co_filename)
            line = trace.tb_lineno
            if relative is not None and type(line) is int and line > 0:
                locations.append({"path": relative, "line": line})
                remaining -= 1
            trace = trace.tb_next
        result.append(
            {"exception_class": _exception_class(current), "locations": locations}
        )
        # Inspect both links even when ``raise ... from None`` suppresses display.
        for linked in (current.__cause__, current.__context__):
            if linked is not None and id(linked) not in seen:
                pending.append(linked)
    return result


def modal_chat_failure_diagnostic(*, phase: str, error: BaseException) -> bytes:
    """Return bounded metadata which cannot authorize a retry or provider effect."""
    if type(phase) is not str or phase not in _PHASES:
        phase = "UNKNOWN"
    return canonical_bytes(
        {
            "schema_version": "synaptic-modal-chat-failure-diagnostic/v1",
            "status": "FAILED",
            "phase": phase,
            "retry_authorized": False,
            "authorizing": False,
            "exception_class": _exception_class(error),
            "exception_chain": _exception_chain(error),
        }
    )


__all__ = ["modal_chat_failure_diagnostic"]
