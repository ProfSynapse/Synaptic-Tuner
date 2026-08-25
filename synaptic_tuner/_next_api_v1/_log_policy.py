"""Closed structured-log policy; provider text never crosses this boundary."""

from __future__ import annotations

from enum import Enum


MAX_QUERY_LIMIT = 500
MAX_RECORD_BYTES = 16 * 1024
MAX_BATCH_RECORDS = 1_000
MAX_BATCH_BYTES = 1024 * 1024
MAX_RETAINED_RECORDS = 50_000
MAX_RETAINED_BYTES = 10 * 1024 * 1024


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EventCode(str, Enum):
    AUTHORIZED = "authorized"
    SUBMISSION_STARTED = "submission_started"
    SUBMITTED = "submitted"
    NOT_SUBMITTED = "not_submitted"
    SUBMISSION_AMBIGUOUS = "submission_ambiguous"
    RECONCILIATION_STARTED = "reconciliation_started"
    RECONCILED = "reconciled"
    STATE_OBSERVED = "state_observed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLATION_STARTED = "cancellation_started"
    CANCELLATION_ACCEPTED = "cancellation_accepted"
    CANCELLATION_FAILED = "cancellation_failed"
    CANCELLATION_AMBIGUOUS = "cancellation_ambiguous"
    LOGS_DROPPED = "logs_dropped"


class MessageCode(str, Enum):
    AUTHORITY_CONSUMED = "authority_consumed"
    EFFECT_STARTED = "effect_started"
    EFFECT_CONFIRMED = "effect_confirmed"
    EFFECT_DEFINITIVELY_ABSENT = "effect_definitively_absent"
    EFFECT_OUTCOME_UNKNOWN = "effect_outcome_unknown"
    EFFECT_COLLISION = "effect_collision"
    AUTHENTICATION_UNAVAILABLE = "authentication_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_PROTOCOL_VIOLATION = "provider_protocol_violation"
    PROVIDER_STATE_OBSERVED = "provider_state_observed"
    RETENTION_LIMIT_REACHED = "retention_limit_reached"


def checked_log_fields(
    level: LogLevel, event: EventCode, message: MessageCode
) -> tuple[str, str, str]:
    if not isinstance(level, LogLevel):
        raise TypeError("level must be LogLevel")
    if not isinstance(event, EventCode):
        raise TypeError("event must be EventCode")
    if not isinstance(message, MessageCode):
        raise TypeError("message must be MessageCode")
    fields = (level.value, event.value, message.value)
    if len("".join(fields).encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("structured log record exceeds policy")
    return fields


__all__ = [
    "EventCode", "LogLevel", "MAX_BATCH_BYTES", "MAX_BATCH_RECORDS",
    "MAX_QUERY_LIMIT", "MAX_RECORD_BYTES", "MAX_RETAINED_BYTES",
    "MAX_RETAINED_RECORDS", "MessageCode", "checked_log_fields",
]
