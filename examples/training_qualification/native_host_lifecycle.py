"""Restricted config-first public lifecycle qualification, never a grant issuer.

Only the root-signed host selection admits these effects. All API and storage
calls remain on the caller's thread. The caller-loss case deliberately exits
without Python cleanup after retaining an authenticated public-progress trigger;
another invocation may read/reconcile that original attempt, never start it.
"""

import hashlib
import os
import time

from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.runs_facade import RunLogsRequest
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from synaptic_tuner.api.v1.native_mlx import NativeCancellationMaterialV1, NativeLifecycleLaunchV1
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_worker_progress import validate_progress_budget


REPORT_LIMIT = 1 << 20
CALLER_LOSS_EXIT = 86
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def validate_policy(value):
    if (type(value) is not dict or set(value) != {
            "launch", "scenario", "trigger_step", "poll_interval_ms", "host_timeout_seconds",
            "recovery_timeout_seconds"}
            or type(value["scenario"]) is not str
            or value["scenario"] not in {"complete", "cancel", "caller-loss"}):
        raise ValueError("native lifecycle host policy differs")
    for name, lower, upper in (("poll_interval_ms", 10, 1000), ("host_timeout_seconds", 1, 600),
                                ("recovery_timeout_seconds", 1, 60)):
        if type(value[name]) is not int or not lower <= value[name] <= upper:
            raise ValueError("native lifecycle host bounds differ")
    if value["scenario"] == "complete":
        if value["trigger_step"] is not None:
            raise ValueError("native completion has no interruption trigger")
    elif type(value["trigger_step"]) is not int or not 1 <= value["trigger_step"] <= 1000000:
        raise ValueError("native interruption trigger differs")
    launch = NativeLifecycleLaunchV1.from_dict(value["launch"])
    if launch.policy.maximum_progress_records > 64:
        raise ValueError("native host report record bound exceeded")
    return launch


def validate_controls(value, projection, limits):
    launch = validate_policy(value)
    validate_progress_budget(projection, launch)
    controls = projection.controls
    if value["host_timeout_seconds"] < limits.timeout_seconds + limits.termination_grace_seconds + 10:
        raise ValueError("native host deadline cannot cover worker cleanup")
    if value["scenario"] != "complete":
        interval = (controls.logging_steps if projection.workload.method == "grpo"
                    else controls.steps_per_report)
        if (value["trigger_step"] >= controls.iterations or interval != 1):
            raise ValueError("native interruption needs an intermediate reported update")


class RestrictedLifecycleAuthority:
    """Authenticate a projection of the externally signed exact host selection.

The admission already commits the config (including mode/launch/operations) and
profile digest. This cannot issue an ordinary hardware/lifecycle qualification.
"""

    def __init__(self, expected, admission, authority):
        self.expected, self.admission, self.authority = expected, admission, authority

    def authenticate(self, grant):
        return (type(grant) is type(self.expected) and grant == self.expected
                and self.authority.authenticate(self.admission) is True)


class LifecycleGrants:
    def __init__(self, start, selection, catalog, authority):
        self.start, self.s, self.catalog, self.authority = start, selection, catalog, authority

    def authorize(self, requirements):
        if tuple(requirements) == (AuthorizationRequirement("training.cancel", False),):
            if (self.s["config"]["lifecycle"]["scenario"] != "cancel"
                    or self.s["root_authority"].authenticate(self.s["admission"]) is not True):
                raise ValueError("native selected scenario does not authorize stop")
            return ExecutionGrant(self.s["grant_ref"])
        return self.start.authorize(requirements)

    def bind(self, grant, *, operation, requirements):
        if operation.operation.effect.kind.value != "cancel":
            return self.start.bind(grant, operation=operation, requirements=requirements)
        if (tuple(requirements) != (AuthorizationRequirement("training.cancel", False),)
                or grant != self.authorize(requirements)):
            raise ValueError("native lifecycle stop grant differs")
        if (operation.preparation.run_id != self.s["run_id"]
                or operation.preparation.project_ref != self.s["project_ref"]):
            raise ValueError("native lifecycle stop scope differs")
        reference = operation.to_dict()["cancellation"]["provider_job_ref"]
        prefix = "native-run-"
        if not reference.startswith(prefix):
            raise ValueError("native lifecycle stop reference differs")
        original = self.catalog.resolve_submission(reference[len(prefix):])
        material = NativeCancellationMaterialV1(
            operation.canonical_bytes, original.command_digest, original.invocation.digest,
            self.s["material_authority_ref"], self.s["material_key_ref"], "0" * 64)
        self.catalog.publish_once(self.authority.issue(material))
        return grant


def _entries(config, api, run):
    """Read all bounded public pages; a page/prefix is never called complete."""
    maximum = validate_policy(config["lifecycle"]).policy.maximum_progress_records
    entries, cursor = [], None
    for _ in range(maximum + 1):
        page = api.runs.logs(RunLogsRequest(run, cursor=cursor, limit=100, maximum_bytes=65536))
        entries.extend(item.to_dict() for item in page.entries)
        if len(entries) > maximum or len(canonical(entries)) > REPORT_LIMIT // 2:
            raise ValueError("native lifecycle public log bound exceeded")
        if not page.truncated:
            return entries
        if page.next_cursor == cursor:
            raise ValueError("native lifecycle public cursor did not advance")
        cursor = page.next_cursor
    raise ValueError("native lifecycle public pagination bound exceeded")


def logs(config, api, run):
    return {"run": run.to_dict(), "entries": _entries(config, api, run)}


def _report(config, run, outcome, entries, *, trigger, cancel_ack=None, pending_progress=()):
    return {"schema": "synaptic-native-host-lifecycle-observation/v1",
            "selection_config_sha256": hashlib.sha256(canonical(config)).hexdigest(),
            "run": run.to_dict(), "scenario": config["lifecycle"]["scenario"],
            "outcome": outcome.to_dict(), "entries": entries,
            "trigger": trigger, "cancel_ack": cancel_ack, "pending_progress": list(pending_progress)}


def qualify(config, api, started, state, write):
    policy = config["lifecycle"]
    validate_policy(policy)
    if started.accepted is not True:
        raise ValueError("native lifecycle start was not accepted")
    deadline = time.monotonic() + policy["host_timeout_seconds"]
    trigger, cancel_ack = None, None
    pending_progress, observed = [], set()
    while time.monotonic() < deadline:
        before = api.runs.outcome(started.run)
        entries = _entries(config, api, started.run)
        outcome = api.runs.outcome(started.run)
        pending = before.state.value == "running" and outcome.state.value == "running"
        for entry in entries:
            if entry["sequence"] not in observed:
                observed.add(entry["sequence"])
                if pending:
                    pending_progress.append({"sequence": entry["sequence"],
                        "before": before.to_dict(), "after": outcome.to_dict()})
        if trigger is None and policy["scenario"] != "complete" and pending:
            for entry in entries:
                event = strict_json(entry["message"].encode(), maximum=4096)
                if event["phase"] == "training" and event["step"] >= policy["trigger_step"]:
                    trigger = {"entry": entry, "observed_outcome": outcome.to_dict(),
                               "before_logs": before.to_dict()}
                    if policy["scenario"] == "caller-loss":
                        write(state, "caller-loss-trigger.json", canonical(_report(
                            config, started.run, outcome, entries, trigger=trigger,
                            pending_progress=pending_progress)), REPORT_LIMIT)
                        # Real process exit: do not execute session/store finally
                        # handlers. The guardian detects anonymous owner-pipe EOF.
                        os._exit(CALLER_LOSS_EXIT)
                    cancel_ack = api.runs.cancel(started.run, "bounded lifecycle qualification").to_dict()
                    break
        if outcome.state.value in _TERMINAL:
            report = _report(config, started.run, outcome, entries,
                             trigger=trigger, cancel_ack=cancel_ack, pending_progress=pending_progress)
            write(state, "lifecycle-observation.json", canonical(report), REPORT_LIMIT)
            # Preserve observed completion races/failures; do not call them a
            # qualified cancellation just because a stop was requested.
            return report
        time.sleep(policy["poll_interval_ms"] / 1000)
    raise ValueError("native lifecycle host observation deadline")


def recover(config, api, run, state, write):
    policy = config["lifecycle"]
    validate_policy(policy)
    deadline = time.monotonic() + policy["recovery_timeout_seconds"]
    while time.monotonic() < deadline:
        api.runs.reconcile(run)
        outcome = api.runs.outcome(run)
        entries = _entries(config, api, run)
        if outcome.state.value in _TERMINAL:
            report = _report(config, run, outcome, entries, trigger=None)
            write(state, "recovery-observation.json", canonical(report), REPORT_LIMIT)
            return report
        time.sleep(policy["poll_interval_ms"] / 1000)
    raise ValueError("native lifecycle cold observation deadline")
