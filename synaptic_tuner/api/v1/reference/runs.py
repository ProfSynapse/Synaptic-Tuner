"""Runs reference composition: the ``RunsOperations`` implementation export.

Location: ``synaptic_tuner/api/v1/reference/runs.py``.

``RunsOperations`` (``api/v1/runs_facade.py``: list, show, outcome, logs,
cancel, reconcile, verify, reverify, artifacts) is implemented by
``RunOperationsV1`` (``tuner/execution/coordinator_v1/operations.py``). This
module is its public export site and builds it for the neutral composition
(``build_run_operations``). The verbs ``outcome``, ``verify`` and ``reverify``
live on the runs family, as the contract test pins; the roadmap text that
once placed them on training was corrected with this module.

Consumed by ``synaptic_tuner/api/v1/reference/provider_family.py``.
"""

from __future__ import annotations

from tuner.execution.coordinator_v1.operations import RunOperationsV1


def build_run_operations(
    *,
    planning,
    planning_store,
    workflow_store,
    coordinator,
    foundation,
    foundation_authenticator,
    assessment_authenticator,
    reader,
    observation_authenticator,
    log_authenticator,
    artifact_verifier,
    cursor_authority,
    clock,
) -> RunOperationsV1:
    """The ``RunsOperations`` implementation over a composed coordinator."""
    return RunOperationsV1(
        planning,
        planning_store,
        workflow_store,
        coordinator,
        foundation,
        foundation_authenticator,
        assessment_authenticator,
        reader,
        observation_authenticator,
        log_authenticator,
        artifact_verifier,
        cursor_authority,
        clock,
    )


__all__ = ["RunOperationsV1", "build_run_operations"]
