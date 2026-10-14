"""Optional public RunsAPI source join for caller-owned publication ports.

The trusted RunsAPI must expose artifacts in stored show() only after native
verification, as the reference RunOperationsV1 does. The verification digest
below commits that public stored projection; it is NOT a native receipt digest.
No method verifies, observes, renews an assessment or submits a training run.
"""

from dataclasses import replace

from synaptic_tuner.api.v1.artifacts_facade import ArtifactsAPI
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunOutcome, RunsAPI
from tuner.execution.coordinator_v1.publication import (
    AuthenticatedVerifiedSourceV1,
    PublicationOperationsV1,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    domain_digest,
    safe_ref,
)

from .authority import ReferencePublicationEvidenceAuthorityV1


class RunsArtifactSourceV1:
    """Sign stored verified manifests; open original public artifact streams."""

    def __init__(self, runs, *, authority_ref, authority):
        if (
            type(runs) is not RunsAPI
            or type(authority) is not ReferencePublicationEvidenceAuthorityV1
        ):
            raise TypeError("exact publication source collaborators required")
        self.runs = runs
        self.authority_ref = safe_ref(authority_ref, "authority_ref")
        self.authority = authority

    def describe(self, run):
        if type(run) is not TrainingRunRef:
            raise TypeError("exact TrainingRunRef required")
        run = TrainingRunRef.from_dict(run.to_dict())
        outcome = self.runs.show(run)
        if type(outcome) is not RunOutcome:
            raise ValueError("publication_source_unavailable")
        outcome = RunOutcome.from_dict(outcome.to_dict())
        if (
            outcome.run != run
            or outcome.state is not TrainingRunState.SUCCEEDED
            or not outcome.artifacts
        ):
            raise ValueError("publication_source_unavailable")
        basis = canonical_bytes(
            {
                "run": run.to_dict(),
                "state": outcome.state.value,
                "artifacts": [item.to_dict() for item in outcome.artifacts],
            }
        )
        value = AuthenticatedVerifiedSourceV1(
            "synaptic-publication-verified-source/v1",
            run,
            outcome.artifacts,
            domain_digest("synaptic-publication-stored-verified-run/v1", basis),
            self.authority_ref,
            self.authority.key_ref,
            "0" * 64,
        )
        return replace(
            value,
            tag=self.authority.sign("publication-verified-source/v1", value.payload),
        )

    def open(self, request):
        if type(request) is not RunArtifactRequest:
            raise TypeError("exact RunArtifactRequest required")
        request = RunArtifactRequest(
            TrainingRunRef.from_dict(request.run.to_dict()),
            request.role,
            request.maximum_bytes,
        )
        current = self.describe(request.run)
        matches = tuple(item for item in current.artifacts if item.role == request.role)
        if len(matches) != 1:
            raise ValueError("publication_source_unavailable")
        stream = self.runs.artifacts(request)
        if stream.artifact != matches[0]:
            raise ValueError("publication_source_unavailable")
        return stream


def compose_publication_artifacts(
    *, runs, store, spool, destinations, authority_ref, authority, clock
):
    """Join existing caller-owned ports into ArtifactsAPI; close nothing.

    The caller owns original authority keys, generic persistence, OS ownership,
    spool/destination deadlines and all object lifetimes. Defaults are untouched.
    Existing publication recovery describes only stored show() metadata; it never
    reopens native streams unless a genuinely new transfer is admitted.
    """
    if not callable(clock):
        raise TypeError("publication clock must be callable")
    source = RunsArtifactSourceV1(
        runs, authority_ref=authority_ref, authority=authority
    )
    return ArtifactsAPI(
        PublicationOperationsV1(
            store=store,
            spool=spool,
            destinations=destinations,
            sources=source,
            authority=authority,
            clock=clock,
        )
    )
