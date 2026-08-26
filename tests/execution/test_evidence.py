from __future__ import annotations

import pytest

from tuner.execution.evidence import (
    EvidenceFreshnessPolicyV1, ReplayDisposition, admit_evidence, canonical_utc,
    validate_evidence_window,
)


@pytest.mark.parametrize("value",[
    "2026-08-25T12:00:00+00:00","2026-08-25T12:00:00.1Z",
    "2026-08-25 12:00:00Z","2026-02-30T12:00:00Z","not-a-timeZ",
])
def test_canonical_utc_rejects_every_noncanonical_form(value):
    with pytest.raises(ValueError):canonical_utc(value,"time")


def test_evidence_window_rejects_stale_expired_future_and_excessive_lifetime():
    policy=EvidenceFreshnessPolicyV1(300,300,30);now="2026-08-25T12:05:00Z"
    invalid=(
        ("2026-08-25T11:59:59Z","2026-08-25T12:09:00Z"),
        ("2026-08-25T12:00:00Z","2026-08-25T12:05:00Z"),
        ("2026-08-25T12:05:31Z","2026-08-25T12:06:00Z"),
        ("2026-08-25T12:04:00Z","2026-08-25T12:10:00Z"),
    )
    for verified,expires in invalid:
        with pytest.raises(ValueError):validate_evidence_window(verified_at=verified,expires_at=expires,now=now,policy=policy)


class Replay:
    def __init__(self):self.values={}
    def admit(self,**value):
        key=(value["purpose"],value["challenge_nonce"]);prior=self.values.get(key)
        if prior is None:self.values[key]=value;return ReplayDisposition.ADMITTED
        return ReplayDisposition.IDEMPOTENT if prior==value else ReplayDisposition.COLLISION


def test_replay_is_idempotent_only_for_the_identical_evidence():
    repo=Replay();values=dict(purpose="source-proof",issuer_ref="issuer",evidence_ref="evidence",challenge_nonce="nonce",audience_ref="project/run",payload_digest="a"*64,expires_at="2026-08-25T12:10:00Z")
    admit_evidence(repo,**values);admit_evidence(repo,**values)
    with pytest.raises(ValueError,match="collision"):admit_evidence(repo,**{**values,"audience_ref":"project/other"})
