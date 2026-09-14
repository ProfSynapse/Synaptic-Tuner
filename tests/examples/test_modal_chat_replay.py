"""Real SQLite replay admission; no remote evidence authentication is simulated."""

import os

import pytest

from examples.modal_chat.replay import ModalChatEvidenceReplay
from examples.modal_chat.storage import ModalChatStorage, ModalChatStorageError
from tuner.execution.evidence import ReplayDisposition, admit_evidence


def _evidence():
    return dict(
        purpose="source-lock-evidence/v1",
        issuer_ref="consumer-source",
        evidence_ref="evidence-a",
        challenge_nonce="challenge-a",
        audience_ref="consumer/run-a",
        payload_digest="a" * 64,
        expires_at="2026-09-14T12:10:00Z",
    )


def test_admission_survives_reopen_and_is_exactly_idempotent(tmp_path):
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "smoke.sqlite3"
    with ModalChatStorage(path, "consumer") as storage:
        replay = ModalChatEvidenceReplay(storage)
        assert replay.admit(**_evidence()) is ReplayDisposition.ADMITTED
        assert replay.admit(**_evidence()) is ReplayDisposition.IDEMPOTENT
    with ModalChatStorage(path, "consumer") as storage:
        replay = ModalChatEvidenceReplay(storage)
        assert replay.admit(**_evidence()) is ReplayDisposition.IDEMPOTENT
        admit_evidence(replay, **_evidence())


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("issuer_ref", "other-issuer"),
        ("evidence_ref", "other-evidence"),
        ("audience_ref", "consumer/other-run"),
        ("payload_digest", "b" * 64),
        ("expires_at", "2026-09-14T12:11:00Z"),
    ],
)
def test_changed_binding_is_collision_not_an_idempotent_read(
    tmp_path, field, replacement
):
    os.chmod(tmp_path, 0o700)
    with ModalChatStorage(tmp_path / "smoke.sqlite3", "consumer") as storage:
        replay = ModalChatEvidenceReplay(storage)
        original = _evidence()
        assert replay.admit(**original) is ReplayDisposition.ADMITTED
        changed = {**original, field: replacement}
        assert replay.admit(**changed) is ReplayDisposition.COLLISION
        with pytest.raises(ValueError, match="replay collision"):
            admit_evidence(replay, **changed)
        assert replay.admit(**original) is ReplayDisposition.IDEMPOTENT


def test_purpose_and_challenge_keys_do_not_alias(tmp_path):
    os.chmod(tmp_path, 0o700)
    with ModalChatStorage(tmp_path / "smoke.sqlite3", "consumer") as storage:
        replay = ModalChatEvidenceReplay(storage)
        assert replay.admit(**_evidence()) is ReplayDisposition.ADMITTED
        assert (
            replay.admit(**{**_evidence(), "purpose": "modal-deployment-evidence/v1"})
            is ReplayDisposition.ADMITTED
        )
        assert (
            replay.admit(**{**_evidence(), "challenge_nonce": "challenge-b"})
            is ReplayDisposition.ADMITTED
        )


@pytest.mark.parametrize(
    "field,value",
    [("payload_digest", "bad"), ("expires_at", "tomorrow"), ("challenge_nonce", "")],
)
def test_invalid_input_never_claims_challenge(tmp_path, field, value):
    os.chmod(tmp_path, 0o700)
    with ModalChatStorage(tmp_path / "smoke.sqlite3", "consumer") as storage:
        replay = ModalChatEvidenceReplay(storage)
        with pytest.raises((ValueError, TypeError)):
            replay.admit(**{**_evidence(), field: value})
        assert replay.admit(**_evidence()) is ReplayDisposition.ADMITTED


def test_closed_storage_does_not_admit(tmp_path):
    os.chmod(tmp_path, 0o700)
    storage = ModalChatStorage(tmp_path / "smoke.sqlite3", "consumer")
    replay = ModalChatEvidenceReplay(storage)
    storage.close()
    with pytest.raises(ModalChatStorageError, match="closed"):
        replay.admit(**_evidence())
