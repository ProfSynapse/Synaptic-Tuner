"""Protected provider-free preparation and one exact HF provisioning operation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from argparse import Namespace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from shared.experiment_tracking.experiment import load_experiment
from shared.experiment_tracking.service import TrackingService
from tuner.cloud.hf_provisioning_claim import (
    build_hf_provisioning_ambiguous_event,
    build_hf_provisioning_claim,
    build_hf_provisioning_succeeded_event,
)
from tuner.cloud.hf_provisioning import canonical_json_bytes
from tuner.cloud.runtime_layout import CloudRuntimeLayout, RuntimeMount
from tuner.core.exceptions import CloudProviderError
from tuner.handlers._hf_secret_file import (
    HFSecretFileClaim,
    preflight_hf_secret_file,
    read_claimed_hf_token,
)
from tuner.handlers.base import BaseHandler
from tuner.project.source_bundle import SourceLock


class HFSourceHandler(BaseHandler):
    """Prepare or provision one immutable Profile-C transport; never submit."""

    def __init__(self, args: Namespace | None = None, context=None, *, provider_factory: Callable[..., Any] | None = None) -> None:
        super().__init__(args=args, context=context)
        self._provider_factory = provider_factory

    @property
    def name(self) -> str:
        return "hf-source"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            action = str(getattr(self.args, "subcommand", "") or "").strip().lower()
            if action == "prepare":
                result = self.prepare()
                message = "HF source transport is PREPARED; no provider was contacted."
            elif action == "provision":
                outcome = self.provision()
                if not outcome.succeeded:
                    assert outcome.failure is not None
                    self.output_error(outcome.failure.message, code=outcome.failure.code)
                    return 1
                result = {"status": "CONSUMABLE", "evidence_sha256": outcome.evidence_sha256, "provider_mutated": outcome.mutated, "submitted": False}
                message = "HF source transport is verified as CONSUMABLE; no job was submitted."
            else:
                raise CloudProviderError("hf-source requires prepare or provision.")
            self.output(result, message)
            return 0
        except Exception as exc:
            self.output_error(str(exc), code="HF_SOURCE_PROVISIONING_ERROR")
            return 1

    def prepare(self) -> dict[str, object]:
        """Create/reuse neutral tracking, preflight Git, and persist PREPARED state."""

        args = self._require_args()
        crossovers = (
            ("--env-file", getattr(args, "env_file", None)),
            ("--actor", getattr(args, "actor", None)),
            ("--authority", getattr(args, "authority", None)),
        )
        selected = [flag for flag, value in crossovers if value is not None]
        if selected:
            raise CloudProviderError(
                f"hf-source prepare does not accept {', '.join(selected)}."
            )
        base_dir = _require_external_base_dir(getattr(args, "base_dir", None), context=self.context)
        config_path = _require_source_config(getattr(args, "source_config", None), context=self.context)
        tracking = TrackingService(base_dir=base_dir)
        experiment_id = str(getattr(args, "experiment_id", "") or "").strip()
        created = False
        if experiment_id:
            experiment = load_experiment(experiment_id, tracking.base_dir)
            _require_recoverable_bootstrap_experiment(experiment)
        else:
            experiment = tracking.create_experiment(
                name="hf-bootstrap-source", dataset_path="",
                dataset_hash=hashlib.sha256(b"").hexdigest(), base_model_name="",
                provider="hf_jobs", method="bootstrap",
                objective="immutable source/bootstrap verification",
            )
            created = True

        # Deliberately below validation; this path never imports a provider, ML, or UI module.
        from tuner.handlers.stages._util import (
            finalize_hf_source_lock,
            prepare_hf_source,
            preflight_hf_source_lock,
            validate_finalized_hf_source_lock,
        )

        try:
            with tracking.hf_source_preparation_execution_lock(experiment.experiment_id):
                experiment = load_experiment(experiment.experiment_id, tracking.base_dir)
                _require_recoverable_bootstrap_experiment(experiment)
                run_id = experiment.experiment_id
                runtime = {"provider": "hf_jobs", "task": "bootstrap_verification"}
                outputs = {"publication": False, "submission": False}
                source_lock_path = (
                    tracking.base_dir / "experiments" / run_id / "source-lock.json"
                )
                source_lock_uri = tracking.tracking_uri(source_lock_path)
                transport_root = (
                    tracking.base_dir
                    / "experiments" / run_id / "cloud" / "hf" / "source-transport"
                ).resolve()
                descriptor_uri = tracking.tracking_uri(transport_root / "descriptor.json")

                if experiment.source_transport_state == "PREPARED":
                    source_lock = tracking.load_source_lock(experiment)
                    validate_finalized_hf_source_lock(
                        self.context, source_lock=source_lock, run_id=run_id,
                        config_path=config_path, source_mode=getattr(args, "source_mode", None),
                        runtime=runtime, outputs=outputs,
                    )
                    prepared = _load_exact_prepared_transport(
                        tracking=tracking, experiment=experiment, context=self.context
                    )
                    _validate_prepared_volume_policy(
                        prepared.descriptor,
                        _load_committed_volume_settings(
                            config_path, context=self.context, source_lock=source_lock
                        ),
                        run_id=run_id,
                    )
                else:
                    if experiment.source_lock_uri is not None:
                        source_lock = tracking.load_source_lock(experiment)
                        validate_finalized_hf_source_lock(
                            self.context, source_lock=source_lock, run_id=run_id,
                            config_path=config_path, source_mode=getattr(args, "source_mode", None),
                            runtime=runtime, outputs=outputs,
                        )
                    else:
                        orphan_path = source_lock_path
                        if not _path_entry_exists(orphan_path) and _path_entry_exists(transport_root):
                            orphan_path = transport_root / "bundle" / "source-lock.json"
                        if _path_entry_exists(orphan_path):
                            source_lock = _load_orphan_source_lock(
                                orphan_path, run_id=run_id, root=tracking.base_dir
                            )
                            validate_finalized_hf_source_lock(
                                self.context, source_lock=source_lock, run_id=run_id,
                                config_path=config_path,
                                source_mode=getattr(args, "source_mode", None),
                                runtime=runtime, outputs=outputs,
                            )
                        else:
                            source_lock = finalize_hf_source_lock(
                                self.context,
                                source_lock=preflight_hf_source_lock(
                                    self.context, run_id=run_id,
                                    source_mode=getattr(args, "source_mode", None),
                                ),
                                run_id=run_id, config_path=config_path,
                                runtime=runtime, outputs=outputs,
                            )
                        source_lock = tracking.persist_source_lock(experiment, source_lock)

                    volume_settings = _load_committed_volume_settings(
                        config_path, context=self.context, source_lock=source_lock
                    )
                    preparation = prepare_hf_source(
                        self.context, run_id=run_id, config_path=config_path,
                        volume_settings=volume_settings,
                        source_mode=getattr(args, "source_mode", None),
                        runtime=runtime, outputs=outputs, source_lock=source_lock,
                        runtime_layout=_external_runtime_layout(self.context, base_dir),
                        source_lock_uri=source_lock_uri, descriptor_uri=descriptor_uri,
                        transport_root=transport_root,
                    )
                    tracking.record_source_transport_prepared(
                        experiment, uri=str(preparation.descriptor_uri),
                        sha256=str(preparation.descriptor_sha256),
                    )
                    prepared = _load_exact_prepared_transport(
                        tracking=tracking, experiment=experiment, context=self.context
                    )

                document = prepared.descriptor
                volume = document["volume"]
        except Exception as exc:
            if created:
                raise CloudProviderError(
                    "hf-source prepare created recoverable neutral experiment "
                    f"{experiment.experiment_id}; rerun with --experiment-id "
                    f"{experiment.experiment_id}: {exc}"
                ) from exc
            raise
        return {
            "status": "PREPARED", "experiment_id": experiment.experiment_id,
            "source_lock": {"uri": experiment.source_lock_uri, "sha256": experiment.source_lock_sha256},
            "descriptor": {"uri": experiment.source_transport_uri, "sha256": experiment.source_transport_sha256},
            "volume": {key: volume[key] for key in ("type", "source", "path", "mount_path", "read_only")},
            "provider_contacted": False, "submitted": False,
        }

    def provision(self):
        args = self._require_args()
        if getattr(args, "source_config", None) is not None:
            raise CloudProviderError("hf-source provision does not accept --source-config.")
        experiment_id = str(getattr(args, "experiment_id", "") or "").strip()
        actor = str(getattr(args, "actor", "") or "").strip()
        authority = str(getattr(args, "authority", "") or "operator").strip()
        if not experiment_id or not actor:
            raise CloudProviderError("HF source provisioning requires experiment_id and actor.")
        base_dir = _require_external_base_dir(getattr(args, "base_dir", None), context=self.context)
        tracking = TrackingService(base_dir=base_dir)
        experiment = load_experiment(experiment_id, tracking.base_dir)
        if experiment.source_transport_state not in {"PREPARED", "ACKNOWLEDGED", "CONSUMABLE"}:
            raise CloudProviderError(
                "HF source provisioning requires PREPARED state or an exact persisted recovery."
            )
        tracking.verify_experiment_provenance(experiment)
        required = (experiment.source_transport_uri, experiment.source_transport_sha256, experiment.source_lock_uri, experiment.source_lock_sha256)
        if any(not isinstance(value, str) or not value for value in required):
            raise CloudProviderError("HF PREPARED tracking provenance is incomplete.")
        descriptor_path = tracking.resolve_uri(experiment.source_transport_uri)
        if descriptor_path.name != "descriptor.json":
            raise CloudProviderError("HF tracked source transport must identify descriptor.json.")

        from tuner.cloud.hf_provisioning import load_hf_source_transport

        prepared = load_hf_source_transport(
            self.context, transport_root=descriptor_path.parent,
            descriptor_uri=experiment.source_transport_uri,
            source_lock_uri=experiment.source_lock_uri,
        )
        if prepared.descriptor_sha256 != experiment.source_transport_sha256:
            raise CloudProviderError("HF PREPARED descriptor tracking digest changed.")
        if prepared.descriptor["source_lock"]["sha256"] != experiment.source_lock_sha256:
            raise CloudProviderError("HF PREPARED SourceLock tracking digest changed.")

        from tuner.cloud.hf_provisioning import consume_hf_source_transport
        from tuner.cloud.hf_provisioning_operator import provision_hf_source_transport

        with tracking.hf_provisioning_execution_lock(experiment.experiment_id):
            experiment = load_experiment(experiment.experiment_id, tracking.base_dir)
            claim = _recover_hf_provisioning_claim(tracking, experiment)
            if claim is not None:
                if experiment.hf_provisioning_state == "SUCCEEDED":
                    return _recover_succeeded_hf_provisioning(
                        tracking=tracking, experiment=experiment, context=self.context,
                        descriptor_path=descriptor_path,
                    )
                if experiment.hf_provisioning_state == "AMBIGUOUS":
                    raise CloudProviderError(
                        "HF source provisioning authority is terminally consumed; retry is prohibited."
                    )
                return _recover_claimed_hf_provisioning(
                    tracking=tracking,
                    experiment=experiment,
                    claim=claim,
                    context=self.context,
                    descriptor_path=descriptor_path,
                )

            # Selection authenticates path metadata only. Recovery never needs
            # a credential; fresh content remains behind the durable claim.
            secret_claim = preflight_hf_secret_file(
                getattr(args, "env_file", None), context=self.context
            )
            claim = build_hf_provisioning_claim(
                experiment_id=experiment.experiment_id,
                descriptor_uri=str(experiment.source_transport_uri),
                descriptor_sha256=str(experiment.source_transport_sha256),
                descriptor=prepared.descriptor,
                actor=actor,
                authority=authority,
            )
            claim_result = tracking.claim_hf_provisioning(experiment, claim)
            if not claim_result.provider_attempt_authorized:
                raise CloudProviderError(
                    "HF source provisioning authority is already consumed; "
                    "provider retry is prohibited."
                )

            try:
                token = read_claimed_hf_token(secret_claim)
            except Exception:
                _record_hf_provisioning_ambiguity(
                    tracking, experiment, claim_result, "CREDENTIAL_REJECTED"
                )
                raise CloudProviderError(
                    "HF source provisioning credential was rejected after authority claim; retry is prohibited."
                ) from None

            try:
                if self._provider_factory is None:
                    from tuner.cloud.hf_provider_adapter import load_hf_jp_provider

                    provider_factory = load_hf_jp_provider
                else:
                    provider_factory = self._provider_factory
                provider = provider_factory(token=token)
            except Exception:
                _record_hf_provisioning_ambiguity(
                    tracking, experiment, claim_result, "LOCAL_POSTCLAIM_FAILURE"
                )
                raise CloudProviderError(
                    "HF source provisioning failed locally after authority claim; retry is prohibited."
                ) from None

            try:
                outcome = provision_hf_source_transport(
                    self.context,
                    transport_root=descriptor_path.parent,
                    descriptor_uri=str(experiment.source_transport_uri),
                    source_lock_uri=str(experiment.source_lock_uri),
                    provider=provider,
                    actor=actor,
                    authority=authority,
                )
            except Exception:
                _record_hf_provisioning_ambiguity(
                    tracking, experiment, claim_result, "LOCAL_POSTCLAIM_FAILURE"
                )
                raise CloudProviderError(
                    "HF source provisioning failed locally after authority claim; retry is prohibited."
                ) from None
            if not outcome.succeeded:
                _record_hf_provisioning_ambiguity(
                    tracking, experiment, claim_result, "PROVIDER_OUTCOME_AMBIGUOUS"
                )
                return outcome

            try:
                assert outcome.evidence is not None
                assert outcome.evidence_sha256 is not None
                evidence_path = descriptor_path.with_name("provisioning-evidence.json")
                evidence_uri = tracking.tracking_uri(evidence_path)
                _persist_immutable(evidence_path, canonical_json_bytes(outcome.evidence))
                consumed = consume_hf_source_transport(
                    self.context,
                    transport_root=descriptor_path.parent,
                    descriptor_uri=str(experiment.source_transport_uri),
                    source_lock_uri=str(experiment.source_lock_uri),
                    evidence=outcome.evidence,
                )
                if consumed.evidence_sha256 != outcome.evidence_sha256:
                    raise CloudProviderError(
                        "HF JP evidence digest changed before local consumption."
                    )
                terminal = build_hf_provisioning_succeeded_event(
                    claim_result.document,
                    claim_uri=claim_result.event_uri,
                    claim_sha256=claim_result.event_sha256,
                    evidence_uri=evidence_uri,
                    evidence_sha256=outcome.evidence_sha256,
                )
                tracking.record_hf_provisioning_succeeded(
                    experiment,
                    terminal,
                    evidence_uri=evidence_uri,
                    evidence_sha256=outcome.evidence_sha256,
                )
                tracking.mark_source_transport_consumable(experiment)
                return outcome
            except Exception:
                durable = load_experiment(experiment.experiment_id, tracking.base_dir)
                if durable.hf_provisioning_state == "SUCCEEDED":
                    return _recover_succeeded_hf_provisioning(
                        tracking=tracking,
                        experiment=durable,
                        context=self.context,
                        descriptor_path=descriptor_path,
                    )
                if durable.hf_provisioning_state == "CLAIMED":
                    _record_hf_provisioning_ambiguity(
                        tracking, durable, claim_result, "PROVIDER_OUTCOME_AMBIGUOUS"
                    )
                raise CloudProviderError(
                    "HF source provisioning ended without verified success; "
                    "authority is consumed and provider retry is prohibited."
                ) from None

    def _require_args(self) -> Namespace:
        if self.args is None:
            raise CloudProviderError("HF source requires explicit arguments.")
        return self.args


def _recover_hf_provisioning_claim(
    tracking: TrackingService, experiment
) -> Mapping[str, object] | None:
    """Return the exact original CLAIMED document for a durable projection."""

    if experiment.hf_provisioning_state is None:
        return None
    uri = str(experiment.hf_provisioning_event_uri or "")
    if not uri:
        raise CloudProviderError("HF provisioning recovery event is unavailable.")
    from tuner.cloud.hf_provisioning import load_canonical_json

    current = load_canonical_json(tracking.resolve_uri(uri), maximum_bytes=64 * 1024)
    if current.get("state") == "CLAIMED":
        return current
    previous = current.get("previous_event")
    if not isinstance(previous, Mapping) or set(previous) != {"uri", "sha256"}:
        raise CloudProviderError("HF provisioning recovery claim is unavailable.")
    claim = load_canonical_json(
        tracking.resolve_uri(str(previous["uri"])), maximum_bytes=64 * 1024
    )
    if claim.get("state") != "CLAIMED":
        raise CloudProviderError("HF provisioning recovery predecessor is invalid.")
    return claim


def _record_hf_provisioning_ambiguity(
    tracking: TrackingService,
    experiment,
    claim_result,
    reason_code: str,
) -> None:
    terminal = build_hf_provisioning_ambiguous_event(
        claim_result.document,
        claim_uri=claim_result.event_uri,
        claim_sha256=claim_result.event_sha256,
        reason_code=reason_code,
    )
    tracking.record_hf_provisioning_ambiguous(experiment, terminal)


def _recover_claimed_hf_provisioning(
    *, tracking: TrackingService, experiment, claim: Mapping[str, object], context,
    descriptor_path: Path,
):
    """Close a stranded CLAIMED state without ever granting provider authority."""

    claim_result = tracking.claim_hf_provisioning(experiment, claim)
    if claim_result.provider_attempt_authorized or claim_result.state != "CLAIMED":
        raise CloudProviderError("HF provisioning recovery claim state is invalid.")
    try:
        terminal = tracking.find_hf_provisioning_terminal(experiment)
    except Exception:
        _record_hf_provisioning_ambiguity(
            tracking, experiment, claim_result, "RECOVERY_EVIDENCE_INVALID"
        )
        raise CloudProviderError(
            "HF provisioning recovery evidence is invalid; retry is prohibited."
        ) from None
    if terminal is not None:
        if terminal.state == "AMBIGUOUS":
            tracking.record_hf_provisioning_ambiguous(experiment, terminal.document)
            raise CloudProviderError(
                "HF source provisioning authority is terminally consumed; retry is prohibited."
            )
        if terminal.state != "SUCCEEDED":
            raise CloudProviderError("HF provisioning orphan terminal state is invalid.")
        evidence_ref = terminal.document.get("evidence")
        if not isinstance(evidence_ref, Mapping):
            raise CloudProviderError("HF provisioning orphan success evidence is unavailable.")
        try:
            _validate_recovery_evidence(
                tracking=tracking, experiment=experiment, claim=claim,
                descriptor_path=descriptor_path, context=context,
                evidence_uri=str(evidence_ref.get("uri") or ""),
                evidence_sha256=str(evidence_ref.get("sha256") or ""),
            )
        except Exception:
            _record_hf_provisioning_ambiguity(
                tracking, experiment, claim_result, "RECOVERY_EVIDENCE_INVALID"
            )
            raise CloudProviderError(
                "HF provisioning recovery evidence is invalid; retry is prohibited."
            ) from None
        tracking.record_hf_provisioning_succeeded(
            experiment, terminal.document,
            evidence_uri=str(evidence_ref["uri"]),
            evidence_sha256=str(evidence_ref["sha256"]),
        )
        return _recover_succeeded_hf_provisioning(
            tracking=tracking, experiment=experiment, context=context,
            descriptor_path=descriptor_path,
        )

    evidence_path = descriptor_path.with_name("provisioning-evidence.json")
    if not _path_entry_exists(evidence_path):
        _record_hf_provisioning_ambiguity(
            tracking, experiment, claim_result, "INTERRUPTED_AFTER_CLAIM"
        )
        raise CloudProviderError(
            "HF provisioning was interrupted after authority claim; retry is prohibited."
        )
    evidence_uri = tracking.tracking_uri(evidence_path)
    try:
        evidence_sha256 = _validate_recovery_evidence(
            tracking=tracking, experiment=experiment, claim=claim,
            descriptor_path=descriptor_path, context=context,
            evidence_uri=evidence_uri, evidence_sha256=None,
        )
        terminal_document = build_hf_provisioning_succeeded_event(
            claim,
            claim_uri=claim_result.event_uri,
            claim_sha256=claim_result.event_sha256,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha256,
        )
        tracking.record_hf_provisioning_succeeded(
            experiment, terminal_document,
            evidence_uri=evidence_uri, evidence_sha256=evidence_sha256,
        )
    except Exception:
        durable = load_experiment(experiment.experiment_id, tracking.base_dir)
        if durable.hf_provisioning_state == "CLAIMED":
            _record_hf_provisioning_ambiguity(
                tracking, durable, claim_result, "RECOVERY_EVIDENCE_INVALID"
            )
        raise CloudProviderError(
            "HF provisioning recovery evidence is invalid; retry is prohibited."
        ) from None
    return _recover_succeeded_hf_provisioning(
        tracking=tracking, experiment=experiment, context=context,
        descriptor_path=descriptor_path,
    )


def _validate_recovery_evidence(
    *, tracking: TrackingService, experiment, claim: Mapping[str, object],
    descriptor_path: Path, context, evidence_uri: str,
    evidence_sha256: str | None,
) -> str:
    from tuner.cloud.hf_provisioning import consume_hf_source_transport, load_canonical_json

    expected_path = descriptor_path.with_name("provisioning-evidence.json").resolve()
    if not evidence_uri or tracking.resolve_uri(evidence_uri).resolve() != expected_path:
        raise CloudProviderError("HF provisioning recovery evidence path is not canonical.")
    evidence = load_canonical_json(expected_path, maximum_bytes=64 * 1024)
    if evidence.get("actor") != claim.get("actor") or evidence.get("authority") != claim.get("authority"):
        raise CloudProviderError("HF provisioning recovery evidence authority changed.")
    consumed = consume_hf_source_transport(
        context,
        transport_root=descriptor_path.parent,
        descriptor_uri=str(experiment.source_transport_uri),
        source_lock_uri=str(experiment.source_lock_uri),
        evidence=evidence,
    )
    if evidence_sha256 is not None and consumed.evidence_sha256 != evidence_sha256:
        raise CloudProviderError("HF provisioning recovery evidence digest changed.")
    return consumed.evidence_sha256


def _load_orphan_source_lock(path: Path, *, run_id: str, root: Path) -> SourceLock:
    """Load one bounded canonical regular SourceLock without following links."""

    _assert_regular_link_free(root.resolve(strict=True), path)
    raw = _read_bounded_regular_nofollow(path, maximum_bytes=256 * 1024)
    try:
        document = json.loads(raw.decode("utf-8"))
        source_lock = SourceLock.from_dict(document)
    except Exception as exc:
        raise CloudProviderError("HF orphan SourceLock is invalid.") from exc
    if source_lock.run_id != run_id or canonical_json_bytes(source_lock.to_dict()) != raw:
        raise CloudProviderError("HF orphan SourceLock is not the exact canonical run lock.")
    return source_lock


def _load_exact_prepared_transport(*, tracking, experiment, context):
    from tuner.cloud.hf_provisioning import load_hf_source_transport

    expected_root = (
        tracking.base_dir / "experiments" / experiment.experiment_id
        / "cloud" / "hf" / "source-transport"
    ).resolve()
    expected_descriptor_uri = tracking.tracking_uri(expected_root / "descriptor.json")
    expected_source_lock_uri = tracking.tracking_uri(
        tracking.base_dir / "experiments" / experiment.experiment_id / "source-lock.json"
    )
    if (
        experiment.source_transport_uri != expected_descriptor_uri
        or experiment.source_lock_uri != expected_source_lock_uri
    ):
        raise CloudProviderError("HF PREPARED recovery references are not canonical.")
    prepared = load_hf_source_transport(
        context,
        transport_root=expected_root,
        descriptor_uri=expected_descriptor_uri,
        source_lock_uri=expected_source_lock_uri,
    )
    if prepared.descriptor_sha256 != experiment.source_transport_sha256:
        raise CloudProviderError("HF PREPARED recovery descriptor digest changed.")
    if prepared.descriptor["source_lock"]["sha256"] != experiment.source_lock_sha256:
        raise CloudProviderError("HF PREPARED recovery SourceLock digest changed.")
    return prepared


def _validate_prepared_volume_policy(
    descriptor: Mapping[str, object], settings: Mapping[str, object], *, run_id: str
) -> None:
    volume = descriptor.get("volume")
    bundle = descriptor.get("bundle")
    if not isinstance(volume, Mapping) or not isinstance(bundle, Mapping):
        raise CloudProviderError("HF PREPARED recovery volume binding is invalid.")
    source = settings.get("source")
    prefix = settings.get("path_prefix")
    digest = bundle.get("content_sha256")
    if not all(isinstance(value, str) and value for value in (source, prefix, digest)):
        raise CloudProviderError("HF PREPARED recovery volume policy is invalid.")
    expected_path = f"{prefix}/{run_id}/{digest}"
    if volume.get("source") != source or volume.get("path") != expected_path:
        raise CloudProviderError("HF PREPARED recovery volume differs from committed policy.")


def _read_bounded_regular_nofollow(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & 0x400
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise CloudProviderError("HF recovery artifact must be a bounded regular non-link file.")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except CloudProviderError:
        raise
    except OSError:
        raise CloudProviderError("HF recovery artifact could not be read safely.") from None
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode
    )
    raw = b"".join(chunks)
    if (
        len(raw) > maximum_bytes
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
    ):
        raise CloudProviderError("HF recovery artifact changed during authentication.")
    return raw


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        raise CloudProviderError("HF recovery artifact path could not be inspected.") from None


def _recover_succeeded_hf_provisioning(
    *,
    tracking: TrackingService,
    experiment,
    context,
    descriptor_path: Path,
):
    """Verify persisted SUCCEEDED evidence and finish local consumption only."""

    if experiment.hf_provisioning_state != "SUCCEEDED":
        raise CloudProviderError("HF provisioning recovery is not SUCCEEDED.")
    evidence_uri = str(experiment.provisioning_evidence_uri or "")
    evidence_sha256 = str(experiment.provisioning_evidence_sha256 or "")
    if not evidence_uri or not evidence_sha256:
        raise CloudProviderError("HF provisioning SUCCEEDED evidence is incomplete.")
    from tuner.cloud.hf_provisioning import (
        consume_hf_source_transport,
        load_canonical_json,
    )
    from tuner.cloud.hf_provisioning_operator import HFProvisioningOutcome

    evidence = load_canonical_json(
        tracking.resolve_uri(evidence_uri), maximum_bytes=64 * 1024
    )
    consumed = consume_hf_source_transport(
        context,
        transport_root=descriptor_path.parent,
        descriptor_uri=str(experiment.source_transport_uri),
        source_lock_uri=str(experiment.source_lock_uri),
        evidence=evidence,
    )
    if consumed.evidence_sha256 != evidence_sha256:
        raise CloudProviderError("HF provisioning recovered evidence digest changed.")
    if experiment.source_transport_state == "ACKNOWLEDGED":
        tracking.mark_source_transport_consumable(experiment)
    elif experiment.source_transport_state != "CONSUMABLE":
        raise CloudProviderError("HF provisioning recovery transport state is invalid.")
    return HFProvisioningOutcome(
        evidence=evidence,
        evidence_sha256=evidence_sha256,
        mutated=False,
    )


def _require_external_base_dir(value: object, *, context) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CloudProviderError("Protected HF routes require an explicit absolute --base-dir.")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise CloudProviderError("Protected HF --base-dir must be absolute.")
    path = Path(os.path.abspath(raw))
    for source in {context.engine_root.resolve(), context.project_root.resolve()}:
        if path == source or _is_relative_to(path, source) or _is_relative_to(source, path):
            raise CloudProviderError("Protected HF --base-dir must remain outside source trees.")
    _assert_existing_chain_link_free(path)
    path.mkdir(parents=True, exist_ok=True)
    _assert_existing_chain_link_free(path)
    return path.resolve(strict=True)


def _require_source_config(value: object, *, context) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CloudProviderError("hf-source prepare requires --source-config.")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = context.invocation_cwd / raw
    path = Path(os.path.abspath(raw))
    containing = [root for root in {context.engine_root.resolve(), context.project_root.resolve()} if _is_relative_to(path, root)]
    if not containing:
        raise CloudProviderError("HF source config must remain within a source tree.")
    _assert_regular_link_free(max(containing, key=lambda item: len(item.parts)), path)
    return path.resolve(strict=True)


def _load_committed_volume_settings(path: Path, *, context, source_lock) -> Mapping[str, object]:
    from tuner.cloud import bootstrap_core

    engine = context.engine_root.resolve()
    project = context.project_root.resolve()
    if _is_relative_to(path, engine):
        repository, commit, relative = engine, source_lock.engine_source.commit, path.relative_to(engine)
    elif _is_relative_to(path, project):
        repository, commit, relative = project, source_lock.project_source.commit, path.relative_to(project)
    else:
        raise CloudProviderError("HF source config must remain within an authenticated source tree.")
    try:
        import yaml
        result = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "blob", f"{commit}:{relative.as_posix()}"],
            check=True, capture_output=True, timeout=30,
            env=bootstrap_core.git_environment(),
        )
        if len(result.stdout) > 1024 * 1024:
            raise CloudProviderError("HF source config exceeds its committed-byte bound.")
        document = yaml.safe_load(result.stdout.decode("utf-8"))
    except (OSError, subprocess.SubprocessError, UnicodeError, CloudProviderError) as exc:
        raise CloudProviderError("HF source config could not be loaded from its exact commit.") from exc
    cloud = document.get("cloud") if isinstance(document, Mapping) else None
    hf_jobs = cloud.get("hf_jobs") if isinstance(cloud, Mapping) else None
    settings = hf_jobs.get("bootstrap_volume") if isinstance(hf_jobs, Mapping) else None
    if not isinstance(settings, Mapping):
        raise CloudProviderError("HF source config requires cloud.hf_jobs.bootstrap_volume.")
    return settings


def _require_recoverable_bootstrap_experiment(experiment) -> None:
    if (experiment.source_lock_uri is None) != (experiment.source_lock_sha256 is None):
        raise CloudProviderError("hf-source prepare found an incomplete SourceLock recovery state.")
    transport_pair = (experiment.source_transport_uri, experiment.source_transport_sha256)
    if experiment.source_transport_state is None and transport_pair != (None, None):
        raise CloudProviderError("hf-source prepare found an incomplete transport recovery state.")
    if experiment.source_transport_state == "PREPARED" and any(value is None for value in transport_pair):
        raise CloudProviderError("hf-source prepare found an incomplete PREPARED recovery state.")
    if experiment.source_transport_state not in {None, "PREPARED"}:
        raise CloudProviderError("hf-source prepare can recover only neutral or PREPARED state.")
    forbidden = (
        experiment.provisioning_evidence_uri, experiment.provisioning_evidence_sha256,
        experiment.hf_run_approval_uri, experiment.hf_submission_state,
    )
    if any(value is not None for value in forbidden):
        raise CloudProviderError("hf-source prepare cannot recover provider or submission state.")
    if experiment.provider not in {"", "hf_jobs"} or experiment.method not in {"", "bootstrap"}:
        raise CloudProviderError("hf-source prepare can reuse only a neutral bootstrap experiment.")


def _external_runtime_layout(context, base_dir: Path) -> CloudRuntimeLayout:
    workspace = PurePosixPath("/workspace")
    writable = tuple(
        RuntimeMount(name, (base_dir / "runtime" / name).resolve(), workspace / name, False)
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    )
    return CloudRuntimeLayout(
        engine=RuntimeMount("engine", context.engine_root.resolve(), workspace / "engine", True),
        project=RuntimeMount("project", context.project_root.resolve(), workspace / "project", True),
        writable=writable,
    )


def _assert_regular_link_free(root: Path, path: Path) -> None:
    current = root
    items = [root]
    for part in path.relative_to(root).parts:
        current = current / part
        items.append(current)
    for index, item in enumerate(items):
        try:
            info = item.lstat()
        except OSError:
            raise CloudProviderError("HF source config is unavailable.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CloudProviderError("HF source config cannot traverse links or reparse points.")
        final = index == len(items) - 1
        if final and not stat.S_ISREG(info.st_mode):
            raise CloudProviderError("HF source config must be a regular file.")
        if not final and not stat.S_ISDIR(info.st_mode):
            raise CloudProviderError("HF source config parent chain is invalid.")


def _assert_existing_chain_link_free(path: Path) -> None:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    chain = [existing]
    current = existing
    for part in path.relative_to(existing).parts:
        current = current / part
        if current.exists():
            chain.append(current)
    for item in chain:
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CloudProviderError("Protected HF --base-dir cannot traverse links or reparse points.")
        if not stat.S_ISDIR(info.st_mode):
            raise CloudProviderError("Protected HF --base-dir must identify a directory.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _persist_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError:
            raise CloudProviderError("HF JP evidence artifact is unavailable.") from None
        if existing != payload:
            raise CloudProviderError("HF JP evidence artifact already exists with different bytes.")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CloudProviderError("HF JP evidence artifact raced with different bytes.") from None
    except OSError:
        raise CloudProviderError("HF JP evidence artifact could not be persisted.") from None


# Compatibility aliases for tests/callers migrating to the shared helper.
_HFSecretFileClaim = HFSecretFileClaim
_preflight_secret_file = preflight_hf_secret_file
_read_claimed_hf_token = read_claimed_hf_token

__all__ = ["HFSourceHandler"]
