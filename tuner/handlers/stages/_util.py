"""Shared utilities for experiment stage runners.

Located at tuner/handlers/stages/_util.py.
Provides helper functions used across multiple stage runner modules.
"""

from __future__ import annotations

import hashlib
import os
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from tuner.cloud import bootstrap_core
from tuner.cloud.checkout import (
    build_source_lock,
    checkout_policy_from_context,
    validate_source_lock_for_cloud,
)
from tuner.cloud.hf_volume_transport import (
    HFVerifiedVolume,
    HFVerifiedVolumeSpec,
    build_runtime_projection_step,
    build_verified_bootstrap_step,
    prove_read_only_volume,
    transport_metadata,
)
from tuner.cloud.hf_provisioning import (
    HFConsumableSourceTransport,
    consume_hf_source_transport,
    load_canonical_json,
    prepare_hf_source_transport,
    validate_hf_bootstrap_volume_config,
)
from tuner.cloud.runtime_layout import CloudRuntimeLayout, build_runtime_layout
from tuner.core.exceptions import CloudProviderError
from tuner.project import ProjectContext
from tuner.project.source_bundle import SourceLock


def _optional_backend_value(value) -> str | None:
    """Return a backend metadata value only when it is a real non-empty string."""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


@dataclass(frozen=True)
class HFSourcePreparation:
    source_lock: SourceLock
    source_lock_sha256: str
    source_lock_uri: str
    volume_spec: HFVerifiedVolumeSpec | None
    runtime_layout: CloudRuntimeLayout
    staging_root: Path | None = None
    descriptor_uri: str | None = None
    descriptor_sha256: str | None = None
    provisioning_evidence_uri: str | None = None
    provisioning_evidence_sha256: str | None = None
    source_transport_state: str = "PREPARED"
    verification_context: ProjectContext | None = None
    consumable_transport: HFConsumableSourceTransport | None = None

    def prove_volume(self, huggingface_hub) -> HFVerifiedVolume:
        self.require_consumable()
        assert self.volume_spec is not None
        proven = prove_read_only_volume(huggingface_hub, self.volume_spec)
        return HFVerifiedVolume(
            spec=proven.spec,
            provider_volume=proven.provider_volume,
            descriptor_sha256=self.descriptor_sha256,
            provisioning_evidence_sha256=self.provisioning_evidence_sha256,
            descriptor_uri=self.descriptor_uri,
            source_lock_uri=self.source_lock_uri,
            transport_root=self.staging_root,
            provisioning_evidence=(
                self.consumable_transport.evidence if self.consumable_transport is not None else None
            ),
            verification_context=self.verification_context,
        )

    def require_consumable(self) -> None:
        if (
            self.source_transport_state != "CONSUMABLE"
            or self.volume_spec is None
            or not self.descriptor_uri
            or not self.descriptor_sha256
            or not self.provisioning_evidence_uri
            or not self.provisioning_evidence_sha256
        ):
            raise CloudProviderError(
                "HF source transport is not externally acknowledged and verified as CONSUMABLE."
            )

    @property
    def remote_project_root(self) -> str:
        return "/workspace/project"

    @property
    def remote_engine_root(self) -> str:
        return "/workspace/engine"

    @property
    def physical_project_root(self) -> str:
        return "/workspace/source/project"

    @property
    def physical_engine_root(self) -> str:
        if self.source_lock.mode == "standalone":
            return self.physical_project_root
        if self.source_lock.mode == "superproject":
            return f"{self.physical_project_root}/{self.source_lock.engine_source.submodule_path}"
        return "/workspace/source/engine"

    @property
    def metadata(self) -> dict[str, object]:
        value = transport_metadata(self.volume_spec) if self.volume_spec is not None else {
            "profile": "hf_read_only_volume",
            "read_only": True,
            "prepared_only": True,
        }
        value.update(
            source_lock_uri=self.source_lock_uri,
            source_lock_mode=self.source_lock.mode,
            source_lock_run_id=self.source_lock.run_id,
            descriptor_uri=self.descriptor_uri,
            descriptor_sha256=self.descriptor_sha256,
            provisioning_evidence_uri=self.provisioning_evidence_uri,
            provisioning_evidence_sha256=self.provisioning_evidence_sha256,
            source_transport_state=self.source_transport_state,
            runtime_layout={
                "schema_version": self.runtime_layout.schema_version,
                "writable_targets": {
                    name: mount.target.as_posix()
                    for name, mount in self.runtime_layout.writable_by_name.items()
                },
            },
        )
        return value


def _bootstrap_volume_policy(volume_settings: Mapping[str, object]) -> tuple[str, str]:
    return validate_hf_bootstrap_volume_config(volume_settings)


def _descriptor_uri_for_source_lock(source_lock_uri: str, run_id: str) -> str:
    if not source_lock_uri.endswith("/source-lock.json"):
        raise CloudProviderError("Canonical SourceLock URI must identify source-lock.json.")
    parent = source_lock_uri.rsplit("/", 1)[0]
    if source_lock_uri.startswith("tracking://experiments/"):
        return f"{parent}/cloud/hf/source-transport/descriptor.json"
    return f"{parent}/source-transport/descriptor.json"


def hf_source_preparation_from_consumable(
    consumed: HFConsumableSourceTransport,
    *,
    context: ProjectContext | None = None,
    runtime_layout: CloudRuntimeLayout,
    provisioning_evidence_uri: str,
) -> HFSourcePreparation:
    """Adapt the pure provisioning result to existing HF workload seams."""

    prepared = consumed.prepared
    source_lock = prepared.source_lock
    return HFSourcePreparation(
        source_lock=source_lock,
        source_lock_sha256=str(prepared.descriptor["source_lock"]["sha256"]),
        source_lock_uri=str(prepared.descriptor["source_lock"]["uri"]),
        volume_spec=consumed.volume_spec,
        runtime_layout=runtime_layout,
        staging_root=prepared.root,
        descriptor_uri=prepared.descriptor_uri,
        descriptor_sha256=prepared.descriptor_sha256,
        provisioning_evidence_uri=provisioning_evidence_uri,
        provisioning_evidence_sha256=consumed.evidence_sha256,
        source_transport_state="CONSUMABLE",
        verification_context=context or prepared.verification_context,
        consumable_transport=consumed,
    )


def preflight_hf_source_lock(
    context: ProjectContext,
    *,
    run_id: str,
    source_mode: str | None = None,
) -> SourceLock:
    """Perform the authorized read-only Git proof before local PREPARED state.

    Unlike descriptor preparation, this operation may contact declared Git
    remotes to prove that the exact commits are pushed.  It performs no HF
    provider, volume, bucket, authentication, or job operation.
    """

    source_lock = build_source_lock(
        context,
        run_id=run_id,
        mode=source_mode,
        environment=os.environ,
    )
    validate_source_lock_for_cloud(source_lock)
    return source_lock


def _source_uri(context: ProjectContext, path: Path) -> str:
    resolved = path.resolve(strict=True)
    # Prefer engine:// when an in-tree submodule path is contained by both roots;
    # the superproject tree stores only the gitlink, not files below it.
    for scheme, root in (("engine", context.engine_root), ("project", context.project_root)):
        try:
            relative = resolved.relative_to(root.resolve(strict=True))
        except ValueError:
            continue
        return f"{scheme}://{relative.as_posix()}"
    raise CloudProviderError("HF source identity must be committed below project or engine root.")


def _file_identity(context: ProjectContext, path: Path, *, key: str = "uri") -> dict[str, str]:
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise CloudProviderError("HF source identity must be a regular non-link file.")
    content = path.resolve(strict=True).read_bytes()
    return {key: _source_uri(context, path), "sha256": hashlib.sha256(content).hexdigest()}


def _verify_committed_identities(context: ProjectContext, source_lock: SourceLock) -> None:
    identities: list[tuple[str, str]] = []
    identities.append((str(source_lock.project["manifest_uri"]), str(source_lock.project["manifest_sha256"])))
    for item in source_lock.configuration["documents"]:
        identities.append((str(item["uri"]), str(item["sha256"])))
    identities.extend((str(item["source"]), str(item["sha256"])) for item in source_lock.plugins)
    identities.extend((str(item["uri"]), str(item["sha256"])) for item in source_lock.inputs)
    for uri, expected in identities:
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise CloudProviderError("HF source identity digest must be lowercase SHA-256.")
        if uri.startswith("engine://"):
            repository = context.engine_root
            commit = source_lock.engine_source.commit
            relative = uri.removeprefix("engine://")
        elif uri.startswith("project://"):
            repository = context.project_root
            commit = source_lock.project_source.commit
            relative = uri.removeprefix("project://")
        else:
            raise CloudProviderError("HF source identity uses an unsupported URI scheme.")
        try:
            committed_oid = bootstrap_core.run_git(
                ["rev-parse", "--verify", f"{commit}:{relative}"],
                cwd=repository,
                env=bootstrap_core.git_environment(),
            )
            object_type = bootstrap_core.run_git(
                ["cat-file", "-t", committed_oid], cwd=repository,
                env=bootstrap_core.git_environment(),
            )
            working_oid = bootstrap_core.run_git(
                ["hash-object", "--", relative], cwd=repository,
                env=bootstrap_core.git_environment(),
            )
        except bootstrap_core.BootstrapError as exc:
            raise CloudProviderError("HF source identity is not a committed regular member.") from exc
        if object_type != "blob" or working_oid != committed_oid:
            raise CloudProviderError("HF source identity does not match its exact committed revision.")


def prepare_hf_source(
    context: ProjectContext,
    *,
    run_id: str,
    config_path: Path,
    volume_settings: Mapping[str, object],
    source_mode: str | None = None,
    plugins: Iterable[Path] = (),
    inputs: Iterable[Path] = (),
    runtime: Mapping[str, object] | None = None,
    outputs: Mapping[str, object] | None = None,
    source_lock: SourceLock,
    runtime_layout: CloudRuntimeLayout | None = None,
    checkout_policy=None,
    source_lock_uri: str | None = None,
    descriptor_uri: str | None = None,
    transport_root: Path | None = None,
) -> HFSourcePreparation:
    """Prepare one immutable local descriptor without provisioning or submission."""

    source, path_prefix = _bootstrap_volume_policy(volume_settings)
    if not isinstance(source_lock, SourceLock):
        raise CloudProviderError(
            "HF PREPARED transport requires an explicit SourceLock from source preflight."
        )
    if source_lock.run_id != run_id:
        raise CloudProviderError("HF source preparation run_id does not match the accepted SourceLock.")
    validate_source_lock_for_cloud(source_lock)
    runtime_layout = runtime_layout or build_runtime_layout(context)
    config_identity = _file_identity(context, config_path)
    if context.manifest_path is not None:
        manifest_identity = _file_identity(context, context.manifest_path)
        engine_requires = source_lock.project.get("engine_requires") or "*"
    else:
        # Standalone has no host manifest. Bind the explicit primary config as
        # its deterministic project contract instead of fabricating a hidden file.
        manifest_identity = dict(config_identity)
        engine_requires = "*"
    plugin_identities = tuple(
        {**_file_identity(context, path, key="source"), "name": path.stem}
        for path in plugins
    )
    input_identities = tuple(
        {**_file_identity(context, path), "name": path.name, "transport": "git"}
        for path in inputs
    )
    source_lock = replace(
        source_lock,
        project={
            **dict(source_lock.project),
            "manifest_uri": manifest_identity["uri"],
            "manifest_sha256": manifest_identity["sha256"],
            "engine_requires": engine_requires,
        },
        configuration={
            "resolved_uri": config_identity["uri"],
            "resolved_sha256": config_identity["sha256"],
            "documents": [config_identity],
        },
        plugins=plugin_identities,
        inputs=input_identities,
        runtime=dict(runtime or {}),
        outputs=dict(outputs or {}),
    )
    _verify_committed_identities(context, source_lock)
    policy = checkout_policy or checkout_policy_from_context(context, source_lock=source_lock)
    staging_root = Path(transport_root).resolve() if transport_root is not None else (
        runtime_layout.writable_by_name["tracking"].source
        / "cloud" / "hf" / run_id / "source-transport"
    ).resolve()
    if source_lock_uri is None:
        canonical_descriptor_uri = descriptor_uri or (
            f"tracking://cloud/hf/{run_id}/source-transport/descriptor.json"
        )
        canonical_source_lock_uri = (
            f"tracking://cloud/hf/{run_id}/source-transport/bundle/source-lock.json"
        )
    else:
        canonical_source_lock_uri = source_lock_uri
        canonical_descriptor_uri = descriptor_uri or _descriptor_uri_for_source_lock(
            canonical_source_lock_uri, run_id
        )
    prepared = prepare_hf_source_transport(
        context,
        source_lock=source_lock,
        source_lock_uri=canonical_source_lock_uri,
        descriptor_uri=canonical_descriptor_uri,
        transport_root=staging_root,
        volume_source=source,
        path_prefix=path_prefix,
        checkout_policy=policy,
    )
    return HFSourcePreparation(
        source_lock=source_lock,
        source_lock_sha256=str(prepared.descriptor["source_lock"]["sha256"]),
        source_lock_uri=canonical_source_lock_uri,
        volume_spec=None,
        runtime_layout=runtime_layout,
        staging_root=staging_root,
        descriptor_uri=canonical_descriptor_uri,
        descriptor_sha256=prepared.descriptor_sha256,
        source_transport_state="PREPARED",
    )


def load_hf_source_preparation(
    context: ProjectContext,
    *,
    run_id: str,
    source_lock_uri: str,
    source_lock_sha256: str,
    volume_settings: Mapping[str, object],
    runtime_layout: CloudRuntimeLayout | None = None,
    descriptor_uri: str | None = None,
    descriptor_sha256: str | None = None,
    provisioning_evidence_uri: str | None = None,
    provisioning_evidence: Mapping[str, object] | None = None,
    provisioning_evidence_sha256: str | None = None,
    transport_root: Path | None = None,
) -> HFSourcePreparation:
    """Rehydrate only an exact descriptor/evidence pair verified CONSUMABLE."""

    _bootstrap_volume_policy(volume_settings)
    runtime_layout = runtime_layout or build_runtime_layout(context)
    staging_root = Path(transport_root).resolve() if transport_root is not None else (
        runtime_layout.writable_by_name["tracking"].source
        / "cloud" / "hf" / run_id / "source-transport"
    ).resolve()
    if (
        descriptor_uri is None
        or descriptor_sha256 is None
        or provisioning_evidence_uri is None
        or provisioning_evidence is None
        or provisioning_evidence_sha256 is None
    ):
        raise CloudProviderError(
            "HF source transport requires exact descriptor and external provisioning-evidence references."
        )
    consumed = consume_hf_source_transport(
        context,
        transport_root=staging_root,
        descriptor_uri=descriptor_uri,
        source_lock_uri=source_lock_uri,
        evidence=provisioning_evidence,
    )
    if consumed.prepared.descriptor_sha256 != descriptor_sha256:
        raise CloudProviderError("Persisted HF descriptor digest does not match tracking state.")
    if consumed.evidence_sha256 != provisioning_evidence_sha256:
        raise CloudProviderError("Persisted HF provisioning evidence digest does not match tracking state.")
    if str(consumed.prepared.descriptor["source_lock"]["sha256"]) != source_lock_sha256:
        raise CloudProviderError("Persisted HF SourceLock digest does not match tracking state.")
    return hf_source_preparation_from_consumable(
        consumed,
        context=context,
        runtime_layout=runtime_layout,
        provisioning_evidence_uri=provisioning_evidence_uri,
    )


def load_tracked_hf_source_preparation(
    context: ProjectContext,
    *,
    tracking_service,
    experiment,
    volume_settings: Mapping[str, object],
) -> HFSourcePreparation:
    """Validate an ACKNOWLEDGED/CONSUMABLE experiment transport locally."""

    required = (
        experiment.source_lock_uri,
        experiment.source_lock_sha256,
        experiment.source_transport_uri,
        experiment.source_transport_sha256,
        experiment.provisioning_evidence_uri,
        experiment.provisioning_evidence_sha256,
    )
    if any(not isinstance(value, str) or not value for value in required):
        raise CloudProviderError("HF tracked source transport references are incomplete.")
    if experiment.source_transport_state not in {"ACKNOWLEDGED", "CONSUMABLE"}:
        raise CloudProviderError(
            "HF source transport must be externally ACKNOWLEDGED before local consumption."
        )
    descriptor_path = tracking_service.resolve_uri(experiment.source_transport_uri)
    evidence_path = tracking_service.resolve_uri(experiment.provisioning_evidence_uri)
    preparation = load_hf_source_preparation(
        context,
        run_id=experiment.experiment_id,
        source_lock_uri=experiment.source_lock_uri,
        source_lock_sha256=experiment.source_lock_sha256,
        volume_settings=volume_settings,
        descriptor_uri=experiment.source_transport_uri,
        descriptor_sha256=experiment.source_transport_sha256,
        provisioning_evidence_uri=experiment.provisioning_evidence_uri,
        provisioning_evidence=load_canonical_json(evidence_path, maximum_bytes=64 * 1024),
        provisioning_evidence_sha256=experiment.provisioning_evidence_sha256,
        transport_root=descriptor_path.parent,
    )
    if experiment.source_transport_state == "ACKNOWLEDGED":
        tracking_service.mark_source_transport_consumable(experiment)
    tracking_service.require_consumable_hf_transport(experiment)
    return preparation


def hf_verified_source_steps(preparation: HFSourcePreparation) -> list[str]:
    """Compile verification/runtime ordering shared by every HF workload."""

    preparation.require_consumable()
    assert preparation.volume_spec is not None
    physical_project_root = preparation.physical_project_root
    physical_engine_root = preparation.physical_engine_root
    quote = shlex.quote
    identity = " ".join(
        [
            f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={quote(physical_engine_root)}",
            "$(command -v python3 || command -v python)",
            "-m tuner.cloud.hf_volume_transport _verify-identities",
            f"--source-lock {quote(preparation.volume_spec.mounted(preparation.volume_spec.source_lock_path))}",
            f"--project-root {quote(physical_project_root)}",
            f"--engine-root {quote(physical_engine_root)}",
        ]
    )
    writable_targets = preparation.runtime_layout.writable_by_name
    writable = " ".join(mount.target.as_posix() for mount in writable_targets.values())
    return [
        build_verified_bootstrap_step(preparation.volume_spec),
        identity,
        build_runtime_projection_step(
            expected_project_root=physical_project_root,
            expected_engine_root=physical_engine_root,
            expected_project_commit=preparation.source_lock.project_source.commit,
            expected_engine_commit=preparation.source_lock.engine_source.commit,
            expected_mode=preparation.source_lock.mode,
        ),
        f"mkdir -p {writable}",
        f"cp -- {quote(preparation.volume_spec.mounted(preparation.volume_spec.source_lock_path))} {quote(writable_targets['artifacts'].target.as_posix() + '/source-lock.json')}",
        f"chmod a-w {quote(writable_targets['artifacts'].target.as_posix() + '/source-lock.json')}",
        f"export SYNAPTIC_PROJECT_ROOT={quote(preparation.remote_project_root)}",
        f"export SYNAPTIC_ENGINE_ROOT={quote(preparation.remote_engine_root)}",
        f"export SYNAPTIC_ARTIFACT_ROOT={quote(writable_targets['artifacts'].target.as_posix())}",
        f"export SYNAPTIC_STATE_ROOT={quote(writable_targets['state'].target.as_posix())}",
        f"export SYNAPTIC_TRACKING_ROOT={quote(writable_targets['tracking'].target.as_posix())}",
        f"export SYNAPTIC_CACHE_ROOT={quote(writable_targets['cache'].target.as_posix())}",
        f"export SYNAPTIC_TMP_ROOT={quote(writable_targets['tmp'].target.as_posix())}",
        f"export SYNAPTIC_SOURCE_LOCK_PATH={quote(preparation.volume_spec.mounted(preparation.volume_spec.source_lock_path))}",
        f"export SYNAPTIC_SOURCE_LOCK_URI={quote(preparation.source_lock_uri)}",
        f"export SYNAPTIC_SOURCE_LOCK_SHA256={quote(preparation.source_lock_sha256)}",
        f"export SYNAPTIC_SOURCE_LOCK_ID={quote(preparation.source_lock_sha256)}",
        f"export SYNAPTIC_SOURCE_TRANSPORT_URI={quote(preparation.descriptor_uri or '')}",
        f"export SYNAPTIC_SOURCE_TRANSPORT_SHA256={quote(preparation.descriptor_sha256 or '')}",
        f"export SYNAPTIC_PROVISIONING_EVIDENCE_URI={quote(preparation.provisioning_evidence_uri or '')}",
        f"export SYNAPTIC_PROVISIONING_EVIDENCE_SHA256={quote(preparation.provisioning_evidence_sha256 or '')}",
        "export SYNAPTIC_SOURCE_TRANSPORT_STATE=CONSUMABLE",
        "export PYTHONDONTWRITEBYTECODE=1",
    ]
