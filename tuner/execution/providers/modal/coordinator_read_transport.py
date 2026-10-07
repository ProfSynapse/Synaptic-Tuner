"""Authenticated provider reads for Foundation-native Modal runs."""
from __future__ import annotations

from collections.abc import Iterator

from tuner.execution.coordinator_v1.model import ProviderLogQueryV1, ProviderRunPhaseV1
from tuner.execution.foundation_v2.canonical import parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2, parse_exact_command
from tuner.training.recipes import RecipeRegistry

from .contracts import ArtifactMemberV1, BoundsPolicyV1, TerminalEvidenceV1, operation_path, provider_entry_identity, sha, strict_int
from .coordinator_binding import ModalCommandBinding
from .coordinator_bundle import ModalCoordinatorBundle
from .coordinator_launch import ModalLaunchEnvelope
from .coordinator_logs import ModalCoordinatorLogChunk, validate_modal_log_chain
from .coordinator_reader import ModalArtifactInventory, ModalLogSnapshot, ModalTerminalSnapshot
from .coordinator_submit_preparation import prepare_modal_submit_dispatch
from .control import CrossPlaneIdentityV1
from .facade import ExplicitModal154ReadFacade
from .manifest import CompletionManifestV1
from .resolution import VerifiedModalDeploymentIdentityV1


class ModalFoundationReadTransport:
    """Read only evidence for one completely authenticated retained launch."""

    def __init__(self, *, facade: ExplicitModal154ReadFacade,
                 deployment: VerifiedModalDeploymentIdentityV1, launch_source,
                 binding_authority, foundation_authenticator,
                 assessment_authenticator, stage_verifier, launch_verifier,
                 evidence_verifier, recipes: RecipeRegistry,
                 bounds: BoundsPolicyV1 = BoundsPolicyV1()) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact explicit Modal facade required")
        if type(deployment) is not VerifiedModalDeploymentIdentityV1:
            raise TypeError("exact verified Modal deployment required")
        if type(recipes) is not RecipeRegistry or type(bounds) is not BoundsPolicyV1:
            raise TypeError("exact Modal read configuration required")
        self._facade, self._deployment, self._launch_source = facade, deployment, launch_source
        self._authority, self._foundation = binding_authority, foundation_authenticator
        self._assessments, self._stage = assessment_authenticator, stage_verifier
        self._launch, self._evidence = launch_verifier, evidence_verifier
        self._recipes, self._bounds = recipes, bounds

    def _context(self, supplied: object, provider_job_ref: str):
        provider_job_ref = safe_ref(provider_job_ref, "provider_job_ref")
        if type(supplied) is not ModalCommandBinding:
            raise ValueError("exact Modal command binding required")
        binding = ModalCommandBinding(supplied.command_bytes, supplied.preparation_snapshot,
                                      supplied.deployment_bytes)
        command = parse_exact_command(binding.command_bytes)
        if (binding != supplied or type(command) is not SubmitCommandV2
                or self._authority.authenticate(binding) is not True
                or binding.deployment != self._deployment
                or binding.client_binding != self._facade.binding):
            raise ValueError("Modal read binding mismatch")
        envelope = self._launch_source.resolve(command.digest)
        if type(envelope) is not ModalLaunchEnvelope or envelope.submit_binding != binding:
            raise ValueError("retained Modal launch mismatch")
        prepare_modal_submit_dispatch(
            envelope, foundation_authenticator=self._foundation,
            assessment_authenticator=self._assessments,
            binding_authority=self._authority, stage_verifier=self._stage,
            launch_verifier=self._launch, recipes=self._recipes, bounds=self._bounds,
        )
        stage_binding = envelope.stage_material.binding
        if type(parse_exact_command(stage_binding.command_bytes)) is not StageCommandV2:
            raise ValueError("retained Modal stage binding invalid")
        bundle = ModalCoordinatorBundle.parse_transport(
            envelope.stage_material.bundle, binding=stage_binding, recipes=self._recipes,
        )
        policy_member = next(x for x in bundle.members if x.name == "log-terminal-policy.json")
        policy = parse_canonical_object(policy_member.content, name="log terminal policy")
        material = envelope.stage_material
        profile = parse_canonical_object(binding.preparation_snapshot, name="preparation snapshot")["configuration"]["profile"]
        if (self._facade.volume_name(material.control_volume_id) != profile["volumes"]["control_ref"]
                or self._facade.volume_name(material.artifact_volume_id) != profile["volumes"]["artifact_ref"]):
            raise ValueError("Modal read volume mismatch")
        self._facade.bound_scope()
        selection = self._deployment.selection
        if self._facade.inspect_deployment(app_name=selection.app_name,
                                           function_name=selection.function_name) != selection:
            raise ValueError("Modal read deployment mismatch")
        return binding, command, material, policy, provider_job_ref

    def _verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> None:
        try:
            valid = self._evidence.verify(purpose, payload, tag, key_ref)
        except Exception:
            raise ValueError("Modal read evidence authentication unavailable") from None
        if valid is not True:
            raise ValueError("Modal read evidence authentication failed")

    def _identity(self, binding, command, material, job_ref, generation):
        return CrossPlaneIdentityV1(
            binding.client_binding, material.control_volume_id, material.artifact_volume_id,
            job_ref, command.operation.effect.effect_id, command.digest,
            command.preparation.plan_fingerprint, binding.deployment.attestation_digest,
            command.operation.invocation_nonce, generation, material.key_ref,
        )

    def _evidence_listing(self, material, effect_id):
        prefix = operation_path(effect_id, "evidence") + "/"
        listing = self._facade.list_prefix(material.control_volume_id, prefix, max_entries=5)
        allowed = {
            prefix + "terminal-evidence.v1.json", prefix + "terminal-evidence.v1.mac",
            prefix + "completion-manifest.v1.json", prefix + "completion-manifest.v1.mac",
        }
        if len({item[0] for item in listing}) != len(listing) or any(
                item[0] not in allowed for item in listing):
            raise ValueError("Modal evidence inventory is invalid")
        return listing

    def _terminal(self, binding, command, material, policy, job_ref, *, allow_absent=False):
        effect = command.operation.effect.effect_id
        root = operation_path(effect, "evidence")
        data_path, tag_path = root + "/terminal-evidence.v1.json", root + "/terminal-evidence.v1.mac"
        listing = self._evidence_listing(material, effect)
        paths = {x[0] for x in listing}
        if data_path not in paths and tag_path not in paths:
            if any(path.endswith("completion-manifest.v1.json") or
                   path.endswith("completion-manifest.v1.mac") for path in paths):
                raise ValueError("Modal completion exists without terminal evidence")
            if allow_absent:
                return None
            raise ValueError("Modal terminal evidence unavailable")
        if (data_path in paths) != (tag_path in paths):
            raise ValueError("Modal terminal evidence is incomplete")
        maximum = min(self._bounds.max_control_bytes, strict_int(policy["max_terminal_bytes"], "max_terminal_bytes", minimum=1))
        data = self._facade.read_complete(material.control_volume_id, data_path, max_bytes=maximum)
        tag = self._facade.read_complete(material.control_volume_id, tag_path, max_bytes=128)
        self._verify("modal-terminal/v1", data, tag, material.key_ref)
        terminal = TerminalEvidenceV1.parse(data, limit=maximum)
        identity = self._identity(binding, command, material, job_ref, policy["generation"])
        if not identity.matches_record(terminal):
            raise ValueError("Modal terminal identity mismatch")
        return terminal

    def observe(self, binding: ModalCommandBinding, *, provider_job_ref: str) -> ModalTerminalSnapshot:
        binding, command, material, policy, job_ref = self._context(binding, provider_job_ref)
        terminal = self._terminal(binding, command, material, policy, job_ref, allow_absent=True)
        identity = self._identity(binding, command, material, job_ref, policy["generation"])
        if terminal is None:
            # A provider timeout proves only that the exact call has not
            # returned; it cannot authenticate either QUEUED or RUNNING.
            raise ValueError("Modal authenticated run phase is unavailable")
        phase = {"completed": ProviderRunPhaseV1.SUCCEEDED,
                 "failed": ProviderRunPhaseV1.FAILED,
                 "cancelled": ProviderRunPhaseV1.CANCELLED}[terminal.status_code]
        return ModalTerminalSnapshot.build(phase, identity, terminal)

    def _logs(self, binding, command, material, policy, job_ref):
        effect, volume = command.operation.effect.effect_id, material.control_volume_id
        root = operation_path(effect, "logs")
        meta_path, tag_path = root + "/log-metadata.v1.json", root + "/log-metadata.v1.mac"
        metadata = self._facade.read_complete(volume, meta_path, max_bytes=self._bounds.max_control_bytes)
        tag = self._facade.read_complete(volume, tag_path, max_bytes=128)
        self._verify("modal-log-metadata/v1", metadata, tag, material.key_ref)
        value = parse_canonical_object(metadata, name="Modal log metadata")
        expected_keys = {"schema","account_ref","workspace_ref","environment_ref","client_ref","sdk_version","control_volume_id","artifact_volume_id","job_ref","effect_id","command_digest","plan_digest","deployment_attestation_digest","invocation_nonce","generation","chain_digest","chunks"}
        if set(value) != expected_keys or value["schema"] != "synaptic.modal-log-metadata/v1":
            raise ValueError("Modal log metadata invalid")
        strict_int(value["generation"], "generation", minimum=1, maximum=2**31 - 1)
        identity = self._identity(binding, command, material, job_ref, policy["generation"])
        expected_identity = {
            "account_ref": identity.binding.account_ref,
            "workspace_ref": identity.binding.workspace_ref,
            "environment_ref": identity.binding.environment_ref,
            "client_ref": identity.binding.client_ref,
            "sdk_version": identity.binding.sdk_version,
            **{name: getattr(identity, name) for name in (
                "control_volume_id", "artifact_volume_id", "job_ref", "effect_id",
                "command_digest", "plan_digest", "deployment_attestation_digest",
                "invocation_nonce", "generation",
            )},
        }
        if any(value[k] != expected for k, expected in expected_identity.items()):
            raise ValueError("Modal log metadata identity mismatch")
        declared = value["chunks"]
        maximum_chunks = min(self._bounds.max_log_records, strict_int(policy["max_log_chunks"], "max_log_chunks", minimum=1))
        maximum_chunk = min(self._bounds.max_log_chunk_bytes, strict_int(policy["max_chunk_bytes"], "max_chunk_bytes", minimum=1))
        if type(declared) is not list or not 1 <= len(declared) <= maximum_chunks:
            raise ValueError("Modal log inventory invalid")
        prefix = root + "/chunks/"
        listing = self._facade.list_prefix(volume, prefix, max_entries=maximum_chunks + 1)
        actual = {p:(s,i) for p,s,i in listing}
        if len(actual) != len(listing) or len(actual) != len(declared):
            raise ValueError("Modal log relist mismatch")
        chunks = []
        declared_paths = set()
        record_count = 0
        for index, item in enumerate(declared):
            if type(item) is not dict or set(item) != {"path","size","sha256","provider_entry_id"}:
                raise ValueError("Modal log inventory member invalid")
            path, size = item["path"], strict_int(item["size"], "log size", minimum=1, maximum=maximum_chunk)
            expected_entry = provider_entry_identity(volume, path, size)
            if (path in declared_paths or path != prefix + f"{index:03d}.json"
                    or item["provider_entry_id"] != expected_entry
                    or actual.get(path) != (size, expected_entry)):
                raise ValueError("Modal log inventory differs from relist")
            declared_paths.add(path)
            raw = self._facade.read_complete(volume, path, max_bytes=maximum_chunk)
            if len(raw) != size or sha(raw) != item["sha256"]:
                raise ValueError("Modal log chunk differs from inventory")
            chunk = ModalCoordinatorLogChunk.parse(raw, bounds=self._bounds)
            record_count += len(chunk.records)
            if record_count > self._bounds.max_log_records:
                raise ValueError("Modal log records exceed aggregate bound")
            chunks.append(chunk)
        chain = validate_modal_log_chain(tuple(chunks), bounds=self._bounds)
        if chain != value["chain_digest"]:
            raise ValueError("Modal log chain mismatch")
        first = chunks[0]
        if (first.generation, first.job_ref, first.effect_id, first.plan_digest,
                first.invocation_nonce) != (
                identity.generation, identity.job_ref, identity.effect_id,
                identity.plan_digest, identity.invocation_nonce):
            raise ValueError("Modal log chain identity mismatch")
        return identity, tuple(chunks), chain

    def logs(self, binding: ModalCommandBinding, query: ProviderLogQueryV1, *, provider_job_ref: str) -> ModalLogSnapshot:
        if type(query) is not ProviderLogQueryV1:
            raise TypeError("exact provider log query required")
        binding, command, material, policy, job_ref = self._context(binding, provider_job_ref)
        identity, chunks, chain = self._logs(binding, command, material, policy, job_ref)
        all_entries = tuple(entry for chunk in chunks for entry in chunk.records)
        after = -1 if query.after_sequence is None else query.after_sequence
        candidates = tuple(x for x in all_entries if x.sequence > after)
        selected, used = [], 0
        for entry in candidates:
            if len(selected) >= query.limit or used + entry.size_bytes > query.maximum_bytes:
                break
            selected.append(entry); used += entry.size_bytes
        high = all_entries[-1].sequence if all_entries else -1
        terminal = self._terminal(binding, command, material, policy, job_ref, allow_absent=True)
        if terminal is not None and terminal.log_chain_digest != chain:
            raise ValueError("Modal terminal log chain mismatch")
        phase = None if terminal is None else {"completed":ProviderRunPhaseV1.SUCCEEDED,"failed":ProviderRunPhaseV1.FAILED,"cancelled":ProviderRunPhaseV1.CANCELLED}[terminal.status_code]
        return ModalLogSnapshot.build(tuple(selected), identity, chain, query.log_query_digest,
                                      identity.generation, high, phase,
                                      identity.generation if phase else None, len(selected) < len(candidates))

    def artifact_inventory(self, binding: ModalCommandBinding, *, provider_job_ref: str) -> ModalArtifactInventory:
        binding, command, material, policy, job_ref = self._context(binding, provider_job_ref)
        terminal = self._terminal(binding, command, material, policy, job_ref)
        if terminal.status_code != "completed":
            raise ValueError("Modal artifacts require completed terminal evidence")
        identity, _, chain = self._logs(binding, command, material, policy, job_ref)
        root = operation_path(command.operation.effect.effect_id, "evidence")
        data_path, tag_path = root + "/completion-manifest.v1.json", root + "/completion-manifest.v1.mac"
        data = self._facade.read_complete(material.control_volume_id, data_path, max_bytes=self._bounds.max_control_bytes)
        tag = self._facade.read_complete(material.control_volume_id, tag_path, max_bytes=128)
        self._verify("modal-completion/v1", data, tag, material.key_ref)
        manifest = CompletionManifestV1.parse(data, limit=self._bounds.max_control_bytes)
        if (any(type(item) is not ArtifactMemberV1 or
                item.size > self._bounds.max_artifact_bytes for item in manifest.members)
                or sum(item.size for item in manifest.members) > self._bounds.max_artifact_total_bytes):
            raise ValueError("Modal artifact inventory exceeds bounds")
        terminal_bytes = self._facade.read_complete(
            material.control_volume_id, root + "/terminal-evidence.v1.json",
            max_bytes=min(self._bounds.max_control_bytes, policy["max_terminal_bytes"]),
        )
        terminal_tag = self._facade.read_complete(
            material.control_volume_id, root + "/terminal-evidence.v1.mac", max_bytes=128,
        )
        self._verify("modal-terminal/v1", terminal_bytes, terminal_tag, material.key_ref)
        if (TerminalEvidenceV1.parse(terminal_bytes) != terminal
                or not identity.matches_record(manifest)
                or manifest.terminal_evidence_digest != sha(terminal_bytes)
                or manifest.log_chain_digest != chain or terminal.log_chain_digest != chain
                or terminal.artifact_set_digest != manifest.artifact_set_digest):
            raise ValueError("Modal completion evidence mismatch")
        prefix = operation_path(identity.effect_id, "output") + "/"
        listing = self._facade.list_prefix(material.artifact_volume_id, prefix, max_entries=6)
        expected = {x.path:(x.size,x.provider_entry_id) for x in manifest.members}
        if len(listing) != 5 or {p:(s,i) for p,s,i in listing} != expected:
            raise ValueError("Modal artifact inventory relist mismatch")
        return ModalArtifactInventory.build(manifest, identity)

    def iter_artifact(self, binding: ModalCommandBinding, member, *, provider_job_ref: str,
                      maximum_bytes: int) -> Iterator[bytes]:
        if (type(member) is not ArtifactMemberV1 or type(maximum_bytes) is not int
                or not member.size <= maximum_bytes <= self._bounds.max_artifact_bytes):
            raise ValueError("Modal artifact stream request invalid")
        inventory = self.artifact_inventory(binding, provider_job_ref=provider_job_ref)
        matches = tuple(x for x in inventory.manifest.members if x == member)
        if len(matches) != 1:
            raise ValueError("Modal artifact stream request invalid")
        volume = inventory.identity.artifact_volume_id
        prefix = operation_path(inventory.identity.effect_id, "output") + "/"
        before = self._facade.list_prefix(volume, prefix, max_entries=6)
        expected = tuple(sorted(
            (item.path, item.size, item.provider_entry_id)
            for item in inventory.manifest.members
        ))
        if tuple(sorted(before)) != expected:
            raise ValueError("Modal artifact changed before stream")
        total = 0
        digest = __import__("hashlib").sha256()
        for chunk in self._facade.iter_complete(volume, member.path, max_bytes=maximum_bytes):
            if type(chunk) is not bytes or not chunk:
                raise ValueError("Modal artifact stream invalid")
            total += len(chunk); digest.update(chunk)
            if total > member.size:
                raise ValueError("Modal artifact exceeds manifest")
            yield chunk
        if total != member.size or digest.hexdigest() != member.sha256:
            raise ValueError("Modal artifact stream incomplete")
        if tuple(sorted(self._facade.list_prefix(volume, prefix, max_entries=6))) != expected:
            raise ValueError("Modal artifact changed during stream")


__all__: list[str] = []
