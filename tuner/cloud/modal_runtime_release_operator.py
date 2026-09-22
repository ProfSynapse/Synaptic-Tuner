"""Durable protected operator for one packaged Modal runtime release.

Release deployment is intentionally separate from per-run training effects.
The durable approval and execute claim contain authority identifiers only;
credentials are supplied to the already-composed explicit-client deployer and
are never serialized by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import stat
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    exact_fields,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.providers.modal.packaged_binding import (
    ModalPackagedRuntimeFactsV1,
)
from tuner.execution.providers.modal.runtime_release_deployment import (
    ExplicitModal154ReleaseDeploymentReader,
    ModalRuntimeReleaseDeployer,
    ModalRuntimeReleaseDeploymentError,
    ModalRuntimeReleaseDeploymentFactsV1,
    ModalRuntimeReleaseDeploymentObservationV1,
    ModalRuntimeReleaseDeploymentPlanV1,
    ModalRuntimeReleaseFunctionFactV1,
    ModalRuntimeReleaseSecretFactV1,
    ModalRuntimeReleaseVolumeFactV1,
)


MODAL_RUNTIME_RELEASE_PREFLIGHT_SCHEMA = "synaptic-modal-runtime-release-preflight/v1"
MODAL_RUNTIME_RELEASE_APPROVAL_SCHEMA = "synaptic-modal-runtime-release-approval/v1"
MODAL_RUNTIME_RELEASE_ATTEMPT_SCHEMA = "synaptic-modal-runtime-release-attempt/v1"
MODAL_RUNTIME_RELEASE_OUTCOME_SCHEMA = "synaptic-modal-runtime-release-outcome/v1"
MODAL_RUNTIME_RELEASE_VERIFICATION_SCHEMA = "synaptic-modal-runtime-release-verification/v1"

_KINDS = frozenset({
    "plan", "preflight", "approval", "attempt", "facts", "outcome",
    "observation", "verification",
})
_MAX_DOCUMENT_BYTES = 64 * 1024


class ModalRuntimeReleaseOperatorError(RuntimeError):
    """Closed nonsecret operator failure."""


def _utc(value: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} is not canonical UTC") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{name} is not canonical UTC")
    rendered = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if rendered != value:
        raise ValueError(f"{name} is not canonical UTC")
    return value


def _now(clock: object) -> datetime:
    value = clock() if callable(clock) else clock.now()  # type: ignore[attr-defined]
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("Modal release clock must return an aware datetime")
    return value.astimezone(timezone.utc)


def _doc_id(schema: str, value: Mapping[str, object]) -> str:
    return domain_digest(schema, canonical_bytes(value))


class ModalRuntimeReleaseStatePort(Protocol):
    def publish_if_absent(self, kind: str, release_ref: str, payload: bytes) -> bool: ...
    def resolve(self, kind: str, release_ref: str) -> bytes | None: ...


class LocalModalRuntimeReleaseState:
    """Private exclusive local document store; no provider or credential access."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("Modal release state root must be an absolute path")
        self._root = root

    @staticmethod
    def _validate(kind: str, release_ref: str, payload: bytes | None = None) -> None:
        if kind not in _KINDS:
            raise ValueError("Modal release state kind is unsupported")
        digest_text(release_ref, "release_ref")
        if payload is not None and (
            type(payload) is not bytes or not payload or len(payload) > _MAX_DOCUMENT_BYTES
        ):
            raise ValueError("Modal release state document is missing or oversized")

    def _path(self, kind: str, release_ref: str) -> Path:
        self._validate(kind, release_ref)
        return self._root / release_ref / f"{kind}.json"

    def publish_if_absent(self, kind: str, release_ref: str, payload: bytes) -> bool:
        self._validate(kind, release_ref, payload)
        path = self._path(kind, release_ref)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            return False
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short durable state write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        return True

    def resolve(self, kind: str, release_ref: str) -> bytes | None:
        path = self._path(kind, release_ref)
        try:
            before = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode) or path.is_symlink() \
                or before.st_size <= 0 or before.st_size > _MAX_DOCUMENT_BYTES:
            raise ModalRuntimeReleaseOperatorError("modal_release_state_invalid")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_DOCUMENT_BYTES + 1)
            after = os.fstat(handle.fileno())
        if len(payload) > _MAX_DOCUMENT_BYTES \
                or (before.st_dev, before.st_ino, before.st_size) \
                != (after.st_dev, after.st_ino, after.st_size):
            raise ModalRuntimeReleaseOperatorError("modal_release_state_changed")
        return payload


class InstalledModalRuntimeReleaseEntrypoints:
    """Resolve only the two global callables named by the authenticated plan."""

    def resolve(self, plan: ModalRuntimeReleaseDeploymentPlanV1) -> dict[str, Callable]:
        if type(plan) is not ModalRuntimeReleaseDeploymentPlanV1:
            raise TypeError("exact Modal release plan required")
        result: dict[str, Callable] = {}
        failed = False
        try:
            for spec in plan.functions:
                value: object = importlib.import_module(spec.module)
                for part in spec.qualname.split("."):
                    value = getattr(value, part)
                if not callable(value):
                    raise TypeError
                result[spec.role] = value
        except Exception:
            failed = True
        if failed or set(result) != {"training", "self_check"}:
            raise ModalRuntimeReleaseOperatorError("modal_release_entrypoint_unavailable")
        return result


class ModalRuntimeReleaseOperator:
    """Protected preflight/approval/single-deploy/recovery state machine."""

    __slots__ = ("_state", "_deployer", "_entrypoints", "_clock")

    def __init__(
        self, *, state: ModalRuntimeReleaseStatePort,
        deployer: ModalRuntimeReleaseDeployer | None, entrypoints: object | None,
        clock: object,
    ) -> None:
        if not callable(getattr(state, "publish_if_absent", None)) \
                or not callable(getattr(state, "resolve", None)):
            raise TypeError("Modal release durable state port required")
        if (deployer is None) != (entrypoints is None):
            raise TypeError("Modal release provider collaborators must be supplied together")
        if deployer is not None and (
            type(deployer) is not ModalRuntimeReleaseDeployer
            or not callable(getattr(entrypoints, "resolve", None))
        ):
            raise TypeError("Modal release deployer and entrypoint resolver are invalid")
        _now(clock)
        self._state, self._deployer = state, deployer
        self._entrypoints, self._clock = entrypoints, clock

    def _provider(self) -> tuple[ModalRuntimeReleaseDeployer, object]:
        if self._deployer is None or self._entrypoints is None:
            raise ModalRuntimeReleaseOperatorError("modal_release_provider_unavailable")
        return self._deployer, self._entrypoints

    def _retain(self, kind: str, ref: str, payload: bytes) -> None:
        created = self._state.publish_if_absent(kind, ref, payload)
        retained = self._state.resolve(kind, ref)
        if retained != payload:
            raise ModalRuntimeReleaseOperatorError("modal_release_state_conflict")
        if not created and kind == "attempt":
            raise ModalRuntimeReleaseOperatorError("modal_release_attempt_already_consumed")

    def _plan(self, ref: str) -> ModalRuntimeReleaseDeploymentPlanV1:
        payload = self._state.resolve("plan", digest_text(ref, "release_ref"))
        if payload is None:
            raise ModalRuntimeReleaseOperatorError("modal_release_plan_unavailable")
        plan = ModalRuntimeReleaseDeploymentPlanV1.parse(payload)
        if plan.deployment_spec_digest != ref:
            raise ModalRuntimeReleaseOperatorError("modal_release_plan_substituted")
        return plan

    def preflight(self, plan_bytes: bytes) -> dict[str, object]:
        plan = ModalRuntimeReleaseDeploymentPlanV1.parse(plan_bytes)
        deployer, _ = self._provider()
        ref = plan.deployment_spec_digest
        self._retain("plan", ref, plan.canonical_bytes)
        observation = deployer.observe(plan)
        volumes, secrets = deployer.inspect_resources(plan)
        unsigned: dict[str, object] = {
            "schema_version": MODAL_RUNTIME_RELEASE_PREFLIGHT_SCHEMA,
            "deployment_spec_digest": ref,
            "runtime_release_digest": plan.release.manifest_digest,
            "observed_at": _now(self._clock).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "current_deployment": None if observation is None else observation.to_dict(),
            "volumes": [item.to_dict() for item in volumes],
            "secrets": [item.to_dict() for item in secrets],
            "ready": True,
            "authorizing": False,
        }
        document = {**unsigned, "preflight_id": _doc_id(
            MODAL_RUNTIME_RELEASE_PREFLIGHT_SCHEMA, unsigned,
        )}
        payload = canonical_bytes(document)
        self._retain("preflight", ref, payload)
        return dict(document)

    def approve(
        self, release_ref: str, *, authorization_reference: str,
        issued_at: str, expires_at: str,
    ) -> dict[str, object]:
        ref = digest_text(release_ref, "release_ref")
        plan = self._plan(ref)
        preflight_bytes = self._state.resolve("preflight", ref)
        if preflight_bytes is None:
            raise ModalRuntimeReleaseOperatorError("modal_release_preflight_unavailable")
        preflight = parse_canonical_object(preflight_bytes, name="Modal release preflight")
        issued, expires = _utc(issued_at, "issued_at"), _utc(expires_at, "expires_at")
        if datetime.fromisoformat(expires.replace("Z", "+00:00")) \
                <= datetime.fromisoformat(issued.replace("Z", "+00:00")):
            raise ValueError("Modal release approval must expire after issuance")
        unsigned: dict[str, object] = {
            "schema_version": MODAL_RUNTIME_RELEASE_APPROVAL_SCHEMA,
            "deployment_spec_digest": plan.deployment_spec_digest,
            "runtime_release_digest": plan.release.manifest_digest,
            "preflight_id": preflight.get("preflight_id"),
            "authorization_reference": safe_ref(
                authorization_reference, "authorization_reference"
            ),
            "issued_at": issued,
            "expires_at": expires,
            "permitted_deploy_calls": 1,
            "credentials_included": False,
        }
        document = {**unsigned, "authorization_id": _doc_id(
            MODAL_RUNTIME_RELEASE_APPROVAL_SCHEMA, unsigned,
        )}
        self._retain("approval", ref, canonical_bytes(document))
        return dict(document)

    def _approval(self, ref: str) -> dict[str, object]:
        payload = self._state.resolve("approval", ref)
        if payload is None:
            raise ModalRuntimeReleaseOperatorError("modal_release_approval_unavailable")
        value = parse_canonical_object(payload, name="Modal release approval")
        exact_fields(value, frozenset({
            "schema_version", "deployment_spec_digest", "runtime_release_digest",
            "preflight_id", "authorization_reference", "issued_at", "expires_at",
            "permitted_deploy_calls", "credentials_included", "authorization_id",
        }), "Modal release approval")
        unsigned = dict(value)
        authorization_id = unsigned.pop("authorization_id")
        if value["schema_version"] != MODAL_RUNTIME_RELEASE_APPROVAL_SCHEMA \
                or value["deployment_spec_digest"] != ref \
                or value["permitted_deploy_calls"] != 1 \
                or value["credentials_included"] is not False \
                or authorization_id != _doc_id(MODAL_RUNTIME_RELEASE_APPROVAL_SCHEMA, unsigned):
            raise ModalRuntimeReleaseOperatorError("modal_release_approval_invalid")
        expires = datetime.fromisoformat(str(value["expires_at"]).replace("Z", "+00:00"))
        if _now(self._clock) >= expires:
            raise ModalRuntimeReleaseOperatorError("modal_release_approval_expired")
        return value

    def execute(self, release_ref: str) -> dict[str, object]:
        ref = digest_text(release_ref, "release_ref")
        plan = self._plan(ref)
        approval = self._approval(ref)
        attempt_unsigned: dict[str, object] = {
            "schema_version": MODAL_RUNTIME_RELEASE_ATTEMPT_SCHEMA,
            "deployment_spec_digest": ref,
            "authorization_id": approval["authorization_id"],
            "claimed_at": _now(self._clock).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "deploy_call_limit": 1,
            "authorizing": True,
            "credentials_included": False,
        }
        attempt = {**attempt_unsigned, "attempt_id": _doc_id(
            MODAL_RUNTIME_RELEASE_ATTEMPT_SCHEMA, attempt_unsigned,
        )}
        # This exclusive durable claim is the point of no replay. It precedes
        # entrypoint resolution, resource rechecks, and the provider mutation.
        self._retain("attempt", ref, canonical_bytes(attempt))
        try:
            deployer, entrypoint_resolver = self._provider()
            entrypoints = entrypoint_resolver.resolve(plan)
            facts = deployer.deploy_once(plan, entrypoints=entrypoints)
        except Exception:
            return self._retain_outcome(
                ref, attempt, status="INDETERMINATE",
                reason="DEPLOYMENT_ACKNOWLEDGEMENT_UNAVAILABLE",
            )
        try:
            facts.validate_plan(plan)
            self._retain("facts", ref, facts.canonical_bytes)
        except Exception:
            return self._retain_outcome(
                ref, attempt, status="INDETERMINATE",
                reason="ACKNOWLEDGED_FACTS_NOT_DURABLE",
            )
        return self._retain_outcome(
            ref, attempt, status="ACKNOWLEDGED", reason="DEPLOY_RETURNED",
            facts_digest=facts.facts_digest,
        )

    def _retain_outcome(
        self, ref: str, attempt: Mapping[str, object], *, status: str,
        reason: str, facts_digest: str | None = None,
    ) -> dict[str, object]:
        if status not in {"ACKNOWLEDGED", "INDETERMINATE"}:
            raise ValueError("Modal release outcome status is unsupported")
        unsigned: dict[str, object] = {
            "schema_version": MODAL_RUNTIME_RELEASE_OUTCOME_SCHEMA,
            "deployment_spec_digest": ref,
            "attempt_id": attempt["attempt_id"],
            "status": status,
            "reason_code": safe_ref(reason, "reason_code"),
            "facts_digest": facts_digest,
            "retry_allowed": False,
            "binding_authorized": status == "ACKNOWLEDGED",
        }
        document = {**unsigned, "outcome_id": _doc_id(
            MODAL_RUNTIME_RELEASE_OUTCOME_SCHEMA, unsigned,
        )}
        try:
            self._retain("outcome", ref, canonical_bytes(document))
        except Exception:
            return {
                "schema_version": MODAL_RUNTIME_RELEASE_OUTCOME_SCHEMA,
                "deployment_spec_digest": ref,
                "status": "INDETERMINATE",
                "reason_code": "OUTCOME_NOT_DURABLE",
                "retry_allowed": False,
                "binding_authorized": False,
            }
        return dict(document)

    @staticmethod
    def _matches_observation(
        facts: ModalRuntimeReleaseDeploymentFactsV1,
        observation: ModalRuntimeReleaseDeploymentObservationV1 | None,
    ) -> bool:
        expected_layout = tuple(sorted(
            (item.spec.name, item.function_id) for item in facts.functions
        ))
        return observation is not None \
            and observation.deployed is True \
            and observation.app_id == facts.app_id \
            and observation.generation == facts.generation \
            and observation.function_ids == expected_layout \
            and observation.class_ids == ()

    def observe(self, release_ref: str) -> dict[str, object]:
        ref = digest_text(release_ref, "release_ref")
        plan = self._plan(ref)
        facts_bytes = self._state.resolve("facts", ref)
        if facts_bytes is None:
            return {
                "status": "INDETERMINATE", "retry_allowed": False,
                "binding_authorized": False,
            }
        facts = ModalRuntimeReleaseDeploymentFactsV1.parse(facts_bytes)
        facts.validate_plan(plan)
        deployer, _ = self._provider()
        observation = deployer.observe(plan)
        if not self._matches_observation(facts, observation):
            return {
                "status": "CHANGED", "retry_allowed": False,
                "binding_authorized": False,
            }
        payload = canonical_bytes(observation.to_dict())  # type: ignore[union-attr]
        self._retain("observation", ref, payload)
        return {
            "status": "CURRENT", "facts_digest": facts.facts_digest,
            "app_id": facts.app_id, "generation": facts.generation,
            "current_state_version_pinned": False,
            "function_image_link_readable": False,
            "retry_allowed": False,
        }

    def recover(self, release_ref: str) -> dict[str, object]:
        ref = digest_text(release_ref, "release_ref")
        if self._state.resolve("attempt", ref) is None:
            raise ModalRuntimeReleaseOperatorError("modal_release_attempt_unavailable")
        if self._state.resolve("facts", ref) is None:
            return {
                "status": "INDETERMINATE", "recovered": False,
                "retry_allowed": False, "binding_authorized": False,
            }
        result = self.observe(ref)
        return {
            **result, "recovered": result.get("status") == "CURRENT",
            "retry_allowed": False,
        }

    def verify(self, release_ref: str) -> dict[str, object]:
        ref = digest_text(release_ref, "release_ref")
        observed = self.observe(ref)
        if observed.get("status") != "CURRENT":
            raise ModalRuntimeReleaseOperatorError("modal_release_not_current")
        facts_bytes = self._state.resolve("facts", ref)
        assert facts_bytes is not None
        facts = ModalRuntimeReleaseDeploymentFactsV1.parse(facts_bytes)
        packaged = ModalPackagedRuntimeFactsV1.from_release_deployment(facts)
        provider = packaged.build_provider_binding(self._plan(ref).release)
        unsigned: dict[str, object] = {
            "schema_version": MODAL_RUNTIME_RELEASE_VERIFICATION_SCHEMA,
            "deployment_spec_digest": ref,
            "release_facts_digest": facts.facts_digest,
            "packaged_runtime_facts_digest": packaged.facts_digest,
            "provider_runtime_binding_digest": provider.binding_digest,
            "verified_at": _now(self._clock).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "current_state_version_pinned": False,
            "function_image_link_readable": False,
        }
        document = {**unsigned, "verification_id": _doc_id(
            MODAL_RUNTIME_RELEASE_VERIFICATION_SCHEMA, unsigned,
        )}
        self._retain("verification", ref, canonical_bytes(document))
        return dict(document)


def run_modal_runtime_release_action(
    action: str, *, operator: ModalRuntimeReleaseOperator,
    release_ref: str | None = None, plan_bytes: bytes | None = None,
    authorization_reference: str | None = None,
    issued_at: str | None = None, expires_at: str | None = None,
) -> dict[str, object]:
    """Dispatch the six exact transitions without inferring missing authority."""
    if type(operator) is not ModalRuntimeReleaseOperator:
        raise TypeError("exact Modal runtime release operator required")
    if action == "preflight":
        if type(plan_bytes) is not bytes:
            raise ValueError("Modal runtime release preflight requires plan bytes")
        return operator.preflight(plan_bytes)
    if release_ref is None:
        raise ValueError("Modal runtime release action requires release_ref")
    if action == "approve":
        if None in (authorization_reference, issued_at, expires_at):
            raise ValueError("Modal runtime release approval inputs are incomplete")
        return operator.approve(
            release_ref, authorization_reference=authorization_reference,  # type: ignore[arg-type]
            issued_at=issued_at, expires_at=expires_at,  # type: ignore[arg-type]
        )
    if action == "execute":
        return operator.execute(release_ref)
    if action == "observe":
        return operator.observe(release_ref)
    if action == "recover":
        return operator.recover(release_ref)
    if action == "verify":
        return operator.verify(release_ref)
    raise ValueError("unsupported Modal runtime release action")


def _cli_state(args: object, context: object) -> LocalModalRuntimeReleaseState:
    value = getattr(args, "base_dir", None)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ModalRuntimeReleaseOperatorError("modal_release_state_root_required")
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ModalRuntimeReleaseOperatorError("modal_release_state_root_not_absolute")
    root = Path(os.path.abspath(root))
    for source in {
        Path(getattr(context, "engine_root")).resolve(),
        Path(getattr(context, "project_root")).resolve(),
    }:
        try:
            root.relative_to(source)
        except ValueError:
            pass
        else:
            raise ModalRuntimeReleaseOperatorError("modal_release_state_root_in_source")
    return LocalModalRuntimeReleaseState((root / "modal-runtime-releases").resolve())


def _read_cli_plan(args: object, context: object) -> bytes:
    value = getattr(args, "release_plan", None)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ModalRuntimeReleaseOperatorError("modal_release_plan_path_required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(getattr(context, "invocation_cwd")) / path
    path = Path(os.path.abspath(path))
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode) \
                or before.st_size <= 0 or before.st_size > _MAX_DOCUMENT_BYTES:
            raise ValueError
        with path.open("rb") as handle:
            payload = handle.read(_MAX_DOCUMENT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except (OSError, ValueError):
        raise ModalRuntimeReleaseOperatorError("modal_release_plan_path_invalid") from None
    if len(payload) > _MAX_DOCUMENT_BYTES \
            or (before.st_dev, before.st_ino, before.st_size) \
            != (after.st_dev, after.st_ino, after.st_size):
        raise ModalRuntimeReleaseOperatorError("modal_release_plan_changed")
    return payload


def _cli_release_ref(args: object) -> str:
    return digest_text(str(getattr(args, "release_ref", "") or "").strip(), "release_ref")


def _compose_cli_provider(
    *, args: object, context: object, plan: ModalRuntimeReleaseDeploymentPlanV1,
) -> tuple[ModalRuntimeReleaseDeployer, InstalledModalRuntimeReleaseEntrypoints]:
    """Compose an explicit Modal client without serializing its credentials."""
    from shared.utilities.env import load_env_file
    from tuner.execution.providers.modal.binding import ModalClientBinding

    explicit_env = getattr(args, "env_file", None)
    load_env_file(
        context=context,
        explicit_path=explicit_env if explicit_env else None,
    )
    token_id = str(os.environ.get("MODAL_TOKEN_ID", "")).strip()
    token_secret = str(os.environ.get("MODAL_TOKEN_SECRET", "")).strip()
    if not token_id or not token_secret:
        raise ModalRuntimeReleaseOperatorError("modal_release_credentials_unavailable")
    try:
        import modal

        client = modal.Client.from_credentials(token_id, token_secret)
        workspace = modal.Workspace.from_context(client=client)
        workspace.hydrate(client)
        workspace_ref = safe_ref(getattr(workspace, "name", None), "workspace_ref")
        if getattr(workspace, "is_hydrated", False) is not True:
            raise ValueError
        binding = ModalClientBinding(
            workspace_ref, workspace_ref, plan.environment_name,
            "modal-runtime-release-explicit-client", getattr(modal, "__version__", ""),
        )
        deployer = ModalRuntimeReleaseDeployer(
            sdk=modal, client=client, client_binding=binding,
            reader=ExplicitModal154ReleaseDeploymentReader(sdk=modal),
        )
    except Exception:
        raise ModalRuntimeReleaseOperatorError("modal_release_provider_unavailable") from None
    return deployer, InstalledModalRuntimeReleaseEntrypoints()


def run_modal_runtime_release_cli_action(
    action: str, *, args: object, context: object,
) -> dict[str, object]:
    """Compose the protected release command without ambient provider fallback."""
    if action not in {"preflight", "approve", "execute", "recover", "observe", "verify"}:
        raise ValueError("unsupported Modal runtime release action")
    state = _cli_state(args, context)
    plan_bytes: bytes | None = None
    release_ref: str | None = None
    if action == "preflight":
        plan_bytes = _read_cli_plan(args, context)
        plan = ModalRuntimeReleaseDeploymentPlanV1.parse(plan_bytes)
    else:
        release_ref = _cli_release_ref(args)
        retained = state.resolve("plan", release_ref)
        if retained is None:
            raise ModalRuntimeReleaseOperatorError("modal_release_plan_unavailable")
        plan = ModalRuntimeReleaseDeploymentPlanV1.parse(retained)

    if action == "approve":
        deployer = entrypoints = None
    else:
        deployer, entrypoints = _compose_cli_provider(
            args=args, context=context, plan=plan,
        )
    operator = ModalRuntimeReleaseOperator(
        state=state, deployer=deployer, entrypoints=entrypoints,
        clock=lambda: datetime.now(timezone.utc),
    )
    return run_modal_runtime_release_action(
        action, operator=operator, release_ref=release_ref, plan_bytes=plan_bytes,
        authorization_reference=getattr(args, "authorization_reference", None),
        issued_at=getattr(args, "issued_at", None),
        expires_at=getattr(args, "expires_at", None),
    )


__all__ = [
    "MODAL_RUNTIME_RELEASE_APPROVAL_SCHEMA",
    "MODAL_RUNTIME_RELEASE_ATTEMPT_SCHEMA",
    "MODAL_RUNTIME_RELEASE_OUTCOME_SCHEMA",
    "MODAL_RUNTIME_RELEASE_PREFLIGHT_SCHEMA",
    "MODAL_RUNTIME_RELEASE_VERIFICATION_SCHEMA",
    "InstalledModalRuntimeReleaseEntrypoints",
    "LocalModalRuntimeReleaseState",
    "ModalRuntimeReleaseOperator",
    "ModalRuntimeReleaseOperatorError",
    "ModalRuntimeReleaseStatePort",
    "run_modal_runtime_release_action",
    "run_modal_runtime_release_cli_action",
]
