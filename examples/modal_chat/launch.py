"""Run the dedicated consumer: checked source, one training attempt, bounded chat.

Execute this checked-in file from a clean consumer's pinned engine submodule.
The default mode checks local inputs only. No Docker or local model staging is
used. Qualification mode records native training evidence while public read
capabilities remain disabled; it never pretends that it exercised public chat.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time

ENGINE = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ENGINE))

from examples.modal_chat.authority import HMACAuthenticator, UTCClock
from examples.modal_chat.configuration import (
    RATE_CALCULATION_PURPOSE,
    _packaged,
    _qualification,
    build_authenticated_inference_configuration,
    issue_authenticated_modal_quote,
)
from examples.modal_chat.consumer import chat_once, submit_training_once
from examples.modal_chat.deployment import ModalChatOwnedDeployment, ModalChatScope
from examples.modal_chat.diagnostics import modal_chat_failure_diagnostic
from examples.modal_chat.provisioning import ModalChatProvisioner
from examples.modal_chat.replay import ModalChatEvidenceReplay
from examples.modal_chat.resolution import _regular_digest
from examples.modal_chat.settings import ModalChatSettings
from examples.modal_chat.storage import ModalChatStorage
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState
from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.providers.modal.composition import ModalVerificationPolicyV1
from tuner.execution.providers.modal.config import ModalRuntimeLockV1
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.inference_preparation import (
    CONFIG_EVIDENCE_PURPOSE,
)
from tuner.execution.providers.modal.inference_transport import (
    ModalChatLaunchSettings,
    _bounded,
)
from tuner.execution.providers.modal.mounted_io import read_regular
from tuner.project.git_verification import GitCliLocalSourceInspector
from tuner.project.manifest import load_project_manifest

_KEY_REF = "consumer-evidence"
_MODEL_KEY = "HF_TOKEN"
_EVIDENCE_KEY = "SYNAPTIC_EVIDENCE_MAC_KEY"
_PURPOSES = frozenset(
    {
        "source-lock-evidence/v1",
        "modal-deployment-evidence/v1",
        "modal-stage-claim/v2",
        "modal-launch-claim/v1",
        "modal-terminal/v1",
        "modal-log-metadata/v1",
        "modal-completion/v1",
        "provider-run-observation/v1",
        "provider-log-page/v1",
        "modal-quote-evidence/v1",
        "modal-inference-launch/v1",
        CONFIG_EVIDENCE_PURPOSE,
        RATE_CALCULATION_PURPOSE,
    }
)


class ModalChatLauncherError(RuntimeError):
    """Closed launcher failure; persisted attempts are never reset."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ModalChatLauncherError("modal_chat_arguments_invalid")


def build_parser():
    parser = _Parser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("check", "qualify-training", "train-chat"), default="check"
    )
    parser.add_argument("--modal-profile")
    parser.add_argument("--hf-token-env-file", type=Path)
    return parser


def _expiry(now, seconds):
    return (
        (
            datetime.fromisoformat(now.replace("Z", "+00:00"))
            + timedelta(seconds=seconds)
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def _record(storage, catalog, key, document):
    payload = canonical_bytes(document)
    target = storage.catalog(catalog, encode=bytes, decode=bytes)
    target.publish_if_absent(key, payload)
    if target.resolve(key) != payload:
        raise ModalChatLauncherError("modal_chat_evidence_conflict")
    return payload


def _private_directory(path):
    """Create only absent consumer output components; never chmod existing data."""
    missing = []
    cursor = path
    while not cursor.exists():
        if cursor.is_symlink():
            raise ModalChatLauncherError("modal_chat_storage_invalid")
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.resolve(strict=True) != cursor or not cursor.is_dir():
        raise ModalChatLauncherError("modal_chat_storage_invalid")
    for selected in reversed(missing):
        selected.mkdir(mode=0o700)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or path.resolve(strict=True) != path
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ModalChatLauncherError("modal_chat_storage_invalid")
    return path


def _client(sdk, profile):
    safe_ref(profile, "modal_profile")
    if sdk.__version__ != "1.5.4":
        raise ModalChatLauncherError("modal_chat_sdk_invalid")
    from modal.config import config

    token_id = config.get("token_id", profile=profile, use_env=False)
    token_secret = config.get("token_secret", profile=profile, use_env=False)
    if any(
        type(value) is not str or not value.strip()
        for value in (token_id, token_secret)
    ):
        raise ModalChatLauncherError("modal_chat_credentials_missing")
    return sdk.Client.from_credentials(token_id, token_secret)


def _check_launcher_python():
    """Reject incompatible serialized-worker Python before credentials or state."""
    locked = ModalRuntimeLockV1.packaged()
    # The pinned image runs this interpreter. Serialized code cannot cross
    # Python minor versions; require the reviewed patch version as well.
    if (
        sys.implementation.name != locked.python_implementation.lower()
        or tuple(sys.version_info[:3])
        != tuple(int(part) for part in locked.python_version.split("."))
        or sys.version_info.releaselevel != "final"
    ):
        raise ModalChatLauncherError("modal_chat_launcher_python_mismatch")


def _model_token(selected_env_file):
    value = os.environ.get(_MODEL_KEY)
    if selected_env_file is not None:
        # Read only the operator-selected existing file and named key. Never
        # search for credentials or export other values from this file.
        from dotenv import get_key

        if not selected_env_file.is_absolute() or not selected_env_file.is_file():
            raise ModalChatLauncherError("modal_chat_credentials_missing")
        value = get_key(str(selected_env_file), _MODEL_KEY)
    if type(value) is not str or not value.strip():
        raise ModalChatLauncherError("modal_chat_credentials_missing")
    return value


def check_inputs(project_root, configuration, *, mode="check"):
    if mode not in {"check", "qualify-training", "train-chat"}:
        raise ModalChatLauncherError("modal_chat_arguments_invalid")
    if (
        not project_root.is_absolute()
        or project_root.resolve(strict=True) != project_root
    ):
        raise ModalChatLauncherError("modal_chat_project_invalid")
    if configuration.is_absolute() or ".." in configuration.parts:
        raise ModalChatLauncherError("modal_chat_configuration_invalid")
    manifest = load_project_manifest(project_root / "synaptic.yaml")
    context = manifest.create_context(engine_root=ENGINE, invocation_cwd=Path.cwd())
    raw = read_regular(project_root, project_root / configuration, 256 * 1024)
    settings = ModalChatSettings.parse(raw[:-1] if raw.endswith(b"\n") else raw)
    _regular_digest(project_root, Path(settings.dataset_project_path), 16 * 1024 * 1024)
    source = GitCliLocalSourceInspector().inspect(context=context)
    allowed = frozenset((item["url"], item["ref"]) for item in settings.allowed_refs)
    observed_refs = set()
    for item in (source.project_source, source.engine_source):
        if item is None or type(item.branch) is not str or not item.branch:
            raise ModalChatLauncherError("modal_chat_source_invalid")
        observed_refs.add((item.location.canonical_url, "refs/heads/" + item.branch))
    if len(observed_refs) != 2 or observed_refs != allowed:
        raise ModalChatLauncherError("modal_chat_source_invalid")
    if mode != "qualify-training":
        # Training qualification never opens chat or relies on inference-image
        # authority. Check the chat image only for the paths that require it.
        reviewed = settings.reviewed_runtime
        _qualification(
            ENGINE / reviewed["capture"]["path"],
            reviewed["capture"]["sha256"],
            ENGINE / reviewed["readback"]["path"],
            reviewed["readback"]["sha256"],
        )
        _packaged()
    return manifest, context, settings, source


def _snapshot_workflow(host, storage, run):
    workflow = host.stores.workflow_store.get(run)
    if workflow is None:
        return None
    _record(storage, "launch-workflows", workflow.record_digest, workflow.to_dict())
    return workflow


def _snapshot_training(host, storage, run):
    workflow = _snapshot_workflow(host, storage, run)
    if workflow is None:
        return None
    snapshots = storage.catalog("launch-foundation", encode=bytes, decode=bytes)
    for intent in (workflow.stage, workflow.submit):
        if intent is None:
            continue
        for payload in (
            intent.canonical_command_bytes,
            *(item.canonical_snapshot_bytes for item in intent.foundation_bindings),
            *(item.canonical_assessment_bytes for item in intent.foundation_bindings),
        ):
            digest = hashlib.sha256(payload).hexdigest()
            snapshots.publish_if_absent(digest, payload)
            if snapshots.resolve(digest) != payload:
                raise ModalChatLauncherError("modal_chat_evidence_conflict")
    if workflow.provider_run_ref is None:
        return workflow
    reference = workflow.provider_run_ref.reference
    _record(
        storage,
        "launch-run-ownership",
        run.run_id,
        {
            "schema_version": "synaptic-modal-chat-known-run/v1",
            "run": run.to_dict(),
            "provider_job_ref": reference.provider_job_ref,
            "workflow_record_digest": workflow.record_digest,
            "submit_command_digest": workflow.submit.command_digest,
            "provider_shutdown_proof": False,
        },
    )
    return workflow


def _wait_training(host, deployment, storage, run, *, timeout_seconds, emit):
    workflow = _snapshot_training(host, storage, run)
    if workflow is None or workflow.provider_run_ref is None:
        raise ModalChatLauncherError("modal_chat_submit_unresolved")
    reference = workflow.provider_run_ref.reference
    facade = deployment.facade()
    deadline = time.monotonic() + timeout_seconds + 120
    while (
        facade.observe_known_call_pending(reference.provider_job_ref)
        is ModalFunctionCallState.PENDING
    ):
        if time.monotonic() >= deadline:
            raise ModalChatLauncherError("modal_chat_training_wait_expired")
        emit("TRAINING_PENDING")
        time.sleep(min(15.0, max(0.0, deadline - time.monotonic())))
    # UNKNOWN is not success: only authenticated terminal evidence may advance
    # the workflow. Do not retry a submit, or invent RUNNING state from this hint.


def _confirm_chat_cleanup(graph, storage, attempt_ref):
    known = []
    lease = graph.runtime.owned_lease
    if lease is not None:
        known.append((lease.submit_command_digest, lease.sandbox))
    unresolved = False
    for ownership in graph.pending_ownership():
        try:
            ownership.close()
        except Exception:
            # A cleanup error is not evidence of absence. Still inspect only
            # that owned handle and persist an unresolved record if necessary.
            pass
        if ownership.sandbox is None:
            unresolved = True
        else:
            known.append((ownership.submit_command_digest, ownership.sandbox))
    seen = set()
    records = []
    for digest, sandbox in known:
        object_id = safe_ref(sandbox.object_id, "sandbox_id")
        if object_id in seen:
            continue
        seen.add(object_id)
        try:
            code = _bounded(sandbox.poll, deadline=time.monotonic() + 15)
            confirmed = type(code) is int and sandbox.object_id == object_id
        except Exception:
            code, confirmed = None, False
        unresolved |= not confirmed
        records.append(
            {
                "sandbox_id": object_id,
                "submit_command_digest": digest,
                "returncode": code if confirmed else None,
                "provider_shutdown_proof": confirmed,
            }
        )
    submitted = bool(graph.catalog.submit_digests())
    unresolved |= submitted and not records
    _record(
        storage,
        "launch-chat-cleanup",
        attempt_ref,
        {
            "schema_version": "synaptic-modal-chat-cleanup-readback/v1",
            "known_instances": records,
            "submit_command_retained": submitted,
            "unresolved_creation_or_cleanup": unresolved,
            "provider_shutdown_proof": bool(records) and not unresolved,
        },
    )
    if unresolved:
        raise ModalChatLauncherError("modal_chat_cleanup_unconfirmed")


def _chat(
    training, deployment, scope, storage, settings, ids, auth, clock, run, *, emit
):
    from examples.modal_chat.chat import compose_modal_run_chat

    now = clock.now_iso()
    session = settings.attempt_ref + "-chat"
    reviewed = settings.reviewed_runtime
    emit("CHAT_CONFIGURATION")
    configuration, trust = build_authenticated_inference_configuration(
        scope=scope,
        reviewed_capture_path=ENGINE / reviewed["capture"]["path"],
        reviewed_capture_sha256=reviewed["capture"]["sha256"],
        reviewed_readback_path=ENGINE / reviewed["readback"]["path"],
        reviewed_readback_sha256=reviewed["readback"]["sha256"],
        **settings.inference,
        volumes={
            "source_artifact_volume_ref": settings.profile.artifact_volume_ref,
            "source_artifact_volume_id": ids["artifacts"],
            "chat_control_volume_ref": settings.profile.control_volume_ref,
            "chat_control_volume_id": ids["training_control"],
            "model_cache_volume_ref": settings.attempt_ref + "-model-cache",
            "model_cache_volume_id": ids["model_cache"],
            "key_ref": _KEY_REF,
        },
        secrets=[
            {
                "name": settings.profile.secrets[0].name,
                "required_keys": list(settings.profile.secrets[0].required_keys),
            }
        ],
        evidence={
            "issuer_ref": "consumer-config",
            "evidence_ref": session + "-configuration",
            "audience_ref": session,
            "challenge_nonce": secrets.token_hex(16),
            "key_ref": _KEY_REF,
            "verified_at": now,
            "expires_at": _expiry(now, 300),
        },
        authenticator=auth,
    )
    _, execution, _ = training.preparation._snapshot()
    now = clock.now_iso()
    emit("CHAT_QUOTE")
    quote, quote_trust, calculation, calculation_tag = issue_authenticated_modal_quote(
        scope=scope,
        configuration=configuration,
        namespace_ref=execution.scope.namespace_ref,
        maximum_cost_minor_units=settings.maximum_chat_cost_minor_units,
        execution_kind="sandbox",
        issued_at=now,
        expires_at=_expiry(now, 300),
        issuer_ref="consumer-quote",
        audience_ref=session,
        challenge_nonce=secrets.token_hex(16),
        key_ref=_KEY_REF,
        authenticator=auth,
        clock=clock,
    )
    _record(
        storage,
        "launch-chat-authority",
        session,
        {
            "schema_version": "synaptic-modal-chat-authority-record/v1",
            "configuration": json.loads(configuration.body_bytes),
            "configuration_tag_hex": configuration.tag.hex(),
            "quote": json.loads(quote.body_bytes),
            "quote_tag_hex": quote.tag.hex(),
            "rate_calculation": json.loads(calculation),
            "rate_tag_hex": calculation_tag.hex(),
        },
    )
    document = json.loads(configuration.body_bytes)
    emit("CHAT_COMPOSITION")
    graph = compose_modal_run_chat(
        host=training.host,
        deployment=deployment,
        launch_source=training.launch_source,
        recipes=training.recipes,
        configuration=configuration,
        configuration_trust=trust,
        quote=quote,
        quote_trust=quote_trust,
        evidence_authenticator=auth,
        clock=clock,
        session_id=session,
        stage_nonce=secrets.token_hex(16),
        submit_nonce=secrets.token_hex(16),
        executor_version="0.1.0",
        policy_digest=hashlib.sha256(settings.canonical_bytes).hexdigest(),
        requirement_digest=quote.body.quote_digest,
        launch_settings=ModalChatLaunchSettings(
            configuration.body_bytes,
            "consumer-chat-launch",
            session,
            _KEY_REF,
            secrets.token_hex(16),
            "/artifacts",
            "/control",
            "/cache",
            _EVIDENCE_KEY,
            _MODEL_KEY,
        ),
        launch_signer=auth,
        issued_at=now,
        expires_at=_expiry(now, 300),
        evidence_ref=session + "-launch",
        profile_ref=document["provider"]["profile_ref"],
        namespace_ref=execution.scope.namespace_ref,
        prequalified_image_id=document["image"]["provider_image_id"],
    )
    with _chat_cleanup_guard(graph, storage, session, emit=emit):
        emit("CHAT_OPEN")
        return chat_once(
            training.host.api.runs,
            storage,
            attempt_ref=session,
            run=run,
            runtime=graph.runtime,
            prompt=settings.prompt,
        )


@contextlib.contextmanager
def _chat_cleanup_guard(graph, storage, session, *, emit):
    try:
        yield
    except BaseException:
        # Always attempt exact cleanup without replacing the original failure.
        try:
            _confirm_chat_cleanup(graph, storage, session)
        except BaseException as cleanup_error:
            try:
                payload = modal_chat_failure_diagnostic(
                    phase="CHAT_CLEANUP", error=cleanup_error
                )
                _record(
                    storage,
                    "launch-chat-cleanup-failures",
                    session,
                    json.loads(payload),
                )
            except BaseException:
                # An unavailable evidence store cannot justify replay or erase
                # the original exception. Its diagnostic remains primary.
                pass
        raise
    else:
        emit("CHAT_CLEANUP")
        _confirm_chat_cleanup(graph, storage, session)


def execute(
    manifest, context, settings, source, *, mode, profile, hf_token_env_file, emit
):
    if type(mode) is not str or mode not in {"qualify-training", "train-chat"}:
        raise ModalChatLauncherError("modal_chat_arguments_invalid")
    _check_launcher_python()
    from tuner.execution.providers.modal.coordinator_adapter import _descriptor

    if mode == "train-chat":
        capabilities = _descriptor().capabilities
        if (
            capabilities.observe is not True
            or capabilities.artifact_streaming is not True
        ):
            raise ModalChatLauncherError("modal_chat_public_capabilities_unqualified")

    import modal
    from examples.modal_chat.training import (
        ModalChatRunIdentity,
        compose_modal_training_host,
    )

    clock = UTCClock()
    private = _private_directory(
        context.state_root / "modal-chat" / settings.attempt_ref
    )
    with ModalChatStorage(private / "smoke.sqlite3", manifest.project_id) as storage:
        storage.attempts.claim(
            "launch",
            canonical_bytes(
                {
                    "schema_version": "synaptic-modal-chat-launch-attempt/v1",
                    "mode": mode,
                    "configuration_sha256": hashlib.sha256(
                        settings.canonical_bytes
                    ).hexdigest(),
                    "consumer_commit": source.project_source.commit,
                    "engine_commit": source.engine_source.commit,
                }
            ),
        )
        token = _model_token(hf_token_env_file)
        scope = ModalChatScope(
            sdk=modal,
            client=_client(modal, profile),
            environment_name=settings.environment_name,
            client_ref="consumer-client",
        )
        key = secrets.token_bytes(32)
        auth = HMACAuthenticator({_KEY_REF: key}, allowed_purposes=_PURPOSES)
        emit("PROVISIONING")
        ids = ModalChatProvisioner(
            scope=scope,
            storage=storage,
            training_control=settings.profile.control_volume_ref,
            artifacts=settings.profile.artifact_volume_ref,
            model_cache=settings.attempt_ref + "-model-cache",
            runtime_secret=settings.profile.secrets[0].name,
            model_key_name=_MODEL_KEY,
            evidence_key_name=_EVIDENCE_KEY,
            secret_values={
                _MODEL_KEY: token,
                _EVIDENCE_KEY: base64.b64encode(key).decode("ascii"),
            },
        ).provision_once(attempt_ref="provision")
        deployment = ModalChatOwnedDeployment(
            scope=scope,
            storage=storage,
            profile=settings.profile,
            runtime_environment=settings.runtime_environment,
            timeout_seconds=settings.training_timeout_seconds,
            evidence_environment_key=_EVIDENCE_KEY,
            evidence_key_ref=_KEY_REF,
            model_token_key=_MODEL_KEY,
        )
        emit("DEPLOYING")
        deployment.deploy_once(attempt_ref="deploy")
        run = TrainingRunRef(settings.attempt_ref, manifest.project_id)
        audience = settings.attempt_ref + "-training"
        emit("RESOLVING")
        training = compose_modal_training_host(
            scope=scope,
            deployment=deployment,
            profile=settings.profile,
            storage=storage,
            context=context,
            request_json=settings.training_json,
            project_ref=manifest.project_id,
            request_identity=ModalChatRunIdentity(
                settings.attempt_ref + "-request",
                run,
                hashlib.sha256(settings.training_json.encode("utf-8")).hexdigest(),
            ),
            dataset_project_path=Path(settings.dataset_project_path),
            allowed_git_refs=frozenset(
                (item["url"], item["ref"]) for item in settings.allowed_refs
            ),
            load_in_4bit=settings.load_in_4bit,
            runtime_environment=settings.runtime_environment,
            timeout_seconds=settings.training_timeout_seconds,
            evidence_authenticator=auth,
            evidence_key_ref=_KEY_REF,
            source_policy=ModalVerificationPolicyV1(
                audience,
                "consumer-source",
                "consumer-deployment",
                _KEY_REF,
                _KEY_REF,
                lambda purpose: secrets.token_hex(16),
                lambda purpose: secrets.token_hex(16),
            ),
            replay=ModalChatEvidenceReplay(storage),
            clock=clock,
            created_at=clock.now_iso(),
            quote_issuer_ref="consumer-quote",
            quote_audience_ref=audience,
            quote_challenge_nonce=secrets.token_hex(16),
            quote_key_ref=_KEY_REF,
            maximum_cost_minor_units=settings.maximum_training_cost_minor_units,
            log_terminal_policy=canonical_bytes(settings.log_terminal_policy),
            control_volume_id=ids["training_control"],
            artifact_volume_id=ids["artifacts"],
            stage_key_ref=_KEY_REF,
            artifact_key=secrets.token_bytes(32),
            grant_key=secrets.token_bytes(32),
            receipt_key=secrets.token_bytes(32),
            invalid_evidence_key=secrets.token_bytes(32),
            assessment_key=secrets.token_bytes(32),
            cursor_key=secrets.token_bytes(32),
            observed_at=clock.now_iso(),
        )
        emit("SUBMITTING")
        try:
            submitted = submit_training_once(
                training.host.api.training,
                storage,
                attempt_ref="training",
                request_json=settings.training_json,
                provider=ProviderRef("modal", settings.profile.profile),
            )
        finally:
            _snapshot_training(training.host, storage, run)
        emit("TRAINING_ACCEPTED")
        _wait_training(
            training.host,
            deployment,
            storage,
            submitted.start.run,
            timeout_seconds=settings.training_timeout_seconds,
            emit=emit,
        )
        if mode == "qualify-training":
            from examples.modal_chat.qualification import qualify_modal_chat_run

            qualify_modal_chat_run(
                host=training.host, storage=storage, run=submitted.start.run
            )
            emit("NATIVE_TRAINING_QUALIFIED")
            return
        outcome = training.host.api.runs.outcome(submitted.start.run)
        if (
            outcome.run != submitted.start.run
            or outcome.state is not TrainingRunState.SUCCEEDED
        ):
            raise ModalChatLauncherError("modal_chat_training_not_succeeded")
        verified = training.host.api.runs.verify(submitted.start.run)
        if verified.run != submitted.start.run or verified.verified is not True:
            raise ModalChatLauncherError("modal_chat_training_not_verified")
        # Ownership is immutable and was recorded at submit. Retain the later
        # workflow by its own digest without rewriting that queued ownership row.
        if _snapshot_workflow(training.host, storage, submitted.start.run) is None:
            raise ModalChatLauncherError("modal_chat_training_not_verified")
        emit("CHATTING")
        _chat(
            training,
            deployment,
            scope,
            storage,
            settings,
            ids,
            auth,
            clock,
            submitted.start.run,
            emit=emit,
        )
        emit("CHAT_SAVED_AND_STOPPED")


def main(argv=None):
    stream = sys.stdout
    phase = "INPUTS"

    def emit(code):
        nonlocal phase
        phase = code
        stream.write(
            json.dumps(
                {"schema_version": "synaptic-modal-chat-progress/v1", "status": code}
            )
            + "\n"
        )
        stream.flush()

    try:
        arguments = build_parser().parse_args(argv)
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            manifest, context, settings, source = check_inputs(
                arguments.project_root, arguments.configuration, mode=arguments.mode
            )
            if arguments.mode == "check":
                emit("LOCAL_INPUTS_CHECKED")
                return 0
            if arguments.modal_profile is None:
                raise ModalChatLauncherError("modal_chat_credentials_missing")
            execute(
                manifest,
                context,
                settings,
                source,
                mode=arguments.mode,
                profile=arguments.modal_profile,
                hf_token_env_file=arguments.hf_token_env_file,
                emit=emit,
            )
        return 0
    except KeyboardInterrupt:
        stream.write(
            json.dumps(
                {"status": "INTERRUPTED", "phase": phase, "retry_authorized": False}
            )
            + "\n"
        )
        return 130
    except Exception as error:
        stream.write(
            modal_chat_failure_diagnostic(phase=phase, error=error).decode("ascii")
            + "\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
