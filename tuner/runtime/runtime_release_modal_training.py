"""Installed package-owned Modal entrypoint for signed packaged training."""

from __future__ import annotations

from pathlib import Path


_CONTROL_ROOT = Path("/mnt/control")
_ARTIFACT_ROOT = Path("/mnt/artifacts")
_MODEL_CACHE_ROOT = Path("/mnt/model-cache")
_PRIVATE_SCRATCH_ROOT = Path("/tmp")


def run_modal_packaged_training(dispatch_bytes: bytes) -> dict[str, object]:
    """Authenticate one dispatch and execute it in the captured image."""
    failed = {
        "schema_version": "synaptic-modal-packaged-worker-result/v1",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
    }
    try:
        import modal

        return _run_with_modal(dispatch_bytes, sdk=modal)
    except BaseException:
        return failed


def _run_with_modal(dispatch_bytes: bytes, *, sdk: object) -> dict[str, object]:
    import base64
    import binascii
    import os
    import tempfile

    from tuner.execution.providers.modal.facade import EXACT_MODAL_SDK_VERSION
    from tuner.execution.providers.modal.model_snapshot import prepare_model_snapshot
    from tuner.execution.providers.modal.packaged_dispatch import parse_modal_packaged_dispatch
    from tuner.execution.providers.modal.packaged_worker import (
        InstalledPackagedSFTTrainerExecutor,
        ModalPackagedWorker,
        ModalPackagedWorkerRoots,
    )
    from tuner.execution.providers.modal.runtime_release_qualification import (
        QUALIFICATION_HMAC_ENV_KEY,
        ModalRuntimeQualificationHmacAuthenticator,
    )

    if type(dispatch_bytes) is not bytes or not dispatch_bytes:
        raise ValueError("packaged training dispatch is invalid")
    encoded_key = os.environ.get(QUALIFICATION_HMAC_ENV_KEY)
    if type(encoded_key) is not str or not encoded_key or not encoded_key.isascii():
        raise ValueError("packaged training authority is unavailable")
    try:
        key = base64.b64decode(encoded_key, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("packaged training authority is invalid") from None
    if len(key) != 32 or base64.b64encode(key).decode("ascii") != encoded_key:
        raise ValueError("packaged training authority is invalid")
    authenticator = ModalRuntimeQualificationHmacAuthenticator(key)
    dispatch = parse_modal_packaged_dispatch(dispatch_bytes, authenticator)
    facts = dispatch.provider_facts
    if (getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION
            or os.environ.get("MODAL_IS_REMOTE") != "1"
            or os.environ.get("MODAL_ENVIRONMENT") != facts.environment_ref
            or os.environ.get("MODAL_IMAGE_ID") != facts.image_id):
        raise ValueError("packaged training provider identity differs")
    client = sdk.Client.from_env()
    handles = {}
    for role, identity in (("control", facts.control_volume_id),
                           ("artifacts", facts.artifact_volume_id),
                           ("model_cache", facts.model_cache_volume_id)):
        handle = sdk.Volume.from_id(identity, client=client)
        handle.hydrate(client)
        if getattr(handle, "is_hydrated", False) is not True or getattr(handle, "object_id", None) != identity:
            raise ValueError("packaged training Volume identity differs")
        handles[role] = handle
    call_id = sdk.current_function_call_id()
    if type(call_id) is not str or not call_id.startswith("fc-") or len(call_id) > 80:
        raise ValueError("packaged training call identity is unavailable")

    control = _CONTROL_ROOT
    artifacts = _ARTIFACT_ROOT
    cache = _MODEL_CACHE_ROOT
    if (not control.is_dir() or not artifacts.is_dir() or not cache.is_dir()
            or control.is_symlink() or artifacts.is_symlink() or cache.is_symlink()):
        raise ValueError("packaged training mounts differ")
    if len({control, artifacts, cache}) != 3:
        raise ValueError("packaged training mounts are not distinct")
    token = os.environ.get("HF_TOKEN")
    credential = token if type(token) is str and token.strip() else None
    with tempfile.TemporaryDirectory(prefix="synaptic-model-", dir=_PRIVATE_SCRATCH_ROOT) as temporary:
        scratch = Path(temporary)

        def prepare(model: dict[str, object], destination: Path) -> Path:
            snapshot = prepare_model_snapshot(
                model_ref=model["ref"], revision=model["revision"], token=credential,
                persistent_root=cache, destination_root=destination,
                scratch_root=scratch,
            )
            # Persist verified reusable files before the credential-free child.
            handles["model_cache"].commit()
            return snapshot

        worker = ModalPackagedWorker(
            expected_facts=facts,
            dispatch_verifier=authenticator,
            trainer_executor=InstalledPackagedSFTTrainerExecutor(model_preparer=prepare),
            evidence_signer=authenticator,
            roots=ModalPackagedWorkerRoots(control, artifacts, cache),
        )
        return worker(
            dispatch_bytes, call_id,
            commit_artifacts=handles["artifacts"].commit,
            commit_control=handles["control"].commit,
        )


__all__ = ["run_modal_packaged_training"]
