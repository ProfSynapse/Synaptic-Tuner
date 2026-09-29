"""Installed package-owned Modal entrypoint for signed packaged training."""

from __future__ import annotations

from pathlib import Path


_CONTROL_ROOT = Path("/mnt/control")
_ARTIFACT_ROOT = Path("/mnt/artifacts")
_MODEL_CACHE_ROOT = Path("/mnt/model-cache")
_PRIVATE_SCRATCH_ROOT = Path("/tmp")


def run_modal_packaged_training(dispatch_bytes: bytes) -> dict[str, object]:
    """Authenticate one dispatch and execute it in the captured image."""
    stage = ["ENTRYPOINT_IMPORTS"]
    failed = {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": "ENTRYPOINT_SETUP",
    }
    try:
        import modal

        stage[0] = "ENTRYPOINT_SETUP"
        return _run_with_modal(dispatch_bytes, sdk=modal, _stage=stage)
    except BaseException:
        failed["failure_stage"] = stage[0]
        return failed


def _run_with_modal(dispatch_bytes: bytes, *, sdk: object,
                    _stage: list[str] | None = None) -> dict[str, object]:
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_IMPORTS"
    import base64
    import binascii
    import os
    import tempfile
    from contextlib import ExitStack

    from tuner.execution.providers.modal.facade import EXACT_MODAL_SDK_VERSION
    from tuner.execution.providers.modal.model_snapshot import (
        MODEL_SNAPSHOT_PREPARATION_STAGES,
        PERSISTENT_PUBLICATION_DIAGNOSTICS,
        ModelSnapshotPreparationError,
        prepare_model_snapshot,
    )
    from tuner.runtime.packaged_sft_execution import PackagedPreparationError
    from tuner.execution.providers.modal.packaged_dispatch import (
        MODAL_PACKAGED_DISPATCH_V2_SCHEMA, parse_modal_packaged_dispatch,
    )
    from tuner.execution.providers.modal.packaged_worker import (
        InstalledPackagedSFTTrainerExecutor,
        ModalPackagedWorker,
        ModalPackagedWorkerRoots,
    )
    from tuner.execution.providers.modal.runtime_release_qualification import (
        QUALIFICATION_HMAC_ENV_KEY,
        ModalRuntimeQualificationHmacAuthenticator,
    )
    from tuner.execution.providers.modal.volume_root_binding import (
        PUBLICATION_DIAGNOSTICS, VolumeRootBinding, VolumeRootBindingError,
    )

    class _ModelPublisher:
        """Translate only binding-owned finite publication failures."""

        def __init__(self, binding):
            self._binding = binding

        def _call(self, operation, *args, **kwargs):
            try:
                return operation(*args, **kwargs)
            except VolumeRootBindingError as error:
                if (type(error) is VolumeRootBindingError
                        and error.code in PUBLICATION_DIAGNOSTICS
                        and error.code in PERSISTENT_PUBLICATION_DIAGNOSTICS):
                    raise ModelSnapshotPreparationError(
                        "PERSISTENT_PUBLICATION_" + error.code
                    ) from None
                raise

        def claim_directory(self, path):
            return self._call(self._binding.claim_directory, path)

        def copy_in_exclusive(self, relative_path, source_path, *, expected_size,
                              expected_sha256, maximum):
            return self._call(
                self._binding.copy_in_exclusive, relative_path, source_path,
                expected_size=expected_size, expected_sha256=expected_sha256,
                maximum=maximum,
            )

    if _stage is not None:
        _stage[0] = "ENTRYPOINT_DISPATCH_AUTH"
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
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_PROVIDER_ID"
    if (getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION
            or os.environ.get("MODAL_IS_REMOTE") != "1"
            or os.environ.get("MODAL_ENVIRONMENT") != facts.environment_ref
            or os.environ.get("MODAL_IMAGE_ID") != facts.image_id):
        raise ValueError("packaged training provider identity differs")
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_VOLUME_ID"
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
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_CALL_ID"
    call_id = sdk.current_function_call_id()
    if type(call_id) is not str or not call_id.startswith("fc-") or len(call_id) > 80:
        raise ValueError("packaged training call identity is unavailable")

    control = _CONTROL_ROOT
    artifacts = _ARTIFACT_ROOT
    cache = _MODEL_CACHE_ROOT
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_MOUNTS"
    for role, root in (("CONTROL", control), ("ARTIFACTS", artifacts),
                       ("MODEL_CACHE", cache)):
        if _stage is not None:
            _stage[0] = "ENTRYPOINT_MOUNT_" + role + "_DIR"
        if not root.is_dir():
            raise ValueError("packaged training mounts differ")
    if getattr(dispatch, "schema_version", None) != MODAL_PACKAGED_DISPATCH_V2_SCHEMA:
        for role, root in (("CONTROL", control), ("ARTIFACTS", artifacts),
                           ("MODEL_CACHE", cache)):
            if _stage is not None:
                _stage[0] = "ENTRYPOINT_MOUNT_" + role + "_LINK"
            if root.is_symlink():
                raise ValueError("packaged training mounts differ")
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_MOUNTS"
    if len({control, artifacts, cache}) != 3:
        raise ValueError("packaged training mounts are not distinct")
    token = os.environ.get("HF_TOKEN")
    credential = token if type(token) is str and token.strip() else None
    if _stage is not None:
        _stage[0] = "ENTRYPOINT_WORKER_SETUP"
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="synaptic-model-", dir=_PRIVATE_SCRATCH_ROOT) as temporary:
        scratch = Path(temporary)
        bindings = None
        if getattr(dispatch, "schema_version", None) == MODAL_PACKAGED_DISPATCH_V2_SCHEMA:
            if _stage is not None:
                _stage[0] = "ENTRYPOINT_MOUNTS"
            roots = {"control": control, "artifacts": artifacts, "model_cache": cache}
            expected = {"control": facts.control_volume_id,
                        "artifacts": facts.artifact_volume_id,
                        "model_cache": facts.model_cache_volume_id}
            bindings = {}
            for marker in dispatch.volume_markers:
                if marker.volume_id != expected[marker.role]:
                    raise ValueError("packaged training marker identity differs")
                bindings[marker.role] = stack.enter_context(VolumeRootBinding.bind(
                    root_path=str(roots[marker.role]), volume_id=marker.volume_id,
                    marker_name=marker.marker_name,
                    marker_sha256=marker.value_sha256,
                ))
            if set(bindings) != set(expected):
                raise ValueError("packaged training marker roles differ")
        persistent = scratch / "persistent-cache"
        persistent.mkdir(mode=0o700)

        def prepare(model: dict[str, object], destination: Path) -> Path:
            args = dict(model_ref=model["ref"], revision=model["revision"], token=credential,
                        persistent_root=persistent if bindings is not None else cache,
                        destination_root=destination, scratch_root=scratch)
            if bindings is not None:
                args["persistent_binding"] = _ModelPublisher(bindings["model_cache"])
            try:
                snapshot = prepare_model_snapshot(**args)
            except ModelSnapshotPreparationError as error:
                suffix = (error.stage if type(error) is ModelSnapshotPreparationError
                          and error.stage in MODEL_SNAPSHOT_PREPARATION_STAGES else None)
                raise PackagedPreparationError("MODEL_" + suffix if suffix is not None
                                               else "MODEL_UNAVAILABLE") from None
            # Persist verified reusable files before the credential-free child.
            try:
                handles["model_cache"].commit()
            except Exception:
                raise PackagedPreparationError("CACHE_COMMIT") from None
            return snapshot

        worker = ModalPackagedWorker(
            expected_facts=facts,
            dispatch_verifier=authenticator,
            trainer_executor=InstalledPackagedSFTTrainerExecutor(model_preparer=prepare),
            evidence_signer=authenticator,
            roots=ModalPackagedWorkerRoots(control, artifacts, cache),
            **({"volume_bindings": bindings, "private_root": scratch}
               if bindings is not None else {}),
        )
        if _stage is not None:
            _stage[0] = "ENTRYPOINT_SETUP"
        return worker(
            dispatch_bytes, call_id,
            commit_artifacts=handles["artifacts"].commit,
            commit_control=handles["control"].commit,
        )


__all__ = ["run_modal_packaged_training"]
