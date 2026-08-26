from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tuner.execution.providers.modal.config import ModalProviderProfileV1, ModalRuntimeLockV1


ROOT=Path(__file__).resolve().parents[3]


def test_checked_in_modal_profile_and_packaged_runtime_lock_resolve_strictly():
    profile=ModalProviderProfileV1.from_mapping(yaml.safe_load((ROOT/"examples/host-project/providers/modal-a10-v1.yaml").read_text(encoding="utf-8")))
    runtime=ModalRuntimeLockV1.packaged()
    assert profile.app_name=="synaptic-training-v1" and profile.function_name=="run_sft_v1"
    assert runtime.registry_reference.endswith("5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39")
    assert len(runtime.locked_digest("deployment_wrapper"))==64


def test_modal_profile_rejects_unknown_fields_and_ambient_secret_values():
    with pytest.raises(ValueError,match="unknown"):
        ModalProviderProfileV1.from_mapping({"schema_version":"synaptic-modal-provider/v1","profile":"p","deployment":{},"runtime_lock":"x","volumes":{},"secrets":[],"token":"secret"})


def test_runtime_lock_is_deeply_immutable_and_detached_from_input():
    document = ModalRuntimeLockV1.packaged().to_dict()
    runtime = ModalRuntimeLockV1(document)
    document["python"]["version"] = "9.9.9"
    document["locked_files"]["sft_runtime"]["sha256"] = "0" * 64
    assert runtime.python_version == "3.11.14"
    assert runtime.locked_digest("sft_runtime") != "0" * 64
    with pytest.raises(TypeError):
        runtime.document["python"]["version"] = "9.9.9"
