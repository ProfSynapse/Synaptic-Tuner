"""Provider-free training inventory controls; no image/model qualification."""

from __future__ import annotations

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from test_modal_inference_runtime_capture import capture, _sdk
from test_modal_inference_runtime_inspection import inspection, _metadata, Distribution

IMAGE = "docker.io/unsloth/unsloth@sha256:" + "a" * 64
COMMIT = "b" * 40
PYTHON = "/opt/conda/bin/python3"
PIN = "2025.06"


def _inspection(monkeypatch):
    values = [
        Distribution(name, version)
        for name, version in (
            ("torch", "2.9.0+cu128"),
            ("transformers", "4.57.1"),
            ("trl", "0.24.0"),
            ("unsloth", "2026.1.2"),
            ("unsloth_zoo", "2026.1.2"),
        )
    ]
    _metadata(monkeypatch, values)
    monkeypatch.setattr(inspection.sys, "executable", PYTHON)
    monkeypatch.setattr(
        inspection, "_executable", lambda: ("/opt/conda/bin/python3.11", "c" * 64)
    )
    monkeypatch.setattr(inspection.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(inspection.platform, "system", lambda: "Linux")
    monkeypatch.setattr(inspection.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(inspection.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(
        inspection.sysconfig,
        "get_paths",
        lambda: {
            "purelib": "/opt/conda/lib/python3.11/site-packages",
            "platlib": "/opt/conda/lib/python3.11/site-packages",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.9.0+cu128", version=SimpleNamespace(cuda="12.8")
        ),
    )
    return inspection.inspect_training_base(
        image=IMAGE, source_commit=COMMIT, interpreter=PYTHON
    )


def _capture_sdk(raw):
    sdk, sandbox, calls = _sdk(raw=raw)

    def from_registry(value, **kwargs):
        calls.append(("registry_options", value, kwargs))
        from test_modal_inference_runtime_capture import Image

        return Image(calls)

    sdk.Image.from_registry = from_registry
    sdk.Image._registry_setup_commands = lambda image, pin, setup, python: [
        "FROM " + image
    ]
    sdk._image = SimpleNamespace(_get_image_builder_version=lambda server: PIN)
    return sdk, sandbox, calls


def _kwargs(sdk):
    return dict(
        sdk=sdk,
        client=object(),
        app_name="existing-app",
        environment_name="isolated",
        image=IMAGE,
        source_commit=COMMIT,
        training_base=True,
        interpreter=PYTHON,
        image_builder_version=PIN,
    )


def test_inspector_actual_metadata_shape_roundtrips_existing_profile_inventory(
    monkeypatch, tmp_path
):
    value = _inspection(monkeypatch)
    assert value["inventory"]["runtime"]["cuda_build"] == "12.8"
    assert value["python"]["executable"] == PYTHON
    assert value["python"]["physical_executable"] != PYTHON
    assert (
        capture._parse_training_candidate(
            inspection._line(value),
            image=IMAGE,
            source_commit=COMMIT,
            interpreter=PYTHON,
        )
        == value
    )
    from tuner.runtime_profiles import _load_inventory
    import hashlib

    raw = inspection._line(value["inventory"])
    path = tmp_path / "actual-shaped-inventory.json"
    path.write_bytes(raw)
    inventory, count = _load_inventory(
        path, expected_sha256="sha256:" + hashlib.sha256(raw).hexdigest(), image=IMAGE
    )
    assert count == 5 and inventory == value["inventory"]


def test_capture_exact_interpreter_pin_and_derived_image_evidence(monkeypatch):
    value = _inspection(monkeypatch)
    sdk, sandbox, calls = _capture_sdk(inspection._line(value))
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", PIN)
    result = capture.capture_modal_inference_runtime(**_kwargs(sdk))
    envelope = json.loads(result.canonical_bytes)
    assert envelope["candidate"] == value
    assert envelope["image_builder_version"] == PIN
    assert envelope["inspection_image"] == {
        "registry_base": IMAGE,
        "derived_provider_image_id": "im-candidate",
        "script_copy_only": True,
        "base_image_byte_identity_attested": False,
    }
    args, options = next(call[1:] for call in calls if call[0] == "create")
    assert args == (
        PYTHON,
        "-I",
        capture._REMOTE_SCRIPT,
        "--image",
        IMAGE,
        "--source-commit",
        COMMIT,
        "--training-base",
        "--interpreter",
        PYTHON,
    )
    assert (
        options["block_network"] is True
        and options["timeout"] == capture._SANDBOX_TIMEOUT_SECONDS
    )
    assert next(call for call in calls if call[0] == "registry_options")[2] == {
        "add_python": None,
        "setup_dockerfile_commands": [],
    }
    assert not any(call[0] == "build_commands" for call in calls)
    assert sandbox.terminate_calls == [False]


@pytest.mark.parametrize(
    "change",
    [
        {"isolated_python": True},
        {"modal_additions": True},
        {"engine_wheel": "unused"},
        {"engine_wheel_sha256": "c" * 64},
        {"check_private_directories": True},
        {"verify_runtime_lock_digest": "d" * 64},
        {"diagnose_distributions": True},
        {"interpreter": None},
        {"interpreter": "/opt/../bin/python"},
        {"image_builder_version": None},
        {"image_builder_version": "2024.10"},
        {"image": "unsloth:latest"},
        {"training_base": 1},
    ],
)
def test_invalid_training_options_refused_before_sdk(change):
    kwargs = _kwargs(SimpleNamespace())
    kwargs.update(change)
    with pytest.raises(
        capture.ModalInferenceRuntimeCaptureError, match="capture_input_invalid"
    ):
        capture.capture_modal_inference_runtime(**kwargs)


@pytest.mark.parametrize("pin", [None, "2024.10"])
def test_library_refuses_unbound_or_conflicting_builder_before_lookup(monkeypatch, pin):
    sdk, _, calls = _capture_sdk(b"unused")
    if pin is None:
        monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION", raising=False)
    else:
        monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", pin)
    with pytest.raises(
        capture.ModalInferenceRuntimeCaptureError, match="capture_input_invalid"
    ):
        capture.capture_modal_inference_runtime(**_kwargs(sdk))
    assert calls == []


def test_sdk_transforming_setup_refused_before_lookup(monkeypatch):
    sdk, _, calls = _capture_sdk(b"unused")
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", PIN)
    sdk.Image._registry_setup_commands = lambda *args: [
        "FROM " + IMAGE,
        "RUN pip install modal",
    ]
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.capture_modal_inference_runtime(**_kwargs(sdk))
    assert calls == []


@pytest.mark.parametrize(
    "mutation", ["duplicate", "changed", "count", "cuda", "python", "path", "order"]
)
def test_changed_or_malformed_capture_rejects(monkeypatch, mutation):
    value = _inspection(monkeypatch)
    if mutation == "duplicate":
        raw = inspection._line(value).replace(
            b'"status":"CANDIDATE_ONLY"',
            b'"status":"CANDIDATE_ONLY","status":"CANDIDATE_ONLY"',
        )
    else:
        changed = deepcopy(value)
        if mutation == "changed":
            changed["inventory"]["runtime"]["torch_version"] = "1.0"
        if mutation == "count":
            changed["inventory"]["distributions"] *= 103
        if mutation == "cuda":
            changed["inventory"]["runtime"]["cuda_build"] = None
        if mutation == "python":
            changed["python"]["executable"] = "/usr/bin/python3"
        if mutation == "path":
            changed["python"]["purelib"] = "/opt/../lib"
        if mutation == "order":
            changed["inventory"]["distributions"].reverse()
        raw = inspection._line(changed)
    with pytest.raises(
        capture.ModalInferenceRuntimeCaptureError, match="capture_output_invalid"
    ):
        capture._parse_training_candidate(
            raw, image=IMAGE, source_commit=COMMIT, interpreter=PYTHON
        )


def test_inspector_metadata_mutation_after_torch_import_refused(monkeypatch):
    _inspection(monkeypatch)
    values = iter(
        [
            {
                "torch": "2.9.0+cu128",
                "transformers": "4.57.1",
                "trl": "0.24.0",
                "unsloth": "2026.1.2",
                "unsloth-zoo": "2026.1.2",
            },
            {},
        ]
    )
    monkeypatch.setattr(inspection, "_distributions", lambda: next(values))
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_METADATA_READ_FAILED"
    ):
        inspection.inspect_training_base(
            image=IMAGE, source_commit=COMMIT, interpreter=PYTHON
        )


def test_cli_incompatible_mode_rejects_before_credentials_or_sdk(monkeypatch, capsys):
    monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION", raising=False)
    monkeypatch.setattr(
        capture.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(AssertionError("SDK opened")),
    )
    assert (
        capture.main(
            [
                "--app",
                "app",
                "--environment",
                "env",
                "--image",
                IMAGE,
                "--source-commit",
                COMMIT,
                "--training-base",
                "--interpreter",
                PYTHON,
                "--image-builder-version",
                PIN,
                "--read-sandbox",
                "sb-existing",
            ]
        )
        == 125
    )
    assert "MODAL_IMAGE_BUILDER_VERSION" not in capture.os.environ
    assert "capture_failed" in capsys.readouterr().err


def test_real_pinned_sdk_nontransforming_builder_commands(monkeypatch):
    modal = pytest.importorskip("modal")
    if modal.__version__ != "1.5.4":
        pytest.skip("requires exact pinned Modal1.5.4; no SDK calls")
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", PIN)
    assert modal._image._get_image_builder_version("2024.10") == PIN
    assert modal.Image._registry_setup_commands(IMAGE, PIN, [], None) == [
        "FROM " + IMAGE
    ]
    assert any(
        "pip install" in command
        for command in modal.Image._registry_setup_commands(IMAGE, "2024.10", [], None)
    )
    capture._training_builder(modal, IMAGE, PIN)


def test_cli_pin_is_set_once_before_sdk_and_not_restored(monkeypatch):
    monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION", raising=False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "fixture-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "fixture-secret")

    def unavailable(name):
        assert capture.os.environ["MODAL_IMAGE_BUILDER_VERSION"] == PIN
        return None

    monkeypatch.setattr(capture.importlib.util, "find_spec", unavailable)
    assert (
        capture.main(
            [
                "--app",
                "app",
                "--environment",
                "env",
                "--image",
                IMAGE,
                "--source-commit",
                COMMIT,
                "--training-base",
                "--interpreter",
                PYTHON,
                "--image-builder-version",
                PIN,
            ]
        )
        == 125
    )
    assert capture.os.environ["MODAL_IMAGE_BUILDER_VERSION"] == PIN


def test_capture_malformed_training_output_still_reaps_exact_sandbox(monkeypatch):
    sdk, sandbox, calls = _capture_sdk(b"{}\n")
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", PIN)
    with pytest.raises(
        capture.ModalInferenceRuntimeCaptureError, match="capture_output_invalid"
    ):
        capture.capture_modal_inference_runtime(**_kwargs(sdk))
    assert sandbox.terminate_calls == [False] and sandbox.poll_calls == 1
    assert len([call for call in calls if call[0] == "create"]) == 1


def test_training_inspector_wrong_interpreter_rejects_before_metadata(monkeypatch):
    monkeypatch.setattr(inspection.sys, "executable", "/other/python")
    monkeypatch.setattr(
        inspection.importlib.metadata,
        "distributions",
        lambda: (_ for _ in ()).throw(AssertionError("metadata read")),
    )
    with pytest.raises(inspection._InspectionFailure, match="PYTHON_INVALID"):
        inspection.inspect_training_base(
            image=IMAGE, source_commit=COMMIT, interpreter=PYTHON
        )


def test_cli_conflicting_builder_pin_rejects_without_overwrite_or_sdk(monkeypatch):
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", "2024.10")
    monkeypatch.setattr(
        capture.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(AssertionError("SDK opened")),
    )
    assert (
        capture.main(
            [
                "--app",
                "app",
                "--environment",
                "env",
                "--image",
                IMAGE,
                "--source-commit",
                COMMIT,
                "--training-base",
                "--interpreter",
                PYTHON,
                "--image-builder-version",
                PIN,
            ]
        )
        == 125
    )
    assert capture.os.environ["MODAL_IMAGE_BUILDER_VERSION"] == "2024.10"


def test_training_capture_byte_bound_rejects_before_json(monkeypatch):
    monkeypatch.setattr(capture, "_MAX_CAPTURE_BYTES", 8)
    monkeypatch.setattr(
        capture.json,
        "loads",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("parsed oversized bytes")
        ),
    )
    with pytest.raises(
        capture.ModalInferenceRuntimeCaptureError, match="capture_output_invalid"
    ):
        capture._parse_training_candidate(
            b"x" * 9, image=IMAGE, source_commit=COMMIT, interpreter=PYTHON
        )
