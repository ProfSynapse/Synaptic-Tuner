from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "scripts" / "inspect_modal_inference_runtime.py"
SPEC = importlib.util.spec_from_file_location("inspect_modal_inference_runtime", PATH)
assert SPEC is not None and SPEC.loader is not None
inspection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspection)

IMAGE = "docker.io/vllm/vllm-openai@sha256:" + "a" * 64
COMMIT = "b" * 40


def _private_tree(monkeypatch, tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    for name in ("model", "base", "scratch"):
        (root / name).mkdir(mode=0o700)
    monkeypatch.setattr(inspection, "_PRIVATE_ROOT", root)
    return root


def test_private_access_is_measured_and_leaves_no_probe_files(monkeypatch, tmp_path):
    root = _private_tree(monkeypatch, tmp_path)
    result = inspection.inspect_private_directories()
    assert result == {
        "effective_uid": os.geteuid(),
        "paths": [
            str(root),
            *(str(root / name) for name in ("model", "base", "scratch")),
        ],
        "mode": "0700",
        "read_write_verified": True,
    }
    assert sorted(p.name for p in root.iterdir()) == ["base", "model", "scratch"]
    assert all(
        not list((root / name).iterdir()) for name in ("model", "base", "scratch")
    )


@pytest.mark.parametrize("mutation", ["missing", "mode", "symlink", "owner"])
def test_private_access_rejects_invalid_directory(monkeypatch, tmp_path, mutation):
    root = _private_tree(monkeypatch, tmp_path)
    target = root / "model"
    if mutation == "missing":
        target.rmdir()
    elif mutation == "mode":
        target.chmod(0o755)
    elif mutation == "symlink":
        target.rmdir()
        target.symlink_to(root / "base", target_is_directory=True)
    else:
        current = os.geteuid()
        monkeypatch.setattr(inspection.os, "geteuid", lambda: current + 1)
    with pytest.raises(inspection._InspectionFailure, match="PRIVATE_DIRECTORY_"):
        inspection.inspect_private_directories()


def test_private_write_failure_is_closed(monkeypatch, tmp_path):
    _private_tree(monkeypatch, tmp_path)

    def deny(**kwargs):
        raise OSError("private-sensitive-error")

    monkeypatch.setattr(inspection.tempfile, "TemporaryFile", deny)
    with pytest.raises(inspection._InspectionFailure) as error:
        inspection.inspect_private_directories()
    assert str(error.value) == "PRIVATE_DIRECTORY_ACCESS_FAILED"


def _runtime_report(candidate):
    return {
        "base_registry_reference": IMAGE,
        "sdk_version": "1.5.4",
        "distributions": candidate["distributions"],
        "python_version": candidate["python"]["version"],
        "python_executable": candidate["python"]["executable"],
        "python_executable_digest": candidate["python"]["executable_sha256"],
        **{
            key: "d" * 64
            for key in (
                "runtime_lock_digest",
                "source_lock_digest",
                "dependency_lock_digest",
                "worker_closure_digest",
            )
        },
    }


def test_installed_verifier_is_called_with_independent_digest(monkeypatch):
    from tuner.execution.providers.modal import inference_runtime

    _metadata(monkeypatch)
    candidate = inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)
    calls = []

    def verify(*, expected_runtime_lock_digest):
        calls.append(expected_runtime_lock_digest)
        return _runtime_report(candidate)

    monkeypatch.setattr(
        inference_runtime, "verify_packaged_modal_inference_runtime", verify
    )
    report = inspection.verify_packaged_runtime(candidate, "d" * 64)
    assert calls == ["d" * 64]
    assert report["status"] == "PACKAGED_RUNTIME_VERIFIED"
    assert report["runtime_lock_digest"] == "d" * 64
    assert candidate["status"] == "CANDIDATE_ONLY"


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_registry_reference", "other"),
        ("sdk_version", "other"),
        ("distributions", {}),
        ("python_version", "other"),
        ("python_executable", "/other"),
        ("python_executable_digest", "a" * 64),
        ("runtime_lock_digest", "a" * 64),
        ("source_lock_digest", "other"),
    ],
)
def test_packaged_verification_must_match_observed_candidate(monkeypatch, field, value):
    from tuner.execution.providers.modal import inference_runtime

    _metadata(monkeypatch)
    candidate = inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)
    report = _runtime_report(candidate)
    report[field] = value
    monkeypatch.setattr(
        inference_runtime,
        "verify_packaged_modal_inference_runtime",
        lambda **kwargs: report,
    )
    with pytest.raises(
        inspection._InspectionFailure, match="PACKAGED_RUNTIME_VERIFICATION_FAILED"
    ):
        inspection.verify_packaged_runtime(candidate, "d" * 64)


def test_packaged_verifier_exception_is_closed(monkeypatch):
    from tuner.execution.providers.modal import inference_runtime

    def fail(**kwargs):
        raise ValueError("private-sensitive-diagnostic")

    monkeypatch.setattr(
        inference_runtime, "verify_packaged_modal_inference_runtime", fail
    )
    with pytest.raises(inspection._InspectionFailure) as error:
        inspection.verify_packaged_runtime({}, "d" * 64)
    assert str(error.value) == "PACKAGED_RUNTIME_VERIFICATION_FAILED"


def test_cli_requires_private_check_for_runtime_verification(monkeypatch, capsys):
    def unexpected(**kwargs):
        raise AssertionError("inspection must not start")

    monkeypatch.setattr(inspection, "inspect_runtime", unexpected)
    assert (
        inspection.main(
            [
                "--image",
                IMAGE,
                "--source-commit",
                COMMIT,
                "--verify-runtime-lock-digest",
                "d" * 64,
            ]
        )
        == 125
    )
    assert "ARGUMENT_INVALID" in capsys.readouterr().err


class Distribution:
    def __init__(self, name: str, version: str):
        self.metadata = {"Name": name}
        self.version = version


def _metadata(
    monkeypatch, values=(Distribution("vllm", "0.17.1"), Distribution("modal", "1.5.4"))
):
    monkeypatch.setattr(inspection.importlib.metadata, "distributions", lambda: values)
    monkeypatch.setattr(
        inspection, "_executable", lambda: ("/usr/bin/python3", "c" * 64)
    )
    monkeypatch.setattr(inspection.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(inspection.platform, "python_version", lambda: "3.12.9")


def test_candidate_is_canonical_bounded_and_records_selection(monkeypatch):
    _metadata(monkeypatch)
    value = inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)
    assert value["status"] == "CANDIDATE_ONLY"
    assert value["operator_selection"] == {"image": IMAGE, "source_commit": COMMIT}
    assert value["requirements"] == {
        "modal": {"present": True, "version": "1.5.4"},
        "vllm": {"present": True, "version": "0.17.1"},
    }
    encoded = inspection._line(value)
    assert len(encoded) <= inspection._MAX_OUTPUT_BYTES
    assert json.loads(encoded) == value


@pytest.mark.parametrize(
    "field,value", [("image", "vllm:v0.17.1"), ("source_commit", "main")]
)
def test_invalid_selection_denied_before_metadata(monkeypatch, field, value):
    monkeypatch.setattr(
        inspection.importlib.metadata,
        "distributions",
        lambda: (_ for _ in ()).throw(AssertionError("metadata")),
    )
    kwargs = {"image": IMAGE, "source_commit": COMMIT, field: value}
    with pytest.raises(inspection._InspectionFailure):
        inspection.inspect_runtime(**kwargs)


@pytest.mark.parametrize(
    "values",
    [
        (Distribution("vllm", "0.17.1"),),
        (Distribution("modal", "1.5.4"),),
    ],
)
def test_missing_requirements_are_candidate_metadata(monkeypatch, values):
    _metadata(monkeypatch, values)
    result = inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)
    assert result["requirements"] == {
        "modal": {
            "present": any(value.metadata["Name"] == "modal" for value in values),
            "version": next(
                (
                    value.version
                    for value in values
                    if value.metadata["Name"] == "modal"
                ),
                None,
            ),
        },
        "vllm": {
            "present": any(value.metadata["Name"] == "vllm" for value in values),
            "version": next(
                (value.version for value in values if value.metadata["Name"] == "vllm"),
                None,
            ),
        },
    }


def test_duplicate_distribution_still_fails(monkeypatch):
    values = (
        Distribution("vllm", "0.17.1"),
        Distribution("VLLM", "other"),
        Distribution("modal", "1.5.4"),
    )
    _metadata(monkeypatch, values)
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_IDENTITY_UNPROVEN"
    ):
        inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)


def _dist_info(root: Path, directory: str, name: str, version: str) -> None:
    metadata = root / directory
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_same_physical_metadata_from_repeated_search_root_is_accepted(
    monkeypatch, tmp_path
):
    _dist_info(tmp_path, "vllm-0.17.1.dist-info", "vllm", "0.17.1")
    _dist_info(tmp_path, "modal-1.5.4.dist-info", "modal", "1.5.4")
    values = inspection.importlib.metadata.distributions(path=[tmp_path, tmp_path])
    _metadata(monkeypatch, values)
    assert inspection._distributions() == {"modal": "1.5.4", "vllm": "0.17.1"}


def test_same_physical_metadata_from_symlink_search_root_is_accepted(
    monkeypatch, tmp_path
):
    packages = tmp_path / "packages"
    alias = tmp_path / "alias"
    _dist_info(packages, "vllm-0.17.1.dist-info", "vllm", "0.17.1")
    alias.symlink_to(packages, target_is_directory=True)
    values = inspection.importlib.metadata.distributions(path=[packages, alias])
    _metadata(monkeypatch, values)
    assert inspection._distributions() == {"vllm": "0.17.1"}


def test_distinct_same_name_version_metadata_is_rejected(monkeypatch, tmp_path):
    _dist_info(tmp_path, "demo-1.dist-info", "demo", "1")
    _dist_info(tmp_path, "demo.other-1.dist-info", "demo", "1")
    values = inspection.importlib.metadata.distributions(path=[tmp_path])
    _metadata(monkeypatch, values)
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_PHYSICAL_DUPLICATE"
    ):
        inspection._distributions()


def test_custom_distributions_cannot_claim_physical_alias(monkeypatch):
    _metadata(monkeypatch, (Distribution("demo", "1"), Distribution("demo", "1")))
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_IDENTITY_UNPROVEN"
    ):
        inspection._distributions()


def test_same_physical_identity_with_different_version_is_rejected(monkeypatch):
    secret = "credential-like-private-version"
    values = (Distribution("demo", "1"), Distribution("demo", secret))
    _metadata(monkeypatch, values)
    monkeypatch.setattr(inspection, "_metadata_identity", lambda value: (1,))
    with pytest.raises(
        inspection._InspectionFailure,
        match="DISTRIBUTION_PHYSICAL_METADATA_MISMATCH",
    ) as caught:
        inspection._distributions()
    assert secret not in str(caught.value)


def test_same_physical_identity_with_different_name_is_rejected(monkeypatch):
    values = (Distribution("first", "1"), Distribution("second", "1"))
    _metadata(monkeypatch, values)
    monkeypatch.setattr(inspection, "_metadata_identity", lambda value: (1,))
    with pytest.raises(
        inspection._InspectionFailure,
        match="DISTRIBUTION_PHYSICAL_METADATA_MISMATCH",
    ):
        inspection._distributions()


@pytest.mark.parametrize(
    "identities,reason",
    (
        ((None, None, None, None), "DISTRIBUTION_IDENTITY_UNPROVEN"),
        (((1,), (1,), (2,), (2,)), "DISTRIBUTION_PHYSICAL_DUPLICATE"),
        (
            ((1,), (1,), (1,), (1,)),
            "DISTRIBUTION_PHYSICAL_METADATA_MISMATCH",
        ),
    ),
)
def test_distribution_collision_diagnostics_do_not_leak_metadata(
    monkeypatch, capsys, identities, reason
):
    secret_name = "private-tokenlike-name"
    secret_version = "private-tokenlike-version"
    values = (
        Distribution(secret_name, "1"),
        Distribution(secret_name, secret_version),
    )
    _metadata(monkeypatch, values)
    observed = iter(identities)
    monkeypatch.setattr(inspection, "_metadata_identity", lambda value: next(observed))
    assert inspection.main(["--image", IMAGE, "--source-commit", COMMIT]) == 125
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": reason,
        "schema_version": inspection._ERROR_SCHEMA,
        "status": "FAILED",
    }
    rendered = captured.err
    assert secret_name not in rendered
    assert secret_version not in rendered


def test_unstable_metadata_identity_cannot_be_deduplicated(monkeypatch):
    values = (Distribution("demo", "1"), Distribution("demo", "1"))
    _metadata(monkeypatch, values)
    identities = iter(((1,), (1,), (2,), (3,)))
    monkeypatch.setattr(
        inspection, "_metadata_identity", lambda value: next(identities)
    )
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_METADATA_READ_FAILED"
    ):
        inspection._distributions()


def test_unstable_unique_metadata_identity_is_rejected(monkeypatch):
    _metadata(monkeypatch, (Distribution("demo", "1"),))
    identities = iter(((1,), (2,)))
    monkeypatch.setattr(
        inspection, "_metadata_identity", lambda value: next(identities)
    )
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_METADATA_READ_FAILED"
    ):
        inspection._distributions()


def test_distribution_count_is_bounded(monkeypatch):
    values = [Distribution(f"package-{index}", "1") for index in range(513)]
    values[:2] = [Distribution("vllm", "0.17.1"), Distribution("modal", "1.5.4")]
    _metadata(monkeypatch, values)
    with pytest.raises(inspection._InspectionFailure, match="DISTRIBUTION_COUNT_LIMIT"):
        inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)


def test_distribution_occurrence_count_is_separately_bounded(monkeypatch):
    value = Distribution("demo", "1")
    monkeypatch.setattr(
        inspection.importlib.metadata,
        "distributions",
        lambda: (value for _ in range(inspection._MAX_DISTRIBUTION_OCCURRENCES + 1)),
    )
    monkeypatch.setattr(inspection, "_metadata_identity", lambda distribution: (1,))
    with pytest.raises(inspection._InspectionFailure, match="DISTRIBUTION_COUNT_LIMIT"):
        inspection._distributions()


def test_distribution_diagnostic_reports_distinct_physical_installs(
    monkeypatch, tmp_path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _dist_info(first, "demo-1.dist-info", "Demo_Name", "1+cpu")
    _dist_info(second, "demo-1.dist-info", "demo-name", "1+cpu")
    values = inspection.importlib.metadata.distributions(path=[first, second, first])
    _metadata(monkeypatch, values)
    result = inspection.diagnose_distributions(image=IMAGE, source_commit=COMMIT)
    assert result == {
        "distributions": [
            {
                "metadata_path": (first / "demo-1.dist-info").as_posix(),
                "name": "demo-name",
                "version": "1+cpu",
            },
            {
                "metadata_path": (second / "demo-1.dist-info").as_posix(),
                "name": "demo-name",
                "version": "1+cpu",
            },
        ],
        "operator_selection": {"image": IMAGE, "source_commit": COMMIT},
        "schema_version": inspection._DIAGNOSTIC_SCHEMA,
        "status": "DIAGNOSTIC_ONLY",
    }


def test_distribution_diagnostic_does_not_relax_normal_inspection(
    monkeypatch, tmp_path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _dist_info(first, "demo-1.dist-info", "demo", "1")
    _dist_info(second, "demo-1.dist-info", "demo", "1")
    values = list(inspection.importlib.metadata.distributions(path=[first, second]))
    _metadata(monkeypatch, values)
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_PHYSICAL_DUPLICATE"
    ):
        inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)


@pytest.mark.parametrize(
    "name,version,reason",
    (
        ("bad name", "1", "DISTRIBUTION_NAME_INVALID"),
        ("demo", "https://private.invalid", "DISTRIBUTION_VERSION_INVALID"),
    ),
)
def test_distribution_diagnostic_rejects_unsafe_metadata(
    monkeypatch, name, version, reason
):
    _metadata(monkeypatch, (Distribution(name, version),))
    with pytest.raises(inspection._InspectionFailure, match=reason):
        inspection.diagnose_distributions(image=IMAGE, source_commit=COMMIT)


def test_distribution_diagnostic_rejects_unbounded_path(monkeypatch, tmp_path):
    _dist_info(tmp_path, "demo-1.dist-info", "demo", "1")
    values = inspection.importlib.metadata.distributions(path=[tmp_path])
    _metadata(monkeypatch, values)
    original = inspection._bounded_text

    def bounded(value, *, maximum):
        if maximum == 4096:
            raise inspection._InspectionFailure("METADATA_INVALID")
        return original(value, maximum=maximum)

    monkeypatch.setattr(inspection, "_bounded_text", bounded)
    with pytest.raises(inspection._InspectionFailure, match="METADATA_INVALID"):
        inspection.diagnose_distributions(image=IMAGE, source_commit=COMMIT)


def test_distribution_diagnostic_reports_null_for_unproven_path(monkeypatch):
    _metadata(monkeypatch, (Distribution("demo", "1"),))
    result = inspection.diagnose_distributions(image=IMAGE, source_commit=COMMIT)
    assert result["distributions"] == [
        {"metadata_path": None, "name": "demo", "version": "1"}
    ]


def test_distribution_diagnostic_unique_name_count_is_bounded(monkeypatch):
    values = [
        Distribution(f"package-{index}", "1")
        for index in range(inspection._MAX_DISTRIBUTIONS + 1)
    ]
    _metadata(monkeypatch, values)
    with pytest.raises(inspection._InspectionFailure, match="DISTRIBUTION_COUNT_LIMIT"):
        inspection.diagnose_distributions(image=IMAGE, source_commit=COMMIT)


def test_distribution_diagnostic_occurrence_count_is_bounded(monkeypatch):
    value = Distribution("demo", "1")
    _metadata(
        monkeypatch,
        (value for _ in range(inspection._MAX_DISTRIBUTION_OCCURRENCES + 1)),
    )
    with pytest.raises(inspection._InspectionFailure, match="DISTRIBUTION_COUNT_LIMIT"):
        inspection.diagnose_distributions(image=IMAGE, source_commit=COMMIT)


def test_distribution_diagnostic_main_is_closed_on_private_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        inspection,
        "diagnose_distributions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private-token-value")),
    )
    assert (
        inspection.main(
            [
                "--image",
                IMAGE,
                "--source-commit",
                COMMIT,
                "--diagnose-distributions",
            ]
        )
        == 125
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": "INSPECTION_FAILED",
        "schema_version": inspection._ERROR_SCHEMA,
        "status": "FAILED",
    }
    assert "private-token-value" not in captured.err


@pytest.mark.parametrize(
    "values,reason",
    (
        ((Distribution("", "1"),), "DISTRIBUTION_NAME_INVALID"),
        ((Distribution("package", ""),), "DISTRIBUTION_VERSION_INVALID"),
    ),
)
def test_invalid_distribution_fields_have_closed_granularity(
    monkeypatch, values, reason
):
    _metadata(monkeypatch, values)
    with pytest.raises(inspection._InspectionFailure, match=reason):
        inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)


def test_distribution_enumeration_failure_is_closed(monkeypatch):
    _metadata(monkeypatch)
    monkeypatch.setattr(
        inspection.importlib.metadata,
        "distributions",
        lambda: (_ for _ in ()).throw(RuntimeError("private-enumeration")),
    )
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_ENUMERATION_FAILED"
    ) as caught:
        inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)
    assert "private" not in str(caught.value)


def test_distribution_metadata_read_failure_is_closed(monkeypatch):
    class Unreadable:
        @property
        def metadata(self):
            raise RuntimeError("private-metadata")

    _metadata(monkeypatch, (Unreadable(),))
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_METADATA_READ_FAILED"
    ) as caught:
        inspection.inspect_runtime(image=IMAGE, source_commit=COMMIT)
    assert "private" not in str(caught.value)


def test_executable_records_canonical_symlink_target(monkeypatch, tmp_path):
    target = tmp_path / "python"
    target.write_bytes(b"python")
    link = tmp_path / "python-link"
    link.symlink_to(target)
    monkeypatch.setattr(inspection.sys, "executable", str(link))
    executable, digest = inspection._executable()
    assert executable == target.as_posix()
    assert digest == inspection.hashlib.sha256(b"python").hexdigest()


def test_main_closes_unexpected_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        inspection,
        "inspect_runtime",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )
    assert inspection.main(["--image", IMAGE, "--source-commit", COMMIT]) == 125
    captured = capsys.readouterr()
    assert "private detail" not in captured.err
    assert json.loads(captured.err)["reason_code"] == "INSPECTION_FAILED"


def test_main_preserves_control_flow(monkeypatch):
    monkeypatch.setattr(
        inspection,
        "inspect_runtime",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        inspection.main(["--image", IMAGE, "--source-commit", COMMIT])


def test_real_interpreter_hash_has_expected_shape():
    executable, digest = inspection._executable()
    assert executable == Path(inspection.sys.executable).resolve(strict=True).as_posix()
    assert len(digest) == 64
