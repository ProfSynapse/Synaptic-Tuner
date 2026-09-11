from __future__ import annotations

import importlib.util
import json
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
        inspection._InspectionFailure, match="DISTRIBUTION_IDENTITY_DUPLICATE"
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
        inspection._InspectionFailure, match="DISTRIBUTION_IDENTITY_DUPLICATE"
    ):
        inspection._distributions()


def test_custom_distributions_cannot_claim_physical_alias(monkeypatch):
    _metadata(monkeypatch, (Distribution("demo", "1"), Distribution("demo", "1")))
    with pytest.raises(
        inspection._InspectionFailure, match="DISTRIBUTION_IDENTITY_DUPLICATE"
    ):
        inspection._distributions()


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
