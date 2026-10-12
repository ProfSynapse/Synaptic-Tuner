"""Restricted consumer admission and evidence, never a production logs switch."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import json
import stat
import subprocess

from synaptic_tuner.api.v1._contract import canonical_bytes as observation_bytes
from synaptic_tuner.api.v1.observations import ObservationKind
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    parse_canonical_object,
    safe_ref,
)
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload,
    packaged_configuration_digest,
)

ADMISSION_SCHEMA = "synaptic-modal-metrics-qualification-admission/v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_ref",
        "engine_commit",
        "profile_sha256",
        "runtime_material_intent_digest",
        "workload_digest",
        "configuration_digest",
        "dataset_digest",
        "dataset_content_digest",
        "modal_profile",
        "modal_environment",
        "account_ref",
        "workspace_ref",
        "accelerator",
        "accelerator_count",
        "timeout_seconds",
        "maximum_cost_minor_units",
        "max_steps",
        "capture_max_records",
    }
)


class MetricsQualificationUnavailable(RuntimeError):
    """Closed admission/evidence diagnostic; never carries rejected input."""

    def __init__(self):
        super().__init__("modal_metrics_qualification_unavailable")


def current_clean_commit(root: Path) -> str:
    try:
        if root.resolve(strict=True) != Path(__file__).resolve().parents[2]:
            raise ValueError
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        commit = head.stdout.decode("ascii").strip()
        if (
            head.returncode
            or status.returncode
            or status.stdout
            or re.fullmatch("[0-9a-f]{40}", commit) is None
        ):
            raise ValueError
        return commit
    except Exception:
        raise MetricsQualificationUnavailable() from None


def candidate_admission(
    plan,
    *,
    engine_commit,
    attempt_ref,
    modal_profile,
    modal_environment,
    account_ref,
    workspace_ref,
):
    recipe = plan.recipe
    capture = recipe.observation_capture
    steps = recipe.hyperparameters.duration.max_steps
    if (
        capture is None
        or capture["every_n_steps"] != 1
        or type(capture["max_records"]) is not int
        or not 2 <= capture["max_records"] <= 32
        or type(steps) is not int
        or not 2 <= steps <= 32
        or recipe.accelerator_count != 1
        or recipe.timeout_seconds > 900
        or recipe.post_training is not None
        or type(recipe.maximum_cost_minor_units) is not int
        or not 1 <= recipe.maximum_cost_minor_units <= 500
    ):
        raise MetricsQualificationUnavailable()
    configuration = recipe.packaged_config(plan.prepared_identity)
    if (
        compile_packaged_sft_workload(resolved_config=configuration).fingerprint
        != plan.workload_digest
        or packaged_configuration_digest(configuration) != plan.configuration_digest
    ):
        raise MetricsQualificationUnavailable()
    document = dict(
        schema_version=ADMISSION_SCHEMA,
        attempt_ref=attempt_ref,
        engine_commit=engine_commit,
        profile_sha256=plan.profile_sha256.removeprefix("sha256:"),
        runtime_material_intent_digest=plan.runtime_material_intent_digest.removeprefix(
            "sha256:"
        ),
        workload_digest=plan.workload_digest.removeprefix("sha256:"),
        configuration_digest=plan.configuration_digest.removeprefix("sha256:"),
        dataset_digest=plan.prepared_identity.revision.removeprefix("sha256:"),
        dataset_content_digest=plan.prepared_identity.content_digest.removeprefix(
            "sha256:"
        ),
        modal_profile=modal_profile,
        modal_environment=modal_environment,
        account_ref=account_ref,
        workspace_ref=workspace_ref,
        accelerator=recipe.accelerator,
        accelerator_count=recipe.accelerator_count,
        timeout_seconds=recipe.timeout_seconds,
        maximum_cost_minor_units=recipe.maximum_cost_minor_units,
        max_steps=steps,
        capture_max_records=capture["max_records"],
    )
    _validate(document)
    return canonical_bytes(document)


def _validate(document):
    if type(document) is not dict or frozenset(document) != _FIELDS:
        raise MetricsQualificationUnavailable()
    if document["schema_version"] != ADMISSION_SCHEMA:
        raise MetricsQualificationUnavailable()
    for name in (
        "attempt_ref",
        "modal_profile",
        "modal_environment",
        "account_ref",
        "workspace_ref",
        "accelerator",
    ):
        if type(document[name]) is not str:
            raise MetricsQualificationUnavailable()
        safe_ref(document[name], name)
    safe_ref("metrics-qual-" + document["attempt_ref"], "attempt_claim_ref")
    if (
        type(document["engine_commit"]) is not str
        or re.fullmatch("[0-9a-f]{40}", document["engine_commit"]) is None
    ):
        raise MetricsQualificationUnavailable()
    for name in (
        "profile_sha256",
        "runtime_material_intent_digest",
        "workload_digest",
        "configuration_digest",
        "dataset_digest",
        "dataset_content_digest",
    ):
        if type(document[name]) is not str:
            raise MetricsQualificationUnavailable()
        digest_text(document[name], name)
    for name, low, high in (
        ("accelerator_count", 1, 1),
        ("timeout_seconds", 1, 900),
        ("maximum_cost_minor_units", 1, 500),
        ("max_steps", 2, 32),
        ("capture_max_records", 2, 32),
    ):
        if type(document[name]) is not int or not low <= document[name] <= high:
            raise MetricsQualificationUnavailable()


@dataclass(frozen=True, slots=True)
class MetricsQualificationAdmission:
    canonical: bytes
    sha256: str

    def __post_init__(self):
        try:
            document = parse_canonical_object(self.canonical, name="metrics admission")
            _validate(document)
            if hashlib.sha256(self.canonical).hexdigest() != self.sha256:
                raise ValueError
        except Exception:
            raise MetricsQualificationUnavailable() from None

    @property
    def document(self):
        return parse_canonical_object(self.canonical, name="metrics admission")

    def validate_plan(self, plan, *, context, modal_profile, modal_environment):
        d = self.document
        expected = candidate_admission(
            plan,
            engine_commit=current_clean_commit(context.engine_root),
            attempt_ref=d["attempt_ref"],
            modal_profile=modal_profile,
            modal_environment=modal_environment,
            account_ref=d["account_ref"],
            workspace_ref=d["workspace_ref"],
        )
        if expected != self.canonical:
            raise MetricsQualificationUnavailable()

    def validate_scope(self, binding):
        d = self.document
        if (
            binding.account_ref != d["account_ref"]
            or binding.workspace_ref != d["workspace_ref"]
            or binding.environment_ref != d["modal_environment"]
        ):
            raise MetricsQualificationUnavailable()


def load_admission(path: Path, expected_sha256: str) -> MetricsQualificationAdmission:
    from tuner.runtime.packaged_worker_closure import stable_read

    try:
        return MetricsQualificationAdmission(stable_read(path, 8192), expected_sha256)
    except Exception:
        raise MetricsQualificationUnavailable() from None


def _create_private(path: Path):
    """Retain no-follow ancestors; exclusive leaf starts at mode 0600."""
    path = Path(os.path.abspath(path))
    parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            os.close(parent)
            parent = child
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            os.fsync(parent)
        except Exception:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(parent)


def write_candidate(path: Path, raw: bytes):
    """Create exact reviewed bytes, no execution claim or credentials."""
    with os.fdopen(_create_private(path), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class MetricsQualificationLedger:
    """Finite private ledger. A delivery proof never qualifies partial-write transport."""

    MAX_POLLS = 512
    MAX_BYTES = 1024 * 1024

    def __init__(self, admission, run, provider_job_ref, root):
        self.admission, self.run, self.provider_job_ref = (
            admission,
            run,
            provider_job_ref,
        )
        self.path = root / ("metrics-qualification-" + admission.sha256 + ".jsonl")
        self.final_path = root / (
            "metrics-qualification-" + admission.sha256 + "-result.json"
        )
        self._fd = None
        self._bytes = self._polls = 0
        self._ledger_digest = hashlib.sha256()
        self._ledger_identity = None
        self._proven = set()
        self._invalid = False
        self._read_failed = False
        self._previous_sequences = set()
        self._retained_compared = False
        self._retained_sha256 = None
        self.finished = False
        self._result = None

    def record(
        self, consumer, *, before_state, after_state, receipt_time, requested_cursor
    ):
        try:
            if self._polls >= self.MAX_POLLS:
                raise ValueError
            records = []
            for sequence, raw in sorted(consumer._seen.items()):
                if sequence not in self._previous_sequences:
                    d = json.loads(raw)
                    records.append(
                        {
                            "sequence": sequence,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "numeric": d["kind"]
                            == ObservationKind.TRAINING_METRIC_OBSERVED.value,
                        }
                    )
            if consumer.disabled:
                self._invalid = True
            if consumer.status == "progress_read_unavailable":
                self._read_failed = True
            intact = not consumer.disabled and consumer.status == "progress_retained"
            if intact and before_state == after_state == "pending":
                self._proven.update(r["sequence"] for r in records if r["numeric"])
            self._previous_sequences.update(r["sequence"] for r in records)
            document = {
                "schema_version": "synaptic-modal-metrics-qualification-poll/v1",
                "admission_sha256": self.admission.sha256,
                "run": self.run.to_dict(),
                "provider_job_ref": self.provider_job_ref,
                "receipt_time": receipt_time,
                "requested_cursor": requested_cursor,
                "retained_cursor": consumer.cursor,
                "before_state": before_state,
                "after_state": after_state,
                "status": consumer.status,
                "records": records,
            }
            raw = observation_bytes(document) + b"\n"
            if self._bytes + len(raw) > self.MAX_BYTES:
                raise ValueError
            if self._fd is None:
                self._fd = _create_private(self.path)
                info = os.fstat(self._fd)
                self._ledger_identity = (info.st_dev, info.st_ino)
            remaining = memoryview(raw)
            while remaining:
                n = os.write(self._fd, remaining)
                if n <= 0:
                    raise OSError
                remaining = remaining[n:]
            os.fsync(self._fd)
            self._ledger_digest.update(raw)
            self._polls += 1
            self._bytes += len(raw)
        except Exception:
            self._invalid = True

    def compare_retained(self, consumer):
        """Compare every accepted public canonical record with the retained file."""
        from tuner.runtime.packaged_worker_closure import stable_read

        try:
            expected = b"".join(
                raw + b"\n" for _, raw in sorted(consumer._seen.items())
            )
            if consumer._created:
                info = consumer.path.lstat()
                if (info.st_dev, info.st_ino) != consumer._file_identity:
                    raise ValueError
                actual = stable_read(consumer.path, consumer.MAX_BYTES)
                # Files preserve first acceptance order, which may differ from
                # sequence order on replay; canonical sequence identity is exact.
                lines = actual.splitlines(keepends=True)
                if any(not line.endswith(b"\n") for line in lines) or sorted(
                    lines
                ) != sorted(expected.splitlines(keepends=True)):
                    raise ValueError
                self._retained_sha256 = hashlib.sha256(actual).hexdigest()
            elif expected:
                raise ValueError
            else:
                self._retained_sha256 = None
            self._retained_compared = True
        except Exception:
            self._invalid = True
            self._retained_compared = False

    def finish(self, *, verified):
        if self.finished:
            return self._result
        self.finished = True
        try:
            from tuner.runtime.packaged_worker_closure import stable_read

            if self._fd is not None:
                info = self.path.lstat()
                if (info.st_dev, info.st_ino) != self._ledger_identity:
                    raise ValueError
                ledger_raw = stable_read(self.path, self.MAX_BYTES)
                if (
                    len(ledger_raw) != self._bytes
                    or hashlib.sha256(ledger_raw).digest()
                    != self._ledger_digest.digest()
                ):
                    raise ValueError
            else:
                raise ValueError
        except Exception:
            self._invalid = True
        result = {
            "schema_version": "synaptic-modal-metrics-qualification-result/v1",
            "admission_sha256": self.admission.sha256,
            "run": self.run.to_dict(),
            "provider_job_ref": self.provider_job_ref,
            "delivery_verdict": (
                "delivered_precompletion"
                if verified
                and not self._invalid
                and self._retained_compared
                and len(self._proven) >= 2
                else "evidence_invalid"
                if self._invalid
                else "inconclusive"
            ),
            "transport_qualified": False,
            "partial_write_behavior_qualified": False,
            "precompletion_numeric_records": len(self._proven),
            "read_rejection_observed": self._read_failed,
            "terminal_artifacts_verified": verified,
            "ledger_path": str(self.path),
            "retained_file_compared": self._retained_compared,
            "retained_file_sha256": self._retained_sha256,
            "durability_acknowledged": False,
        }
        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raw = observation_bytes(result)
            with os.fdopen(_create_private(self.final_path), "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            from tuner.runtime.packaged_worker_closure import stable_read

            if stable_read(self.final_path, 8192) != raw:
                raise ValueError
            # Saved reports are provisional (acknowledged=false). Only the
            # returned digest-bound result acknowledges successful persistence.
            result["report_sha256"] = hashlib.sha256(raw).hexdigest()
            result["durability_acknowledged"] = True
        except Exception:
            result["delivery_verdict"] = "evidence_invalid"
        self._result = result
        return result
