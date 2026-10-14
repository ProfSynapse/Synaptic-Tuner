"""Fixed isolated GRPO child: sealed rows, installed sources, credential-free ML."""
from __future__ import annotations

import math
import os
from pathlib import Path
import stat
import sys
import time
from types import SimpleNamespace

from tuner.runtime import packaged_sft_execution as shared
from tuner.runtime.packaged_sft_child import _classified_failure, _installed_import_guard, _metric_sink_for_fd
from tuner.runtime.packaged_grpo_execution import _CHILD_SCHEMA, _admit_contracts, _admit_dataset, _invocation_spec, _model
from tuner.runtime.releases import PackagedExecutionBindingV1, ProviderRuntimeBindingV1, parse_packaged_runtime_release
from tuner.training.contracts import ArtifactPolicy


def _sealed_input(descriptor, execution):
    from Trainers.sft import runtime_v1 as seals
    import fcntl
    if type(descriptor) is not int or descriptor <= 2 or not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise ValueError
    required = (seals._LINUX_F_SEAL_SEAL | seals._LINUX_F_SEAL_SHRINK
                | seals._LINUX_F_SEAL_GROW | seals._LINUX_F_SEAL_WRITE)
    if fcntl.fcntl(descriptor, seals._LINUX_F_GET_SEALS) & required != required:
        raise ValueError
    raw = os.pread(descriptor, execution.prepared_input_size_bytes + 1, 0)
    if (len(raw) != execution.prepared_input_size_bytes
            or shared._digest(raw) != execution.prepared_input_content_digest):
        raise ValueError
    return raw


def _metric_sink(descriptor, maximum, total_steps):
    from synaptic_tuner.api.v1.observations import TrainingMetricObservedPayloadV1
    sink = _metric_sink_for_fd(descriptor, maximum)
    def emit(record):
        try:
            payload = TrainingMetricObservedPayloadV1(
                step=record["step"], total_steps=total_steps, loss=record.get("loss"),
                learning_rate=record.get("learning_rate"), epoch=record.get("epoch"),
                train_elapsed_seconds=None, job_elapsed_seconds=None,
                gpu_current_allocated_bytes=None, gpu_current_reserved_bytes=None,
                gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None,
                gpu_utilization_percent=None, tokens_per_second=None,
                estimated_samples_per_second=None)
            sink(payload)
        except BaseException:
            # Observation transport is optional; terminal native records retain
            # actual GRPO trainer statistics, including any measured rewards.
            pass
    return emit


def _run(argv, phase):
    args = tuple(sys.argv[1:] if argv is None else argv)
    if (not sys.flags.isolated or len(args) != 4 or args[0] != "--transport"
            or args[2] != "--digest"):
        raise ValueError
    raw = shared.stable_read(Path(args[1]), shared._MAX_CHILD_BYTES)
    if shared._digest(raw) != args[3]:
        raise ValueError
    payload = shared._document(raw, shared._MAX_CHILD_BYTES)
    if (set(payload) != {"schema_version", "release", "provider_binding", "execution", "workload",
                         "policy", "paths", "environment", "model_snapshot", "dataset_fd",
                         "metric_fd", "deadline"} or payload["schema_version"] != _CHILD_SCHEMA):
        raise ValueError
    phase[0] = "RELEASE"
    release = parse_packaged_runtime_release(payload["release"])
    provider = ProviderRuntimeBindingV1.from_dict(payload["provider_binding"])
    execution = PackagedExecutionBindingV1.from_dict(payload["execution"])
    policy = payload["policy"]
    if type(policy) is not dict or set(policy) != {"required_kinds", "retain_checkpoints"}:
        raise ValueError
    policy = ArtifactPolicy(tuple(policy["required_kinds"]), policy["retain_checkpoints"])
    workload = shared._canonical(payload["workload"])
    compiled = _admit_contracts(release, provider, execution, workload, policy)
    trainer = shared._inspect_release(release, method="grpo")
    phase[0] = "INPUT"
    paths = shared.PackagedSFTPaths(**{key: Path(value) for key, value in payload["paths"].items()})
    deadline = payload["deadline"]
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or not 0 < deadline - time.monotonic() <= compiled.document["limits"]["child_max_seconds"]):
        raise ValueError
    content = _sealed_input(payload["dataset_fd"], execution)
    _admit_dataset(compiled, content)
    environment = tuple(tuple(pair) for pair in payload["environment"])
    if (len(dict(environment)) != len(environment)
            or not set(dict(environment)) <= {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LANG", "LC_ALL"}
            or any(type(value) is not str or "\0" in value for _, value in environment)):
        raise ValueError
    physical = SimpleNamespace(paths=paths, environment=environment, release=release, execution=execution,
        provider_binding=provider, artifact_policy=policy, workload_bytes=workload)
    invocation, transport, expected = _invocation_spec(physical, payload["model_snapshot"],
        payload["dataset_fd"], payload["metric_fd"], deadline)
    if (expected != raw or tuple(invocation.argv[5:]) != args or str(transport) != args[1]
            or dict(os.environ) != dict(invocation.environment) or Path.cwd() != paths.tmp):
        raise ValueError
    phase[0] = "IMPORT"
    guard = _installed_import_guard(release, trainer, method="grpo")
    sys.meta_path.insert(0, guard)
    sys.dont_write_bytecode = True
    retained = None
    state = None
    output = None
    try:
        phase[0] = "INPUT"
        retained = shared._retain_private_snapshot(payload["model_snapshot"], paths, _model(compiled))
        retained.check()
        state = shared._HeldDirectory(paths.state)
        state.check()
        os.mkdir("grpo-output", mode=0o700, dir_fd=state.fd)
        output = shared._HeldDirectory(paths.state / "grpo-output", parent_fd=state.fd)
        state.check()
        output.check()
        core = compiled.document["core_workload"]
        from Trainers.grpo.runtime_v1 import run_grpo
        sink = _metric_sink(payload["metric_fd"], min(10000, core["config"]["training"]["max_steps"]),
                            core["config"]["training"]["max_steps"])
        if time.monotonic() >= deadline:
            raise TimeoutError
        phase[0] = "EXEC"
        run_grpo(config=core["config"], dataset_bytes=content,
            reward_documents=core["reward_documents"], reward_schema=core["reward_schema"],
            model_snapshot=retained.snapshot, model_identity=core["model"], output_root=output.path,
            workload=core, metric_sink=sink, _packaged_artifacts=True)
        phase[0] = "POSTCHECK"
        retained.check()
        state.check()
        output.check()
        if time.monotonic() >= deadline:
            raise TimeoutError
        return 0
    finally:
        failed = False
        for descriptor in (payload["dataset_fd"], payload["metric_fd"]):
            try:
                os.close(descriptor)
            except OSError:
                failed = True
        failed = shared._close_resources(tuple(value for value in (output, state, retained)
                                               if value is not None)) or failed
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            failed = True
        if failed:
            phase[0] = "POSTCHECK"
            raise RuntimeError("PACKAGED_CHILD_CLEANUP_UNCERTAIN") from None


def main():
    phase = ["TRANSPORT"]
    try:
        return _run(None, phase)
    except BaseException as error:
        print("PACKAGED_GRPO_CHILD_REJECTED", file=sys.stderr)
        return _classified_failure(phase[0], error).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
