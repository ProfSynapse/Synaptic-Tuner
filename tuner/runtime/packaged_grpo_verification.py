"""Native GRPO artifacts reuse existing model/tokenizer archive validation."""
from __future__ import annotations

import os
import math
from pathlib import Path
from types import SimpleNamespace

from tuner.runtime import packaged_sft_execution as shared
from tuner.training.packaged_grpo_compilation import ROLES, parse_packaged_grpo_workload


def _validate_core_records(core, lineage, metrics):
    expected_plan = [{"type": "rubric", "name": item["name"], "weight": float(item["weight"]),
                      "source": str(Path(item["name"]))} for item in core["config"]["rewards"]["items"]]
    from Trainers.grpo.runtime_v1 import _finite_tree
    _finite_tree(lineage)
    _finite_tree(metrics)
    if (type(lineage) is not dict
            or set(lineage) != {"method", "model", "input_commitments", "reward_plan", "actual_steps"}
            or lineage["method"] != "grpo" or lineage["model"] != core["model"]
            or lineage["input_commitments"] != {key: core[key] for key in
                ("config_sha256", "dataset_sha256", "rewards_sha256")}
            or shared._canonical({"plan": lineage["reward_plan"]}) !=
                shared._canonical({"plan": expected_plan})):
        raise ValueError("PACKAGED_GRPO_LINEAGE_REJECTED")
    if (type(metrics) is not dict or set(metrics) != {"global_step", "log_history", "train_metrics"}
            or type(metrics["global_step"]) is not int
            or not 1 <= metrics["global_step"] <= core["config"]["training"]["max_steps"]
            or lineage["actual_steps"] != metrics["global_step"]
            or type(lineage["actual_steps"]) is not int
            or type(metrics["log_history"]) is not list or type(metrics["train_metrics"]) is not dict):
        raise ValueError("PACKAGED_GRPO_METRICS_REJECTED")


def assemble_grpo_artifacts(admitted, compiled, manifest):
    from Trainers.sft import runtime_v1 as utils
    from tuner.runtime.packaged_grpo_execution import LINEAGE_SCHEMA
    core = compiled.document["core_workload"]
    output = admitted.paths.state / "grpo-output"
    held, original = shared._snapshot_inventory(output, admitted.paths.state)
    try:
        records = {role: shared._document(shared.stable_read(output / (role + ".json"), 2 * 1024 * 1024),
                                         2 * 1024 * 1024)
                   for role in ("training_lineage", "training_metrics", "workload_record")}
        if shared._canonical(records["workload_record"]) != shared._canonical(core):
            raise ValueError("PACKAGED_GRPO_WORKLOAD_REJECTED")
        _validate_core_records(core, records["training_lineage"], records["training_metrics"])
        lineage = {"schema_version": LINEAGE_SCHEMA, **shared._bindings(admitted),
            "prepared_input": core["prepared_input"], "model_snapshot": manifest,
            "trainer_lineage": records["training_lineage"], "status": "completed", "trainer_exit_code": 0,
            "child_transport_sha256": shared._digest(shared.stable_read(
                admitted.paths.state / "packaged-grpo-child.json", shared._MAX_CHILD_BYTES))}
        artifacts = [utils._write_artifact(admitted.paths.artifacts, "workload_record", "workload.json",
                                          admitted.workload_bytes),
            utils._write_artifact(admitted.paths.artifacts, "training_lineage", "training_lineage.json",
                                  shared._canonical(lineage)),
            utils._write_artifact(admitted.paths.artifacts, "training_metrics", "training_metrics.json",
                                  shared._canonical(records["training_metrics"]))]
        for role, kind in (("final_model", "model"), ("tokenizer", "tokenizer")):
            artifacts.append(utils._archive_artifact(admitted.paths.artifacts, role=role,
                filename=role + ".tar", source=output / role, artifact_kind=kind,
                locked_model_ref=core["model"]["model_id"]))
        for directory in held:
            directory.check()
        confirmation, current = shared._snapshot_inventory(output, admitted.paths.state)
        try:
            if current != original:
                raise ValueError("PACKAGED_GRPO_OUTPUT_CHANGED")
        finally:
            if shared._close_resources(confirmation):
                raise ValueError("PACKAGED_GRPO_OUTPUT_CHANGED")
        inventory = shared._canonical({"schema_version": "synaptic-artifact-inventory/v1",
            "workload_fingerprint": compiled.fingerprint, "artifacts": artifacts})
        state = next(directory for directory in admitted._directories if directory.path == admitted.paths.state)
        path = admitted.paths.state / "runtime-v1-inventory.json"
        shared._write_terminal_exclusive(path, inventory, state, lambda: [directory.check() for directory in held])
        return SimpleNamespace(workload_fingerprint=compiled.fingerprint, inventory_path=path,
                               artifacts=tuple(artifacts))
    finally:
        if shared._close_resources(held):
            raise ValueError("PACKAGED_GRPO_OUTPUT_CHANGED")


def verify_packaged_grpo_artifacts(*, admitted, inventory_bytes, terminal_bytes):
    from tuner.runtime.packaged_grpo_execution import LINEAGE_SCHEMA, TERMINAL_SCHEMA, _invocation_spec
    from tuner.runtime.verification import _validate_sft_archive_stream
    compiled = parse_packaged_grpo_workload(admitted.workload_bytes)
    inventory, members = shared._retain_result_artifacts(admitted, SimpleNamespace(
        workload_fingerprint=compiled.fingerprint, inventory_path=admitted.paths.state / "runtime-v1-inventory.json",
        artifacts=tuple(shared._document(inventory_bytes)["artifacts"])))
    try:
        if inventory != inventory_bytes:
            raise ValueError
        expected_terminal = {"schema_version": TERMINAL_SCHEMA, **shared._bindings(admitted),
            "status": "completed", "trainer_exit_code": 0,
            "inventory_sha256": shared._digest(inventory_bytes), "artifact_roles": list(ROLES)}
        terminal = shared._document(terminal_bytes)
        if (terminal != expected_terminal or type(terminal.get("trainer_exit_code")) is not int):
            raise ValueError
        contents = {}
        for member in members:
            member.check()
            if member.path.suffix == ".json" and member.path.name != "runtime-v1-inventory.json":
                contents[member.path.name] = shared._document(os.pread(member.fd, member.member["size_bytes"] + 1, 0),
                                                             2 * 1024 * 1024)
        if shared._canonical(contents["workload.json"]) != admitted.workload_bytes:
            raise ValueError
        lineage = contents["training_lineage.json"]
        core = compiled.document["core_workload"]
        fields = {"schema_version", *shared._bindings(admitted), "prepared_input", "model_snapshot",
                  "trainer_lineage", "status", "trainer_exit_code", "child_transport_sha256"}
        if (set(lineage) != fields or lineage["schema_version"] != LINEAGE_SCHEMA
                or any(lineage[key] != value for key, value in shared._bindings(admitted).items())
                or lineage["prepared_input"] != core["prepared_input"] or lineage["status"] != "completed"
                or type(lineage["trainer_exit_code"]) is not int or lineage["trainer_exit_code"] != 0
                or lineage["child_transport_sha256"] != shared._digest(shared.stable_read(
                    admitted.paths.state / "packaged-grpo-child.json", shared._MAX_CHILD_BYTES))):
            raise ValueError
        child_raw = shared.stable_read(admitted.paths.state / "packaged-grpo-child.json", shared._MAX_CHILD_BYTES)
        child = shared._document(child_raw, shared._MAX_CHILD_BYTES)
        if (child.get("model_snapshot") != lineage["model_snapshot"]
                or any(type(child.get(key)) is not int or child[key] <= 2
                       for key in ("dataset_fd", "metric_fd"))
                or type(child.get("deadline")) not in (int, float)
                or not math.isfinite(child["deadline"])):
            raise ValueError
        _, _, expected_child = _invocation_spec(admitted, lineage["model_snapshot"],
            child["dataset_fd"], child["metric_fd"], child["deadline"])
        if expected_child != child_raw:
            raise ValueError
        artifact_root = next(directory for directory in admitted._directories
                             if directory.path == admitted.paths.artifacts)
        shared._check_artifact_namespace(artifact_root, members)
        shared._private_manifest(lineage["model_snapshot"], admitted.paths,
            {"ref": core["model"]["model_id"], "revision": core["model"]["revision"]})
        _validate_core_records(core, lineage["trainer_lineage"], contents["training_metrics.json"])
        for member in members:
            if member.path.suffix == ".tar":
                kind = "model" if member.path.name == "final_model.tar" else "tokenizer"
                with os.fdopen(os.dup(member.fd), "rb") as stream:
                    stream.seek(0)
                    _, valid = _validate_sft_archive_stream(stream, kind,
                                                              locked_model_ref=core["model"]["model_id"])
                if not valid:
                    raise ValueError
            member.check()
        shared._check_artifact_namespace(artifact_root, members)
        return True
    finally:
        if shared._close_resources(members):
            raise ValueError("PACKAGED_GRPO_ARTIFACT_REJECTED")
