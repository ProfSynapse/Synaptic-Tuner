"""Explicit pure GRPO recipe registration; no provider dispatch is enabled."""
from tuner.project.execution_source import ExecutionSourceV1
from tuner.training.contracts import CanonicalDocument
from tuner.training.grpo_compilation import GRPO_CONFIG_SCHEMA, compile_grpo_input
from tuner.training.recipes import CompiledWorkload, canonical_json_bytes, selected_execution_mode
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity


class GRPORecipe:
    method = "grpo"

    def compile(self, *, resolved_config, execution_source):
        if type(resolved_config) is not CanonicalDocument or type(execution_source) is not ExecutionSourceV1:
            raise TypeError("exact GRPO config and execution source required")
        value = resolved_config.to_dict()
        if (set(value) != {"schema_version", "method", "execution", "training_input", "prepared_input"}
                or value["schema_version"] != GRPO_CONFIG_SCHEMA or value["method"] != "grpo"
                or selected_execution_mode(resolved_config) != "developer_integration"):
            raise ValueError("GRPO configuration has unsupported schema or fields")
        workload = compile_grpo_input(TrainingInputV1.from_dict(value["training_input"]),
            PreparedTrainingInputIdentity.from_dict(value["prepared_input"]))
        document = workload.document
        document["execution_source"] = execution_source.to_dict()
        return CompiledWorkload(workload.method, workload.schema_version, workload.entrypoint,
                                canonical_json_bytes(document))
