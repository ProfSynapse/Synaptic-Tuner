# Environment Rollouts

This folder defines the canonical dataset shape for environment-backed synthetic
episodes used to train and evaluate workflow behavior in real runtimes.

Design goals:
- keep one rich base dataset with full environment traces
- derive `KTO` and `GRPO` views from that base dataset later
- keep schema/format failures separate from behavior failures
- keep scenario/task definition separate from projection logic

Recommended workflow:
1. Generate canonical rollout records from SynthChat + environment execution.
2. Inspect and filter by `labels`, `results`, and `stop_reason`.
3. Materialize a `KTO` projection with `prompt`, `completion`, and `label`.
4. Materialize a `GRPO` projection with string `prompt` plus reward columns.

Primary files in this folder:
- [vault_workflows_v1_SPEC.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/vault_workflows_v1_SPEC.md)
- [canonical_rollout.schema.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/canonical_rollout.schema.json)
- [kto_projection.schema.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/kto_projection.schema.json)
- [grpo_projection.schema.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/grpo_projection.schema.json)

The initial 10-scenario pack uses:
- scenario source: [vault_kto_pilot.yaml](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/SynthChat/scenarios/vault_kto_pilot.yaml)
- local target counts: [targets_vault_workflows_v1.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/SynthChat/config/targets_vault_workflows_v1.json)
