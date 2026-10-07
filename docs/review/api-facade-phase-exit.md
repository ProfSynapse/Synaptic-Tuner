# API facade phase exit: evidence
Verified on engine commit `af04218c` (all eleven facade slices merged) on 2026-09-18.

## Exit criterion (`docs/plans/api-facade-slices.md`, "Phase exit")

> The phase is complete when `compose_reference_host` over the fake provider family
> drives all seven facades through the conformance ladder, both import-closure gates
> pass, `synaptic_tuner/host/v1` is gone, and no module under `api/v1/reference/`
> imports `api/v1/persistence.py`. No live provider effect is claimed; the frozen
> matrix rows for Modal cancel/reconcile and for publication are untouched.

## Proof, clause by clause (`tests/contract/test_api_facade_phase_exit_v1.py`)

| Clause | Test id |
|---|---|
| One `compose_reference_host` over the fake family, in-memory stores, all four optional ports, wrapped in `APIHost(HostPorts(...))`; every property is its `*API` type; training load→resolve→plan→preflight→start; runs show/outcome/logs/verify/reverify/artifacts/list and a real idempotent cancel; artifacts destinations/publications/publish/verify; evaluation plan→preflight→start→show→result→list→observations plus a cooperative cancel; data plan→preflight→start→show→list→datasets→validate→observations (homogeneous JSONL, sidecar present); pipelines plan→start→show→resume→list→observations with the train and evaluate children visible through `host.runs` / `host.evaluation`; chat open→turn→turn→show→list→close→observations; effects partition holds seven fake-provider stage/submit/cancel effects and no `spend`; no `.synaptic` / `.tracking` under the tmp root; socket guard active | `test_reference_host_drives_all_seven_facades_through_their_ladders_over_the_fake_family` |
| Package gate (fresh `-I` interpreter, `import synaptic_tuner.api.v1`) | `test_package_gate_importing_the_public_package_loads_no_provider_engine_or_database_module` |
| Module gate (fourteen contract submodules imported directly, fresh `-I` interpreter) | `test_module_gate_importing_every_contract_submodule_directly_loads_no_engine_module` |
| Source gate (AST over the sixteen contract sources) | `test_source_gate_contract_sources_import_no_engine_provider_or_database_module` |
| `synaptic_tuner/host` is gone | `test_the_superseded_host_package_is_gone` |
| No `api/v1/reference/*.py` imports `api/v1/persistence.py` | `test_no_reference_module_imports_the_frozen_persistence_module` |

Not driven, by contract: `RunsAPI` has no `observations` verb (nine verbs; nothing reads the `training`
observation family), and `RunsAPI.reconcile` is skipped because the grant policy is per effect kind, so an
indeterminate submit would also reroute the pipeline's train child; both stay proven in
`test_reference_composition_v1.py`. Fake-provider evidence only: no Modal, Docker, HF or network call.

## Commands and counts (2026-09-18, `af04218c`)

```
rtk proxy python3 -m pytest tests/contract -q -p no:cacheprovider                         # 789 passed in 58.45s
rtk proxy python3 -m pytest tests/contract/test_api_facade_phase_exit_v1.py -q -p no:cacheprovider -p no:randomly
                                                                                          # 6 passed in 2.44s, then 6 passed in 2.37s
python3 scripts/regenerate_offline_sft_worker_closure.py                                  # {"member_count": 66, "status": "CURRENT"}
python3 scripts/regenerate_modal_inference_lock.py                                        # {"member_count": 118, "status": "CURRENT"}
python3 scripts/regenerate_modal_runtime_lock.py                                          # {"locked_file_count": 97, "status": "CURRENT"}
```

One merge interaction was fixed on the branch as `4a89c110` before this note: slice 11 (`d0c2b1a6`)
added the completion call-site census and slice 8 (`af04218c`) later added `live.session.chat(...)` in
`synaptic_tuner/api/v1/reference/chat.py`, an engine `ChatSession` caller, now classified in
`OTHER_PROTOCOL_CALLERS` of `tests/contract/test_llm_completion_call_sites_v1.py` next to
`tuner/inference/chat_session.py`. Before that pin `tests/contract` was 787 passed, 1 failed.

## Known pre-existing failures outside `tests/contract` on this base

`tests/inference/test_retrieved_model.py::test_cleanup_preserves_replaced_owned_role_leaf` (intermittent);
6 in `tests/test_evaluator_multistep.py`; 3 in `tests/shared/experiment_tracking/`; 2 in
`tests/shared/test_vault_gym_scenarios.py`; 3 in `tests/synthchat/test_schemas.py`; 3 in
`tests/synthchat/test_workspace.py`; 10 in `tests/test_synthchat_generator.py`; 1 in
`tests/flywheel/test_experiment_loop.py`; 2 in `tests/flywheel/test_token_capture.py`;
`tests/execution/coordinator_v1/test_boundaries.py::test_coordinator_import_is_provider_sdk_and_storage_neutral`
only under collection-order pollution. Six SynthChat tool-wrapper tests fail on every baseline back to before the facade work.

## Follow-ups the slices left open

SynthChat fallback-provider client creation bypasses the usage meter; the paid judge remains
`backend_unmetered`; `tuner/inference/owned_process.py`, `vllm_runtime.py`, `Evaluator/verified_vllm_chat.py`
and the Modal inference modules still attach `cleanup_lease` to exceptions while `model_chat.py` uses the
typed failure; training observations (`ObservationFamily.TRAINING`) have no public reader on `RunsAPI`;
`ChatOperationCode` has no `cursor_invalid`, so a malformed chat list cursor surfaces as `session_missing`.
