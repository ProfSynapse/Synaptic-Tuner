# Embedded run-chat consumer acceptance slice

This slice exercises the smallest host-consumer composition for talking to a
verified trained model. A minimal `APIHost` supplies its public `RunsAPI` to
`open_run_chat`; the generic helper delegates run preparation to the selected
runtime and verifies its returned run/model/artifact projections. The local
consumer adapter performs real artifact reverification/materialization and
serving-target preparation before opening its bounded `ChatSession`.

Correction (2026-09-10, runtime-first): the consumer acceptance tests now cover
real local LoRA preparation plus an independent remote-style runtime that performs
current run reverification/outcome admission without artifact-body streaming or
local preparation. Local model and tokenizer paths remain available after
teardown; remote results expose model/artifact metadata and `local_model=None`.
Failed remote reverification reaches neither serving startup nor the backend.
Real local full/LoRA and backend-error teardown remain covered by the separate
`tests/evaluator/test_run_chat_local_integration.py` selection.

The first fake runtime models a local adapter. A second independently
implemented remote-style adapter exercises the same `RunChatRuntime.open(runs, run)` seam.
The generic helper receives no provider identifier, provider knobs, or global
runtime registry, and it adds no prompt or wrapper: the sole caller message is
passed through the existing generic role/content `BackendClient` contract.

These are embedded-library acceptance tests, not a CLI, configuration loader,
provider call, GPU smoke, or proof of a concrete runtime's model loading. Actual
runtime security bindings and process launch remain responsibilities of the
trusted adapter supplied by composition.
