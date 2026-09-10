# Embedded run-chat consumer acceptance slice

This slice exercises the smallest host-consumer composition for talking to a
verified trained model. A minimal `APIHost` supplies its public `RunsAPI` to
`open_run_chat`; the generic helper then performs real authenticated artifact
reverification and materialization, prepares the exact serving target, and
opens a consumer-supplied bounded `ChatSession` runtime.

The acceptance tests cover both LoRA and full SFT results. LoRA preparation is
called once with the exact retained model identity; full-model chat performs no
base-model preparation. Retrieved model and tokenizer paths remain available
after runtime teardown. A backend error closes the owned runtime exactly once,
while failed run authentication reaches neither the preparer nor runtime.

The first fake runtime models a local vLLM adapter. A second independently
implemented local adapter exercises the same `RunChatRuntime.open(target)` seam.
The generic helper receives no provider identifier, provider knobs, or global
runtime registry, and it adds no prompt or wrapper: the sole caller message is
passed through the existing generic role/content `BackendClient` contract.

These are embedded-library acceptance tests, not a CLI, configuration loader,
provider call, GPU smoke, or proof of a concrete runtime's model loading. Actual
runtime security bindings and process launch remain responsibilities of the
trusted adapter supplied by composition.
