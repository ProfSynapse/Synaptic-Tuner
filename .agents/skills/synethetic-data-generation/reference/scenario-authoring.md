# Scenario Authoring Reference

Scenarios define what SynthChat should generate. Tool-call structure is config-driven; scenarios should reference the active configured format instead of assuming a hardcoded wrapper in code.

---

## Config-Driven Tool Scenario Shape

```yaml
scenarios:
  workspace_move_file:
    type: tool
    tool: storage move
    system: true
    tool_call_format: default
    workspace_format: cli_only
    rubrics:
      system_prompt: [system_prompt_format]
      response: [tool_alignment, factuality]
    prompts:
      system: |
        Generate a realistic system prompt with:
        - <session_context> containing sessionId and workspaceId
        - <vault_structure>
        - <available_workspaces>
        - <available_prompts>
        - <selected_workspace> JSON
      user: |
        Generate a natural request that requires moving a file.
        OUTPUT ONLY THE REQUEST TEXT.
      assistant: |
        Generate a single tool response using the configured tool_call_format.
        Follow the active wrapper name, argument shape, and command format from config.
```

The active format is only a scenario/config choice, not a repo-wide runtime
truth. In this repo, the default expectation is that scenarios are
rubric-backed and judge-backed unless there is a deliberate reason to run a raw
smoke test without them.

For new or production-bound scenarios:
- add `rubrics` for at least the stages you care about
- prefer a response-stage rubric for tool correctness
- add `final_judge` and/or in-loop `judge` when the scenario needs an explicit
  quality gate beyond schema/runtime validation

Judge-less scenarios are allowed for narrow plumbing checks, but they should be
treated as temporary smoke-test scaffolding rather than the standard setup.

---

## Format Rule

The tool wrapper, top-level fields, command style, and any context payload shape
must come from config:
- `SynthChat/config/tool_call_formats.yaml`
- scenario YAML
- evaluator/tool schema config

Do not treat the currently active wrapper as permanent. A different user should
be able to define a different format without changing code.

---

## Scenario Types

### Tool
Use for tool-calling datasets. Define the model-facing tool surface in config
or scenario YAML. Do not assume a particular wrapper, manager namespace, CLI
command shape, or argument field set unless the scenario explicitly configures
it.

### Behavioral
Use for clarification, verification, destructive safety, and similar behaviors.

### Docs-Based
Use when `--docs` is supplying source material.

---

## Environment Validation

Use the `environment` block when the tool call should be executed locally during generation:

```yaml
environment:
  allowed_tools:
    - storage move
    - content read
  max_steps: 4
  execution:
    strict_schema: true
  assertions:
    - type: path_exists
      path: "archive/today.md"
```

`allowed_tools`, `expected_tools`, scoring paths, and prose examples should use
the configured model-facing tool surface. The runtime may internally expand
those names for execution, but the scenario should not leak a different
implementation namespace into the training prompt unless that is the surface
being trained.

For multi-turn environment-backed scenarios, configure the loop explicitly:

```yaml
environment:
  loop:
    enabled: true
    mode: agentic
    max_turns: 6
    max_tool_steps: 8
    continue_on_execution_error: true
    continue_on_validation_error: true
    stop_on_text_response: true
    stop_on_environment_pass: true
    tool_result_name_format: command
  require_expected_tools: true
```

Use `tool_result_name_format` when the model-facing feedback should display
configured command/tool names rather than internal executor identifiers. This
keeps generated conversations aligned with the surface being trained.

When a scenario requires a final natural-language answer after tool work, make
that behavior a positive completion rule in the model-facing prompt, not only a
negative prohibition in the judge. For example, tell the model that once the
needed tool results answer the request or confirm the action is complete, it
should stop calling tools and send concise text summarizing the answer or
completed action. Keep the exact transport/wrapper details in config, but make
the completion behavior clear in shared assistant instructions, tool-options
sections, or equivalent scenario prompt blocks.

For GRPO curriculum design, consider separating "environment solved" from
"final text emitted correctly." Early shaped rewards can give credit for
correct tool/environment completion while applying a penalty for missing final
text. Once the model reliably acts in the environment, tighten final gates or
reward config so extra post-completion tool calls become clear failures.

For generated environments, prefer deterministic stage gates for structural
validity: payload shape, canonical schema, placeholder scans, and minimum
fixture size. Use an LLM environment-generation judge only for semantic checks
that deterministic gates cannot express. If that judge starts contradicting the
normalized JSON (for example claiming `fixture.files` is an array when the
schema gate passed an object), disable or narrow the LLM judge in config rather
than adding parser/runtime repairs.

Model selection is stage-specific. The model that authors a generated
workspace does not have to be the same model that produces the assistant
trajectory, and neither has to match the in-loop or final judge. In
environment-backed training data, a reliable fixture author can reduce noise
while the rollout model remains the model being trained or evaluated. Express
this only in config:

```yaml
environment_generation:
  provider: your_provider
  model: fixture_author_model
  response_format: json_object
  gates: [...]
  judge:
    enabled: true
    provider: your_provider
    model: environment_review_model

assistant_generation:
  provider: your_provider
  model: rollout_model

judge:
  in_loop:
    provider: your_provider
    model: turn_review_model
```

Validate each stage separately. Run `env-generate` to prove fixture shape and
hidden answer keys before running an agentic rollout. Then inspect the rollout
trace to decide whether failures came from environment authoring, the
assistant model, the in-loop judge, or final judging. Do not patch runtime code
to make one stage compensate for another stage's bad config.

When a generated environment contains loose dynamic keys such as arbitrary file
paths or task-specific answer-key fields, provider JSON mode is the simplest
starting point for that stage:

```yaml
environment_generation:
  schema: canonical_environment
  response_format: json_object
  gates:
    - type: json_schema
      field: generated_environment
      schema: canonical_environment
```

Strict `json_schema` mode is best for fixed response shapes such as tool-call
wrappers, and it can also work for generated environments when the scenario
provides a complete inline schema for the allowed structure. Use it when you
can constrain dynamic file maps with typed `additionalProperties`, require the
needed `task_context` keys, enforce fixed command counts, and add ASCII/path
patterns. Prefer this path when JSON mode produces malformed JSON, corrupted
hidden anchors, or sparse environments that repeatedly need review feedback.
If the environment shape is too open-ended to express cleanly as JSON Schema,
keep `json_object` and let deterministic stage gates reject bad samples.

Shared environment gates are only a baseline. If a generated scenario depends
on exact `task_context` keys, a fixed command count, or assertion semantics,
add scenario-specific gates in YAML. Prefer `required_mapping_keys` for hidden
answer-key fields and an inline `json_schema` gate for constraints such as
"expected_command_sequence has exactly two commands" or "assertion types may
only be `path_exists` and `file_contains`." This catches bad generated
environments before the agent loop wastes turns on impossible assertions.
If a scenario references a deterministic gate type that the runner does not
support, either express the check with supported gates or add a generic gate
implementation. Do not add scenario-specific validation logic to runtime code.
Also gate generated answer-key commands against stale or unsupported command
surfaces. If the trained interface is a configured tool wrapper or CLI, reject
shell examples such as `grep`, `cat`, pipes, semicolons, and chained shell
operators in `task_context.expected_command_sequence` with
`no_placeholder_strings` or an inline schema/pattern gate.

For CLI-style tools, keep raw command validation config-driven. If debug
artifacts show model-emitted commands with non-ASCII whitespace, markdown
backticks, shell syntax, or other provider-specific artifacts, add
`invalid_cli_patterns` in the environment execution config or scenario
execution overrides. Do not repair those strings in scenario-specific runtime
code; the model should receive structured validation/tool feedback and retry.

Assistant turns and in-loop judges can choose their provider response format
the same way environment generation does. For routed providers, strict
`json_schema` may be useful for simple fixed outputs, but complex nested
tool-call arrays can trigger provider-internal artifacts or long stalls. In
that case, set `assistant_generation.response_format: json_object` or
`judge.in_loop.response_format: json_object` in YAML, then rely on local schema
validation, environment execution, and judge feedback for retries.

Use `continue_on_validation_error` when malformed assistant turns should be
fed back to the model and retried inside the same episode. This is usually the
right default for agentic data generation because otherwise an invalid first
tool call exits before any environment result can guide correction.

For GRPO-style environment datasets, prefer `stop_on_environment_pass: true`
unless the reward/scenario explicitly requires a natural-language final answer.
This keeps the rollout focused on the completed environment trajectory and
avoids brittle post-pass text generation through a structured tool-call schema.

Only opt into post-environment single-response repair for an agentic scenario
when you intentionally want a failed trajectory rewritten as one assistant
message. For normal multi-turn data, debug and gate the full episode trace.

Only opt into post-loop response validation/improvement when you intentionally
want a separate response-stage rewrite after the agentic episode. For normal
multi-turn datasets, leave that off so the saved conversation remains the
actual environment-backed trajectory.

For recovery-style multi-turn data, decide explicitly whether earlier
recoverable tool errors should fail the example. If the behavior being trained
is "recover after a bad tool call," final gates should usually require the final
environment assertions to pass, while final judges should allow earlier
recoverable errors that were corrected later in the same rollout.

---

## Authoring Rules

- Use `tool_call_format: default` unless there is a strong reason to override it.
- Default to rubric-backed scenarios. Do not leave `rubrics`, `judge`, and
  `final_judge` empty by accident.
- For tool scenarios, the baseline should usually include:
  - `rubrics.response` with tool-format/tool-alignment checks
  - `final_judge` when you need a final pass/fail gate on overall quality
  - `judge.in_loop` only when the scenario benefits from iterative correction
- If environment validation is enabled, make sure the response-stage judge or
  improver can see environment/runtime failures through template variables or
  stage payloads. Those failures should inform improvement recommendations.
- For content edits, define concrete read/write/insert/replace commands in
  scenario or tool-schema config and make examples match that syntax exactly.
- For early training on edit tools, prefer command-safe generated anchors:
  hyphenated search tokens and underscore-only replacement strings. Add quoting
  and multi-word arguments as a later difficulty tier after the model reliably
  performs search -> read -> edit.
- Keep format-specific instructions inside scenario/config text, not runtime code.
- Use checked-in target manifests for smoke tests.
- If you intentionally omit rubrics/judges for a smoke test, note that choice in
  the scenario comments or plan so it is clearly an exception.
- For retrieval-then-action tasks, do not require a fallback search/list tool
  as an expected tool if the primary search can satisfy the environment. Put
  fallback behavior in judge/improver guidance, not in mandatory expected tools,
  unless every passing trajectory must execute that fallback.
- When a later action depends on unknown tool output, instruct the model to stop
  after the discovery command and wait for feedback. Chain multiple commands
  only when every path and argument is already known.
- In-loop judge feedback should be short and copyable. Prefer one next command:
  `Call useTools with tool exactly: <command>`. Avoid wrapper JSON, markdown
  fences, multiple alternatives, and escaped command examples in feedback shown
  back to the model.
- For loops that require a final text answer after environment pass, make the
  in-loop judge distinguish final-text turns from tool turns. Once the
  environment preview has passed and the runner asks for final text, the judge
  should accept a concise text-only answer with no tool calls. Do not let the
  judge invent extra verification/search/read steps unless those exact commands
  are still missing from the configured/generated expected command sequence.
  Grounded supporting details from the latest tool output are acceptable unless
  they contradict the environment result or violate the scenario's requested
  answer style.

---

## Dry-Run First

Before broad generation:

```bash
python3 -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```
