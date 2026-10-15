# Harness-Reviewed Student Rollouts

`harness_review` turns an environment-backed agentic scenario into on-policy
data collection (Harness-Zero, arXiv:2609.24974). A student model proposes
each assistant turn from student-visible context. A reviewing agent, guided
by a private reference harness, either passes the proposal or replaces it with
the smallest correction before it executes. Only accepted turns enter the
trajectory.

Use it when a deployable model should learn procedures (locate, read, verify,
recover) that it does not show on its own. The paper's ablations found that
training on clean teacher trajectories, or on review that has access to the
task answer, transferred much less than review guided by procedural rules.

## Scenario block

```yaml
harness_review:
  enabled: true
  replace_budget: 3             # budgeted replacements per rollout
  max_submission_attempts: 3    # reviewer re-prompts before the turn is marked FAILED
  student:                      # stage LLM spec plus the student's turn prompt
    provider: openrouter
    model: qwen/qwen3.5-9b
    provider_routing: {require_parameters: true}   # only schema-enforcing providers
    schema: use_tools_response
    prompt: "Write the assistant's next message ..."
  reviewer:                     # stage LLM spec plus system/prompt templates
    provider: openrouter
    model: openai/gpt-5.6-terra
    system: "..."
    prompt: "... {reference_harness} ... {proposal_json} ..."
  reference_harness:
    components:                 # procedural guidance; never task answers
      - {id: read-before-answering, kind: skill, content: "..."}
    middleware:                 # deterministic checks that add a private hint
      - {id: final-before-read, final_only: true, unless_prior_pattern: "content read", hint: "..."}
  leak_patterns: ["\\breviewer\\b"]          # replacements matching these are rejected
  reviewer_voice_patterns: ["\\bthe student\\b"]  # flagged in the trace
```

Middleware fields: `pattern` (regex on the stringified proposal), `final_only`,
`tool_only` and `unless_prior_pattern` (suppress when earlier assistant turns
match).

Reviewer prompt variables: `scenario_key`, `turn_index`, `reference_harness`,
`student_action_space`, `student_transcript`, `new_events`,
`review_history_json`, `proposal_json`, `proposal_kind`,
`structural_defects`, `triggered_middleware`, `replacements_used`,
`replace_budget`, `replacements_remaining`.

## Rules

- **The student's system prompt must not contain the answer.** Use the
  `cli_student` workspace format, which omits `<note_contents>`. Do not add
  answer-key `extra_sections`. Keep `task_context` for the user prompt and
  final judge only.
- **Proposals that fail response validation, or that the student cannot
  produce as parseable output, must be replaced.** These structural fixes do
  not count against `replace_budget`.
- **A turn with no valid reviewer submission keeps the student proposal,** is
  traced as `FAILED`, and adds the `harness_review` stage failure.
- **For read-only tasks, set `stop_on_environment_pass: false`.** Fixture
  assertions are already true at the start, so an environment pass would
  falsely tell the model the task is complete. Let the final judge check the
  answer instead.
- **Disable the in-loop judge.** The reviewer replaces it.

## Student action format

Give the student `tool_call_format: {arguments_encoding: object}` so the
schema constrains `function.arguments` as an object. Parsing re-encodes it as
the usual JSON string and drops null optional fields. With the default
`string` encoding, small models write pseudo-JSON into the string, and every
tool turn becomes a reviewer-authored structural replacement.

## Trace and inspection

Each rollout stores `metadata.environment.harness_review`. It holds counts
(`reviews`, `passes`, `replacements`, `budgeted_replacements`,
`reviewer_voice_turns`, `failed`) and a per-turn list. Each turn records the
original proposal, the decision, the accepted replacement, triggered
middleware, structural defects, the reason and the attempts. Every REPLACE
gives a rejected/accepted pair from the same state.

```bash
python SynthChat/scripts/inspect_rollout_trace.py --input <run>.jsonl --max-chars 400
```

Project with `--sft-projection per_turn` (one SFT row per accepted turn) and
`--dpo-output` (one pair per REPLACE). See `rollout-projection.md`. For the
full worked example, see the `case-studies` skill:
`reference/harness-distillation-pipeline.md`.

Pilot scenario: `SynthChat/scenarios/workspace_harness_review_pilot.yaml`.
Targets: `targets_workspace_harness_review_pilot_smoke.json` (1 rollout) and
`targets_workspace_harness_review_pilot_dryrun.json` (3 rollouts).
