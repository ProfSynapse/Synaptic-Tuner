# Case Study: Tool-Calling Pipeline

Use this case study when designing a dataset where the assistant must emit
structured tool actions and then continue based on environment feedback. The
exact wrapper name, argument fields, action syntax, and environment semantics
belong in scenario/config files, not in the skill or runtime code.

---

## What The Model Must Learn

1. Use the configured tool-call response shape.
2. Include the configured required context fields exactly where the schema says
   they belong.
3. Express concrete operations using the configured action payload format.
4. Preserve ordering when the scenario requires multiple actions.
5. Ask for clarification before vague, risky, or destructive operations.
6. End a completed trajectory with a normal text response when the scenario
   expects one.

---

## Config Ownership

The source of truth should be declarative:

- tool-call response shape: configured format/schema file
- available actions and arguments: configured tool schema
- workspace or fixture state: scenario/environment YAML
- expected action sequence: scenario metadata or gates
- reward/judge behavior: rubric/final-judge config

Skills may describe the workflow, but they should not name the active wrapper,
current context fields, or concrete action strings as though they are universal.

---

## Generic Example Shape

```json
{
  "tool_calls": [
    {
      "id": "call_0001",
      "type": "function",
      "function": {
        "name": "CONFIGURED_WRAPPER_NAME",
        "arguments": "{\"FIELD_A\":\"value\",\"FIELD_B\":\"value\",\"ACTION_FIELD\":\"CONFIGURED_ACTION_PAYLOAD\"}"
      }
    }
  ],
  "content": null
}
```

This is only an illustrative placeholder. Replace the wrapper, fields, and
payload with whatever the active scenario config declares.

---

## Dataset Creation Flow

1. Start from the canonical schema/config for the capability being trained.
2. Author or update scenario YAML with the configured tool-call format,
   generated environment shape, expected actions, and rubrics.
3. Run environment-generation-only probes before full rollouts.
4. Run a tiny full smoke with raw trace output enabled.
5. Inspect the model-facing prompt, assistant payloads, environment responses,
   judge feedback, and final accepted JSONL rows.
6. Fix scenario/config/rubric gaps before changing runtime code.
7. Scale in stages and keep passed-only artifacts when failures are expected.

---

## System Prompt Alignment

System prompts should be aligned with the deployment environment, but should
remain schema-driven. If deployment injects selected workspaces, compacted
context, prompt references, or available tool summaries, represent those as
generic sections in the dataset profile and keep the concrete field names in
config.

For SFT, keep the prompt as small as possible for the target behavior. For GRPO
or environment-backed rollouts, include only the strategy guidance needed for
the model to explore, act, observe tool feedback, and finish with a text answer.

---

## Validation

Prefer deterministic gates for structural facts:

- configured wrapper/action shape parses
- required fields are present
- disallowed fields are absent
- action names and arguments exist in the configured tool schema
- expected actions were executed in the required order when order matters
- destructive actions require confirmation when the scenario says so

Use LLM judges for semantic quality:

- whether the model gathered enough context before acting
- whether it chose an efficient-enough path
- whether the final answer accurately reflects environment results
- whether recovery turns improved the trajectory instead of adding noise

---

## Smoke Test Pattern

```bash
python -m SynthChat.run generate \
  --targets-file path/to/targets.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_tool_calls.jsonl
```

The important part is not the concrete command above; it is the inspection loop:
generate a small sample, inspect raw traces, repair config/rubrics, then scale.
