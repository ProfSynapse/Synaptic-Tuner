# Evaluator Config-Driven Assertion Refactor Plan

**Status**: Proposed
**Scope**: `Evaluator/`, `shared/environments/validator.py`, active evaluator config under `Evaluator/config/`, evaluation skill docs
**Goal**: Replace tool-specific and behavior-specific evaluator logic with a generic assertion engine where scenarios define the prompt and the exact allowed correct response patterns using JSONPath, regex, exact values, and environment assertions. There is no target concept of "expected tools"; there are only expected responses.

---

## Executive Summary

The evaluator is currently only partially config-driven. Scenario YAML can describe some expectations, but Python code still encodes the old semantic tool model: concrete manager-style tool names, wrapper expansion into `agent_tool`, fixed `expect.tool` semantics, named behavior predicates, and scoring paths that reason over `tool_names`.

The target design is assertion-driven:

1. The config defines the question and the allowed correct responses.
2. The runtime extracts configured views from the model response, such as raw text, structured tool calls, first tool-call name, `arguments.tool`, or environment state.
3. A generic assertion engine evaluates those views using reusable primitives like `equals`, `regex`, `contains`, `jsonpath_exists`, `jsonpath_equals`, `jsonpath_regex`, `json_subset`, `length_min`, `none`, `all`, and `any`.
4. Tool calling, behavior checks, scoring, environment validation, and reporting are not separate correctness systems. They all reduce to the same configured assertion result model.

In the target architecture, "tool eval" and "behavior eval" are labels for groups of scenarios, not separate validator types. A tool case is just a response assertion that happens to inspect a tool-call object or CLI command string. A behavior case is just a response assertion that happens to inspect text, absence of a tool call, required context fields, or a sequence of generated actions.

This makes the CLI schema just data. If the current expected output is:

```json
{
  "tool_calls": [
    {
      "function": {
        "name": "useTools",
        "arguments": {
          "workspaceId": "workspace_main",
          "sessionId": "session_eval",
          "memory": "Need to run the prompt requested by the user.",
          "goal": "Execute the selected prompt.",
          "constraints": "Use the current workspace context.",
          "tool": "prompt execute-prompt \"weekly planning\""
        }
      }
    }
  ]
}
```

then the scenario should test that exact shape directly, for example with `jsonpath_equals` on `$.tool_calls[0].function.name` and `jsonpath_regex` on `$.tool_calls[0].function.arguments.tool`. The evaluator should not normalize that into `promptManager_executePrompt`, because that recreates the old system inside runtime code.

---

## Current State Findings

### 1. Scenario Config Still Contains Old Semantic Tool IDs

Active scenarios still encode correctness as manager-style tool names:

| File | Current Pattern | Why It Blocks Config-First Evaluation |
|---|---|---|
| `Evaluator/config/scenarios/tool_prompts.yaml` | `expected_tools`, `expect.tool`, `params_include`, IDs like `storageManager_copy` | Tests expect old semantic tool IDs instead of the current CLI command string. |
| `Evaluator/config/scenarios/behavior_prompts.yaml` | `expected_tools`, `acceptable_tools`, `behavior_expectations`, `anti_patterns`, old manager IDs | Behavior cases still depend on runtime knowing named behaviors and concrete tool identifiers. |
| `Evaluator/config/scenarios/vault_gym.yaml` | `expected_tools`, `allowed_tools`, ordered tool names, environment `require_expected_tools` | Multi-turn environment checks still validate and constrain execution with manager-style tool names. |
| `Evaluator/config/behaviors.yaml` | behavior profiles with old tool names | Behavior correctness is still defined as named runtime concepts instead of assertions. |
| `Evaluator/config/response_types.yaml` | response types such as `information_gathering` with hardcoded categories like `search`, `read`, `list` | Even config-level abstractions assume this specific product/tool taxonomy. |
| `Evaluator/config/rubrics/*.yaml` | judge rubrics that mention expected tools or fixed judge meanings | Judge behavior can become another hardcoded correctness layer if its prompt variables and expected output are not config-defined. |

The config has started moving toward the new wrapper by expecting `useTools`, but the actual correctness still points at old names such as `promptManager_executePrompts` and `storageManager_createFolder`.

### 2. `ConfigLoader` Promotes Old Fields Into Runtime Tool Expectations

`Evaluator/config_loader.py` still treats `expected_tools`, `acceptable_tools`, `expect.tool`, `expect.acceptable[].tool`, `params_include`, `first_tool`, and `first_tool_any_of` as first-class runtime fields. That means the data model carries old assumptions before validation even starts.

The desired loader behavior is simpler: load `correct.any[].assertions` as opaque config and attach it to the case metadata. The loader should not interpret "tool", "acceptable tool", "first tool", or parameter expectations as special concepts.

### 3. `ConfigDrivenValidator` Is Not Truly Config Driven

`Evaluator/config_validator.py` parses responses into `ParsedToolCall.name` with comments and logic centered on full names like `storageManager_move`. Its CLI wrapper path builds `full_name = f"{agent}_{name}"`, parses `arguments.tool`, and then converts the CLI command back into the old semantic name.

That is the core alignment bug. For current CLI-centric evaluation, the expected value is the CLI string itself. The validator should not know that `storage copy` means `storageManager_copy`; it should only expose configured response views and run configured assertions against them.

### 4. Lower-Level Schema Validation Still Has Wrapper Expansion Logic

`Evaluator/schema_validator.py` expands wrapper calls with a `calls` array into concrete tool names and includes prompt-manager-specific ID validation. It also leans on a lower-level dataset validator with fixed parsing assumptions.

This should become format extraction, not semantic validation:

- Extract raw response text.
- Extract OpenAI-style tool-call objects when present.
- Extract configured text-embedded tool-call formats when configured.
- Expose the extracted objects to assertions.
- Do not infer product-specific tool names.

### 5. Runner, Scoring, Judge Context, and Reporting Still Use `expected_tools`

`Evaluator/runner.py` computes pass/fail through `validator.passed`, `_check_expected_tools`, behavior validation, optional judge validation, and path scoring over `tool_names`. Scoring config supports keys like `all_tools`, `any_tools`, `ordered_tools`, and `first_tool`.

These should be replaced by assertion paths:

- Required correctness path: `correct.any[]`.
- Optional scoring paths: `score.paths[].assertions`.
- Judge prompt context: include configured prompt inputs, configured expected judge response shape, and judge assertion results, not `expected_tools`.
- Reporting: list passed/failed assertion paths and selected actual values, not "expected tools".

### 6. The Behavior Validator Should Not Exist in the Active Path

`Evaluator/behavior_validator.py` has named predicates such as `does_not_call_tool`, `calls_tool_directly`, `explains_choice`, `delegates_complex_task`, `uses_execute_prompts`, and anti-pattern names with hardcoded interpretations. Some branches explicitly mention old tools like `promptManager_executePrompts`.

Behavior evals should not be a separate hardcoded subsystem. The active evaluator should not call a behavior validator. Behavior scenarios should be ordinary response expectations:

- "Should ask a clarifying question" becomes `text_regex`.
- "Should not call a tool" becomes `jsonpath_length_equals` or `not_exists` on tool calls.
- "Should call the prompt CLI" becomes regex on `arguments.tool`.
- "Should include memory/goals/constraints" becomes JSONPath existence/min-length assertions.

If behavior names are useful for authoring, they should be config templates only. Expanding `templates.clarifying_question` into assertions is acceptable. Calling Python code that knows what `clarifying_question` means is not.

### 7. Environment Validation Has Two Models

`shared/environments/validator.py` already has config-defined state assertions such as `path_exists`, `file_contains`, and `frontmatter_field_equals`. That part is directionally correct, but it also still accepts `expected_tools` and checks whether semantic tool names executed.

Environment validation should keep state assertions, but tool/action expectations should move into the same generic assertion model. Executed actions can be exposed as a view such as `$.environment.executed_actions[*].command` or `$.environment.executed_actions[*].name`, and scenarios can assert against those values.

`allowed_tools` is also a legacy semantic surface. If environment execution needs an allowlist, it should be expressed in the command/action vocabulary the configured runtime actually accepts, not old manager IDs. It must not become an indirect correctness check.

### 8. Execution Parsers Must Not Become Validators

`shared/environments/tool_executor.py` may need to parse CLI command strings into executable runtime operations. That is execution plumbing, not evaluation truth.

Hard rule:

> CLI-to-manager or CLI-to-runtime expansion may exist only inside environment execution. Its expanded names must not feed correctness assertions, scoring, judge prompts, report expected/actual fields, dashboard metrics, or pass policy.

The assertion engine should evaluate what the model actually emitted unless a scenario explicitly asserts against a configured execution trace view.

---

## Target Mental Model

The evaluator should answer one question per case:

> Given this model response and optional environment trace, does it satisfy one configured correct answer path?

It should not know:

- What `storageManager_copy` means.
- That `useTools` is the preferred wrapper.
- That CLI commands live in `arguments.tool`.
- That `memory`, `goal`, or `constraints` are special fields.
- That a "clarification" response means a particular hardcoded keyword list.
- That "prompt execution" is better than direct content modification.

It may know generic mechanics:

- How to parse JSON.
- How to select data with JSONPath.
- How to apply regex.
- How to compare values.
- How to combine assertions with `all`, `any`, and `not`.
- How to report expected versus actual values.

Everything product-specific belongs in YAML/JSON config.

The target model deliberately removes these runtime concepts:

- `expected_tools`
- `acceptable_tools`
- `expect.tool`
- `params_include`
- `first_tool`
- `first_tool_any_of`
- `expected_response_type`
- named `behavior_expectations`
- named `anti_patterns`

Those ideas can be represented in config when needed, but only as ordinary assertions over actual model output. For example, "does not call a tool" is not a behavior-validator rule; it is an assertion that `$.tool_calls[0]` is absent. "Calls storage copy" is not an expected-tool rule; it is an assertion that `$.tool_calls[0].function.arguments.tool` matches a configured regex.

---

## Proposed Config Shape

### Case-Level Shape

```yaml
- id: prompt_execute_weekly_planning
  tags: [tool, prompt, cli]
  messages:
    - role: system
      template: workspace_cli_system
      variables:
        workspaceId: workspace_main
        sessionId: session_eval
    - role: user
      content: Run the weekly planning prompt.

  correct:
    any:
      - name: use_tools_prompt_execute_cli
        assertions:
          - type: jsonpath_equals
            path: $.tool_calls[0].function.name
            value: useTools
          - type: jsonpath_exists
            path: $.tool_calls[0].function.arguments.memory
          - type: jsonpath_exists
            path: $.tool_calls[0].function.arguments.goal
          - type: jsonpath_exists
            path: $.tool_calls[0].function.arguments.constraints
          - type: jsonpath_regex
            path: $.tool_calls[0].function.arguments.tool
            pattern: '^prompt execute-prompt\s+"?weekly planning"?$'
```

The evaluator does not know `prompt execute-prompt` is a prompt manager call. It only checks the configured JSONPath and regex.

### Multiple Allowed Correct Answers

```yaml
- id: missing_target_requires_clarification_or_search
  messages:
    - role: user
      content: Add this to the file from yesterday.

  correct:
    any:
      - name: asks_clarifying_question
        assertions:
          - type: jsonpath_absent
            path: $.tool_calls[0]
          - type: text_regex
            pattern: '(?i)\b(which|what|where|clarify|specify|confirm)\b'
      - name: searches_before_acting
        assertions:
          - type: jsonpath_equals
            path: $.tool_calls[0].function.name
            value: useTools
          - type: jsonpath_regex
            path: $.tool_calls[0].function.arguments.tool
            pattern: '^search search-(content|directory)\b'
```

This replaces `acceptable_tools`, `TEXT_ONLY`, and named behavior expectations with explicit alternatives.

### Exact JSON Object Expectation

```yaml
- id: structured_status_response
  messages:
    - role: user
      content: Return the current status as JSON.

  correct:
    any:
      - name: exact_status_payload
        assertions:
          - type: json_equals
            selector: $.content_json
            value:
              status: ready
              requires_action: false
```

This supports use cases that are not tool-calling at all.

### Ordered Multi-Action CLI Expectation

```yaml
- id: archive_then_open
  messages:
    - role: user
      content: Archive old notes, then open the summary.

  correct:
    any:
      - name: ordered_cli_batch
        assertions:
          - type: jsonpath_equals
            path: $.tool_calls[0].function.name
            value: useTools
          - type: jsonpath_regex
            path: $.tool_calls[0].function.arguments.tool
            pattern: '^storage archive\s+"Old Notes";\s*storage open\s+"Summary\.md"$'
```

No `first_tool`, `ordered_tools`, or semantic batch parser is needed for correctness. If a project wants command-level parsing, it can add a configured extractor and assert against extracted command tokens.

### Environment State Expectation

```yaml
- id: copy_runbook_environment
  messages:
    - role: user
      content: Copy Incident-Response.md to Incident-Response-Template.md.

  environment:
    fixture: fixtures/runbooks.yaml
    execute: true

  correct:
    any:
      - name: cli_and_state_changed
        assertions:
          - type: jsonpath_regex
            path: $.tool_calls[0].function.arguments.tool
            pattern: '^storage copy\s+"Projects/Runbooks/Incident-Response\.md"\s+"Projects/Runbooks/Incident-Response-Template\.md"$'
          - type: environment_path_exists
            path: Projects/Runbooks/Incident-Response-Template.md
          - type: environment_file_contains
            path: Projects/Runbooks/Incident-Response-Template.md
            text: "# Incident Response"
```

Environment assertions remain config-defined, but they join the same result model as response assertions.

---

## Target Architecture

### 1. `PromptCase` Becomes Format-Agnostic

Replace first-class `expected_tools` and `acceptable_tools` with a generic case payload:

```python
@dataclass
class PromptCase:
    case_id: str
    messages: list[dict[str, Any]]
    tags: list[str]
    correct: dict[str, Any]
    scoring: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Compatibility shims can exist only as migration tooling, not active evaluator semantics. The runtime case model should not preserve old tool fields.

### 2. Config Loader Only Loads and Validates Shape

`ConfigLoader` should:

- Load YAML/JSON.
- Resolve templates and variables.
- Validate the file against a generic scenario schema.
- Preserve `correct.any[].assertions` without translating them into `expected_tools`.
- Validate assertion definitions structurally, not semantically.

It should not:

- Build expected tool lists.
- Interpret `expect.tool`.
- Dedupe acceptable tools.
- Infer `expected_response_type`.
- Convert behavior names into runtime checks.

### 3. Response Views Are Explicit and Configurable

The evaluator needs an intermediate response object, but it should be generic and lossless. It must preserve the raw API message exactly and then add normalized views for assertion convenience.

```json
{
  "raw_api_message": {},
  "raw": "...",
  "content": "...",
  "content_json": {},
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "useTools",
        "arguments": {
          "tool": "storage copy \"a.md\" \"b.md\""
        }
      }
    }
  ],
  "environment": {
    "executed_actions": [],
    "snapshot": {}
  }
}
```

Exact raw preservation matters because assertions need to distinguish absent fields, null fields, empty arrays, raw tool-call arguments, and text-only responses. Normalized views may make these easier to select, but they must not discard raw response shape.

Extraction should be driven by config:

```yaml
response_formats:
  openai_tool_calls:
    enabled: true
    source: raw_response
  qwen_xml_tool_calls:
    enabled: true
    source: text
    extractor: regex_json_blocks
    start: '<tool_call>'
    end: '</tool_call>'
  chatml_tool_call:
    enabled: true
    source: text
    extractor: regex
    pattern: '(?s)tool_call:\s*(?P<name>\w+)\s*arguments:\s*(?P<arguments>\{.*\})'
```

The extractor registry can be generic and small. The configured format decides which extractor runs and where data lands.

Extraction may normalize syntax, but not meaning. Parsing Qwen XML into `$.tool_calls[0]` is acceptable. Preserving `function.name: useTools` and `function.arguments.tool: storage copy "a.md" "b.md"` is required. Converting that command into `storageManager_copy` is not allowed in the validation view.

### 4. Assertion Engine Is the Only Correctness Authority

Introduce a generic assertion engine:

```python
@dataclass
class AssertionResult:
    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str

@dataclass
class AssertionPathResult:
    name: str
    passed: bool
    assertions: list[AssertionResult]

@dataclass
class CorrectnessResult:
    passed: bool
    matched_path: str | None
    paths: list[AssertionPathResult]
```

The engine evaluates:

```yaml
correct:
  any:
    - name: preferred
      assertions: [...]
    - name: acceptable_fallback
      assertions: [...]
```

It should also support:

```yaml
correct:
  all:
    - assertions: [...]
```

and assertion-level combinators:

```yaml
- type: any
  assertions:
    - type: text_regex
      pattern: '(?i)clarify'
    - type: text_regex
      pattern: '\?'
```

### 5. Assertion Primitive Vocabulary

Keep the primitive set generic. Initial primitives:

| Primitive | Purpose |
|---|---|
| `equals` | Compare selected value to scalar/object/list. |
| `not_equals` | Inverse equality. |
| `contains` | String/list/object containment. |
| `not_contains` | Inverse containment. |
| `regex` | Regex match on a selected string. |
| `not_regex` | Regex must not match. |
| `exists` | Selected value exists and is not null. |
| `absent` | Selected value does not exist or is null. |
| `jsonpath_equals` | Select from response view and compare. |
| `jsonpath_regex` | Select from response view and regex-match. |
| `jsonpath_contains` | Select from response view and require containment. |
| `json_subset` | Require a configured JSON object subset. |
| `length_equals` | Length of selected value equals integer. |
| `length_min` | Length of selected value is at least integer. |
| `length_max` | Length of selected value is at most integer. |
| `all` | Nested assertions must all pass. |
| `any` | At least one nested assertion must pass. |
| `not` | Nested assertion must fail. |

Environment-specific assertions may remain as domain adapters, but they should be exposed through the same engine and config:

| Primitive | Purpose |
|---|---|
| `environment_path_exists` | Validate runtime snapshot path exists. |
| `environment_path_absent` | Validate runtime snapshot path does not exist. |
| `environment_file_contains` | Validate file content contains text. |
| `environment_file_regex` | Validate file content matches regex. |
| `environment_jsonpath_equals` | Validate arbitrary environment snapshot fields. |

This is acceptable hardcoding because these are generic assertion operations, not use-case-specific correctness rules.

### 6. Scoring Uses Assertion Paths

Replace scoring keys like `all_tools`, `any_tools`, `ordered_tools`, `first_tool`, and `max_tool_calls` with weighted assertion paths:

```yaml
scoring:
  max_score: 1.0
  paths:
    - name: exact_cli
      score: 1.0
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].function.arguments.tool
          pattern: '^storage copy\s+"a\.md"\s+"b\.md"$'
    - name: right_command_wrong_quotes
      score: 0.5
      assertions:
        - type: jsonpath_regex
          path: $.tool_calls[0].function.arguments.tool
          pattern: '^storage copy\b'
```

This keeps grading policy in config, including partial credit.

### 7. LLM-as-Judge Is Also Configured Assertions

LLM-as-judge should remain available, but it must follow the same config-first model. The evaluator should not contain hardcoded judge meanings such as "semantic pass", "tool correctness", "behavior correctness", or any specific rubric fields. It should only:

1. Load a configured judge prompt/template.
2. Render that prompt with configured inputs.
3. Call the configured judge model.
4. Parse the judge response using a configured response format.
5. Evaluate the parsed judge response with configured assertions.

Example:

```yaml
judge:
  enabled: true
  model: gpt-4.1-mini
  prompt:
    template: semantic_tool_response_judge
    variables:
      task: $.case.messages[-1].content
      expected: $.case.correct
      actual_response: $.response.raw
      extracted_response: $.response
  response_format:
    type: json
  correct:
    any:
      - name: judge_passes_response
        assertions:
          - type: jsonpath_equals
            path: $.pass
            value: true
          - type: jsonpath_exists
            path: $.reasoning
          - type: jsonpath_regex
            path: $.reasoning
            pattern: '.{30,}'
```

The judge prompt can ask for any structure the scenario owner wants:

```json
{
  "pass": true,
  "confidence": 0.91,
  "reasoning": "The response invokes the requested CLI command with the correct target path.",
  "issues": []
}
```

or:

```json
{
  "verdict": "fail",
  "reason_codes": ["wrong_target"],
  "reasoning": "The command copied the wrong source file."
}
```

Both are valid as long as the scenario config declares how to parse and judge the returned structure.

The judge input may include:

- Case messages.
- `correct` assertion paths.
- Model response view.
- Assertion result summary.

It should not receive `expected_tools`, because that field biases the judge toward the old semantic model. If the judge needs to know what was expected, it receives the configured `correct` object or a rendered explanation from config.

Judge pass/fail should be composed by config, not code. Examples:

```yaml
pass_policy:
  type: all
  assertions:
    - ref: response.correct
    - ref: judge.correct
```

```yaml
pass_policy:
  type: any
  assertions:
    - ref: response.correct
    - ref: judge.correct
```

If a case wants judge-only grading, it can express that without a special runtime mode:

```yaml
pass_policy:
  type: all
  assertions:
    - ref: judge.correct
```

Old mode names like `and`, `or`, and `judge_only` can exist only as migration compatibility shims. They are not target architecture concepts. The target pass policy is a configured assertion/combinator tree over response, judge, environment, and scoring result fields.

### 8. Reporting Shows Assertion Results

Reports should answer:

- Which correct path matched?
- Which assertions failed?
- What selected actual values were compared?
- What was the expected exact value or regex?
- Did environment execution produce the expected state?

Example report block:

```text
Case: prompt_execute_weekly_planning
Status: FAIL
Matched path: none

Path: use_tools_prompt_execute_cli
  PASS $.tool_calls[0].function.name equals useTools
  FAIL $.tool_calls[0].function.arguments.tool regex ^prompt execute-prompt\s+"?weekly planning"?$
       actual: prompt run "weekly planning"
```

This is much more useful than "expected promptManager_executePrompt, got useTools".

---

## Migration Plan

### Phase 0: Freeze and Inventory

Deliverables:

- Add an inventory report of all active config keys that imply old semantics:
  - `expected_tools`
  - `acceptable_tools`
  - `allowed_tools`
  - `expect.tool`
  - `expect.acceptable[].tool`
  - `params_include`
  - `first_tool`
  - `first_tool_any_of`
  - `behavior_expectations`
  - `anti_patterns`
  - `expected_response_type`
  - `require_expected_tools`
  - scoring keys such as `all_tools`, `any_tools`, `ordered_tools`
  - judge prompt variables such as `{expected_tools}`
  - dashboard/progress fields such as `behavior_tested`, `behavior_passed`, and `schema_passed`
- Add a search/lint target that fails if active evaluator configs contain manager-style expected IDs such as `promptManager_`, `storageManager_`, `contentManager_`, `memoryManager_`, or `searchManager_`.
- Extend the search/lint target to docs and skills so old authoring guidance does not remain canonical.
- Capture a small set of golden current responses from local vLLM for representative tool, behavior, and environment cases.

No runtime behavior changes in this phase.

### Phase 1: Introduce Generic Assertion Schema

Deliverables:

- Add `Evaluator/config/assertion_schema.yaml` or `Evaluator/config/assertion_schema.json`.
- Document required case fields:
  - `id`
  - `messages` or `question` plus optional `system`
  - `tags`
  - `correct.any[].name`
  - `correct.any[].assertions[]`
- Define assertion primitives and their allowed fields.
- Add config validation tests for malformed assertion configs.

The schema validates structure only. It must not know about `useTools`, CLI commands, manager tools, memory fields, or Synaptic-specific concepts.

### Phase 2: Add Assertion Engine Beside Existing Validator

Deliverables:

- Add a generic assertion engine that receives:
  - `response_view`
  - `correct` config
  - optional `environment_result`
- Add generic selector support:
  - raw text selector
  - JSONPath over response view
  - JSONPath over environment snapshot
- Add unit tests for every primitive and combinator.

During this phase, the old validator can still run for comparison, but it must not be extended with more use-case-specific logic.

### Phase 3: Add Config-Driven Response View Extraction

Deliverables:

- Define response format config for:
  - OpenAI-style `tool_calls`
  - text-only responses
  - Qwen XML tool-call blocks
  - existing ChatML `tool_call:` blocks if still needed
- Return a normalized response view without semantic expansion.
- Preserve wrapper calls exactly as emitted:
  - `function.name: useTools`
  - `function.arguments.tool: storage copy ...`
- Remove or bypass wrapper-to-manager expansion in the assertion path.

Important rule: extraction may normalize syntax into a common response view, but must not normalize meaning. Parsing a Qwen `<tool_call>` into `tool_calls[0]` is acceptable. Turning `storage copy` into `storageManager_copy` is not.

### Phase 4: Convert Active Scenarios to Assertions

Deliverables:

- Convert `Evaluator/config/scenarios/tool_prompts.yaml`:
  - Replace `expected_tools: [useTools]` plus `expect.tool: storageManager_copy` with assertions on `function.name` and `function.arguments.tool`.
  - Replace `params_include` with regex or structured argument assertions against the actual configured response shape.
- Convert `Evaluator/config/scenarios/behavior_prompts.yaml`:
  - Replace `expected_response_type`, `behavior_expectations`, and `anti_patterns` with explicit assertion alternatives.
  - Use `text_regex`, `jsonpath_absent`, `jsonpath_exists`, and CLI regex assertions directly.
- Convert `Evaluator/config/scenarios/vault_gym.yaml`:
  - Replace `expected_tools` and `require_expected_tools` with response assertions and environment state assertions.
  - Keep environment fixtures and state checks where they are useful.
- Convert `Evaluator/config/behaviors.yaml` into reusable assertion templates or remove it from active validation.
- Convert `Evaluator/config/response_types.yaml` into reusable assertion templates or remove it from active validation.

Example reusable template shape:

```yaml
assertion_templates:
  text_only_clarification:
    assertions:
      - type: jsonpath_absent
        path: $.tool_calls[0]
      - type: text_regex
        pattern: '(?i)\b(which|what|where|clarify|specify|confirm)\b'

  use_tools_cli:
    assertions:
      - type: jsonpath_equals
        path: $.tool_calls[0].function.name
        value: useTools
      - type: jsonpath_exists
        path: $.tool_calls[0].function.arguments.memory
      - type: jsonpath_exists
        path: $.tool_calls[0].function.arguments.goal
      - type: jsonpath_exists
        path: $.tool_calls[0].function.arguments.constraints
```

Templates are config convenience only. The evaluator does not know what they mean.

### Phase 5: Make Assertion Results the Runtime Authority

Deliverables:

- `EvaluationRecord.status` derives from `CorrectnessResult.passed`, environment execution errors, and optional judge policy.
- Remove `_check_expected_tools` from the active path.
- Replace `ConfigDrivenValidator.validate(expect=...)` with assertion engine evaluation.
- Replace behavior validator calls with assertion paths.
- Replace scoring over `tool_names` with weighted assertion paths.
- Replace hardcoded judge combination modes with configured pass-policy assertions over response and judge results.

At the end of this phase, active pass/fail should no longer depend on `expected_tools`, `acceptable_tools`, `expect.tool`, or manager-style names.

### Phase 6: Retire Legacy Validators or Demote Them to Generic Helpers

Deliverables:

- Remove use-case-specific wrapper expansion from `Evaluator/schema_validator.py`.
- Remove prompt-manager-specific ID validation from the default validation path.
- Replace `ConfigDrivenValidator` with:
  - response extraction helpers
  - assertion engine invocation
  - config shape validation
- Either delete `behavior_validator.py` or keep it only as a deprecated compatibility adapter outside active presets.
- Ensure `shared/environments/validator.py` no longer receives `expected_tools`; environment checks enter through assertion config.

If compatibility migration tooling is needed, keep it in a separate script with explicit naming, for example `scripts/migrate_legacy_eval_expectations.py`. It should not run inside evaluator runtime.

### Phase 7: Update Docs and Skills

Deliverables:

- Update `.skills/evaluation/SKILL.md` and `reference/scenario-authoring.md`.
- Sync `.agents/skills` and `.claude/skills`.
- Document:
  - how to write `correct.any` assertions
  - how to test CLI command strings
  - how to allow multiple correct answers
  - how to use environment assertions
  - how to add a new assertion primitive without adding use-case logic
- Remove examples that teach `expected_tools`, `acceptable_tools`, manager-style IDs, or `expect.tool`.

---

## File-Level Refactor Map

| File | Refactor Direction |
|---|---|
| `Evaluator/base_client.py` | Preserve raw API messages consistently so assertions can distinguish missing fields, nulls, empty arrays, and raw tool-call objects. |
| `Evaluator/cli.py` | Replace hardcoded judge-mode flags and legacy expectation language with config-selected pass policies and assertion reporting options. |
| `Evaluator/prompt_sets.py` | Replace `expected_tools`/`acceptable_tools` fields with generic `messages`, `correct`, `scoring`, and `environment` payloads. |
| `Evaluator/config_loader.py` | Stop interpreting old expectation keys; load assertions and templates generically. |
| `Evaluator/config_validator.py` | Replace with generic assertion evaluation or split into response extraction plus assertion runner. Remove CLI-to-manager normalization. |
| `Evaluator/schema_validator.py` | Keep syntax extraction only. Remove wrapper expansion into concrete semantic names and product-specific ID checks from default path. |
| `Evaluator/behavior_validator.py` | Remove from active path or turn existing behavior configs into assertion templates. |
| `Evaluator/judge_validator.py` | Keep only generic judge orchestration: render configured prompt, call configured model, parse configured response format, and evaluate configured judge assertions. |
| `Evaluator/runner.py` | Make correctness result, not tool validation, the source of pass/fail. Compose response, judge, environment, and scoring results through configured pass-policy assertions. |
| `Evaluator/reporting.py` | Report matched assertion path and assertion-level expected/actual values. |
| `Evaluator/ui.py` | Display assertion pass/fail summaries instead of expected/called tools. |
| `Evaluator/config/rubrics/*.yaml` | Convert judge prompts to configured input/output contracts with assertion-checked judge responses. Remove `{expected_tools}` as a special prompt variable. |
| `shared/agentic_loop.py` | Remove `expected_tools` and `require_expected_tools` from loop pass/fail; expose turn responses and environment traces to assertion evaluation. |
| `shared/environments/validator.py` | Remove `expected_tools`; expose executed actions and state snapshots to the assertion engine. |
| `shared/environments/tool_executor.py` | Keep CLI parsing only for execution. Do not return expanded manager names as validation truth. |
| `shared/cloud_eval_progress.py` | Replace schema/behavior pass metrics with assertion/correctness result fields. |
| `shared/ui/evaluation.py` | Replay and display assertion result vocabulary instead of behavior-specific metrics. |
| `Evaluator/config/scenarios/*.yaml` | Convert to `correct.any[].assertions`. Remove manager-style expected IDs. |
| `Evaluator/config/tool_schema.yaml` | Treat CLI command metadata as prompt/config source data only, not evaluator semantic truth. |
| `cli-first-tool-schemas.json` | Remains a source of tool command metadata. It can help generate scenario examples or templates, but evaluator correctness still comes from case assertions. |
| `.skills/evaluation/**` | Update canonical evaluation authoring guidance first, then sync `.agents/skills` and `.claude/skills` and verify with `--check`. |
| `docs/architecture/llm-judge-integration.md` | Update judge architecture docs to describe configured prompt/parser/assertion contracts instead of expected-tools context. |

---

## What "Config-First" Means Here

Config-first does not mean the evaluator has no code. It means evaluator code implements generic mechanics, and config supplies use-case truth.

Allowed in code:

- Load YAML/JSON.
- Validate config shape.
- Parse JSON.
- Extract configured response formats.
- Select values using JSONPath-like selectors.
- Evaluate generic string, regex, equality, subset, length, and existence assertions.
- Combine assertions.
- Execute configured environment fixtures and expose resulting snapshots.
- Report assertion results.

Not allowed in code:

- Knowing that `useTools` is the wrapper for the current product.
- Knowing that `arguments.tool` contains CLI commands.
- Knowing that `storage copy` maps to `storageManager_copy`.
- Knowing that prompt execution is preferable to content modification.
- Knowing that `memory`, `goal`, or `constraints` are required fields.
- Knowing named behavior labels like `delegates_complex_task`.
- Hardcoding current manager names, command names, wrapper fields, or response-type categories into pass/fail logic.

---

## Test Strategy

### Unit Tests

- Assertion primitives:
  - equality, regex, contains, absent, exists, length, JSON subset
- Combinators:
  - `any`, `all`, `not`
- JSONPath selector behavior:
  - missing path
  - array index
  - object field
  - selected list values
- Response extraction:
  - OpenAI tool calls
  - text-only
  - configured Qwen XML tool calls
  - malformed tool-call JSON reports extraction error but preserves raw text

### Golden Config Tests

Add tiny fixture configs:

- Exact JSON response case.
- Text-only clarification case.
- CLI tool-call case with `useTools`.
- Multi-allowed-answer case.
- Environment state case.

Each fixture should assert expected pass/fail without calling a model.

### Config Lint Tests

- Active evaluator scenario files must not contain old manager-style expected IDs.
- Active evaluator scenario files must not contain old runtime expectation keys after migration.
- Assertion templates must resolve.
- Every assertion has required fields.
- Regex patterns compile.

### Integration Tests

- Dry-run config validation over all active scenarios.
- Replay saved model responses against the new assertion engine.
- Run a small local vLLM smoke suite:
  - one tool case
  - one behavior/text case
  - one environment case
- Run full local vLLM tools + behavior suite after scenario conversion.

---

## Acceptance Criteria

The refactor is complete when:

- Active evaluator configs define correctness through `correct.any[].assertions`, not `expected_tools`, `acceptable_tools`, or `expect.tool`.
- Active evaluator configs contain no manager-style expected output names such as `promptManager_executePrompts` or `storageManager_copy`.
- Runtime pass/fail does not depend on semantic tool-name normalization.
- A CLI case can check `prompt execute-prompt "..."` directly from the emitted `arguments.tool` string.
- A behavior case can be expressed as text/JSON assertions without a named Python behavior predicate.
- An LLM-as-judge case can define the judge prompt, expected judge response structure, optional fields such as `reasoning`, and judge pass/fail assertions entirely in config.
- Environment cases validate response shape and state changes through assertion config, not `expected_tools`.
- Reports list assertion paths, expected values/patterns, actual selected values, and matched path.
- Evaluation artifacts report `correctness` and `assertion` results, not `schema_passed` plus `behavior_passed` as the primary outcome model.
- Environment/runtime errors are exposed as structured response/judge context and assertion failures, not hidden or "fixed" through ad hoc parser repair.
- Adding a new use case requires config changes only unless it needs a new generic assertion primitive.
- Adding a new generic assertion primitive does not mention any product, tool, wrapper, or scenario-specific name.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scenario conversion is large and error-prone. | Convert in batches by scenario file and use golden fixtures plus dry-run config validation after each batch. |
| Regex expectations become too brittle. | Prefer targeted regex with named allowed variation. Use multiple `correct.any` paths for truly acceptable alternatives. |
| JSONPath implementation adds dependency risk. | Start with a small supported selector subset if dependency policy is strict; otherwise use a maintained JSONPath package and pin it. |
| Environment execution still needs tool-command parsing. | Keep command execution parser separate from correctness evaluation. Execution may parse commands to mutate state, but pass/fail comes from configured assertions. |
| Backward compatibility pressure reintroduces old semantics. | Keep migration scripts separate from runtime and add lint tests that fail on old expectation keys in active configs. |
| Reports may become verbose. | Summarize by matched path and failed assertions; include selected actual values only for failing assertions by default. |

---

## Open Design Decisions

1. Should scenarios use `messages` only, or preserve `question`/`system` as shorthand that the loader expands into `messages`?
2. Should JSONPath support full JSONPath or a strict subset like `$.a.b[0].c` and `[*]`?
3. Should assertion templates live in one global file or alongside scenario files?
4. Should environment assertions be evaluated by the shared assertion engine directly, or should environment validator return a response-view-compatible snapshot that the assertion engine evaluates?
5. Should legacy scenario keys be rejected immediately after migration, or allowed only through an explicit `--legacy-eval-config` flag?

Recommended defaults:

- Preserve `question`/`system` shorthand for author ergonomics, but normalize to `messages` before execution.
- Start with a strict JSONPath subset to keep behavior predictable.
- Put reusable templates in `Evaluator/config/assertion_templates.yaml`.
- Have environment validator return snapshots/traces; the assertion engine evaluates them.
- Reject legacy keys in active configs after migration.

---

## Recommended Implementation Order

1. Add assertion schema and config linter.
2. Add assertion engine and response view model.
3. Add golden tests with static responses.
4. Convert three smoke scenarios manually to prove the config shape.
5. Wire runner pass/fail to assertions behind an explicit temporary flag.
6. Convert all active tool scenarios.
7. Convert all active behavior scenarios.
8. Convert environment/vault gym scenarios.
9. Remove old active runtime fields and validators from pass/fail.
10. Update reports, UI, docs, and skills.
11. Run local vLLM smoke and full suite.

The key sequencing point is that scenario conversion should follow the assertion engine, not precede it. Otherwise the config will again be forced to fit whatever hardcoded runtime fields already exist.
