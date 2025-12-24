# Config-Driven Evaluator Design

## Core Principle

**You define what's valid. The evaluator checks against it. No hardcoding.**

---

## 1. TOOL SCHEMA DEFINITION

Define your tools however YOU want them structured:

```yaml
# Evaluator/config/tool_schema.yaml

# Your tool format - completely custom
tool_format:
  # How tool calls appear in the response
  wrapper: "useTools"  # or null if direct tool names

  # If using a wrapper, define its structure
  wrapper_structure:
    context:
      required: [sessionId, workspaceId, memory, goal]
      optional: [constraints]
    calls:
      type: array
      item_fields: [agent, tool, params]
      tool_name_pattern: "{agent}_{tool}"  # How to construct full tool name

  # OR if using direct tool calls (no wrapper):
  # wrapper: null
  # direct_structure:
  #   context_in_args: true
  #   context_fields: [sessionId, workspaceId]

# Define your actual tools
tools:
  vaultManager:
    - name: moveFolder
      params:
        required: [path, newPath]
        optional: [overwrite]
    - name: createFolder
      params:
        required: [path]
    - name: deleteFolder
      params:
        required: [path]
        optional: [recursive]

  contentManager:
    - name: readContent
      params:
        required: [filePath]
        optional: [limit, offset]
    - name: appendContent
      params:
        required: [filePath, content]

  vaultLibrarian:
    - name: searchContent
      params:
        required: [query]
        optional: [path, limit]
```

---

## 2. RESPONSE TYPES

Define what valid responses look like:

```yaml
# Evaluator/config/response_types.yaml

response_types:
  tool_only:
    description: "Response contains only tool call(s), no explanation"
    requirements:
      has_tool_calls: true
      text_content:
        max_length: 20  # Allow minimal formatting text

  tool_with_explanation:
    description: "Tool call with explanation before/after"
    requirements:
      has_tool_calls: true
      text_content:
        min_length: 30

  text_only:
    description: "Text response, no tool calls"
    requirements:
      has_tool_calls: false
      text_content:
        min_length: 10

  clarification:
    description: "Asking user for more info"
    requirements:
      has_tool_calls: false
      text_content:
        contains_any: ["?", "which", "what", "could you", "can you clarify"]
```

---

## 3. BEHAVIORS (Generic Checks)

Define reusable behavior checks:

```yaml
# Evaluator/config/behaviors.yaml

behaviors:
  # Tool-related
  uses_tool:
    check: tool_called
    params:
      tool: "{tool_name}"  # Parameterized - filled in by test

  uses_any_of_tools:
    check: any_tool_called
    params:
      tools: "{tool_list}"

  # Text-related
  asks_clarification:
    check: text_contains_any
    params:
      patterns: ["?", "which", "what", "clarify", "specify"]

  explains_reasoning:
    check: text_min_length
    params:
      min: 50

  minimal_text:
    check: text_max_length
    params:
      max: 30

  # Context-related
  context_complete:
    check: fields_present
    params:
      in: context
      fields: [sessionId, workspaceId, memory, goal]

  # Custom regex
  mentions_file:
    check: text_matches
    params:
      pattern: "\\b[\\w/]+\\.(md|txt|json)\\b"
```

---

## 4. TEST SCENARIOS

Define test cases that reference the above:

```yaml
# Evaluator/config/scenarios/tool_coverage.yaml

name: Tool Coverage Tests
description: Verify each tool can be called correctly

# Shared context for all tests in this file
defaults:
  system_prompt_template: standard_vault
  response_type: tool_only

tests:
  - id: vaultManager_moveFolder
    question: "Move Projects/Old to Archive/Old"
    expect:
      tool: vaultManager_moveFolder
      params_include:
        path: "Projects/Old"
        newPath: "Archive/Old"
    behaviors:
      - context_complete

  - id: vaultManager_createFolder
    question: "Create a new folder called Research/2024"
    expect:
      tool: vaultManager_createFolder
      params_include:
        path: "Research/2024"

  - id: contentManager_readContent
    question: "Read the contents of README.md"
    expect:
      tool: contentManager_readContent
      params_include:
        filePath: "README.md"
```

```yaml
# Evaluator/config/scenarios/behavior_tests.yaml

name: Behavior Tests
description: Test model behaviors in various situations

tests:
  - id: ambiguous_request_clarifies
    question: "Delete the old files"
    # No specific tool expected - multiple valid responses
    expect:
      any_of:
        - response_type: clarification
        - tool: vaultLibrarian_searchContent  # Search first is also valid
    behaviors:
      - asks_clarification OR uses_tool(vaultLibrarian_searchContent)

  - id: destructive_action_verifies
    question: "Delete everything in the Archive folder"
    expect:
      # Should verify before deleting
      tool_sequence:
        - any: [vaultManager_listDirectory, vaultLibrarian_searchDirectory]
        # NOT immediately deleteFolder
    behaviors:
      - context_complete

  - id: batch_operation
    question: "Move all .md files from Inbox to Notes"
    expect:
      response_type: tool_with_explanation
      # Multiple tool calls expected
      min_tool_calls: 2
    behaviors:
      - explains_reasoning
```

---

## 5. SYSTEM PROMPT TEMPLATES

Reusable system prompts:

```yaml
# Evaluator/config/templates/system_prompts.yaml

templates:
  standard_vault:
    content: |
      <session_context>
      - sessionId: "{session_id}"
      - workspaceId: "{workspace_id}"
      </session_context>

      <vault_structure>
      {vault_structure}
      </vault_structure>

      <available_tools>
      {available_tools}
      </available_tools>

    # Default values for placeholders
    defaults:
      session_id: "session_eval_001"
      workspace_id: "default"
      vault_structure: |
        - Projects/
        - Archive/
        - Notes/
        - README.md
      available_tools: "{{auto_from_tool_schema}}"

  minimal:
    content: |
      You are a helpful assistant with access to vault tools.
      Session: {session_id}
```

---

## 6. EVALUATION RUN

How it all comes together:

```yaml
# Evaluator/config/eval_run.yaml

run:
  name: "Full Evaluation Suite"

  # Which scenario files to run
  scenarios:
    - scenarios/tool_coverage.yaml
    - scenarios/behavior_tests.yaml

  # Model to test
  model:
    backend: lmstudio  # or ollama, openrouter
    name: "your-model-name"

  # Output
  output:
    format: [json, markdown]
    path: "results/{timestamp}/"

  # Scoring
  scoring:
    pass_threshold: 0.8
    weights:
      tool_correct: 1.0
      params_correct: 0.5
      behavior_pass: 0.3
```

---

## 7. THE GENERIC VALIDATOR

The Python code is **completely generic**:

```python
# Pseudocode - the validator knows NOTHING specific

class ConfigDrivenValidator:
    def __init__(self, config_dir: Path):
        self.tool_schema = load_yaml(config_dir / "tool_schema.yaml")
        self.response_types = load_yaml(config_dir / "response_types.yaml")
        self.behaviors = load_yaml(config_dir / "behaviors.yaml")

    def validate(self, response: str, test_case: dict) -> Result:
        # Parse response according to tool_schema
        parsed = self.parse_response(response)

        # Check expected tool (if specified)
        if "tool" in test_case["expect"]:
            self.check_tool(parsed, test_case["expect"]["tool"])

        # Check response type (if specified)
        if "response_type" in test_case["expect"]:
            self.check_response_type(parsed, test_case["expect"]["response_type"])

        # Check behaviors
        for behavior in test_case.get("behaviors", []):
            self.check_behavior(parsed, behavior)

        return self.result

    def parse_response(self, response):
        """Parse according to tool_schema.yaml - no hardcoding"""
        if self.tool_schema["tool_format"]["wrapper"]:
            # Extract wrapper, then expand tools using configured pattern
            pattern = self.tool_schema["tool_format"]["wrapper_structure"]["calls"]["tool_name_pattern"]
            # ... generic expansion
        else:
            # Direct tool calls
            # ...

    def check_behavior(self, parsed, behavior_name):
        """Look up behavior in config, run generic check"""
        behavior_def = self.behaviors["behaviors"][behavior_name]
        check_type = behavior_def["check"]

        # Generic dispatch - no hardcoded behavior names
        return self.run_check(check_type, parsed, behavior_def["params"])
```

---

## 8. ADDING NEW TOOLS/BEHAVIORS

**To add a new tool:**
1. Add to `tool_schema.yaml` under the appropriate agent
2. Create test in `scenarios/tool_coverage.yaml`
3. Done. No Python changes.

**To add a new behavior check:**
1. Add to `behaviors.yaml`
2. Reference in test scenarios
3. Done. No Python changes.

**To add a new response type:**
1. Add to `response_types.yaml`
2. Reference in tests
3. Done. No Python changes.

---

## 9. EXAMPLE: COMPLETELY DIFFERENT TOOL FORMAT

Say someone uses a different format (not useTools):

```yaml
# Their tool_schema.yaml
tool_format:
  wrapper: null  # No wrapper
  direct_structure:
    # Tools called directly by name
    naming: "{tool_name}"  # e.g., "searchContent"
    context_location: "first_param"
    context_fields: [session, workspace]

tools:
  searchContent:
    params:
      required: [query]
      optional: [path]

  writeFile:
    params:
      required: [path, content]
```

**Same evaluator, different config. No code changes.**

---

## Summary

| What | Where | Python Code |
|------|-------|-------------|
| Tool names & structure | `tool_schema.yaml` | Generic parser |
| Valid response types | `response_types.yaml` | Generic checker |
| Behavior definitions | `behaviors.yaml` | Generic dispatcher |
| Test cases | `scenarios/*.yaml` | Generic runner |
| System prompts | `templates/*.yaml` | String substitution |

**The Python code never knows about your specific tools, behaviors, or test cases.**
