# SynthChat Config-Driven Refactoring Plan

## Executive Summary

Three categories of hardcoded, scenario-specific behavior in SynthChat must become config-driven so users can define entirely different tool-call formats, workspace prompt structures, and metadata label mappings via YAML -- without touching Python code. This plan targets the **post-decomposition** module structure (where `workspace/`, `schemas/tool_response_schema.py`, and `labeling/metadata_labels.py` already exist as separate modules).

**Principle**: NO HARDCODING for specific scenarios. Everything is CONFIG-DRIVEN. Users configure any scenario via YAML without touching code.

---

## Current State: What Is Hardcoded

### Category 1: Tool Call Schema (Biggest Offender)

**Post-decomposition location**: `SynthChat/schemas/tool_response_schema.py`

Three tightly coupled pieces define a single, hardcoded tool-call format:

| Function | What It Hardcodes |
|---|---|
| `_build_use_tools_response_schema()` | Full JSON Schema: `{content, tool_calls: [{id, type:"function", function:{name, arguments:{context:{sessionId, workspaceId, memory, goal}, calls:[{agent, tool, params}], strategy}}}]}` |
| `_build_use_tools_generation_prompt()` | 10-line prompt instructing the LLM to produce exactly this wrapper format |
| `_tool_wrapper_name()` | Default wrapper name `"useTools"` |
| `_render_available_tools()` | Hardcoded instruction: "Required wrapper context fields: sessionId, workspaceId, memory, goal." |

**Impact**: Every scenario that uses tool calls must produce the exact `useTools` wrapper with `context.sessionId`, `context.workspaceId`, `context.memory`, `context.goal`. No scenario can define a different tool-call structure.

**Partial config exists**: The `tool_format.wrapper` field in scenario YAML can override the wrapper name (e.g., from `useTools` to `callTools`), and `tool_format.context.required` lists required fields. But the **schema structure** (nested `tool_calls[].function.arguments.context` + `calls[]` + `strategy`) and the **generation prompt** are entirely hardcoded.

### Category 2: Workspace/Environment Structure

**Post-decomposition location**: `SynthChat/workspace/renderer.py`, `SynthChat/workspace/sections.py`

| Function | What It Hardcodes |
|---|---|
| `_render_mocked_workspace_system_prompt()` | Fixed section ordering: session_context, vault_structure, available_workspaces, available_prompts, available_tools, selected_workspace, note_contents, extra_sections, assistant_instructions |
| `_build_session_context_section()` | Fixed XML tag `<session_context>` with hardcoded instruction text about sessionId/workspaceId |
| `_build_selected_workspace_section()` | Fixed JSON structure: `{context, workspaceStructure, recentFiles, keyFiles, workflows, preferences, sessions}` |
| `_render_available_tools()` | Fixed instruction text about wrapper context fields |

**Impact**: Every workspace-based scenario produces a Synaptic-specific prompt structure. Users cannot define a different workspace product format (e.g., a coding IDE, a CRM, a project management tool).

### Category 3: Metadata Label Mappings

**Post-decomposition location**: `SynthChat/labeling/metadata_labels.py`

| Code Region | What It Hardcodes |
|---|---|
| `_classify_environment_issue()` (lines 3278-3307) | 12 string-matching rules mapping issue messages to labels (e.g., `"expected tool(s) not executed"` -> `"missing_expected_tool"`) |
| `_build_metadata_labels()` (lines 1771-1785) | 6 label-to-behavior mappings (e.g., `"frontmatter_missing"` -> `"behavior:structure_failure"`) |
| `_build_metadata_labels()` (lines 1771-1776) | 3 stage-to-failure-type mappings (e.g., `"environment"` -> `"failure_type:environment"`) |

**Impact**: Adding a new issue type or behavior label requires editing Python code.

---

## Proposed Architecture

### Design Principles

1. **Config as schema definition**: The YAML config defines the structure; Python code interprets it generically
2. **Backward compatibility via defaults**: Existing scenarios work unchanged -- defaults produce the current behavior
3. **Layered resolution**: Global defaults -> scenario-level overrides -> runtime resolution
4. **No Jinja2 dependency**: The project uses a simple `{placeholder}` template system (`_render_template_object`). The config-driven approach uses YAML-defined structures, not template engines, to avoid adding a new dependency

### Overview Diagram

```
                    Scenario YAML
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
   tool_call_format  workspace_format  label_mappings
          |              |              |
          v              v              v
   +-----------+  +-----------+  +------------+
   | Defaults  |  | Defaults  |  | Defaults   |
   | Registry  |  | Registry  |  | Registry   |
   +-----------+  +-----------+  +------------+
          |              |              |
          v              v              v
  tool_response_  workspace/   labeling/
  schema.py       renderer.py  metadata_labels.py
  (generic        (generic     (generic
   builder)        renderer)    classifier)
```

---

## Category 1: Config-Driven Tool Call Schema

### 1.1 New Config File: `SynthChat/config/tool_call_formats.yaml`

This file defines **named tool-call formats**. Each format specifies the JSON Schema structure, required context fields, and LLM generation prompt instructions.

```yaml
# SynthChat/config/tool_call_formats.yaml
# Named tool-call response formats.
# Scenarios reference these by name via `tool_call_format: <name>`.
# The "default" format reproduces today's hardcoded behavior.

formats:
  # ── Current Synaptic useTools wrapper (backward-compatible default) ──
  default:
    wrapper_name: useTools

    # Fields required inside function.arguments.context
    context_fields:
      required:
        - sessionId
        - workspaceId
        - memory
        - goal
      properties:
        sessionId: { type: string, minLength: 1 }
        workspaceId: { type: string, minLength: 1 }
        memory: { type: string, minLength: 1 }
        goal: { type: string, minLength: 1 }

    # Schema for each item in function.arguments.calls[]
    call_item:
      properties:
        agent: { type: string, minLength: 1 }
        tool: { type: string, minLength: 1 }
        params: { type: object, additionalProperties: true }
      required: [agent, tool, params]

    # Additional top-level fields inside function.arguments
    extra_argument_fields:
      strategy:
        type: string
        enum: [serial, parallel]

    # Required top-level fields inside function.arguments
    argument_required: [context, calls]

    # LLM generation prompt lines (appended before the base prompt)
    generation_instructions:
      - "Return a single JSON object only."
      - "Your job is to either call tools or respond via text."
      - "If tools are needed, use exactly one tool_calls entry whose function.name is '{wrapper_name}'."
      - "If no tool call is needed, respond with normal text in content and set tool_calls to null or []."
      - "Inside function.arguments.calls, each item must use this exact shape:"
      - '{"agent": "AgentName", "tool": "toolName", "params": {...}}'
      - "Do not use dotted names like 'contentManager.read' for either agent or tool."
      - "Do not use nested wrappers like params.tool, params.parameters, or assistant as the agent name."
      - "Put the real tool arguments directly inside params."
      - "Use content as null when the response is tool-only."
      - "When the task is already complete, when clarification is needed, or when you are asked for a final confirmation, respond with text instead of calling tools."

    # Instruction line for the available_tools section of workspace prompts
    available_tools_instruction: >-
      Required wrapper context fields: {context_required_csv}.

  # ── Example: Plain OpenAI-style tool calls (no wrapper) ──
  openai_native:
    wrapper_name: null  # No wrapper -- each tool gets its own tool_calls entry

    context_fields:
      required: []
      properties: {}

    call_item: null  # Not used -- tools are called directly

    extra_argument_fields: {}
    argument_required: []

    generation_instructions:
      - "Return a single JSON object only."
      - "Use the standard OpenAI tool_calls format."
      - "Each tool call should be a separate entry in tool_calls."
      - "Set content to null when the response is tool-only."

    available_tools_instruction: >-
      Call tools directly using the standard function calling format.

  # ── Example: MCP-style tool calls ──
  mcp_style:
    wrapper_name: use_mcp_tool

    context_fields:
      required: [server_name]
      properties:
        server_name: { type: string, minLength: 1 }

    call_item:
      properties:
        tool_name: { type: string, minLength: 1 }
        arguments: { type: object, additionalProperties: true }
      required: [tool_name, arguments]

    extra_argument_fields: {}
    argument_required: [server_name, tool_name, arguments]

    generation_instructions:
      - "Return a single JSON object only."
      - "Use the MCP tool calling format with function.name = 'use_mcp_tool'."
      - "Inside function.arguments, provide server_name, tool_name, and arguments."

    available_tools_instruction: >-
      Call tools using the MCP format. Specify server_name and tool_name.
```

### 1.2 How Scenarios Reference Formats

**Existing scenarios** (no YAML changes needed):
```yaml
# No tool_call_format key -> uses "default" -> reproduces current behavior
storageManager_createFolder:
  type: tool
  agent: storageManager
  tool: storageManager_createFolder
  system: true
```

**New scenario using a different format**:
```yaml
# Uses native OpenAI tool calls
direct_api_call:
  type: tool
  tool: search_api
  tool_call_format: openai_native   # <-- references named format
  prompts:
    ...
```

**Inline format override** (for one-off customization):
```yaml
custom_scenario:
  type: tool
  tool_call_format:
    wrapper_name: customWrapper
    context_fields:
      required: [projectId, userId]
      properties:
        projectId: { type: string }
        userId: { type: string }
    call_item:
      properties:
        action: { type: string }
        payload: { type: object, additionalProperties: true }
      required: [action, payload]
    generation_instructions:
      - "Return JSON with a single tool_calls entry using 'customWrapper'."
      - "Inside arguments, provide projectId, userId, and your action calls."
    available_tools_instruction: "Required fields: projectId, userId."
```

### 1.3 Resolution Logic

```
resolve_tool_call_format(scenario):
  1. Read scenario["tool_call_format"]
  2. If missing or null -> return defaults_registry["default"]
  3. If string (e.g., "openai_native") -> look up in tool_call_formats.yaml
  4. If dict (inline) -> deep-merge with defaults_registry["default"]
  5. Return resolved format config
```

### 1.4 Code Changes (Post-Decomposition Targets)

**`SynthChat/schemas/tool_response_schema.py`** -- refactored functions:

| Current Function | New Behavior |
|---|---|
| `_build_use_tools_response_schema(wrapper_name, allowed_tools, session_id, workspace_id)` | Becomes `build_tool_response_schema(format_config, allowed_tools, context_overrides)`. Reads schema structure from `format_config` dict instead of hardcoding. Builds `context_properties` from `format_config["context_fields"]["properties"]`, `call_properties` from `format_config["call_item"]["properties"]`, `extra_argument_fields` from `format_config["extra_argument_fields"]`. When `wrapper_name` is null (native mode), generates one `tool_calls` entry per tool instead of a single wrapper. |
| `_build_use_tools_generation_prompt(base_prompt, wrapper_name, allowed_tools)` | Becomes `build_tool_generation_prompt(format_config, base_prompt, allowed_tools)`. Reads instruction lines from `format_config["generation_instructions"]`, performs `{placeholder}` substitution for `{wrapper_name}` and `{allowed_tools_csv}`. |
| `_tool_wrapper_name(tool_schema)` | Becomes `resolve_wrapper_name(format_config, tool_schema)`. Prefers `format_config["wrapper_name"]`, falls back to `tool_schema.tool_format.wrapper`, then `"useTools"`. |

**`SynthChat/workspace/sections.py`** -- one change:

| Current Function | New Behavior |
|---|---|
| `_render_available_tools(tool_schema)` | Becomes `render_available_tools(tool_schema, format_config)`. Reads the instruction line from `format_config["available_tools_instruction"]`, substitutes `{context_required_csv}` with the comma-separated list from `format_config["context_fields"]["required"]`. |

**`SynthChat/config/__init__.py`** -- new function:

```python
def load_tool_call_formats(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load named tool-call format definitions.

    Returns dict mapping format names to their config dicts.
    Falls back to built-in defaults if file not found.
    """
```

### 1.5 Backward Compatibility

- The `"default"` format in `tool_call_formats.yaml` reproduces the exact current behavior
- Scenarios without `tool_call_format` key automatically use `"default"`
- The existing `tool_format.wrapper` field in scenario YAML continues to work (it overrides `wrapper_name` in the resolved format)
- Existing `tool_format.context.required` continues to work (merged into resolved format's `context_fields.required`)

---

## Category 2: Config-Driven Workspace Prompt Structure

### 2.1 New Config File: `SynthChat/config/workspace_formats.yaml`

```yaml
# SynthChat/config/workspace_formats.yaml
# Named workspace prompt formats.
# Scenarios reference these by name via `workspace_format: <name>`.
# The "default" format reproduces today's mocked_workspace_vault behavior.

formats:
  # ── Current Synaptic workspace format (backward-compatible default) ──
  default:
    # Ordered list of sections to render. Each section is a dict with:
    #   tag: XML tag name for wrapping
    #   source: where to get the content (built-in renderer name or "system_context.<key>")
    #   optional: if true, section is omitted when content is empty
    #   template: inline text template (overrides source)
    sections:
      - tag: session_context
        source: session_context
        template: |
          IMPORTANT: When using tools, include these values in your tool call parameters:

          - sessionId: "{session_id}"
          - workspaceId: "{workspace_id}" (current workspace)

          Include these in the "context" parameter of your tool calls.

      - tag: vault_structure
        source: vault_structure
        optional: true

      - tag: available_workspaces
        source: available_workspaces
        optional: true

      - tag: available_prompts
        source: available_prompts
        optional: true

      - tag: available_tools
        source: available_tools
        optional: true

      - tag: selected_workspace
        source: selected_workspace
        # selected_workspace has special rendering (name/id attributes on tag)

      - tag: note_contents
        source: note_contents
        optional: true

      - source: extra_sections

      - source: assistant_instructions
        raw: true  # No XML tag wrapping

    # JSON structure for the selected_workspace payload
    selected_workspace_fields:
      - context       # Object with id, name, description, rootFolder
      - workspaceStructure
      - recentFiles
      - keyFiles
      - workflows
      - preferences
      - sessions

    # Defaults for missing system_context values
    defaults:
      session_id: "session_eval_001"
      workspace_name: "Current Workspace"

  # ── Example: Coding IDE format ──
  coding_ide:
    sections:
      - tag: session
        template: |
          Session ID: {session_id}
          Project: {project_name}

      - tag: project_structure
        source: vault_structure
        optional: true

      - tag: open_files
        source: note_contents
        optional: true

      - tag: available_commands
        source: available_tools
        optional: true

      - source: assistant_instructions
        raw: true

    selected_workspace_fields:
      - context
      - projectStructure
      - openFiles
      - buildConfig
      - gitStatus

    defaults:
      session_id: "ide_session_001"
      workspace_name: "Default Project"
```

### 2.2 How Scenarios Reference Formats

**Existing scenarios** (no YAML changes needed):
```yaml
# system_template: mocked_workspace_vault -> uses workspace_format: "default"
envfs_update_config_note:
  type: tool
  system_template: mocked_workspace_vault
  # No workspace_format key -> resolved from system_template mapping
```

**New scenario with a different workspace format**:
```yaml
ide_code_generation:
  type: tool
  system_template: mocked_workspace_vault
  workspace_format: coding_ide   # <-- overrides the default workspace format
  system_context:
    project_name: "my-api-server"
  prompts:
    ...
```

### 2.3 Resolution Logic

```
resolve_workspace_format(scenario):
  1. Read scenario["workspace_format"]
  2. If present (string) -> look up in workspace_formats.yaml
  3. If present (dict) -> inline format definition
  4. If missing and system_template == "mocked_workspace_vault" -> use "default"
  5. If missing and no system_template -> no workspace rendering
  6. Return resolved format config
```

### 2.4 Code Changes (Post-Decomposition Targets)

**`SynthChat/workspace/renderer.py`** -- refactored:

The current `render_mocked_workspace_system_prompt()` becomes a **generic section renderer** that iterates over the `sections` list from the format config:

```python
def render_workspace_prompt(
    system_context: Dict[str, Any],
    environment_config: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]],
    format_config: Dict[str, Any],           # <-- NEW: from workspace_formats.yaml
    tool_call_format: Optional[Dict[str, Any]] = None,  # <-- for available_tools rendering
) -> str:
    """Render workspace system prompt from format config.

    Iterates over format_config["sections"], resolving each section's content
    from its source, applying templates, and wrapping in XML tags.
    """
```

The function iterates over `format_config["sections"]` and for each section:
1. If `template` is present: render it with `{placeholder}` substitution
2. If `source` is a built-in renderer name: call the appropriate renderer
3. If `source` starts with `system_context.`: pull from system_context dict
4. Wrap in XML tag if `tag` is specified and `raw` is not true
5. Skip if `optional` and content is empty

**Built-in source renderers** (unchanged functions, just called generically):

| Source Name | Existing Function |
|---|---|
| `session_context` | Uses `template` from format config (replaces `_build_session_context_section`) |
| `vault_structure` | `_vault_structure_text_from_fixture()` |
| `available_workspaces` | `_render_available_workspaces()` |
| `available_prompts` | `_render_available_prompts()` |
| `available_tools` | `render_available_tools(tool_schema, tool_call_format)` |
| `selected_workspace` | `_build_selected_workspace_section()` -- uses `selected_workspace_fields` from format config |
| `note_contents` | `_render_note_contents()` |
| `extra_sections` | `_render_extra_sections()` |
| `assistant_instructions` | Direct string from `system_context["assistant_instructions"]` |

**`SynthChat/workspace/sections.py`** -- changes:

| Current Function | Change |
|---|---|
| `_build_session_context_section()` | Removed. Replaced by `template` field in format config. |
| `_build_selected_workspace_section()` | Reads field list from `format_config["selected_workspace_fields"]` instead of hardcoding the 7 fields. |
| `_build_wrapped_section()` | Unchanged. Used generically by the renderer. |

### 2.5 Backward Compatibility

- `system_template: mocked_workspace_vault` continues to work -- it maps to `workspace_format: "default"`
- The `"default"` format reproduces the exact current section ordering and content
- `system_context.extra_sections` continues to work (it's a built-in source renderer)
- `system_context.assistant_instructions` continues to work

---

## Category 3: Config-Driven Metadata Label Mappings

### 3.1 New Config File: `SynthChat/config/label_mappings.yaml`

```yaml
# SynthChat/config/label_mappings.yaml
# Maps environment issues and stage failures to behavioral/failure-type labels.
# The classifier uses these rules instead of hardcoded if/elif chains.

# ── Issue message -> issue label classification ──
# Each rule: if the message (lowercased) contains the `match` string, assign the `label`.
# Rules are evaluated in order; a message can match multiple rules.
issue_classifiers:
  - match: "expected tool(s) not executed"
    label: missing_expected_tool
  - match: "no acceptable tool called"
    label: wrong_tool_called
  - match: "front matter"
    label: frontmatter_missing
  - match: "yaml front matter"
    label: frontmatter_missing
  - match: "expected path to exist"
    label: path_state_mismatch
  - match: "expected path to be absent"
    label: path_state_mismatch
  - match: "does not contain expected text"
    label: content_mismatch
  - match: "contains forbidden text"
    label: content_mismatch
  - match: "failed reading"
    label: read_failure
  - match: "is a directory"
    label: path_type_error
  - match: "file exists"
    label: path_type_error
  - match: "strict schema"
    label: schema_error
  - match: "missing required args"
    label: schema_error
  - match: "searchmanager_searchcontent"
    label: retrieval_missing
  - match: "searchmanager_searchdirectory"
    label: retrieval_missing
  - match: "clarification"
    label: clarification_expected
  - match: "tool '"
    match_also: "failed:"
    label: tool_runtime_error

# ── Issue label -> behavior label rollup ──
# Groups of issue labels that map to a single behavior label.
behavior_rollups:
  "behavior:retrieval_failure":
    - missing_expected_tool
    - retrieval_missing
  "behavior:structure_failure":
    - frontmatter_missing
  "behavior:tool_execution_failure":
    - wrong_tool_called
    - tool_runtime_error
  "behavior:clarification_failure":
    - clarification_expected

# ── Stage failure -> failure type rollup ──
# Groups of stage names that map to a single failure_type label.
failure_type_rollups:
  "failure_type:environment":
    - environment
  "failure_type:behavior":
    - response
    - thinking
  "failure_type:generation":
    - system_prompt
    - user
    - system_generation
    - user_generation
    - assistant_generation
    - environment_generation
    - final
```

### 3.2 Code Changes (Post-Decomposition Targets)

**`SynthChat/labeling/metadata_labels.py`** -- refactored:

| Current Function | New Behavior |
|---|---|
| `_classify_environment_issue(message)` | Becomes `classify_environment_issue(message, classifiers)`. Iterates over `classifiers` list from config. Each rule: check if `match` (and optionally `match_also`) appears in lowercased message. If so, add `label`. |
| `_build_metadata_labels()` (behavior rollup section, lines 1778-1785) | Reads `behavior_rollups` from config. For each rollup label, checks if any of its trigger labels exist in `issue_labels`. If so, adds the rollup label to `flat_labels`. |
| `_build_metadata_labels()` (failure type section, lines 1771-1776) | Reads `failure_type_rollups` from config. For each rollup label, checks if any of its trigger stages exist in `stage_failure_labels`. If so, adds the rollup label to `flat_labels`. |

**New loader in `SynthChat/config/__init__.py`**:

```python
def load_label_mappings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load label mapping config.

    Returns dict with keys: issue_classifiers, behavior_rollups, failure_type_rollups.
    Falls back to built-in defaults if file not found.
    """
```

### 3.3 Backward Compatibility

- The `label_mappings.yaml` file ships with all current hardcoded rules
- If the file is missing, the code falls back to the same built-in defaults
- Existing behavior is identical -- same input produces same labels

### 3.4 Extensibility

Users can add new rules by editing the YAML:

```yaml
# User adds a new classifier for their custom tool
issue_classifiers:
  # ... existing rules ...
  - match: "api rate limit"
    label: rate_limited

# User adds a new behavior rollup
behavior_rollups:
  # ... existing rollups ...
  "behavior:rate_limit_failure":
    - rate_limited
```

---

## Config Loading Architecture

### Where Configs Are Loaded

All three config files are loaded once at `SynthChatGenerator.__init__()` time and stored as instance attributes:

```python
class SynthChatGenerator:
    def __init__(self, ...):
        # ... existing init ...
        self._tool_call_formats = load_tool_call_formats()
        self._workspace_formats = load_workspace_formats()
        self._label_mappings = load_label_mappings()
```

### Per-Scenario Resolution

At generation time, the scenario's format references are resolved:

```python
# In generate_single (or wherever the scenario is processed):
tool_call_format = resolve_tool_call_format(scenario, self._tool_call_formats)
workspace_format = resolve_workspace_format(scenario, self._workspace_formats)
# label_mappings are global, not per-scenario
```

### Config Resolution Functions

New file: **`SynthChat/config/format_resolver.py`** (~80 lines)

```python
def resolve_tool_call_format(
    scenario: Dict[str, Any],
    formats_registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the tool call format for a scenario.

    Priority:
    1. scenario["tool_call_format"] as inline dict
    2. scenario["tool_call_format"] as string -> lookup in registry
    3. scenario["tool_format"]["wrapper"] -> merge into default
    4. "default" from registry
    """

def resolve_workspace_format(
    scenario: Dict[str, Any],
    formats_registry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve the workspace format for a scenario.

    Priority:
    1. scenario["workspace_format"] as inline dict
    2. scenario["workspace_format"] as string -> lookup in registry
    3. system_template == "mocked_workspace_vault" -> "default"
    4. None (no workspace rendering)
    """
```

---

## New File Summary

| File | Lines (est.) | Purpose |
|---|---|---|
| `SynthChat/config/tool_call_formats.yaml` | ~120 | Named tool-call format definitions |
| `SynthChat/config/workspace_formats.yaml` | ~80 | Named workspace prompt format definitions |
| `SynthChat/config/label_mappings.yaml` | ~70 | Issue classifiers and label rollup rules |
| `SynthChat/config/format_resolver.py` | ~80 | Resolution logic for per-scenario format lookup |

### Modified Files (Post-Decomposition)

| File | Change Summary |
|---|---|
| `SynthChat/schemas/tool_response_schema.py` | Functions read structure from `format_config` dict instead of hardcoding. New function signatures accept `format_config` parameter. |
| `SynthChat/workspace/renderer.py` | `render_mocked_workspace_system_prompt` becomes `render_workspace_prompt` with generic section iteration driven by format config. |
| `SynthChat/workspace/sections.py` | `_build_session_context_section` removed (replaced by template). `_build_selected_workspace_section` reads field list from config. `render_available_tools` accepts `format_config`. |
| `SynthChat/labeling/metadata_labels.py` | `_classify_environment_issue` iterates over config rules. `_build_metadata_labels` reads rollup mappings from config. |
| `SynthChat/config/__init__.py` | New loader functions for the three config files. |
| `SynthChat/generator.py` | Loads configs in `__init__`, passes resolved format configs to extracted module functions. |

---

## Migration Strategy

### Phase 1: Ship Config Files with Current Behavior (Zero User Impact)

1. Create the three YAML config files with rules that reproduce current hardcoded behavior exactly
2. Add loader functions that read the YAMLs with hardcoded fallback defaults
3. No scenario YAML changes needed

### Phase 2: Wire Config Into Code

1. Modify `tool_response_schema.py` to accept and use `format_config`
2. Modify `workspace/renderer.py` to iterate over format config sections
3. Modify `labeling/metadata_labels.py` to use config-driven classifiers and rollups
4. Modify `generator.py` to load configs and pass them through

### Phase 3: Verify Backward Compatibility

1. Run full test suite -- all existing tests must pass with zero changes
2. Generate a sample from each scenario type and diff against pre-refactoring output
3. Verify that removing the config files causes fallback to built-in defaults (same behavior)

### Recommended Implementation Order

```
1. label_mappings.yaml + metadata_labels.py changes     (simplest, isolated)
2. tool_call_formats.yaml + tool_response_schema.py     (most impactful, but well-scoped)
3. workspace_formats.yaml + workspace renderer changes  (most complex, most files touched)
4. format_resolver.py + generator.py wiring              (ties it all together)
```

Steps 1 and 2 can be done in parallel (they touch different modules). Step 3 depends on step 2 (workspace renderer uses tool_call_format for available_tools rendering). Step 4 is the integration step.

---

## Testing Strategy

### Existing Tests Affected

| Test | What Changes |
|---|---|
| `test_build_use_tools_response_schema_*` | Function signature changes. Tests must pass `format_config` instead of relying on hardcoded defaults. Add helper that constructs default format_config. |
| `test_build_use_tools_generation_prompt_*` | Same -- function signature changes, pass format_config. |
| Tests that exercise `_render_mocked_workspace_system_prompt` | Function renamed/refactored, but called through `SynthChatGenerator` facade, so integration tests should pass unchanged. |

### New Tests Needed

| Test | What It Verifies |
|---|---|
| `test_resolve_tool_call_format_default` | Missing key -> returns "default" format |
| `test_resolve_tool_call_format_named` | String key -> looks up in registry |
| `test_resolve_tool_call_format_inline` | Dict key -> deep-merged with default |
| `test_build_tool_response_schema_custom_format` | Custom format produces correct JSON Schema |
| `test_build_tool_response_schema_no_wrapper` | `wrapper_name: null` produces per-tool entries |
| `test_classify_environment_issue_from_config` | Config-driven classifier produces same labels as hardcoded version |
| `test_classify_environment_issue_custom_rule` | User-added rule is applied |
| `test_render_workspace_prompt_custom_sections` | Custom section ordering works |
| `test_render_workspace_prompt_custom_template` | Template substitution in section content works |
| `test_backward_compat_no_config_files` | Missing YAML files -> fallback to defaults -> same behavior |

### Regression Test

A dedicated regression test that:
1. Runs `SynthChatGenerator.generate_single()` with a known scenario
2. Captures the output (system prompt, tool schema, metadata labels)
3. Compares against a golden snapshot saved before the refactoring
4. Fails if any output differs

---

## Risk Assessment

### Low Risk

| Risk | Mitigation |
|---|---|
| Config file not found | Fallback to hardcoded defaults in loader functions |
| Malformed YAML | Validate config structure at load time; raise clear error |
| New dependency (Jinja2) | NOT introducing Jinja2. Using existing `{placeholder}` substitution. |

### Medium Risk

| Risk | Mitigation |
|---|---|
| Schema builder produces different JSON Schema than current code | Golden snapshot regression test; careful manual comparison during implementation |
| Workspace renderer produces different prompt text | Same -- golden snapshot test |
| `tool_format.wrapper` and `tool_call_format` config interact unexpectedly | Clear precedence rules documented in resolution logic; test edge cases |

### Low-Medium Risk

| Risk | Mitigation |
|---|---|
| `label_mappings.yaml` `match_also` semantics need careful implementation | The only `match_also` rule is `tool ' ... failed:`. Simple AND condition. |
| Performance of iterating over config rules vs. hardcoded if-chains | Negligible -- these functions run once per example, not in hot loops |

---

## Scope and Effort

**Estimated scope**: ~400 lines of new code + ~200 lines of modified code + ~270 lines of new YAML config.

**Implementable in a single PR** after the decomposition merges. The changes are well-scoped to the post-decomposition modules:
- `schemas/tool_response_schema.py` (Category 1)
- `workspace/renderer.py` + `workspace/sections.py` (Category 2)
- `labeling/metadata_labels.py` (Category 3)
- `config/` (new files)
- `generator.py` (wiring)

No changes to `run.py`, CLI, or external consumer imports.

---

## Decision Log

| Decision | Rationale | Alternatives Considered |
|---|---|---|
| YAML config files (not Python dicts) | Users edit YAML without touching code; consistent with existing SynthChat config pattern | Python dicts in code (violates "no code changes" principle); JSON (less readable) |
| Named format registry (not per-scenario inline only) | Reuse across scenarios; single source of truth for a format; inline override still available | Per-scenario inline only (too verbose, DRY violation) |
| No Jinja2 | Project doesn't use it; simple `{placeholder}` substitution is sufficient for current needs; avoids new dependency | Jinja2 (more powerful but overkill for placeholder substitution in a few instruction lines) |
| `wrapper_name: null` for native mode | Clean signal that no wrapper is used; avoids magic string like `"none"` | Boolean `use_wrapper: false` (less expressive) |
| `selected_workspace_fields` as a list | Controls which fields appear in the JSON payload without duplicating the full schema | Full JSON Schema for the payload (over-engineered for a list of field names) |
| Config loaded once in `__init__` | Consistent with existing SynthChat config loading pattern; no per-example overhead | Lazy loading (unnecessary complexity); per-example loading (wasteful) |
