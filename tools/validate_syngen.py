#!/usr/bin/env python3
"""
Validator for synthetic Claudesidian tool-calling datasets in ChatML format.

Validates JSONL files with the following structure:
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "label": true  // optional (true = desirable, false = undesirable)
}

Usage:
    python tools/validate_syngen.py Synthetic\\ Conversations/syngen_toolset_v1.0.0.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

SESSION_ID_RE = re.compile(r"^session_\d{13}_[a-z0-9]{9}$")
WORKSPACE_ID_RE = re.compile(r"^ws_\d{13}_[a-z0-9]{9}$")
# Labels are now boolean: true = desirable, false = undesirable
ALLOWED_ROLES = {"system", "user", "assistant"}
MIN_TOOL_CALLS = 2

# System prompt context extraction patterns
SESSION_CONTEXT_RE = re.compile(r'<session_context>(.*?)</session_context>', re.DOTALL)
AVAILABLE_WORKSPACES_RE = re.compile(r'<available_workspaces>(.*?)</available_workspaces>', re.DOTALL)
AVAILABLE_AGENTS_RE = re.compile(r'<available_agents>(.*?)</available_agents>', re.DOTALL)
SESSION_ID_EXTRACT_RE = re.compile(r'sessionId:\s*["\']([^"\']+)["\']')
WORKSPACE_ID_EXTRACT_RE = re.compile(r'workspaceId:\s*["\']([^"\']+)["\']')
WORKSPACE_LIST_RE = re.compile(r'\(id:\s*["\']([^"\']+)["\']\)')
AGENT_LIST_RE = re.compile(r'\(id:\s*["\']([^"\']+)["\']\)')

# Load tool schemas
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}
SCHEMAS_FILE = Path(__file__).parent / "tool_schemas.json"

def load_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Load tool schemas from JSON file."""
    if SCHEMAS_FILE.exists():
        try:
            with open(SCHEMAS_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load tool schemas: {e}", file=sys.stderr)
    return {}

TOOL_SCHEMAS = load_tool_schemas()


@dataclass
class ValidationIssue:
    level: str  # "ERROR" or "WARN"
    message: str


@dataclass
class ExampleReport:
    index: int
    issues: List[ValidationIssue] = field(default_factory=list)
    label: Optional[bool] = None

    def add(self, level: str, message: str) -> None:
        self.issues.append(ValidationIssue(level, message))

    @property
    def is_valid(self) -> bool:
        # For undesirable examples (label=False), we expect them to have errors (they demonstrate bad behavior)
        # So we only fail validation if there are structural issues, not tool parameter issues
        if self.label is False:  # Explicitly check for False
            # Allow tool parameter errors and schema mismatches in undesirable examples
            structural_errors = [
                issue for issue in self.issues
                if issue.level == "ERROR" and not any(x in issue.message for x in [
                    "Missing required parameter",
                    "Unexpected parameter",
                    "Invalid parameter value",
                    "Missing required 'context'",  # Allow missing context in undesirable
                    "Context object must be the first field",  # Allow context ordering issues
                    "does not match system prompt"  # Allow ID mismatches in undesirable
                ])
            ]
            return len(structural_errors) == 0
        # For desirable examples (label=True), all errors are failures
        return all(issue.level != "ERROR" for issue in self.issues)


@dataclass
class SystemPromptContext:
    """Extracted context from system prompt for validation."""
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    available_workspace_ids: List[str] = field(default_factory=list)
    available_agent_ids: List[str] = field(default_factory=list)
    has_system_prompt: bool = False


def extract_system_prompt_context(conversations: list) -> SystemPromptContext:
    """Extract IDs from system prompt for cross-validation with tool calls."""
    ctx = SystemPromptContext()

    # Find system message
    system_msg = None
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system_msg = msg.get("content", "")
            ctx.has_system_prompt = True
            break

    if not system_msg:
        return ctx

    # Extract session context
    session_match = SESSION_CONTEXT_RE.search(system_msg)
    if session_match:
        session_content = session_match.group(1)

        # Extract sessionId
        sid_match = SESSION_ID_EXTRACT_RE.search(session_content)
        if sid_match:
            ctx.session_id = sid_match.group(1)

        # Extract workspaceId
        wid_match = WORKSPACE_ID_EXTRACT_RE.search(session_content)
        if wid_match:
            ctx.workspace_id = wid_match.group(1)

    # Extract available workspaces
    workspaces_match = AVAILABLE_WORKSPACES_RE.search(system_msg)
    if workspaces_match:
        workspaces_content = workspaces_match.group(1)
        ctx.available_workspace_ids = WORKSPACE_LIST_RE.findall(workspaces_content)

    # Extract available agents
    agents_match = AVAILABLE_AGENTS_RE.search(system_msg)
    if agents_match:
        agents_content = agents_match.group(1)
        ctx.available_agent_ids = AGENT_LIST_RE.findall(agents_content)

    return ctx


def validate_ids_match_system_prompt(
    tool_name: str,
    args: dict,
    system_ctx: SystemPromptContext,
    report: ExampleReport,
    tool_call_num: int
) -> None:
    """Validate that tool call IDs match those provided in the system prompt."""
    if not system_ctx.has_system_prompt:
        return  # No system prompt to validate against

    context = args.get("context", {})
    if not isinstance(context, dict):
        return

    # Validate sessionId matches
    tool_session_id = context.get("sessionId")
    if tool_session_id and system_ctx.session_id:
        if tool_session_id != system_ctx.session_id:
            report.add("ERROR",
                f"Tool call #{tool_call_num} ({tool_name}): sessionId '{tool_session_id}' "
                f"does not match system prompt sessionId '{system_ctx.session_id}'")

    # Validate workspaceId matches
    tool_workspace_id = context.get("workspaceId")
    if tool_workspace_id and system_ctx.workspace_id:
        # workspaceId in tool should match either:
        # 1. The workspaceId in session_context, OR
        # 2. One of the available_workspace_ids (for workspace operations)
        valid_workspace_ids = [system_ctx.workspace_id] + system_ctx.available_workspace_ids

        if tool_workspace_id not in valid_workspace_ids:
            report.add("ERROR",
                f"Tool call #{tool_call_num} ({tool_name}): workspaceId '{tool_workspace_id}' "
                f"does not match system prompt (expected one of: {valid_workspace_ids})")

    # For workspace operations, validate the target workspace ID
    if "id" in args and tool_name.startswith("memoryManager_"):
        target_id = args.get("id")
        if target_id and system_ctx.available_workspace_ids:
            # For loadWorkspace, createWorkspace, etc. the target should be in available list
            # or be a new workspace being created
            if target_id.startswith("ws_") and target_id not in system_ctx.available_workspace_ids:
                # Only warn for loadWorkspace (should exist), not createWorkspace (new)
                if "load" in tool_name.lower():
                    report.add("WARN",
                        f"Tool call #{tool_call_num} ({tool_name}): target workspace '{target_id}' "
                        f"not in available_workspaces list")

    # For agent operations, validate agent IDs
    if tool_name.startswith("agentManager_"):
        # Check 'id' parameter for updateAgent, deleteAgent, getAgent, toggleAgent
        agent_id = args.get("id")
        if agent_id and system_ctx.available_agent_ids:
            if agent_id.startswith("agent_") and agent_id not in system_ctx.available_agent_ids:
                # Only warn - agent might be newly created
                if any(op in tool_name.lower() for op in ["update", "delete", "get", "toggle"]):
                    report.add("WARN",
                        f"Tool call #{tool_call_num} ({tool_name}): agent '{agent_id}' "
                        f"not in available_agents list")


def load_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {idx}: invalid JSON - {exc}") from exc
            yield idx, payload


def validate_conversations_array(conversations: list, report: ExampleReport) -> None:
    if not isinstance(conversations, list):
        report.add("ERROR", "conversations must be an array")
        return
    if len(conversations) == 0:
        report.add("ERROR", "conversations array must not be empty")
        return

    # Check for required roles
    roles = [msg.get("role") for msg in conversations if isinstance(msg, dict)]
    # System message is no longer required (new format)
    if "user" not in roles:
        report.add("ERROR", "conversations must include at least one 'user' role message")

    # Validate each message
    for idx, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            report.add("ERROR", f"Message at index {idx} must be an object")
            continue

        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")  # OpenAI format

        if not role:
            report.add("ERROR", f"Message at index {idx} missing 'role' field")
        elif role not in ALLOWED_ROLES:
            report.add("ERROR", f"Message at index {idx} has invalid role '{role}' (must be system/user/assistant)")

        # For OpenAI format, assistant can have null content if tool_calls present
        if role == "assistant" and tool_calls is not None:
            # OpenAI format: content can be null when tool_calls is present
            if content is not None and not isinstance(content, str):
                report.add("ERROR", f"Message at index {idx} has invalid 'content' field (must be string or null)")
        elif not isinstance(content, str):
            report.add("ERROR", f"Message at index {idx} missing or invalid 'content' field (must be string)")


def extract_tool_calls(content: str) -> List[Tuple[str, dict]]:
    """Extract tool calls from ChatML format assistant content, returning (tool_name, arguments_dict) tuples."""
    entries: List[Tuple[str, dict]] = []
    marker = "tool_call:"
    pos = 0
    while True:
        idx = content.find(marker, pos)
        if idx == -1:
            break
        start = idx + len(marker)
        rest = content[start:]
        parts = rest.split("arguments:", 1)
        if len(parts) != 2:
            break
        tool_name = parts[0].strip().splitlines()[0].strip()
        json_start = content.index("{", content.index("arguments:", idx))
        json_blob, end_index = extract_json_block(content, json_start)
        try:
            args = json.loads(json_blob, strict=False)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse arguments JSON for tool {tool_name}")
        entries.append((tool_name, args))
        pos = end_index
    return entries


def extract_tool_calls_openai(tool_calls_array: list) -> List[Tuple[str, dict]]:
    """Extract tool calls from OpenAI format, returning (tool_name, arguments_dict) tuples."""
    entries: List[Tuple[str, dict]] = []
    for tool_call in tool_calls_array:
        if not isinstance(tool_call, dict):
            continue

        # OpenAI format has nested function object
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue

        tool_name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        # Arguments might be a string that needs parsing
        if isinstance(arguments_str, str):
            try:
                args = json.loads(arguments_str, strict=False)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse arguments JSON for tool {tool_name}")
        elif isinstance(arguments_str, dict):
            args = arguments_str
        else:
            raise ValueError(f"Invalid arguments format for tool {tool_name}")

        entries.append((tool_name, args))
    return entries


def extract_tool_calls_mistral(content: str) -> List[Tuple[str, dict]]:
    """Extract tool calls from Mistral [TOOL_CALLS] format, returning (tool_name, arguments_dict) tuples.

    Mistral format: [TOOL_CALLS] [{"name": "tool_name", "arguments": "{...}", "id": "..."}]
    """
    entries: List[Tuple[str, dict]] = []

    if "[TOOL_CALLS]" not in content:
        return entries

    # Split on marker and get the JSON part
    parts = content.split("[TOOL_CALLS]", 1)
    if len(parts) < 2:
        return entries

    json_part = parts[1].strip()
    if not json_part.startswith("["):
        return entries

    # Find the matching closing bracket for the array
    try:
        json_blob, _ = extract_json_block(json_part, 0)
        tool_calls_array = json.loads(json_blob, strict=False)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse Mistral tool calls JSON: {e}")

    if not isinstance(tool_calls_array, list):
        raise ValueError("Mistral tool calls must be an array")

    for tool_call in tool_calls_array:
        if not isinstance(tool_call, dict):
            continue

        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # Arguments might be a string that needs parsing
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments, strict=False)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse arguments JSON for tool {tool_name}")
        elif isinstance(arguments, dict):
            args = arguments
        else:
            raise ValueError(f"Invalid arguments format for tool {tool_name}")

        entries.append((tool_name, args))

    return entries


def validate_tool_call_structure(content: str, report: ExampleReport) -> None:
    """Validate the structure and formatting of tool calls in assistant content."""
    tool_call_positions = []
    pos = 0

    # Find all tool_call markers
    while True:
        idx = content.find("tool_call:", pos)
        if idx == -1:
            break
        tool_call_positions.append(idx)
        pos = idx + 1

    if not tool_call_positions:
        return  # No tool calls to validate

    for call_idx, call_pos in enumerate(tool_call_positions):
        tool_call_num = call_idx + 1

        # Extract the section for this tool call
        next_pos = tool_call_positions[call_idx + 1] if call_idx + 1 < len(tool_call_positions) else len(content)
        section = content[call_pos:next_pos]

        # 1. Validate tool_call line format
        first_line_end = section.find("\n")
        if first_line_end == -1:
            report.add("ERROR", f"Tool call #{tool_call_num}: Missing newline after tool_call:")
            continue

        first_line = section[:first_line_end]
        if not first_line.startswith("tool_call: "):
            report.add("ERROR", f"Tool call #{tool_call_num}: Must have space after 'tool_call:'")

        tool_name = first_line.replace("tool_call:", "").strip()
        if not tool_name:
            report.add("ERROR", f"Tool call #{tool_call_num}: Missing tool name after 'tool_call:'")
        elif "_" not in tool_name:
            report.add("WARN", f"Tool call #{tool_call_num}: Tool name '{tool_name}' doesn't follow manager_mode convention")

        # 2. Validate arguments presence and format
        if "arguments:" not in section:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing 'arguments:' line")
            continue

        args_idx = section.find("arguments:")
        args_line_start = section.rfind("\n", 0, args_idx)
        args_line = section[args_line_start:args_idx + len("arguments:")].strip()

        if not args_line.startswith("arguments:"):
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): 'arguments:' must be at start of line")

        # 3. Validate arguments JSON structure
        try:
            json_start = section.index("{", args_idx)
            json_blob, _ = extract_json_block(section, json_start)
            args = json.loads(json_blob, strict=False)

            # Validate it's a dict
            if not isinstance(args, dict):
                report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Arguments must be a JSON object, not {type(args).__name__}")
        except ValueError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Invalid JSON in arguments - {e}")
        except json.JSONDecodeError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Failed to parse arguments JSON - {e}")

        # 4. Validate Result format if present
        if "Result:" not in section:
            # Results are optional - tool calls can exist without results
            continue

        result_idx = section.find("Result:")

        # Check that Result comes after arguments
        if result_idx < args_idx:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): 'Result:' appears before 'arguments:'")

        # 5. Validate Result JSON structure
        try:
            result_json_start = section.index("{", result_idx)
            result_json_blob, _ = extract_json_block(section, result_json_start)
            result = json.loads(result_json_blob, strict=False)

            # Validate it's a dict
            if not isinstance(result, dict):
                report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Result must be a JSON object, not {type(result).__name__}")
        except ValueError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Invalid JSON in Result - {e}")
        except json.JSONDecodeError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Failed to parse Result JSON - {e}")


def validate_tool_against_schema(tool_name: str, args: dict, report: ExampleReport, tool_call_num: int) -> None:
    """Validate tool call arguments against the tool's schema."""
    if not TOOL_SCHEMAS:
        report.add("WARN", "Tool schemas not loaded - skipping schema validation")
        return

    if tool_name not in TOOL_SCHEMAS:
        report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): No schema found for this tool")
        return

    schema = TOOL_SCHEMAS[tool_name]
    required_params = schema.get('required_params', [])
    all_params = {p['name']: p for p in schema.get('parameters', [])}

    # Special handling for get_tools meta-tool
    if tool_name == 'get_tools':
        # Validate 'managers' parameter is present and is an array
        if 'managers' not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} (get_tools): Missing required parameter 'managers'")
        elif not isinstance(args.get('managers'), list):
            report.add("ERROR", f"Tool call #{tool_call_num} (get_tools): Parameter 'managers' must be an array")
        elif len(args.get('managers', [])) == 0:
            report.add("WARN", f"Tool call #{tool_call_num} (get_tools): Parameter 'managers' is an empty array")

        # Validate context is present (required for all tools)
        if 'context' not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} (get_tools): Missing required parameter 'context'")

        # Skip further validation for get_tools (it's a meta-tool with special handling)
        return

    # Check required parameters are present
    for req_param in required_params:
        if req_param not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing required parameter '{req_param}'")

    # Check for unexpected parameters (skip 'context' as it's standard across all tools via CommonParams)
    for arg_name in args.keys():
        if arg_name not in all_params and arg_name not in ['context', 'workspaceContext']:
            report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): Unexpected parameter '{arg_name}' not in schema")

    # Validate context structure if present
    if 'context' in args and 'context_schema' in schema:
        context = args['context']
        if not isinstance(context, dict):
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): 'context' must be an object")
        else:
            context_schema = schema['context_schema']
            if context_schema and 'fields' in context_schema:
                required_context_fields = [f['name'] for f in context_schema['fields'] if not f.get('optional', False)]
                for field in required_context_fields:
                    if field not in context:
                        report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing required context field '{field}'")


def extract_json_block(text: str, start_index: int) -> Tuple[str, int]:
    stack = []
    in_string = False
    escape = False
    i = start_index
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        elif not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack.pop():
                    raise ValueError("Unbalanced JSON block")
                if not stack:
                    return text[start_index : i + 1], i + 1
        i += 1
    raise ValueError("Unterminated JSON block")


def validate_context(args: dict, report: ExampleReport) -> None:
    if not isinstance(args, dict) or not args:
        report.add("ERROR", "Arguments must be a JSON object")
        return

    # ALL tools require context from CommonParams - it must be present and first
    if "context" not in args:
        report.add("ERROR", "Missing required 'context' field in arguments (all tools require context from CommonParams)")
        return

    keys = list(args.keys())
    if keys[0] != "context":
        report.add("ERROR", "Context object must be the first field in arguments")
        return
    ctx = args.get("context")
    required_fields = [
        "sessionId",
        "workspaceId",
        "sessionDescription",
        "sessionMemory",
        "toolContext",
        "primaryGoal",
        "subgoal",
    ]
    if not isinstance(ctx, dict):
        report.add("ERROR", "context must be an object")
        return
    for field in required_fields:
        if field not in ctx:
            report.add("ERROR", f"context missing field '{field}'")
    sess_id = ctx.get("sessionId")
    if isinstance(sess_id, str) and not SESSION_ID_RE.match(sess_id):
        report.add("ERROR", f"sessionId '{sess_id}' does not match generator format")
    ws_id = ctx.get("workspaceId")
    # Accept "default" as a valid workspaceId (used for default workspace)
    if isinstance(ws_id, str) and ws_id != "default" and not WORKSPACE_ID_RE.match(ws_id):
        report.add("ERROR", f"workspaceId '{ws_id}' does not match generator format")


def validate_assistant_content(content: str, report: ExampleReport, system_ctx: Optional[SystemPromptContext] = None) -> None:
    """Validate assistant message content, including tool calls if present.

    Supports multiple formats:
    - ChatML format: tool_call: toolName\\narguments: {...}
    - Mistral format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
    """
    if not content.strip():
        report.add("ERROR", "Assistant content may not be empty")
        return

    tool_calls = []

    # Check for Mistral format first (more specific marker)
    if "[TOOL_CALLS]" in content:
        try:
            tool_calls = extract_tool_calls_mistral(content)
        except ValueError as e:
            report.add("ERROR", f"Invalid Mistral tool call format: {e}")
            return

        if not tool_calls:
            report.add("ERROR", "Assistant content has '[TOOL_CALLS]' marker but no valid tool calls found")
            return

    # Check for ChatML format
    elif "tool_call:" in content:
        try:
            tool_calls = extract_tool_calls(content)
        except ValueError as e:
            report.add("ERROR", f"Invalid ChatML tool call format: {e}")
            return

        if not tool_calls:
            report.add("ERROR", "Assistant content has 'tool_call:' marker but no valid tool calls found")
            return

    # Validate each tool call (if any were found)
    for idx, (tool_name, args) in enumerate(tool_calls, 1):
        if not tool_name:
            report.add("ERROR", f"Tool call #{idx} missing name")
        validate_context(args, report)
        # Validate against actual tool schema
        validate_tool_against_schema(tool_name, args, report, idx)
        # Validate IDs match system prompt context
        if system_ctx:
            validate_ids_match_system_prompt(tool_name, args, system_ctx, report, idx)


def validate_assistant_message_openai(msg: dict, report: ExampleReport, system_ctx: Optional[SystemPromptContext] = None) -> None:
    """Validate OpenAI format assistant message with tool_calls array."""
    tool_calls_array = msg.get("tool_calls")
    if not isinstance(tool_calls_array, list):
        report.add("ERROR", "tool_calls must be an array")
        return

    if len(tool_calls_array) == 0:
        report.add("ERROR", "tool_calls array must not be empty if present")
        return

    try:
        tool_calls = extract_tool_calls_openai(tool_calls_array)
    except ValueError as e:
        report.add("ERROR", str(e))
        return

    if not tool_calls:
        report.add("ERROR", "No valid tool calls found in tool_calls array")
        return

    # Validate each tool call
    for idx, (tool_name, args) in enumerate(tool_calls, 1):
        if not tool_name:
            report.add("ERROR", f"Tool call #{idx} missing name")
        validate_context(args, report)
        # Validate against actual tool schema
        validate_tool_against_schema(tool_name, args, report, idx)
        # Validate IDs match system prompt context
        if system_ctx:
            validate_ids_match_system_prompt(tool_name, args, system_ctx, report, idx)


def validate_example(idx: int, example: dict) -> ExampleReport:
    label = example.get("label")
    report = ExampleReport(index=idx, label=label)
    conversations = example.get("conversations")

    # Validate conversations field
    if conversations is None:
        report.add("ERROR", "Missing 'conversations' field")
        return report
    if not isinstance(conversations, list):
        report.add("ERROR", "'conversations' must be an array")
        return report

    # Validate conversations array structure
    validate_conversations_array(conversations, report)

    # Extract system prompt context for ID validation
    system_ctx = extract_system_prompt_context(conversations)

    # Validate assistant messages specifically
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            # Check format: OpenAI (tool_calls array) or ChatML (content string)
            if "tool_calls" in msg:
                # OpenAI format
                validate_assistant_message_openai(msg, report, system_ctx)
            else:
                # ChatML format
                content = msg.get("content", "")
                validate_assistant_content(content, report, system_ctx)

    # Label is optional in ChatML format
    # Labels should now be boolean: true = desirable, false = undesirable
    if label is not None:
        if not isinstance(label, bool):
            report.add("ERROR", f"Label must be a boolean (true/false) if present, got: {type(label).__name__}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Claudesidian synthetic data JSONL files")
    parser.add_argument("path", type=Path, help="Path to JSONL file")
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"File not found: {args.path}")

    reports: List[ExampleReport] = []
    try:
        for idx, payload in load_jsonl(args.path):
            reports.append(validate_example(idx, payload))
    except ValueError as exc:
        sys.exit(str(exc))

    # Only count failures from label=true examples (or no label)
    # label=false examples are intentionally incorrect and should be ignored
    invalid = [r for r in reports if not r.is_valid and r.label is not False]

    for report in reports:
        if report.issues and report.label is not False:
            print(f"Example line {report.index}:")
            for issue in report.issues:
                print(f"  [{issue.level}] {issue.message}")
            print()

    # Print schema validation status
    if TOOL_SCHEMAS:
        print(f"✓ Schema validation enabled ({len(TOOL_SCHEMAS)} tool schemas loaded)\n", file=sys.stderr)
    else:
        print(f"⚠ Schema validation disabled (tool_schemas.json not found)\n", file=sys.stderr)

    # Count label=false examples separately for informational purposes
    label_false_count = len([r for r in reports if r.label is False])
    summary = f"Validated {len(reports)} example(s): {len(invalid)} failed (ignoring {label_false_count} label=false examples)."
    if invalid:
        sys.exit(summary)
    print(summary)


if __name__ == "__main__":
    main()
