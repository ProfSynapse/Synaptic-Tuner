"""Scope extractor - extracts scope-specific content using config.

Single Responsibility: Extract scope content using config-defined patterns.
NO HARDCODING - all patterns come from ScopeConfig.
"""

import json
import re
from typing import Optional, Any, List, Dict

from .conversation_parser import ParsedConversation, ConversationParser
from ...config import ScopeConfig
from ...utils.logger import ImproveLogger


class ScopeExtractor:
    """
    Extract scope-specific content using configuration.

    Responsibility: ONLY extraction using config patterns (SRP).
    NO hardcoding - all patterns from ScopeConfig.
    """

    def __init__(self, scope_config: ScopeConfig, logger: Optional[ImproveLogger] = None):
        """
        Initialize scope extractor.

        Args:
            scope_config: Scope configuration with patterns/markers
            logger: Logger instance
        """
        self.config = scope_config
        self.logger = logger or ImproveLogger()
        self.parser = ConversationParser()

    def extract(self, example: dict, scope_name: str) -> Optional[Any]:
        """
        Extract content for a specific scope.

        Args:
            example: Example dict
            scope_name: Scope to extract (e.g., "thinking", "system_prompt")

        Returns:
            Extracted content (format depends on scope) or None if not found
        """
        scope_def = self.config.get_scope(scope_name)
        if not scope_def:
            self.logger.warning(f"Unknown scope: {scope_name}")
            return None

        # Parse conversation
        conversation = self.parser.parse(example)

        # Get message for this scope's role
        message = conversation.get_by_role(scope_def.conversation_role)
        if not message:
            return None

        # Extract based on method
        if scope_def.extraction.method == "role_based":
            return self._extract_role_based(message, scope_def)

        elif scope_def.extraction.method == "pattern":
            return self._extract_pattern_based(message, scope_def)

        elif scope_def.extraction.method == "exclusion":
            return self._extract_exclusion_based(message, scope_def, example)

        else:
            self.logger.warning(f"Unknown extraction method: {scope_def.extraction.method}")
            return None

    def _extract_role_based(self, message, scope_def) -> str:
        """Extract content directly from message (for role-based scopes)."""
        return message.content or ""

    def _extract_pattern_based(self, message, scope_def) -> Optional[Any]:
        """Extract content using regex pattern."""
        pattern = scope_def.extraction.pattern
        if not pattern:
            return None

        # Handle None content (e.g., non-thinking assistant messages)
        content = message.content
        if content is None:
            return None

        # Compile pattern with flags
        flags = 0
        if scope_def.extraction.flags:
            for flag_name in scope_def.extraction.flags:
                flags |= getattr(re, flag_name, 0)

        regex = re.compile(pattern, flags)
        match = regex.search(content)

        if not match:
            return None

        # Extract matched content
        extracted = match.group(1).strip()

        # Parse if format is specified
        if scope_def.format:
            return self._parse_format(extracted, scope_def.format)

        return extracted

    def _extract_exclusion_based(self, message, scope_def, example: dict) -> str:
        """Extract content by excluding other scopes."""
        content = message.content or ""

        # Remove each excluded scope
        if scope_def.extraction.exclude:
            for excluded_scope in scope_def.extraction.exclude:
                excluded_content = self.extract(example, excluded_scope)
                if excluded_content:
                    # Remove the excluded content
                    excluded_scope_def = self.config.get_scope(excluded_scope)
                    if excluded_scope_def and excluded_scope_def.markers.start:
                        # Remove using markers
                        pattern = f"{re.escape(excluded_scope_def.markers.start)}.*?{re.escape(excluded_scope_def.markers.end)}"
                        content = re.sub(pattern, "", content, flags=re.DOTALL)

        return content.strip()

    def _parse_format(self, content: str, format_config) -> Optional[Any]:
        """Parse content according to format configuration."""
        # Try primary format
        if format_config.primary == "json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        # Try fallback format
        if format_config.fallback == "yaml":
            return self._parse_yaml_style(content)

        return content

    def _parse_yaml_style(self, content: str) -> Optional[Dict]:
        """
        Parse YAML-style content into dict.

        This is a generic YAML-style parser that doesn't assume
        specific field names (unlike the old hardcoded version).
        """
        result = {}
        lines = content.split("\n")
        current_key = None
        nested_key = None
        nested_obj = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Array item
            if stripped.startswith("- "):
                if current_key:
                    if not isinstance(result.get(current_key), list):
                        result[current_key] = []
                    result[current_key].append(stripped[2:].strip())
                continue

            # Nested key (indented)
            if line.startswith("  ") and ":" in stripped and nested_key:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    nested_obj[k] = True
                elif v.lower() == "false":
                    nested_obj[k] = False
                else:
                    nested_obj[k] = v
                continue

            # Top-level key
            if ":" in stripped:
                # Save previous nested object
                if nested_key and nested_obj:
                    result[nested_key] = nested_obj
                    nested_obj = {}
                    nested_key = None

                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                current_key = k

                if v == "" or v.lower() == "none":
                    nested_key = k
                    result[k] = {}
                elif v.lower() in ["true", "false"]:
                    result[k] = v.lower() == "true"
                else:
                    try:
                        result[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        result[k] = v

        # Handle final nested object
        if nested_key and nested_obj:
            result[nested_key] = nested_obj

        return result if result else None

    def extract_tool_calls_from_text(self, content: str) -> Optional[List[Dict]]:
        """
        Extract tool calls from text content using config-defined markers.

        Supports three formats:
        1. OpenAI format: {"tool_calls": [{"id": "...", "type": "function", "function": {...}}]}
        2. JSON block format: ```tool {"name": "X", "arguments": {...}} ```
        3. Legacy format: tool_name: X, arguments: {...}

        Args:
            content: Text content with potential tool calls

        Returns:
            List of tool call dicts or None if no tool calls found
        """
        if content is None:
            return None

        # First, try OpenAI tool_calls format (from rubric improvers)
        # This is the preferred format for improved output
        try:
            # Strip any surrounding whitespace or code fences
            stripped = content.strip()
            if stripped.startswith("```"):
                # Remove code fence (```json or ```)
                stripped = re.sub(r'^```\w*\n?', '', stripped)
                stripped = re.sub(r'\n?```$', '', stripped)
                stripped = stripped.strip()

            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls_data = parsed["tool_calls"]
                if isinstance(tool_calls_data, list) and len(tool_calls_data) > 0:
                    # Validate the structure matches OpenAI format
                    valid_tool_calls = []
                    for tc in tool_calls_data:
                        if isinstance(tc, dict) and "function" in tc:
                            # Already in correct OpenAI format
                            valid_tool_calls.append(tc)
                    if valid_tool_calls:
                        return valid_tool_calls
        except (json.JSONDecodeError, TypeError):
            pass

        tool_calls = []

        # Second, try to extract from ```tool JSON blocks
        # This handles improved content from older rubric improvers
        tool_block_pattern = r'```tool\s*\n(.*?)```'
        tool_blocks = re.findall(tool_block_pattern, content, re.DOTALL)

        for block in tool_blocks:
            try:
                tool_json = json.loads(block.strip())
                tool_name = tool_json.get("name")
                arguments = tool_json.get("arguments", {})

                if tool_name:
                    tool_calls.append({
                        "id": f"call_{hash(tool_name + str(len(tool_calls))) % 10000:04x}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments
                        }
                    })
            except json.JSONDecodeError:
                pass

        # If we found tool calls from JSON blocks, return them
        if tool_calls:
            return tool_calls

        # Fallback to legacy format parsing
        scope_def = self.config.get_scope("tool_calls")
        if not scope_def:
            return None

        markers = scope_def.markers

        # Check for legacy tool call markers
        if not any([
            markers.block_start and markers.block_start in content,
            markers.block_start_alt and markers.block_start_alt in content,
            markers.name_prefix and markers.name_prefix in content,
            markers.name_prefix_alt and markers.name_prefix_alt in content
        ]):
            return None

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Look for tool name
            tool_name = None
            if markers.name_prefix and line.startswith(markers.name_prefix):
                tool_name = line.split(markers.name_prefix)[1].strip()
            elif markers.name_prefix_alt and line.startswith(markers.name_prefix_alt):
                tool_name = line.split(markers.name_prefix_alt)[1].strip()

            if tool_name:
                # Look for arguments
                args_str = ""
                if i + 1 < len(lines) and markers.args_prefix in lines[i + 1]:
                    i += 2  # Skip "arguments:" line

                    # Collect JSON
                    json_lines = []
                    brace_count = 0
                    while i < len(lines):
                        json_line = lines[i]
                        json_lines.append(json_line)
                        brace_count += json_line.count("{") - json_line.count("}")
                        if brace_count <= 0 and "{" in "".join(json_lines):
                            break
                        i += 1
                    args_str = "\n".join(json_lines)

                try:
                    args = json.loads(args_str) if args_str else {}
                    tool_calls.append({
                        "id": f"call_{hash(tool_name) % 10000:04x}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args)
                        }
                    })
                except json.JSONDecodeError:
                    pass

            i += 1

        return tool_calls if tool_calls else None
