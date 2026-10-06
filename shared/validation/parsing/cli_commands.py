"""Parse CLI command strings against a CLI command catalog.

A CLI-style tool-call wrapper carries one or more commands in a single string,
for example ``content read "a.md" 1, content write "b.md" "text"``. This module
is the one parser for those strings. The environment executor, the Evaluator
schema validator, path scoring and the dataset tools all use it.

The command catalog (command words, tool name and argument specs) is data:
callers build it from a CLI schema file. Which backslash escapes are decoded is
set by the tool-call format config (``command_escapes``), not by this module.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CliCommandSpec:
    """One catalog command: its words, the tool it runs and its argument specs."""

    command: str
    agent: str
    tool: str
    arguments: Sequence[Mapping[str, Any]] = ()

    @property
    def tool_name(self) -> str:
        return f"{self.agent}_{self.tool}"


@dataclass
class ParsedCliCommand:
    """One parsed command; ``spec`` is ``None`` when no catalog command matched."""

    tokens: List[str]
    spec: Optional[CliCommandSpec] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def load_cli_command_catalog(schema_path: Path) -> Dict[str, CliCommandSpec]:
    """Load a CLI schema JSON file (``{"tools": [{agent, tool, command, arguments}]}``)."""
    payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    catalog: Dict[str, CliCommandSpec] = {}
    for item in payload.get("tools", []):
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        tool = str(item.get("tool", "")).strip()
        command = str(item.get("command", "")).strip()
        if not agent or not tool or not command:
            continue
        catalog[command] = CliCommandSpec(
            command=command,
            agent=agent,
            tool=tool,
            arguments=tuple(item.get("arguments", []) or []),
        )
    return catalog


def parse_cli_commands(
    tool_value: str,
    catalog: Mapping[str, CliCommandSpec],
    escapes: Mapping[str, str],
) -> List[ParsedCliCommand]:
    """Parse every command in ``tool_value`` against ``catalog``.

    Raises ``ValueError`` when the string cannot be tokenized (an unterminated
    quote or a trailing backslash). A command whose words match no catalog
    entry is returned with ``spec=None`` so each caller decides whether that
    rejects the whole string or skips the command.
    """
    sorted_commands = sorted(catalog, key=lambda value: len(value.split()), reverse=True)
    parsed: List[ParsedCliCommand] = []
    for tokens in tokenize_cli_commands(tool_value, escapes):
        spec = None
        remaining: List[str] = []
        for command in sorted_commands:
            command_tokens = command.split()
            if tokens[: len(command_tokens)] == command_tokens:
                spec = catalog[command]
                remaining = tokens[len(command_tokens):]
                break
        if spec is None:
            parsed.append(ParsedCliCommand(tokens=tokens))
            continue
        arguments, errors = parse_cli_arguments(remaining, spec.arguments)
        parsed.append(ParsedCliCommand(tokens=tokens, spec=spec, arguments=arguments, errors=errors))
    return parsed


_DOUBLE_QUOTES = {'"': {'"'}, "“": {'"', "“", "”"}, "”": {'"', "“", "”"}}
_SINGLE_QUOTES = {"'": {"'"}, "‘": {"'", "‘", "’"}, "’": {"'", "‘", "’"}}
# An undeclared token is option-shaped when its part before any ``=`` is ``--``
# or ``--name`` with no whitespace. A value that merely starts with dashes, such
# as ``---`` YAML front matter, is an argument, not an option.
_OPTION_NAME = re.compile(r"--(?:[^-\s]\S*)?")


def _is_cli_separator(char: str) -> bool:
    return char.isspace() or unicodedata.category(char) == "Zs"


def tokenize_cli_commands(tool_value: str, escapes: Mapping[str, str]) -> List[List[str]]:
    """Split a CLI command sequence into commands and their argument tokens.

    Tokenizing follows POSIX shell quoting, as ``shlex.split`` does: whitespace
    separates arguments outside quotes, a backslash outside quotes takes the next
    character literally, single quotes are literal, and inside double quotes a
    backslash escapes ``"`` and ``\\\\``. Quoted text is kept exactly, including
    newlines and other whitespace. ``escapes`` maps the character after a
    backslash inside double quotes to its decoded text; it comes from the
    tool-call format config, so a format without escapes keeps plain POSIX
    semantics.

    Commands are separated by commas outside quotes, braces and brackets. Curly
    quotes outside an argument open a quoted argument that a straight or curly
    quote of the same kind closes. Raises ``ValueError`` like ``shlex.split`` on
    an unterminated quote or a trailing backslash.
    """
    commands: List[List[str]] = []
    tokens: List[str] = []
    buffer: List[str] = []
    in_token = False
    closers: Optional[set] = None
    double_quoted = False
    brace_depth = 0
    bracket_depth = 0

    def end_token() -> None:
        nonlocal buffer, in_token
        if in_token:
            tokens.append("".join(buffer))
        buffer = []
        in_token = False

    i = 0
    length = len(tool_value)
    while i < length:
        char = tool_value[i]
        if closers is not None:
            if char in closers:
                closers = None
            elif double_quoted and char == "\\" and i + 1 < length:
                following = tool_value[i + 1]
                if following == "\\" or following in closers:
                    buffer.append(following)
                    i += 1
                elif following in escapes:
                    buffer.append(escapes[following])
                    i += 1
                else:
                    buffer.append(char)
            else:
                buffer.append(char)
            i += 1
            continue

        if char == "\\":
            if i + 1 >= length:
                raise ValueError("No escaped character")
            buffer.append(tool_value[i + 1])
            in_token = True
            i += 2
            continue
        if char in _DOUBLE_QUOTES or char in _SINGLE_QUOTES:
            double_quoted = char in _DOUBLE_QUOTES
            closers = (_DOUBLE_QUOTES if double_quoted else _SINGLE_QUOTES)[char]
            in_token = True
            i += 1
            continue
        if _is_cli_separator(char):
            end_token()
            i += 1
            continue
        if char == "," and brace_depth == 0 and bracket_depth == 0:
            end_token()
            if tokens:
                commands.append(tokens)
            tokens = []
            i += 1
            continue
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        buffer.append(char)
        in_token = True
        i += 1

    if closers is not None:
        raise ValueError("No closing quotation")
    end_token()
    if tokens:
        commands.append(tokens)
    return commands


def split_cli_option(token: str, flag_specs: Mapping[str, Any]) -> Optional[Tuple[str, Optional[str]]]:
    """Return ``(flag, inline_value)`` when ``token`` is an option, else ``None``."""
    flag_token, separator, inline_value = token.partition("=")
    if flag_token in flag_specs or _OPTION_NAME.fullmatch(flag_token):
        return flag_token, inline_value if separator else None
    return None


def parse_cli_arguments(
    tokens: Sequence[str],
    argument_specs: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Bind argument tokens to their specs; return ``(arguments, errors)``.

    Positional specs take non-option tokens in order. A declared flag takes the
    next token or its inline ``--flag=value``; a boolean flag takes no value.
    Undeclared options are skipped.
    """
    parsed: Dict[str, Any] = {}
    errors: List[str] = []
    positional_specs = [arg for arg in argument_specs if arg.get("positional")]
    flag_specs = {
        str(arg.get("flag")).strip(): arg
        for arg in argument_specs
        if str(arg.get("flag", "")).strip()
    }

    def bind(arg_spec: Mapping[str, Any], raw_value: str) -> None:
        value = parse_cli_value(raw_value, arg_spec.get("type", "string"))
        parsed[arg_spec["name"]] = value
        error = validate_cli_arg_value(value, arg_spec)
        if error:
            errors.append(error)

    positional_index = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        option = split_cli_option(token, flag_specs)
        if option is not None:
            flag_token, inline_value = option
            arg_spec = flag_specs.get(flag_token)
            if not arg_spec:
                i += 1
                continue
            if arg_spec.get("type", "string") == "boolean":
                parsed[arg_spec["name"]] = True
                i += 1
                continue
            if inline_value is not None:
                bind(arg_spec, inline_value)
                i += 1
                continue
            if i + 1 < len(tokens):
                bind(arg_spec, tokens[i + 1])
                i += 2
                continue
            i += 1
            continue

        if positional_index < len(positional_specs):
            bind(positional_specs[positional_index], token)
            positional_index += 1
        i += 1

    return parsed, errors


def parse_cli_value(raw_value: str, value_type: str) -> Any:
    lowered = (value_type or "").strip().lower()
    if lowered.startswith("array") or lowered == "object":
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            if lowered == "array<string>":
                parts = [part.strip() for part in raw_value.split(",") if part.strip()]
                if parts:
                    return parts
            return raw_value
    if lowered == "boolean":
        if raw_value.lower() in {"true", "1", "yes"}:
            return True
        if raw_value.lower() in {"false", "0", "no"}:
            return False
        return raw_value
    if lowered == "number":
        try:
            return int(raw_value) if "." not in raw_value else float(raw_value)
        except ValueError:
            return raw_value
    return raw_value


def validate_cli_arg_value(value: Any, spec: Mapping[str, Any]) -> Optional[str]:
    value_type = str(spec.get("type", "string") or "string").strip().lower()
    name = str(spec.get("name", "value") or "value")

    if value_type.startswith("array"):
        if not isinstance(value, list):
            return f"{name} must be valid JSON array"
        if value_type == "array<object>" and any(not isinstance(item, dict) for item in value):
            return f"{name} must be an array of objects"
        return None

    if value_type == "object" and not isinstance(value, dict):
        return f"{name} must be valid JSON object"

    if value_type == "number" and not isinstance(value, (int, float)):
        return f"{name} must be numeric"

    if value_type == "boolean" and not isinstance(value, bool):
        return f"{name} must be boolean"

    return None
