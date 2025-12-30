"""Shared parsing utilities for response and tool call parsers.

This module provides common utilities used across the parsing module,
reducing code duplication and providing a single source of truth for
shared functionality.
"""
from __future__ import annotations


def fix_json_newlines(json_str: str) -> str:
    """Fix literal newlines inside JSON string values.

    Models sometimes output literal newlines inside JSON strings instead of
    escaped \\n sequences. This makes the JSON invalid.

    This function attempts to fix this by escaping newlines that appear
    inside string values (between quotes).

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Fixed JSON string with escaped newlines

    Example:
        >>> malformed = '{"key": "value with\nliteral newline"}'
        >>> fixed = fix_json_newlines(malformed)
        >>> fixed
        '{"key": "value with\\nliteral newline"}'
    """
    # Strategy: find all string values and escape newlines within them
    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\':
            escape_next = True
            result.append(char)
            i += 1
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if in_string:
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return ''.join(result)
