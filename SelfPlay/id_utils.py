"""
ID Generation Utilities

Auto-generates session and workspace IDs following the required format:
- sessionId: session_{13-digit-timestamp}_{9-char-random}
- workspaceId: ws_{13-digit-timestamp}_{9-char-random}
"""

import time
import random
import string
from typing import Tuple


def generate_session_id() -> str:
    """
    Generate a session ID in the format: session_{timestamp}_{random}

    Format: session_1732300800000_a1b2c3d4e
    - timestamp: 13 digits (milliseconds since epoch)
    - random: 9 characters (lowercase letters + digits)

    Returns:
        str: Generated session ID
    """
    timestamp = int(time.time() * 1000)  # 13-digit millisecond timestamp
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"session_{timestamp}_{random_suffix}"


def generate_workspace_id() -> str:
    """
    Generate a workspace ID in the format: ws_{timestamp}_{random}

    Format: ws_1732300800000_f5g6h7i8j
    - timestamp: 13 digits (milliseconds since epoch)
    - random: 9 characters (lowercase letters + digits)

    Returns:
        str: Generated workspace ID
    """
    timestamp = int(time.time() * 1000)  # 13-digit millisecond timestamp
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"ws_{timestamp}_{random_suffix}"


def generate_ids() -> Tuple[str, str]:
    """
    Generate both session and workspace IDs at once.

    Returns:
        Tuple[str, str]: (session_id, workspace_id)
    """
    return generate_session_id(), generate_workspace_id()


def validate_session_id(session_id: str) -> bool:
    """
    Validate a session ID matches the required format.

    Args:
        session_id: Session ID to validate

    Returns:
        bool: True if valid format, False otherwise
    """
    import re
    pattern = r'^session_\d{13}_[a-z0-9]{9}$'
    return bool(re.match(pattern, session_id))


def validate_workspace_id(workspace_id: str) -> bool:
    """
    Validate a workspace ID matches the required format.

    Args:
        workspace_id: Workspace ID to validate

    Returns:
        bool: True if valid format, False otherwise
    """
    import re
    pattern = r'^ws_\d{13}_[a-z0-9]{9}$'
    return bool(re.match(pattern, workspace_id))


# Example usage
if __name__ == "__main__":
    # Generate IDs
    session_id = generate_session_id()
    workspace_id = generate_workspace_id()

    print(f"Session ID:   {session_id}")
    print(f"Workspace ID: {workspace_id}")
    print()

    # Generate both at once
    sid, wid = generate_ids()
    print(f"Generated pair:")
    print(f"  Session:   {sid}")
    print(f"  Workspace: {wid}")
    print()

    # Validate
    print(f"Session ID valid:   {validate_session_id(session_id)}")
    print(f"Workspace ID valid: {validate_workspace_id(workspace_id)}")
    print()

    # Test invalid IDs
    invalid_session = "session_123_abc"  # Wrong format
    invalid_workspace = "workspace_1732300800000_a1b2c3d4e"  # Should be 'ws_'
    print(f"Invalid session valid:   {validate_session_id(invalid_session)}")
    print(f"Invalid workspace valid: {validate_workspace_id(invalid_workspace)}")
