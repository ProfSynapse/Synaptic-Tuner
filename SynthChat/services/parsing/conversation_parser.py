"""Conversation parser - parses conversation structure only.

Single Responsibility: Parse conversation format into structured data.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedMessage:
    """A single parsed message from a conversation."""
    role: str
    content: str
    tool_calls: Optional[List[dict]] = None


@dataclass
class ParsedConversation:
    """Parsed conversation with structured access to messages."""
    messages: List[ParsedMessage]

    def get_by_role(self, role: str) -> Optional[ParsedMessage]:
        """Get first message by role."""
        for msg in self.messages:
            if msg.role == role:
                return msg
        return None

    def get_all_by_role(self, role: str) -> List[ParsedMessage]:
        """Get all messages by role."""
        return [msg for msg in self.messages if msg.role == role]

    def get_system_message(self) -> Optional[ParsedMessage]:
        """Get system message."""
        return self.get_by_role("system")

    def get_user_message(self) -> Optional[ParsedMessage]:
        """Get user message."""
        return self.get_by_role("user")

    def get_assistant_message(self) -> Optional[ParsedMessage]:
        """Get assistant message."""
        return self.get_by_role("assistant")


class ConversationParser:
    """
    Parse conversation structure.

    Responsibility: ONLY parsing conversation format (SRP).
    Does NOT extract scope-specific content (that's ScopeExtractor's job).
    """

    def parse(self, example: dict) -> ParsedConversation:
        """
        Parse example into structured conversation.

        Args:
            example: Example dict with "conversations" list

        Returns:
            ParsedConversation with structured messages
        """
        messages = []

        for conv in example.get("conversations", []):
            role = conv.get("role", "")
            content = conv.get("content", "")
            tool_calls = conv.get("tool_calls")

            messages.append(ParsedMessage(
                role=role,
                content=content,
                tool_calls=tool_calls
            ))

        return ParsedConversation(messages=messages)

    def to_dict(self, conversation: ParsedConversation) -> dict:
        """
        Convert parsed conversation back to dict format.

        Args:
            conversation: ParsedConversation

        Returns:
            Dict in example format
        """
        conversations = []

        for msg in conversation.messages:
            conv = {
                "role": msg.role,
                "content": msg.content
            }
            if msg.tool_calls:
                conv["tool_calls"] = msg.tool_calls

            conversations.append(conv)

        return {"conversations": conversations}
