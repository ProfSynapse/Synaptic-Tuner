"""Interactive console UI for labeling."""

import json
import re
import textwrap
from typing import Dict, List, Optional, Any

from ..core.models import Example, LabelingSession


class InteractiveDisplay:
    """
    Console UI for interactive labeling.

    Features:
    - Formatted example display
    - Category selection menu
    - Progress indicators
    - Input validation
    """

    def __init__(self, display_config: Dict[str, Any], ui_config: Dict[str, Any]):
        """
        Initialize interactive display.

        Args:
            display_config: Display configuration from labeling_categories.yaml
            ui_config: UI formatting configuration
        """
        self.display_config = display_config
        self.ui_config = ui_config
        self.max_line_length = ui_config.get("max_line_length", 100)
        self.separator_char = ui_config.get("separator_char", "─")
        self.header_char = ui_config.get("header_char", "═")
        self.indent_size = ui_config.get("indent_size", 2)

    def display_example(self, example: Example, line_number: int, file_path: str = ""):
        """
        Display example with formatting based on display configuration.

        Args:
            example: Example to display
            line_number: Line number in the dataset
            file_path: Path to the dataset file (for display)
        """
        # Clear screen (optional - uncomment if desired)
        # print("\033[2J\033[H")

        # Header
        self._print_header(line_number, file_path)

        # Display conversation
        if self.display_config.get("show_conversation", True):
            self._display_conversation(example.conversations)

        # Display existing labels if present
        if self.display_config.get("show_existing_labels", True) and example.quality_labels:
            self._display_existing_labels(example.quality_labels)

        # Footer separator
        self._print_separator()

    def _print_header(self, line_number: int, file_path: str):
        """Print header with line number and file path."""
        header_line = self.header_char * self.max_line_length
        print(f"\n{header_line}")

        title = f"Example #{line_number}"
        if file_path:
            # Extract just the filename
            import os
            filename = os.path.basename(file_path)
            title += f" ({filename})"

        print(f"{title}")
        print(f"{header_line}\n")

    def _print_separator(self):
        """Print separator line."""
        print(f"{self.separator_char * self.max_line_length}")

    def _display_conversation(self, conversations: List[Dict[str, str]]):
        """Display conversation turns."""
        for turn in conversations:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")

            # Role header
            print(f"{role}:")

            # Check if content contains thinking block
            thinking_block = self._extract_thinking_block(content)

            if thinking_block and self.display_config.get("show_thinking_block", True):
                # Display thinking block separately
                self._display_thinking_block(thinking_block)

                # Display rest of content (response text and tool calls)
                remaining_content = self._remove_thinking_block(content)
                if remaining_content.strip():
                    if self.display_config.get("show_response_text", True):
                        self._display_response_text(remaining_content)
                    if self.display_config.get("show_tool_calls", True):
                        self._display_tool_calls(remaining_content)
            else:
                # No thinking block, just display content
                if self.display_config.get("show_response_text", True):
                    wrapped = textwrap.fill(content, width=self.max_line_length - self.indent_size)
                    indented = textwrap.indent(wrapped, " " * self.indent_size)
                    print(indented)

            print()  # Blank line between turns

    def _extract_thinking_block(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract thinking block from content if present."""
        # Look for <thinking>...</thinking> tags
        match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
        if match:
            thinking_text = match.group(1).strip()
            try:
                # Try to parse as JSON
                return json.loads(thinking_text)
            except json.JSONDecodeError:
                # If not JSON, return as text
                return {"raw": thinking_text}
        return None

    def _remove_thinking_block(self, content: str) -> str:
        """Remove thinking block from content."""
        return re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL).strip()

    def _display_thinking_block(self, thinking: Dict[str, Any]):
        """Display thinking block in formatted way."""
        print(f"{'  '}THINKING:")

        if "raw" in thinking:
            # Raw text thinking block
            wrapped = textwrap.fill(thinking["raw"], width=self.max_line_length - 4)
            indented = textwrap.indent(wrapped, " " * 4)
            print(indented)
        else:
            # Structured thinking block
            indent = " " * 4
            for key, value in thinking.items():
                if isinstance(value, list):
                    print(f"{indent}{key}:")
                    for item in value:
                        item_wrapped = textwrap.fill(f"- {item}", width=self.max_line_length - 6)
                        print(textwrap.indent(item_wrapped, " " * 6))
                elif isinstance(value, dict):
                    print(f"{indent}{key}:")
                    for k, v in value.items():
                        print(f"{indent}  {k}: {v}")
                else:
                    # Wrap long values
                    value_str = str(value)
                    if len(value_str) > 80:
                        wrapped = textwrap.fill(value_str, width=self.max_line_length - 4 - len(key) - 2)
                        first_line, *rest = wrapped.split('\n')
                        print(f"{indent}{key}: {first_line}")
                        for line in rest:
                            print(f"{indent}{'  ' * len(key)}{line}")
                    else:
                        print(f"{indent}{key}: {value_str}")

    def _display_response_text(self, content: str):
        """Display response text."""
        # Remove tool call sections
        text = re.sub(r'tool_call:.*?(?=\n\n|$)', '', content, flags=re.DOTALL)
        text = re.sub(r'Result:.*?(?=\n\n|$)', '', text, flags=re.DOTALL)
        text = text.strip()

        if text:
            print(f"{'  '}ASSISTANT RESPONSE:")
            wrapped = textwrap.fill(text, width=self.max_line_length - 4)
            indented = textwrap.indent(wrapped, " " * 4)
            print(indented)
            print()

    def _display_tool_calls(self, content: str):
        """Display tool calls if present."""
        # Look for tool_call: pattern
        tool_calls = re.findall(r'tool_call: (\w+)\narguments: ({.*?})', content, re.DOTALL)

        if tool_calls:
            print(f"{'  '}TOOL CALL:")
            for tool_name, args in tool_calls:
                print(f"{'    '}{tool_name}(")
                try:
                    args_dict = json.loads(args)
                    args_str = json.dumps(args_dict, indent=2)
                    indented = textwrap.indent(args_str, " " * 6)
                    print(indented)
                except json.JSONDecodeError:
                    print(f"{'      '}{args}")
                print(f"{'    '})")
            print()

    def _display_existing_labels(self, labels: List[str]):
        """Display existing quality labels."""
        print(f"\n{'  '}EXISTING LABELS: {', '.join(labels)}")
        print()

    def display_rubric(self, rubric_instructions: str, categories: List[Dict[str, Any]]):
        """
        Display the full labeling rubric with all category criteria.

        Args:
            rubric_instructions: General rubric instructions
            categories: List of category definitions
        """
        print(f"\n{self.header_char * self.max_line_length}")
        print("LABELING RUBRIC")
        print(f"{self.header_char * self.max_line_length}\n")

        # Show general instructions
        if rubric_instructions:
            wrapped = textwrap.fill(rubric_instructions.strip(), width=self.max_line_length)
            print(wrapped)
            print()

        # Show each category's criteria
        print("CATEGORY CRITERIA:")
        print()
        for category in categories:
            tag = category.get("tag", "").upper()
            description = category.get("description", "").strip()
            aliases = category.get("aliases", [])

            # Tag header with aliases
            alias_str = f" (shortcuts: {', '.join(aliases)})" if aliases else ""
            print(f"[{tag}]{alias_str}")

            # Description
            desc_wrapped = textwrap.fill(description, width=self.max_line_length - 2)
            desc_indented = textwrap.indent(desc_wrapped, "  ")
            print(desc_indented)
            print()

        print(f"{self.header_char * self.max_line_length}\n")

    def display_categories(self, categories: List[Dict[str, Any]]):
        """
        Show available categories with descriptions.

        Args:
            categories: List of category definitions from config
        """
        print("\nAvailable Labels:")
        for i, category in enumerate(categories, 1):
            tag = category.get("tag", "")
            aliases = category.get("aliases", [])

            # Format aliases
            alias_str = ""
            if aliases:
                alias_str = f" [{', '.join(aliases)}]"

            print(f"  {i}. [{tag}]{alias_str}")

        print("\n  s. Skip this example")
        print("  r. Show full rubric")
        print("  q. Quit labeling session")

    def prompt_selection(
        self,
        categories: List[Dict[str, Any]],
        allow_multiple: bool,
        rubric_instructions: str = ""
    ) -> Optional[List[str]]:
        """
        Prompt for category selection with validation.

        Args:
            categories: List of available categories
            allow_multiple: Whether to allow multiple selections
            rubric_instructions: Rubric instructions to show if user requests

        Returns:
            List of selected tags, None if skipped, or raises KeyboardInterrupt if quit
        """
        # Build lookup maps
        tag_by_number = {str(i+1): cat["tag"] for i, cat in enumerate(categories)}
        tag_by_alias = {}
        tag_by_name = {}

        for cat in categories:
            tag = cat["tag"]
            tag_by_name[tag.lower()] = tag
            for alias in cat.get("aliases", []):
                tag_by_alias[alias.lower()] = tag

        while True:
            if allow_multiple:
                prompt = "\nSelect label(s) (comma-separated for multiple, or number/tag/alias): "
            else:
                prompt = "\nSelect label (number/tag/alias): "

            try:
                user_input = input(prompt).strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\nLabeling session interrupted.")
                raise KeyboardInterrupt()

            # Handle special commands
            if user_input in ['q', 'quit', 'exit']:
                raise KeyboardInterrupt()

            if user_input in ['s', 'skip']:
                return None

            if user_input in ['r', 'rubric']:
                # Show rubric and redisplay example context
                self.display_rubric(rubric_instructions, categories)
                self.display_categories(categories)
                continue

            # Parse input
            selections = [s.strip() for s in user_input.split(',')]
            selected_tags = []

            for selection in selections:
                # Try number
                if selection in tag_by_number:
                    selected_tags.append(tag_by_number[selection])
                # Try alias
                elif selection in tag_by_alias:
                    selected_tags.append(tag_by_alias[selection])
                # Try tag name
                elif selection in tag_by_name:
                    selected_tags.append(tag_by_name[selection])
                else:
                    print(f"Invalid selection: '{selection}'. Please try again.")
                    selected_tags = []
                    break

            if selected_tags:
                # Remove duplicates while preserving order
                selected_tags = list(dict.fromkeys(selected_tags))

                # Validate multiple selection if not allowed
                if not allow_multiple and len(selected_tags) > 1:
                    print("Only one label allowed. Please select a single label.")
                    continue

                return selected_tags

    def show_progress(self, session: LabelingSession):
        """
        Display progress bar and statistics.

        Args:
            session: Current labeling session
        """
        total_processed = session.labeled_count + session.skipped_count
        progress_pct = session.completion_rate

        # Progress bar
        bar_width = 50
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"\n{'─' * self.max_line_length}")
        print(f"Progress: [{bar}] {progress_pct:.1f}%")
        print(f"Processed: {total_processed}/{session.total_examples} | "
              f"Labeled: {session.labeled_count} | "
              f"Skipped: {session.skipped_count}")

        if session.categories_used:
            # Show top 5 most used categories
            top_categories = sorted(
                session.categories_used.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            categories_str = ", ".join([f"{tag}({count})" for tag, count in top_categories])
            print(f"Categories: {categories_str}")

        print(f"{'─' * self.max_line_length}\n")

    def confirm_action(self, message: str) -> bool:
        """
        Prompt for yes/no confirmation.

        Args:
            message: Confirmation message

        Returns:
            True if confirmed, False otherwise
        """
        while True:
            try:
                response = input(f"{message} (y/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return False

            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def print_summary(self, session: LabelingSession):
        """
        Print final summary of labeling session.

        Args:
            session: Completed labeling session
        """
        print(f"\n{self.header_char * self.max_line_length}")
        print("Labeling Session Complete")
        print(f"{self.header_char * self.max_line_length}\n")

        print(session)

        if session.categories_used:
            print("\nCategory Usage:")
            for tag, count in sorted(session.categories_used.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tag}: {count}")

        print(f"\n{self.header_char * self.max_line_length}\n")
