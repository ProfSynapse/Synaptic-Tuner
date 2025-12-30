"""
Arrow-key navigation menus using simple-term-menu.

Location: /mnt/f/Code/Toolset-Training/tuner/ui/terminal_menu.py
Purpose: Provide modern arrow-key menu navigation
Used by: shared/ui/console.py as primary menu system
"""

from typing import List, Tuple, Optional

# Check if simple-term-menu is available
TERMINAL_MENU_AVAILABLE = False
try:
    from simple_term_menu import TerminalMenu
    TERMINAL_MENU_AVAILABLE = True
except ImportError:
    pass


def arrow_menu(
    options: List[Tuple[str, str]],
    title: str = "Select an option",
    show_search: bool = True,
    cursor: str = "→",
    cursor_style: Tuple[str, str] = ("fg_cyan", "bold"),
    highlight_style: Tuple[str, str] = ("fg_cyan",),
) -> Optional[str]:
    """
    Display an arrow-key navigable menu.

    Args:
        options: List of (key, description) tuples
        title: Menu title displayed above options
        show_search: Enable '/' to search/filter options
        cursor: Cursor character shown on selected option
        cursor_style: Styling for cursor (simple-term-menu styles)
        highlight_style: Styling for highlighted option

    Returns:
        Selected option key or None if cancelled (Escape/q)

    Example:
        choice = arrow_menu([
            ("train", "★ Train a model (SFT, KTO, GRPO)"),
            ("upload", "• Upload model to HuggingFace"),
        ], "What would you like to do?")

        if choice == "train":
            # User selected train
            pass
    """
    if not TERMINAL_MENU_AVAILABLE:
        return None  # Caller should fall back to numbered menu

    # Build display labels and add exit option
    labels = [desc for _, desc in options]
    labels.append("[dim]Exit[/dim]" if False else "Exit")  # No rich markup in terminal menu

    # Create menu
    menu = TerminalMenu(
        labels,
        title=f"\n  {title}\n",
        cursor_index=0,
        menu_cursor=f" {cursor} ",
        menu_cursor_style=cursor_style,
        menu_highlight_style=highlight_style,
        cycle_cursor=True,
        clear_screen=False,
        show_search_hint=show_search,
        search_key="/",
        quit_keys=("escape", "q"),
        accept_keys=("enter",),
        skip_empty_entries=True,
    )

    # Show menu and get selection
    selected_index = menu.show()

    # Handle selection
    if selected_index is None:
        # User pressed Escape or q
        return None
    elif selected_index == len(options):
        # User selected "Exit"
        return None
    else:
        # Return the key for the selected option
        return options[selected_index][0]


def arrow_select_multiple(
    options: List[Tuple[str, str]],
    title: str = "Select options (Space to toggle, Enter to confirm)",
    preselected: List[str] = None,
) -> Optional[List[str]]:
    """
    Display a multi-select menu with checkboxes.

    Args:
        options: List of (key, description) tuples
        title: Menu title
        preselected: List of keys to pre-select

    Returns:
        List of selected option keys or None if cancelled

    Example:
        selected = arrow_select_multiple([
            ("sft", "Supervised Fine-Tuning"),
            ("kto", "KTO Preference Learning"),
            ("grpo", "GRPO Training"),
        ], preselected=["sft"])
    """
    if not TERMINAL_MENU_AVAILABLE:
        return None

    labels = [desc for _, desc in options]

    # Determine preselected indices
    preselected_indices = []
    if preselected:
        for i, (key, _) in enumerate(options):
            if key in preselected:
                preselected_indices.append(i)

    menu = TerminalMenu(
        labels,
        title=f"\n  {title}\n",
        multi_select=True,
        show_multi_select_hint=True,
        preselected_entries=preselected_indices if preselected_indices else None,
        multi_select_cursor_style=("fg_cyan", "bold"),
        multi_select_select_on_accept=False,
        quit_keys=("escape", "q"),
    )

    selected_indices = menu.show()

    if selected_indices is None:
        return None

    # Convert tuple to list if needed
    if isinstance(selected_indices, int):
        selected_indices = (selected_indices,)

    return [options[i][0] for i in selected_indices]


def arrow_confirm(
    message: str,
    default: bool = False,
) -> bool:
    """
    Arrow-key confirmation prompt (Yes/No).

    Args:
        message: Confirmation message
        default: Default selection (False = No, True = Yes)

    Returns:
        True if confirmed, False otherwise

    Example:
        if arrow_confirm("Start training?", default=True):
            # User confirmed
            pass
    """
    if not TERMINAL_MENU_AVAILABLE:
        return None  # Caller should fall back

    options = ["Yes", "No"]
    default_index = 0 if default else 1

    menu = TerminalMenu(
        options,
        title=f"\n  {message}\n",
        cursor_index=default_index,
        menu_cursor=" → ",
        menu_cursor_style=("fg_cyan", "bold"),
        quit_keys=("escape", "q"),
    )

    selected = menu.show()

    if selected is None:
        return False  # Escape = No
    return selected == 0  # 0 = Yes
