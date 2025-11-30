"""
Main CLI entry point.

Location: tuner/cli/main.py
Purpose: Main entry point for Synaptic Tuner CLI
Used by: tuner.py wrapper, python -m tuner
"""

import sys
from .parser import create_parser
from .router import route_command


def main():
    """
    Main CLI entry point.

    Parses command-line arguments, routes to appropriate handler,
    and handles top-level errors gracefully.

    Exit Codes:
        0: Success
        1: General error
        130: Interrupted by user (Ctrl+C)

    Example:
        >>> if __name__ == "__main__":
        ...     main()
    """
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Route to handler and exit with its code
        exit_code = route_command(args)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\n\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
