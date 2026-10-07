"""Trusted host-project plug-in used by embedded-engine contract tests."""


def render(subject: str) -> str:
    """Return deterministic host-owned content without writing source files."""
    return f"Assess the evidence for: {subject}"
