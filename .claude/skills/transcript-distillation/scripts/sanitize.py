"""Deterministic secret redaction for transcript rows.

Config-driven (see configs/transcript_import/default.yaml `sanitize:`). Catches
high-entropy credential formats and sensitive KEY=VALUE assignments. Applied at
emit time so raw secrets never reach disk.

This is a SECRET scrubber, not a PII scrubber — it targets keys/tokens that a
model could memorize and regurgitate. For names/emails/paths, run the OPF-backed
SynthChat sanitize as a separate pass.
"""
from __future__ import annotations
import re
from collections import Counter


class Redactor:
    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", True)
        self.replacement = cfg.get("replacement", "[REDACTED:{name}]")
        self._patterns = [(name, re.compile(pat))
                          for name, pat in cfg.get("patterns", {}).items()]
        env = cfg.get("env_assignment") or {}
        self._env_re = None
        if env.get("sensitive_key_re"):
            key_re = env["sensitive_key_re"].replace("(?i)", "")  # flag -> compile arg
            # capture a sensitive key followed by = or : then the value to drop
            self._env_re = re.compile(
                rf"({key_re}\s*[:=]\s*)"
                r"(['\"]?)([^\s'\"]{4,})(\2)", re.IGNORECASE)
            self._env_replacement = env.get("replacement", "[REDACTED:env_secret]")

    def redact(self, text: str, counts: Counter | None = None):
        """Return redacted text; tally hits into `counts` if given."""
        if not self.enabled or not text:
            return text
        for name, rx in self._patterns:
            def _sub(m, n=name):
                if counts is not None:
                    counts[n] += 1
                return self.replacement.format(name=n)
            text = rx.sub(_sub, text)
        if self._env_re is not None:
            def _env_sub(m):
                if counts is not None:
                    counts["env_secret"] += 1
                return m.group(1) + self._env_replacement
            text = self._env_re.sub(_env_sub, text)
        return text
