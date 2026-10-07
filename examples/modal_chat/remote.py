"""Exact public HTTPS Git-ref reads for the minimal consumer's source finalizer."""

from __future__ import annotations

import subprocess

from tuner.project.source_bundle import RepositoryLocation, canonical_remote_branch_ref
from tuner.project.errors import RepositoryUrlError, SourceLockError


class ModalChatGitRemote:
    """Read only declared public repository/ref pairs with no ambient Git auth.

    This sample targets public HTTPS source. Private repositories need their
    own explicitly scoped credential resolver; this class never searches for
    credentials or falls back to an SSH agent or credential helper.
    """

    def __init__(self, allowed_refs: frozenset[tuple[str, str]]):
        if type(allowed_refs) is not frozenset or not allowed_refs:
            raise ValueError("explicit source refs required")
        normalized = set()
        for pair in allowed_refs:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("source ref is invalid")
            url, ref = pair
            try:
                location = RepositoryLocation.parse(url, allowed_schemes={"https"})
            except RepositoryUrlError:
                raise ValueError("modal_chat_source_ref_invalid") from None
            if location.canonical_url != url or type(ref) is not str:
                raise ValueError("source ref is not canonical")
            try:
                if (
                    not ref.startswith("refs/heads/")
                    or canonical_remote_branch_ref(ref[11:]) != ref
                ):
                    raise ValueError("exact branch ref required")
            except SourceLockError:
                raise ValueError("modal_chat_source_ref_invalid") from None
            normalized.add((url, ref))
        self._refs = frozenset(normalized)

    def read_ref(self, *, canonical_url: str, exact_ref: str) -> bytes:
        if (canonical_url, exact_ref) not in self._refs:
            raise ValueError("modal_chat_source_ref_not_allowed")
        try:
            result = subprocess.run(
                [
                    "/usr/bin/git",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "http.followRedirects=false",
                    "-c",
                    "protocol.allow=never",
                    "-c",
                    "protocol.https.allow=always",
                    "ls-remote",
                    "--exit-code",
                    "--refs",
                    canonical_url,
                    exact_ref,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
                cwd="/tmp",
                env={
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GCM_INTERACTIVE": "Never",
                },
            )
            if type(result.stdout) is not bytes or not 0 < len(result.stdout) <= 4096:
                raise ValueError
            # The production pushed-ref verifier checks exact SHA/ref equality.
            return result.stdout
        except Exception:
            raise ValueError("modal_chat_source_ref_unavailable") from None


__all__ = ["ModalChatGitRemote"]
