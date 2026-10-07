"""Production Git inspection and pushed-ref verification algorithms."""

from __future__ import annotations

import base64
import configparser
import hashlib
import io
import os
import subprocess
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from tuner.execution.evidence import (
    EvidenceAuthenticator,
    SOURCE_EVIDENCE_POLICY,
    SOURCE_EVIDENCE_PURPOSE,
    parse_utc,
)

from .context import ProjectContext
from .execution_source import (
    AuthenticatedSourceEvidenceV1,
    LocalSourceInspectionPort,
    PushedSourceVerificationPort,
    _capture_source_evidence_snapshot,
    _require_source_evidence_snapshot,
)
from .source_bundle import (
    RepositoryLocation, SourceLock, SourceLockError,
    _capture_source_lock_snapshot, _require_source_lock_snapshot,
    canonical_remote_branch_ref, inspect_git_source,
    resolve_relative_repository_url,
)


def _local_git(repository: Path, *arguments: str) -> bytes:
    environment = {
        "PATH": os.environ.get("PATH", ""), "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never",
        "LC_ALL": "C", "LANG": "C",
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments], capture_output=True,
            check=True, timeout=10, env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceLockError("local Git inspection failed") from exc
    if len(result.stdout) > 1_048_576:
        raise SourceLockError("local Git inspection output exceeded its bound")
    return result.stdout


class GitCliLocalSourceInspector(LocalSourceInspectionPort):
    """Derive the superproject, submodule, committed URL, and gitlink from Git."""

    def inspect(self, *, context: ProjectContext) -> SourceLock:
        if not isinstance(context, ProjectContext) or context.mode != "host":
            raise SourceLockError("local Modal source inspection requires a host project")
        project = context.project_root.resolve(strict=True)
        engine = context.engine_root.resolve(strict=True)
        try:
            submodule_path = engine.relative_to(project).as_posix()
        except ValueError as exc:
            raise SourceLockError("engine checkout is outside the project") from exc
        project_source = inspect_git_source(project, remote_proof=lambda *_: False)
        raw_modules = _local_git(project, "show", "HEAD:.gitmodules")
        try:
            text = raw_modules.decode("utf-8")
            parser = configparser.ConfigParser(interpolation=None, strict=True)
            parser.read_file(io.StringIO(text))
        except (UnicodeError, configparser.Error) as exc:
            raise SourceLockError("committed .gitmodules is malformed") from exc
        matches = []
        for section in parser.sections():
            if section.startswith('submodule "') and section.endswith('"') and parser.get(section,"path",fallback=None)==submodule_path:
                matches.append((
                    parser.get(section,"url",fallback=None),
                    parser.get(section,"branch",fallback=None),
                ))
        if len(matches)!=1 or not matches[0][0] or not matches[0][1]:
            raise SourceLockError("engine submodule path is not uniquely committed")
        committed_url, committed_branch = matches[0]
        canonical_remote_branch_ref(committed_branch)
        committed_location = resolve_relative_repository_url(
            committed_url, project_source.location
        )
        raw_tree = _local_git(project,"ls-tree","-z","HEAD","--",submodule_path)
        records=[record for record in raw_tree.split(b"\0") if record]
        if len(records)!=1:
            raise SourceLockError("engine gitlink is not uniquely committed")
        try:
            metadata,path=records[0].split(b"\t",1);mode,kind,gitlink=metadata.decode("ascii").split(" ")
        except (ValueError,UnicodeError) as exc:
            raise SourceLockError("committed engine gitlink is malformed") from exc
        if mode!="160000" or kind!="commit" or path.decode("utf-8")!=submodule_path:
            raise SourceLockError("committed engine gitlink is malformed")
        engine_source=inspect_git_source(engine,submodule_path=submodule_path,gitlink_commit=gitlink,remote_proof=lambda *_:False)
        if (
            engine_source.location.canonical_url != committed_location.canonical_url
            or engine_source.commit.lower() != gitlink.lower()
            or engine_source.branch not in {None, committed_branch}
        ):
            raise SourceLockError("engine checkout differs from committed submodule identity")
        engine_source = replace(engine_source, branch=committed_branch)
        return SourceLock(run_id="local-inspection",mode="superproject",project_source=project_source,engine_source=engine_source,project={},configuration={})


@runtime_checkable
class ScopedGitRemoteRunner(Protocol):
    def read_ref(self, *, canonical_url: str, exact_ref: str) -> bytes: ...


class GitLsRemotePushedCommitVerifier(PushedSourceVerificationPort):
    """Verify both exact upstream refs and seal the result without owning credentials."""

    def __init__(self,runner:ScopedGitRemoteRunner,authenticator:EvidenceAuthenticator,*,clock:Callable[[],str],audience_ref:str,issuer_ref:str,key_ref:str,challenge_factory:Callable[[],str],evidence_ref_factory:Callable[[],str]):
        if not isinstance(runner,ScopedGitRemoteRunner) or not isinstance(authenticator,EvidenceAuthenticator):raise TypeError("remote runner and authenticator are required")
        self.runner=runner;self.authenticator=authenticator;self.clock=clock;self.audience_ref=audience_ref;self.issuer_ref=issuer_ref;self.key_ref=key_ref;self.challenge_factory=challenge_factory;self.evidence_ref_factory=evidence_ref_factory

    def _verify_ref(self,source,source_lock,baseline)->None:
        ref=canonical_remote_branch_ref(source.branch)
        try:raw=self.runner.read_ref(canonical_url=source.location.canonical_url,exact_ref=ref)
        except BaseException:raise SourceLockError("authenticated remote ref verification failed") from None
        _require_source_lock_snapshot(source_lock,baseline)
        if type(raw) is not bytes or len(raw)>4096:raise SourceLockError("remote ref proof is malformed")
        try:lines=[line for line in raw.decode("ascii").splitlines() if line]
        except BaseException:raise SourceLockError("remote ref proof is malformed") from None
        if len(lines)!=1 or lines[0]!=f"{source.commit.lower()}\t{ref}":raise SourceLockError("remote ref does not equal the locked commit")

    def verify(self,source_lock:SourceLock)->AuthenticatedSourceEvidenceV1:
        if type(source_lock) is not SourceLock or source_lock.mode!="superproject":raise TypeError("superproject source lock is required")
        baseline = _capture_source_lock_snapshot(source_lock)
        locked = baseline.canonical_lock
        self._verify_ref(locked.project_source,source_lock,baseline)
        self._verify_ref(locked.engine_source,source_lock,baseline)
        try:verified=self.clock()
        except BaseException:raise SourceLockError("source evidence clock is unavailable") from None
        _require_source_lock_snapshot(source_lock,baseline)
        try:expires=(parse_utc(verified)+timedelta(seconds=SOURCE_EVIDENCE_POLICY.maximum_lifetime_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except BaseException:raise SourceLockError("source evidence clock is unavailable") from None
        try:evidence_ref=self.evidence_ref_factory()
        except BaseException:raise SourceLockError("source evidence reference is unavailable") from None
        _require_source_lock_snapshot(source_lock,baseline)
        try:challenge=self.challenge_factory()
        except BaseException:raise SourceLockError("source evidence challenge is unavailable") from None
        _require_source_lock_snapshot(source_lock,baseline)
        value=AuthenticatedSourceEvidenceV1(project_url=locked.project_source.location.canonical_url,project_commit=locked.project_source.commit,engine_url=locked.engine_source.location.canonical_url,engine_commit=locked.engine_source.commit,engine_submodule_path=locked.engine_source.submodule_path,gitlink_commit=locked.engine_source.gitlink_commit,source_lock_binding=baseline.binding,issuer_ref=self.issuer_ref,evidence_ref=evidence_ref,audience_ref=self.audience_ref,challenge_nonce=challenge,verified_at=verified,expires_at=expires,key_ref=self.key_ref,tag_base64="dGFn",attestation_digest="0"*64)
        evidence_baseline=_capture_source_evidence_snapshot(value)
        attestation=hashlib.sha256(evidence_baseline.authenticated_payload).hexdigest()
        try:tag=self.authenticator.sign(SOURCE_EVIDENCE_PURPOSE,evidence_baseline.authenticated_payload,self.key_ref)
        except BaseException:raise SourceLockError("source evidence authentication is unavailable") from None
        _require_source_lock_snapshot(source_lock,baseline)
        _require_source_evidence_snapshot(value,evidence_baseline)
        if type(tag) is not bytes or not tag:
            raise SourceLockError("source evidence authentication is malformed")
        result=replace(evidence_baseline.evidence,attestation_digest=attestation,tag_base64=base64.b64encode(tag).decode("ascii"))
        result_baseline=_capture_source_evidence_snapshot(result)
        _require_source_lock_snapshot(source_lock,baseline)
        _require_source_evidence_snapshot(result,result_baseline)
        return result_baseline.evidence


__all__=["GitCliLocalSourceInspector","GitLsRemotePushedCommitVerifier","ScopedGitRemoteRunner"]
