from __future__ import annotations

from dataclasses import replace

import pytest

from tests.execution.providers.test_modal_source_resolution import _source
from tuner.project.git_verification import (
    GitCliLocalSourceInspector, GitLsRemotePushedCommitVerifier,
)
from tuner.project.context import ProjectContext
from tuner.project.source_bundle import SourceLockError


class Auth:
    def sign(self,purpose,payload,key_ref):return b"authenticated-tag"
    def verify(self,purpose,payload,tag,key_ref):return tag==b"authenticated-tag"


class Runner:
    def __init__(self,values):self.values=values;self.calls=[]
    def read_ref(self,*,canonical_url,exact_ref):self.calls.append((canonical_url,exact_ref));return self.values[(canonical_url,exact_ref)]


def source_with_branches():
    value=_source();return replace(value,project_source=replace(value.project_source,branch="main"),engine_source=replace(value.engine_source,branch="release"))


def verifier(source,runner):
    return GitLsRemotePushedCommitVerifier(runner,Auth(),clock=lambda:"2026-08-25T12:00:00Z",audience_ref="project/run-1",issuer_ref="git-verifier",key_ref="git-key",challenge_factory=lambda:"source-challenge",evidence_ref_factory=lambda:"source-evidence")


def test_pushed_verifier_requires_both_exact_named_refs_and_seals_evidence():
    source=source_with_branches();values={(source.project_source.location.canonical_url,"refs/heads/main"):f"{source.project_source.commit}\trefs/heads/main\n".encode(),(source.engine_source.location.canonical_url,"refs/heads/release"):f"{source.engine_source.commit}\trefs/heads/release\n".encode()};evidence=verifier(source,Runner(values)).verify(source)
    assert evidence.binds(source) and evidence.audience_ref=="project/run-1" and evidence.tag==b"authenticated-tag"


@pytest.mark.parametrize("bad",[b"",b"a"*40+b"\trefs/heads/other\n",b"a"*40+b"\trefs/heads/main\n"+b"a"*40+b"\trefs/heads/main\n"])
def test_pushed_verifier_fails_closed_on_moved_malformed_or_duplicate_ref(bad):
    source=source_with_branches();values={(source.project_source.location.canonical_url,"refs/heads/main"):bad,(source.engine_source.location.canonical_url,"refs/heads/release"):f"{source.engine_source.commit}\trefs/heads/release\n".encode()}
    with pytest.raises(SourceLockError):verifier(source,Runner(values)).verify(source)


def test_local_inspector_uses_committed_branch_for_detached_submodule(
    tmp_path, monkeypatch,
):
    project = tmp_path / "project"
    engine = project / "synaptic-tuner"
    engine.mkdir(parents=True)
    source = _source()
    project_source = replace(
        source.project_source, branch="host-branch",
    )
    detached_engine = replace(source.engine_source, branch=None)

    def inspect(repository, **kwargs):
        return project_source if repository == project else detached_engine

    modules = (
        '[submodule "synaptic-tuner"]\n'
        'path = synaptic-tuner\n'
        f'url = {detached_engine.location.canonical_url}\n'
        'branch = engine-branch\n'
    ).encode()
    tree = (
        f"160000 commit {detached_engine.commit}\tsynaptic-tuner\0"
    ).encode()

    def local_git(repository, *arguments):
        return modules if arguments[:2] == ("show", "HEAD:.gitmodules") else tree

    monkeypatch.setattr("tuner.project.git_verification.inspect_git_source", inspect)
    monkeypatch.setattr("tuner.project.git_verification._local_git", local_git)
    locked = GitCliLocalSourceInspector().inspect(
        context=ProjectContext.host(engine_root=engine, project_root=project)
    )
    assert locked.engine_source.branch == "engine-branch"
