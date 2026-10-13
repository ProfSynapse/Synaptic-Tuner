from dataclasses import replace
import os
from unittest.mock import Mock

import pytest

from shared.environments.bounded_local_runtime import BoundedLocalEnvironmentRuntime, FilesystemLimits
from shared.environments.fixture_parser import EnvironmentFixture


@pytest.fixture
def limits():
    return FilesystemLimits(30, 1024, 512, 2048, 8, 128, 30)


@pytest.fixture
def parent(tmp_path):
    path = tmp_path / 'private'
    path.mkdir(mode=0o700)
    return path


@pytest.fixture
def runtime(parent, limits):
    current = BoundedLocalEnvironmentRuntime(parent, limits)
    current.setup(EnvironmentFixture(['nested'], {'nested/source.txt': 'hello'}))
    yield current
    current.teardown()


def owned(runtime, parent):
    return parent / runtime._name


def test_actual_interface_actions_and_snapshot(runtime, parent):
    runtime.mkdir('a/b')
    runtime.write_text('a/b/x', 'world')
    assert runtime.read_text('a/b/x') == 'world'
    assert runtime.list_dir('a/b') == ['x']
    assert runtime.search('world') == ['a/b/x']
    runtime.copy('a', 'copy')
    runtime.move('copy/b/x', 'moved')
    assert runtime.read_text('moved') == 'world'
    assert not runtime.exists('copy/b/x')
    runtime.delete('copy', recursive=True)
    result = runtime.snapshot(limit=30)
    assert str(parent) not in str(result)
    assert 'root' not in result
    assert {'path': 'moved', 'size': 5} in result['files']
    runtime.teardown()
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize('path', ['/absolute', '../outside', 'a/../b', 'a//b', 'a/./b', r'a\b', 'C:/x', ''])
def test_invalid_fixture_paths_before_effects(parent, limits, path):
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    with pytest.raises(ValueError):
        runtime.setup(EnvironmentFixture([], {path: 'x'}))
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize('value', [0, -1, True, 1.5, float('inf')])
def test_exact_positive_limits(value):
    with pytest.raises(ValueError):
        FilesystemLimits(value, 1, 1, 1, 1, 1, 1)


def test_oversized_fixture_prevalidated(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    with pytest.raises(ValueError):
        runtime.setup(EnvironmentFixture([], {'file': 'x' * 1025}))
    assert list(parent.iterdir()) == []


def test_unsupported_platform_no_effects(parent, limits, monkeypatch):
    monkeypatch.setattr(os, 'supports_dir_fd', set())
    with pytest.raises(RuntimeError, match='facilities'):
        BoundedLocalEnvironmentRuntime(parent, limits).setup(EnvironmentFixture())
    assert list(parent.iterdir()) == []


def test_private_parent_required(tmp_path, limits):
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match='0700'):
        BoundedLocalEnvironmentRuntime(tmp_path, limits).setup(EnvironmentFixture())
    assert list(tmp_path.iterdir()) == []


def test_setup_failure_cleans_owned_partial(parent, limits, monkeypatch):
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    monkeypatch.setattr(runtime, 'write_text', Mock(side_effect=OSError('injected')))
    with pytest.raises(OSError):
        runtime.setup(EnvironmentFixture(['partial/sub'], {'partial/sub/x': 'x'}))
    assert list(parent.iterdir()) == []
    assert runtime._root_fd is None


def test_capacity_overwrite_copy_no_destructive_rejection(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, replace(limits, max_content_bytes=8))
    runtime.setup(EnvironmentFixture([], {'a': '1234', 'b': 'xy'}))
    runtime.write_text('a', '123456')
    with pytest.raises(ValueError):
        runtime.write_text('a', '1234567')
    assert runtime.read_text('a') == '123456'
    with pytest.raises(ValueError):
        runtime.copy('a', 'b', overwrite=True)
    assert runtime.read_text('b') == 'xy'
    runtime.teardown()


def test_collision_copy_and_explicit_overwrite(runtime):
    runtime.write_text('destination', 'before')
    with pytest.raises(FileExistsError):
        runtime.copy('nested/source.txt', 'destination')
    assert runtime.read_text('destination') == 'before'
    runtime.copy('nested/source.txt', 'destination', overwrite=True)
    assert runtime.read_text('destination') == 'hello'


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'fifo'])
def test_hostile_leaf_rejected(runtime, parent, tmp_path, kind):
    root = owned(runtime, parent)
    outside = tmp_path / 'outside'
    outside.write_text('untouched')
    leaf = root / 'hostile'
    if kind == 'symlink':
        leaf.symlink_to(outside)
    elif kind == 'hardlink':
        os.link(outside, leaf)
    else:
        os.mkfifo(leaf)
    with pytest.raises(ValueError):
        runtime.write_text('hostile', 'overwrite')
    with pytest.raises(ValueError):
        runtime.snapshot(limit=30)
    assert outside.read_text() == 'untouched'
    leaf.unlink()


def test_symlink_ancestor_rejected(runtime, parent, tmp_path):
    link = owned(runtime, parent) / 'redirect'
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        runtime.write_text('redirect/outside', 'x')
    assert not (tmp_path / 'outside').exists()
    link.unlink()


def test_scratch_substitution_retains_uncertainty(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture([], {'file': 'retained'}))
    root = owned(runtime, parent)
    moved = parent / 'retained'
    root.rename(moved)
    root.mkdir(mode=0o700)
    (root / 'foreign').write_text('preserved')
    with pytest.raises(RuntimeError, match='identity'):
        runtime.teardown()
    assert (root / 'foreign').read_text() == 'preserved'
    assert (moved / 'file').read_text() == 'retained'
    (root / 'foreign').unlink()
    root.rmdir()
    moved.rename(root)
    runtime.teardown()


def test_private_ancestor_substitution(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture([], {'file': 'retained'}))
    moved = parent.with_name('retained-parent')
    parent.rename(moved)
    parent.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match='ancestor'):
        runtime.read_text('file')
    with pytest.raises(RuntimeError, match='ancestor'):
        runtime.teardown()
    assert list(parent.iterdir()) == []
    parent.rmdir()
    moved.rename(parent)
    runtime.teardown()


def test_deadline_cooperative_cleanup_still_bounded(runtime):
    runtime._deadline = 0
    with pytest.raises(TimeoutError):
        runtime.read_text('nested/source.txt')
    runtime.teardown()


def test_returned_bytes_and_snapshot_bounds(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, replace(limits, max_output_bytes=4, max_snapshot_bytes=20))
    runtime.setup(EnvironmentFixture([], {'unicode': 'ééé'}))
    with pytest.raises(ValueError, match='output'):
        runtime.read_text('unicode')
    with pytest.raises(ValueError, match='output'):
        runtime.list_dir()
    with pytest.raises(ValueError, match='output'):
        runtime.snapshot(limit=30)
    runtime.teardown()


def test_entry_depth_and_snapshot_limits(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, replace(limits, max_entries=2, max_depth=2))
    runtime.setup(EnvironmentFixture([], {'a': 'a'}))
    with pytest.raises(ValueError):
        runtime.mkdir('x/y')
    with pytest.raises(ValueError):
        runtime.write_text('x/y/z', 'z')
    with pytest.raises(ValueError):
        runtime.snapshot(limit=3)
    runtime.write_text('b', 'b')
    assert runtime.snapshot(limit=1)['truncated'] is True
    runtime.teardown()


def test_cleanup_entry_substitution_never_deletes_replacement(parent, limits, monkeypatch):
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture([], {'file': 'original'}))
    original_inventory = runtime._inventory
    root = owned(runtime, parent)
    def substitute(*, cleanup=False):
        rows = original_inventory(cleanup=cleanup)
        if cleanup:
            (root / 'file').rename(root / 'retained-original')
            (root / 'file').write_text('replacement')
        return rows
    monkeypatch.setattr(runtime, '_inventory', substitute)
    with pytest.raises(RuntimeError, match='authority uncertain'):
        runtime.teardown()
    assert (root / 'file').read_text() == 'replacement'
    assert (root / 'retained-original').read_text() == 'original'
    monkeypatch.setattr(runtime, '_inventory', original_inventory)
    runtime.teardown()


def test_relative_ancestor_changed_during_read_rejected(runtime, parent, monkeypatch):
    original_read = os.read
    root = owned(runtime, parent)
    changed = False
    def substitute(fd, count):
        nonlocal changed
        raw = original_read(fd, count)
        if not changed:
            changed = True
            (root / 'nested').rename(root / 'retained-nested')
            (root / 'nested').mkdir(mode=0o700)
        return raw
    monkeypatch.setattr(os, 'read', substitute)
    with pytest.raises(RuntimeError, match='ancestor'):
        runtime.read_text('nested/source.txt')
    assert (root / 'retained-nested/source.txt').read_text() == 'hello'


def test_recursive_work_bound_rejects_externally_added_entries(runtime, parent):
    root = owned(runtime, parent)
    for index in range(runtime.limits.max_entries):
        (root / f'extra{index}').write_text('')
    for operation in [lambda: runtime.search('hello'), lambda: runtime.snapshot(limit=30),
                      lambda: runtime.copy('nested', 'copy')]:
        with pytest.raises(ValueError, match='entry'):
            operation()
    with pytest.raises(ValueError, match='entry'):
        runtime.teardown()
    assert (root / 'nested/source.txt').read_text() == 'hello'
    for index in range(runtime.limits.max_entries):
        (root / f'extra{index}').unlink()


def test_open_failure_after_scratch_creation_cleans_owned_child(parent, limits, monkeypatch):
    original_open = os.open
    def refuse(path, flags, *args, **kwargs):
        if str(path).startswith('fixture-'):
            raise OSError('injected open failure')
        return original_open(path, flags, *args, **kwargs)
    # Platform membership uses function identity, so preflight before replacing open.
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    runtime._platform()
    monkeypatch.setattr(runtime, '_platform', lambda: None)
    monkeypatch.setattr(os, 'open', refuse)
    with pytest.raises(OSError):
        runtime.setup(EnvironmentFixture())
    assert list(parent.iterdir()) == []
    assert runtime._anchors == []


def test_unrepresentable_deadline_rejected_before_effects(parent):
    with pytest.raises(ValueError, match='represented'):
        FilesystemLimits(1, 1, 1, 1, 1, 1, 10 ** 1000)
    assert list(parent.iterdir()) == []


def test_fixture_cumulative_budget_before_retaining_next_file(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, replace(limits, max_content_bytes=8))
    with pytest.raises(ValueError, match='content'):
        runtime.setup(EnvironmentFixture([], {'a': '123456', 'b': '123456'}))
    assert list(parent.iterdir()) == []


def test_move_requires_temporary_copy_capacity(parent, limits):
    runtime = BoundedLocalEnvironmentRuntime(parent, replace(limits, max_content_bytes=8))
    runtime.setup(EnvironmentFixture([], {'file': '123456'}))
    with pytest.raises(ValueError):
        runtime.move('file', 'moved')
    assert runtime.read_text('file') == '123456'
    assert not runtime.exists('moved')
    runtime.teardown()


def test_fresh_import_does_not_attempt_provider_sdk():
    import subprocess
    import sys
    from pathlib import Path
    source = str(Path(__file__).resolve().parents[2])
    script = '''
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def forbid(name, *args, **kwargs):
    if name.split('.')[0] in {'e2b', 'e2b_code_interpreter', 'modal', 'huggingface_hub'}:
        raise AssertionError('provider SDK import attempted')
    return original(name, *args, **kwargs)
builtins.__import__ = forbid
from shared.environments.bounded_local_runtime import BoundedLocalEnvironmentRuntime
assert BoundedLocalEnvironmentRuntime.__name__ == 'BoundedLocalEnvironmentRuntime'
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, source],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_lazy_exports_preserve_actual_class_identity():
    import shared.environments as public
    from shared.environments.validator import EnvironmentSession, EnvironmentValidator
    assert public.EnvironmentSession is EnvironmentSession
    assert public.EnvironmentValidator is EnvironmentValidator
    assert 'EnvironmentSession' in public.__all__
    assert 'EnvironmentValidator' in public.__all__


def test_first_stat_failure_retains_unestablished_creation(parent, limits, monkeypatch):
    original_stat = os.stat
    def refuse(path, *args, **kwargs):
        if str(path).startswith('fixture-'):
            raise OSError('injected first stat failure')
        return original_stat(path, *args, **kwargs)
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    monkeypatch.setattr(runtime, '_platform', lambda: None)
    monkeypatch.setattr(os, 'stat', refuse)
    with pytest.raises(RuntimeError, match='unestablished'):
        runtime.setup(EnvironmentFixture())
    assert runtime._created and runtime._created_identity is None
    assert runtime._anchors and runtime._name
    assert len(list(parent.iterdir())) == 1
    with pytest.raises(RuntimeError, match='already initialized'):
        runtime.setup(EnvironmentFixture())
    with pytest.raises(RuntimeError, match='unestablished'):
        runtime.teardown()
    # Test-only host reconciliation, not runtime's deletion authority.
    monkeypatch.setattr(os, 'stat', original_stat)
    (parent / runtime._name).rmdir()
    runtime._close()


def test_opened_root_identity_must_match_created(parent, limits, monkeypatch):
    original_open = os.open
    def substitute(path, flags, *args, **kwargs):
        if str(path).startswith('fixture-'):
            return os.dup(kwargs['dir_fd'])
        return original_open(path, flags, *args, **kwargs)
    runtime = BoundedLocalEnvironmentRuntime(parent, limits)
    monkeypatch.setattr(runtime, '_platform', lambda: None)
    monkeypatch.setattr(os, 'open', substitute)
    with pytest.raises(RuntimeError, match='before opening'):
        runtime.setup(EnvironmentFixture())
    assert runtime._root_fd is None
    assert list(parent.iterdir()) == []
