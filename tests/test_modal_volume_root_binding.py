"""Linux checks for the isolated Modal root-link transport candidate."""

import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

import pytest

from tuner.execution.providers.modal import volume_root_binding as binding_module
from tuner.execution.providers.modal.volume_root_binding import (
    VolumeRootBinding,
    VolumeRootBindingError,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor API")


@pytest.fixture
def layout(monkeypatch):
    # /tmp is sticky and world writable, so the fixture lives under a private
    # home directory just as a reviewed provider target must.
    with tempfile.TemporaryDirectory(prefix="modal-volume-root-", dir=Path.home()) as directory:
        base = Path(directory)
        base.chmod(0o700)
        (base / "roots").mkdir(mode=0o700)
        (base / "volumes").mkdir(mode=0o700)
        monkeypatch.setattr(binding_module, "_PROVIDER_VOLUME_ROOT", str(base / "volumes"))
        target = base / "volumes" / "vo-Control"
        target.mkdir(mode=0o700)
        marker = b"m" * 32
        (target / "claim.marker").write_bytes(marker)
        root = base / "roots" / "control"
        root.symlink_to(target, target_is_directory=True)
        yield root, base / "volumes", target, marker


def _bind(layout):
    root, _, _, marker = layout
    return VolumeRootBinding.bind(
        root_path=str(root),
        volume_id="vo-Control",
        marker_name="claim.marker",
        marker_sha256=hashlib.sha256(marker).hexdigest(),
    )


def test_bound_root_reads_lists_and_exclusively_writes(layout):
    _, _, target, marker = layout
    with _bind(layout) as bound:
        assert bound.read_regular("claim.marker", 32) == marker
        assert bound.list_regular("", 1) == (("claim.marker", 32),)
        bound.claim_directory("output")
        bound.write_exclusive("output/member", b"ok", 2)
        assert bound.read_regular("output/member", 2) == b"ok"
        assert bound.list_regular("output", 1) == (("member", 2),)
        with pytest.raises(VolumeRootBindingError) as collision:
            bound.claim_directory("output")
        assert collision.value.code == "CLAIM_CREATE_EXISTS"
        bound.write_exclusive("result.json", b"{}", 2)
        assert bound.read_regular("result.json", 2) == b"{}"
        with pytest.raises(VolumeRootBindingError):
            bound.list_regular("", 3)  # directories cannot pass as regular files
        with pytest.raises(VolumeRootBindingError):
            bound.write_exclusive("result.json", b"new", 3)
    assert (target / "result.json").read_bytes() == b"{}"


def test_marker_mismatch_fails_closed(layout):
    root, _, _, _ = layout
    with pytest.raises(VolumeRootBindingError):
        VolumeRootBinding.bind(
            root_path=str(root), volume_id="vo-Control",
            marker_name="claim.marker", marker_sha256="0" * 64,
        )


def test_wrong_volume_id_fails_closed(layout):
    root, _, target, marker = layout
    with pytest.raises(VolumeRootBindingError):
        VolumeRootBinding.bind(
            root_path=str(root), volume_id="vo-Other",
            marker_name="claim.marker", marker_sha256=hashlib.sha256(marker).hexdigest(),
        )


def test_exact_volume_id_target_is_allowed(layout):
    root, _, target, marker = layout
    with VolumeRootBinding.bind(
        root_path=str(root), volume_id="vo-Control",
        marker_name="claim.marker", marker_sha256=hashlib.sha256(marker).hexdigest(),
    ) as bound:
        assert bound.read_regular("claim.marker", 32) == marker
        assert (bound.volume_id, bound.marker_name, bound.marker_sha256) == (
            "vo-Control", "claim.marker", hashlib.sha256(marker).hexdigest(),
        )


@pytest.mark.parametrize("volume_id", ("vo-Control/child", "vo-", "control", "vo-Control..", "vo-Control\x00"))
def test_invalid_volume_id_fails_before_root_access(layout, volume_id):
    root, _, _, marker = layout
    with pytest.raises(VolumeRootBindingError):
        VolumeRootBinding.bind(
            root_path=str(root), volume_id=volume_id,
            marker_name="claim.marker", marker_sha256=hashlib.sha256(marker).hexdigest(),
        )


def test_link_to_child_of_correct_volume_is_rejected(layout):
    root, _, target, marker = layout
    child = target / "child"
    child.mkdir()
    (child / "claim.marker").write_bytes(marker)
    root.unlink()
    root.symlink_to(child, target_is_directory=True)
    with pytest.raises(VolumeRootBindingError):
        _bind(layout)


def test_swapped_link_to_other_volume_with_same_marker_is_rejected(layout):
    root, _, target, marker = layout
    other = target.parent / "vo-Other"
    other.mkdir()
    (other / "claim.marker").write_bytes(marker)
    with _bind(layout) as bound:
        root.unlink()
        root.symlink_to(other, target_is_directory=True)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("claim.marker", 32)


def test_ancestor_replacement_is_rejected(layout):
    root, _, target, _ = layout
    with _bind(layout) as bound:
        parent = root.parent
        moved = parent.with_name("old-roots")
        parent.rename(moved)
        parent.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("claim.marker", 32)


def test_target_ancestor_replacement_is_rejected(layout):
    root, prefix, target, marker = layout
    with _bind(layout) as bound:
        moved = prefix.with_name("old-volumes")
        prefix.rename(moved)
        prefix.mkdir(mode=0o700)
        replacement = prefix / "vo-Control"
        replacement.mkdir(mode=0o700)
        (replacement / "claim.marker").write_bytes(marker)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("claim.marker", 32)
    assert (moved / "vo-Control" / "claim.marker").read_bytes() == marker


def test_failed_bind_closes_retained_descriptors_without_removing_marker(layout, monkeypatch):
    root, _, target, marker = layout
    original_chain = binding_module._chain
    retained = []

    def track_chain(parts):
        chain = original_chain(parts)
        retained.extend(chain.descriptors)
        return chain

    monkeypatch.setattr(binding_module, "_chain", track_chain)
    with pytest.raises(VolumeRootBindingError):
        VolumeRootBinding.bind(
            root_path=str(root), volume_id="vo-Control",
            marker_name="claim.marker", marker_sha256="0" * 64,
        )
    assert retained
    for descriptor in retained:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert root.is_symlink()
    assert (target / "claim.marker").read_bytes() == marker


def test_stream_copy_in_and_out_with_exclusive_destinations(layout, monkeypatch):
    _, _, target, _ = layout
    scratch = target.parent.parent / "scratch"
    scratch.mkdir(mode=0o700)
    content = b"a" * (1024 * 1024 + 17)
    source = scratch / "source.bin"
    source.write_bytes(content)
    output = scratch / "output.bin"
    digest = hashlib.sha256(content).hexdigest()
    original_read = os.read
    sizes = []

    def bounded_read(fd, size):
        sizes.append(size)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", bounded_read)
    with _bind(layout) as bound:
        bound.copy_in_exclusive("cache.bin", str(source), expected_size=len(content), expected_sha256=digest, maximum=32 * 1024**3)
        bound.copy_out("cache.bin", str(output), expected_size=len(content), expected_sha256=digest, maximum=32 * 1024**3)
        with pytest.raises(VolumeRootBindingError):
            bound.copy_out("cache.bin", str(output), expected_size=len(content), expected_sha256=digest, maximum=len(content))
        with pytest.raises(VolumeRootBindingError) as collision:
            bound.copy_in_exclusive("cache.bin", str(source), expected_size=len(content), expected_sha256=digest, maximum=len(content))
        assert collision.value.code == "COPY_DEST_CREATE_EXISTS"
    assert sizes and max(sizes) <= 1024 * 1024
    assert (target / "cache.bin").read_bytes() == content
    assert output.read_bytes() == content


def test_copy_rejects_bad_digest_and_retains_failed_volume_leaf(layout):
    _, _, target, _ = layout
    scratch = target.parent.parent / "scratch"
    scratch.mkdir(mode=0o700)
    source = scratch / "source.bin"
    source.write_bytes(b"source")
    with _bind(layout) as bound:
        with pytest.raises(VolumeRootBindingError) as digest_failure:
            bound.copy_in_exclusive("bad.bin", str(source), expected_size=6, expected_sha256="0" * 64, maximum=6)
        assert digest_failure.value.code == "COPY_STREAM_HASH"
        assert (target / "bad.bin").read_bytes() == b"source"
        (target / "good.bin").write_bytes(b"good")
        with pytest.raises(VolumeRootBindingError):
            bound.copy_out("good.bin", str(scratch / "bad-output.bin"), expected_size=4, expected_sha256="0" * 64, maximum=4)
        assert not (scratch / "bad-output.bin").exists()
    assert source.read_bytes() == b"source"
    assert (target / "good.bin").read_bytes() == b"good"


def test_copy_rejects_private_source_link_and_destination_collision(layout):
    _, _, target, _ = layout
    scratch = target.parent.parent / "scratch"
    scratch.mkdir(mode=0o700)
    source = scratch / "source.bin"
    source.write_bytes(b"source")
    (scratch / "alias.bin").symlink_to(source)
    output = scratch / "output.bin"
    output.write_bytes(b"existing")
    digest = hashlib.sha256(b"source").hexdigest()
    with _bind(layout) as bound:
        with pytest.raises(VolumeRootBindingError):
            bound.copy_in_exclusive("linked.bin", str(scratch / "alias.bin"), expected_size=6, expected_sha256=digest, maximum=6)
        bound.copy_in_exclusive("source.bin", str(source), expected_size=6, expected_sha256=digest, maximum=6)
        with pytest.raises(VolumeRootBindingError) as collision:
            bound.copy_out("source.bin", str(output), expected_size=6, expected_sha256=digest, maximum=6)
        assert collision.value.code == "COPY_DEST_CREATE_EXISTS"
    assert not (target / "linked.bin").exists()
    assert output.read_bytes() == b"existing"


def test_copy_rejects_changed_source_identity(layout, monkeypatch):
    _, _, target, _ = layout
    scratch = target.parent.parent / "scratch"
    scratch.mkdir(mode=0o700)
    source = scratch / "source.bin"
    source.write_bytes(b"source")
    original_read = os.read
    changed = False

    def replace_after_read(fd, size):
        nonlocal changed
        data = original_read(fd, size)
        if not changed:
            changed = True
            source.rename(scratch / "old.bin")
            source.write_bytes(b"source")
        return data

    with _bind(layout) as bound:
        monkeypatch.setattr(os, "read", replace_after_read)
        with pytest.raises(VolumeRootBindingError) as changed_source:
            bound.copy_in_exclusive("changed.bin", str(source), expected_size=6, expected_sha256=hashlib.sha256(b"source").hexdigest(), maximum=6)
        assert changed_source.value.code == "COPY_SOURCE_RECHECK"
    assert changed
    assert (target / "changed.bin").read_bytes() == b"source"


@pytest.mark.parametrize("operation,code", (
    ("write", "COPY_STREAM_WRITE"), ("fsync", "COPY_STREAM_FSYNC"),
))
def test_copy_reports_fixed_stream_operation_without_os_detail(layout, monkeypatch, operation, code):
    _, _, target, _ = layout
    scratch = target.parent.parent / "scratch"
    scratch.mkdir(mode=0o700)
    source = scratch / "source.bin"
    source.write_bytes(b"source")

    def denied(*_args, **_kwargs):
        raise PermissionError("PRIVATE_SENTINEL")

    with _bind(layout) as bound:
        monkeypatch.setattr(os, operation, denied)
        with pytest.raises(VolumeRootBindingError) as caught:
            bound.copy_in_exclusive(
                "stream.bin", str(source), expected_size=6,
                expected_sha256=hashlib.sha256(b"source").hexdigest(), maximum=6,
            )
    assert caught.value.code == code
    assert str(caught.value) == "volume root binding invalid"
    assert caught.value.__cause__ is None


def test_copy_accepts_private_scratch_under_sticky_tmp(layout):
    _, _, target, _ = layout
    with tempfile.TemporaryDirectory(prefix="modal-copy-", dir="/tmp") as directory:
        scratch = Path(directory)
        assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
        source = scratch / "source.bin"
        output = scratch / "output.bin"
        source.write_bytes(b"scratch")
        digest = hashlib.sha256(b"scratch").hexdigest()
        with _bind(layout) as bound:
            bound.copy_in_exclusive("scratch.bin", str(source), expected_size=7, expected_sha256=digest, maximum=7)
            bound.copy_out("scratch.bin", str(output), expected_size=7, expected_sha256=digest, maximum=7)
        assert output.read_bytes() == b"scratch"
        assert (target / "scratch.bin").read_bytes() == b"scratch"


def test_selected_root_child_is_private_and_admitted_for_publication(layout):
    from tuner.runtime import runtime_release_modal_training as entrypoint

    parent = entrypoint._PRIVATE_SCRATCH_ROOT
    assert parent == Path("/")
    if not os.access(parent, os.W_OK):
        pytest.skip("selected scratch parent requires a privileged local fixture")
    _, _, target, _ = layout
    with tempfile.TemporaryDirectory(prefix="synaptic-model-", dir=parent) as directory:
        scratch = Path(directory)
        assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
        source = scratch / "source.bin"
        source.write_bytes(b"source")
        with _bind(layout) as bound:
            bound.copy_in_exclusive(
                "root-child.bin", str(source), expected_size=6,
                expected_sha256=hashlib.sha256(b"source").hexdigest(), maximum=6,
            )
    assert (target / "root-child.bin").read_bytes() == b"source"


@pytest.mark.parametrize("owner,mode,code", (
    (1, stat.S_IFDIR | 0o1777, "SOURCE_CHAIN_TMP_OWNER"),
    (0, stat.S_IFDIR | 0o0755, "SOURCE_CHAIN_TMP_MODE_NONWRITABLE"),
    (0, stat.S_IFDIR | 0o0777, "SOURCE_CHAIN_TMP_MODE_WRITABLE"),
    (0, stat.S_IFREG | 0o0644, "SOURCE_CHAIN_TMP"),
))
def test_tmp_chain_reports_exact_closed_predicate(layout, monkeypatch, owner, mode, code):
    _, _, target, _ = layout
    with tempfile.TemporaryDirectory(prefix="modal-copy-", dir="/tmp") as directory:
        source = Path(directory) / "source.bin"
        source.write_bytes(b"source")
        original_fstat = os.fstat

        def reported_tmp_stat(fd):
            info = original_fstat(fd)
            if os.readlink(f"/proc/self/fd/{fd}") != "/tmp":
                return info
            fields = list(info)
            fields[0], fields[4] = mode, owner
            return os.stat_result(fields)

        with _bind(layout) as bound:
            monkeypatch.setattr(os, "fstat", reported_tmp_stat)
            with pytest.raises(VolumeRootBindingError) as caught:
                bound.copy_in_exclusive(
                    "rejected.bin", str(source), expected_size=6,
                    expected_sha256=hashlib.sha256(b"source").hexdigest(), maximum=6,
                )
        assert caught.value.code == code
        assert str(caught.value) == "volume root binding invalid"
        assert not (target / "rejected.bin").exists()


def test_copy_accepts_sdk_directories_beneath_private_tmp_anchor(layout):
    _, _, target, _ = layout
    with tempfile.TemporaryDirectory(prefix="modal-copy-", dir="/tmp") as directory:
        scratch = Path(directory)
        repository = scratch / "repository"
        repository.mkdir(mode=0o755)
        nested = repository / "config"
        nested.mkdir(mode=0o755)
        source = nested / "model.json"
        source.write_bytes(b"model")
        digest = hashlib.sha256(b"model").hexdigest()
        with _bind(layout) as bound:
            bound.copy_in_exclusive(
                "model.json", str(source), expected_size=5,
                expected_sha256=digest, maximum=5,
            )
        assert (target / "model.json").read_bytes() == b"model"


@pytest.mark.parametrize("mode", (0o1777, 0o0777, 0o0770))
def test_copy_rejects_hostile_private_ancestor(layout, mode):
    _, _, target, _ = layout
    hostile = target.parent.parent / "hostile"
    hostile.mkdir(mode=0o700)
    scratch = hostile / "scratch"
    scratch.mkdir(mode=0o700)
    source = scratch / "source.bin"
    source.write_bytes(b"source")
    hostile.chmod(mode)
    try:
        with _bind(layout) as bound:
            with pytest.raises(VolumeRootBindingError) as hostile_source:
                bound.copy_in_exclusive("rejected.bin", str(source), expected_size=6, expected_sha256=hashlib.sha256(b"source").hexdigest(), maximum=6)
            assert hostile_source.value.code == "SOURCE_CHAIN_MODE"
        assert not (target / "rejected.bin").exists()
    finally:
        hostile.chmod(0o700)


def test_world_writable_ancestor_fails_closed(layout):
    root, prefix, _, marker = layout
    original_mode = stat.S_IMODE(prefix.stat().st_mode)
    prefix.chmod(0o777)
    try:
        with pytest.raises(VolumeRootBindingError):
            VolumeRootBinding.bind(
                root_path=str(root), volume_id="vo-Control",
                marker_name="claim.marker", marker_sha256=hashlib.sha256(marker).hexdigest(),
            )
    finally:
        prefix.chmod(original_mode)


def test_leaf_and_intermediate_symlinks_are_rejected(layout):
    _, _, target, _ = layout
    (target / "alias").symlink_to("claim.marker")
    (target / "nest").mkdir()
    (target / "nest-link").symlink_to("nest", target_is_directory=True)
    with _bind(layout) as bound:
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("alias", 32)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("nest-link/file", 32)
        with pytest.raises(VolumeRootBindingError):
            bound.list_regular("", 4)
        for bad in ("../claim.marker", "/claim.marker", "nest//file", "./claim.marker"):
            with pytest.raises(VolumeRootBindingError):
                bound.read_regular(bad, 32)


def test_retargeted_root_is_rejected(layout):
    root, _, target, _ = layout
    with _bind(layout) as bound:
        root.unlink()
        root.symlink_to(target, target_is_directory=True)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("claim.marker", 32)


def test_marker_must_be_exactly_32_bytes(layout):
    root, _, target, _ = layout
    (target / "claim.marker").write_bytes(b"m" * 33)
    with pytest.raises(VolumeRootBindingError):
        VolumeRootBinding.bind(
            root_path=str(root), volume_id="vo-Control",
            marker_name="claim.marker", marker_sha256=hashlib.sha256(b"m" * 33).hexdigest(),
        )


def test_nested_directory_replacement_is_detected(layout, monkeypatch):
    _, _, target, _ = layout
    (target / "nested").mkdir()
    (target / "nested" / "file").write_bytes(b"old")
    original_read = os.read
    changed = False

    def replace_after_read(fd, size):
        nonlocal changed
        result = original_read(fd, size)
        if not changed:
            changed = True
            (target / "nested").rename(target / "moved")
            (target / "nested").mkdir()
        return result

    with _bind(layout) as bound:
        monkeypatch.setattr(os, "read", replace_after_read)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("nested/file", 3)


def test_fifo_swap_cannot_block_regular_read(layout, monkeypatch):
    _, _, target, _ = layout
    (target / "swappable").write_bytes(b"x")
    original_open = os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "swappable" and not swapped:
            assert flags & os.O_NONBLOCK
            swapped = True
            (target / "swappable").unlink()
            os.mkfifo(target / "swappable")
        return original_open(path, flags, *args, **kwargs)

    with _bind(layout) as bound:
        monkeypatch.setattr(os, "open", swap_before_open)
        with pytest.raises(VolumeRootBindingError):
            bound.read_regular("swappable", 1)
    assert swapped


def test_enter_failure_closes_owned_descriptors(layout):
    root, _, target, _ = layout
    bound = _bind(layout)
    root.unlink()
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(VolumeRootBindingError):
        bound.__enter__()
    assert bound._closed


def test_directory_claim_rejects_replacement(layout, monkeypatch):
    _, _, target, _ = layout
    original_open = os.open
    swapped = False

    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        fd = original_open(path, flags, *args, **kwargs)
        if path == "newdir" and not swapped:
            swapped = True
            (target / "newdir").rename(target / "old-dir")
            (target / "newdir").mkdir()
        return fd

    with _bind(layout) as bound:
        monkeypatch.setattr(os, "open", replace_after_open)
        with pytest.raises(VolumeRootBindingError):
            bound.claim_directory("newdir")
    assert swapped


def test_bind_interrupt_closes_retained_parent_descriptors(layout, monkeypatch):
    root, _, _, marker = layout
    original_chain = binding_module._chain
    retained = []

    def interrupt_after_parent(parts):
        if retained:
            raise KeyboardInterrupt
        chain = original_chain(parts)
        retained.extend(chain.descriptors)
        return chain

    monkeypatch.setattr(binding_module, "_chain", interrupt_after_parent)
    with pytest.raises(KeyboardInterrupt):
        VolumeRootBinding.bind(
            root_path=str(root), volume_id="vo-Control",
            marker_name="claim.marker", marker_sha256=hashlib.sha256(marker).hexdigest(),
        )
    assert retained
    for descriptor in retained:
        with pytest.raises(OSError):
            os.fstat(descriptor)
