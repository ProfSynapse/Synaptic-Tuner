"""Unit tests for the pure helpers in ``tuner.handlers.local_run_handler``.

Covers: _validate_user_field, _current_host_ids, _resolve_user_spec,
_collect_chown_paths, _build_bash_wrapper, _chown_host_tree.

No docker, no network, no filesystem side effects (tmp_path OK).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tuner.handlers.local_run_handler import (
    LocalRunError,
    UserSpec,
    _build_bash_wrapper,
    _build_persistent_docker_run_args,
    _cache_mount_args,
    _chown_host_tree,
    _collect_chown_paths,
    _current_host_ids,
    _derive_container_name,
    _ensure_host_cache_dirs,
    _pip_marker_hash,
    _resolve_tty_flags,
    _resolve_user_spec,
    _validate_bool_field,
    _validate_tty_field,
    _validate_user_field,
)


# ---------------------------------------------------------------------------
# _validate_user_field
# ---------------------------------------------------------------------------


class TestValidateUserField:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, "auto"),
            ("auto", "auto"),
            ("AUTO", "auto"),
            ("  auto  ", "auto"),
            ("root", "root"),
            ("Root", "root"),
            ("image", "image"),
            ("IMAGE", "image"),
            ("1000:1000", "1000:1000"),
            ("0:0", "0:0"),
            ("65534:65534", "65534:65534"),
            ("  1000:1000  ", "1000:1000"),
        ],
    )
    def test_accepts_valid(self, raw, expected):
        assert _validate_user_field(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "1000",          # missing gid
            "a:b",           # non-numeric
            "",              # empty
            "   ",           # whitespace only
            "1000:",         # trailing colon
            ":1000",         # leading colon
            "1000:1000:1",   # too many parts
            "user",          # unknown keyword
            "-1:0",          # negative uid
        ],
    )
    def test_rejects_invalid(self, raw):
        with pytest.raises(LocalRunError):
            _validate_user_field(raw)


# ---------------------------------------------------------------------------
# _current_host_ids
# ---------------------------------------------------------------------------


class TestCurrentHostIds:
    def test_windows_returns_zero_zero(self, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "win32")
        assert _current_host_ids() == (0, 0)

    def test_linux_returns_os_ids(self, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.getuid", lambda: 1234, raising=False)
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.getgid", lambda: 5678, raising=False)
        assert _current_host_ids() == (1234, 5678)

    def test_darwin_returns_os_ids(self, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "darwin")
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.getuid", lambda: 501, raising=False)
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.getgid", lambda: 20, raising=False)
        assert _current_host_ids() == (501, 20)


# ---------------------------------------------------------------------------
# _resolve_user_spec
# ---------------------------------------------------------------------------


class TestResolveUserSpec:
    @pytest.mark.parametrize(
        "job_user,transfer,platform,expected",
        [
            # auto branches
            ("auto", "bind",  "linux",  ("0:0", 1000, 1000, False)),
            ("auto", "bind",  "darwin", ("0:0", 1000, 1000, False)),
            ("auto", "bind",  "win32",  ("0:0", None, None, True)),
            ("auto", "copy",  "linux",  (None,  1000, 1000, False)),
            ("auto", "copy",  "darwin", (None,  1000, 1000, False)),
            ("auto", "copy",  "win32",  (None,  None, None, True)),
            # root — independent of transfer/platform
            ("root", "bind",  "linux",  ("0:0", None, None, True)),
            ("root", "copy",  "linux",  ("0:0", None, None, True)),
            ("root", "bind",  "win32",  ("0:0", None, None, True)),
            # image — always no user flag, no chown
            ("image", "bind", "linux",  (None,  None, None, True)),
            ("image", "copy", "linux",  (None,  None, None, True)),
            ("image", "bind", "win32",  (None,  None, None, True)),
            # explicit uid:gid — flag set, uid/gid captured, skip_chown True
            ("1005:1005", "bind", "linux", ("1005:1005", 1005, 1005, True)),
            ("1005:1005", "copy", "darwin", ("1005:1005", 1005, 1005, True)),
            ("1005:1005", "bind", "win32",  ("1005:1005", 1005, 1005, True)),
            ("0:0",       "bind", "linux", ("0:0", 0, 0, True)),
        ],
    )
    def test_resolution_table(self, job_user, transfer, platform, expected):
        spec = _resolve_user_spec(job_user, transfer, 1000, 1000, platform)
        assert isinstance(spec, UserSpec)
        assert (
            spec.docker_user_flag,
            spec.chown_host_uid,
            spec.chown_host_gid,
            spec.skip_chown,
        ) == expected

    def test_root_ignores_host_ids(self):
        # root never reports host uid/gid as chown target.
        spec = _resolve_user_spec("root", "bind", 1234, 5678, "linux")
        assert spec.chown_host_uid is None and spec.chown_host_gid is None

    def test_explicit_uid_uses_literal_not_host(self):
        # Host ids are irrelevant when job_user is explicit uid:gid.
        spec = _resolve_user_spec("1005:1005", "bind", 9999, 8888, "linux")
        assert spec.docker_user_flag == "1005:1005"
        assert spec.chown_host_uid == 1005
        assert spec.chown_host_gid == 1005


# ---------------------------------------------------------------------------
# _collect_chown_paths
# ---------------------------------------------------------------------------


class TestCollectChownPaths:
    def test_ordering_distinct_paths(self):
        plan = {
            "container_artifact_path": "/workspace/repo/out/run-1",
            "workdir": "/workspace/repo/Trainers/sft",
        }
        assert _collect_chown_paths(plan) == [
            "/workspace/repo/out/run-1",
            "/workspace/repo/Trainers/sft",
            "/workspace/repo/toolset-training-artifacts",
        ]

    def test_dedup_when_workdir_equals_artifacts_root(self):
        plan = {
            "container_artifact_path": "/workspace/repo/out/run-1",
            "workdir": "/workspace/repo/toolset-training-artifacts",
        }
        # The workdir matches the default artifacts root — dedup to one entry.
        assert _collect_chown_paths(plan) == [
            "/workspace/repo/out/run-1",
            "/workspace/repo/toolset-training-artifacts",
        ]

    def test_dedup_when_artifact_equals_artifacts_root(self):
        plan = {
            "container_artifact_path": "/workspace/repo/toolset-training-artifacts",
            "workdir": "/workspace/repo/Trainers/sft",
        }
        assert _collect_chown_paths(plan) == [
            "/workspace/repo/toolset-training-artifacts",
            "/workspace/repo/Trainers/sft",
        ]

    def test_missing_artifact_and_workdir(self):
        # Only the implicit artifacts root survives.
        assert _collect_chown_paths({}) == ["/workspace/repo/toolset-training-artifacts"]

    def test_missing_workdir_only(self):
        plan = {"container_artifact_path": "/workspace/repo/out/run-1"}
        assert _collect_chown_paths(plan) == [
            "/workspace/repo/out/run-1",
            "/workspace/repo/toolset-training-artifacts",
        ]


# ---------------------------------------------------------------------------
# _build_bash_wrapper
# ---------------------------------------------------------------------------


class TestBuildBashWrapper:
    def _plan(self, **overrides):
        plan = {
            "command": ["python", "train.py", "--epochs", "3"],
            "workdir": "/workspace/repo/Trainers/sft",
            "container_artifact_path": "/workspace/repo/out/run-1",
            "pip": [],
        }
        plan.update(overrides)
        return plan

    def test_skip_chown_no_pip(self):
        spec = UserSpec(docker_user_flag=None, chown_host_uid=None, chown_host_gid=None, skip_chown=True)
        out = _build_bash_wrapper(self._plan(), spec)
        assert "trap" not in out
        assert "exec" not in out
        assert "pip install" not in out
        assert out == "python train.py --epochs 3"

    def test_no_chown_uid_no_pip(self):
        # skip_chown False but chown_host_uid None (e.g. image mode) — still no wrapper.
        spec = UserSpec(docker_user_flag=None, chown_host_uid=None, chown_host_gid=None, skip_chown=False)
        out = _build_bash_wrapper(self._plan(), spec)
        assert "trap" not in out
        assert "exec" not in out

    def test_skip_chown_with_pip(self):
        spec = UserSpec(docker_user_flag=None, chown_host_uid=None, chown_host_gid=None, skip_chown=True)
        plan = self._plan(pip=["torch==2.3.0", "transformers"])
        out = _build_bash_wrapper(plan, spec)
        assert "trap" not in out
        assert "exec" not in out
        assert out.startswith("pip install --upgrade ")
        assert "torch==2.3.0" in out
        assert "transformers" in out
        assert out.endswith(" && python train.py --epochs 3")

    def test_chown_active_no_pip(self):
        spec = UserSpec(docker_user_flag="0:0", chown_host_uid=1000, chown_host_gid=1000, skip_chown=False)
        out = _build_bash_wrapper(self._plan(), spec)
        assert out.startswith("trap ")
        assert "chown -R 1000:1000" in out
        assert " EXIT" in out
        assert "exec python train.py --epochs 3" in out
        assert "pip install" not in out

    def test_chown_active_with_pip(self):
        spec = UserSpec(docker_user_flag="0:0", chown_host_uid=1000, chown_host_gid=1000, skip_chown=False)
        plan = self._plan(pip=["datasets", "peft"])
        out = _build_bash_wrapper(plan, spec)
        # Expected shape: 'trap ...; pip install --upgrade datasets peft && exec python ...'
        assert out.startswith("trap ")
        assert "; pip install --upgrade " in out
        assert "datasets" in out
        assert "peft" in out
        assert " && exec python train.py --epochs 3" in out

    def test_command_arg_with_space_is_shlex_quoted(self):
        spec = UserSpec(docker_user_flag=None, chown_host_uid=None, chown_host_gid=None, skip_chown=True)
        plan = self._plan(command=["python", "train.py", "--name", "run one"])
        out = _build_bash_wrapper(plan, spec)
        # shlex.quote wraps args containing spaces in single quotes.
        assert "'run one'" in out

    def test_trap_targets_include_chown_targets_quoted(self):
        spec = UserSpec(docker_user_flag="0:0", chown_host_uid=1000, chown_host_gid=1000, skip_chown=False)
        plan = self._plan(
            container_artifact_path="/workspace/repo/out/run with space",
            workdir="/workspace/repo/Trainers/sft",
        )
        out = _build_bash_wrapper(plan, spec)
        # The path with a space must be shlex-quoted inside the trap.
        assert "'/workspace/repo/out/run with space'" in out
        # The artifacts root default is safe-to-unquote but present.
        assert "/workspace/repo/toolset-training-artifacts" in out
        # Workdir appears in trap too.
        assert "/workspace/repo/Trainers/sft" in out
        # Trap swallows errors.
        assert "|| true" in out
        assert "2>/dev/null" in out

    def test_pip_items_are_shlex_quoted(self):
        spec = UserSpec(docker_user_flag=None, chown_host_uid=None, chown_host_gid=None, skip_chown=True)
        plan = self._plan(pip=["pkg with space"])
        out = _build_bash_wrapper(plan, spec)
        assert "'pkg with space'" in out


# ---------------------------------------------------------------------------
# _chown_host_tree
# ---------------------------------------------------------------------------


class TestChownHostTree:
    def test_nonexistent_path_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")
        chown_mock = MagicMock()
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.chown", chown_mock, raising=False)
        missing = tmp_path / "does-not-exist"
        _chown_host_tree(missing, 1000, 1000)
        chown_mock.assert_not_called()

    def test_windows_no_chown_calls(self, tmp_path, monkeypatch):
        # Even if the path exists, Windows must early-return.
        (tmp_path / "file.txt").write_text("x")
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "win32")
        chown_mock = MagicMock()
        # os.chown may not exist on Windows; we patch with raising=False.
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.chown", chown_mock, raising=False)
        _chown_host_tree(tmp_path, 1000, 1000)
        chown_mock.assert_not_called()

    def test_recurses_over_tree(self, tmp_path, monkeypatch):
        # Build a small tree:  tmp_path/{a.txt, sub/{b.txt, c.txt}}
        (tmp_path / "a.txt").write_text("a")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")
        (sub / "c.txt").write_text("c")

        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")
        chown_mock = MagicMock()
        monkeypatch.setattr("tuner.handlers.local_run_handler.os.chown", chown_mock, raising=False)

        _chown_host_tree(tmp_path, 42, 24)

        # Every entry in the tree must be chown'd to (42, 24).
        called_paths = {str(call.args[0]) for call in chown_mock.call_args_list}
        expected = {
            str(tmp_path),
            str(tmp_path / "a.txt"),
            str(sub),
            str(sub / "b.txt"),
            str(sub / "c.txt"),
        }
        assert expected.issubset(called_paths)
        # And all got the right (uid, gid) tuple.
        for call in chown_mock.call_args_list:
            assert call.args[1] == 42
            assert call.args[2] == 24

    def test_permission_error_swallowed_on_root(self, tmp_path, monkeypatch, capsys):
        # When os.chown on the root raises PermissionError, the function must not raise,
        # and must print a warning to stdout.
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")

        def raising_chown(path, uid, gid):
            raise PermissionError("not permitted")

        monkeypatch.setattr("tuner.handlers.local_run_handler.os.chown", raising_chown, raising=False)
        # Should not raise.
        _chown_host_tree(tmp_path, 1000, 1000)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "chown" in captured.out

    def test_permission_error_swallowed_on_descendant_continues(self, tmp_path, monkeypatch):
        # A PermissionError on a descendant must not stop the walk.
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")

        calls = []

        def partial_chown(path, uid, gid):
            calls.append(str(path))
            # Fail on a.txt only.
            if str(path).endswith("a.txt"):
                raise PermissionError("nope")

        monkeypatch.setattr("tuner.handlers.local_run_handler.os.chown", partial_chown, raising=False)
        _chown_host_tree(tmp_path, 1000, 1000)
        # Both files should have been attempted despite a.txt failing.
        assert any(c.endswith("a.txt") for c in calls)
        assert any(c.endswith("b.txt") for c in calls)

    def test_oserror_swallowed_on_root(self, tmp_path, monkeypatch, capsys):
        # Also covers the OSError branch on the root chown.
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")

        def raising_chown(path, uid, gid):
            raise OSError("filesystem readonly")

        monkeypatch.setattr("tuner.handlers.local_run_handler.os.chown", raising_chown, raising=False)
        _chown_host_tree(tmp_path, 1000, 1000)
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# Smoke: compose wrapper for a minimal auto+bind+linux plan
# ---------------------------------------------------------------------------


class TestBashWrapperSmoke:
    def test_auto_bind_linux_snapshot(self):
        spec = _resolve_user_spec("auto", "bind", 1000, 1000, "linux")
        plan = {
            "command": ["python", "train_sft.py", "--epochs", "1"],
            "workdir": "/workspace/repo/Trainers/sft",
            "container_artifact_path": "/workspace/repo/out/run-1",
            "pip": ["datasets"],
        }
        out = _build_bash_wrapper(plan, spec)
        assert out.startswith("trap ")
        assert "chown -R 1000:1000" in out
        assert "/workspace/repo/out/run-1" in out
        assert "/workspace/repo/Trainers/sft" in out
        assert "/workspace/repo/toolset-training-artifacts" in out
        assert "pip install --upgrade datasets && exec python train_sft.py --epochs 1" in out


# ---------------------------------------------------------------------------
# _validate_tty_field
# ---------------------------------------------------------------------------


class TestValidateTtyField:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, "auto"),
            ("auto", "auto"),
            ("AUTO", "auto"),
            ("  auto  ", "auto"),
            ("always", "always"),
            ("Always", "always"),
            ("never", "never"),
            ("NEVER", "never"),
        ],
    )
    def test_accepts_valid(self, raw, expected):
        assert _validate_tty_field(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",           # empty
            "   ",        # whitespace only
            "yes",        # unknown keyword
            "no",         # unknown keyword
            "true",       # bool-ish string
            "1",          # int-ish
            "on",         # unknown keyword
        ],
    )
    def test_rejects_invalid(self, raw):
        with pytest.raises(LocalRunError):
            _validate_tty_field(raw)


# ---------------------------------------------------------------------------
# _resolve_tty_flags
# ---------------------------------------------------------------------------


class TestResolveTtyFlags:
    @pytest.mark.parametrize(
        "tty_mode,isatty,expected",
        [
            ("always", True,  ["-i", "-t"]),
            ("always", False, ["-i", "-t"]),
            ("never",  True,  []),
            ("never",  False, []),
            ("auto",   True,  ["-i", "-t"]),
            ("auto",   False, []),
        ],
    )
    def test_resolution_matrix(self, tty_mode, isatty, expected):
        assert _resolve_tty_flags(tty_mode, isatty) == expected

    def test_unknown_mode_raises(self):
        # Defensive guard — validator should have caught it first.
        with pytest.raises(LocalRunError):
            _resolve_tty_flags("bogus", True)


# ---------------------------------------------------------------------------
# _validate_bool_field
# ---------------------------------------------------------------------------


class TestValidateBoolField:
    @pytest.mark.parametrize(
        "raw,default,expected",
        [
            (None, False, False),
            (None, True, True),
            (True, False, True),
            (False, True, False),
            (1, False, True),
            (0, True, False),
            ("true", False, True),
            ("FALSE", True, False),
            ("  yes  ", False, True),
            ("no", True, False),
            ("1", False, True),
            ("0", True, False),
            ("on", False, True),
            ("off", True, False),
        ],
    )
    def test_accepts_valid(self, raw, default, expected):
        assert _validate_bool_field(raw, "persist", default) is expected

    @pytest.mark.parametrize(
        "raw",
        [
            "maybe",
            "",
            "   ",
            2,
            -1,
            [],
            {"a": 1},
        ],
    )
    def test_rejects_invalid(self, raw):
        with pytest.raises(LocalRunError):
            _validate_bool_field(raw, "persist", default=False)


# ---------------------------------------------------------------------------
# _pip_marker_hash
# ---------------------------------------------------------------------------


class TestPipMarkerHash:
    def test_empty_returns_empty(self):
        assert _pip_marker_hash([]) == ""
        assert _pip_marker_hash(["", None, ""]) == ""  # filters falsy

    def test_order_independent(self):
        a = _pip_marker_hash(["b", "a", "c"])
        b = _pip_marker_hash(["c", "b", "a"])
        assert a == b
        assert len(a) == 12

    def test_different_sets_hash_differently(self):
        a = _pip_marker_hash(["transformers==5.5.0"])
        b = _pip_marker_hash(["transformers==5.5.1"])
        assert a != b

    def test_hex_chars_only(self):
        result = _pip_marker_hash(["foo", "bar"])
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# _derive_container_name
# ---------------------------------------------------------------------------


class TestDeriveContainerName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("qwen35-2b-sft", "local-run-qwen35-2b-sft"),
            ("qwen35_2b_sft", "local-run-qwen35-2b-sft"),
            ("Qwen3.5-2B-SFT", "local-run-qwen3-5-2b-sft"),
            ("a__b__c", "local-run-a-b-c"),
            ("---weird---name---", "local-run-weird-name"),
            ("local-run-already", "local-run-already"),
            ("LOCAL-RUN-PREFIXED", "local-run-prefixed"),
            ("!!!", "local-run-job"),  # falls back to "job" when slug is empty
        ],
    )
    def test_slug(self, raw, expected):
        assert _derive_container_name(raw) == expected

    def test_idempotent(self):
        first = _derive_container_name("my_job_name")
        second = _derive_container_name(first)
        assert first == second


# ---------------------------------------------------------------------------
# _build_persistent_docker_run_args
# ---------------------------------------------------------------------------


class TestBuildPersistentDockerRunArgs:
    def _base_plan(self, **overrides):
        plan = {
            "persistent_container_name": "local-run-example",
            "stop_timeout": 60,
            "image": "unsloth/unsloth:latest",
            "user_spec": UserSpec("0:0", 1000, 1000, skip_chown=False),
            "mount_hf_cache": True,
            "mount_pip_cache": True,
        }
        plan.update(overrides)
        return plan

    def test_required_flags_present(self):
        args = _build_persistent_docker_run_args(
            self._base_plan(), Path("/repo"), Path("/home/u")
        )
        # Essential shape.
        assert args[:2] == ["docker", "run"]
        assert "-d" in args
        assert "--init" in args
        assert "--name" in args and "local-run-example" in args
        assert "--stop-timeout" in args
        idx = args.index("--stop-timeout")
        assert args[idx + 1] == "60"
        assert "--gpus" in args
        idx = args.index("--gpus")
        assert args[idx + 1] == "all"
        assert "-u" in args
        idx = args.index("-u")
        assert args[idx + 1] == "0:0"
        # Repo bind-mount.
        assert "-v" in args
        assert "/repo:/workspace/repo" in args
        # Entrypoint + sleep infinity tail.
        assert args[-4:] == ["unsloth/unsloth:latest", "-c", "sleep infinity"][-3:] or args[-3:] == [
            "unsloth/unsloth:latest",
            "-c",
            "sleep infinity",
        ]
        # Entrypoint bash must appear just before the image.
        image_idx = args.index("unsloth/unsloth:latest")
        assert args[image_idx - 2:image_idx] == ["--entrypoint", "bash"]

    def test_hf_cache_mount_toggle(self):
        plan = self._base_plan(mount_hf_cache=True, mount_pip_cache=False)
        args = _build_persistent_docker_run_args(plan, Path("/repo"), Path("/home/u"))
        assert "/home/u/.cache/huggingface:/root/.cache/huggingface" in args
        assert not any("pip:" in a for a in args)

    def test_pip_cache_mount_toggle(self):
        plan = self._base_plan(mount_hf_cache=False, mount_pip_cache=True)
        args = _build_persistent_docker_run_args(plan, Path("/repo"), Path("/home/u"))
        assert "/home/u/.cache/pip:/root/.cache/pip" in args
        assert not any("huggingface:" in a for a in args)

    def test_no_cache_mounts(self):
        plan = self._base_plan(mount_hf_cache=False, mount_pip_cache=False)
        args = _build_persistent_docker_run_args(plan, Path("/repo"), Path("/home/u"))
        assert not any("huggingface" in a for a in args)
        assert not any(".cache/pip" in a for a in args)

    def test_user_spec_flag_propagates(self):
        plan = self._base_plan(user_spec=UserSpec("1000:1000", None, None, skip_chown=True))
        args = _build_persistent_docker_run_args(plan, Path("/repo"), Path("/home/u"))
        idx = args.index("-u")
        assert args[idx + 1] == "1000:1000"

    def test_defaults_to_root_when_user_spec_none(self):
        plan = self._base_plan(user_spec=UserSpec(None, None, None, skip_chown=True))
        args = _build_persistent_docker_run_args(plan, Path("/repo"), Path("/home/u"))
        idx = args.index("-u")
        assert args[idx + 1] == "0:0"


# ---------------------------------------------------------------------------
# _container_exists (subprocess-mocked)
# ---------------------------------------------------------------------------


class TestContainerExists:
    def _handler(self):
        from tuner.handlers.local_run_handler import LocalRunHandler

        h = LocalRunHandler()
        return h

    @pytest.mark.parametrize(
        "returncode,stdout,expected",
        [
            (0, "running\n", "running"),
            (0, "RUNNING", "running"),
            (0, "exited\n", "exited"),
            (0, "created\n", "exited"),  # unknown states coerced to exited
            (0, "paused\n", "exited"),
            (1, "", "absent"),
            (125, "no such container", "absent"),
        ],
    )
    def test_inspect_parse(self, returncode, stdout, expected):
        h = self._handler()
        fake = MagicMock()
        fake.returncode = returncode
        fake.stdout = stdout
        with patch.object(h, "_run", return_value=fake) as run_mock:
            assert h._container_exists("some-name") == expected
        # Invoked with docker inspect and the expected format.
        argv = run_mock.call_args[0][0]
        assert argv[:4] == ["docker", "inspect", "--format", "{{.State.Status}}"]
        assert argv[-1] == "some-name"


# ---------------------------------------------------------------------------
# _ensure_persistent_container (state-machine smoke)
# ---------------------------------------------------------------------------


class TestEnsurePersistentContainer:
    def _handler_and_plan(self):
        from tuner.handlers.local_run_handler import LocalRunHandler

        h = LocalRunHandler()
        plan = {
            "persistent_container_name": "local-run-example",
            "stop_timeout": 60,
            "image": "unsloth/unsloth:latest",
            "user_spec": UserSpec("0:0", 1000, 1000, skip_chown=False),
            "mount_hf_cache": False,
            "mount_pip_cache": False,
        }
        return h, plan

    def test_running_short_circuits(self):
        h, plan = self._handler_and_plan()
        with patch.object(h, "_container_exists", return_value="running"), \
             patch.object(h, "_check") as check_mock:
            assert h._ensure_persistent_container(plan) == "reused"
            check_mock.assert_not_called()

    def test_exited_starts(self):
        h, plan = self._handler_and_plan()
        with patch.object(h, "_container_exists", return_value="exited"), \
             patch.object(h, "_check") as check_mock:
            assert h._ensure_persistent_container(plan) == "started"
            check_mock.assert_called_once()
            argv = check_mock.call_args[0][0]
            assert argv[:2] == ["docker", "start"]
            assert argv[-1] == "local-run-example"

    def test_absent_creates(self):
        h, plan = self._handler_and_plan()
        with patch.object(h, "_container_exists", return_value="absent"), \
             patch.object(h, "_check") as check_mock:
            assert h._ensure_persistent_container(plan) == "created"
            check_mock.assert_called_once()
            argv = check_mock.call_args[0][0]
            assert argv[:2] == ["docker", "run"]
            assert "-d" in argv
            assert "--init" in argv
            assert "local-run-example" in argv


# ---------------------------------------------------------------------------
# _cache_mount_args
# ---------------------------------------------------------------------------


class TestCacheMountArgs:
    def test_both_enabled(self):
        plan = {"mount_hf_cache": True, "mount_pip_cache": True}
        args = _cache_mount_args(plan, Path("/home/u"))
        assert "-v" in args
        assert "/home/u/.cache/huggingface:/root/.cache/huggingface" in args
        assert "/home/u/.cache/pip:/root/.cache/pip" in args
        assert args.count("-v") == 2

    def test_hf_only(self):
        plan = {"mount_hf_cache": True, "mount_pip_cache": False}
        args = _cache_mount_args(plan, Path("/home/u"))
        assert args == ["-v", "/home/u/.cache/huggingface:/root/.cache/huggingface"]

    def test_pip_only(self):
        plan = {"mount_hf_cache": False, "mount_pip_cache": True}
        args = _cache_mount_args(plan, Path("/home/u"))
        assert args == ["-v", "/home/u/.cache/pip:/root/.cache/pip"]

    def test_neither(self):
        plan = {"mount_hf_cache": False, "mount_pip_cache": False}
        args = _cache_mount_args(plan, Path("/home/u"))
        assert args == []

    def test_missing_keys_treated_as_false(self):
        # Defensive: _compile always populates both, but be resilient.
        args = _cache_mount_args({}, Path("/home/u"))
        assert args == []


# ---------------------------------------------------------------------------
# _ensure_host_cache_dirs
# ---------------------------------------------------------------------------


class TestEnsureHostCacheDirs:
    def test_creates_both_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")
        plan = {"mount_hf_cache": True, "mount_pip_cache": True}
        _ensure_host_cache_dirs(plan, tmp_path)
        assert (tmp_path / ".cache" / "huggingface").is_dir()
        assert (tmp_path / ".cache" / "pip").is_dir()

    def test_creates_only_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")
        plan = {"mount_hf_cache": True, "mount_pip_cache": False}
        _ensure_host_cache_dirs(plan, tmp_path)
        assert (tmp_path / ".cache" / "huggingface").is_dir()
        assert not (tmp_path / ".cache" / "pip").exists()

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")
        plan = {"mount_hf_cache": True, "mount_pip_cache": True}
        _ensure_host_cache_dirs(plan, tmp_path)
        # Second call must not raise despite dirs already existing.
        _ensure_host_cache_dirs(plan, tmp_path)
        assert (tmp_path / ".cache" / "huggingface").is_dir()

    def test_noop_on_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "win32")
        plan = {"mount_hf_cache": True, "mount_pip_cache": True}
        _ensure_host_cache_dirs(plan, tmp_path)
        assert not (tmp_path / ".cache").exists()

    def test_swallows_oserror(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("tuner.handlers.local_run_handler.sys.platform", "linux")

        class FakePath:
            def __init__(self, p):
                self._p = Path(p)

            def __truediv__(self, other):
                return FakePath(self._p / other)

            def mkdir(self, *args, **kwargs):
                raise OSError("disk full")

        # Patch mkdir on the derived target path to raise.
        import tuner.handlers.local_run_handler as mod

        original_mkdir = Path.mkdir

        def failing_mkdir(self, *args, **kwargs):
            if ".cache/huggingface" in str(self) or ".cache\\huggingface" in str(self):
                raise OSError("disk full")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)
        plan = {"mount_hf_cache": True, "mount_pip_cache": False}
        _ensure_host_cache_dirs(plan, tmp_path)
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# Persistent docker-run args include cache mounts by default via _cache_mount_args
# ---------------------------------------------------------------------------


class TestBuildPersistentDockerRunArgsDefaults:
    def test_both_caches_on_by_default_via_plan(self):
        # Simulates a plan compiled with the new defaults (mounts always-on).
        plan = {
            "persistent_container_name": "local-run-example",
            "stop_timeout": 60,
            "image": "unsloth/unsloth:latest",
            "user_spec": UserSpec("0:0", 1000, 1000, skip_chown=False),
            "mount_hf_cache": True,
            "mount_pip_cache": True,
        }
        args = _build_persistent_docker_run_args(plan, Path("/repo"), Path("/home/u"))
        assert "/home/u/.cache/huggingface:/root/.cache/huggingface" in args
        assert "/home/u/.cache/pip:/root/.cache/pip" in args


# ---------------------------------------------------------------------------
# _execute_bind_mode (ephemeral --rm path) includes cache mounts by default
# ---------------------------------------------------------------------------


class TestEphemeralBindModeCacheMounts:
    def _handler_and_plan(self, monkeypatch, **overrides):
        from tuner.handlers.local_run_handler import LocalRunHandler

        h = LocalRunHandler()
        # Pin HOME so the cache mount path assertion is deterministic.
        monkeypatch.setenv("HOME", "/home/testuser")
        plan = {
            "user_spec": UserSpec("0:0", 1000, 1000, skip_chown=False),
            "tty_mode": "never",
            "stop_timeout": 60,
            "command": ["python", "train.py"],
            "workdir": "/workspace/repo/Trainers/sft",
            "container_artifact_path": "/workspace/repo/out/run-1",
            "pip": [],
            "image": "unsloth/unsloth:latest",
            "mount_hf_cache": True,
            "mount_pip_cache": True,
        }
        plan.update(overrides)
        return h, plan

    def test_defaults_emit_both_cache_mounts(self, monkeypatch):
        h, plan = self._handler_and_plan(monkeypatch)
        with patch.object(h, "_check") as check_mock:
            h._execute_bind_mode(plan)
        argv = check_mock.call_args[0][0]
        assert argv[:2] == ["docker", "run"]
        assert "--rm" in argv
        assert "/home/testuser/.cache/huggingface:/root/.cache/huggingface" in argv
        assert "/home/testuser/.cache/pip:/root/.cache/pip" in argv

    def test_opt_out_removes_cache_mounts(self, monkeypatch):
        h, plan = self._handler_and_plan(
            monkeypatch, mount_hf_cache=False, mount_pip_cache=False
        )
        with patch.object(h, "_check") as check_mock:
            h._execute_bind_mode(plan)
        argv = check_mock.call_args[0][0]
        assert not any("huggingface" in a for a in argv)
        assert not any(".cache/pip" in a for a in argv)

    def test_hf_only(self, monkeypatch):
        h, plan = self._handler_and_plan(
            monkeypatch, mount_hf_cache=True, mount_pip_cache=False
        )
        with patch.object(h, "_check") as check_mock:
            h._execute_bind_mode(plan)
        argv = check_mock.call_args[0][0]
        assert "/home/testuser/.cache/huggingface:/root/.cache/huggingface" in argv
        assert not any(".cache/pip" in a for a in argv)
