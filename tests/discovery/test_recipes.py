"""Tests for the unified training recipe discovery module.

Covers tuner.discovery.recipes (list_recipes, load_recipe, RecipeMeta) and
the handler integration that points local_run_handler / cloud_run_handler at
Trainers/recipes/. Pure-function discovery is tested in isolation against
synthetic recipes under tmp_path; handler integration is verified at the
_jobs_dir() level (full handler invocation pulls in Docker/HF deps).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import pytest

from tuner.discovery.recipes import (
    RECIPE_DIRNAME,
    RecipeMeta,
    list_recipes,
    load_recipe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_repo(tmp_path: Path) -> Path:
    """Create a synthetic repo root with an empty Trainers/recipes/ dir."""
    (tmp_path / RECIPE_DIRNAME).mkdir(parents=True)
    return tmp_path


def _write_recipe(repo_root: Path, filename: str, body: str) -> Path:
    path = repo_root / RECIPE_DIRNAME / filename
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# list_recipes — basic discovery + filtering
# ---------------------------------------------------------------------------


class TestListRecipesBasic:
    def test_returns_recipemeta_for_each_valid_yaml(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _write_recipe(
            repo,
            "alpha.yaml",
            """
            name: alpha
            description: First recipe
            target: local
            method: sft
            """,
        )
        _write_recipe(
            repo,
            "beta.yaml",
            """
            name: beta
            description: Second recipe
            target: cloud
            method: kto
            """,
        )

        results = list_recipes(repo)

        assert len(results) == 2
        assert all(isinstance(r, RecipeMeta) for r in results)
        names = {r.name for r in results}
        assert names == {"alpha", "beta"}

    def test_results_sorted_by_path(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        for filename in ("zebra.yaml", "alpha.yaml", "mango.yaml"):
            _write_recipe(
                repo,
                filename,
                f"""
                name: {filename}
                target: local
                method: sft
                """,
            )

        results = list_recipes(repo)
        paths = [r.path.name for r in results]
        assert paths == sorted(paths)

    def test_recipemeta_fields_populated(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "single.yaml",
            """
            name: single-recipe
            description: A single test recipe
            target: cloud
            method: grpo
            """,
        )

        [meta] = list_recipes(repo)
        assert meta.path == path
        assert meta.name == "single-recipe"
        assert meta.description == "A single test recipe"
        assert meta.target == "cloud"
        assert meta.method == "grpo"

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        # No Trainers/recipes/ subdirectory at all.
        assert list_recipes(tmp_path) == []

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert list_recipes(repo) == []


# ---------------------------------------------------------------------------
# list_recipes — filtering
# ---------------------------------------------------------------------------


class TestListRecipesFilters:
    @pytest.fixture
    def populated_repo(self, tmp_path: Path) -> Path:
        repo = _make_repo(tmp_path)
        _write_recipe(
            repo,
            "local_sft.yaml",
            """
            name: local-sft
            target: local
            method: sft
            """,
        )
        _write_recipe(
            repo,
            "local_kto.yaml",
            """
            name: local-kto
            target: local
            method: kto
            """,
        )
        _write_recipe(
            repo,
            "cloud_sft.yaml",
            """
            name: cloud-sft
            target: cloud
            method: sft
            """,
        )
        _write_recipe(
            repo,
            "both_sft.yaml",
            """
            name: both-sft
            target: both
            method: sft
            """,
        )
        return repo

    def test_filter_target_local_includes_both(self, populated_repo: Path) -> None:
        results = list_recipes(populated_repo, target="local")
        names = {r.name for r in results}
        # target=both should match either local or cloud filter.
        assert names == {"local-sft", "local-kto", "both-sft"}

    def test_filter_target_cloud_includes_both(self, populated_repo: Path) -> None:
        results = list_recipes(populated_repo, target="cloud")
        names = {r.name for r in results}
        assert names == {"cloud-sft", "both-sft"}

    def test_filter_method_sft(self, populated_repo: Path) -> None:
        results = list_recipes(populated_repo, method="sft")
        names = {r.name for r in results}
        assert names == {"local-sft", "cloud-sft", "both-sft"}

    def test_filter_method_kto(self, populated_repo: Path) -> None:
        results = list_recipes(populated_repo, method="kto")
        names = {r.name for r in results}
        assert names == {"local-kto"}

    def test_filter_target_and_method_combined(self, populated_repo: Path) -> None:
        results = list_recipes(populated_repo, target="local", method="sft")
        names = {r.name for r in results}
        # local-sft and both-sft (target=both also matches local)
        assert names == {"local-sft", "both-sft"}

    def test_filter_target_case_insensitive(self, populated_repo: Path) -> None:
        results = list_recipes(populated_repo, target="LOCAL")
        names = {r.name for r in results}
        assert "local-sft" in names

    def test_filter_no_matches_returns_empty(self, populated_repo: Path) -> None:
        assert list_recipes(populated_repo, method="nonexistent") == []


# ---------------------------------------------------------------------------
# list_recipes — robustness against malformed/incomplete files
# ---------------------------------------------------------------------------


class TestListRecipesRobustness:
    def test_missing_target_field_skipped_silently(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _write_recipe(
            repo,
            "no_target.yaml",
            """
            name: no-target
            method: sft
            """,
        )
        _write_recipe(
            repo,
            "valid.yaml",
            """
            name: valid
            target: local
            method: sft
            """,
        )

        results = list_recipes(repo)
        names = {r.name for r in results}
        assert names == {"valid"}

    def test_malformed_yaml_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        bad = repo / RECIPE_DIRNAME / "bad.yaml"
        bad.write_text("name: broken\n  - unbalanced: [\n", encoding="utf-8")
        _write_recipe(
            repo,
            "good.yaml",
            """
            name: good
            target: local
            method: sft
            """,
        )

        results = list_recipes(repo)
        names = {r.name for r in results}
        assert names == {"good"}

    def test_non_dict_top_level_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        # YAML list at top level — not a recipe.
        (repo / RECIPE_DIRNAME / "list.yaml").write_text(
            "- one\n- two\n", encoding="utf-8"
        )
        _write_recipe(
            repo,
            "good.yaml",
            """
            name: good
            target: local
            method: sft
            """,
        )

        results = list_recipes(repo)
        names = {r.name for r in results}
        assert names == {"good"}

    def test_non_yaml_files_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        (repo / RECIPE_DIRNAME / "README.md").write_text("# Recipes\n", encoding="utf-8")
        (repo / RECIPE_DIRNAME / "notes.txt").write_text("notes", encoding="utf-8")
        _write_recipe(
            repo,
            "valid.yaml",
            """
            name: valid
            target: local
            method: sft
            """,
        )

        results = list_recipes(repo)
        assert [r.name for r in results] == ["valid"]

    def test_subdirectories_not_recursed(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        nested = repo / RECIPE_DIRNAME / "nested"
        nested.mkdir()
        (nested / "buried.yaml").write_text(
            "name: buried\ntarget: local\nmethod: sft\n", encoding="utf-8"
        )
        _write_recipe(
            repo,
            "top.yaml",
            """
            name: top
            target: local
            method: sft
            """,
        )

        results = list_recipes(repo)
        assert {r.name for r in results} == {"top"}

    def test_name_falls_back_to_stem(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _write_recipe(
            repo,
            "fallback_name.yaml",
            """
            target: local
            method: sft
            """,
        )

        [meta] = list_recipes(repo)
        assert meta.name == "fallback_name"


# ---------------------------------------------------------------------------
# load_recipe — passthrough + deep merge
# ---------------------------------------------------------------------------


class TestLoadRecipe:
    def test_target_local_returns_data_unchanged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "local.yaml",
            """
            name: local-only
            target: local
            method: sft
            training:
              batch_size: 4
              learning_rate: 0.001
            """,
        )

        loaded = load_recipe(path, runner="local")
        assert loaded["name"] == "local-only"
        assert loaded["target"] == "local"
        assert loaded["training"] == {"batch_size": 4, "learning_rate": 0.001}

    def test_target_cloud_returns_data_unchanged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "cloud.yaml",
            """
            name: cloud-only
            target: cloud
            method: sft
            run:
              steps:
                - "echo hello"
            """,
        )

        loaded = load_recipe(path, runner="cloud")
        assert loaded["name"] == "cloud-only"
        assert loaded["run"]["steps"] == ["echo hello"]

    def test_target_both_local_runner_merges_local_subblock(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "both.yaml",
            """
            name: shared
            target: both
            method: sft
            training:
              epochs: 2
              batch_size: 1
            local:
              training:
                batch_size: 4
              job:
                image: unsloth/unsloth:latest
            cloud:
              training:
                batch_size: 16
              job:
                flavor: a10g-small
            """,
        )

        loaded = load_recipe(path, runner="local")
        # Sub-block wins on conflict (batch_size 1 -> 4); shared values
        # preserved (epochs); local-only blocks present.
        assert loaded["training"]["batch_size"] == 4
        assert loaded["training"]["epochs"] == 2
        assert loaded["job"] == {"image": "unsloth/unsloth:latest"}
        # Both sub-blocks stripped before return.
        assert "local" not in loaded
        assert "cloud" not in loaded

    def test_target_both_cloud_runner_merges_cloud_subblock(
        self, tmp_path: Path
    ) -> None:
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "both.yaml",
            """
            name: shared
            target: both
            method: sft
            training:
              epochs: 2
              batch_size: 1
            local:
              training:
                batch_size: 4
            cloud:
              training:
                batch_size: 16
              job:
                flavor: a10g-small
            """,
        )

        loaded = load_recipe(path, runner="cloud")
        assert loaded["training"]["batch_size"] == 16
        assert loaded["training"]["epochs"] == 2
        assert loaded["job"] == {"flavor": "a10g-small"}
        assert "local" not in loaded
        assert "cloud" not in loaded

    def test_target_both_with_no_runner_subblock(self, tmp_path: Path) -> None:
        """target=both but the runner sub-block is missing; still strips both."""
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "both_partial.yaml",
            """
            name: partial
            target: both
            method: sft
            training:
              batch_size: 8
            cloud:
              training:
                batch_size: 32
            """,
        )

        loaded = load_recipe(path, runner="local")
        # No local sub-block; original training value preserved; cloud stripped.
        assert loaded["training"]["batch_size"] == 8
        assert "local" not in loaded
        assert "cloud" not in loaded

    def test_deep_merge_lists_replaced_not_concatenated(self, tmp_path: Path) -> None:
        """Lists are replaced, not extended."""
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "lists.yaml",
            """
            name: list-merge
            target: both
            method: datagen
            run:
              steps:
                - "step-a"
                - "step-b"
            cloud:
              run:
                steps:
                  - "cloud-step"
            """,
        )

        loaded = load_recipe(path, runner="cloud")
        assert loaded["run"]["steps"] == ["cloud-step"]

    def test_deep_merge_three_levels_deep(self, tmp_path: Path) -> None:
        """Nested dicts merge key-wise at every level."""
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "nested.yaml",
            """
            name: deep
            target: both
            method: sft
            artifacts:
              storage:
                bucket: shared-bucket
                prefix: runs/
            local:
              artifacts:
                storage:
                  prefix: local-runs/
            """,
        )

        loaded = load_recipe(path, runner="local")
        assert loaded["artifacts"]["storage"]["bucket"] == "shared-bucket"
        assert loaded["artifacts"]["storage"]["prefix"] == "local-runs/"

    def test_invalid_runner_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        path = _write_recipe(
            repo,
            "any.yaml",
            """
            name: any
            target: local
            method: sft
            """,
        )

        with pytest.raises(ValueError, match="runner must be 'local' or 'cloud'"):
            load_recipe(path, runner="bogus")

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        path = repo / RECIPE_DIRNAME / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a YAML object"):
            load_recipe(path, runner="local")

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """Empty YAML parses to None; loader normalizes to empty dict.

        Documents the loader's lenient contract: an empty file is not a
        recipe but does not crash the loader. list_recipes() will skip it
        (no target field), and a handler that calls load_recipe() directly
        on an empty file gets back {} rather than an exception.
        """
        repo = _make_repo(tmp_path)
        path = repo / RECIPE_DIRNAME / "empty.yaml"
        path.write_text("", encoding="utf-8")

        loaded = load_recipe(path, runner="local")
        assert loaded == {}


# ---------------------------------------------------------------------------
# Handler integration — surface-level path correctness
# ---------------------------------------------------------------------------


class TestHandlerJobsDir:
    """Verify both run handlers point _jobs_dir() at Trainers/recipes/.

    Per scope decision: surface-level check only — full handler invocation
    pulls in Docker/HF Jobs deps. We instantiate the handler with a stub
    Namespace and inspect _jobs_dir() / load_recipe import.
    """

    def test_local_run_handler_jobs_dir_points_at_recipes(self) -> None:
        from argparse import Namespace

        from tuner.handlers.local_run_handler import LocalRunHandler

        handler = LocalRunHandler(args=Namespace())
        jobs_dir = handler._jobs_dir()
        assert jobs_dir.name == "recipes"
        assert jobs_dir.parent.name == "Trainers"
        # And it does NOT point at the legacy Trainers/local/jobs/.
        assert "local/jobs" not in jobs_dir.as_posix()
        assert "local\\jobs" not in str(jobs_dir)

    def test_cloud_run_handler_jobs_dir_points_at_recipes(self) -> None:
        from argparse import Namespace

        from tuner.handlers.cloud_run_handler import CloudRunHandler

        handler = CloudRunHandler(args=Namespace())
        jobs_dir = handler._jobs_dir()
        assert jobs_dir.name == "recipes"
        assert jobs_dir.parent.name == "Trainers"
        assert "cloud/jobs" not in jobs_dir.as_posix()
        assert "cloud\\jobs" not in str(jobs_dir)

    def test_local_run_handler_uses_load_recipe(self) -> None:
        """local_run_handler module imports load_recipe from the discovery layer."""
        from tuner.handlers import local_run_handler

        assert hasattr(local_run_handler, "load_recipe")
        # Sanity: it's the discovery module's symbol, not a local re-bind.
        from tuner.discovery.recipes import load_recipe as discovery_load

        assert local_run_handler.load_recipe is discovery_load

    def test_cloud_run_handler_uses_load_recipe(self) -> None:
        from tuner.handlers import cloud_run_handler

        assert hasattr(cloud_run_handler, "load_recipe")
        from tuner.discovery.recipes import load_recipe as discovery_load

        assert cloud_run_handler.load_recipe is discovery_load


# ---------------------------------------------------------------------------
# Reference completeness — no stale Trainers/local/jobs/ or Trainers/cloud/jobs/
# ---------------------------------------------------------------------------


REFERENCE_GREP_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / ".skills",
    REPO_ROOT / ".agents" / "skills",
    REPO_ROOT / ".claude" / "skills",
    REPO_ROOT / "docs",
    REPO_ROOT / "tests",
    REPO_ROOT / "tuner",
    REPO_ROOT / "Trainers",
    REPO_ROOT / "Evaluator",
    REPO_ROOT / "shared",
    REPO_ROOT / "SynthChat",
)

REFERENCE_GREP_FILES: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "AGENTS.md",
)

# docs/plans/ may legitimately reference legacy paths in the migration plan.
REFERENCE_EXCLUDE_DIRS: tuple[str, ...] = (
    "docs/plans",
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
)


def _git_grep(needle: str, paths: Iterable[Path]) -> list[str]:
    """Run `git grep -nF -- needle paths…` from REPO_ROOT.

    Returns hit lines; missing paths are silently skipped. Falls back to
    plain string scan if git grep is unavailable.
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        return []
    try:
        result = subprocess.run(
            ["git", "grep", "-nF", "--", needle, *[str(p) for p in existing]],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _fallback_grep(needle, existing)

    if result.returncode == 1:
        # No matches.
        return []
    if result.returncode != 0:
        # Some other failure (e.g. not in a git tree). Fall back.
        return _fallback_grep(needle, existing)
    return [line for line in result.stdout.splitlines() if line]


def _fallback_grep(needle: str, paths: Iterable[Path]) -> list[str]:
    hits: list[str] = []
    for root in paths:
        if root.is_file():
            files: Iterable[Path] = [root]
        else:
            files = root.rglob("*")
        for path in files:
            if not path.is_file():
                continue
            posix = path.as_posix()
            if any(excl in posix for excl in REFERENCE_EXCLUDE_DIRS):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    hits.append(f"{posix}:{lineno}:{line}")
    return hits


def _filter_excluded(lines: Iterable[str]) -> list[str]:
    return [
        line
        for line in lines
        if not any(excl in line.split(":", 1)[0] for excl in REFERENCE_EXCLUDE_DIRS)
    ]


class TestReferenceCompleteness:
    """Verify no live references to the legacy job-config paths remain.

    Excludes docs/plans/ (the migration plan documents the rename) and
    .git/. All other source/test/docs/skill paths must be clean.
    """

    def test_no_local_jobs_references(self) -> None:
        hits = _filter_excluded(
            _git_grep("Trainers/local/jobs", [*REFERENCE_GREP_ROOTS, *REFERENCE_GREP_FILES])
        )
        assert hits == [], (
            "Stale references to Trainers/local/jobs/ found:\n"
            + "\n".join(hits[:20])
        )

    def test_no_cloud_jobs_references(self) -> None:
        hits = _filter_excluded(
            _git_grep("Trainers/cloud/jobs", [*REFERENCE_GREP_ROOTS, *REFERENCE_GREP_FILES])
        )
        assert hits == [], (
            "Stale references to Trainers/cloud/jobs/ found:\n"
            + "\n".join(hits[:20])
        )
