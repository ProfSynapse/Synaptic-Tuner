import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(".skills/scripts/sync_skill_trees.py").resolve()
    spec = importlib.util.spec_from_file_location("sync_skill_trees_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_skill_tree_adds_missing_and_removes_stale(tmp_path):
    module = _load_module()
    source = tmp_path / "source"
    target = tmp_path / "target"

    canonical = source / "alpha"
    canonical.mkdir(parents=True)
    (canonical / "SKILL.md").write_text("alpha", encoding="utf-8")
    (canonical / "reference").mkdir()
    (canonical / "reference" / "doc.md").write_text("doc", encoding="utf-8")

    stale = target / "stale"
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text("stale", encoding="utf-8")

    module.sync_skill_tree(source, target)

    assert (target / "alpha" / "SKILL.md").read_text(encoding="utf-8") == "alpha"
    assert (target / "alpha" / "reference" / "doc.md").read_text(encoding="utf-8") == "doc"
    assert not (target / "stale").exists()


def test_check_skill_tree_detects_drift(tmp_path):
    module = _load_module()
    source = tmp_path / "source"
    target = tmp_path / "target"

    canonical = source / "alpha"
    canonical.mkdir(parents=True)
    (canonical / "SKILL.md").write_text("alpha", encoding="utf-8")

    drifted = target / "alpha"
    drifted.mkdir(parents=True)
    (drifted / "SKILL.md").write_text("different", encoding="utf-8")

    issues = module.check_skill_tree(source, target)

    assert issues
    assert any("file differs" in issue for issue in issues)
