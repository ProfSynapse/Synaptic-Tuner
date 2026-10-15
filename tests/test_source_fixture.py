"""Provider-free checks of the actual committed consumer/gitlink topology."""

from pathlib import Path
import subprocess
import sys
import unittest
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "synaptic-tuner"
PIN = "91c4bbb2272e91b92f3b7c8169dfe725e0fa4e22"
ORIGIN = "https://github.com/ProfSynapse/Synaptic-Tuner.git"

if ENGINE.resolve(strict=True) != ENGINE:
    raise RuntimeError("engine path is not canonical")
sys.path.insert(0, str(ENGINE))

from tuner.project.git_verification import GitCliLocalSourceInspector
from tuner.project.manifest import load_project_manifest


class ConsumerSourceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_project_manifest(ROOT / "synaptic.yaml")
        self.context = self.manifest.create_context(
            engine_root=ENGINE, invocation_cwd=ROOT
        )

    def test_real_host_context_and_private_output_roots(self):
        self.assertEqual(self.manifest.project_id, "synaptic-hf-sft-fixture")
        self.assertEqual(self.context.mode, "host")
        self.assertEqual(self.context.path_mode, "project_v1")
        self.assertEqual(self.context.engine_root, ENGINE)
        self.assertEqual(self.context.project_root, ROOT)
        self.assertNotEqual(self.context.engine_root, self.context.project_root)
        for path in self.context.writable_roots:
            self.assertTrue(path.is_relative_to(ROOT / ".synaptic"))
            self.assertFalse(path.is_relative_to(ENGINE))

    def test_fail_closed_source_policy(self):
        policy = self.manifest.policies
        self.assertEqual(policy["repository_schemes"], ["https"])
        self.assertEqual(policy["repository_hosts"], ["github.com"])
        self.assertEqual(policy["engine_writes"], "deny")
        self.assertEqual(policy["source_writes"], "deny")
        self.assertEqual(policy["external_paths"], "deny")
        self.assertEqual(policy["dirty_sources"], "deny")
        self.assertIs(policy["nested_submodules"], False)
        self.assertEqual(policy["max_submodule_depth"], 0)

    def test_committed_gitlink_passes_production_local_inspector(self):
        # The production local inspector passes remote_proof=False internally:
        # this checks Git bytes, not credentials, remote refs, or serving authority.
        source = GitCliLocalSourceInspector().inspect(context=self.context)
        self.assertEqual(source.mode, "superproject")
        self.assertEqual(source.project_source.location.canonical_url, ORIGIN)
        self.assertEqual(source.engine_source.location.canonical_url, ORIGIN)
        self.assertEqual(source.engine_source.commit, PIN)
        self.assertEqual(source.engine_source.gitlink_commit, PIN)
        self.assertEqual(source.engine_source.submodule_path, "synaptic-tuner")
        self.assertEqual(source.engine_source.branch, "release/stable-diagnostics-integration")
        self.assertFalse(source.project_source.dirty)
        self.assertFalse(source.engine_source.dirty)
        self.assertFalse(source.project_source.pushed)
        self.assertFalse(source.engine_source.pushed)

    def test_real_v1_publication_and_complete_paired_rows(self):
        from tuner.dataset_prep import snapshot_prepared_dataset_v1
        digest = "7aa26aba079ac1210466c5c2cdf747f00c764b5bc4ed366e2f67af7ffa0d3770"
        publication = ROOT / "data/hf-sft-v1/prepared" / ("dataset-" + digest)
        # Git retains original bytes, not private publication permission modes.
        # Admit an exact test-owned copy without changing historical evidence.
        with tempfile.TemporaryDirectory() as temporary:
            private_publication = Path(temporary) / publication.name
            shutil.copytree(publication, private_publication)
            for member in private_publication.rglob("*"):
                member.chmod(0o700 if member.is_dir() else 0o600)
            private_publication.chmod(0o700)
            verified, raw = snapshot_prepared_dataset_v1(private_publication)
        self.assertEqual(verified.semantic_identity.dataset_digest, digest)
        self.assertEqual(len(raw), 2821)
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        self.assertEqual(len(rows), 8)
        texts = {row["text"] for row in rows}
        sources = list((ROOT / "data/hf-sft-v1/sources").glob("*.md"))
        self.assertEqual(len(sources), 8)
        self.assertEqual(texts, {p.read_text() for p in sources})
        self.assertEqual(sum(row["split"] == "train" for row in rows), 6)
        self.assertEqual(sum(row["split"] == "validation" for row in rows), 2)
        config = json.loads((ROOT / "configuration/hf-sft-v1/sft-config.json").read_bytes())
        self.assertEqual(config["dataset"]["revision"], digest)
        self.assertEqual(config["dataset"]["format"], "syntunia-sft-row/v1")


if __name__ == "__main__":
    unittest.main()
