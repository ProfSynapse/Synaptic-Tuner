"""Provider-free checks of the actual committed consumer/gitlink topology."""

from pathlib import Path
import subprocess
import sys
import unittest
import json

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "synaptic-tuner"
PIN = "ba65e3c8ff590aa79387ddd07a70c70233a8b90a"
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
        self.assertEqual(self.manifest.project_id, "synaptic-modal-chat-smoke")
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
        self.assertEqual(source.engine_source.branch, "smoke/modal-chat-engine")
        self.assertFalse(source.project_source.dirty)
        self.assertFalse(source.engine_source.dirty)
        self.assertFalse(source.project_source.pushed)
        self.assertFalse(source.engine_source.pushed)

    def test_committed_tree_is_only_the_reviewed_fixture(self):
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                ".gitignore",
                ".gitmodules",
                "AGENTS.md",
                "README.md",
                "configuration/smoke.json",
                "data/smoke.jsonl",
                "synaptic-tuner",
                "synaptic.yaml",
                "tests/test_source_fixture.py",
            ],
        )

    def test_committed_smoke_uses_real_typed_configuration(self):
        from examples.modal_chat.settings import ModalChatSettings

        raw = (ROOT / "configuration/smoke.json").read_bytes()
        settings = ModalChatSettings.parse(raw[:-1] if raw.endswith(b"\n") else raw)
        self.assertEqual(settings.environment_name, "synaptic-smoke-v1")
        self.assertEqual(settings.runtime_environment["PATH"], "/opt/conda/bin:/usr/bin:/bin")
        self.assertEqual(settings.training_input.hyperparameters.duration.max_steps, 2)
        rows = [
            json.loads(line)
            for line in (ROOT / settings.dataset_project_path).read_text().splitlines()
        ]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["messages"] for row in rows))


if __name__ == "__main__":
    unittest.main()
