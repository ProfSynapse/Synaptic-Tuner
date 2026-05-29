"""Tests for SynthChat prompt optimization overlay hook."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from SynthChat.modes.generate import (
    _PromptOptimizationResultWriter,
    _apply_prompt_overlays,
    _prepare_prompt_optimization,
)


class TestPromptOptimizationOverlays:
    def test_applies_scenario_overlay_in_memory(self, tmp_path):
        scenarios_dir = tmp_path / "SynthChat" / "scenarios"
        scenarios_dir.mkdir(parents=True)
        source = scenarios_dir / "demo.yaml"
        source.write_text("scenarios: {}\n", encoding="utf-8")
        generator = SimpleNamespace(
            scenario_loader=SimpleNamespace(
                scenarios={"demo": {"prompts": {"assistant": "old"}}}
            ),
            _tool_call_formats={},
            _workspace_formats={},
            _label_mappings={},
        )
        overlays = {
            "subjects": {
                "demo_assistant": {
                    "source_path": str(source),
                    "dotted_path": "scenarios.demo.prompts.assistant",
                    "optimized_prompt": "new",
                }
            }
        }

        applied = _apply_prompt_overlays(
            overlays=overlays,
            generator=generator,
            scenarios_dir=scenarios_dir,
            config_dir=tmp_path / "SynthChat" / "config",
        )

        assert applied == ["demo_assistant"]
        assert generator.scenario_loader.scenarios["demo"]["prompts"]["assistant"] == "new"

    def test_applies_tool_call_format_overlay_as_lines(self, tmp_path):
        config_dir = tmp_path / "SynthChat" / "config"
        config_dir.mkdir(parents=True)
        source = config_dir / "tool_call_formats.yaml"
        source.write_text("formats: {}\n", encoding="utf-8")
        generator = SimpleNamespace(
            scenario_loader=SimpleNamespace(scenarios={}),
            _tool_call_formats={"default": {"generation_instructions": ["old"]}},
            _workspace_formats={},
            _label_mappings={},
        )
        overlays = {
            "subjects": {
                "default_instructions": {
                    "source_path": str(source),
                    "dotted_path": "formats.default.generation_instructions",
                    "optimized_prompt": "first\nsecond",
                }
            }
        }

        applied = _apply_prompt_overlays(
            overlays=overlays,
            generator=generator,
            scenarios_dir=tmp_path / "SynthChat" / "scenarios",
            config_dir=config_dir,
        )

        assert applied == ["default_instructions"]
        assert generator._tool_call_formats["default"]["generation_instructions"] == [
            "first",
            "second",
        ]

    def test_applies_overlay_using_repo_relative_source_path(self, tmp_path):
        repo_root = tmp_path
        (repo_root / "tuner.py").write_text("", encoding="utf-8")
        config_dir = repo_root / "SynthChat" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "tool_call_formats.yaml").write_text("formats: {}\n", encoding="utf-8")
        generator = SimpleNamespace(
            scenario_loader=SimpleNamespace(scenarios={}),
            _tool_call_formats={"default": {"generation_instructions": ["old"]}},
            _workspace_formats={},
            _label_mappings={},
        )
        overlays = {
            "subjects": {
                "default_instructions": {
                    "source_path": "tool_call_formats.yaml",
                    "source_path_repo_relative": "SynthChat/config/tool_call_formats.yaml",
                    "dotted_path": "formats.default.generation_instructions",
                    "optimized_prompt": "new",
                }
            }
        }

        applied = _apply_prompt_overlays(
            overlays=overlays,
            generator=generator,
            scenarios_dir=repo_root / "SynthChat" / "scenarios",
            config_dir=config_dir,
        )

        assert applied == ["default_instructions"]
        assert generator._tool_call_formats["default"]["generation_instructions"] == ["new"]

    def test_prepare_fails_when_no_overlay_subject_applies(self, tmp_path):
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "overlays.json").write_text(
            """
{
  "subjects": {
    "missing": {
      "source_path": "missing.yaml",
      "dotted_path": "formats.default.generation_instructions",
      "optimized_prompt": "new"
    }
  }
}
""".lstrip(),
            encoding="utf-8",
        )
        generator = SimpleNamespace(
            scenario_loader=SimpleNamespace(scenarios={}),
            _tool_call_formats={},
            _workspace_formats={},
            _label_mappings={},
        )
        args = SimpleNamespace(prompt_opt_config=None, prompt_opt_artifact=str(artifact))

        with pytest.raises(ValueError, match="none of the overlay subjects"):
            _prepare_prompt_optimization(
                args,
                generator=generator,
                scenarios_dir=tmp_path / "SynthChat" / "scenarios",
                config_dir=tmp_path / "SynthChat" / "config",
            )


class TestPromptOptimizationResultWriter:
    def test_attaches_prompt_optimization_metadata_before_write(self):
        captured = []

        def write(result):
            captured.append(result.example)
            return True

        result = SimpleNamespace(example={"metadata": {"scenario": "demo"}})
        metadata = {"artifact_path": "artifact", "selected_candidate_id": "candidate"}
        writer = _PromptOptimizationResultWriter(write, metadata)

        assert writer.write(result) is True
        assert captured[0]["metadata"]["prompt_optimization"] == metadata
