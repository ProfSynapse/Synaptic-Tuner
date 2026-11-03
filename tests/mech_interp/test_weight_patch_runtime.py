"""CPU tests for the loader/config/lineage wiring of ``surface.weight_patch``,
the dose-calibrate revision fix, pipeline module-form commands and CLI surface."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from MechInterp import cell as cell_mod
from MechInterp import cli as mi_cli
from MechInterp.config import (
    DoseCalibrationConfig,
    ExtractConfig,
    SteerCellConfig,
    load_dose_calibration_config,
    load_extract_config,
    load_steer_config,
)
from MechInterp.pipeline import (
    PipelineConfig,
    _env_for_stage,
    build_pipeline_plan,
)
from tests.mech_interp.test_batched_generation import _build_tiny_model, _build_tiny_tokenizer
from tests.mech_interp.weight_patch_fixtures import (
    OTHER_REVISION,
    REVISION,
    file_sha,
    make_spec,
    write_checkpoint,
)

# Golden config shas computed from the BASE commit (11e8da19) before weight_patch
# existed; an absent patch must keep hashing exactly the same.
GOLDEN_STEER_SHA = "cdb718c556dfa0f765683d05d1d4bd0b9f71ae141a45be312f9c765853a0f329"
GOLDEN_DOSE_SHA = "b042dd7cefa36c27e154b973c68b0fe57f497e58aafd475fc99e47f6183d77dc"


def render_row(row: dict) -> str:  # referenced by "module:callable" specs
    return row["prompt"]


def _steer_cfg(**surface_extra):
    return SteerCellConfig(
        surface={
            "rows_path": "rows.jsonl",
            "generation": {"max_new_tokens": 8},
            "seed": 3,
            **surface_extra,
        },
        readouts=[{"name": "axis", "path": "axis.json"}],
        law={"kind": "erase_write", "readout": "axis", "position": "final"},
        arms=[{"name": "baseline", "strength": 0.0}, {"name": "dose", "strength": 1.5}],
        execution={"output_path": "out.jsonl", "render_fn": "m:r"},
    )


def _dose_cfg(**surface_extra):
    return DoseCalibrationConfig(
        surface={"rows_path": "rows.jsonl", **surface_extra},
        readouts=[{"name": "axis", "path": "axis.json"}],
        law={"kind": "additive", "readout": "axis"},
        calibration={"doses": [0.0, 1.0]},
        execution={"output_path": "o.jsonl", "summary_path": "s.json"},
    )


# --------------------------------------------------------------------------
# Config + fingerprint
# --------------------------------------------------------------------------


def test_absent_weight_patch_keeps_historical_config_shas():
    assert cell_mod.compute_config_sha(_steer_cfg()) == GOLDEN_STEER_SHA
    assert cell_mod.compute_config_sha(_dose_cfg()) == GOLDEN_DOSE_SHA


def test_weight_patch_changes_config_sha_and_pins_content():
    ref_a = {"path": "p.yaml", "sha256": "a" * 64}
    ref_b = {"path": "p.yaml", "sha256": "b" * 64}
    sha_none = cell_mod.compute_config_sha(_steer_cfg())
    sha_a = cell_mod.compute_config_sha(_steer_cfg(weight_patch=ref_a))
    sha_b = cell_mod.compute_config_sha(_steer_cfg(weight_patch=ref_b))
    assert len({sha_none, sha_a, sha_b}) == 3
    assert cell_mod.compute_config_sha(_dose_cfg(weight_patch=ref_a)) != GOLDEN_DOSE_SHA


@pytest.mark.parametrize(
    "bad",
    [{"path": "p.yaml"}, {"path": "p.yaml", "sha256": "short"}, {"path": "p.yaml", "sha256": "A" * 64},
     {"path": "p.yaml", "sha256": "a" * 64, "extra": 1}],
)
def test_weight_patch_ref_requires_exact_sha(bad):
    with pytest.raises(Exception):
        _steer_cfg(weight_patch=bad)


def test_extract_config_capture_sites_validation():
    base = dict(rows_path="r.jsonl", output_dir="out", render_fn="m:r", content_end_fn="m:c")
    cfg = ExtractConfig(**base, capture_sites=[{"layer": 20, "module": "mlp.down_proj", "tensor": "input"}])
    assert cfg.capture_sites[0].layer == 20
    assert ExtractConfig(**base).capture_sites is None and ExtractConfig(**base).weight_patch is None
    for bad in (
        [],
        [{"layer": -1, "module": "mlp.down_proj", "tensor": "input"}],
        [{"layer": 1, "module": "mlp..down", "tensor": "input"}],
        [{"layer": 1, "module": "mlp.down_proj", "tensor": "weights"}],
        [{"layer": 1, "module": "mlp.down_proj", "tensor": "input", "x": 1}],
        [{"layer": 1, "module": "m", "tensor": "input"}] * 2,
    ):
        with pytest.raises(Exception):
            ExtractConfig(**base, capture_sites=bad)


def test_loaders_accept_and_resolve_weight_patch(tmp_path):
    p = tmp_path / "steer.yaml"
    data = _steer_cfg(weight_patch={"path": "patch.yaml", "sha256": "c" * 64}).model_dump(mode="json")
    p.write_text(yaml.safe_dump(data))
    assert load_steer_config(p).surface.weight_patch.sha256 == "c" * 64
    d = tmp_path / "dose.yaml"
    d.write_text(yaml.safe_dump(_dose_cfg(weight_patch={"path": "patch.yaml", "sha256": "c" * 64}).model_dump(mode="json")))
    assert load_dose_calibration_config(d).surface.weight_patch.path == "patch.yaml"
    e = tmp_path / "ex.yaml"
    e.write_text(yaml.safe_dump({
        "rows_path": "r.jsonl", "output_dir": "o", "render_fn": "m:r", "content_end_fn": "m:c",
        "weight_patch": {"path": "patch.yaml", "sha256": "c" * 64},
        "capture_sites": [{"layer": 3, "module": "mlp.down_proj", "tensor": "input"}],
    }))
    cfg = load_extract_config(e)
    assert cfg.weight_patch.sha256 == "c" * 64 and cfg.capture_sites[0].module == "mlp.down_proj"


# --------------------------------------------------------------------------
# Smoke state + resume lineage
# --------------------------------------------------------------------------


def test_smoke_state_binds_patch_digest_and_legacy_states_still_match(tmp_path):
    out = tmp_path / "o.jsonl"
    cell_mod.record_smoke(out, "sha1", {"passed": True})
    assert "weight_patch_digest" not in json.loads(cell_mod.smoke_state_path(out).read_text())
    assert cell_mod.smoke_passed(out, "sha1")  # unchanged legacy behavior
    assert not cell_mod.smoke_passed(out, "sha1", "d" * 64)  # a patched run needs its own smoke
    cell_mod.record_smoke(out, "sha1", {"passed": True}, patch_digest="d" * 64)
    assert cell_mod.smoke_passed(out, "sha1", "d" * 64)
    assert not cell_mod.smoke_passed(out, "sha1", "e" * 64)
    assert not cell_mod.smoke_passed(out, "sha1")  # patch removed -> smoke not reusable
    assert not cell_mod.smoke_passed(out, "sha2", "d" * 64)


def _receipt(digest="d" * 64):
    return {"patch_digest": digest, "spec_sha256": "1" * 64, "payload_sha256": "2" * 64, "revision": REVISION}


def test_lineage_creates_then_enforces_and_never_rewrites(tmp_path):
    out = tmp_path / "rows.jsonl"
    cell_mod.enforce_weight_patch_lineage(out, None)  # no patch, no file: no-op
    assert not cell_mod.weight_patch_lineage_path(out).exists()

    cell_mod.enforce_weight_patch_lineage(out, _receipt())
    lineage = cell_mod.weight_patch_lineage_path(out)
    assert json.loads(lineage.read_text())["patch_digest"] == "d" * 64
    cell_mod.enforce_weight_patch_lineage(out, _receipt())  # same digest resumes
    with pytest.raises(cell_mod.WeightPatchLineageError):
        cell_mod.enforce_weight_patch_lineage(out, _receipt("e" * 64))  # patch changed
    with pytest.raises(cell_mod.WeightPatchLineageError):
        cell_mod.enforce_weight_patch_lineage(out, None)  # patch removed
    assert json.loads(lineage.read_text())["patch_digest"] == "d" * 64


def test_lineage_refuses_patch_over_pre_existing_unbound_rows(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text('{"row_key": "a"}\n')
    with pytest.raises(cell_mod.WeightPatchLineageError):
        cell_mod.enforce_weight_patch_lineage(out, _receipt())
    cell_mod.enforce_weight_patch_lineage(out, None)  # historical unpatched resume unchanged


def test_corrupt_lineage_record_refuses(tmp_path):
    out = tmp_path / "rows.jsonl"
    cell_mod.weight_patch_lineage_path(out).write_text("{not json")
    with pytest.raises(cell_mod.WeightPatchLineageError):
        cell_mod.enforce_weight_patch_lineage(out, _receipt())


# --------------------------------------------------------------------------
# Loader hook ordering (real _load_model_and_tokenizer, fake transformers)
# --------------------------------------------------------------------------


def test_post_base_load_runs_after_base_load_and_before_adapter(monkeypatch):
    import transformers

    events = []

    class FakeConfig:
        pass

    class FakeModel(torch.nn.Module):
        def eval(self):
            events.append("eval")
            return self

    class FakeAuto:
        _model_mapping = {FakeConfig: FakeModel}
        @staticmethod
        def from_pretrained(name, **kw):
            events.append(("model", kw.get("revision")))
            return FakeModel()

    class FakeTok:
        pad_token = "p"
        eos_token = "e"

        @staticmethod
        def from_pretrained(name, **kw):
            return FakeTok()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAuto)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: FakeConfig())
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTok)
    model, _ = mi_cli._load_model_and_tokenizer(
        "m", None, REVISION, post_base_load=lambda m: events.append(("hook", type(m).__name__))
    )
    assert events == [("model", REVISION), ("hook", "FakeModel"), "eval"]
    events.clear()
    mi_cli._load_model_and_tokenizer("m", None, None)  # no hook: unchanged path
    assert events == [("model", None), "eval"]


# --------------------------------------------------------------------------
# End-to-end steer with a tiny real GPT-2 (patch applied before generation)
# --------------------------------------------------------------------------

TARGET = "transformer.h.0.mlp.c_proj.weight"


def _write_rows_and_axis(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    rows = [{"row_key": f"r{i}", "prompt": p} for i, p in enumerate(
        ["w1 w2 w3", "w4 w5 w6 w7", "w9 w10"]
    )]
    (tmp / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = torch.Generator().manual_seed(2)
    v = torch.randn(32, generator=g)
    v = (v / v.norm()).tolist()
    (tmp / "axis.json").write_text(json.dumps({"layer": 0, "hidden_dim": 32, "vector": v, "sigma": 1.0}))


def _patch_spec(tmp: Path, scale: float, name: str = "patch") -> tuple[Path, str]:
    model = _build_tiny_model()
    w = dict(model.named_parameters())[TARGET].detach()
    g = torch.Generator().manual_seed(11)
    spec = make_spec(
        tmp,
        {TARGET: w},
        [{"id": "edit", "tensor": TARGET, "op": "add_rank_one", "scale": scale, "arrays": {"u": "u", "v": "v"}}],
        {"u": torch.randn(w.shape[0], generator=g), "v": torch.randn(w.shape[1], generator=g)},
        budget={"allowed_tensors": ["transformer.h.*"], "max_relative_frobenius": 1000.0},
        name=name,
    )
    return spec, file_sha(spec)


def _write_steer_config(tmp: Path, out_name: str, patch=None) -> Path:
    surface = {
        "rows_path": str(tmp / "rows.jsonl"),
        "generation": {"max_new_tokens": 5, "do_sample": False},
    }
    if patch is not None:
        surface["weight_patch"] = {"path": str(patch[0]), "sha256": patch[1]}
    cfg = {
        "surface": surface,
        "readouts": [{"name": "axis", "path": str(tmp / "axis.json")}],
        "law": {"kind": "additive", "readout": "axis", "position": "anchor"},
        "arms": [{"name": "baseline", "strength": 0.0}, {"name": "dose", "strength": 1.0}],
        "execution": {"output_path": str(tmp / out_name), "render_fn": f"{__name__}:render_row"},
    }
    p = tmp / f"{out_name}.cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def steer_env(tmp_path, monkeypatch):
    _write_rows_and_axis(tmp_path)
    seen = {"loads": []}

    def fake_load(model_name, adapter=None, revision=None, post_base_load=None):
        model = _build_tiny_model()
        seen["loads"].append({"revision": revision, "hook": post_base_load is not None})
        if post_base_load is not None:
            post_base_load(model)
        return model, _build_tiny_tokenizer()

    monkeypatch.setattr(mi_cli, "_load_model_and_tokenizer", fake_load)
    return tmp_path, seen


def _run(tmp, cfg_path, revision=REVISION):
    cfg = load_steer_config(cfg_path)
    return mi_cli.run_steer(cfg, "tiny", revision, None, f"{__name__}:render_row", True)


def _texts(path: Path, arm: str):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r["answer_text"] for r in rows if r["arm"] == arm]


def test_steer_applies_patch_before_generation_and_binds_lineage(steer_env, capsys):
    tmp, seen = steer_env
    assert _run(tmp, _write_steer_config(tmp, "plain.jsonl"), None) == 0
    patch = _patch_spec(tmp / "p", scale=0.05)
    cfg_path = _write_steer_config(tmp, "patched.jsonl", patch)
    assert _run(tmp, cfg_path) == 0
    assert seen["loads"][-1] == {"revision": REVISION, "hook": True}

    plain = _texts(tmp / "plain.jsonl", "baseline")
    patched = _texts(tmp / "patched.jsonl", "baseline")
    assert patched != plain  # the edited weights were in place for generation

    rows = [json.loads(l) for l in (tmp / "patched.jsonl").read_text().splitlines()]
    digest = rows[0]["weight_patch_digest"]
    assert all(r["weight_patch_digest"] == digest for r in rows)
    assert not any("weight_patch_digest" in json.loads(l) for l in (tmp / "plain.jsonl").read_text().splitlines())
    manifest = json.loads((tmp / "patched.jsonl.manifest.json").read_text())
    assert manifest["weight_patch"]["patch_digest"] == digest
    assert manifest["surface"]["weight_patch"]["sha256"] == patch[1]
    assert "weight_patch" not in json.loads((tmp / "plain.jsonl.manifest.json").read_text())["surface"]
    assert json.loads(cell_mod.weight_patch_lineage_path(tmp / "patched.jsonl").read_text())["patch_digest"] == digest
    assert json.loads(cell_mod.smoke_state_path(tmp / "patched.jsonl").read_text())["weight_patch_digest"] == digest

    before = (tmp / "patched.jsonl").read_text()
    assert _run(tmp, cfg_path) == 0  # same patch resumes: nothing pending, nothing rewritten
    assert (tmp / "patched.jsonl").read_text() == before


def test_zero_patch_run_matches_unpatched_outputs(steer_env):
    tmp, _ = steer_env
    assert _run(tmp, _write_steer_config(tmp, "plain.jsonl"), None) == 0
    zero = _patch_spec(tmp / "z", scale=0.0)
    assert _run(tmp, _write_steer_config(tmp, "zero.jsonl", zero)) == 0
    for arm in ("baseline", "dose"):
        assert _texts(tmp / "zero.jsonl", arm) == _texts(tmp / "plain.jsonl", arm)
    manifest = json.loads((tmp / "zero.jsonl.manifest.json").read_text())
    assert manifest["weight_patch"]["noop"] is True


def test_stale_resume_is_refused_when_patch_changes_or_disappears(steer_env, capsys):
    tmp, _ = steer_env
    first = _patch_spec(tmp / "a", scale=0.05)
    assert _run(tmp, _write_steer_config(tmp, "rows_out.jsonl", first)) == 0
    rows_before = (tmp / "rows_out.jsonl").read_text()

    second = _patch_spec(tmp / "b", scale=0.025)
    assert _run(tmp, _write_steer_config(tmp, "rows_out.jsonl", second)) == mi_cli.EXIT_PATCH_LINEAGE
    assert _run(tmp, _write_steer_config(tmp, "rows_out.jsonl", None)) == mi_cli.EXIT_PATCH_LINEAGE
    assert (tmp / "rows_out.jsonl").read_text() == rows_before  # nothing spliced in


def test_patch_refusals_return_7_before_any_row_is_written(steer_env, capsys):
    tmp, seen = steer_env
    spec, sha = _patch_spec(tmp / "p", scale=0.01)
    # wrong pinned sha in the config
    assert _run(tmp, _write_steer_config(tmp, "a.jsonl", (spec, "0" * 64))) == mi_cli.EXIT_PATCH_REFUSED
    # spec edited on disk after pinning
    spec.write_text(spec.read_text() + "\n# edited\n")
    assert _run(tmp, _write_steer_config(tmp, "b.jsonl", (spec, sha))) == mi_cli.EXIT_PATCH_REFUSED
    # missing / wrong declared model revision
    spec2, sha2 = _patch_spec(tmp / "q", scale=0.01)
    cfg = _write_steer_config(tmp, "c.jsonl", (spec2, sha2))
    assert _run(tmp, cfg, None) == mi_cli.EXIT_PATCH_REFUSED
    assert _run(tmp, cfg, OTHER_REVISION) == mi_cli.EXIT_PATCH_REFUSED
    assert not seen["loads"]  # all refused before the model was even loaded
    # payload tampered
    payload = next((tmp / "q").glob("*.payload.safetensors"))
    payload.write_bytes(payload.read_bytes()[:-1] + b"\x00")
    assert _run(tmp, cfg) == mi_cli.EXIT_PATCH_REFUSED
    for name in "abc":
        assert not (tmp / f"{name}.jsonl").exists()
    out = capsys.readouterr().out
    assert "Weight patch refused" in out


def test_wrong_loaded_base_is_refused_after_load(steer_env, monkeypatch):
    tmp, _ = steer_env
    spec, sha = _patch_spec(tmp / "p", scale=0.01)

    def other_base(model_name, adapter=None, revision=None, post_base_load=None):
        model = _build_tiny_model()
        with torch.no_grad():
            dict(model.named_parameters())[TARGET].add_(1.0)
        post_base_load(model)
        return model, _build_tiny_tokenizer()

    monkeypatch.setattr(mi_cli, "_load_model_and_tokenizer", other_base)
    assert _run(tmp, _write_steer_config(tmp, "x.jsonl", (spec, sha))) == mi_cli.EXIT_PATCH_REFUSED
    assert not (tmp / "x.jsonl").exists()


# --------------------------------------------------------------------------
# dose-calibrate revision + pipeline module form
# --------------------------------------------------------------------------


class _Stop(Exception):
    pass


def test_run_dose_calibration_propagates_exact_revision(tmp_path, monkeypatch):
    _write_rows_and_axis(tmp_path)
    cfg = DoseCalibrationConfig(
        surface={"rows_path": str(tmp_path / "rows.jsonl")},
        readouts=[{"name": "axis", "path": str(tmp_path / "axis.json")}],
        law={"kind": "additive", "readout": "axis", "layer": 0},
        calibration={"doses": [1.0]},
        execution={"output_path": str(tmp_path / "o.jsonl"), "summary_path": str(tmp_path / "s.json")},
    )
    captured = {}

    def fake_load(model_name, adapter=None, revision=None, post_base_load=None):
        captured.update(model=model_name, adapter=adapter, revision=revision, hook=post_base_load)
        raise _Stop

    monkeypatch.setattr(mi_cli, "_load_model_and_tokenizer", fake_load)
    with pytest.raises(_Stop):
        mi_cli.run_dose_calibration(cfg, "m", None, f"{__name__}:render_row", True, revision=REVISION)
    assert captured["revision"] == REVISION and captured["hook"] is None
    with pytest.raises(_Stop):
        mi_cli.run_dose_calibration(cfg, "m", None, f"{__name__}:render_row", True)
    assert captured["revision"] is None  # absent revision behaves as before


def _write_dose_yaml(tmp: Path) -> Path:
    p = tmp / "dose.yaml"
    p.write_text(yaml.safe_dump(_dose_cfg().model_dump(mode="json")))
    return p


def test_pipeline_dose_stage_passes_stage_and_global_revision(tmp_path):
    cfg_path = _write_dose_yaml(tmp_path)
    for stage_rev, global_rev, expect in ((None, REVISION, REVISION), (OTHER_REVISION, REVISION, OTHER_REVISION), (None, None, None)):
        cfg = PipelineConfig(
            name="d", model="Tiny/Model", model_revision=global_rev, runtime={"python": "python"},
            stages=[{"name": "dose", "kind": "mechinterp.dose-calibrate", "config": str(cfg_path), "model_revision": stage_rev}],
        )
        cmd = build_pipeline_plan(cfg, repo_root=tmp_path, provider="local")["stages"][0]["command"]
        if expect is None:
            assert "--model-revision" not in cmd
        else:
            assert cmd[cmd.index("--model-revision") + 1] == expect


def test_pipeline_commands_use_installed_module_form_for_every_verb(tmp_path):
    cfg_path = _write_dose_yaml(tmp_path)
    cfg = PipelineConfig(
        name="all", model="M", runtime={"python": "pyx"},
        stages=[
            {"name": "e", "kind": "mechinterp.extract", "config": "e.yaml"},
            {"name": "p", "kind": "mechinterp.probe-fit", "config": "p.yaml"},
            {"name": "s", "kind": "mechinterp.steer", "config": "s.yaml", "render_fn": "m:r"},
            {"name": "d", "kind": "mechinterp.dose-calibrate", "config": str(cfg_path)},
            {"name": "g", "kind": "mechinterp.score-gates", "gates_config": "g.yaml", "rows_path": "r.jsonl"},
        ],
    )
    for stage in build_pipeline_plan(cfg, repo_root=tmp_path, provider="local")["stages"]:
        assert stage["command"][:4] == ["pyx", "-m", "tuner", "mechinterp"], stage["name"]
        assert not any(str(a).endswith("tuner.py") for a in stage["command"])


def test_stage_env_puts_source_checkout_engine_on_pythonpath_only_when_present(tmp_path):
    cfg = PipelineConfig(name="x", stages=[{"name": "e", "kind": "mechinterp.extract", "config": "e.yaml"}])
    stage = cfg.stages[0]
    engine = tmp_path / "engine"
    (engine / "tuner").mkdir(parents=True)
    (engine / "tuner" / "__main__.py").write_text("")
    (engine / "tuner.py").write_text("")
    (engine / "pyproject.toml").write_text("")
    env = _env_for_stage(cfg, stage, engine)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(engine)
    bare = tmp_path / "wheel_like"
    (bare / "tuner").mkdir(parents=True)
    (bare / "tuner" / "__main__.py").write_text("")
    assert str(bare) not in _env_for_stage(cfg, stage, bare).get("PYTHONPATH", "")
    command = PipelineConfig(name="c", stages=[{"name": "c", "kind": "command", "command": ["true"]}])
    assert str(engine) not in _env_for_stage(command, command.stages[0], engine).get("PYTHONPATH", "")


# --------------------------------------------------------------------------
# CLI surface (parser + handler)
# --------------------------------------------------------------------------


def test_parser_accepts_weights_flags_and_keeps_export_config():
    from tuner.cli.parser import create_parser

    parser = create_parser()
    args = parser.parse_args(["mechinterp", "weights-patch", "--weights-config", "w.yaml", "--validate"])
    assert (args.command, args.subcommand, args.weights_config) == ("mechinterp", "weights-patch", "w.yaml")
    assert args.weights_validate is True and args.weights_export is False
    args = parser.parse_args(["mechinterp", "weights-patch", "--weights-config", "w.yaml", "--export"])
    assert args.weights_export is True and args.weights_validate is False
    args = parser.parse_args(["mechinterp", "weights-inspect", "--weights-config", "i.yaml"])
    assert args.subcommand == "weights-inspect" and args.weights_validate is False
    args = parser.parse_args(["flywheel", "export-fixtures", "--export-config", "e.yaml"])
    assert args.export_config == "e.yaml" and args.weights_export is False
    args = parser.parse_args(["mechinterp", "dose-calibrate", "--mi-config", "d.yaml", "--model", "m", "--model-revision", REVISION])
    assert args.model_revision == REVISION


def _handler(**kw):
    from tuner.handlers.mechinterp_handler import MechInterpHandler

    base = dict(json=True, weights_validate=False, weights_export=False)
    base.update(kw)
    return MechInterpHandler(args=Namespace(**base))


@pytest.fixture
def cli_ckpt(tmp_path):
    root = tmp_path / "ck"
    tensors = write_checkpoint(root)
    from tests.mech_interp.weight_patch_fixtures import standard_ops_and_arrays

    ops, arrays = standard_ops_and_arrays()
    spec = make_spec(
        tmp_path / "s", tensors, ops, arrays, checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors.index.json")},
        export_dir=tmp_path / "exp",
    )
    return tmp_path, root, spec


def test_handler_inspect_validate_export_and_flag_exclusivity(cli_ckpt, capsys):
    tmp, root, spec = cli_ckpt
    inspect_cfg = tmp / "inspect.yaml"
    inspect_cfg.write_text(yaml.safe_dump({
        "schema_version": "mechinterp-weights-inspect/v1", "checkpoint": str(root),
        "tensors": ["blk.0.*"], "hash_tensors": True,
    }))
    assert _handler(subcommand="weights-inspect", weights_config=str(inspect_cfg)).handle() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["success"] and out["data"]["n_tensors"] == 5 and len(out["data"]["tensors"]) == 3

    h = lambda **kw: _handler(subcommand="weights-patch", weights_config=str(spec), **kw)  # noqa: E731
    assert h().handle() == 1  # neither flag
    assert h(weights_validate=True, weights_export=True).handle() == 1  # both
    capsys.readouterr()
    assert h(weights_validate=True).handle() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["data"]["written"] is False and not (tmp / "exp").exists()
    assert out["data"]["receipt"]["patch_digest"]

    assert h(weights_export=True).handle() == 0
    exported = json.loads(capsys.readouterr().out)["data"]
    assert (tmp / "exp" / "mechinterp-weight-export.manifest.json").is_file()
    assert exported["patch_digest"] == out["data"]["receipt"]["patch_digest"]

    assert h(weights_export=True).handle() == 1  # second export refused
    err = capsys.readouterr().out
    assert "WEIGHTS_OUTPUT_EXISTS" in err


def test_handler_refusal_codes_surface(cli_ckpt, capsys):
    tmp, root, spec = cli_ckpt
    data = yaml.safe_load(spec.read_text())
    data["budget"]["max_relative_frobenius"] = 1e-9
    spec.write_text(yaml.safe_dump(data))
    h = _handler(subcommand="weights-patch", weights_config=str(spec), weights_validate=True)
    assert h.handle() == 1
    assert "WEIGHTS_BUDGET" in capsys.readouterr().out
    missing = _handler(subcommand="weights-inspect", weights_config=str(tmp / "nope.yaml"))
    assert missing.handle() == 1


def test_handler_dose_calibrate_forwards_model_revision(tmp_path, monkeypatch):
    seen = {}

    def fake_run(config, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(mi_cli, "run_dose_calibration", fake_run)
    d = tmp_path / "dose.yaml"
    cfg = _dose_cfg().model_dump(mode="json")
    cfg["execution"]["render_fn"] = f"{__name__}:render_row"
    d.write_text(yaml.safe_dump(cfg))
    h = _handler(
        subcommand="dose-calibrate", mechinterp_config=str(d), model="Org/Model",
        model_revision=REVISION, adapter=None, render_fn=None, i_know_this_runs_on_gpu=True,
    )
    assert h.handle() == 0
    assert seen["revision"] == REVISION and seen["model_name"] == "Org/Model"


def test_handler_extract_forwards_project_context(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(mi_cli, "run_extract", lambda config, **kw: seen.update(kw) or 0)
    e = tmp_path / "ex.yaml"
    e.write_text(yaml.safe_dump({"rows_path": "r", "output_dir": "o", "render_fn": "m:r", "content_end_fn": "m:c"}))
    h = _handler(subcommand="extract", mechinterp_config=str(e), model="M", model_revision=REVISION,
                 adapter=None, i_know_this_runs_on_gpu=True)
    assert h.handle() == 0
    assert seen["project_context"] is h.context and seen["revision"] == REVISION
