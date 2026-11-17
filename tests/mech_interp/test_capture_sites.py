"""CPU tests for explicit module-site activation capture (extract capture_sites)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from MechInterp.extraction import PositionSpec, extract_rows
from MechInterp.extraction.sites import SiteCapture, SiteSpec, resolve_sites, to_site_specs
from tests.mech_interp.test_batched_generation import _build_tiny_model, _build_tiny_tokenizer

D, H = 6, 10  # model width, MLP width


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D, H, bias=False)
        self.up_proj = nn.Linear(D, H, bias=False)
        self.down_proj = nn.Linear(H, D, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()

    def forward(self, x):
        return x + self.mlp(x)


class _Inner(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Tiny(nn.Module):
    """Layout ``model.model.layers`` -> resolved by get_decoder_layer's 'model.layers' path."""

    def __init__(self, n_layers=3, dtype=torch.float32):
        super().__init__()
        torch.manual_seed(0)
        self.model = _Inner(n_layers)
        self.to(dtype)

    def forward(self, x):
        return self.model(x)


def _x(seq=7, dtype=torch.float32, seed=1):
    return torch.randn(1, seq, D, generator=torch.Generator().manual_seed(seed)).to(dtype)


def _hook_counts(model):
    n = 0
    for m in model.modules():
        n += len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._forward_hooks_with_kwargs)
    return n


def _site_hook_count(model):
    """Hooks installed by SiteCapture (HF models also carry their own lazy hooks)."""
    n = 0
    for m in model.modules():
        for table in (m._forward_hooks, m._forward_pre_hooks):
            n += sum("SiteCapture" in getattr(h, "__qualname__", "") for h in table.values())
    return n


class ManualHook:
    """The reference: a hook registered by hand that keeps the whole tensor."""

    def __init__(self, module, kind):
        self.value = None
        if kind == "input":
            self.h = module.register_forward_pre_hook(lambda m, a: setattr(self, "value", a[0].detach().clone()))
        else:
            self.h = module.register_forward_hook(lambda m, a, o: setattr(self, "value", o.detach().clone()))


@pytest.mark.parametrize("tensor", ["input", "output"])
def test_capture_equals_manual_hook_at_selected_positions_only(tensor):
    model = Tiny()
    manual = ManualHook(model.model.layers[1].mlp.down_proj, tensor)
    x = _x(seq=9)
    positions = [0, 3, 8]
    with SiteCapture(model, [SiteSpec(1, "mlp.down_proj", tensor)]) as cap:
        cap.begin(positions)
        with torch.no_grad():
            model(x)
    manual.h.remove()
    got = cap.results()[f"L1.mlp.down_proj.{tensor}"]
    full = manual.value[0]
    assert got.shape == (3, H if tensor == "input" else D)  # only selected positions retained
    assert got.dtype == torch.float32 and got.device.type == "cpu"
    assert torch.equal(got, full[positions])
    assert cap.positions == positions
    meta = cap.metadata()[f"L1.mlp.down_proj.{tensor}"]
    assert meta["layer"] == 1 and meta["layer_convention"] == "decoder_block"


def test_multiple_sites_and_layer_indexing_is_decoder_block_explicit():
    model = Tiny(n_layers=3)
    sites = [SiteSpec(0, "mlp.down_proj", "input"), SiteSpec(2, "mlp.gate_proj", "output")]
    refs = [ManualHook(model.model.layers[0].mlp.down_proj, "input"), ManualHook(model.model.layers[2].mlp.gate_proj, "output")]
    with SiteCapture(model, sites) as cap:
        cap.begin([2, 4])
        model(_x())
    res = cap.results()
    assert torch.equal(res["L0.mlp.down_proj.input"], refs[0].value[0][[2, 4]])
    assert torch.equal(res["L2.mlp.gate_proj.output"], refs[1].value[0][[2, 4]])
    for r in refs:
        r.h.remove()


def test_bf16_source_is_persisted_as_exact_float32_with_dtype_metadata():
    model = Tiny(dtype=torch.bfloat16)
    manual = ManualHook(model.model.layers[0].mlp.down_proj, "input")
    with SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input")]) as cap:
        cap.begin([1])
        model(_x(dtype=torch.bfloat16))
    out = cap.results()["L0.mlp.down_proj.input"]
    assert out.dtype == torch.float32
    assert torch.equal(out, manual.value[0][[1]].float())
    assert cap.metadata()["L0.mlp.down_proj.input"]["source_dtype"] == "bfloat16"


def test_hooks_are_removed_after_normal_exit_and_after_forward_failure():
    model = Tiny()
    assert _hook_counts(model) == 0
    with SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input"), SiteSpec(1, "mlp.down_proj", "output")]) as cap:
        assert _hook_counts(model) == 2
        cap.begin([0])
        model(_x())
    assert _hook_counts(model) == 0

    class Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("forward failed")

    model.model.layers[2].mlp = Boom()
    with pytest.raises(RuntimeError, match="forward failed"):
        with SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input")]) as cap:
            cap.begin([0])
            model(_x())
    assert _hook_counts(model) == 0


def test_hooks_are_removed_when_registration_fails_midway(monkeypatch):
    model = Tiny()
    target = model.model.layers[1].mlp.down_proj

    def refuse(*a, **k):
        raise RuntimeError("cannot register")

    monkeypatch.setattr(target, "register_forward_hook", refuse)
    cap = SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input"), SiteSpec(1, "mlp.down_proj", "output")])
    with pytest.raises(RuntimeError, match="cannot register"):
        cap.__enter__()
    assert _hook_counts(model) == 0  # the first (already registered) hook was removed


def test_error_inside_capture_still_removes_hooks_for_bad_positions():
    model = Tiny()
    with pytest.raises(ValueError, match="outside sequence length"):
        with SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input")]) as cap:
            cap.begin([50])
            model(_x(seq=7))
    assert _hook_counts(model) == 0


def test_unarmed_forward_captures_nothing_and_results_refuse():
    model = Tiny()
    with SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input")]) as cap:
        model(_x())
        with pytest.raises(ValueError, match="did not fire"):
            cap.results()


def test_module_firing_twice_and_batched_input_are_refused():
    model = Tiny()

    class Twice(nn.Module):  # calls its down_proj twice in one forward
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(D, D, bias=False)

        def forward(self, x):
            return self.down_proj(self.down_proj(x))

    model.model.layers[0].mlp = Twice()
    with pytest.raises(ValueError, match="more than once"):
        with SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input")]) as cap:
            cap.begin([0])
            model(_x())
    fresh = Tiny()
    with pytest.raises(ValueError, match=r"\[1, seq, features\]"):
        with SiteCapture(fresh, [SiteSpec(0, "mlp.down_proj", "input")]) as cap:
            cap.begin([0])
            fresh(torch.randn(2, 5, D))
    assert _hook_counts(fresh) == 0


def test_bad_sites_are_refused_before_any_hook_exists():
    model = Tiny(n_layers=2)
    for bad, msg in (
        (SiteSpec(5, "mlp.down_proj", "input"), "out of range"),
        (SiteSpec(0, "mlp.nope", "input"), "no submodule"),
    ):
        with pytest.raises(ValueError, match=msg):
            SiteCapture(model, [bad])
    with pytest.raises(ValueError, match="unique"):
        to_site_specs([SiteSpec(0, "mlp.down_proj", "input")] * 2)
    with pytest.raises(ValueError, match="'input' or 'output'"):
        to_site_specs([{"layer": 0, "module": "mlp.down_proj", "tensor": "grad"}])
    with pytest.raises(ValueError, match="decoder layers"):
        resolve_sites(nn.Linear(2, 2), [SiteSpec(0, "x", "input")])
    assert _hook_counts(model) == 0
    with pytest.raises(ValueError, match="non-empty"):
        SiteCapture(model, [SiteSpec(0, "mlp.down_proj", "input")]).begin([])


# --------------------------------------------------------------------------
# extract_rows integration on a tiny real GPT-2
# --------------------------------------------------------------------------

ROWS = [
    {"row_key": "a", "prompt": "w1 w2 w3"},
    {"row_key": "b", "prompt": "w4 w5 w6 w7 w8"},
]
SITE = {"layer": 1, "module": "mlp.c_proj", "tensor": "input"}


def _content_end(full, prompt_len, tokenizer):
    return int(full.shape[0]) - 1


def _extract(tmp, name, **kw):
    out = tmp / name
    manifest = extract_rows(
        _build_tiny_model(),
        _build_tiny_tokenizer(),
        ROWS,
        render_fn=lambda r: r["prompt"],
        content_end_fn=_content_end,
        spec=PositionSpec(families=["anchor", "answer_end"], layers=[1, 2]),
        out_dir=out,
        max_new_tokens=4,
        **kw,
    )
    return out, manifest


def test_extract_without_capture_sites_is_unchanged_and_with_them_is_additive(tmp_path):
    plain, plain_manifest = _extract(tmp_path, "plain")
    sited, sited_manifest = _extract(tmp_path, "sited", capture_sites=[SITE])
    names = sorted(p.name for p in plain.iterdir())
    assert not any(".sites." in n for n in names)
    assert "capture_sites" not in plain_manifest
    # residual files are byte-identical with and without the option
    for n in names:
        assert (plain / n).read_bytes() == (sited / n).read_bytes() or n == "manifest.json"
    extra = sorted(set(p.name for p in sited.iterdir()) - set(names))
    assert extra == [f"{k}__{f}.sites.safetensors" for k in ("a", "b") for f in ("anchor", "answer_end")]
    assert {k: v for k, v in sited_manifest.items() if k not in {"capture_sites", "capture_site_layer_convention", "capture_site_persist_dtype", "rows"}} == {
        k: v for k, v in plain_manifest.items() if k != "rows"
    }
    assert sited_manifest["capture_site_layer_convention"] == "decoder_block"
    assert sited_manifest["capture_sites"] == [{"key": "L1.mlp.c_proj.input", **SITE}]


def test_extract_site_capture_equals_manual_hook_and_records_metadata(tmp_path):
    model, tok = _build_tiny_model(), _build_tiny_tokenizer()
    out = tmp_path / "o"
    manifest = extract_rows(
        model, tok, ROWS, render_fn=lambda r: r["prompt"], content_end_fn=_content_end,
        spec=PositionSpec(families=["anchor", "answer_end"], layers=[]), out_dir=out,
        max_new_tokens=4, capture_sites=[SITE],
    )
    manual = ManualHook(model.transformer.h[1].mlp.c_proj, "input")
    for rec in manifest["rows"]:
        ids = tok(next(r["prompt"] for r in ROWS if r["row_key"] == rec["row_key"]), return_tensors="pt")["input_ids"]
        full = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False, num_beams=1, return_dict_in_generate=True).sequences[0]
        with torch.no_grad():
            model(input_ids=full[: rec["content_end"] + 1].unsqueeze(0), attention_mask=torch.ones(1, rec["content_end"] + 1, dtype=torch.long), use_cache=False)
        ref = manual.value[0]
        for family, pos in (("anchor", rec["prompt_len"] - 1), ("answer_end", rec["content_end"])):
            path = out / f"{rec['row_key']}__{family}.sites.safetensors"
            got = load_file(str(path))["L1.mlp.c_proj.input"]
            assert got.shape == (1, 128) and got.dtype == torch.float32
            assert torch.equal(got, ref[[pos]].float())
            with safe_open(str(path), "pt") as f:
                meta = f.metadata()
            assert meta["layer_convention"] == "decoder_block" and json.loads(meta["positions"]) == [pos]
            info = rec["site_captures"][family]
            assert info["positions"] == [pos]
            assert info["sites"]["L1.mlp.c_proj.input"] == {"shape": [1, 128], "persist_dtype": "float32", "source_dtype": "float32"}
    manual.h.remove()
    assert _site_hook_count(model) == 0
    # no residual layers requested -> no residual files, only site files
    assert all(".sites." in p.name or p.name == "manifest.json" for p in out.iterdir())


def test_extract_refuses_bad_site_before_generating(tmp_path):
    model = _build_tiny_model()
    calls = []
    orig = model.generate
    model.generate = lambda *a, **k: calls.append(1) or orig(*a, **k)
    with pytest.raises(ValueError, match="no submodule"):
        extract_rows(
            model, _build_tiny_tokenizer(), ROWS, render_fn=lambda r: r["prompt"], content_end_fn=_content_end,
            spec=PositionSpec(), out_dir=tmp_path / "o", capture_sites=[{**SITE, "module": "mlp.missing"}],
        )
    assert not calls


def test_extract_forward_failure_removes_site_hooks(tmp_path):
    model = _build_tiny_model()
    block = model.transformer.h[1]
    original_forward = block.mlp.c_proj.forward

    def failing(x):
        original_forward(x)  # fire the hook, then fail the pass
        raise RuntimeError("mid-forward failure")

    # generate() must succeed, the clean forward then fails after the hook fired
    real_forward = model.forward

    def forward(*a, **k):
        if k.get("output_hidden_states"):
            block.mlp.c_proj.forward = failing
        return real_forward(*a, **k)

    model.forward = forward
    with pytest.raises(RuntimeError, match="mid-forward failure"):
        extract_rows(
            model, _build_tiny_tokenizer(), ROWS, render_fn=lambda r: r["prompt"], content_end_fn=_content_end,
            spec=PositionSpec(), out_dir=tmp_path / "o", max_new_tokens=3, capture_sites=[SITE],
        )
    assert _site_hook_count(model) == 0


# --------------------------------------------------------------------------
# use_cache and forward_only
# --------------------------------------------------------------------------


class _ForwardSpy:
    """Records the use_cache argument of every model(...) call."""

    def __init__(self, model):
        self.calls = []
        self.generate_calls = 0
        real_forward, real_generate = model.forward, model.generate

        def forward(*a, **k):
            if k.get("output_hidden_states"):
                self.calls.append(k.get("use_cache"))
            return real_forward(*a, **k)

        def generate(*a, **k):
            self.generate_calls += 1
            return real_generate(*a, **k)

        model.forward, model.generate = forward, generate


def _extract_spied(tmp, name, **kw):
    model = _build_tiny_model()
    spy = _ForwardSpy(model)
    out = tmp / name
    manifest = extract_rows(
        model, _build_tiny_tokenizer(), ROWS, render_fn=lambda r: r["prompt"],
        content_end_fn=kw.pop("content_end_fn", _content_end),
        spec=kw.pop("spec", PositionSpec(families=["anchor", "answer_end"], layers=[1])),
        out_dir=out, max_new_tokens=kw.pop("max_new_tokens", 4), **kw,
    )
    return model, spy, out, manifest


def test_capture_use_cache_is_configurable_and_default_is_unchanged(tmp_path):
    _, spy_default, out_default, m_default = _extract_spied(tmp_path, "d", capture_sites=[SITE])
    assert set(spy_default.calls) == {False}  # historical teacher-forced behavior
    assert "capture_use_cache" not in m_default

    _, spy_true, out_true, m_true = _extract_spied(tmp_path, "t", capture_sites=[SITE], use_cache=True)
    assert set(spy_true.calls) == {True}
    assert m_true["capture_use_cache"] is True
    # for a plain causal LM the captured values do not depend on the cache flag
    for n in sorted(p.name for p in out_default.iterdir() if p.name != "manifest.json"):
        a, b = load_file(str(out_default / n)), load_file(str(out_true / n))
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_site_capture_equals_manual_hook_with_use_cache_true(tmp_path):
    model, spy, out, manifest = _extract_spied(
        tmp_path, "c", spec=PositionSpec(families=["anchor", "answer_end"], layers=[]),
        capture_sites=[SITE], use_cache=True,
    )
    assert set(spy.calls) == {True}
    manual = ManualHook(model.transformer.h[1].mlp.c_proj, "input")
    for rec in manifest["rows"]:
        ids = _build_tiny_tokenizer()(next(r["prompt"] for r in ROWS if r["row_key"] == rec["row_key"]), return_tensors="pt")["input_ids"]
        full = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False, num_beams=1, return_dict_in_generate=True).sequences[0]
        n = rec["content_end"] + 1
        with torch.no_grad():
            model(input_ids=full[:n].unsqueeze(0), attention_mask=torch.ones(1, n, dtype=torch.long), use_cache=True)
        ref = manual.value[0]
        got = load_file(str(out / f"{rec['row_key']}__anchor.sites.safetensors"))["L1.mlp.c_proj.input"]
        assert torch.equal(got, ref[[rec["prompt_len"] - 1]].float())
    manual.h.remove()


def test_forward_only_anchor_capture_never_generates(tmp_path):
    spec = PositionSpec(families=["anchor"], layers=[1, 2])
    model, spy, out, manifest = _extract_spied(
        tmp_path, "f", spec=spec, content_end_fn=None, capture_sites=[SITE],
        use_cache=True, forward_only=True, max_new_tokens=999,
    )
    assert spy.generate_calls == 0 and set(spy.calls) == {True}
    assert manifest["forward_only"] is True and manifest["n_answered"] == 2
    manual = ManualHook(model.transformer.h[1].mlp.c_proj, "input")
    for rec in manifest["rows"]:
        prompt = next(r["prompt"] for r in ROWS if r["row_key"] == rec["row_key"])
        ids = _build_tiny_tokenizer()(prompt, return_tensors="pt")["input_ids"]
        assert rec["seq_total"] == rec["prompt_len"] == ids.shape[1] and rec["answer_text"] == ""
        with torch.no_grad():
            res = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
        site = load_file(str(out / f"{rec['row_key']}__anchor.sites.safetensors"))["L1.mlp.c_proj.input"]
        assert torch.equal(site, manual.value[0][[-1]].float())
        resid = load_file(str(out / f"{rec['row_key']}__anchor.safetensors"))
        assert torch.equal(resid["L1"][0], res.hidden_states[1][0, -1].float())
    manual.h.remove()


def test_forward_only_requires_anchor_family_and_content_end_otherwise():
    model, tok = _build_tiny_model(), _build_tiny_tokenizer()
    common = dict(render_fn=lambda r: r["prompt"], out_dir="unused")
    with pytest.raises(ValueError, match="'anchor' family"):
        extract_rows(model, tok, ROWS, content_end_fn=None, spec=PositionSpec(families=["anchor", "answer_end"]), forward_only=True, **common)
    with pytest.raises(ValueError, match="content_end_fn is required"):
        extract_rows(model, tok, ROWS, content_end_fn=None, spec=PositionSpec(), **common)


def test_extract_config_use_cache_forward_only_contract():
    from MechInterp.config import ExtractConfig

    base = dict(rows_path="r", output_dir="o", render_fn="m:r")
    legacy = ExtractConfig(**base, content_end_fn="m:c")
    assert legacy.use_cache is False and legacy.forward_only is False
    assert ExtractConfig(**base, content_end_fn="m:c", use_cache=True).use_cache is True
    fo = ExtractConfig(**base, families=["anchor"], forward_only=True)
    assert fo.content_end_fn is None
    with pytest.raises(Exception, match="content_end_fn"):
        ExtractConfig(**base)  # still required when generating
    with pytest.raises(Exception, match="forward_only requires"):
        ExtractConfig(**base, forward_only=True)  # default families include answer_end
