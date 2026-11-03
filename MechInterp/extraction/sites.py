"""Explicit module-site activation capture.

A *site* is ``(layer, module, tensor)``: the ``input`` (first positional
argument) or ``output`` of a named submodule inside a decoder block. Examples
of what a recipe may name -- all supplied by config, none known to this module:
a block's ``mlp.down_proj`` input, an attention projection's output.

``layer`` is the decoder BLOCK index resolved with ``get_decoder_layer`` (the
convention steering uses), not a ``hidden_states`` index.

``SiteCapture`` registers hooks only for the duration of a ``with`` block and
removes every one of them on exit, including when registration or the forward
pass raises. Only the selected token positions are indexed out inside the hook,
so no per-sequence activation cache is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from MechInterp.intervention.hooks import get_decoder_layer

LAYER_CONVENTION = "decoder_block"


@dataclass(frozen=True)
class SiteSpec:
    layer: int
    module: str
    tensor: str  # "input" | "output"

    @property
    def key(self) -> str:
        return f"L{self.layer}.{self.module}.{self.tensor}"


def to_site_specs(sites) -> list[SiteSpec]:
    """Accept SiteSpec or any object/dict with layer/module/tensor."""
    out = []
    for s in sites:
        if isinstance(s, SiteSpec):
            out.append(s)
        elif isinstance(s, dict):
            out.append(SiteSpec(int(s["layer"]), str(s["module"]), str(s["tensor"])))
        else:
            out.append(SiteSpec(int(s.layer), str(s.module), str(s.tensor)))
    for s in out:
        if s.tensor not in ("input", "output"):
            raise ValueError(f"site {s.key}: tensor must be 'input' or 'output'")
    if len({s.key for s in out}) != len(out):
        raise ValueError("capture sites must be unique")
    return out


def resolve_sites(model, sites: Sequence[SiteSpec]) -> dict[str, torch.nn.Module]:
    """Resolve each site to its module; raise ValueError naming the bad site."""
    resolved: dict[str, torch.nn.Module] = {}
    for site in sites:
        try:
            block = get_decoder_layer(model, site.layer)
        except IndexError:
            raise ValueError(f"site {site.key}: decoder block {site.layer} is out of range") from None
        except AttributeError as exc:
            raise ValueError(f"site {site.key}: {exc}") from None
        try:
            resolved[site.key] = block.get_submodule(site.module)
        except AttributeError:
            raise ValueError(
                f"site {site.key}: block {site.layer} has no submodule {site.module!r}"
            ) from None
    return resolved


class SiteCapture:
    """Context manager capturing selected token positions at explicit sites."""

    def __init__(self, model, sites: Sequence[SiteSpec]):
        self.sites = to_site_specs(sites)
        self._modules = resolve_sites(model, self.sites)
        self._handles: list = []
        self._positions: list[int] = []
        self._armed = False
        self._captured: dict[str, torch.Tensor] = {}
        self._meta: dict[str, dict] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "SiteCapture":
        try:
            for site in self.sites:
                module = self._modules[site.key]
                if site.tensor == "input":
                    handle = module.register_forward_pre_hook(
                        self._pre_hook(site), with_kwargs=True
                    )
                else:
                    handle = module.register_forward_hook(self._post_hook(site))
                self._handles.append(handle)
        except BaseException:
            self.remove()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()
        self._armed = False

    # -- capture -----------------------------------------------------------

    def begin(self, positions: Sequence[int]) -> None:
        """Arm for the next forward pass: capture exactly these columns."""
        pos = sorted({int(p) for p in positions})
        if not pos or pos[0] < 0:
            raise ValueError("capture positions must be a non-empty set of non-negative indices")
        self._positions = pos
        self._armed = True
        self._captured = {}
        self._meta = {}

    def _record(self, site: SiteSpec, x) -> None:
        if not self._armed:
            return
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"site {site.key}: expected a tensor, got {type(x).__name__}")
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(
                f"site {site.key}: expected shape [1, seq, features], got {list(x.shape)}"
            )
        if site.key in self._captured:
            raise ValueError(f"site {site.key}: module fired more than once in one forward pass")
        if self._positions[-1] >= x.shape[1]:
            raise ValueError(
                f"site {site.key}: position {self._positions[-1]} outside sequence length {x.shape[1]}"
            )
        index = torch.tensor(self._positions, dtype=torch.long, device=x.device)
        selected = x.detach()[0].index_select(0, index)
        self._captured[site.key] = selected.to("cpu", torch.float32).contiguous()
        self._meta[site.key] = {
            "layer": site.layer,
            "module": site.module,
            "tensor": site.tensor,
            "source_dtype": str(x.dtype).removeprefix("torch."),
            "layer_convention": LAYER_CONVENTION,
        }

    def _pre_hook(self, site: SiteSpec):
        def hook(module, args, kwargs):
            if not args:
                raise ValueError(f"site {site.key}: module was called without a positional input")
            self._record(site, args[0])

        return hook

    def _post_hook(self, site: SiteSpec):
        def hook(module, args, output):
            value = output
            if isinstance(output, (tuple, list)) and output:
                value = output[0]
            self._record(site, value)

        return hook

    def results(self) -> dict[str, torch.Tensor]:
        """{site key: float32 [n_positions, features]} for the armed positions."""
        missing = [s.key for s in self.sites if s.key not in self._captured]
        if missing:
            raise ValueError(f"sites did not fire during the forward pass: {missing}")
        return dict(self._captured)

    @property
    def positions(self) -> list[int]:
        return list(self._positions)

    def metadata(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self._meta.items()}
