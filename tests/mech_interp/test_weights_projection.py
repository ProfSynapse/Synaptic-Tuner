"""CPU tests for bounded numeric reads in ``weights-inspect`` (projections)."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch
import yaml
from safetensors.torch import save_file

from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.spec import parse_inspect_request
from MechInterp.weights.workflow import inspect_from_config
from tests.mech_interp.weight_patch_fixtures import (
    DOWN0,
    GATE0,
    NORM,
    base_tensors,
    file_sha,
    write_checkpoint,
)


def refusal(code):
    return pytest.raises(WeightPatchError, match=rf"^\[{code}\]")


@pytest.fixture
def env(tmp_path):
    tensors = base_tensors()
    tensors[DOWN0][:, 5] = 0.0  # a zero column for the cosine-null case
    root = tmp_path / "ck"
    write_checkpoint(root, tensors)
    vec = torch.randn(8, generator=torch.Generator().manual_seed(3))
    save_file({"axis": vec}, str(tmp_path / "vec.safetensors"))
    (tmp_path / "dir.json").write_text(
        json.dumps({"schema_version": "mechinterp-direction/v1", "vector": vec.tolist()})
    )
    return tmp_path, root, tensors, vec


def _inspect(tmp, root, projections, **extra):
    doc = {
        "schema_version": "mechinterp-weights-inspect/v1",
        "checkpoint": str(root),
        "projections": projections,
        **extra,
    }
    p = tmp / "inspect.yaml"
    p.write_text(yaml.safe_dump(doc))
    return inspect_from_config(p)


def _vec_ref(tmp):
    return {"safetensors": {"path": str(tmp / "vec.safetensors"), "key": "axis", "sha256": file_sha(tmp / "vec.safetensors")}}


def test_selected_column_dots_match_manual_float64_reference(env):
    tmp, root, tensors, vec = env
    idx = [0, 3, 9, 15]
    report = _inspect(tmp, root, [{"id": "d", "tensor": DOWN0, "axis": "cols", "indices": idx, "vector": _vec_ref(tmp), "include_norms": True}])
    (p,) = report["projections"]
    w = tensors[DOWN0].double()  # [8, 16]
    expected = [float(w[:, j] @ vec.double()) for j in idx]
    assert p["values"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert p["line_norms"] == pytest.approx([float(w[:, j].norm()) for j in idx], rel=1e-12)
    assert p["indices"] == idx and p["n_values"] == 8 and p["kind"] == "dot"
    assert p["shape"] == [8, 16] and p["dtype"] == "bfloat16"
    from tests.mech_interp.weight_patch_fixtures import raw_sha
    assert p["tensor_sha256"] == raw_sha(tensors[DOWN0])
    assert p["vector"]["source"] == "safetensors" and p["vector"]["length"] == 8
    assert p["vector"]["l2_norm"] == pytest.approx(float(vec.double().norm()))
    json.dumps(report, allow_nan=False)  # strictly JSON-able


def test_all_lines_rows_axis_and_other_indices(env):
    tmp, root, tensors, vec = env
    sub = [1, 4, 6]
    report = _inspect(tmp, root, [
        {"id": "all", "tensor": DOWN0, "axis": "cols", "vector": _vec_ref(tmp)},
        {"id": "rows", "tensor": GATE0, "axis": "rows", "indices": [2, 7], "other_indices": sub,
         "vector": {"inline": [0.5, -1.0, 2.0]}},
    ])
    allp, rows = report["projections"]
    assert allp["indices"] == "all" and len(allp["values"]) == 16
    w = tensors[DOWN0].double()
    assert allp["values"] == pytest.approx([float(w[:, j] @ vec.double()) for j in range(16)], abs=1e-12)
    g = tensors[GATE0].double()  # [16, 8]
    v = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    assert rows["values"] == pytest.approx([float(g[i, sub] @ v) for i in (2, 7)], abs=1e-12)
    assert rows["other_indices"] == sub and rows["vector"]["source"] == "inline"


def test_cosine_and_zero_line_is_null(env):
    tmp, root, tensors, vec = env
    report = _inspect(tmp, root, [{"id": "c", "tensor": DOWN0, "kind": "cosine", "axis": "cols", "indices": [5, 2], "vector": _vec_ref(tmp)}])
    zero, other = report["projections"][0]["values"]
    assert zero is None
    w = tensors[DOWN0].double()
    assert other == pytest.approx(float(w[:, 2] @ vec.double() / (w[:, 2].norm() * vec.double().norm())), abs=1e-12)


def test_slice_returns_exact_selected_values(env):
    tmp, root, tensors, _ = env
    report = _inspect(tmp, root, [{"id": "s", "tensor": GATE0, "kind": "slice", "axis": "rows", "indices": [3, 4], "other_indices": [0, 7]}])
    (p,) = report["projections"]
    g = tensors[GATE0].double()
    assert p["values"] == [[float(g[3, 0]), float(g[3, 7])], [float(g[4, 0]), float(g[4, 7])]]
    assert p["n_values"] == 4 and "vector" not in p


def test_direction_json_vector_source(env):
    tmp, root, tensors, vec = env
    ref = {"direction_json": {"path": str(tmp / "dir.json"), "sha256": file_sha(tmp / "dir.json")}}
    a = _inspect(tmp, root, [{"id": "a", "tensor": DOWN0, "axis": "cols", "indices": [1, 2], "vector": ref}])
    b = _inspect(tmp, root, [{"id": "a", "tensor": DOWN0, "axis": "cols", "indices": [1, 2], "vector": _vec_ref(tmp)}])
    assert a["projections"][0]["values"] == pytest.approx(b["projections"][0]["values"], abs=1e-6)
    assert a["projections"][0]["vector"]["source"] == "direction_json"
    (tmp / "dir.json").write_text(json.dumps({"schema_version": "other/v1", "vector": [1.0] * 8}))
    ref["direction_json"]["sha256"] = file_sha(tmp / "dir.json")
    with refusal("payload"):
        _inspect(tmp, root, [{"id": "a", "tensor": DOWN0, "axis": "cols", "vector": ref}])


def test_unsafe_or_unbounded_requests_are_refused(env):
    tmp, root, tensors, _ = env
    base = {"id": "p", "tensor": DOWN0, "axis": "cols", "vector": _vec_ref(tmp)}

    def run(**kw):
        extra = kw.pop("extra", {})
        return _inspect(tmp, root, [{**base, **kw}], **extra)

    with refusal("payload"):
        run(indices=[16])  # out of range
    with refusal("payload"):
        run(indices=[1, 1])  # not distinct
    with refusal("payload"):
        run(other_indices=[0, 1])  # vector length mismatch
    with refusal("payload"):
        run(tensor=NORM)  # 1-D tensor
    with refusal("binding"):
        run(tensor="missing.weight")
    with refusal("budget"):
        run(extra={"max_output_values": 8})  # 16 lines > 8
    with refusal("budget"):
        run(indices=[0, 1, 2], include_norms=True, extra={"max_output_values": 5})
    with refusal("budget"):
        run(extra={"max_tensor_bytes": 16})
    with refusal("budget"):  # slice size = lines * positions
        _inspect(tmp, root, [{"id": "s", "tensor": GATE0, "kind": "slice", "axis": "rows"}], max_output_values=100)
    # the total across requests is what is capped
    with refusal("budget"):
        _inspect(tmp, root, [
            {"id": "a", "tensor": DOWN0, "axis": "cols", "indices": [0, 1, 2], "vector": _vec_ref(tmp)},
            {"id": "b", "tensor": DOWN0, "axis": "cols", "indices": [3, 4, 5], "vector": _vec_ref(tmp)},
        ], max_output_values=5)
    # pinned vector file must match its hash
    bad = _vec_ref(tmp)
    bad["safetensors"]["sha256"] = "0" * 64
    with refusal("payload"):
        run(vector=bad)


def test_nonfinite_checkpoint_values_and_vectors_are_refused(env):
    tmp, root, tensors, _ = env
    inf = {k: v.clone() for k, v in tensors.items()}
    inf[DOWN0][0, 0] = float("inf")
    bad_root = tmp / "bad"
    write_checkpoint(bad_root, inf)
    with refusal("nonfinite"):
        _inspect(tmp, bad_root, [{"id": "p", "tensor": DOWN0, "axis": "cols", "indices": [0], "vector": _vec_ref(tmp)}])
    with pytest.raises(WeightPatchError) as exc:  # NaN inline is rejected by the schema
        _inspect(tmp, root, [{"id": "p", "tensor": DOWN0, "axis": "cols", "vector": {"inline": [float("nan")] * 8}}])
    assert exc.value.code == "schema"


def test_schema_rules_for_projection_requests(env):
    tmp, root, *_ = env
    ok = {"schema_version": "mechinterp-weights-inspect/v1", "checkpoint": str(root)}
    for bad in (
        {"id": "a", "tensor": DOWN0, "axis": "cols"},  # dot needs vector
        {"id": "a", "tensor": DOWN0, "kind": "slice", "axis": "cols", "vector": {"inline": [1.0]}},
        {"id": "a", "tensor": DOWN0, "axis": "cols", "vector": {"inline": [1.0], "safetensors": {"path": "x", "key": "k", "sha256": "0" * 64}}},
        {"id": "a", "tensor": DOWN0, "axis": "diag", "vector": {"inline": [1.0]}},
        {"id": "a", "tensor": DOWN0, "axis": "cols", "vector": {"inline": [1.0]}, "surprise": 1},
    ):
        with refusal("schema"):
            parse_inspect_request({**ok, "projections": [bad]})
    twin = {"id": "a", "tensor": DOWN0, "axis": "cols", "vector": {"inline": [1.0]}}
    with refusal("schema"):
        parse_inspect_request({**ok, "projections": [twin, twin]})


def test_inspect_without_projections_is_unchanged(env):
    tmp, root, *_ = env
    report = _inspect(tmp, root, [])
    assert "projections" not in report


def test_cli_inspect_runs_projections_through_handler(env, capsys):
    from tuner.handlers.mechinterp_handler import MechInterpHandler

    tmp, root, tensors, vec = env
    cfg = tmp / "inspect.yaml"
    cfg.write_text(yaml.safe_dump({
        "schema_version": "mechinterp-weights-inspect/v1", "checkpoint": "ck",
        "projections": [{"id": "d", "tensor": DOWN0, "axis": "cols", "indices": [0],
                         "vector": {"safetensors": {"path": "vec.safetensors", "key": "axis", "sha256": file_sha(tmp / "vec.safetensors")}}}],
    }))
    h = MechInterpHandler(args=Namespace(json=True, subcommand="weights-inspect", weights_config=str(cfg)))
    assert h.handle() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["data"]["projections"][0]["n_values"] == 1  # relative paths resolve against the document
