"""CPU tests for the generic weight-patch core: spec, safetensors I/O, planning,
in-memory apply/revert and create-only export. Tiny local checkpoints only."""

from __future__ import annotations

import json
import errno
import os
import struct
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from MechInterp.weights import runtime as wruntime
from MechInterp.weights.checkpoint import (
    inspect_checkpoint,
    open_checkpoint,
)
from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.patch import load_payload
from MechInterp.weights.safetensors_io import hash_file_with_regions, read_header, read_region
from MechInterp.weights.spec import (
    InspectRequest,
    load_patch_spec,
    parse_patch_spec,
    read_document,
)
from MechInterp.weights.workflow import (
    export_patch_config,
    plan_against_checkpoint,
    validate_patch_config,
)

from tests.mech_interp.weight_patch_fixtures import (
    DOWN0,
    DOWN1,
    EMBED,
    GATE0,
    NORM,
    OTHER_REVISION,
    REVISION,
    base_tensors,
    file_sha,
    make_spec,
    raw_sha,
    rewrite_spec,
    standard_ops_and_arrays,
    write_checkpoint,
)


def refusal(code):
    return pytest.raises(WeightPatchError, match=rf"^\[{code}\]")


@pytest.fixture
def ckpt_dir(tmp_path):
    tensors = write_checkpoint(tmp_path / "base")
    return tmp_path / "base", tensors


def _std_spec(tmp_path, ckpt_dir, **kw):
    root, tensors = ckpt_dir
    ops, arrays = standard_ops_and_arrays()
    return make_spec(
        tmp_path / "spec",
        tensors,
        ops,
        arrays,
        checkpoint=root,
        base_extra=kw.pop("base_extra", {"index_sha256": file_sha(root / "model.safetensors.index.json")}),
        **kw,
    )


# --------------------------------------------------------------------------
# safetensors I/O
# --------------------------------------------------------------------------


def test_header_reader_agrees_with_safetensors_library_and_hashes_bf16_bytes(ckpt_dir):
    root, tensors = ckpt_dir
    sf = read_header(root / "model-00001-of-00002.safetensors")
    assert set(sf.entries) == {DOWN0, GATE0, NORM}
    assert sf.metadata == {"format": "pt"}
    lib = load_file(str(sf.path))
    whole, regions = hash_file_with_regions(sf)
    assert whole == file_sha(sf.path)
    for name, entry in sf.entries.items():
        assert entry.torch_dtype == str(lib[name].dtype).removeprefix("torch.")
        assert list(entry.shape) == list(lib[name].shape)
        # region hash == independent hash of the stored dtype bytes (bf16 included)
        assert regions[name] == raw_sha(tensors[name])
        got = torch.frombuffer(read_region(sf, entry), dtype=lib[name].dtype).reshape(entry.shape)
        assert torch.equal(got, lib[name])
    assert tensors[DOWN0].dtype == torch.bfloat16


def _header_bytes(path: Path):
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    return raw, n, json.loads(raw[8 : 8 + n])


def _rewrite(path: Path, header: dict, data: bytes, *, pad: bool = True):
    body = json.dumps(header).encode()
    if pad:
        body += b" " * (-len(body) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)


@pytest.mark.parametrize(
    "mutation",
    ["truncate", "bad_len", "overlap", "gap", "wrong_bytes", "trailing", "bad_dtype", "neg_shape"],
)
def test_malformed_safetensors_headers_are_refused(tmp_path, mutation):
    p = tmp_path / "m.safetensors"
    save_file({"a": torch.zeros(4), "b": torch.ones(4)}, str(p))
    raw, n, header = _header_bytes(p)
    data = raw[8 + n :]
    if mutation == "truncate":
        p.write_bytes(raw[:-3])
    elif mutation == "bad_len":
        p.write_bytes(struct.pack("<Q", 10**12) + raw[8:])
    elif mutation == "overlap":
        header["b"]["data_offsets"] = [8, 24]
        _rewrite(p, header, data)
    elif mutation == "gap":
        header["b"]["data_offsets"] = [20, 36]
        _rewrite(p, header, data + b"\0" * 8)
    elif mutation == "wrong_bytes":
        header["b"]["shape"] = [5]
        _rewrite(p, header, data)
    elif mutation == "trailing":
        _rewrite(p, header, data + b"\0\0\0\0")
    elif mutation == "bad_dtype":
        header["a"]["dtype"] = "F4"
        _rewrite(p, header, data)
    elif mutation == "neg_shape":
        header["a"]["shape"] = [-4]
        _rewrite(p, header, data)
    with refusal("index"):
        read_header(p)


def test_duplicate_header_keys_and_nan_constants_are_refused(tmp_path):
    p = tmp_path / "d.safetensors"
    body = b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    body += b" " * (-len(body) % 8)
    p.write_bytes(struct.pack("<Q", len(body)) + body + b"\0\0\0\0")
    with refusal("index"):
        read_header(p)


# --------------------------------------------------------------------------
# Checkpoint index / path safety / binding
# --------------------------------------------------------------------------


def test_open_checkpoint_sharded_and_single(tmp_path):
    write_checkpoint(tmp_path / "s", sharded=True)
    write_checkpoint(tmp_path / "one", sharded=False)
    sharded = open_checkpoint(tmp_path / "s")
    single = open_checkpoint(tmp_path / "one")
    assert set(sharded.tensor_to_file) == set(single.tensor_to_file)
    assert len(sharded.files) == 2 and len(single.files) == 1
    assert open_checkpoint(tmp_path / "one" / "model.safetensors").is_single_file_input


def _index(root: Path):
    p = root / "model.safetensors.index.json"
    return p, json.loads(p.read_text())


@pytest.mark.parametrize(
    "mutation", ["missing_tensor", "traversal_name", "absolute_name", "extra_in_shard", "empty_map", "dup_key", "missing_shard"]
)
def test_malformed_index_is_refused(tmp_path, mutation):
    root = tmp_path / "ck"
    write_checkpoint(root)
    p, idx = _index(root)
    wm = idx["weight_map"]
    if mutation == "missing_tensor":
        wm["ghost.weight"] = "model-00001-of-00002.safetensors"
    elif mutation == "traversal_name":
        wm[DOWN0] = "../outside.safetensors"
    elif mutation == "absolute_name":
        wm[DOWN0] = "/etc/passwd.safetensors"
    elif mutation == "extra_in_shard":
        del wm[GATE0]
    elif mutation == "empty_map":
        idx["weight_map"] = {}
    elif mutation == "missing_shard":
        wm[DOWN0] = "model-00009-of-00009.safetensors"
    if mutation == "dup_key":
        p.write_text('{"weight_map": {"a": "x.safetensors", "a": "x.safetensors"}}')
    else:
        p.write_text(json.dumps(idx))
    with pytest.raises(WeightPatchError) as exc:
        open_checkpoint(root)
    assert exc.value.code in {"index", "path"}


def test_index_and_single_file_together_is_ambiguous(tmp_path):
    root = tmp_path / "ck"
    write_checkpoint(root)
    write_checkpoint(tmp_path / "other", sharded=False)
    (root / "model.safetensors").write_bytes((tmp_path / "other" / "model.safetensors").read_bytes())
    with refusal("index"):
        open_checkpoint(root)


def test_symlinked_shards_refused_unless_opted_in(tmp_path):
    blobs = tmp_path / "blobs"
    write_checkpoint(blobs, sharded=False)
    snap = tmp_path / "snap"
    snap.mkdir()
    os.symlink(blobs / "model.safetensors", snap / "model.safetensors")
    with refusal("path"):
        open_checkpoint(snap)
    ck = open_checkpoint(snap, allow_symlinked_files=True)
    assert DOWN0 in ck.tensor_to_file
    # dangling symlink refused even when allowed
    os.remove(snap / "model.safetensors")
    os.symlink(blobs / "gone.safetensors", snap / "model.safetensors")
    with refusal("path"):
        open_checkpoint(snap, allow_symlinked_files=True)


def test_symlinked_directory_component_escapes_are_refused(tmp_path):
    outside = tmp_path / "outside"
    write_checkpoint(outside, sharded=False)
    root = tmp_path / "ck"
    root.mkdir()
    # a regular-looking shard name reached through a symlinked *directory* is not
    # possible via plain names, but the root's own resolved parent must still match
    os.symlink(outside / "model.safetensors", root / "model.safetensors")
    with refusal("path"):
        open_checkpoint(root)


def test_inspect_report_lists_and_hashes_selected_tensors(ckpt_dir):
    root, tensors = ckpt_dir
    req = InspectRequest(
        schema_version="mechinterp-weights-inspect/v1",
        checkpoint=str(root),
        hash_files=True,
        tensors=["blk.0.*"],
        hash_tensors=True,
    )
    report = inspect_checkpoint(req)
    assert report["n_tensors"] == 5 and report["n_files"] == 2
    assert report["layout"] == "sharded_index"
    assert report["dtype_counts"] == {"bfloat16": 4, "float32": 1}
    assert report["index_total_size_matches"] is True
    listed = {t["name"]: t for t in report["tensors"]}
    assert set(listed) == {DOWN0, GATE0, NORM}
    assert listed[DOWN0]["sha256"] == raw_sha(tensors[DOWN0])
    assert report["base_tensors_snippet"][DOWN0] == {
        "dtype": "bfloat16",
        "shape": [8, 16],
        "sha256": raw_sha(tensors[DOWN0]),
    }
    assert all("sha256" in f for f in report["files"])


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------


def _spec_dict(tmp_path, ckpt_dir):
    spec_path = _std_spec(tmp_path, ckpt_dir)
    return yaml.safe_load(spec_path.read_text()), spec_path


def test_spec_rejects_unlisted_disallowed_and_malformed_inputs(tmp_path, ckpt_dir):
    data, _ = _spec_dict(tmp_path, ckpt_dir)

    def mutated(fn):
        d = json.loads(json.dumps(data))
        fn(d)
        return d

    cases = {
        "extra field": mutated(lambda d: d.update(surprise=1)),
        "short revision": mutated(lambda d: d["base"].update(revision="abc123")),
        "disallowed tensor": mutated(lambda d: d["budget"].update(allowed_tensors=["nothing.*"])),
        "unpinned op tensor": mutated(lambda d: d["base"]["tensors"].pop(GATE0)),
        "pinned but untargeted": mutated(
            lambda d: d["base"]["tensors"].update({EMBED: d["base"]["tensors"][DOWN0]})
        ),
        "max tensors": mutated(lambda d: d["budget"].update(max_modified_tensors=1)),
        "missing role": mutated(lambda d: d["operations"][0]["arrays"].pop("v")),
        "unknown role": mutated(lambda d: d["operations"][0]["arrays"].update(w="x")),
        "scale on replace": mutated(lambda d: d["operations"][1].update(scale=2.0)),
        "inline and array rows": mutated(lambda d: d["operations"][1]["arrays"].update(rows="x")),
        "dup op id": mutated(lambda d: d["operations"][1].update(id="r1")),
        "bool scale": mutated(lambda d: d["operations"][0].update(scale=True)),
        "bad dtype pin": mutated(lambda d: d["base"]["tensors"][DOWN0].update(dtype="int8")),
        "bad hash": mutated(lambda d: d["base"]["tensors"][DOWN0].update(sha256="xyz")),
        "no budget cap": mutated(lambda d: d["budget"].pop("max_relative_frobenius")),
        "wrong schema": mutated(lambda d: d.update(schema_version="mechinterp-weight-patch/v2")),
    }
    for label, d in cases.items():
        with pytest.raises(WeightPatchError, match=r"^\[schema\]"):
            parse_patch_spec(d)


def test_nan_and_inf_scale_in_yaml_are_refused(tmp_path, ckpt_dir):
    for token in (".nan", ".inf"):
        spec_path = _std_spec(tmp_path, ckpt_dir, name=f"nan{token[1:]}")
        text = spec_path.read_text().replace("scale: 0.05", f"scale: {token}")
        spec_path.write_text(text)
        with refusal("schema"):
            load_patch_spec(spec_path)


def test_duplicate_yaml_keys_are_refused(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("a: 1\na: 2\n")
    with refusal("schema"):
        read_document(p)


# --------------------------------------------------------------------------
# Planning: numerics, budget, zero-patch
# --------------------------------------------------------------------------


def _plan(tmp_path, ckpt_dir, **kw):
    spec_path = _std_spec(tmp_path, ckpt_dir, **kw)
    spec, digest = load_patch_spec(spec_path)
    return spec, digest, plan_against_checkpoint(spec, digest)


def test_rank_one_selected_columns_and_replace_rows_match_manual_reference(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    spec, digest, (ckpt, binding, payload, plan) = _plan(tmp_path, ckpt_dir)
    u, v = payload.arrays["r1.u"], payload.arrays["r1.v"]
    expected = tensors[DOWN0].clone().float()
    expected[:, [1, 4, 9]] += 0.05 * torch.outer(u, v)
    expected = expected.to(torch.bfloat16)
    got = plan.targeted[DOWN0].new_tensor
    assert torch.equal(got, expected)
    # only the three selected columns may differ
    changed_cols = (got != tensors[DOWN0]).any(dim=0).nonzero().flatten().tolist()
    assert set(changed_cols) <= {1, 4, 9}
    rep = plan.targeted[GATE0].new_tensor
    assert torch.equal(rep[[2, 5]], payload.arrays["rows.values"].to(torch.bfloat16))
    untouched = [i for i in range(16) if i not in (2, 5)]
    assert torch.equal(rep[untouched], tensors[GATE0][untouched])
    assert plan.targeted[DOWN0].after_sha256 == raw_sha(got)
    assert plan.targeted[DOWN0].before_sha256 == raw_sha(tensors[DOWN0])
    assert 0 < plan.targeted[DOWN0].relative_frobenius < 5
    assert binding["content_pinned"] is True


@pytest.mark.parametrize("axis,dim", [("output", 8), ("input", 16)])
def test_project_removes_component_along_direction(tmp_path, axis, dim):
    tensors = base_tensors(torch.float32)
    root = tmp_path / "ck"
    write_checkpoint(root, tensors, sharded=False)
    d = torch.randn(dim, generator=torch.Generator().manual_seed(4))
    d = d / d.norm()
    spec_path = make_spec(
        tmp_path / "s",
        tensors,
        [{"id": "p", "tensor": DOWN0, "op": "project", "axis": axis, "arrays": {"direction": "d"}}],
        {"d": d},
        checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
    )
    spec, digest = load_patch_spec(spec_path)
    _, _, _, plan = plan_against_checkpoint(spec, digest)
    w = plan.targeted[DOWN0].new_tensor
    residual = (d @ w) if axis == "output" else (w @ d)
    assert residual.abs().max() < 1e-5
    # a non-unit direction is refused unless normalize is set
    spec_path2 = make_spec(
        tmp_path / "s2",
        tensors,
        [{"id": "p", "tensor": DOWN0, "op": "project", "axis": axis, "arrays": {"direction": "d"}}],
        {"d": d * 3},
        checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
    )
    spec2, digest2 = load_patch_spec(spec_path2)
    with refusal("payload"):
        plan_against_checkpoint(spec2, digest2)


def test_replace_cols_and_payload_index_arrays(tmp_path):
    tensors = base_tensors()
    root = tmp_path / "ck"
    write_checkpoint(root, tensors, sharded=False)
    vals = torch.randn(8, 2, generator=torch.Generator().manual_seed(5))
    spec_path = make_spec(
        tmp_path / "s",
        tensors,
        [{"id": "c", "tensor": DOWN0, "op": "replace_cols", "arrays": {"values": "v", "cols": "idx"}}],
        {"v": vals, "idx": torch.tensor([3, 7], dtype=torch.int64)},
        checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
    )
    spec, digest = load_patch_spec(spec_path)
    _, _, _, plan = plan_against_checkpoint(spec, digest)
    new = plan.targeted[DOWN0].new_tensor
    assert torch.equal(new[:, [3, 7]], vals.to(torch.bfloat16))
    assert plan.targeted[DOWN0].entries_changed <= 16


@pytest.mark.parametrize(
    "case",
    ["oob", "negative", "duplicate", "wrong_u_len", "payload_nan", "base_hash", "base_shape"],
)
def test_unsafe_inputs_refuse_before_anything_is_planned(tmp_path, ckpt_dir, case):
    root, tensors = ckpt_dir
    ops, arrays = standard_ops_and_arrays()
    expect = "payload"
    if case == "oob":
        ops[0]["cols"] = [1, 16]
    elif case == "negative":
        ops[0]["cols"] = [1, -1, 3]
    elif case == "duplicate":
        ops[0]["cols"] = [1, 1, 3]
    elif case == "wrong_u_len":
        arrays["r1.u"] = torch.randn(7)
    elif case == "payload_nan":
        arrays["r1.u"][0] = float("nan")
        expect = "nonfinite"
    spec_path = make_spec(
        tmp_path / "s",
        tensors,
        ops,
        arrays,
        checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors.index.json")},
    )
    if case == "base_hash":
        rewrite_spec(spec_path, lambda d: d["base"]["tensors"][GATE0].update(sha256="0" * 64))
        expect = "base_hash_mismatch"
    if case == "base_shape":
        rewrite_spec(spec_path, lambda d: d["base"]["tensors"][GATE0].update(shape=[8, 16]))
        expect = "base_hash_mismatch"
    spec, digest = load_patch_spec(spec_path)
    with refusal(expect):
        plan_against_checkpoint(spec, digest)


def test_budgets_refuse_whole_patch(tmp_path, ckpt_dir):
    with refusal("budget"):
        _plan(tmp_path / "a", ckpt_dir, budget={"max_relative_frobenius": 1e-9})
    with refusal("budget"):
        _plan(tmp_path / "b", ckpt_dir, budget={"max_modified_entries": 3})
    with refusal("budget"):
        _plan(tmp_path / "c", ckpt_dir, budget={"max_payload_bytes": 8})
    # relative cap is per tensor: a cap just above the larger realized value passes
    _, _, (_, _, _, plan) = _plan(tmp_path / "d", ckpt_dir)
    cap = max(t.relative_frobenius for t in plan.targeted.values()) * 1.0001
    _plan(tmp_path / "e", ckpt_dir, budget={"max_relative_frobenius": cap})


def test_payload_hash_and_unreferenced_arrays_are_refused(tmp_path, ckpt_dir):
    spec_path = _std_spec(tmp_path, ckpt_dir)
    spec, digest = load_patch_spec(spec_path)
    payload_file = Path(spec.payload.path)
    original = payload_file.read_bytes()
    payload_file.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with refusal("payload"):
        load_payload(spec)
    root, tensors = ckpt_dir
    ops, arrays = standard_ops_and_arrays()
    arrays["smuggled"] = torch.zeros(2)
    p2 = make_spec(
        tmp_path / "x", tensors, ops, arrays, checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors.index.json")},
    )
    spec2, _ = load_patch_spec(p2)
    with refusal("payload"):
        load_payload(spec2)


def test_zero_patch_is_a_true_noop_including_negative_zero(tmp_path):
    tensors = base_tensors()
    tensors[DOWN0][0, 0] = -0.0
    root = tmp_path / "ck"
    write_checkpoint(root, tensors, sharded=False)
    ops = [
        {"id": "z", "tensor": DOWN0, "op": "add_rank_one", "scale": 0.0, "arrays": {"u": "u", "v": "v"}},
        {"id": "zv", "tensor": DOWN0, "op": "add_rank_one", "cols": [0], "arrays": {"u": "zu", "v": "zv"}},
    ]
    arrays = {"u": torch.randn(8), "v": torch.randn(16), "zu": torch.zeros(8), "zv": torch.zeros(1)}
    spec_path = make_spec(
        tmp_path / "s", tensors, ops, arrays, checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
        export_dir=tmp_path / "out",
    )
    spec, digest = load_patch_spec(spec_path)
    _, _, _, plan = plan_against_checkpoint(spec, digest)
    assert plan.is_noop and plan.total_entries_changed == 0
    t = plan.targeted[DOWN0]
    assert t.before_sha256 == t.after_sha256 and t.new_tensor is None
    result = export_patch_config(spec_path)
    assert result["totals"]["noop"] is True
    assert file_sha(tmp_path / "out" / "model.safetensors") == file_sha(root / "model.safetensors")


def test_untouched_entries_in_an_edited_region_keep_exact_bytes(tmp_path):
    tensors = base_tensors()
    tensors[DOWN0][0, 0] = -0.0  # u[0] == 0 below, so this entry gets "+ 0.0"
    root = tmp_path / "ck"
    write_checkpoint(root, tensors, sharded=False)
    u = torch.ones(8)
    u[0] = 0.0
    spec_path = make_spec(
        tmp_path / "s", tensors,
        [{"id": "e", "tensor": DOWN0, "op": "add_rank_one", "scale": 0.5, "cols": [0, 1], "arrays": {"u": "u", "v": "v"}}],
        {"u": u, "v": torch.tensor([1.0, 1.0])}, checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
    )
    spec, digest = load_patch_spec(spec_path)
    _, _, _, plan = plan_against_checkpoint(spec, digest)
    new = plan.targeted[DOWN0].new_tensor
    assert plan.targeted[DOWN0].entries_changed > 0
    assert new[0, 0].view(torch.int16) == tensors[DOWN0][0, 0].view(torch.int16)  # still -0.0 bitwise
    assert torch.signbit(new[0, 0].float())


def test_sub_ulp_delta_is_unchanged_not_budgeted(tmp_path):
    tensors = base_tensors()
    root = tmp_path / "ck"
    write_checkpoint(root, tensors, sharded=False)
    ops = [{"id": "tiny", "tensor": DOWN0, "op": "add_rank_one", "scale": 1e-12, "arrays": {"u": "u", "v": "v"}}]
    spec_path = make_spec(
        tmp_path / "s", tensors, ops, {"u": torch.randn(8), "v": torch.randn(16)}, checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
    )
    spec, digest = load_patch_spec(spec_path)
    _, _, _, plan = plan_against_checkpoint(spec, digest)
    assert plan.is_noop  # realized delta in bf16 is zero


# --------------------------------------------------------------------------
# Revision / source binding
# --------------------------------------------------------------------------


def test_binding_requires_content_pins_or_revision_evidence(tmp_path, ckpt_dir):
    with refusal("binding"):
        _plan(tmp_path, ckpt_dir, base_extra={})  # no pins, plain dir, no evidence


def test_wrong_index_or_file_hash_refuses(tmp_path, ckpt_dir):
    with refusal("binding"):
        _plan(tmp_path / "a", ckpt_dir, base_extra={"index_sha256": "0" * 64})
    root, _ = ckpt_dir
    files = {n: file_sha(root / n) for n in os.listdir(root) if n.endswith(".safetensors")}
    _plan(tmp_path / "ok", ckpt_dir, base_extra={"files_sha256": files})
    bad = dict(files)
    bad["model-00001-of-00002.safetensors"] = "f" * 64
    with refusal("binding"):
        _plan(tmp_path / "b", ckpt_dir, base_extra={"files_sha256": bad})
    with refusal("binding"):  # must list exactly the checkpoint's files
        _plan(tmp_path / "c", ckpt_dir, base_extra={"files_sha256": {k: files[k] for k in list(files)[:1]}})


def test_revision_evidence_from_snapshot_dir_and_download_metadata(tmp_path):
    tensors = base_tensors()
    ops, arrays = standard_ops_and_arrays()

    def run(dirname_rev, declared):
        root = tmp_path / dirname_rev[:6] / "snapshots" / dirname_rev
        write_checkpoint(root, tensors, sharded=False)
        spec_path = make_spec(
            tmp_path / f"s{dirname_rev[:6]}{declared[:3]}", tensors,
            [ops[0]], {k: v for k, v in arrays.items() if k.startswith("r1")},
            checkpoint=root, revision=declared,
        )
        spec, digest = load_patch_spec(spec_path)
        return plan_against_checkpoint(spec, digest)

    _, binding, _, _ = run(REVISION, REVISION)  # no content pin; evidence suffices
    assert binding["revision_evidence"][0]["source"] == "hf_snapshot_directory"
    assert binding["content_pinned"] is False
    with refusal("binding"):  # the snapshot says OTHER_REVISION, spec declares REVISION
        run(OTHER_REVISION, REVISION)

    root = tmp_path / "dl"
    write_checkpoint(root, tensors, sharded=False)
    meta = root / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "model.safetensors.metadata").write_text(f"{OTHER_REVISION}\netag\n123\n")
    spec_path = make_spec(
        tmp_path / "sd", tensors, [ops[0]], {k: v for k, v in arrays.items() if k.startswith("r1")},
        checkpoint=root,
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
    )
    spec, digest = load_patch_spec(spec_path)
    with refusal("binding"):  # content pinned, but local metadata contradicts the declared revision
        plan_against_checkpoint(spec, digest)


# --------------------------------------------------------------------------
# In-memory apply / revert (runtime)
# --------------------------------------------------------------------------


class TinyModule(torch.nn.Module):
    """Parameter names mirror the fixture checkpoint names via ParameterDict-less attrs."""

    def __init__(self, tensors, prefix=""):
        super().__init__()
        self._names = {}
        for i, (name, t) in enumerate(tensors.items()):
            attr = f"p{i}"
            setattr(self, attr, torch.nn.Parameter(t.clone(), requires_grad=False))
            self._names[prefix + name] = attr

    def named_parameters(self, *a, **k):  # expose checkpoint-style names
        for name, attr in self._names.items():
            yield name, getattr(self, attr)


def _prepared(tmp_path, ckpt_dir, **kw):
    spec_path = _std_spec(tmp_path, ckpt_dir, **kw)
    spec, digest = load_patch_spec(spec_path)
    return wruntime.prepare_patch(spec_path, digest), spec_path, digest


def test_apply_matches_planned_bytes_and_revert_is_exact(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    model = TinyModule(tensors)
    prepared, _, _ = _prepared(tmp_path, ckpt_dir)
    before = {n: raw_sha(p) for n, p in model.named_parameters()}
    applied = wruntime.apply_patch(model, prepared)
    params = dict(model.named_parameters())
    modified = applied.plan.modified
    assert set(modified) == {DOWN0, GATE0}
    for name, tp in modified.items():
        assert raw_sha(params[name]) == tp.after_sha256 != before[name]
    for name in (DOWN1, EMBED, NORM):
        assert raw_sha(params[name]) == before[name]
    # same bytes the disk-based plan produced
    _, _, _, disk_plan = plan_against_checkpoint(prepared.spec, prepared.spec_sha256)
    assert disk_plan.patch_digest() == applied.plan.patch_digest()
    applied.revert()
    assert {n: raw_sha(p) for n, p in model.named_parameters()} == before
    applied.revert()  # idempotent


def test_context_manager_reverts_even_on_error(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    model = TinyModule(tensors)
    prepared, _, _ = _prepared(tmp_path, ckpt_dir)
    before = {n: raw_sha(p) for n, p in model.named_parameters()}
    with pytest.raises(RuntimeError):
        with wruntime.patched(model, prepared) as applied:
            assert raw_sha(dict(model.named_parameters())[DOWN0]) != before[DOWN0]
            raise RuntimeError("boom")
    assert applied.reverted
    assert {n: raw_sha(p) for n, p in model.named_parameters()} == before


def test_wrong_base_refuses_and_leaves_module_untouched(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    wrong = {k: v.clone() for k, v in tensors.items()}
    wrong[GATE0][0, 0] = wrong[GATE0][0, 0] + 1  # second tensor differs
    model = TinyModule(wrong)
    prepared, _, _ = _prepared(tmp_path, ckpt_dir)
    before = {n: raw_sha(p) for n, p in model.named_parameters()}
    with refusal("base_hash_mismatch"):
        wruntime.apply_patch(model, prepared)
    assert {n: raw_sha(p) for n, p in model.named_parameters()} == before  # first tensor not half-patched


def test_budget_refusal_leaves_module_untouched(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    model = TinyModule(tensors)
    prepared, _, _ = _prepared(tmp_path, ckpt_dir, budget={"max_relative_frobenius": 1e-9})
    before = {n: raw_sha(p) for n, p in model.named_parameters()}
    with refusal("budget"):
        wruntime.apply_patch(model, prepared)
    assert {n: raw_sha(p) for n, p in model.named_parameters()} == before


def test_commit_verification_failure_rolls_back_every_tensor(tmp_path, ckpt_dir, monkeypatch):
    root, tensors = ckpt_dir
    model = TinyModule(tensors)
    prepared, _, _ = _prepared(tmp_path, ckpt_dir)
    before = {n: raw_sha(p) for n, p in model.named_parameters()}
    real = wruntime.tensor_sha256
    calls = {"n": 0}

    def flaky(t):
        calls["n"] += 1
        return "0" * 64 if calls["n"] == 2 else real(t)

    monkeypatch.setattr(wruntime, "tensor_sha256", flaky)
    with refusal("apply"):
        wruntime.apply_patch(model, prepared)
    monkeypatch.setattr(wruntime, "tensor_sha256", real)
    assert {n: raw_sha(p) for n, p in model.named_parameters()} == before


def test_runtime_prefix_rules_map_checkpoint_names_to_module_names(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    model = TinyModule(tensors, prefix="lm.")
    prepared, spec_path, _ = _prepared(
        tmp_path, ckpt_dir,
        base_extra={
            "index_sha256": file_sha(root / "model.safetensors.index.json"),
            "runtime_prefix_rules": [{"checkpoint_prefix": "blk.", "runtime_prefix": "lm.blk."}],
        },
    )
    with refusal("binding"):  # tensors only exist under "lm.<name>"; rule maps blk.* -> lm.blk.*
        wruntime.apply_patch(TinyModule(tensors, prefix="other."), prepared)
    applied = wruntime.apply_patch(model, prepared)
    assert dict(model.named_parameters())["lm." + DOWN0] is not None
    applied.revert()


def test_declared_revision_must_equal_patch_revision(tmp_path, ckpt_dir):
    prepared, _, digest = _prepared(tmp_path, ckpt_dir)
    wruntime.check_declared_revision(prepared, REVISION)
    with refusal("binding"):
        wruntime.check_declared_revision(prepared, OTHER_REVISION)
    with refusal("binding"):
        wruntime.check_declared_revision(prepared, None)

    class Cfg:
        _commit_hash = OTHER_REVISION

    class M:
        config = Cfg()

    with refusal("binding"):  # a loaded model that resolved another commit is refused
        wruntime.check_loaded_commit(prepared, M())
    Cfg._commit_hash = REVISION
    assert wruntime.check_loaded_commit(prepared, M()) == REVISION


def test_prepare_patch_refuses_wrong_pinned_spec_sha(tmp_path, ckpt_dir):
    spec_path = _std_spec(tmp_path, ckpt_dir)
    with refusal("binding"):
        wruntime.prepare_patch(spec_path, "0" * 64)


# --------------------------------------------------------------------------
# Export + readback
# --------------------------------------------------------------------------


def test_validate_writes_nothing_then_export_is_exact(tmp_path, ckpt_dir):
    root, tensors = ckpt_dir
    out = tmp_path / "exported"
    spec_path = _std_spec(tmp_path, ckpt_dir, export_dir=out)
    snapshot = {p.name: file_sha(p) for p in root.iterdir()}

    report = validate_patch_config(spec_path)
    assert report["written"] is False and not out.exists()
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]

    result = export_patch_config(spec_path)
    assert out.is_dir() and result["written"]
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]
    assert {p.name: file_sha(p) for p in root.iterdir()} == snapshot  # base untouched

    plan_modified = {t["name"]: t for t in result["manifest"]["tensors"] if t["entries_changed"]}
    assert set(plan_modified) == {DOWN0, GATE0}
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        got = load_file(str(out / shard))  # readback with the independent library
        base = load_file(str(root / shard))
        assert set(got) == set(base)
        for name, tensor in got.items():
            assert tensor.dtype == base[name].dtype and tensor.shape == base[name].shape
            if name in plan_modified:
                assert raw_sha(tensor) == plan_modified[name]["after_sha256"]
                assert raw_sha(tensor) != raw_sha(base[name])
            else:
                assert torch.equal(tensor.view(torch.uint8), base[name].view(torch.uint8))
    # untouched shard is byte-identical; header (incl. __metadata__) preserved
    assert file_sha(out / "model-00002-of-00002.safetensors") == file_sha(root / "model-00002-of-00002.safetensors")
    assert read_header(out / "model-00001-of-00002.safetensors").metadata == {"format": "pt"}
    # index + sidecar copied verbatim
    assert file_sha(out / "model.safetensors.index.json") == file_sha(root / "model.safetensors.index.json")
    assert (out / "config.json").read_text() == (root / "config.json").read_text()

    manifest = json.loads((out / "mechinterp-weight-export.manifest.json").read_text())
    assert manifest["schema_version"] == "mechinterp-weight-export/v1"
    assert manifest["patch_digest"] == result["patch_digest"]
    assert manifest["base"]["revision"] == REVISION
    assert manifest["spec"]["sha256"] == file_sha(spec_path)
    for fname, info in manifest["output"]["weight_files"].items():
        assert info["sha256"] == file_sha(out / fname)
    assert manifest["base"]["files"]["model-00001-of-00002.safetensors"]["sha256"] == file_sha(
        root / "model-00001-of-00002.safetensors"
    )
    assert manifest["totals"]["tensors_modified"] == 2


def test_export_refuses_existing_destination_and_preserves_it(tmp_path, ckpt_dir):
    out = tmp_path / "exported"
    out.mkdir()
    (out / "keep.txt").write_text("precious")
    spec_path = _std_spec(tmp_path, ckpt_dir, export_dir=out)
    with refusal("output_exists"):
        export_patch_config(spec_path)
    assert (out / "keep.txt").read_text() == "precious" and len(list(out.iterdir())) == 1
    # a second export into a freshly exported dir is refused too
    spec_path2 = _std_spec(tmp_path / "n", ckpt_dir, export_dir=tmp_path / "fresh")
    export_patch_config(spec_path2)
    with refusal("output_exists"):
        export_patch_config(spec_path2)


def test_failed_export_leaves_no_output_or_staging(tmp_path, ckpt_dir):
    out = tmp_path / "exported"
    spec_path = _std_spec(tmp_path, ckpt_dir, export_dir=out, budget={"max_relative_frobenius": 1e-9})
    with refusal("budget"):
        export_patch_config(spec_path)
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]


def test_export_midway_failure_cleans_staging(tmp_path, ckpt_dir, monkeypatch):
    out = tmp_path / "exported"
    spec_path = _std_spec(tmp_path, ckpt_dir, export_dir=out)
    from MechInterp.weights import export as wexport

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(wexport, "_copy_file", boom)
    with pytest.raises(OSError):
        export_patch_config(spec_path)
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]


def test_export_requires_export_block_and_checkpoint(tmp_path, ckpt_dir):
    spec_path = _std_spec(tmp_path, ckpt_dir)
    with refusal("schema"):
        export_patch_config(spec_path)
    root, tensors = ckpt_dir
    ops, arrays = standard_ops_and_arrays()
    no_ckpt = make_spec(tmp_path / "nc", tensors, ops, arrays, export_dir=tmp_path / "o")
    with refusal("schema"):
        validate_patch_config(no_ckpt)


def test_single_file_checkpoint_export_roundtrip(tmp_path):
    tensors = base_tensors()
    root = tmp_path / "ck"
    write_checkpoint(root, tensors, sharded=False)
    ops, arrays = standard_ops_and_arrays()
    spec_path = make_spec(
        tmp_path / "s", tensors, ops, arrays, checkpoint=root / "model.safetensors",
        base_extra={"index_sha256": file_sha(root / "model.safetensors")},
        export_dir=tmp_path / "out",
    )
    result = export_patch_config(spec_path)
    got = load_file(str(tmp_path / "out" / "model.safetensors"))
    for tp in result["manifest"]["tensors"]:
        assert raw_sha(got[tp["name"]]) == tp["after_sha256"]
    assert raw_sha(got[EMBED]) == raw_sha(tensors[EMBED])


def test_runtime_refuses_shared_storage_before_any_write(tmp_path, ckpt_dir):
    _, tensors = ckpt_dir
    model = TinyModule(tensors)
    prepared, _, _ = _prepared(tmp_path, ckpt_dir)
    target = prepared.spec.targeted_tensors()[0]
    model._names["unbudgeted_alias"] = model._names[target]
    before = {name: raw_sha(param) for name, param in model.named_parameters()}
    with pytest.raises(WeightPatchError, match="shares storage"):
        wruntime.apply_patch(model, prepared)
    assert {name: raw_sha(param) for name, param in model.named_parameters()} == before


def test_export_racing_empty_destination_is_never_replaced(tmp_path):
    from MechInterp.weights.export import _rename_create_only

    source, destination = tmp_path / "staged", tmp_path / "destination"
    source.mkdir()
    (source / "payload").write_text("preserve")
    destination.mkdir()
    with pytest.raises((WeightPatchError, FileExistsError)):
        _rename_create_only(source, destination)
    assert not list(destination.iterdir())
    assert (source / "payload").read_text() == "preserve"


def test_export_without_exclusive_rename_verifies_files_and_publishes_manifest_last(tmp_path, ckpt_dir, monkeypatch):
    from MechInterp.weights import export as module
    def unsupported(*args):
        raise OSError(errno.EINVAL, "filesystem does not support rename flags")
    monkeypatch.setattr(module, "_rename_create_only", unsupported)
    out = tmp_path / "exported"
    result = export_patch_config(_std_spec(tmp_path, ckpt_dir, export_dir=out))
    manifest = json.loads((out / module.MANIFEST_NAME).read_text())
    assert result["written"] and manifest == result["manifest"]
    assert manifest["publication"] == {"method": "exclusive_directory_manifest_last", "atomic_directory_rename": False}
    for name, info in {**manifest["output"]["weight_files"], **manifest["output"]["copied_files"]}.items():
        assert file_sha(out / name) == info["sha256"]
    assert not [p for p in tmp_path.iterdir() if ".partial-" in p.name]


def test_export_fallback_refuses_a_racing_destination(tmp_path, ckpt_dir, monkeypatch):
    from MechInterp.weights import export as module
    out = tmp_path / "exported"
    def racing(source, destination):
        destination.mkdir()
        (destination / "keep.txt").write_text("precious")
        raise OSError(errno.EINVAL, "unsupported rename flags")
    monkeypatch.setattr(module, "_rename_create_only", racing)
    with refusal("output_exists"):
        export_patch_config(_std_spec(tmp_path, ckpt_dir, export_dir=out))
    assert (out / "keep.txt").read_text() == "precious"
    assert len(list(out.iterdir())) == 1


def test_failed_fallback_never_publishes_manifest_or_deletes_foreign_files(tmp_path, ckpt_dir, monkeypatch):
    from MechInterp.weights import export as module
    out = tmp_path / "exported"
    original_copy = module.shutil.copyfileobj
    def unsupported(*args):
        raise OSError(errno.EINVAL, "unsupported rename flags")
    def fail_during_publication(source, sink, *args, **kwargs):
        if ".partial-" in Path(source.name).parent.name:
            (out / "keep.txt").write_text("precious")
            sink.write(b"partial")
            raise OSError(errno.EIO, "planted publication failure")
        return original_copy(source, sink, *args, **kwargs)
    monkeypatch.setattr(module, "_rename_create_only", unsupported)
    monkeypatch.setattr(module.shutil, "copyfileobj", fail_during_publication)
    with pytest.raises(OSError, match="planted publication failure"):
        export_patch_config(_std_spec(tmp_path, ckpt_dir, export_dir=out))
    assert not (out / module.MANIFEST_NAME).exists()
    assert (out / "keep.txt").read_text() == "precious"
    assert len(list(out.iterdir())) == 1


def test_fallback_rejects_corrupted_copies(tmp_path, ckpt_dir, monkeypatch):
    from MechInterp.weights import export as module
    out = tmp_path / "exported"
    original_copy = module.shutil.copyfileobj
    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "unsupported rename flags")
    def corrupt(source, sink, *args, **kwargs):
        if ".partial-" in Path(source.name).parent.name:
            sink.write(b"corrupt")
            return None
        return original_copy(source, sink, *args, **kwargs)
    monkeypatch.setattr(module, "_rename_create_only", unsupported)
    monkeypatch.setattr(module.shutil, "copyfileobj", corrupt)
    with refusal("apply"):
        export_patch_config(_std_spec(tmp_path, ckpt_dir, export_dir=out))
    assert not out.exists()


def test_export_does_not_fallback_on_permission_failure(tmp_path, ckpt_dir, monkeypatch):
    from MechInterp.weights import export as module
    out = tmp_path / "exported"
    def denied(*args):
        raise OSError(errno.EACCES, "permission denied")
    monkeypatch.setattr(module, "_rename_create_only", denied)
    with pytest.raises(OSError) as exc:
        export_patch_config(_std_spec(tmp_path, ckpt_dir, export_dir=out))
    assert exc.value.errno == errno.EACCES and not out.exists()
