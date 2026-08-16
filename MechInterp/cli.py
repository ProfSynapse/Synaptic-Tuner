"""
CLI layer for the MechInterp verbs.

This is the model-facing glue that turns a validated recipe into a run: it loads
the model, resolves arms and readouts, registers the intervention hook, generates
per row, grades, and writes per-row provenance plus a manifest. The lane-agnostic
logic (arm resolution, resume, smoke gating, config sha) lives in cell.py and is
imported here so this layer stays thin.

Verbs:
  extract     generate + capture hidden states to safetensors + manifest.
  probe_fit   fit a linear readout from extracted activations and freeze it.
  steer       run the six-block steer cell (smoke-gated).
  dose_calibrate run a resumable dose ladder over frozen readouts.
  score_gates evaluate a gates.yaml against a per-row output JSONL.

The steer and extract verbs touch a GPU; they refuse to run without an explicit
acknowledgement flag so a recipe never surprises a shared machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tuner.project import ProjectContext

from MechInterp import cell as cell_mod
from MechInterp.config import (
    DoseCalibrationConfig,
    SteerCellConfig,
    ExtractConfig,
    ProbeFitConfig,
    load_dose_calibration_config,
    load_steer_config,
    load_extract_config,
    load_probe_fit_config,
)


# --------------------------------------------------------------------------
# Model loading (self-contained; handles the transformers 5.x dtype rename)
# --------------------------------------------------------------------------


def _load_model_and_tokenizer(model_name: str, adapter: Optional[str] = None, revision: Optional[str] = None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    major = int(transformers.__version__.split(".")[0])
    dtype_kwarg = {"dtype": torch.bfloat16} if major >= 5 else {"torch_dtype": torch.bfloat16}
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, token=token, device_map="auto", **dtype_kwarg
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def _require_gpu_ack(ack: bool) -> Optional[int]:
    if not ack:
        print(
            "Refusing to run a GPU verb without --i-know-this-runs-on-gpu. "
            "This verb loads a model and runs generation."
        )
        return 2
    return None


def _load_callable(spec: str) -> Callable:
    import importlib

    module_path, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_path), attr)


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


def run_extract(config: ExtractConfig, model_name: str, revision: Optional[str], adapter: Optional[str], gpu_ack: bool) -> int:
    guard = _require_gpu_ack(gpu_ack)
    if guard is not None:
        return guard
    from MechInterp.extraction import PositionSpec, extract_rows

    rows = cell_mod.load_jsonl(config.rows_path)
    render_fn = _load_callable(config.render_fn)
    content_end_fn = _load_callable(config.content_end_fn)
    model, tokenizer = _load_model_and_tokenizer(model_name, adapter, revision)
    spec = PositionSpec(
        families=config.families, every_k=config.every_k, layers=config.layers
    )
    manifest = extract_rows(
        model,
        tokenizer,
        rows,
        render_fn=render_fn,
        content_end_fn=content_end_fn,
        spec=spec,
        out_dir=config.output_dir,
        max_new_tokens=config.max_new_tokens,
    )
    print(
        f"Extracted {manifest['n_answered']}/{manifest['n_rows']} answered rows "
        f"to {config.output_dir}"
    )
    return 0


# --------------------------------------------------------------------------
# probe_fit
# --------------------------------------------------------------------------


def _load_activation_matrix(activations_dir: str, family: str, labels: list[dict]):
    """Assemble (X, y, layer_keys) from extracted safetensors for one layer sweep.

    Returns a dict {layer_index: X} plus aligned y, using each label row's
    row_key to find its safetensors file. Only rows with a captured file for the
    family are included.
    """
    from safetensors.torch import load_file

    act_dir = Path(activations_dir)
    per_layer: dict[int, list] = {}
    y = []
    for rec in labels:
        rk = str(rec.get("row_key", rec.get("id")))
        safe_key = rk.replace("::", "__").replace("|", "_").replace("/", "_")
        f = act_dir / f"{safe_key}__{family}.safetensors"
        if not f.exists():
            continue
        tensors = load_file(str(f))
        for key, t in tensors.items():
            li = int(key[1:])  # "L23" -> 23
            per_layer.setdefault(li, []).append(t[0].numpy())  # first captured pos
        y.append(int(rec["label"]))
    matrices = {li: np.stack(vals, axis=0) for li, vals in per_layer.items()}
    return matrices, np.asarray(y, dtype=int)


def run_probe_fit(config: ProbeFitConfig) -> int:
    from MechInterp.probe import sweep_layers, freeze_direction

    labels = cell_mod.load_jsonl(config.labels_path)
    matrices, y = _load_activation_matrix(
        config.activations_path, config.position_family, labels
    )
    if not matrices:
        print("No activation matrices found for the requested family.")
        return 1
    clf_kwargs = {"solver": config.solver, "tol": config.tol, "C": config.C}
    sweep = sweep_layers(
        matrices,
        y,
        n_components=config.n_components,
        n_splits=config.n_splits,
        seed=config.seed,
        **clf_kwargs,
    )
    best_layer = sweep["best_layer"]
    record = freeze_direction(
        matrices[best_layer],
        y,
        layer=best_layer,
        out_path=config.output_direction,
        n_components=config.n_components,
        seed=config.seed,
        normalize=config.normalize,
        provenance={
            "activations_path": config.activations_path,
            "position_family": config.position_family,
            "auroc_by_layer": sweep["auroc_by_layer"],
        },
        **clf_kwargs,
    )
    print(
        f"Froze direction at layer {best_layer} "
        f"(AUROC {sweep['auroc_by_layer'][best_layer]:.4f}) -> {config.output_direction}"
    )
    return 0


# --------------------------------------------------------------------------
# steer (six-block cell)
# --------------------------------------------------------------------------


def _direction_tensor(readout_record: dict):
    import torch

    return torch.tensor(readout_record["vector_np"], dtype=torch.float32)


# Keys _run_one_pass / _run_batch compute themselves; a same-named pool-row
# field is never allowed to shadow them (see _passthrough_fields).
_COMPUTED_RECORD_KEYS = frozenset(
    {
        "row_key",
        "strength",
        "active",
        "answer_text",
        "prompt_len",
        "n_new_tokens",
        "terminated_naturally",
    }
)


def _passthrough_fields(row: dict) -> dict:
    """Pool-row metadata to carry into the output record, verbatim.

    Without this, the record _run_one_pass/_run_batch return is limited to the
    fields those passes compute, so a grader (and mechinterp score-gates,
    grouping by a project-chosen field like a class label) can only ever see
    row_key/strength/active/answer_text/prompt_len -- never the pool row's own
    metadata (e.g. gold-answer aliases or a class/cell label), no matter what
    the recipe's rows_path actually puts on each row. Carrying every non-
    internal field through keeps the grader interface generic: a project's
    rows dictate what a project's grader can read, with no tuner-side
    allowlist of "known" field names.

    Excludes underscore-prefixed keys (internal bookkeeping set by
    cell.pending_rows, e.g. _strength/_active) and the keys the pass itself
    computes, so a pool row can never shadow those via a same-named field.
    """
    return {
        k: v
        for k, v in row.items()
        if not k.startswith("_") and k not in _COMPUTED_RECORD_KEYS
    }


def _redact_record_fields(value, redact_fields: set[str]):
    if not redact_fields:
        return value
    if isinstance(value, dict):
        return {
            k: _redact_record_fields(v, redact_fields)
            for k, v in value.items()
            if k not in redact_fields
        }
    if isinstance(value, list):
        return [_redact_record_fields(v, redact_fields) for v in value]
    return value


def _write_checkpoint_record(handle, rec: dict, redact_fields: set[str] | None = None) -> None:
    rec = _redact_record_fields(rec, redact_fields or set())
    handle.write(json.dumps(rec) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
def _eos_token_ids(tokenizer, generation) -> set[int]:
    ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.add(int(eos_id))
    for token in getattr(generation, "extra_eos_tokens", []) or []:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tok_id, int) and tok_id >= 0 and tok_id != unk_id:
            ids.add(int(tok_id))
    return ids


def _generation_kwargs(tokenizer, generation) -> dict:
    kwargs = {
        "max_new_tokens": generation.max_new_tokens,
        "min_new_tokens": generation.min_new_tokens,
        "do_sample": generation.do_sample,
        "num_beams": 1,
        "return_dict_in_generate": True,
    }
    if generation.do_sample:
        kwargs["temperature"] = generation.temperature
        kwargs["top_p"] = generation.top_p
    eos_ids = sorted(_eos_token_ids(tokenizer, generation))
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        kwargs["pad_token_id"] = int(pad_id)
    return kwargs


def _continuation_record(tokenizer, continuation, generation) -> dict:
    token_ids = [int(t) for t in continuation.detach().cpu().tolist()]
    eos_ids = _eos_token_ids(tokenizer, generation)
    stop_len = len(token_ids)
    if eos_ids:
        for idx, tok_id in enumerate(token_ids):
            if tok_id in eos_ids:
                stop_len = idx + 1
                break
    effective_ids = token_ids[:stop_len]
    text = tokenizer.decode(effective_ids, skip_special_tokens=True)
    return {
        "answer_text": text,
        "n_new_tokens": int(stop_len),
        "terminated_naturally": bool(stop_len < generation.max_new_tokens),
    }


def _run_one_pass(
    model,
    tokenizer,
    controller,
    row: dict,
    generation,
    generation_mode: str,
    render_fn: Callable,
) -> dict:
    import torch

    prompt = render_fn(row)
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    strength = float(row.get("_strength", 0.0))
    is_active = bool(row.get("_active", False))
    # Engagement is decided by _active (set centrally in cell.pending_rows), not
    # by strength != 0: a force_active arm (ablate) must still engage the
    # controller at strength 0.0, and force_active also tells the hook to write
    # that zero setpoint rather than skip the row as a no-op.
    controller.begin_pass(
        generation_mode if is_active else "off",
        strength,
        attention_mask=enc["attention_mask"],
        force_active=is_active,
    )
    gen = model.generate(
        **enc,
        **_generation_kwargs(tokenizer, generation),
    )
    controller.reset()
    full = gen.sequences[0]
    prompt_len = enc["input_ids"].shape[1]
    generated = _continuation_record(tokenizer, full[prompt_len:], generation)
    return {
        **_passthrough_fields(row),
        "row_key": cell_mod.row_key_of(row),
        "strength": strength,
        "active": bool(row.get("_active", False)),
        **generated,
        "prompt_len": int(prompt_len),
    }


def _run_batch(
    model,
    tokenizer,
    controller,
    rows: list[dict],
    generation,
    generation_mode: str,
    render_fn: Callable,
) -> list[dict]:
    """Batched counterpart of _run_one_pass: run a whole chunk of rows through
    one model.generate() call instead of one row at a time.

    The batch is encoded with LEFT padding, restoring the tokenizer's original
    padding_side afterward. Left padding right-aligns every row's real prompt
    tokens, which matters for two things: the prefill step's shared "anchor"
    column (hard-coded to seq_len - 1 by the controller, see hooks.py) lands on
    every row's true last prompt token instead of a pad token, and every row's
    generated continuation starts at the same column -- the batch's padded
    prompt length -- so no per-row slice offset is needed to cut it out of the
    generated sequence (the padding tokens all sit before that column).

    Per-row strength and force_active are passed to the controller as
    length-batch tensors (see _strength_per_row and _resolve_force_active in
    hooks.py), so a single batch may mix active and inactive rows: an inactive
    row's strength is 0 and its force_active entry is False, which is a true
    no-op under both intervention laws (additive: 0 * direction changes
    nothing; erase_write: the per-row active mask built from strength/override
    excludes it from the edit entirely). The whole pass runs under mode="off"
    only when every row in the chunk is inactive, matching _run_one_pass's
    per-row "off" mode exactly for a batch of size 1.

    Sampling caveat: generation.do_sample=True is not covered by the batching
    equivalence guarantee -- torch's global RNG stream is consumed in a
    different order depending on batch shape, so a sampled batch is not
    expected to reproduce a sampled single-row run token-for-token. Every
    steer cell's default (and the correctness tests) use greedy decoding
    (do_sample=False), where this does not apply.
    """
    import torch

    device = next(model.parameters()).device
    prompts = [render_fn(row) for row in rows]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_padding_side
    enc = {k: v.to(device) for k, v in enc.items()}

    strengths = torch.tensor(
        [float(row.get("_strength", 0.0)) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    active = torch.tensor(
        [bool(row.get("_active", False)) for row in rows],
        dtype=torch.bool,
        device=device,
    )
    mode = generation_mode if bool(active.any()) else "off"
    controller.begin_pass(
        mode,
        strengths,
        attention_mask=enc["attention_mask"],
        force_active=active,
    )
    gen = model.generate(
        **enc,
        **_generation_kwargs(tokenizer, generation),
    )
    controller.reset()

    padded_prompt_len = enc["input_ids"].shape[1]
    row_prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    records = []
    for i, row in enumerate(rows):
        generated = _continuation_record(
            tokenizer, gen.sequences[i, padded_prompt_len:], generation
        )
        records.append(
            {
                **_passthrough_fields(row),
                "row_key": cell_mod.row_key_of(row),
                "strength": float(row.get("_strength", 0.0)),
                "active": bool(row.get("_active", False)),
                **generated,
                "prompt_len": int(row_prompt_lens[i]),
            }
        )
    return records


def run_steer(
    config: SteerCellConfig,
    model_name: str,
    revision: Optional[str],
    adapter: Optional[str],
    render_fn_spec: str,
    gpu_ack: bool,
    force: bool = False,
    project_context: ProjectContext | None = None,
) -> int:
    guard = _require_gpu_ack(gpu_ack)
    if guard is not None:
        return guard
    import torch

    from MechInterp.intervention import InterventionHook, GenerationInterventionController, get_decoder_layer
    from MechInterp.probe import load_frozen_direction
    from MechInterp.grading import load_grader

    config_sha = cell_mod.compute_config_sha(config)
    if config.surface.expected_config_sha and config.surface.expected_config_sha != config_sha:
        print(
            f"Config sha mismatch: expected {config.surface.expected_config_sha}, "
            f"got {config_sha}. Aborting."
        )
        return 3

    rows = cell_mod.load_jsonl(config.surface.rows_path)
    render_fn = _load_callable(render_fn_spec)
    readout_by_name = {
        r.name: load_frozen_direction(r.path) for r in config.readouts
    }
    active_readout = readout_by_name[config.law.readout]
    layer = config.law.layer if config.law.layer is not None else active_readout["layer"]

    model, tokenizer = _load_model_and_tokenizer(model_name, adapter, revision)
    direction = _direction_tensor(active_readout)
    sigma = float(active_readout.get("sigma", 1.0))
    hook = InterventionHook(
        law=config.law.kind,
        direction=direction,
        sigma=sigma,
        position=config.law.position,
    )
    controller = GenerationInterventionController(hook)
    layer_module = get_decoder_layer(model, layer)
    handle = layer_module.register_forward_hook(controller)

    grader = (
        load_grader(config.execution.grader, project_context=project_context)
        if config.execution.grader
        else None
    )

    # Smoke gate: run n_rows through the intervention with readback and record.
    if not force and not cell_mod.smoke_passed(config.execution.output_path, config_sha):
        readback = _run_smoke(
            model, tokenizer, config, rows, render_fn, direction, sigma, layer
        )
        verdict = cell_mod.evaluate_smoke_readback(readback, config.smoke)

        # A gen_stream cell additionally needs proof that the decode hook fires
        # during a real generate() call: the forward-only readback above never
        # exercises the decode loop, so it cannot catch a decode hook that is
        # wired up but never actually invoked (see gen_stream_fires below).
        # Skip this on anchor/off cells, whose behavior is unchanged by this
        # guard.
        gen_stream_fired = None
        if verdict["passed"] and config.law.generation_mode == "gen_stream":
            smoke_enc = tokenizer(render_fn(rows[0]), return_tensors="pt").to(
                next(model.parameters()).device
            )
            gen_stream_fired = gen_stream_fires(
                model,
                controller,
                smoke_enc,
                strength=config.smoke.gen_stream_probe_strength
                or _GEN_STREAM_SMOKE_STRENGTH,
            )
            if not gen_stream_fired:
                verdict["passed"] = False
        verdict["gen_stream_fired"] = gen_stream_fired

        readback["passed"] = verdict["passed"]
        cell_mod.record_smoke(config.execution.output_path, config_sha, {**readback, **verdict})
        if not verdict["passed"]:
            handle.remove()
            if gen_stream_fired is False:
                print(
                    "gen_stream decode hook did not fire -- check that the model "
                    "routes generate() through the hooked module's forward() on "
                    "every decode step. An optimized cached-generation path that "
                    "bypasses the hooked module's forward() (as with Unsloth's "
                    "FastLanguageModel.for_inference) will silently produce this "
                    "failure mode."
                )
            print(f"Smoke failed: {verdict}. Full arms not run. Fix or pass --force.")
            return 4
        print(f"Smoke passed: {verdict}")

    strengths_by_arm = cell_mod.resolve_all_arms(config, rows)
    arm_summaries = {}
    out_path = Path(config.execution.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as out:
        redact_fields = set(config.execution.redact_fields)
        for arm in config.arms:
            strengths = strengths_by_arm[arm.name]
            # A gain_field arm always writes at its computed value even when
            # that value is exactly 0.0 (the couple law with g==0 IS the
            # ablate arm, per the amendment); force_active on the ArmConfig is
            # the same "apply-at-zero" intent for a plain fixed-strength arm.
            write_at_zero = bool(arm.force_active) or arm.gain_field is not None
            pending = cell_mod.pending_rows(
                rows, strengths, arm.name, out_path, config.execution.resume,
                write_at_zero=write_at_zero,
            )
            arm_summaries[arm.name] = {"n_active": len(strengths), "n_pending": len(pending)}
            batch_size = config.execution.batch_size
            if batch_size <= 1:
                for row in pending:
                    rec = _run_one_pass(
                        model, tokenizer, controller, row, config.surface.generation,
                        config.law.generation_mode, render_fn,
                    )
                    rec["arm"] = arm.name
                    if grader is not None:
                        rec.update(grader(rec))
                    _write_checkpoint_record(out, rec, redact_fields)
            else:
                for i in range(0, len(pending), batch_size):
                    chunk = pending[i : i + batch_size]
                    recs = _run_batch(
                        model, tokenizer, controller, chunk, config.surface.generation,
                        config.law.generation_mode, render_fn,
                    )
                    for rec in recs:
                        rec["arm"] = arm.name
                        if grader is not None:
                            rec.update(grader(rec))
                        _write_checkpoint_record(out, rec, redact_fields)

    handle.remove()
    cell_mod.write_manifest(out_path, config, config_sha, arm_summaries)
    print(f"Steer cell complete. Output {out_path}, config_sha {config_sha}")
    return 0


# --------------------------------------------------------------------------
# dose_calibrate
# --------------------------------------------------------------------------


def _target_readout_names(config: DoseCalibrationConfig) -> list[str]:
    if config.law.readout == "*":
        return [readout.name for readout in config.readouts]
    return [config.law.readout]


def _strength_for_dose(config: DoseCalibrationConfig, dose: float, sigma: float) -> float:
    if config.calibration.dose_kind == "strength":
        return float(dose)
    if config.law.kind == "erase_write":
        return float(dose) / float(sigma or 1.0)
    return float(dose)


def _readback_for_record(readback: dict | None, row_index: int) -> dict:
    if not readback:
        return {}
    active_rows = [int(i) for i in readback.get("active_rows", [])]
    if row_index not in active_rows:
        return {
            "readback_offtarget_abs_max": readback.get("offtarget_abs_max"),
            "readback_offtarget_abs_mean": readback.get("offtarget_abs_mean"),
        }
    idx = active_rows.index(row_index)
    commanded = readback.get("commanded", [])
    measured = readback.get("measured", [])
    return {
        "readback_commanded": commanded[idx] if idx < len(commanded) else None,
        "readback_measured": measured[idx] if idx < len(measured) else None,
        "readback_offtarget_abs_max": readback.get("offtarget_abs_max"),
        "readback_offtarget_abs_mean": readback.get("offtarget_abs_mean"),
    }


def run_dose_calibration(
    config: DoseCalibrationConfig,
    model_name: str,
    adapter: Optional[str],
    render_fn_spec: str,
    gpu_ack: bool,
    project_context: ProjectContext | None = None,
) -> int:
    guard = _require_gpu_ack(gpu_ack)
    if guard is not None:
        return guard
    import torch

    from MechInterp.intervention import (
        GenerationInterventionController,
        InterventionHook,
        get_decoder_layer,
    )
    from MechInterp.grading import load_grader
    from MechInterp.probe import load_frozen_direction

    config_sha = cell_mod.compute_config_sha(config)
    if (
        config.surface.expected_config_sha
        and config.surface.expected_config_sha != config_sha
    ):
        print(
            f"Config sha mismatch: expected {config.surface.expected_config_sha}, "
            f"got {config_sha}. Aborting."
        )
        return 3

    rows = cell_mod.load_jsonl(config.surface.rows_path)
    render_fn = _load_callable(render_fn_spec)
    grader = (
        load_grader(config.execution.grader, project_context=project_context)
        if config.execution.grader
        else None
    )
    readout_by_name = {
        readout.name: load_frozen_direction(readout.path)
        for readout in config.readouts
    }

    model, tokenizer = _load_model_and_tokenizer(model_name, adapter)
    out_path = Path(config.execution.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(config.execution.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    run_summaries = {}
    with open(out_path, "a") as out:
        redact_fields = set(config.execution.redact_fields)
        for readout_name in _target_readout_names(config):
            readout = readout_by_name[readout_name]
            layer = config.law.layer if config.law.layer is not None else readout["layer"]
            sigma = float(readout.get("sigma", 1.0))
            direction = _direction_tensor(readout)
            hook = InterventionHook(
                law=config.law.kind,
                direction=direction,
                sigma=sigma,
                position=config.law.position,
                measure_readback=True,
            )
            controller = GenerationInterventionController(hook)
            layer_module = get_decoder_layer(model, layer)
            handle = layer_module.register_forward_hook(controller)
            try:
                for dose in config.calibration.doses:
                    strength = _strength_for_dose(config, float(dose), sigma)
                    pending = cell_mod.dose_pending_rows(
                        rows,
                        out_path,
                        resume=config.execution.resume,
                        readout=readout_name,
                        dose=float(dose),
                        strength=strength,
                        selection=config.calibration.selection,
                        write_at_zero=False,
                    )
                    run_summaries[f"{readout_name}:{float(dose):.17g}"] = {
                        "layer": layer,
                        "sigma": sigma,
                        "strength": strength,
                        "n_pending": len(pending),
                    }
                    for start in range(0, len(pending), config.execution.batch_size):
                        chunk = pending[start : start + config.execution.batch_size]
                        hook.last_readback = None
                        if len(chunk) == 1:
                            recs = [
                                _run_one_pass(
                                    model,
                                    tokenizer,
                                    controller,
                                    chunk[0],
                                    config.surface.generation,
                                    config.law.generation_mode,
                                    render_fn,
                                )
                            ]
                        else:
                            recs = _run_batch(
                                model,
                                tokenizer,
                                controller,
                                chunk,
                                config.surface.generation,
                                config.law.generation_mode,
                                render_fn,
                            )
                        readback = hook.last_readback
                        for idx, rec in enumerate(recs):
                            rec.update(
                                {
                                    "readout": readout_name,
                                    "readout_path": next(
                                        r.path for r in config.readouts if r.name == readout_name
                                    ),
                                    "layer": layer,
                                    "dose": float(dose),
                                    "dose_kind": config.calibration.dose_kind,
                                    "sigma": sigma,
                                    "config_sha": config_sha,
                                }
                            )
                            rec.update(_readback_for_record(readback, idx))
                            if grader is not None:
                                rec.update(grader(rec))
                            _write_checkpoint_record(out, rec, redact_fields)
            finally:
                handle.remove()

    summary = cell_mod.summarize_dose_calibration(out_path)
    summary["config_sha"] = config_sha
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    cell_mod.write_dose_manifest(out_path, config, config_sha, run_summaries)
    print(
        f"Dose calibration complete. Output {out_path}, summary {summary_path}, "
        f"config_sha {config_sha}"
    )
    return 0


# Default firing-probe strength. The premise is that it is large relative to any
# real dose ladder, so byte-identical output to "off" mode unambiguously means
# the hook did not fire (not merely "fired too weakly to move the argmax token").
# That premise breaks on high-activation-scale substrates (e.g. bnb-4bit bases)
# whose coherent doses run into the hundreds, where 100.0 sits in the inert
# regime and false-negatives. Override per cell via
# SmokeConfig.gen_stream_probe_strength when the dose ladder exceeds this.
_GEN_STREAM_SMOKE_STRENGTH = 100.0
_GEN_STREAM_SMOKE_MAX_NEW_TOKENS = 8


def _generate_under_mode(model, controller, enc, mode: str, strength: float, max_new_tokens: int):
    """Run one short greedy generate() pass with controller armed for mode.

    Shared by the gen_stream firing guard below and mirrors the begin_pass /
    generate / reset sequence _run_one_pass uses for the real arm passes.
    """
    import torch

    controller.begin_pass(mode, strength, attention_mask=enc["attention_mask"])
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
    controller.reset()
    return gen.sequences[0]


def gen_stream_fires(
    model,
    controller,
    enc,
    strength: float = _GEN_STREAM_SMOKE_STRENGTH,
    max_new_tokens: int = _GEN_STREAM_SMOKE_MAX_NEW_TOKENS,
) -> bool:
    """Return whether the gen_stream decode hook measurably changes generate() output.

    Runs one short greedy generate() under "gen_stream" mode at a strength
    large relative to any real dose ladder, and one under "off" mode, on the
    same encoded prompt, then compares the resulting token ids. Identical ids
    at that strength means the decode hook never fired during generate(): an
    optimized cached-generation decode path that never routes through the
    hooked module's Python forward() will silently produce this failure mode
    (documented for Unsloth's FastLanguageModel.for_inference). On a plain HF
    model the decode hook is expected to fire every decode step, so this
    should return True; a False here is the fail-closed signal the run_steer
    smoke gate acts on.

    enc must carry "input_ids" and "attention_mask" tensors already on the
    model's device, e.g. the output of tokenizer(prompt, return_tensors="pt").
    """
    import torch

    dosed = _generate_under_mode(model, controller, enc, "gen_stream", strength, max_new_tokens)
    off = _generate_under_mode(model, controller, enc, "off", strength, max_new_tokens)
    return not torch.equal(dosed, off)


def _run_smoke(model, tokenizer, config, rows, render_fn, direction, sigma, layer):
    """Run a small forward-only readback over the first smoke rows.

    For each smoke row, a fresh forward hook (in final-position mode with
    readback on) applies the dose arm's strength at the anchor token, and the
    realized projection is compared to the commanded value. Inactive rows probe
    off-target movement. A dedicated hook is registered and removed here so it
    does not interfere with the generation controller registered by the caller.
    """
    import torch

    from MechInterp.intervention import InterventionHook, get_decoder_layer

    smoke_rows = rows[: config.smoke.n_rows]
    dose_arm = next((a for a in config.arms if a.strength != 0.0), config.arms[0])
    strengths = cell_mod.resolve_arm_strengths(
        dose_arm, rows, {a.name: a for a in config.arms}
    )

    smoke_hook = InterventionHook(
        law=config.law.kind,
        direction=direction,
        sigma=sigma,
        position="final",
        measure_readback=True,
    )
    layer_module = get_decoder_layer(model, layer)
    handle = layer_module.register_forward_hook(smoke_hook)

    commanded, measured, offtarget = [], [], []
    dev = next(model.parameters()).device
    try:
        for row in smoke_rows:
            prompt = render_fn(row)
            enc = tokenizer(prompt, return_tensors="pt").to(dev)
            rk = cell_mod.row_key_of(row)
            g = float(strengths.get(rk, 0.0))
            smoke_hook.strength = g
            smoke_hook.attention_mask = enc["attention_mask"]
            smoke_hook.active = True
            with torch.no_grad():
                model(**enc, output_hidden_states=False, use_cache=False)
            rb = smoke_hook.last_readback or {}
            if g != 0.0:
                commanded.extend(rb.get("commanded", []))
                measured.extend(rb.get("measured", []))
            offtarget.append(float(rb.get("offtarget_abs_max", 0.0)))
    finally:
        handle.remove()
        smoke_hook.active = False

    return {
        "commanded": commanded,
        "measured": measured,
        "offtarget_abs_max": max(offtarget) if offtarget else 0.0,
    }


# --------------------------------------------------------------------------
# score_gates
# --------------------------------------------------------------------------


def run_score_gates(gates_config_path: str, rows_path: str, arm_field: str = "arm") -> int:
    from MechInterp.stats import evaluate_gates
    from MechInterp.stats.evaluator import load_gates_config

    gates_config = load_gates_config(gates_config_path)
    rows = cell_mod.load_jsonl(rows_path)
    report = evaluate_gates(gates_config, rows, arm_field=arm_field)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["overall_pass"] else 5
