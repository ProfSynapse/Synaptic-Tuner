"""Describe the model artifact an evaluation ran against.

Location: ``Evaluator/model_artifact.py``.

The Evaluator records *what* was evaluated alongside the results so that a
quantized artifact can later be compared against its full-precision reference
(``python -m Evaluator.compare``). This module is deliberately generic: it only
knows llama.cpp quantization type *names* and the snake_case keys of a GGUF
build manifest; it never inspects model weights or tool formats.

Quantization precedence: ``--quantization`` flag > manifest ``quant_type`` >
label detected from the model path/name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Mapping, Optional

# llama.cpp quantization type names (ggml ftypes as used in GGUF filenames),
# e.g. Q4_K_M, Q5_K_S, Q8_0, Q4_0_4_8, IQ2_XS, IQ4_NL, TQ1_0, MXFP4, F16, BF16.
# Bounded by non-alphanumerics so ``BF16`` never also matches ``F16`` and
# ``Q4_K_M`` is found in ``model-Q4_K_M.gguf``, ``model.q4_k_m.gguf`` or
# ``repo:Q4_K_M``.
GGUF_QUANT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])("
    r"IQ[1-4]_(?:XXS|XS|NL|S|M)"
    r"|TQ[12]_0"
    r"|Q[2-8]_K(?:_[SML])?"
    r"|Q[4-8]_[01](?:_[48]_[48])?"
    r"|MXFP4"
    r"|BF16|F16|F32"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Labels that denote an unquantized (full or half precision) artifact.
FULL_PRECISION_LABELS = frozenset({"F32", "F16", "BF16"})


def detect_quantization(name: Optional[str]) -> Optional[str]:
    """Return the llama.cpp quant label found in a model path or name.

    Only the final path component is searched so directory names such as
    ``runs/F16_sweep/final_model`` do not produce a false label. Returns the
    canonical upper-case label, or None when nothing matches.
    """
    if not name:
        return None
    text = str(name).rstrip("/\\")
    basename = re.split(r"[/\\]", text)[-1]
    match = GGUF_QUANT_PATTERN.search(basename)
    return match.group(1).upper() if match else None


def is_full_precision(label: Optional[str]) -> bool:
    return bool(label) and str(label).upper() in FULL_PRECISION_LABELS


def _iter_manifest_file_entries(manifest: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield per-file entries (dicts carrying ``filename``) from a manifest.

    The canonical container is ``files``. When it is absent, any other
    top-level list of such dicts (or mapping of name -> entry) is accepted so
    the reader stays tolerant of the manifest writer's exact layout.
    ``calibration`` is never treated as a file container.
    """
    if "files" in manifest:
        containers: List[Any] = [manifest["files"]]
    else:
        containers = [value for key, value in manifest.items() if key != "calibration"]
    for container in containers:
        if isinstance(container, Mapping):
            values: Iterable[Any] = container.values()
        elif isinstance(container, list):
            values = container
        else:
            continue
        for entry in values:
            if isinstance(entry, Mapping) and entry.get("filename"):
                yield dict(entry)


def match_manifest_entry(
    manifest: Mapping[str, Any],
    model_path: str,
) -> Optional[Dict[str, Any]]:
    """Return the manifest file entry whose basename matches ``model_path``."""
    target = PurePath(str(model_path).replace("\\", "/")).name
    for entry in _iter_manifest_file_entries(manifest):
        if PurePath(str(entry["filename"]).replace("\\", "/")).name == target:
            return entry
    return None


def load_manifest_details(manifest_path: Path, model_path: str) -> Dict[str, Any]:
    """Read a GGUF build manifest and extract what applies to ``model_path``.

    Never raises for content problems: an unreadable manifest or a missing
    entry is reported under ``warnings`` so the evaluation itself is unaffected.
    """
    details: Dict[str, Any] = {"path": str(manifest_path), "warnings": []}
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        details["warnings"].append(f"could not read artifact manifest: {type(exc).__name__}: {exc}")
        return details
    if not isinstance(manifest, Mapping):
        details["warnings"].append("artifact manifest is not a JSON object")
        return details

    # Scalar top-level provenance (schema_version, source_model, base_dtype, ...).
    header = {
        key: value
        for key, value in manifest.items()
        if key not in ("files", "calibration") and isinstance(value, (str, int, float, bool))
    }
    if header:
        details["header"] = header
    entry = match_manifest_entry(manifest, model_path)
    if entry is None:
        details["warnings"].append(
            f"no manifest file entry matches '{PurePath(str(model_path)).name}'"
        )
    else:
        details["file"] = entry
    if isinstance(manifest.get("calibration"), Mapping):
        details["calibration"] = dict(manifest["calibration"])
    return details


def build_model_artifact(
    *,
    model: str,
    backend: str,
    quantization: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    load_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``model_artifact`` block for results metadata and lineage."""
    warnings: List[str] = []
    manifest = load_manifest_details(manifest_path, model) if manifest_path else None
    if manifest is not None:
        warnings.extend(manifest.pop("warnings"))
    manifest_quant = (manifest or {}).get("file", {}).get("quant_type")

    if quantization:
        label, source = quantization, "flag"
        if manifest_quant and str(manifest_quant).upper() != str(quantization).upper():
            warnings.append(
                f"--quantization {quantization} disagrees with manifest quant_type {manifest_quant}"
            )
    elif manifest_quant:
        label, source = str(manifest_quant), "manifest"
    else:
        label = detect_quantization(model)
        source = "detected" if label else None

    artifact: Dict[str, Any] = {
        "model": model,
        "backend": backend,
        "quantization": label,
        "quantization_source": source,
    }
    if load_settings:
        artifact["load_settings"] = dict(load_settings)
    if manifest is not None:
        artifact["manifest"] = manifest
    if warnings:
        artifact["warnings"] = warnings
    return artifact
