"""Training recipe discovery and loading.

A recipe is a YAML file under ``Trainers/recipes/`` that describes a
training, eval, or job-pipeline run. Recipes carry two required header
fields used for filtering and dispatch:

* ``target``: ``local`` | ``cloud`` | ``both`` — which runner consumes it
* ``method``: ``sft`` | ``kto`` | ``grpo`` | ``gguf`` | ``loss-bench`` |
  ``datagen`` | ``eval`` (or any string — discovery does not validate the
  value)

``list_recipes()`` does header-only scans (cheap), ``load_recipe()`` does
the full parse and, for ``target: both`` recipes, deep-merges the runner
sub-block into the top level so the handler always sees a flat dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RECIPE_DIRNAME = "Trainers/recipes"

VALID_TARGETS = {"local", "cloud", "both"}


@dataclass(frozen=True)
class RecipeMeta:
    """Header metadata for a single recipe.

    Held by callers (TUI, CLI) for selection. The full body is loaded via
    ``load_recipe(path, runner)`` only when the user picks one.
    """

    path: Path
    name: str
    description: str
    target: str
    method: str


def _recipes_dir(repo_root: Path) -> Path:
    return repo_root / RECIPE_DIRNAME


def _read_header(path: Path) -> dict[str, Any] | None:
    """Parse a recipe YAML and return its top-level dict, or None on failure.

    Returns None for unreadable files, non-dict YAML, or YAML parse errors.
    Discovery should not crash if a single file is malformed — the file is
    skipped and the rest of the directory is still listed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def list_recipes(
    repo_root: Path,
    *,
    target: str | None = None,
    method: str | None = None,
) -> list[RecipeMeta]:
    """Scan ``Trainers/recipes/*.yaml`` and return matching recipe headers.

    Recipes without a ``target`` field, or whose top-level YAML is not a
    dict, are skipped silently. ``target=both`` recipes match both
    ``target='local'`` and ``target='cloud'`` filters.
    """
    recipes_dir = _recipes_dir(repo_root)
    if not recipes_dir.exists():
        return []

    results: list[RecipeMeta] = []
    for path in sorted(recipes_dir.glob("*.yaml")):
        if not path.is_file():
            continue
        data = _read_header(path)
        if data is None:
            continue

        recipe_target = str(data.get("target", "")).strip().lower()
        if not recipe_target:
            continue

        if target is not None:
            wanted = target.strip().lower()
            if recipe_target != wanted and recipe_target != "both":
                continue

        recipe_method = str(data.get("method", "")).strip().lower()
        if method is not None and recipe_method != method.strip().lower():
            continue

        results.append(
            RecipeMeta(
                path=path,
                name=str(data.get("name") or path.stem),
                description=str(data.get("description", "")).strip(),
                target=recipe_target,
                method=recipe_method,
            )
        )

    return results


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``. Overlay wins on conflicts.

    Lists and scalars are replaced (not concatenated). Dicts merge key-wise.
    Returns a new dict; inputs are not mutated.
    """
    result: dict[str, Any] = dict(base)
    for key, overlay_value in overlay.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            result[key] = _deep_merge(base_value, overlay_value)
        else:
            result[key] = overlay_value
    return result


def load_recipe(path: Path, runner: str) -> dict[str, Any]:
    """Load a recipe YAML and normalize it for the given runner.

    For ``target: both`` recipes, the ``local:``/``cloud:`` sub-block
    matching ``runner`` is deep-merged into the top level (sub-block wins
    on conflict), and both sub-blocks are removed before returning. Other
    targets are returned as-is.

    ``runner`` must be ``'local'`` or ``'cloud'``.
    """
    if runner not in {"local", "cloud"}:
        raise ValueError(f"runner must be 'local' or 'cloud', got: {runner!r}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Recipe must be a YAML object: {path}")

    target = str(data.get("target", "")).strip().lower()
    if target != "both":
        return data

    overlay = data.get(runner)
    if isinstance(overlay, dict):
        merged = _deep_merge(data, overlay)
    else:
        merged = dict(data)

    merged.pop("local", None)
    merged.pop("cloud", None)
    return merged
