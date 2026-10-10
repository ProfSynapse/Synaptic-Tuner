"""
Captioned image-LoRA dataset builder (Obsidian-style notes -> ai-toolkit folder).

Location: Trainers/image_lora/src/dataset_builder.py
Purpose:  Turn a library of note-described images into an ai-toolkit dataset
          folder (``<stem>.png`` + ``<stem>.txt`` pairs) with a full decision log.
          Everything project-specific (note globs, statuses, token rules, aliases,
          boilerplate, holdouts) lives in a recipe YAML kept with the dataset, not
          in this module (config-first; see AGENTS.md).
Used by:  Trainers/image_lora/train_image_lora.py ``build-dataset``.

Pipeline (each step is logged per candidate image in ``manifest.json``):
  1. Read the configured notes (YAML frontmatter). Moment-like notes contribute
     one image each; entity notes contribute their card image and every image
     variant. Entity notes also form the token registry.
  2. Approval: moment images need an approved status and a linked image that
     exists; entity images are dropped when the entity status is excluded.
  3. Filter: holdout ids, filename patterns (layers, chromakey) and real alpha.
  4. Same-file merges and perceptual-hash near-duplicate removal.
  5. Transform: downscale oversized upscales, grayscale effectively monochrome
     images, flatten to RGB PNG.
  6. Caption: trigger phrase, entity tokens, relation phrases, then the cleaned
     prompt with names replaced by tokens.
Every image file found under ``candidate_dirs`` that no note selected is logged
as excluded too, so the manifest accounts for the whole library.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


# --------------------------------------------------------------------------- notes


def read_frontmatter(path: Path) -> dict[str, Any]:
    """Return a note's YAML frontmatter (empty dict when absent or invalid)."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def link_target(value: Any) -> str | None:
    """Basename of an image reference: ``[[a/b.png]]``, ``"a/b.png"`` or ``b.png``.

    Tolerates the stray quote characters that hand-edited YAML sometimes leaves
    inside the value (for example ``[[x.png]]'``).
    """
    if value is None:
        return None
    text = str(value).strip().strip("'\"").strip()
    match = WIKILINK.search(text)
    if match:
        text = match.group(1)
    text = text.strip().strip("[]").strip("'\"").strip()
    return os.path.basename(text) or None


def link_names(values: Any) -> list[str]:
    """Note names from a frontmatter list of wikilinks (or a single string)."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    names = []
    for value in values:
        if value is None:
            continue
        for match in WIKILINK.finditer(str(value)):
            names.append(match.group(1).strip())
    return names


# --------------------------------------------------------------------------- tokens


def ascii_fold(text: str) -> str:
    """Strip diacritics: ``Gō`` -> ``Go``."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def derive_token(name: str, entity_type: str, rules: dict[str, Any]) -> str:
    """Stable token from a note name, e.g. ``The Ash Fields`` -> ``ash_fields_loc``.

    Rules (all from the recipe ``tokens`` section): strip leading articles,
    strip a possessive owner (``Gō's Kanabō`` -> ``Kanabō``), strip an
    ``of <Name>`` suffix (``Bokken of Adauchi`` -> ``Bokken``), drop
    parenthesised qualifiers, ASCII-fold, lowercase, join words with ``_`` and
    append the type suffix.
    """
    core = re.sub(r"\([^)]*\)", " ", name)
    for article in rules.get("strip_articles", ["The", "A", "An"]):
        core = re.sub(rf"^\s*{re.escape(article)}\s+", "", core)
    if rules.get("strip_possessive_owner", True):
        core = re.sub(r"^[^\s']+['’]s\s+", "", core)
    if rules.get("strip_of_suffix", True):
        core = re.sub(r"\s+of\s+.+$", "", core)
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_fold(core).lower()).strip("_")
    suffix = rules.get("suffixes", {}).get(entity_type, entity_type[:4])
    return f"{slug}_{suffix}" if suffix else slug


@dataclass
class Entity:
    name: str
    entity_type: str
    token: str
    note: str
    aliases: list[str] = field(default_factory=list)
    status: str | None = None


def build_registry(entity_notes: Iterable[tuple[Path, dict[str, Any], str]],
                   recipe: dict[str, Any]) -> dict[str, Entity]:
    """Note name -> Entity, with collision-safe tokens and text aliases."""
    token_rules = recipe.get("tokens", {})
    overrides = token_rules.get("overrides", {})
    text_aliases = recipe.get("text_aliases", {})
    registry: dict[str, Entity] = {}
    for path, fm, entity_type in entity_notes:
        name = path.stem
        token = overrides.get(name) or derive_token(name, entity_type, token_rules)
        aliases = [name]
        folded = ascii_fold(name)
        if folded != name:
            aliases.append(folded)
        aliases.extend(text_aliases.get(name, []))
        registry[name] = Entity(name, entity_type, token, str(path), aliases,
                                _status(fm, recipe))
    # Collisions fall back to the full slugged name (logged in tokens.md by count).
    by_token: dict[str, list[Entity]] = {}
    for entity in registry.values():
        by_token.setdefault(entity.token, []).append(entity)
    for token, group in by_token.items():
        if len(group) > 1:
            for entity in group:
                if entity.name in overrides:
                    continue
                full = re.sub(r"[^a-z0-9]+", "_", ascii_fold(entity.name).lower()).strip("_")
                suffix = token_rules.get("suffixes", {}).get(entity.entity_type, "")
                entity.token = f"{full}_{suffix}" if suffix else full
    return registry


def _status(fm: dict[str, Any], recipe: dict[str, Any]) -> str | None:
    value = fm.get(recipe.get("status_field", "imageStatus"))
    return str(value).strip().lower() if value is not None else None


# --------------------------------------------------------------------------- captions


def prompt_to_text(value: Any, recipe: dict[str, Any]) -> str:
    """Plain prompt text from a string, a JSON-object string, a dict or a list.

    Structured prompts keep only the dotted paths listed in the recipe's
    ``captions.structured_fields`` (scene description), never style/exclusion
    keys. Lists are rejoined with commas (some hand-edited YAML splits a prompt
    string on its commas).
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                value = json.loads(stripped)
            except ValueError:
                return value
        else:
            return value
    if isinstance(value, dict):
        parts = []
        for dotted in recipe.get("captions", {}).get(
                "structured_fields", ["scene.composition", "scene.setting"]):
            node: Any = value
            for key in dotted.split("."):
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, str) and node.strip():
                part = node.strip().rstrip(".")
                parts.append(part[0].upper() + part[1:] + ".")
        return " ".join(parts)
    return str(value)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text) if s.strip()]


def clean_prompt(text: str, recipe: dict[str, Any]) -> str:
    """Drop negative and boilerplate sentences, then inline boilerplate phrases."""
    cfg = recipe.get("captions", {})
    negative = [re.compile(p) for p in cfg.get("negative_sentence_patterns", [])]
    drop = [re.compile(p, re.IGNORECASE) for p in cfg.get("drop_sentence_patterns", [])]
    kept = [
        s for s in split_sentences(text)
        if not any(p.search(s) for p in negative) and not any(p.search(s) for p in drop)
    ]
    out = " ".join(kept)
    for pattern in cfg.get("strip_phrase_patterns", []):
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:])", r"\2", out)
    out = re.sub(r"(^|\.\s*)[.,;:]+\s*", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip(" ,;:.")
    if out:
        out += "."
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    max_words = int(cfg.get("max_prompt_words", 0) or 0)
    if max_words and len(out.split()) > max_words:
        out = " ".join(out.split()[:max_words]).rstrip(",;:") + "."
    return out


def _alias_pattern(alias: str) -> re.Pattern[str]:
    # Proper names (capitalised) match case-sensitively so "Go" never eats "go";
    # lowercase aliases (object nouns like "kanabō") match case-insensitively.
    flags = 0 if alias[:1].isupper() else re.IGNORECASE
    return re.compile(rf"(?<![\w-]){re.escape(alias)}(?:['’]s)?(?![\w])", flags)


def replace_names(text: str, registry: dict[str, Entity]) -> tuple[str, set[str]]:
    """Replace entity names/aliases with tokens; return the text and tokens found."""
    found: set[str] = set()
    pairs = sorted(
        ((alias, entity) for entity in registry.values() for alias in entity.aliases),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for alias, entity in pairs:
        pattern = _alias_pattern(alias)

        def _sub(match: re.Match[str], token: str = entity.token) -> str:
            found.add(token)
            possessive = match.group(0)[len(alias):]
            return token + ("'s" if possessive else "")

        text = pattern.sub(_sub, text)
    return text, found


def structured_names(value: Any) -> list[str]:
    """``references.*.name`` values of a structured (JSON) prompt."""
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, dict):
        return []
    refs = value.get("references")
    if not isinstance(refs, dict):
        return []
    names = []
    for ref in refs.values():
        if isinstance(ref, dict) and ref.get("name"):
            role = str(ref.get("role_in_scene") or "")
            if not re.search(r"\b(omitted|offscreen|off-screen|not shown)\b", role, re.IGNORECASE):
                names.append(str(ref["name"]))
    return names


def is_shown(entity: Entity, mention_text: str, recipe: dict[str, Any]) -> bool:
    """True when the prompt mentions the entity on-screen.

    Moment links list everyone involved in the beat, not everyone drawn. For the
    configured types, a linked entity is kept only when the positive prompt text
    names it (or uses one of its descriptive ``mention_aliases``) somewhere that
    is not immediately marked offscreen.
    """
    offscreen = recipe.get("offscreen_pattern",
                           r"[^.;]{0,40}?\b(offscreen|off-screen|outside the (close )?(frame|crop))\b")
    aliases = entity.aliases + recipe.get("mention_aliases", {}).get(entity.name, [])
    for alias in aliases:
        for match in _alias_pattern(alias).finditer(mention_text):
            tail = mention_text[match.end():match.end() + 80]
            if not re.match(offscreen, tail, re.IGNORECASE):
                return True
    return False


def positive_text(text: str, recipe: dict[str, Any]) -> str:
    """Prompt text used for entity detection: negative sentences and mid-sentence
    ``no ...`` / ``without ...`` clauses removed, so exclusion lists never add tokens."""
    negative = [re.compile(p) for p in recipe.get("captions", {}).get("negative_sentence_patterns", [])]
    kept = " ".join(s for s in split_sentences(text) if not any(p.search(s) for p in negative))
    return re.sub(r"\b(no|without|never)\b[^.;]*", " ", kept, flags=re.IGNORECASE)


def relation_phrases(tokens: set[str], raw_text: str, registry: dict[str, Entity],
                     recipe: dict[str, Any]) -> list[str]:
    """Phrases such as ``go_char holding kanabo_obj`` for configured pairs."""
    phrases = []
    for rule in recipe.get("relations", []):
        subject = registry.get(rule["subject"])
        obj = registry.get(rule["object"])
        if not subject or not obj or subject.token not in tokens or obj.token not in tokens:
            continue
        phrase = rule.get("default", "{subject} holding {object}")
        for alt in rule.get("patterns", []):
            if re.search(alt["regex"], raw_text, re.IGNORECASE):
                phrase = alt["phrase"]
                break
        phrases.append(phrase.format(subject=subject.token, object=obj.token))
    return phrases


def compose_caption(trigger: str, header_tokens: list[str], relations: list[str],
                    body: str) -> str:
    head = ", ".join([trigger, *header_tokens, *relations])
    return f"{head}. {body}".strip() if body else head


# --------------------------------------------------------------------------- pixels


def phash(image: Any) -> int:
    """64-bit DCT perceptual hash (numpy only)."""
    import numpy as np

    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    n = 32
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    dct = basis @ gray @ basis.T
    low = dct[:8, :8].flatten()[1:]
    bits = low > np.median(low)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def pixel_stats(image: Any) -> dict[str, float]:
    """Alpha coverage and chroma stats used by the filter/transform rules."""
    import numpy as np

    stats: dict[str, float] = {"transparency": 0.0}
    if image.mode in ("RGBA", "LA", "P"):
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
        stats["transparency"] = float((alpha < 250).mean())
    rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float64)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    stats["mean_chroma"] = float(chroma.mean())
    stats["p99_chroma"] = float(np.percentile(chroma, 99))
    # Tint residual: how much colour is left once each channel is predicted from
    # luminance alone (per 16 luminance bins). Sepia, cyanotype or warm-paper
    # monochrome scores near 0; genuinely multi-coloured images score high.
    flat = rgb.reshape(-1, 3)
    luma = flat @ np.array([0.299, 0.587, 0.114])
    offset = flat - luma[:, None]                 # the colour part of each pixel
    bins = np.minimum((luma / 16).astype(int), 15)
    residual = np.zeros_like(offset)
    for b in range(16):
        mask = bins == b
        if mask.any():
            residual[mask] = offset[mask] - offset[mask].mean(axis=0)
    stats["tint_residual"] = float(np.sqrt((residual ** 2).mean()))
    return stats


def transform_image(image: Any, stats: dict[str, float], cfg: dict[str, Any]) -> tuple[Any, list[str]]:
    """Downscale oversized images and grayscale effectively monochrome ones."""
    from PIL import Image

    steps: list[str] = []
    if image.mode != "RGB":
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[3])
        image = background
        steps.append("flattened to RGB")
    long_side = max(image.size)
    if long_side > int(cfg.get("downscale_above", 2048)):
        target = int(cfg.get("target_long_side", 1328))
        scale = target / long_side
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.LANCZOS)
        steps.append(f"downscaled {long_side}px -> {target}px long side")
    gray_cfg = cfg.get("grayscale", {})
    low_chroma = stats["mean_chroma"] <= float(gray_cfg.get("max_mean_chroma", -1)) and \
        stats["p99_chroma"] <= float(gray_cfg.get("max_p99_chroma", -1))
    single_tint = stats.get("tint_residual", 1e9) <= float(gray_cfg.get("max_tint_residual", -1))
    if gray_cfg and (low_chroma or single_tint):
        image = image.convert("L").convert("RGB")
        why = "low chroma" if low_chroma else "single tint"
        steps.append(f"grayscale (effectively monochrome: {why})")
    return image, steps


# --------------------------------------------------------------------------- build


@dataclass
class Candidate:
    file_name: str
    path: Path
    kind: str                  # moment | entity_card | entity_variant
    priority: int
    note: str
    source_id: str
    prompt: str
    tokens: list[str]
    raw_prompt: str
    stem: str
    entity: str | None = None
    merged_from: list[str] = field(default_factory=list)
    dropped_links: list[str] = field(default_factory=list)


def _resolve(file_name: str | None, image_dirs: list[Path]) -> Path | None:
    if not file_name:
        return None
    for directory in image_dirs:
        candidate = directory / file_name
        if candidate.is_file():
            return candidate
    return None


def _safe_stem(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ascii_fold(text).lower()).strip("_")[:80]


def _glob_all(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    return paths


def load_recipe(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle) or {}
    for key in ("name", "trigger", "image_dirs", "sources"):
        if key not in recipe:
            raise ValueError(f"recipe is missing required key: {key}")
    return recipe


def gather(recipe: dict[str, Any]) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Entity]]:
    """Collect candidate images and early exclusions from the configured notes."""
    image_dirs = [Path(p) for p in recipe["image_dirs"]]
    entity_notes = []
    for source in recipe["sources"]:
        if source["kind"] != "entity":
            continue
        for path in _glob_all(source["notes"]):
            fm = read_frontmatter(path)
            entity_type = source.get("entity_type") or str(fm.get("entity_type") or "entity")
            entity_notes.append((path, fm, entity_type))
    registry = build_registry(entity_notes, recipe)

    candidates: list[Candidate] = []
    excluded: list[dict[str, Any]] = []
    holdouts = set(recipe.get("holdout_ids", []))
    approved = {s.lower() for s in recipe.get("approved_statuses", ["completed", "complete"])}
    entity_excluded = {s.lower() for s in recipe.get("entity_excluded_statuses", ["not started"])}
    object_types = set(recipe.get("text_detected_types", ["object"]))

    def log(file_name: str | None, note: Path, reason: str, **extra: Any) -> None:
        excluded.append({"source_file": file_name, "source_note": str(note),
                         "decision": "exclude", "reason": reason, **extra})

    for source in recipe["sources"]:
        prompt_field = source.get("prompt_field", "imagePrompt")
        if source["kind"] == "moment":
            for path in _glob_all(source["notes"]):
                fm = read_frontmatter(path)
                if not fm:
                    continue
                source_id = str(fm.get(source.get("id_field", "id")) or path.stem)
                file_name = link_target(fm.get(source.get("image_field", "card_image")))
                status = _status(fm, recipe)
                resolved = _resolve(file_name, image_dirs)
                if status not in approved:
                    if file_name:
                        log(file_name, path, f"moment imageStatus '{status}' is not approved", source_id=source_id)
                    continue
                if resolved is None:
                    log(file_name, path, "approved moment has no existing linked card image", source_id=source_id)
                    continue
                if source_id in holdouts:
                    log(file_name, path, "held out for evaluation (recipe holdout_ids)", source_id=source_id)
                    continue
                raw = prompt_to_text(fm.get(prompt_field), recipe) or str(fm.get("description") or "")
                mention_text = " ".join([positive_text(raw, recipe),
                                         *structured_names(fm.get(prompt_field))])
                must_mention = set(recipe.get("mention_required_types", []))
                dropped_links: list[dict[str, str]] = []
                linked = []
                for link_field in source.get("entity_link_fields", []):
                    for name in link_names(fm.get(link_field)):
                        entity = registry.get(name)
                        if entity is None or entity.token in linked:
                            continue
                        if entity.entity_type in must_mention and not is_shown(entity, mention_text, recipe):
                            dropped_links.append({"source_id": source_id, "token": entity.token,
                                                  "reason": "linked but not shown in the prompt"})
                            continue
                        linked.append(entity.token)
                cleaned, _ = replace_names(clean_prompt(raw, recipe), registry)
                _, found = replace_names(positive_text(raw, recipe), registry)
                tokens = linked + sorted(
                    t for t in found
                    if t not in linked and _entity_type_of(t, registry) in object_types
                )
                candidates.append(Candidate(resolved.name, resolved, "moment", 0, str(path),
                                            source_id, cleaned, tokens, raw, _safe_stem(source_id),
                                            dropped_links=[d["token"] for d in dropped_links]))
        elif source["kind"] == "entity":
            for path in _glob_all(source["notes"]):
                fm = read_frontmatter(path)
                entity = registry.get(path.stem)
                if entity is None:
                    continue
                items = [("card", fm.get(source.get("image_field", "card_image")),
                          prompt_to_text(fm.get(prompt_field), recipe) or str(fm.get("description") or ""),
                          str(fm.get("description") or ""))]
                for variant in fm.get(source.get("variants_field", "imageVariants")) or []:
                    if isinstance(variant, dict):
                        fallback = ". ".join(str(variant.get(k)) for k in ("label", "narrative_context")
                                             if variant.get(k))
                        items.append((str(variant.get("id") or "variant"), variant.get("image"),
                                      prompt_to_text(variant.get("imagePrompt"), recipe), fallback))
                for item_id, ref, raw, fallback in items:
                    file_name = link_target(ref)
                    if not file_name:
                        continue
                    resolved = _resolve(file_name, image_dirs)
                    if resolved is None:
                        log(file_name, path, "linked image file does not exist", source_id=item_id)
                        continue
                    if entity.status in entity_excluded:
                        log(file_name, path, f"entity imageStatus '{entity.status}' is excluded", source_id=item_id)
                        continue
                    cleaned, _ = replace_names(clean_prompt(raw, recipe), registry)
                    if not cleaned and fallback:
                        cleaned, _ = replace_names(clean_prompt(fallback, recipe), registry)
                    # A single-entity reference never shows the other characters it
                    # mentions ("facing left toward Gō"): neutralise their tokens.
                    absent = recipe.get("entity_reference_absent_text")
                    if absent:
                        for other in registry.values():
                            if other.entity_type == "character" and other.token != entity.token:
                                cleaned = re.sub(rf"\b{re.escape(other.token)}('s)?(?![\w])",
                                                 absent, cleaned)
                    _, found = replace_names(positive_text(raw, recipe), registry)
                    tokens = [entity.token] + sorted(
                        t for t in found
                        if t != entity.token and _entity_type_of(t, registry) in object_types
                    )
                    kind = "entity_card" if item_id == "card" else "entity_variant"
                    stem = _safe_stem(f"{entity.token}_{item_id if item_id != 'card' else 'card'}")
                    candidates.append(Candidate(resolved.name, resolved, kind,
                                                1 if kind == "entity_card" else 2, str(path),
                                                item_id, cleaned, tokens, raw, stem, entity.name))
    return candidates, excluded, registry


def _entity_type_of(token: str, registry: dict[str, Entity]) -> str | None:
    for entity in registry.values():
        if entity.token == token:
            return entity.entity_type
    return None


def build_dataset(recipe_path: Path, out_dir: Path, *, write: bool = True) -> dict[str, Any]:
    """Build the dataset into ``out_dir`` and return the manifest dict."""
    from PIL import Image

    recipe = load_recipe(recipe_path)
    candidates, excluded, registry = gather(recipe)
    image_cfg = recipe.get("image", {})
    patterns = recipe.get("exclude_filename_patterns", [])
    max_transparency = float(recipe.get("max_transparency", 0.02))
    dedupe_distance = int(recipe.get("dedupe", {}).get("phash_max_distance", 6))

    # 1. Same file selected by several notes: keep the highest-priority record,
    #    merge the others' tokens into it.
    candidates.sort(key=lambda c: (c.priority, c.note, c.source_id))
    by_file: dict[str, Candidate] = {}
    for cand in candidates:
        keeper = by_file.get(str(cand.path))
        if keeper is None:
            by_file[str(cand.path)] = cand
            continue
        for token in cand.tokens:
            if token not in keeper.tokens:
                keeper.tokens.append(token)
        keeper.merged_from.append(f"{cand.note}#{cand.source_id}")
        excluded.append({"source_file": cand.file_name, "source_note": cand.note,
                         "source_id": cand.source_id, "decision": "exclude",
                         "reason": f"same file already selected by {keeper.note}#{keeper.source_id}; tokens merged"})

    included: list[dict[str, Any]] = []
    kept: list[tuple[Candidate, Any, dict[str, float], int]] = []
    for cand in by_file.values():
        lowered = cand.file_name.lower()
        hit = next((p for p in patterns if fnmatch.fnmatch(lowered, p.lower())), None)
        if hit:
            excluded.append(_exclusion(cand, f"filename matches excluded pattern '{hit}' (layer/chromakey component)"))
            continue
        with Image.open(cand.path) as opened:
            image = opened.copy()
        stats = pixel_stats(image)
        if stats["transparency"] > max_transparency:
            excluded.append(_exclusion(cand, f"transparent layer ({stats['transparency']:.0%} of pixels see-through)", stats=stats))
            continue
        kept.append((cand, image, stats, phash(image)))

    # 2. Near-duplicates by perceptual hash, in priority order.
    survivors: list[tuple[Candidate, Any, dict[str, float], int]] = []
    for cand, image, stats, digest in kept:
        clash = next((s for s in survivors if hamming(s[3], digest) <= dedupe_distance), None)
        if clash:
            excluded.append(_exclusion(
                cand, f"near-duplicate of {clash[0].file_name} (phash distance {hamming(clash[3], digest)})",
                stats=stats, phash=f"{digest:016x}"))
            continue
        survivors.append((cand, image, stats, digest))

    images_dir = out_dir / "images"
    if write:
        images_dir.mkdir(parents=True, exist_ok=True)
    used_stems: set[str] = set()
    for cand, image, stats, digest in survivors:
        stem = cand.stem
        n = 2
        while stem in used_stems:
            stem, n = f"{cand.stem}_{n}", n + 1
        used_stems.add(stem)
        out_image, steps = transform_image(image, stats, image_cfg)
        relations = relation_phrases(set(cand.tokens), cand.raw_prompt, registry, recipe)
        body = cand.prompt
        # Hand corrections after visual review (recipe caption_fixes, keyed by
        # source file): replace relation phrases and/or rewrite body text.
        fix = recipe.get("caption_fixes", {}).get(cand.file_name)
        if fix:
            if "relations" in fix:
                relations = list(fix["relations"])
            for pattern, replacement in fix.get("replace", []):
                body = re.sub(pattern, replacement, body)
            for token in fix.get("add_tokens", []):
                if token not in cand.tokens:
                    cand.tokens.append(token)
            steps.append("caption fixed by hand (recipe caption_fixes)")
        caption = compose_caption(recipe["trigger"], cand.tokens, relations, body)
        if write:
            out_image.save(images_dir / f"{stem}.png", optimize=True)
            (images_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
        included.append({
            "decision": "include", "output_file": f"images/{stem}.png", "source_file": cand.file_name,
            "source_path": str(cand.path), "source_note": cand.note, "source_id": cand.source_id,
            "kind": cand.kind, "entity": cand.entity, "tokens": cand.tokens, "relations": relations,
            "caption": caption, "transforms": steps, "size": list(out_image.size),
            "source_size": list(image.size), "phash": f"{digest:016x}",
            "sha256_source": _sha256(cand.path), "merged_from": cand.merged_from,
            "links_not_shown": cand.dropped_links,
            "pixel_stats": {k: round(v, 3) for k, v in stats.items()},
            "provenance": recipe.get("provenance_default", "not recorded in notes"),
        })

    # 3. Account for every other image file in the library.
    seen = {e["source_file"] for e in excluded} | {i["source_file"] for i in included}
    for directory in recipe.get("candidate_dirs", []):
        for path in sorted(Path(directory).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name not in seen:
                excluded.append({"source_file": path.name, "source_path": str(path), "decision": "exclude",
                                 "reason": "not linked as an approved image by any configured note"})
                seen.add(path.name)

    token_counts: dict[str, int] = {}
    for item in included:
        for token in item["tokens"]:
            token_counts[token] = token_counts.get(token, 0) + 1
    manifest = {
        "schema_version": "synaptic-image-lora-dataset/v1",
        "name": recipe["name"], "trigger": recipe["trigger"],
        "recipe": str(recipe_path), "recipe_sha256": _sha256(recipe_path),
        "counts": {"included": len(included), "excluded": len(excluded),
                   "by_kind": _count(included, "kind"),
                   "exclusion_reasons": _reason_counts(excluded)},
        "tokens": {e.token: {"entity": e.name, "type": e.entity_type, "note": e.note,
                             "images": token_counts.get(e.token, 0)}
                   for e in sorted(registry.values(), key=lambda e: e.token)},
        "included": included, "excluded": excluded,
    }
    if write:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        (out_dir / "tokens.md").write_text(render_tokens_md(manifest), encoding="utf-8")
    return manifest


def _exclusion(cand: Candidate, reason: str, **extra: Any) -> dict[str, Any]:
    return {"source_file": cand.file_name, "source_path": str(cand.path), "source_note": cand.note,
            "source_id": cand.source_id, "decision": "exclude", "reason": reason, **extra}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item[key]] = counts.get(item[key], 0) + 1
    return counts


def _reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reason = re.sub(r"\(.*?\)|'[^']*'|\S+\.(png|jpg|jpeg|webp)", "…", item["reason"], flags=re.IGNORECASE)
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def render_tokens_md(manifest: dict[str, Any]) -> str:
    lines = [f"# Tokens: {manifest['name']}", "",
             f"Trigger phrase: `{manifest['trigger']}`. Tokens are derived from note names "
             "(articles, possessive owners and `of <Name>` suffixes stripped, ASCII-folded, type suffix added).",
             "", "| Token | Entity | Type | Images |", "|---|---|---|---|"]
    for token, info in manifest["tokens"].items():
        lines.append(f"| `{token}` | {info['entity']} | {info['type']} | {info['images']} |")
    lines.append("")
    return "\n".join(lines)


def contact_sheets(out_dir: Path, manifest: dict[str, Any], *, per_sheet: int = 48,
                   thumb: int = 192) -> list[Path]:
    """Write labelled contact sheets of the included images for visual review."""
    from PIL import Image, ImageDraw

    review = out_dir / "review"
    review.mkdir(parents=True, exist_ok=True)
    items = manifest["included"]
    sheets = []
    cols = 8
    for start in range(0, len(items), per_sheet):
        chunk = items[start:start + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 16)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, item in enumerate(chunk):
            with Image.open(out_dir / item["output_file"]) as img:
                img.thumbnail((thumb, thumb))
                x, y = (i % cols) * thumb, (i // cols) * (thumb + 16)
                sheet.paste(img, (x + (thumb - img.width) // 2, y))
            draw.text((x + 2, y + thumb + 2), Path(item["output_file"]).stem[:30], fill=(0, 0, 0))
        path = review / f"contact_{start // per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=85)
        sheets.append(path)
    return sheets
