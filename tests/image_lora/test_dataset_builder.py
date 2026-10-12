"""build-dataset on a tiny synthetic vault: decisions, tokens, captions, pixels."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import dataset_builder as db  # noqa: E402


def _note(path: Path, fm: dict, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + yaml.safe_dump(fm, allow_unicode=True) + "---\n" + body, encoding="utf-8")


def _img(path: Path, *, size=(64, 96), color=(200, 190, 180), mode="RGB", pattern=0) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    channels = len(color)
    arr = np.empty((h, w, channels), dtype=np.uint8)
    arr[...] = color
    # A distinct checker per image keeps perceptual hashes apart (scaled to the size).
    ys, xs = np.mgrid[0:h, 0:w]
    cell_x, cell_y = max(1, w * (4 + pattern) // 64), max(1, h * (6 + 2 * pattern) // 96)
    mask = ((xs // cell_x) + (ys // cell_y)) % 2 == 1
    arr[mask, :3] = np.clip(arr[mask, :3].astype(int) - 120, 0, 255)
    assert mode == ("RGBA" if channels == 4 else "RGB")
    Image.fromarray(arr).save(path)


@pytest.fixture
def vault(tmp_path):
    images = tmp_path / "Images"
    world = tmp_path / "Book" / "World"
    moments = tmp_path / "Book" / "Moments"
    # Entities
    _note(world / "Characters" / "Gō.md", {
        "imageStatus": "in progress", "card_image": "[[Images/go-prime.png]]",
        "imagePrompt": "Sumi-e ink wash painting of a bald warrior. A massive kanabō strapped to his back.",
        "imageVariants": [
            {"id": "go-pose", "image": "[[Images/go-pose.png]]",
             "imagePrompt": "Gō swings his kanabō toward Hinata. Pure black and white ink wash only, no color."},
            {"id": "go-layer", "image": "[[Images/go-seated-layer-v1.png]]", "imagePrompt": "Gō seated."},
            {"id": "go-missing", "image": "[[Images/nope.png]]'", "imagePrompt": "x"},
        ]})
    _note(world / "Characters" / "Hinata.md", {"imageStatus": "in progress", "card_image": "Images/hinata.png",
                                               "imagePrompt": "A young woman with long black hair."})
    _note(world / "Objects" / "Gō's Kanabō.md", {"imageStatus": "in progress",
                                                  "card_image": "[[Images/kanabo.png]]",
                                                  "imagePrompt": "A war club. NO PEOPLE."})
    _note(world / "Objects" / "Old Armor of Gō.md", {"imageStatus": "not started",
                                                      "card_image": "[[Images/armor.png]]"})
    _note(world / "Locations" / "The Ash Fields.md", {"imageStatus": "completed",
                                                       "card_image": "[[Images/ash.png]]",
                                                       "imagePrompt": "A grey plain."})
    # Moments
    _note(moments / "M_001.md", {
        "id": "M_001", "imageStatus": "completed", "card_image": "[[Images/M_001-v1.png]]",
        "relatedEntities": ["[[Gō]]", "[[Hinata]]"], "notableLocations": ["[[The Ash Fields]]"],
        "imagePrompt": ("Vertical 2:3 monochrome sumi-e, graphite, and charcoal illustration on textured rice paper. "
                        "Gō leans on his kanabō like a cane while the wind blows. "
                        "No Hinata, bokken, text, or color.")})
    _note(moments / "M_002.md", {"id": "M_002", "imageStatus": "in progress",
                                 "card_image": "[[Images/M_002.png]]", "imagePrompt": "draft"})
    _note(moments / "M_003.md", {"id": "M_003", "imageStatus": "completed",
                                 "card_image": "[[Images/M_003.png]]", "imagePrompt": "Held out."})
    _note(moments / "M_004.md", {"id": "M_004", "imageStatus": "complete", "imagePrompt": "no image yet"})
    _note(moments / "M_005.md", {"id": "M_005", "imageStatus": "completed",
                                 "card_image": "[[Images/M_005.png]]", "relatedEntities": ["[[Hinata]]"],
                                 "imagePrompt": json.dumps({
                                     "style": {"medium": "sumi-e"},
                                     "scene": {"composition": "Hinata kneels by a well",
                                               "setting": "a forest at night"},
                                     "exclusions": ["no kanabō"]})})
    # Pixels
    _img(images / "go-prime.png", pattern=1)
    _img(images / "go-pose.png", pattern=2)
    _img(images / "go-seated-layer-v1.png", mode="RGBA", color=(0, 0, 0, 0), pattern=3)
    _img(images / "hinata.png", mode="RGBA", color=(10, 10, 10, 0), pattern=4)    # true alpha, not named layer
    _img(images / "kanabo.png", size=(3000, 4500), pattern=5)                     # oversized upscale
    _img(images / "armor.png", pattern=6)
    # Genuinely multi-coloured: green and magenta at the same luminance.
    ash = Image.new("RGB", (64, 96), (40, 200, 40))
    ash.paste((255, 60, 255), (0, 0, 32, 96))
    ash.save(images / "ash.png")
    _img(images / "M_001-v1.png", color=(210, 190, 160), pattern=7)              # sepia tint
    _img(images / "M_002.png", pattern=8)
    _img(images / "M_003.png", pattern=9)
    _img(images / "M_005.png", pattern=7, color=(205, 192, 165))                 # near-duplicate of M_001
    _img(images / "unrelated.png", pattern=10)
    recipe = {
        "name": "demo", "trigger": "demo ink style",
        "image_dirs": [str(images)], "candidate_dirs": [str(images)],
        "approved_statuses": ["completed", "complete"], "entity_excluded_statuses": ["not started"],
        "holdout_ids": ["M_003"],
        "sources": [
            {"kind": "moment", "notes": [str(moments / "*.md")], "entity_link_fields":
             ["relatedEntities", "notableLocations"]},
            {"kind": "entity", "entity_type": "character", "notes": [str(world / "Characters" / "*.md")]},
            {"kind": "entity", "entity_type": "object", "notes": [str(world / "Objects" / "*.md")]},
            {"kind": "entity", "entity_type": "location", "notes": [str(world / "Locations" / "*.md")]},
        ],
        "tokens": {"suffixes": {"character": "char", "object": "obj", "location": "loc"}},
        "text_detected_types": ["object"],
        "mention_required_types": ["character"],
        "text_aliases": {"Gō's Kanabō": ["kanabō", "kanabo"]},
        "relations": [{"subject": "Gō", "object": "Gō's Kanabō",
                       "patterns": [{"regex": "strapped to his back",
                                     "phrase": "{subject} with {object} strapped to his back"}]}],
        "exclude_filename_patterns": ["*-layer-*"],
        "dedupe": {"phash_max_distance": 6},
        "image": {"downscale_above": 2048, "target_long_side": 1328,
                  "grayscale": {"max_mean_chroma": 3, "max_p99_chroma": 10, "max_tint_residual": 6}},
        "captions": {
            "structured_fields": ["scene.composition", "scene.setting"],
            "negative_sentence_patterns": ["^(No|NO)\\b"],
            "drop_sentence_patterns": ["^(Pure black|Vertical 2:3)"],
            "strip_phrase_patterns": ["\\bsumi-e ink wash painting of\\b"],
        },
    }
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, allow_unicode=True), encoding="utf-8")
    return tmp_path, recipe_path


def _by_source(manifest):
    out = {}
    for item in manifest["included"] + manifest["excluded"]:
        out.setdefault(item["source_file"], []).append(item)
    return out


def test_tokens_follow_note_names():
    rules = {"suffixes": {"character": "char", "object": "obj", "location": "loc"}}
    assert db.derive_token("Gō", "character", rules) == "go_char"
    assert db.derive_token("Gō's Kanabō", "object", rules) == "kanabo_obj"
    assert db.derive_token("Bokken of Adauchi", "object", rules) == "bokken_obj"
    assert db.derive_token("The Ash Fields", "location", rules) == "ash_fields_loc"


def test_link_target_tolerates_hand_edited_yaml():
    assert db.link_target("[[Images/a b.png]]'") == "a b.png"
    assert db.link_target("Images/c.jpg") == "c.jpg"
    assert db.link_target(None) is None


def test_build_decisions_and_captions(vault):
    root, recipe = vault
    out = root / "out"
    manifest = db.build_dataset(recipe, out)
    src = _by_source(manifest)
    included = {i["source_file"]: i for i in manifest["included"]}

    # Approval, holdout and missing-image rules.
    assert src["M_002.png"][0]["decision"] == "exclude" and "not approved" in src["M_002.png"][0]["reason"]
    assert "held out" in src["M_003.png"][0]["reason"]
    assert any("no existing linked card image" in e["reason"] for e in manifest["excluded"])
    assert "armor.png" not in included and "excluded" in src["armor.png"][0]["reason"]
    assert any(e["reason"] == "linked image file does not exist" for e in manifest["excluded"])
    # Layers: by name and by real alpha.
    assert "excluded pattern" in src["go-seated-layer-v1.png"][0]["reason"]
    assert "transparent layer" in src["hinata.png"][0]["reason"]
    # Near-duplicate: lower-priority moment M_005 loses to M_001 (same pixels, other tint).
    assert "near-duplicate" in src["M_005.png"][0]["reason"]
    # Every library file is accounted for.
    assert src["unrelated.png"][0]["reason"].startswith("not linked")

    moment = included["M_001-v1.png"]
    # Hinata is linked but explicitly not drawn ("No Hinata ..."): no token for her.
    assert moment["tokens"] == ["go_char", "ash_fields_loc", "kanabo_obj"]
    assert moment["links_not_shown"] == ["hinata_char"]
    assert moment["caption"].startswith("demo ink style, go_char, ash_fields_loc, kanabo_obj, go_char holding kanabo_obj. ")
    assert "Vertical" not in moment["caption"] and "bokken" not in moment["caption"]
    assert "go_char leans on his kanabo_obj like a cane" in moment["caption"]

    card = included["go-prime.png"]
    assert "go_char with kanabo_obj strapped to his back" in card["caption"]
    pose = included["go-pose.png"]
    assert "go_char holding kanabo_obj" in pose["caption"]
    assert "Pure black" not in pose["caption"]
    kanabo = included["kanabo.png"]
    assert kanabo["tokens"] == ["kanabo_obj"] and kanabo["caption"].endswith("A war club.")
    assert max(kanabo["size"]) == 1328 and any("downscaled" in t for t in kanabo["transforms"])

    # Files and sidecars on disk.
    for item in manifest["included"]:
        image = out / item["output_file"]
        assert image.is_file()
        assert image.with_suffix(".txt").read_text().strip() == item["caption"]
    assert "| `kanabo_obj` | Gō's Kanabō | object | 4 |" in (out / "tokens.md").read_text()
    assert json.loads((out / "manifest.json").read_text())["counts"]["included"] == len(manifest["included"])


def test_grayscale_only_for_single_tint(vault):
    root, recipe = vault
    manifest = db.build_dataset(recipe, root / "out")
    included = {i["source_file"]: i for i in manifest["included"]}
    assert any("grayscale" in t for t in included["M_001-v1.png"]["transforms"])      # sepia
    assert not any("grayscale" in t for t in included["ash.png"]["transforms"])       # green + magenta


def test_structured_prompt_keeps_scene_only():
    recipe = {"captions": {"structured_fields": ["scene.composition", "scene.setting"]}}
    text = db.prompt_to_text(json.dumps({"style": {"medium": "ink"}, "scene": {
        "composition": "a woman kneels", "setting": "forest"}, "exclusions": ["no club"]}), recipe)
    assert text == "A woman kneels. Forest."
    assert db.prompt_to_text(["one", "two"], recipe) == "one, two"


def test_example_recipe_is_valid():
    example = Path(db.__file__).resolve().parent.parent / "configs" / "dataset_recipe.example.yaml"
    recipe = db.load_recipe(example)
    assert recipe["trigger"] and recipe["sources"]
    # Unmatched globs are fine: nothing is gathered, nothing fails.
    candidates, excluded, registry = db.gather(recipe)
    assert candidates == [] and registry == {}


def test_dry_run_writes_nothing(vault):
    root, recipe = vault
    db.build_dataset(recipe, root / "dry", write=False)
    assert not (root / "dry").exists()
