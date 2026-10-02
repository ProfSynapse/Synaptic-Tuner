from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldMapping,
    FieldSelector,
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    MarkdownProfileV1,
    SourceMatcher,
    StructureBinding,
    StructureDefinition,
    StructureSet,
    TextProjection,
)
from tuner.dataset_prep import (
    ContextPackageV2,
    DatasetPrepConfigV2,
    DatasetPrepValidationError,
    GroupSplitV2,
    ItemLineageV2,
    ProjectionV1,
    PromptVariantV2,
    SplitAllocationV1,
    TargetTransformsV2,
    build_prepared_dataset_v2,
)
from tuner.dataset_prep import publication
from tuner.dataset_prep.context_messages import ParagraphGapPolicyV2, normalize_paragraph_gaps_v2
from tuner.ingestion import (
    NormalizedItemInputV1,
    load_verified_normalized_bundle_v1,
    write_normalized_bundle_v1,
)


def _bundle(tmp_path: Path, overrides=None):
    definition = StructureDefinition.define(
        name="GenericDocument",
        version="1",
        markdown=MarkdownProfileV1(FrontmatterMode.OPTIONAL),
        fields=(
            FieldMapping(
                "body",
                FieldSelector(FieldSelectorKind.DOCUMENT_BODY),
                FieldValueKind.STRING,
                True,
            ),
        ),
        text_projections=(TextProjection("text", "body"),),
    )
    structures = StructureSet(
        (definition,),
        (StructureBinding("markdown", SourceMatcher("**/*.md"), definition.ref),),
    )
    documents = {
        "context-root.md": "Earliest whole document.",
        "context-a.md": "Earlier whole document A.",
        "context-b.md": "Earlier whole document B.",
        "target-a.md": "KEEP: answer A\nDROP: private line\n```omit\nhidden\n```\n",
        "target-b.md": "answer B",
        "target-c.md": "answer C",
        "target-d.md": "answer D",
        "target-e.md": "answer E",
    }
    if overrides is not None:
        documents.update(overrides)
    inputs = tuple(
        NormalizedItemInputV1(
            logical_path=path,
            source_sha256=f"{index + 1:064x}",
            source_size_bytes=len(text.encode("utf-8")),
            structure_ref=definition.ref.to_dict(),
            fields={"body": text},
        )
        for index, (path, text) in enumerate(documents.items())
    )
    written = write_normalized_bundle_v1(tmp_path / "bundles", structures.to_dict(), inputs)
    loaded = load_verified_normalized_bundle_v1(written.path)
    ids = {item.logical_path: item.item_id for item in loaded.items}
    return definition, loaded, ids, documents


def _config(tmp_path: Path, overrides=None):
    definition, bundle, ids, documents = _bundle(tmp_path, overrides)
    lineage = (
        ItemLineageV2(ids["context-root.md"], "reference-root", 0, (), "reference-root"),
        ItemLineageV2(ids["context-a.md"], "reference-a", 1, (), "reference-a"),
        ItemLineageV2(ids["context-b.md"], "reference-b", 2, (), "reference-b"),
        ItemLineageV2(ids["target-a.md"], "target-a", 10, (), "group-ab"),
        ItemLineageV2(ids["target-b.md"], "target-b", 11, (ids["target-a.md"],), "group-ab"),
        ItemLineageV2(ids["target-c.md"], "target-c", 12, (), "group-c"),
        ItemLineageV2(ids["target-d.md"], "target-d", 13, (), "group-d"),
        ItemLineageV2(ids["target-e.md"], "target-e", 14, (), "group-e"),
    )
    packages = tuple(
        ContextPackageV2(
            ids[f"target-{letter}.md"],
            (ids["context-b.md"], ids["context-a.md"]),
            "alternate" if letter == "e" else "plain",
        )
        for letter in "abcde"
    )
    config = DatasetPrepConfigV2(
        source_bundle_path=bundle.path,
        expected_bundle_digest=bundle.semantic_identity.bundle_digest,
        target_projection=ProjectionV1.from_dict(
            {"structure_ref": definition.ref.to_dict(), "name": "text"}
        ),
        lineage=lineage,
        packages=packages,
        prompt_variants=(
            PromptVariantV2("plain", "Use the context.", "\n\n---\n\n"),
            PromptVariantV2("alternate", "Consider the material.", "\n\n***\n\n"),
        ),
        target_transforms=TargetTransformsV2(("omit",), ("DROP:",)),
        split=GroupSplitV2(
            "stable-fixture-split",
            (SplitAllocationV1("train", 3), SplitAllocationV1("validation", 1)),
        ),
    )
    return bundle, ids, documents, config


@pytest.mark.parametrize("tail", [False, True])
def test_document_cleanup_context_only_group(tmp_path, tail):
    bundle, ids, docs, config = _config(tmp_path, {
        "context-a.md": "Context\r\n  \r\n\r\nNext\r\n```widget\r\nhidden\r\n```\r\n  ![[media]]\r\n<iframe>\r\n",
        "context-b.md": "Unselected\n  \n\nKeep\n",
        "target-a.md": "Target\n  \n\nNext\n```widget\nhidden\n```\n  ![[media]]\n",
    })
    if tail:
        config = _tail_config(config, ids)
    transforms = TargetTransformsV2(("widget",), ("  ![[", "<iframe"))
    updated = replace(config, target_transforms=transforms, context_transforms=transforms,
        paragraph_gap_policy=ParagraphGapPolicyV2(("reference-a", "series" if tail else "group-ab")))
    assert DatasetPrepConfigV2.from_dict(updated.to_dict()) == updated
    prepared = build_prepared_dataset_v2(bundle, updated)
    assert prepared.rows[0].messages[1]["content"] == "Target\n\nNext\n"
    assert prepared.rows[0].messages[0]["content"] == config.prompt_variants[0].prompt + config.prompt_variants[0].separator + docs["context-b.md"] + config.prompt_variants[0].separator + "Context\r\n\r\nNext\r\n"
    assert prepared.identity.dataset_digest != build_prepared_dataset_v2(bundle, config).identity.dataset_digest
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity
    manifest = json.loads(prepared.manifest_raw)
    manifest["recipe"]["context_package"]["paragraph_gap_policy"]["kind"] = "unknown"
    (retained / "manifest.json").write_bytes(publication._canonical_bytes(manifest))
    with pytest.raises(DatasetPrepValidationError):
        publication.verify_prepared_dataset_v2(retained)


@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_gap_fences_and_crlf(marker):
    fenced = "   " + marker + "code\r\n  \r\n\r\n    indented\r\n" + marker + "\r\n"
    assert normalize_paragraph_gaps_v2("  prose\r\n  \r\n\r\nscene\r\n" + fenced + "\r\n\t\r\nend") == "  prose\r\n\r\nscene\r\n" + fenced + "\r\nend"
    assert normalize_paragraph_gaps_v2("```\n  \n\n") == "```\n  \n\n"


@pytest.mark.parametrize("policy", [None, {}, {"kind":"wrong","group_ids":["reference-a"]},
    {"kind":"collapse_blank_lines/v1","group_ids":[]},
    {"kind":"collapse_blank_lines/v1","group_ids":["reference-a","reference-a"]},
    {"kind":"collapse_blank_lines/v1","group_ids":[True]},
    {"kind":"collapse_blank_lines/v1","group_ids":["missing"]},
    {"kind":"collapse_blank_lines/v1","group_ids":["reference-a"],"extra":1}])
def test_invalid_gap_policy(tmp_path, policy):
    _, _, _, config = _config(tmp_path)
    raw = config.to_dict()
    raw["context_package"]["paragraph_gap_policy"] = policy
    with pytest.raises(DatasetPrepValidationError):
        DatasetPrepConfigV2.from_dict(raw)


def test_cleanup_absence_empty_prose_and_bounds(tmp_path):
    bundle, _, _, config = _config(tmp_path, {"context-a.md":"<iframe>\n  \n"})
    raw = config.to_dict()
    assert "context_transforms" not in raw["context_package"]
    assert "paragraph_gap_policy" not in config.semantic_recipe()["context_package"]
    assert build_prepared_dataset_v2(bundle, DatasetPrepConfigV2.from_dict(raw)).dataset_raw == build_prepared_dataset_v2(bundle, config).dataset_raw
    with pytest.raises(DatasetPrepValidationError, match="all document prose"):
        build_prepared_dataset_v2(bundle, replace(config, context_transforms=TargetTransformsV2((), ("<iframe",))))
    with pytest.raises(DatasetPrepValidationError):
        replace(config, context_transforms=TargetTransformsV2((), tuple(str(i) for i in range(257))))
    with pytest.raises(DatasetPrepValidationError):
        ParagraphGapPolicyV2(tuple(str(i) for i in range(257)))
    raw["context_package"]["context_transforms"] = None
    with pytest.raises(DatasetPrepValidationError):
        DatasetPrepConfigV2.from_dict(raw)


def test_gap_fence_closer_lengths_and_pseudo_fences():
    fenced = "````code\n  \n\n```\n~~~\n  \n\n````\n"
    assert normalize_paragraph_gaps_v2(fenced + "  \n\nend") == fenced + "\nend"
    pseudo = "    ```code\n  \n\ntext\n    ```\n"
    assert normalize_paragraph_gaps_v2(pseudo) == "    ```code\n\ntext\n    ```\n"
    with pytest.raises(DatasetPrepValidationError):
        ParagraphGapPolicyV2(("reference-a",), True)
    class StringSubclass(str):
        pass
    with pytest.raises(DatasetPrepValidationError):
        ParagraphGapPolicyV2(("reference-a",), StringSubclass("collapse_blank_lines/v1"))


def test_unclosed_context_fence_does_not_cross_document_boundary(tmp_path):
    bundle, _, _, config = _config(tmp_path, {
        "context-b.md": "```code\n  \n\nkept",
        "context-a.md": "Next\n  \n\nparagraph",
    })
    prepared = build_prepared_dataset_v2(bundle, replace(config,
        paragraph_gap_policy=ParagraphGapPolicyV2(("reference-a", "reference-b"))))
    variant = config.prompt_variants[0]
    assert prepared.rows[0].messages[0]["content"] == (
        variant.prompt + variant.separator + "```code\n  \n\nkept" +
        variant.separator + "Next\n\nparagraph")


@pytest.mark.parametrize("text,expected", [
    ("Prose [[name]] and [[destination|Display words]].", "Prose name and Display words."),
    ("[[Unicode Ω]]\r\n[[Dest|Alias Ω]]", "Unicode Ω\r\nAlias Ω"),
    ("![[image]] \\[[escaped]] `[[code]]` ``[[code]]``", "![[image]] \\[[escaped]] `[[code]]` ``[[code]]``"),
    ("[[path/item]] [[note#heading]] [[note^block]] [[x|y|z]] [[|alias]] [[x|]]", "[[path/item]] [[note#heading]] [[note^block]] [[x|y|z]] [[|alias]] [[x|]]"),
    ("[[outer [[inner]]]] [[unclosed", "[[outer [[inner]]]] [[unclosed"),
    ("   ~~~~code\r\n[[literal]]\r\n~~~\r\n[[still literal]]\r\n~~~~\r\n[[prose]]", "   ~~~~code\r\n[[literal]]\r\n~~~\r\n[[still literal]]\r\n~~~~\r\nprose"),
    ("```\n[[literal]]\n", "```\n[[literal]]\n"),
    ("`code\n    [[literal]]\n[[still literal]]` [[prose]]", "`code\n    [[literal]]\n[[still literal]]` prose"),
    ("    [[indented code]]\n\t[[tab code]]\n[[prose]]", "    [[indented code]]\n\t[[tab code]]\nprose"),
    ("[[extra closer]]]", "[[extra closer]]]"),
    ("Intro\r    [[code]]\r[[prose]]", "Intro\r    [[code]]\rprose"),
    ("    [[code]]\r[[prose]]", "    [[code]]\rprose"),
    ("Intro\r\n\t[[code]]\r\n[[prose]]", "Intro\r\n\t[[code]]\r\nprose"),
    ("~~~code\r[[literal]]\r~~~\r[[prose]]", "~~~code\r[[literal]]\r~~~\rprose"),
])
def test_wiki_display_preserves_protected_syntax(text, expected):
    from tuner.dataset_prep.context_messages import apply_target_transforms_v2
    assert apply_target_transforms_v2(text, TargetTransformsV2(inline_links="wiki_display/v1")) == expected
    assert apply_target_transforms_v2(text, TargetTransformsV2()) == text


@pytest.mark.parametrize("policy", [None, True, "wiki_display/v1", {}, {"kind":None},
    {"kind":True}, {"kind":"unknown"}, {"kind":"wiki_display/v1","extra":1}])
def test_wiki_display_rejects_invalid_policy(policy):
    raw = TargetTransformsV2().to_dict()
    raw["inline_links"] = policy
    with pytest.raises(DatasetPrepValidationError):
        TargetTransformsV2.from_dict(raw)


def test_wiki_display_identity_role_independence_and_public_verifier(tmp_path):
    bundle, _, _, config = _config(tmp_path, {
        "target-a.md": " ".join("[[word]]" for _ in range(31)),
        "context-a.md": "[[context]]",
    })
    old = build_prepared_dataset_v2(bundle, config)
    legacy = TargetTransformsV2().hashed_recipe()
    assert "inline_links" not in legacy
    assert legacy["selection_digest"] == publication._domain_digest("syntunia-target-transforms/v2", TargetTransformsV2().to_dict())
    transformed = replace(config, target_transforms=TargetTransformsV2(inline_links="wiki_display/v1"))
    assert DatasetPrepConfigV2.from_dict(transformed.to_dict()) == transformed
    prepared = build_prepared_dataset_v2(bundle, transformed)
    assert prepared.rows[0].messages[1]["content"] == " ".join("word" for _ in range(31))
    assert prepared.rows[0].messages[0]["content"] == old.rows[0].messages[0]["content"]
    assert prepared.identity.dataset_digest != old.identity.dataset_digest
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity
    manifest = json.loads(prepared.manifest_raw)
    manifest["recipe"]["context_package"]["target_transforms"]["inline_links"]["kind"] = "unknown"
    (retained / "manifest.json").write_bytes(publication._canonical_bytes(manifest))
    with pytest.raises(DatasetPrepValidationError):
        publication.verify_prepared_dataset_v2(retained)
    context = build_prepared_dataset_v2(bundle, replace(config, context_transforms=TargetTransformsV2(inline_links="wiki_display/v1")))
    assert context.rows[0].messages[1]["content"] == old.rows[0].messages[1]["content"]
    assert "[[context]]" not in context.rows[0].messages[0]["content"]


@pytest.mark.parametrize("different_structure", [False, True])
def test_independent_context_projection_public_boundary(tmp_path, different_structure):
    def definition(name):
        return StructureDefinition.define(name=name, version="1",
            markdown=MarkdownProfileV1(FrontmatterMode.OPTIONAL),
            fields=(FieldMapping("body", FieldSelector(FieldSelectorKind.DOCUMENT_BODY), FieldValueKind.STRING, True),
                    FieldMapping("full", FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "content"), FieldValueKind.STRING, True)),
            text_projections=(TextProjection("prose", "body"), TextProjection("source", "full")))
    target_definition = definition("TargetDoc")
    context_definition = definition("ContextDoc") if different_structure else target_definition
    definitions = (target_definition, context_definition) if different_structure else (target_definition,)
    structures = StructureSet(definitions, (
        StructureBinding("target", SourceMatcher("target*.md"), target_definition.ref),
        StructureBinding("context", SourceMatcher("context*.md"), context_definition.ref)))
    inputs = tuple(NormalizedItemInputV1(logical_path=path, source_sha256=f"{i+1:064x}",
        source_size_bytes=10, structure_ref=selected.ref.to_dict(),
        fields={"body":"Prose only", "full":"---\nkey: value\n---\nFull source"})
        for i, (path, selected) in enumerate((
            ("context-a.md", context_definition), ("context-b.md", context_definition),
            ("target-a.md", target_definition), ("target-b.md", target_definition))))
    written = write_normalized_bundle_v1(tmp_path / "bundles", structures.to_dict(), inputs)
    bundle = load_verified_normalized_bundle_v1(written.path)
    ids = {item.logical_path:item.item_id for item in bundle.items}
    config = DatasetPrepConfigV2(bundle.path, bundle.semantic_identity.bundle_digest,
        ProjectionV1.from_dict({"structure_ref":target_definition.ref.to_dict(),"name":"prose"}),
        tuple(ItemLineageV2(ids[path], "family-"+str(i), i, (), "group-"+str(i))
              for i,path in enumerate(("context-a.md","context-b.md","target-a.md","target-b.md"))),
        tuple(ContextPackageV2(ids["target-"+x+".md"],(ids["context-"+x+".md"],),"plain") for x in "ab"),
        (PromptVariantV2("plain","Write.","\n\n"),),TargetTransformsV2(),
        GroupSplitV2("seed",(SplitAllocationV1("train",1),SplitAllocationV1("validation",1))),
        context_projection=ProjectionV1.from_dict({"structure_ref":context_definition.ref.to_dict(),"name":"source"}))
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    assert prepared.rows[0].messages[0]["content"] == "Write.\n\n---\nkey: value\n---\nFull source"
    assert prepared.rows[0].messages[1]["content"] == "Prose only"
    manifest = json.loads(prepared.manifest_raw)
    assert manifest["context_projection"]["field_ref"] == "full"
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json",prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl",prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity
    manifest["context_projection"]["field_ref"] = "body"
    (retained / "manifest.json").write_bytes(publication._canonical_bytes(manifest))
    with pytest.raises(DatasetPrepValidationError):
        publication.verify_prepared_dataset_v2(retained)
    manifest["context_projection_digest"] = publication._domain_digest("syntunia-dataset-projection/v2", manifest["context_projection"])
    (retained / "manifest.json").write_bytes(publication._canonical_bytes(manifest))
    with pytest.raises(DatasetPrepValidationError,match="row_id"):
        publication.verify_prepared_dataset_v2(retained)
    with pytest.raises(DatasetPrepValidationError,match="exactly once"):
        build_prepared_dataset_v2(bundle,replace(config,context_projection=replace(config.context_projection,name="missing")))
    if different_structure:
        with pytest.raises(DatasetPrepValidationError,match="context item"):
            build_prepared_dataset_v2(bundle,replace(config,context_projection=config.target_projection))
    else:
        old = build_prepared_dataset_v2(bundle,replace(config,context_projection=None))
        assert old.rows[0].messages[0]["content"] == "Write.\n\nProse only"
        assert "context_projection" not in json.loads(old.manifest_raw)
        assert old.rows[0].row_id != prepared.rows[0].row_id
        explicit_same = build_prepared_dataset_v2(bundle,replace(config,context_projection=config.target_projection))
        assert explicit_same.rows[0].messages == old.rows[0].messages
        assert explicit_same.rows[0].row_id != old.rows[0].row_id


@pytest.mark.parametrize("projection", [None, True, {}, {"name":"source"}, {"structure_ref":{},"name":"source"}])
def test_invalid_context_projection(tmp_path, projection):
    _, _, _, config = _config(tmp_path)
    raw = config.to_dict()
    raw["context_projection"] = projection
    with pytest.raises((DatasetPrepValidationError,ValueError)):
        DatasetPrepConfigV2.from_dict(raw)


def _tail_config(config, ids) -> DatasetPrepConfigV2:
    targets = {ids[f"target-{letter}.md"] for letter in "abcde"}
    lineage = tuple(
        replace(entry, group_id="series") if entry.item_id in targets else entry
        for entry in config.lineage
    )
    return replace(
        config, lineage=lineage,
        split=GroupSplitV2(
            None,
            (SplitAllocationV1("train", 4), SplitAllocationV1("validation", 1)),
            "group_sequence_tail",
        ),
    )


def _relabel_and_rehash(prepared, index: int, split: str):
    rows = [json.loads(line) for line in prepared.dataset_raw.splitlines()]
    rows[index]["split"] = split
    dataset_raw = b"".join(publication._canonical_bytes(row) + b"\n" for row in rows)
    manifest = json.loads(prepared.manifest_raw)
    manifest["dataset_bytes"] = len(dataset_raw)
    manifest["dataset_sha256"] = hashlib.sha256(dataset_raw).hexdigest()
    manifest["split_counts"] = {
        name: sum(row["split"] == name for row in rows)
        for name in ("train", "validation")
    }
    basis = {
        "source_bundle_digest": manifest["source"]["bundle_digest"],
        "source_structure_set_digest": manifest["source"]["structure_set_digest"],
        "projection": manifest["projection"],
        "projection_digest": manifest["projection_digest"],
        "recipe": manifest["recipe"],
        "lineage": manifest["lineage"],
        "lineage_digest": manifest["lineage_digest"],
        "group_count": manifest["group_count"],
        "group_ids_sha256": manifest["group_ids_sha256"],
        "row_count": manifest["row_count"],
        "dataset_bytes": manifest["dataset_bytes"],
        "dataset_sha256": manifest["dataset_sha256"],
        "row_ids_sha256": manifest["row_ids_sha256"],
        "split_counts": manifest["split_counts"],
    }
    if "split_lineage" in manifest:
        basis["split_lineage"] = manifest["split_lineage"]
    manifest["dataset_digest"] = publication._domain_digest(publication._DATASET_DOMAIN_V2, basis)
    manifest["dataset_id"] = "dataset-" + manifest["dataset_digest"]
    return manifest, publication._canonical_bytes(manifest), dataset_raw






def _own_support_config(config, ids, tail, transitive=False, sequence=15):
    config = _tail_config(config, ids) if tail else config
    target = next(e for e in config.lineage if e.item_id == ids["target-a.md"])
    support = ids["context-a.md"]
    lineage = tuple(replace(e, group_id=target.group_id, sequence=sequence, derived_from=(target.item_id,))
                    if e.item_id == support else e for e in config.lineage)
    if transitive:
        lineage = tuple(replace(e, group_id=target.group_id, sequence=sequence, derived_from=(support,))
                        if e.item_id == ids["context-b.md"] else e for e in lineage)
        support = ids["context-b.md"]
    packages = (replace(config.packages[0], context_item_ids=(support,)), *(
        replace(p, context_item_ids=(ids["context-root.md"],)) for p in config.packages[1:]))
    return replace(config, lineage=lineage, packages=packages, context_projection=config.target_projection,
                   conditioning_policy="target_derived_support/v1")


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("transitive,sequence", [(False, 10), (False, 15), (True, 15)])
def test_own_support_roundtrip_public_boundary(tmp_path, tail, transitive, sequence):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _own_support_config(legacy, ids, tail, transitive, sequence)
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    manifest = json.loads(prepared.manifest_raw)
    assert ("conditioning_lineage" in manifest) is not tail
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity
    with pytest.raises(DatasetPrepValidationError):
        build_prepared_dataset_v2(bundle, replace(config, conditioning_policy=None))
    old = build_prepared_dataset_v2(bundle, legacy)
    roundtrip = build_prepared_dataset_v2(bundle, DatasetPrepConfigV2.from_dict(legacy.to_dict()))
    assert (old.dataset_raw, old.manifest_raw, old.identity) == (roundtrip.dataset_raw, roundtrip.manifest_raw, roundtrip.identity)
    assert "conditioning_policy" not in legacy.semantic_recipe()["context_package"]


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("bad", ["group", "revision", "coancestor", "cycle", "unresolved"])
def test_own_support_rejects_invalid_graph_builder_and_rehashed_public_artifact(tmp_path, tail, bad):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _own_support_config(legacy, ids, tail)
    prepared = build_prepared_dataset_v2(bundle, config)
    manifest = json.loads(prepared.manifest_raw)
    key = "split_lineage" if tail else "conditioning_lineage"
    target = next(e for e in manifest[key] if e["item_id"] == ids["target-a.md"])
    for e in manifest[key]:
        if e["item_id"] == ids["context-a.md"]:
            if bad == "group":
                e["group_id"] = "other"
            elif bad == "revision":
                e["revision_family"] = target["revision_family"]
            elif bad == "coancestor":
                e["derived_from"].append(ids["context-root.md"])
            elif bad == "cycle":
                e["derived_from"].append(e["item_id"])
            else:
                e["derived_from"].append("item-" + "f" * 64)
        if bad == "coancestor" and e["item_id"] == ids["context-root.md"]:
            e["sequence"] = 11
    with pytest.raises(DatasetPrepValidationError):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(ItemLineageV2.from_dict(e) for e in manifest[key])))
    manifest["recipe"]["context_package"]["lineage_digest"] = publication._domain_digest(publication._LINEAGE_DOMAIN_V2, manifest[key])
    basis = {name: manifest[name] for name in ("projection", "projection_digest", "recipe", "lineage", "lineage_digest", "group_count", "group_ids_sha256", "row_count", "dataset_bytes", "dataset_sha256", "row_ids_sha256", "split_counts", key, "context_projection", "context_projection_digest")}
    basis.update(source_bundle_digest=manifest["source"]["bundle_digest"], source_structure_set_digest=manifest["source"]["structure_set_digest"])
    manifest["dataset_digest"] = publication._domain_digest(publication._DATASET_DOMAIN_V2, basis)
    manifest["dataset_id"] = "dataset-" + manifest["dataset_digest"]
    retained = tmp_path / manifest["dataset_id"]
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", publication._canonical_bytes(manifest))
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    with pytest.raises(DatasetPrepValidationError):
        publication.verify_prepared_dataset_v2(retained)


@pytest.mark.parametrize("tail", [False, True])
def test_own_support_rejects_heldout_derivative_and_literal_target(tmp_path, tail):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _own_support_config(legacy, ids, tail)
    prepared = build_prepared_dataset_v2(bundle, config)
    heldout = next(r.target_item_id for r in prepared.rows if r.split == "validation")
    train = next(r.target_item_id for r in prepared.rows if r.split == "train")
    parent = next(e for e in config.lineage if e.item_id == heldout)
    lineage = tuple(replace(e, sequence=parent.sequence, derived_from=(heldout,), group_id=parent.group_id)
                    if e.item_id == ids["context-a.md"] else e for e in config.lineage)
    packages = tuple(replace(p, context_item_ids=(ids["context-a.md"],)) if p.target_item_id == train else
                     replace(p, context_item_ids=(ids["context-root.md"],)) for p in config.packages)
    with pytest.raises(DatasetPrepValidationError):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage, packages=packages))
    with pytest.raises(DatasetPrepValidationError):
        replace(config.packages[0], context_item_ids=(config.packages[0].target_item_id,))


@pytest.mark.parametrize("value", [None, True, {}, {"kind":None}, {"kind":"other"}, {"kind":"target_derived_support/v1","extra":1}])
def test_own_support_closed_policy_config(tmp_path, value):
    _, _, _, config = _config(tmp_path)
    raw = config.to_dict()
    raw["context_package"]["conditioning_policy"] = value
    with pytest.raises(DatasetPrepValidationError):
        DatasetPrepConfigV2.from_dict(raw)


def test_sequence_tail_preserves_rows_and_verifies_declared_assignment(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    old = build_prepared_dataset_v2(bundle, legacy)
    assert [row.messages for row in prepared.rows] == [row.messages for row in old.rows]
    assert [row.split for row in prepared.rows] == ["train"] * 4 + ["validation"]
    assert "split_lineage" in prepared.manifest
    assert "split_lineage" not in old.manifest
    assert prepared.manifest["recipe"]["split"] == {
        "kind": "group_sequence_tail",
        "allocations": [{"name": "train", "weight": 4}, {"name": "validation", "weight": 1}],
    }
    assert publication._verify_bytes_v2(
        tmp_path / prepared.identity.dataset_id,
        prepared.manifest_raw, prepared.dataset_raw,
    ).semantic_identity == prepared.identity
    rows = [
        (row.target_item_id, row.group_id, row.split, row.context_item_ids)
        for row in prepared.rows
    ]
    wrong = list(rows)
    target, group, _split, contexts = wrong[0]
    wrong[0] = (target, group, "validation", contexts)
    with pytest.raises(DatasetPrepValidationError, match="assignment"):
        publication._verify_sequence_tail(
            dict(prepared.manifest), prepared.manifest["recipe"], wrong,
            bundle.semantic_identity.item_count,
        )


def test_sequence_tail_keeps_equal_sequence_cohort_together(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    lineage = _lineage_with(config.lineage, ids["target-d.md"], sequence=14)
    prepared = build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))
    assert [row.split for row in prepared.rows] == ["train"] * 3 + ["validation"] * 2
    assert publication._verify_bytes_v2(
        tmp_path / prepared.identity.dataset_id,
        prepared.manifest_raw, prepared.dataset_raw,
    ).semantic_identity == prepared.identity


def test_sequence_tail_rejects_single_target_group_and_revision_leak(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = replace(legacy, split=GroupSplitV2(
        None,
        (SplitAllocationV1("train", 4), SplitAllocationV1("validation", 1)),
        "group_sequence_tail",
    ))
    with pytest.raises(DatasetPrepValidationError, match="chronology boundary"):
        build_prepared_dataset_v2(bundle, config)
    config = _tail_config(legacy, ids)
    lineage = _lineage_with(config.lineage, ids["context-root.md"], revision_family="target-e")
    packages = (
        replace(config.packages[0], context_item_ids=(ids["context-root.md"], ids["context-a.md"])),
        *config.packages[1:],
    )
    with pytest.raises(DatasetPrepValidationError, match="held-out target lineage"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage, packages=packages))


def test_sequence_tail_allows_validation_to_use_prior_training_target(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    packages = (*config.packages[:-1], replace(config.packages[-1], context_item_ids=(ids["target-d.md"],)))
    prepared = build_prepared_dataset_v2(bundle, replace(config, packages=packages))
    assert prepared.rows[-2].split == "train"
    assert prepared.rows[-1].split == "validation"
    assert prepared.rows[-1].context_item_ids == (ids["target-d.md"],)
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity


def test_sequence_tail_rejects_transitive_heldout_revision_and_target_family_crossing(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    lineage = _lineage_with(config.lineage, ids["context-root.md"], revision_family="target-e")
    lineage = _lineage_with(lineage, ids["context-a.md"], derived_from=(ids["context-root.md"],))
    packages = tuple(
        replace(package, context_item_ids=(ids["context-b.md"],))
        if package.target_item_id == ids["target-e.md"] else package for package in config.packages
    )
    with pytest.raises(DatasetPrepValidationError, match="held-out target lineage"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage, packages=packages))
    same_family = _lineage_with(config.lineage, ids["target-e.md"], revision_family="target-a")
    with pytest.raises(DatasetPrepValidationError, match="revision family crosses"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=same_family))


def test_sequence_tail_assignment_does_not_depend_on_package_array_order(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    forward = build_prepared_dataset_v2(bundle, config)
    reversed_rows = build_prepared_dataset_v2(bundle, replace(config, packages=tuple(reversed(config.packages))))
    assert {row.target_item_id: row.split for row in forward.rows} == {
        row.target_item_id: row.split for row in reversed_rows.rows
    }


def test_artifact_verifier_rejects_rehashed_wrong_tail_assignment_and_old_mixed_group(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    tail = build_prepared_dataset_v2(bundle, _tail_config(legacy, ids))
    manifest, manifest_raw, dataset_raw = _relabel_and_rehash(tail, 0, "validation")
    wrong_tail = tmp_path / manifest["dataset_id"]
    wrong_tail.mkdir(mode=0o700)
    publication._write_member(wrong_tail / "manifest.json", manifest_raw)
    publication._write_member(wrong_tail / "dataset.jsonl", dataset_raw)
    with pytest.raises(DatasetPrepValidationError, match="assignment"):
        publication.verify_prepared_dataset_v2(wrong_tail)
    old = build_prepared_dataset_v2(bundle, legacy)
    old_split = old.rows[0].split
    manifest, manifest_raw, dataset_raw = _relabel_and_rehash(
        old, 1, "validation" if old_split == "train" else "train",
    )
    wrong_old = tmp_path / manifest["dataset_id"]
    wrong_old.mkdir(mode=0o700)
    publication._write_member(wrong_old / "manifest.json", manifest_raw)
    publication._write_member(wrong_old / "dataset.jsonl", dataset_raw)
    with pytest.raises(DatasetPrepValidationError, match="group crosses declared splits"):
        publication.verify_prepared_dataset_v2(wrong_old)


def test_five_generic_packages_render_exact_two_message_rows_and_safe_manifest(tmp_path: Path) -> None:
    bundle, ids, documents, config = _config(tmp_path)
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    repeated = build_prepared_dataset_v2(bundle, config)

    assert prepared.dataset_raw == repeated.dataset_raw
    assert prepared.manifest_raw == repeated.manifest_raw
    assert len(prepared.rows) == 5
    first = prepared.rows[0].to_dict()
    assert [message["role"] for message in first["messages"]] == ["user", "assistant"]
    assert first["messages"][1]["content"] == "KEEP: answer A\n"
    assert "system" not in json.dumps(first)
    expected_user = (
        "Use the context.\n\n---\n\n"
        + documents["context-b.md"]
        + "\n\n---\n\n"
        + documents["context-a.md"]
    )
    assert first["messages"][0]["content"] == expected_user
    assert tuple(first["context_item_ids"]) == (ids["context-b.md"], ids["context-a.md"])
    assert prepared.rows[0].split == prepared.rows[1].split
    assert prepared.rows[-1].messages[0]["content"].startswith("Consider the material.")

    manifest_text = prepared.manifest_raw.decode("utf-8")
    assert str(tmp_path) not in manifest_text
    assert "Use the context." not in manifest_text
    assert "Earlier whole document" not in manifest_text
    assert "answer A" not in manifest_text
    assert prepared.manifest["lineage_digest"]
    assert prepared.manifest["lineage"][0]["context_item_ids"] == [
        ids["context-b.md"],
        ids["context-a.md"],
    ]


def test_context_order_and_prompt_template_bind_row_identity(tmp_path: Path) -> None:
    bundle, _ids, _documents, config = _config(tmp_path)
    baseline = build_prepared_dataset_v2(bundle, config)
    changed_package = replace(
        config.packages[0], context_item_ids=tuple(reversed(config.packages[0].context_item_ids))
    )
    reordered = build_prepared_dataset_v2(
        bundle, replace(config, packages=(changed_package, *config.packages[1:]))
    )
    changed_variant = replace(config.prompt_variants[0], prompt="Apply the context.")
    reprompted = build_prepared_dataset_v2(
        bundle, replace(config, prompt_variants=(changed_variant, config.prompt_variants[1]))
    )
    assert baseline.rows[0].row_id != reordered.rows[0].row_id
    assert baseline.rows[0].row_id != reprompted.rows[0].row_id


def test_rejects_target_in_own_context(tmp_path: Path) -> None:
    bundle, _ids, _documents, config = _config(tmp_path)
    package = config.packages[0]
    with pytest.raises(DatasetPrepValidationError, match="own context"):
        replace(package, context_item_ids=(package.target_item_id,))


def test_rejects_alternate_revision_context(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    context_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["context-a.md"])
    entries[context_index] = replace(entries[context_index], revision_family="target-a")
    with pytest.raises(DatasetPrepValidationError, match="alternate revision"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def test_rejects_transitive_ancestor_in_target_revision_family(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    lineage = _lineage_with(
        config.lineage,
        ids["context-b.md"],
        revision_family="target-a",
        sequence=2,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=3,
        derived_from=(ids["context-b.md"],),
    )
    with pytest.raises(DatasetPrepValidationError, match="alternate revision"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_rejects_context_derived_from_target(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    context_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["context-a.md"])
    entries[context_index] = replace(
        entries[context_index], sequence=11, derived_from=(ids["target-a.md"],)
    )
    with pytest.raises(DatasetPrepValidationError, match="derived from the target"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def test_rejects_future_context(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    context_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["context-a.md"])
    entries[context_index] = replace(entries[context_index], sequence=99)
    with pytest.raises(DatasetPrepValidationError, match="newer than the target"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def test_rejects_lineage_child_that_predates_its_source(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    lineage = _lineage_with(
        config.lineage,
        ids["context-b.md"],
        sequence=99,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=1,
        derived_from=(ids["context-b.md"],),
    )
    with pytest.raises(DatasetPrepValidationError, match="causal sequence"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_rejects_transitive_future_ancestor_closure(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    lineage = _lineage_with(
        config.lineage,
        ids["target-a.md"],
        sequence=10,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-b.md"],
        sequence=11,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=12,
        derived_from=(ids["context-b.md"],),
    )
    with pytest.raises(DatasetPrepValidationError, match="newer than the target"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_rejects_cross_group_derivative_targets(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    target_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["target-b.md"])
    entries[target_index] = replace(entries[target_index], group_id="different-group")
    with pytest.raises(DatasetPrepValidationError, match="derivative targets"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def _lineage_with(entries, item_id: str, **changes):
    return tuple(
        replace(entry, **changes) if entry.item_id == item_id else entry
        for entry in entries
    )


def _context_derivative_config(config, ids, *, transitive: bool, ancestor: str):
    lineage = _lineage_with(
        config.lineage,
        ids["target-b.md"],
        sequence=1,
        derived_from=(),
        group_id="group-b",
    )
    lineage = _lineage_with(
        lineage,
        ids["context-b.md"],
        sequence=2,
        derived_from=(ids[ancestor],),
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=3,
        derived_from=(ids["context-b.md"],) if transitive else (ids[ancestor],),
    )
    packages = tuple(
        replace(
            package,
            context_item_ids=(
                (ids["context-a.md"],)
                if package.target_item_id == ids["target-a.md"]
                else (ids["context-root.md"],)
            ),
        )
        for package in config.packages
    )
    return replace(
        config,
        lineage=lineage,
        packages=packages,
        split=replace(config.split, seed="held-out-probe-4"),
    )


@pytest.mark.parametrize("transitive", [False, True])
def test_rejects_context_descending_from_target_in_other_split(
    tmp_path: Path, transitive: bool
) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    leaked = _context_derivative_config(
        config,
        ids,
        transitive=transitive,
        ancestor="target-b.md",
    )
    baseline_lineage = _lineage_with(
        leaked.lineage, ids["context-a.md"], derived_from=()
    )
    baseline_lineage = _lineage_with(
        baseline_lineage, ids["context-b.md"], derived_from=()
    )
    baseline = build_prepared_dataset_v2(
        bundle, replace(leaked, lineage=baseline_lineage)
    )
    baseline_rows = {row.target_item_id: row for row in baseline.rows}
    assert baseline_rows[ids["target-a.md"]].split == "train"
    assert baseline_rows[ids["target-b.md"]].split == "validation"
    with pytest.raises(DatasetPrepValidationError, match="different split"):
        build_prepared_dataset_v2(bundle, leaked)


def test_allows_earlier_derived_context_when_ancestor_target_is_split_safe(
    tmp_path: Path,
) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    safe = _context_derivative_config(
        config,
        ids,
        transitive=False,
        ancestor="target-c.md",
    )
    lineage = _lineage_with(
        safe.lineage,
        ids["target-c.md"],
        sequence=1,
        derived_from=(),
    )
    # context-b is the direct derivative and context-a remains an independently
    # configured derivative of the same target; both precede target A.
    safe = replace(safe, lineage=lineage)
    prepared = build_prepared_dataset_v2(bundle, safe)
    rows = {row.target_item_id: row for row in prepared.rows}
    assert rows[ids["target-a.md"]].split == "train"
    assert rows[ids["target-c.md"]].split == "train"


@pytest.mark.parametrize("kind", ["duplicate", "unresolved"])
def test_rejects_duplicate_or_unresolved_references(tmp_path: Path, kind: str) -> None:
    bundle, _ids, _documents, config = _config(tmp_path)
    if kind == "duplicate":
        bad = replace(config, lineage=(*config.lineage, config.lineage[0]))
        message = "unique"
    else:
        bad_package = replace(config.packages[0], context_item_ids=("item-" + "f" * 64,))
        bad = replace(config, packages=(bad_package, *config.packages[1:]))
        message = "unresolved"
    with pytest.raises(DatasetPrepValidationError, match=message):
        build_prepared_dataset_v2(bundle, bad)
