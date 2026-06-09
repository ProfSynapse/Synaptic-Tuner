#!/usr/bin/env python3
"""Paper library manager for the epistemic-humility research program.

Reads manifest.yaml and maintains an Obsidian-style library:
  notes/<arxiv-id>--<slug>.md   YAML frontmatter + sections (always available)
  pdfs/<arxiv-id>.pdf           downloaded from arXiv (requires network allowlist)
  fulltext/<arxiv-id>.html      ar5iv HTML full text (requires network allowlist)

Modes:
  --stub     create/refresh note stubs from manifest metadata (offline-safe)
  --enrich   query export.arxiv.org for title/authors/abstract, download PDFs
             and ar5iv full text; updates frontmatter status to "fetched"

Idempotent: existing note bodies are never overwritten — only frontmatter
fields managed by this script (title, year, status, url, pdf) are updated.

Network note: this execution environment allowlists outbound hosts. --enrich
needs arxiv.org, export.arxiv.org, and ar5iv.labs.arxiv.org to be allowed.
"""

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

LIB = Path(__file__).resolve().parent.parent
NOTES = LIB / "notes"
PDFS = LIB / "pdfs"
FULLTEXT = LIB / "fulltext"

ATOM = "{http://www.w3.org/2005/Atom}"

NOTE_TEMPLATE = """\
## Summary

<!-- filled during extraction -->

## Extracted numbers

<!-- rows feeding meta-analysis/evidence/*.csv; cite table/figure of origin -->

## Relevance to experiment

<!-- how this informs the Synaptic Tuner experiment design -->
"""


def note_path(paper: dict) -> Path:
    return NOTES / f"{paper['arxiv']}--{paper['slug']}.md"


def split_note(text: str):
    """Return (frontmatter dict, body) for an existing note."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def render_note(fm: dict, body: str) -> str:
    fm_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=1000)
    return f"---\n{fm_text}---\n{body}"


def base_frontmatter(paper: dict) -> dict:
    return {
        "title": paper.get("title", ""),
        "arxiv": paper["arxiv"],
        "year": paper.get("year"),
        "url": f"https://arxiv.org/abs/{paper['arxiv']}",
        "area": paper.get("area", "uncategorized"),
        "status": paper.get("status", "stubbed"),
        "tags": ["paper", "epistemic-humility", paper.get("area", "uncategorized")],
        "authors": paper.get("authors", []),
        "models": [],
        "metrics": [],
        "pdf": f"../pdfs/{paper['arxiv']}.pdf",
    }


def stub(papers: list) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        path = note_path(paper)
        if path.exists():
            fm, body = split_note(path.read_text())
            merged = base_frontmatter(paper)
            merged.update({k: v for k, v in fm.items() if v not in (None, [], "")})
            path.write_text(render_note(merged, body))
        else:
            path.write_text(render_note(base_frontmatter(paper), NOTE_TEMPLATE))
        print(f"stubbed {path.name}")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "synaptic-tuner-library/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def arxiv_metadata(arxiv_id: str) -> dict:
    data = fetch(f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1")
    entry = ET.fromstring(data).find(f"{ATOM}entry")
    if entry is None:
        return {}
    return {
        "title": re.sub(r"\s+", " ", entry.findtext(f"{ATOM}title", "")).strip(),
        "abstract": re.sub(r"\s+", " ", entry.findtext(f"{ATOM}summary", "")).strip(),
        "authors": [a.findtext(f"{ATOM}name", "") for a in entry.findall(f"{ATOM}author")],
        "published": entry.findtext(f"{ATOM}published", ""),
    }


def enrich(papers: list) -> None:
    PDFS.mkdir(parents=True, exist_ok=True)
    FULLTEXT.mkdir(parents=True, exist_ok=True)
    failures = []
    for paper in papers:
        aid = paper["arxiv"]
        path = note_path(paper)
        if not path.exists():
            stub([paper])
        fm, body = split_note(path.read_text())
        try:
            meta = arxiv_metadata(aid)
            if meta:
                fm["title"] = meta["title"] or fm.get("title")
                fm["authors"] = meta["authors"]
                if meta["published"]:
                    fm["year"] = int(meta["published"][:4])
                if meta["abstract"] and "## Abstract" not in body:
                    body = f"## Abstract\n\n{meta['abstract']}\n\n{body}"
            pdf_path = PDFS / f"{aid}.pdf"
            if not pdf_path.exists():
                pdf_path.write_bytes(fetch(f"https://arxiv.org/pdf/{aid}"))
            ft_path = FULLTEXT / f"{aid}.html"
            if not ft_path.exists():
                try:
                    ft_path.write_bytes(fetch(f"https://ar5iv.labs.arxiv.org/html/{aid}"))
                except Exception:
                    pass  # ar5iv lacks some papers; PDF remains canonical
            if fm.get("status") in (None, "candidate", "stubbed"):
                fm["status"] = "fetched"
            path.write_text(render_note(fm, body))
            print(f"fetched {aid}: pdf={pdf_path.exists()} fulltext={ft_path.exists()}")
        except Exception as exc:
            failures.append((aid, str(exc)))
            print(f"FAILED {aid}: {exc}", file=sys.stderr)
    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--only", help="comma-separated arXiv ids to restrict to")
    args = parser.parse_args()

    papers = yaml.safe_load((LIB / "manifest.yaml").read_text())["papers"]
    if args.only:
        wanted = set(args.only.split(","))
        papers = [p for p in papers if p["arxiv"] in wanted]
    if args.stub:
        stub(papers)
    if args.enrich:
        enrich(papers)
    if not (args.stub or args.enrich):
        parser.error("choose --stub and/or --enrich")


if __name__ == "__main__":
    main()
