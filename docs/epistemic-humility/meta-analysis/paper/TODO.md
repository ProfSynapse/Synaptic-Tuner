# Pre-submission checklist (draft-v0 -> arXiv v1)

- [ ] **PDF verification pass.** Re-verify all 49 `verified=false` rows in
      `evidence/effects.csv` against primary PDFs once network access to
      arxiv.org/aclanthology.org/openreview.net is available. Flip flags,
      remove daggers (†) in the draft prose, and re-run
      `analysis/synthesize.py` so C1-C5 counts/medians regenerate. Priority
      order: starred sycophancy figures (98%, ~6% from 2310.13548), GPT-4 TR
      Figure 8 ECE values, Cheng Table values, AbstentionBench 24%.
- [ ] **PRISMA flow counts.** Reconstruct records-identified / screened /
      excluded / included numbers from the five search agents' logs (82
      queries) + the follow-up search; add a PRISMA-style flow figure and an
      explicit exclusion log. Currently only inclusion criteria are stated.
- [ ] **Integrate raw-reports/06** (mech-interp/probing follow-up search,
      running) into Sections 3, 6.2, 6.3 gap 4, and 8; add any new effects rows.
- [ ] **Figure regeneration.** Regenerate `analysis/figures/forest.png` from
      final effects.csv; add (a) per-claim-family vote/sign-test figure,
      (b) Section 5.3 recall-vs-over-refusal operating-point scatter,
      (c) L1-L4 coverage map (stack level x rows). Verify figure numbers match
      regenerated synthesis-summary.md.
- [ ] **Related-surveys positioning paragraph.** Expand Section 6.4 into a
      proper comparison with 2407.18418 (abstention survey) and 2409.18786
      (honesty survey), plus any 2026 surveys found during the PDF pass.
- [ ] **Author/affiliation block.** Add author name(s), affiliation, contact,
      and the companion-essay citation in final form; decide author-name form
      for the essay attribution (Section 3) and References.
- [ ] **Finish the interim columns** of `idk-method-reanalysis.csv`
      (exact-match answer grader for `approx_correct_on_known` /
      `approx_truthful`) or drop them from the released CSV; the draft
      currently excludes them from claims.
- [ ] **Abstract trim** to ~200 words (currently ~250) and final word-count /
      style pass; convert references to BibTeX; arXiv formatting (LaTeX or
      arXiv-ready markdown pipeline).
- [ ] **Verify 2026 arXiv IDs** flagged in library manifest (2603.xxxxx-
      2606.xxxxx cluster cited via the essay) resolve to the intended papers.
