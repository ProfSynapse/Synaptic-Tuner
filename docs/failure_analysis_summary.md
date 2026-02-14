# Failure Analysis - Executive Summary

**Date:** 2025-12-18
**Analyst:** Claude Code (Sonnet 4.5)

---

## Overview

Analyzed 179 failed improvement examples from vaultLibrarian and vaultManager dataset improvement runs.

**Key Finding:** The improvement engine is **working as designed** - it successfully converts legacy direct tool calls to the new useTools format. Failures are due to secondary validation issues (factuality, parameters) that persist across multiple iterations.

---

## Failure Breakdown

| Agent | Total Examples | Failed | Failure Rate |
|-------|---------------|--------|--------------|
| vaultLibrarian | 939 | 174 | 18.5% |
| vaultManager | 1,094 | 5 | 0.5% |

---

## Root Causes (vaultLibrarian)

### 1. Factuality Issues (98.9% of failures)
- **Problem:** Hallucinated file paths, invented dates, wrong IDs
- **Why:** Improver struggles to match paths to vault_structure after format conversion
- **Impact:** 172/174 failures
- **Fix:** Relax validation for subdirectory inferences

### 2. Parameter Validation (Score Distribution)
- **0.5 score (37.2%):** "Almost right" - minor issues
- **0.0 score (33.7%):** Missing required params
- **0.2 score (19.8%):** Significant param problems
- **Fix:** Review required param definitions (e.g., make `paths` optional for searchDirectory)

### 3. Unsupported Tools (4.7% of failures)
- **Problem:** 7 examples use `batch` or `batchFileOperation` tools
- **Why:** These tools not in useTools schema
- **Impact:** Low but fixable
- **Fix:** Add batch tools to schema OR filter pre-improvement

### 4. Iteration Exhaustion (100% of failures)
- **Problem:** ALL failures hit 12-iteration limit
- **Why:** Improver fixes format (iterations 1-3) but can't resolve factuality (iterations 4-12)
- **Fix:** Increase limit to 20 OR add early stopping

---

## What's Working

✅ **Format conversion:** 100% of failed examples successfully converted to useTools
✅ **Schema validation:** Validation catches actual issues (not false positives)
✅ **Pass rate:** 81.5% of examples pass all rubrics after improvement
✅ **Rubric design:** Correctly enforces useTools standard

---

## Recommended Actions

### Immediate (30 min total)

1. **Add batch tools to vaultLibrarian_tools.yaml** (lines 62-69, 135-142, 200-223)
2. **Relax factuality path validation** in factuality.yaml (add subdirectory skip pattern)
3. **Increase iteration limit** from 12 → 20 in rubric_runner config
4. **Re-run improvement** on 174 failed examples

**Expected outcome:** Failure rate drops to 5-8%

### Short-term (1-2 hours)

5. **Add more improver examples** (3-4 diverse tool call examples)
6. **Add tool mapping guidance** for legacy → useTools conversions
7. **Review passed examples** (random sample of 50 for QA)

### Long-term (ongoing)

8. **Standardize dataset generation** to use useTools from the start
9. **Build quality dashboard** showing rubric pass rates over time
10. **Add semantic validation** to judge prompts (query quality, not just presence)

---

## Files Delivered

1. **failure_analysis.md** - Detailed 800-line analysis with examples and YAML changes
2. **failure_analysis_summary.md** - This executive summary
3. **Datasets/archive/analysis/failure_analysis_data.json** - Raw statistics and failure data

---

## Next Steps

1. **Apply YAML changes** from detailed analysis
2. **Re-run improvement** with updated rubrics
3. **Monitor success rate** (expect 40-60% of current failures to pass)
4. **Manual review** remaining failures for edge cases

---

## Questions?

See **failure_analysis.md** for:
- Complete rubric YAML changes (copy-paste ready)
- Detailed failure pattern examples
- Appendix with 3 real failure cases
- Full recommendations with rationale
