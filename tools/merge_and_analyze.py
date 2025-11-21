#!/usr/bin/env python3
"""
Merge scored batches and generate quality triage report.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median, stdev

def load_scored_batches():
    """Load all scored batch files."""
    review_dir = Path(__file__).parent.parent / "Datasets" / "quality_review"
    examples = []

    for i in range(1, 11):  # Updated to load batches 1-10
        batch_file = review_dir / f"scored_batch_{i}.jsonl"
        if not batch_file.exists():
            print(f"Warning: {batch_file.name} not found")
            continue

        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                examples.append(example)

    return examples

def analyze_quality_scores(examples):
    """Analyze quality scores across all dimensions."""
    all_scores = []
    dimension_scores = defaultdict(list)

    for ex in examples:
        if 'quality_scores' not in ex:
            continue

        scores = ex['quality_scores']
        all_scores.append(scores['overall_quality'])

        for dim in ['sessionMemory_quality', 'toolContext_quality', 'goal_coherence',
                    'prompt_naturalness', 'response_realism']:
            if dim in scores:
                dimension_scores[dim].append(scores[dim])

    analysis = {
        'total_scored': len(all_scores),
        'overall': {
            'mean': round(mean(all_scores), 2),
            'median': round(median(all_scores), 2),
            'min': round(min(all_scores), 2),
            'max': round(max(all_scores), 2),
            'stdev': round(stdev(all_scores), 2) if len(all_scores) > 1 else 0
        },
        'dimensions': {}
    }

    for dim, scores in dimension_scores.items():
        analysis['dimensions'][dim] = {
            'mean': round(mean(scores), 2),
            'median': round(median(scores), 2),
            'min': round(min(scores), 2),
            'max': round(max(scores), 2)
        }

    return analysis

def categorize_by_quality(examples):
    """Categorize examples by quality tier."""
    categories = {
        'excellent': [],  # 4.0+
        'good': [],       # 3.0-3.9
        'fair': [],       # 2.0-2.9
        'poor': []        # < 2.0
    }

    for idx, ex in enumerate(examples):
        if 'quality_scores' not in ex:
            continue

        score = ex['quality_scores']['overall_quality']
        ex['_index'] = idx  # Track original index

        if score >= 4.0:
            categories['excellent'].append(ex)
        elif score >= 3.0:
            categories['good'].append(ex)
        elif score >= 2.0:
            categories['fair'].append(ex)
        else:
            categories['poor'].append(ex)

    return categories

def extract_common_issues(examples):
    """Extract common issues from quality notes."""
    issue_patterns = {
        'empty_sessionMemory': 0,
        'generic_toolContext': 0,
        'missing_results': 0,
        'weak_goals': 0,
        'command_style_prompt': 0,
        'minimal_metadata': 0
    }

    keywords = {
        'empty_sessionMemory': ['empty sessionMemory', 'sessionMemory is empty', 'sessionMemory: ""'],
        'generic_toolContext': ['generic toolContext', 'toolContext just', 'toolContext lacks'],
        'missing_results': ['missing Result', 'no Result', 'Result section'],
        'weak_goals': ['goals overlap', 'goals identical', 'weak hierarchy'],
        'command_style_prompt': ['command-style', 'robotic', 'not natural'],
        'minimal_metadata': ['minimal metadata', 'no executionTime', 'sparse result']
    }

    for ex in examples:
        if 'quality_scores' not in ex:
            continue

        notes = ex['quality_scores'].get('notes', '').lower()

        for issue, patterns in keywords.items():
            if any(pattern.lower() in notes for pattern in patterns):
                issue_patterns[issue] += 1

    return issue_patterns

def generate_markdown_report(examples, analysis, categories, issues):
    """Generate comprehensive markdown report."""

    report = f"""# Interaction Quality Review Report

**Date:** 2025-11-21
**Total Examples Scored:** {analysis['total_scored']}
**Dataset:** syngen_tools_sft_merged_complete_11.21.25.jsonl (5,505 total)
**Sample Strategy:** Stratified systematic sampling across 2 rounds (300 examples, ~5.4% coverage)

---

## Executive Summary

### Overall Quality Scores

| Metric | Score |
|--------|-------|
| **Mean** | **{analysis['overall']['mean']}** / 5.0 |
| **Median** | {analysis['overall']['median']} / 5.0 |
| **Range** | {analysis['overall']['min']} - {analysis['overall']['max']} |
| **Std Dev** | {analysis['overall']['stdev']} |

**Interpretation:** The dataset shows **fair to good quality** (2.88 avg) with significant room for improvement, especially in context fields and response structures.

### Quality Distribution

| Tier | Count | Percentage | Description |
|------|-------|------------|-------------|
| **Excellent** (4.0-5.0) | {len(categories['excellent'])} | {len(categories['excellent'])/analysis['total_scored']*100:.1f}% | Best examples - use as templates |
| **Good** (3.0-3.9) | {len(categories['good'])} | {len(categories['good'])/analysis['total_scored']*100:.1f}% | Minor improvements needed |
| **Fair** (2.0-2.9) | {len(categories['fair'])} | {len(categories['fair'])/analysis['total_scored']*100:.1f}% | Needs enhancement |
| **Poor** (1.0-1.9) | {len(categories['poor'])} | {len(categories['poor'])/analysis['total_scored']*100:.1f}% | Major rework required |

**Key Finding:** {len(categories['fair']) + len(categories['poor'])} examples ({(len(categories['fair']) + len(categories['poor']))/analysis['total_scored']*100:.1f}%) need improvement.

---

## Dimension Analysis

### Strengths and Weaknesses

"""

    # Dimension table
    dims = analysis['dimensions']
    dim_data = [
        ('prompt_naturalness', dims.get('prompt_naturalness', {})),
        ('goal_coherence', dims.get('goal_coherence', {})),
        ('response_realism', dims.get('response_realism', {})),
        ('toolContext_quality', dims.get('toolContext_quality', {})),
        ('sessionMemory_quality', dims.get('sessionMemory_quality', {}))
    ]

    # Sort by mean score descending
    dim_data.sort(key=lambda x: x[1].get('mean', 0), reverse=True)

    report += "| Dimension | Mean | Median | Range | Assessment |\n"
    report += "|-----------|------|--------|-------|------------|\n"

    for dim_name, dim_scores in dim_data:
        if not dim_scores:
            continue
        mean_score = dim_scores['mean']
        assessment = "🟢 Strong" if mean_score >= 4.0 else "🟡 Average" if mean_score >= 3.0 else "🟠 Weak" if mean_score >= 2.0 else "🔴 Critical"

        report += f"| **{dim_name}** | {mean_score} | {dim_scores['median']} | {dim_scores['min']}-{dim_scores['max']} | {assessment} |\n"

    report += f"""

### Key Insights

**🟢 Strengths:**
- **prompt_naturalness** ({dims.get('prompt_naturalness', {}).get('mean', 'N/A')}): Users write natural, conversational requests
- **goal_coherence** ({dims.get('goal_coherence', {}).get('mean', 'N/A')}): Clear hierarchies between primaryGoal and subgoal

**🔴 Critical Weaknesses:**
- **sessionMemory_quality** ({dims.get('sessionMemory_quality', {}).get('mean', 'N/A')}): Many empty or generic entries
- **response_realism** ({dims.get('response_realism', {}).get('mean', 'N/A')}): Missing Result objects and metadata
- **toolContext_quality** ({dims.get('toolContext_quality', {}).get('mean', 'N/A')}): Generic descriptions lacking workflow reasoning

---

## Common Issues Analysis

| Issue | Occurrences | % of Sample |
|-------|-------------|-------------|
| **Missing Results** | {issues['missing_results']} | {issues['missing_results']/analysis['total_scored']*100:.1f}% |
| **Empty sessionMemory** | {issues['empty_sessionMemory']} | {issues['empty_sessionMemory']/analysis['total_scored']*100:.1f}% |
| **Generic toolContext** | {issues['generic_toolContext']} | {issues['generic_toolContext']/analysis['total_scored']*100:.1f}% |
| **Minimal metadata** | {issues['minimal_metadata']} | {issues['minimal_metadata']/analysis['total_scored']*100:.1f}% |
| **Weak goal hierarchy** | {issues['weak_goals']} | {issues['weak_goals']/analysis['total_scored']*100:.1f}% |
| **Command-style prompts** | {issues['command_style_prompt']} | {issues['command_style_prompt']/analysis['total_scored']*100:.1f}% |

---

## Priority Triage

### Bottom 20% - High Priority Fixes ({int(analysis['total_scored'] * 0.2)} examples)

Examples most in need of enhancement:

"""

    # Get bottom 20%
    sorted_examples = sorted(examples, key=lambda x: x.get('quality_scores', {}).get('overall_quality', 5.0))
    bottom_20_count = int(len(sorted_examples) * 0.2)
    bottom_20 = sorted_examples[:bottom_20_count]

    for ex in bottom_20:
        if 'quality_scores' not in ex:
            continue

        score = ex['quality_scores']['overall_quality']
        user_prompt = ex['conversations'][0]['content'][:80] if len(ex['conversations']) > 0 else 'N/A'
        notes = ex['quality_scores']['notes'][:150]

        report += f"- **Score {score}**: \"{user_prompt}...\"\n"
        report += f"  - Issues: {notes}...\n\n"

    report += f"""
### Top 10% - Template Examples ({int(analysis['total_scored'] * 0.1)} examples)

Highest quality examples to use as templates:

"""

    # Get top 10%
    top_10_count = int(len(sorted_examples) * 0.1)
    top_10 = sorted_examples[-top_10_count:]
    top_10.reverse()

    for ex in top_10:
        if 'quality_scores' not in ex:
            continue

        score = ex['quality_scores']['overall_quality']
        user_prompt = ex['conversations'][0]['content'][:80] if len(ex['conversations']) > 0 else 'N/A'
        notes = ex['quality_scores']['notes'][:150]

        report += f"- **Score {score}**: \"{user_prompt}...\"\n"
        report += f"  - Why excellent: {notes}...\n\n"

    report += f"""
---

## Recommendations

### Immediate Actions (High Impact)

1. **Add Result structures to all examples** ({issues['missing_results']} need this)
   - Include realistic metadata: executionTime, totalResults, searchCapabilities
   - Add realistic scores/confidence values (not always 1.0)
   - Show edge cases: warnings, partial results, informative failures

2. **Enrich sessionMemory fields** ({issues['empty_sessionMemory']} empty)
   - Replace empty strings with contextual information
   - Reference specific prior tool calls: "Searched 23 files via vaultLibrarian_searchContent"
   - Include concrete details: numbers, paths, results
   - Minimum 50 chars with high information density

3. **Enhance toolContext reasoning** ({issues['generic_toolContext']} generic)
   - Explain WHY this tool was chosen (not just WHAT it does)
   - Show workflow reasoning: "After confirming path via search, now appending content"
   - Consider alternatives: "Using append instead of replace to preserve existing entries"

### Medium Priority

4. **Add metadata to responses** ({issues['minimal_metadata']} minimal)
   - executionTime (ms)
   - Timestamps
   - Result counts (totalResults, displayed, filtered)
   - Capability flags (semanticSearch, workspaceFiltering)

5. **Strengthen goal hierarchies** ({issues['weak_goals']} weak)
   - Ensure primaryGoal and subgoal are distinct
   - Show strategic decomposition, not just restatement
   - Subgoal should indicate current step in larger workflow

### Long-term Improvements

6. **Increase prompt diversity**
   - More ambiguous prompts requiring inference
   - References to implicit context: "that folder we created"
   - Corrections and clarifications: "Actually, make that Friday"
   - Domain-specific language (technical, creative, personal)

7. **Include more error scenarios**
   - Realistic failures with helpful error messages
   - Recovery workflows (retry with different parameters)
   - Edge cases (missing files, permission issues)

---

## Next Steps

1. **Review high-priority examples** (bottom 20%) using this report
2. **Apply enhancements** following the recommendations
3. **Use template examples** (top 10%) as reference for quality standards
4. **Re-score enhanced examples** to validate improvements
5. **Extrapolate patterns** to improve remaining 5,355 unscored examples

---

## Appendix: Scoring Methodology

- **Rubric:** 5-point scale across 5 dimensions
- **Sample Size:** 300 examples (5.4% of 5,505)
- **Sampling:** Stratified systematic across 2 rounds
  - Round 1: 150 examples (batches 1-5)
  - Round 2: 150 examples (batches 6-10)
- **Scoring:** 10 parallel agents (5 per round) following detailed rubric
- **Output:** Scored examples with detailed reasoning notes

**Files Generated:**
- `scored_batch_[1-10].jsonl` - Individual batch scores
- `scored_complete.jsonl` - Merged scored dataset (300 examples)
- `quality_triage_report.md` - This report
"""

    return report

def main():
    print("Loading scored batches...")
    examples = load_scored_batches()
    print(f"Loaded {len(examples)} scored examples")

    print("Analyzing quality scores...")
    analysis = analyze_quality_scores(examples)

    print("Categorizing by quality...")
    categories = categorize_by_quality(examples)

    print("Extracting common issues...")
    issues = extract_common_issues(examples)

    print("Generating report...")
    report = generate_markdown_report(examples, analysis, categories, issues)

    # Write merged scored dataset
    output_dir = Path(__file__).parent.parent / "Datasets" / "quality_review"
    merged_file = output_dir / "scored_complete.jsonl"

    with open(merged_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"✓ Merged dataset: {merged_file.name}")

    # Write report
    report_file = output_dir / "quality_triage_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report generated: {report_file.name}")
    print(f"\n{'='*60}")
    print(f"Overall Quality: {analysis['overall']['mean']}/5.0")
    print(f"Needs Improvement: {len(categories['fair']) + len(categories['poor'])}/{len(examples)} ({(len(categories['fair']) + len(categories['poor']))/len(examples)*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
