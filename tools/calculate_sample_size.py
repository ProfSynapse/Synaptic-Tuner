#!/usr/bin/env python3
"""
Calculate statistically representative sample size based on tool distribution.
"""
import json
import math
from pathlib import Path
from collections import Counter

def extract_tool_name(example):
    """Extract tool name from assistant content."""
    try:
        if len(example.get('conversations', [])) < 2:
            return None

        content = example['conversations'][1].get('content', '')

        # Look for tool_call: pattern
        if 'tool_call:' not in content:
            return None

        # Extract tool name
        tool_line = content.split('tool_call:')[1].split('\n')[0].strip()
        return tool_line
    except:
        return None

def calculate_sample_size(population_size, confidence_level=0.95, margin_error=0.05):
    """
    Calculate sample size using Cochran's formula for finite populations.

    n = (Z^2 * p * (1-p)) / e^2
    adjusted = n / (1 + (n-1)/N)

    Where:
    - Z = Z-score for confidence level (1.96 for 95%, 2.576 for 99%)
    - p = population proportion (use 0.5 for maximum variance)
    - e = margin of error
    - N = population size
    """
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }

    z = z_scores.get(confidence_level, 1.96)
    p = 0.5  # Maximum variance assumption

    # Cochran's formula
    n = (z**2 * p * (1 - p)) / (margin_error**2)

    # Finite population correction
    adjusted_n = n / (1 + ((n - 1) / population_size))

    return math.ceil(adjusted_n)

def stratified_sample_size(tool_counts, total_population, confidence=0.95, margin=0.05):
    """
    Calculate sample size for stratified sampling across tools.
    Ensures minimum representation per stratum (tool).
    """
    num_strata = len(tool_counts)

    # Minimum samples per stratum for reliable statistics
    # Rule of thumb: 5-10 per stratum for pattern detection
    min_per_stratum = 8

    # Base sample size from formula
    base_sample = calculate_sample_size(total_population, confidence, margin)

    # Ensure minimum per stratum
    stratum_minimum = min_per_stratum * num_strata

    # Use the larger of the two
    recommended = max(base_sample, stratum_minimum)

    return {
        'base_statistical_sample': base_sample,
        'stratum_minimum': stratum_minimum,
        'recommended_sample': recommended,
        'num_tools': num_strata,
        'min_per_tool': min_per_stratum
    }

def main():
    # Load full dataset and analyze tool distribution
    dataset_path = Path(__file__).parent.parent / "Datasets" / "syngen_tools_sft_merged_complete_11.21.25.jsonl"

    print("Analyzing full dataset tool distribution...")
    tool_counts = Counter()
    total = 0

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                total += 1
                tool = extract_tool_name(example)
                if tool:
                    tool_counts[tool] += 1
            except:
                continue

    print(f"\nDataset Analysis:")
    print(f"  Total examples: {total}")
    print(f"  Unique tools found: {len(tool_counts)}")
    print(f"  Tools with examples: {sum(1 for c in tool_counts.values() if c > 0)}")

    # Show top 10 tools
    print(f"\n  Top 10 tools by frequency:")
    for tool, count in tool_counts.most_common(10):
        print(f"    {tool}: {count} ({count/total*100:.1f}%)")

    # Calculate sample sizes for different scenarios
    print("\n" + "="*70)
    print("STATISTICAL SAMPLE SIZE RECOMMENDATIONS")
    print("="*70)

    scenarios = [
        (0.95, 0.05, "95% confidence, ±5% margin (Standard)"),
        (0.95, 0.03, "95% confidence, ±3% margin (Stricter)"),
        (0.99, 0.05, "99% confidence, ±5% margin (High confidence)"),
    ]

    for conf, margin, desc in scenarios:
        sample = calculate_sample_size(total, conf, margin)
        print(f"\n{desc}:")
        print(f"  Required sample: {sample} ({sample/total*100:.1f}% of dataset)")

    # Stratified sampling recommendation
    print("\n" + "="*70)
    print("STRATIFIED SAMPLING RECOMMENDATION (Tool Coverage)")
    print("="*70)

    strat = stratified_sample_size(tool_counts, total)

    print(f"\nBased on {strat['num_tools']} unique tools:")
    print(f"  Base statistical sample: {strat['base_statistical_sample']} examples")
    print(f"  Minimum for tool coverage: {strat['stratum_minimum']} examples")
    print(f"    ({strat['min_per_tool']} per tool × {strat['num_tools']} tools)")
    print(f"  \n  🎯 RECOMMENDED SAMPLE: {strat['recommended_sample']} examples")
    print(f"     ({strat['recommended_sample']/total*100:.1f}% of dataset)")

    # Current progress
    print("\n" + "="*70)
    print("CURRENT PROGRESS")
    print("="*70)

    scored_path = Path(__file__).parent.parent / "Datasets" / "quality_review" / "scored_complete.jsonl"
    if scored_path.exists():
        scored_count = sum(1 for _ in open(scored_path))
        print(f"\n  Already scored: {scored_count} examples")
        print(f"  Progress: {scored_count/total*100:.1f}% of dataset")
        print(f"  Progress toward recommended: {scored_count/strat['recommended_sample']*100:.1f}%")

        remaining = strat['recommended_sample'] - scored_count
        if remaining > 0:
            rounds_needed = math.ceil(remaining / 150)
            print(f"\n  Remaining to reach target: {remaining} examples")
            print(f"  Rounds needed (150 per round): {rounds_needed}")
            print(f"  Total when complete: {scored_count + (rounds_needed * 150)} examples")

    # Tool coverage in current sample
    print("\n" + "="*70)
    print("TOOL COVERAGE ANALYSIS (Current 300 scored)")
    print("="*70)

    if scored_path.exists():
        scored_tools = Counter()
        with open(scored_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    tool = extract_tool_name(example)
                    if tool:
                        scored_tools[tool] += 1
                except:
                    continue

        print(f"\n  Tools represented in sample: {len(scored_tools)}/{len(tool_counts)}")
        print(f"  Coverage: {len(scored_tools)/len(tool_counts)*100:.1f}%")

        # Find underrepresented tools
        underrep = []
        for tool in tool_counts:
            if tool not in scored_tools or scored_tools[tool] < strat['min_per_tool']:
                underrep.append((tool, scored_tools.get(tool, 0)))

        if underrep:
            print(f"\n  Underrepresented tools (< {strat['min_per_tool']} examples): {len(underrep)}")
            print(f"  Top 10 needing more coverage:")
            for tool, count in sorted(underrep, key=lambda x: x[1])[:10]:
                print(f"    {tool}: {count} scored")

if __name__ == "__main__":
    main()
