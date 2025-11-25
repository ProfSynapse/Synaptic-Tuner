#!/usr/bin/env python3
"""
Relabel low-quality examples from true to false for KTO preference training.

Criteria for relabeling (true -> false):
1. Overall quality score < 3.0 (below quality standards)
2. sessionMemory_quality == 1 (missing/empty sessionMemory)

This preserves examples for contrastive learning while marking them as undesirable.
"""

import json
from pathlib import Path
from collections import Counter

def main():
    # Paths
    input_file = Path("Datasets/quality_review/scored_complete.jsonl")
    output_file = Path("Datasets/quality_review/scored_complete_relabeled.jsonl")

    print(f"Reading scored examples from: {input_file}")

    # Load all scored examples
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} scored examples")

    # Track relabeling statistics
    stats = {
        'total': len(examples),
        'originally_true': 0,
        'relabeled_to_false': 0,
        'reason_low_quality': 0,
        'reason_missing_memory': 0,
        'reason_both': 0,
        'already_false': 0
    }

    # Relabel examples
    relabeled_examples = []
    for ex in examples:
        original_label = ex.get('label', True)

        if original_label:
            stats['originally_true'] += 1
        else:
            stats['already_false'] += 1

        # Check relabeling criteria
        quality_scores = ex.get('quality_scores', {})
        overall_quality = quality_scores.get('overall_quality', 5.0)
        sessionMemory_quality = quality_scores.get('sessionMemory_quality', 5)

        should_relabel = False
        low_quality = overall_quality < 3.0
        missing_memory = sessionMemory_quality == 1

        if low_quality and missing_memory:
            should_relabel = True
            stats['reason_both'] += 1
        elif low_quality:
            should_relabel = True
            stats['reason_low_quality'] += 1
        elif missing_memory:
            should_relabel = True
            stats['reason_missing_memory'] += 1

        # Apply relabeling
        if should_relabel and original_label:
            ex['label'] = False
            stats['relabeled_to_false'] += 1

            # Add relabeling metadata
            if 'quality_scores' in ex:
                ex['quality_scores']['relabeled'] = True
                ex['quality_scores']['relabel_reason'] = []
                if low_quality:
                    ex['quality_scores']['relabel_reason'].append(f"low_quality ({overall_quality:.1f})")
                if missing_memory:
                    ex['quality_scores']['relabel_reason'].append("missing_sessionMemory")

        relabeled_examples.append(ex)

    # Write relabeled dataset
    print(f"\nWriting relabeled dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in relabeled_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Print statistics
    print("\n" + "="*60)
    print("RELABELING STATISTICS")
    print("="*60)
    print(f"Total examples:              {stats['total']}")
    print(f"Originally labeled true:     {stats['originally_true']}")
    print(f"Originally labeled false:    {stats['already_false']}")
    print(f"\nRelabeled true → false:      {stats['relabeled_to_false']}")
    print(f"  Due to low quality:        {stats['reason_low_quality']}")
    print(f"  Due to missing memory:     {stats['reason_missing_memory']}")
    print(f"  Due to both:               {stats['reason_both']}")
    print(f"\nFinal label distribution:")

    final_true = stats['originally_true'] - stats['relabeled_to_false']
    final_false = stats['already_false'] + stats['relabeled_to_false']
    print(f"  True (desirable):          {final_true} ({final_true/stats['total']*100:.1f}%)")
    print(f"  False (undesirable):       {final_false} ({final_false/stats['total']*100:.1f}%)")

    print("\n✅ Relabeling complete!")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
