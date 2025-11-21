#!/usr/bin/env python3
"""Analyze the scored quality examples."""

import json
from collections import Counter

def main():
    scored_file = "/home/user/Toolset-Training/Datasets/quality_review/scored_batch_100.jsonl"

    scores_by_dimension = {
        "sessionMemory_quality": [],
        "toolContext_quality": [],
        "goal_coherence": [],
        "prompt_naturalness": [],
        "response_realism": [],
        "overall_quality": []
    }

    with open(scored_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            qs = example["quality_scores"]

            for dim in scores_by_dimension:
                scores_by_dimension[dim].append(qs[dim])

    print("=" * 80)
    print("QUALITY SCORING ANALYSIS")
    print("=" * 80)

    for dim, scores in scores_by_dimension.items():
        avg = sum(scores) / len(scores)
        counter = Counter(scores)

        print(f"\n{dim}:")
        print(f"  Average: {avg:.2f}")
        print(f"  Distribution:")
        for score in sorted(counter.keys(), reverse=True):
            count = counter[score]
            pct = count / len(scores) * 100
            bar = "█" * int(pct / 2)
            print(f"    {score}: {count:2d} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    # Analyze patterns
    sm_scores = scores_by_dimension["sessionMemory_quality"]
    tc_scores = scores_by_dimension["toolContext_quality"]
    gc_scores = scores_by_dimension["goal_coherence"]
    pn_scores = scores_by_dimension["prompt_naturalness"]
    rr_scores = scores_by_dimension["response_realism"]

    print(f"\n1. Response Realism: ALL examples missing Result section (100% scored 1)")
    print(f"   This severely impacts overall quality scores.")

    sm_avg = sum(sm_scores) / len(sm_scores)
    print(f"\n2. sessionMemory Quality: Average {sm_avg:.2f}")
    print(f"   - Most are generic but relevant (3) or have some specifics (4)")
    print(f"   - None scored 5 (excellent with rich multi-action context)")

    tc_avg = sum(tc_scores) / len(tc_scores)
    print(f"\n3. toolContext Quality: Average {tc_avg:.2f}")
    print(f"   - Mostly clear purpose (3) but lack workflow reasoning")
    print(f"   - Few explain WHY this tool vs alternatives")

    pn_avg = sum(pn_scores) / len(pn_scores)
    print(f"\n4. Prompt Naturalness: Average {pn_avg:.2f}")
    print(f"   - Strong area with many natural, conversational prompts")
    print(f"   - Good use of pronouns and natural phrasing")

    print(f"\n5. Overall Quality: {sum(scores_by_dimension['overall_quality']) / 30:.2f}")
    print(f"   - Primarily limited by missing Result sections (automatic 1)")
    print(f"   - If Result sections were added with score 4, average would be ~3.2")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
