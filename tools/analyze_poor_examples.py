#!/usr/bin/env python3
"""
Analyze poor-quality examples to understand what needs improvement.
"""

import json
from pathlib import Path
from collections import Counter
import re

def load_scored_dataset():
    """Load the scored and relabeled dataset."""
    dataset_path = Path("Datasets/quality_review/scored_complete_relabeled.jsonl")
    examples = []

    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples

def analyze_poor_examples(examples):
    """Analyze poor-quality examples to find common issues."""
    poor_examples = [ex for ex in examples if ex['label'] == False]

    print(f"\n=== Poor Example Analysis ===")
    print(f"Total examples: {len(examples)}")
    print(f"Poor examples (label=false): {len(poor_examples)}")
    print(f"Good examples (label=true): {len(examples) - len(poor_examples)}")
    print(f"Percentage poor: {len(poor_examples)/len(examples)*100:.1f}%")

    # Analyze quality dimension scores
    print(f"\n=== Quality Dimension Breakdown (Poor Examples) ===")

    sessionMemory_scores = [ex['quality_scores']['sessionMemory_quality'] for ex in poor_examples]
    toolContext_scores = [ex['quality_scores']['toolContext_quality'] for ex in poor_examples]
    goal_scores = [ex['quality_scores']['goal_coherence'] for ex in poor_examples]
    prompt_scores = [ex['quality_scores']['prompt_naturalness'] for ex in poor_examples]
    response_scores = [ex['quality_scores']['response_realism'] for ex in poor_examples]
    overall_scores = [ex['quality_scores']['overall_quality'] for ex in poor_examples]

    print(f"sessionMemory: avg={sum(sessionMemory_scores)/len(sessionMemory_scores):.2f}")
    print(f"  - Score 1: {sessionMemory_scores.count(1)} ({sessionMemory_scores.count(1)/len(sessionMemory_scores)*100:.1f}%)")
    print(f"  - Score 2: {sessionMemory_scores.count(2)} ({sessionMemory_scores.count(2)/len(sessionMemory_scores)*100:.1f}%)")
    print(f"  - Score 3: {sessionMemory_scores.count(3)} ({sessionMemory_scores.count(3)/len(sessionMemory_scores)*100:.1f}%)")

    print(f"\ntoolContext: avg={sum(toolContext_scores)/len(toolContext_scores):.2f}")
    print(f"  - Score 1: {toolContext_scores.count(1)} ({toolContext_scores.count(1)/len(toolContext_scores)*100:.1f}%)")
    print(f"  - Score 2: {toolContext_scores.count(2)} ({toolContext_scores.count(2)/len(toolContext_scores)*100:.1f}%)")
    print(f"  - Score 3: {toolContext_scores.count(3)} ({toolContext_scores.count(3)/len(toolContext_scores)*100:.1f}%)")

    print(f"\ngoal_coherence: avg={sum(goal_scores)/len(goal_scores):.2f}")
    print(f"response_realism: avg={sum(response_scores)/len(response_scores):.2f}")
    print(f"prompt_naturalness: avg={sum(prompt_scores)/len(prompt_scores):.2f}")
    print(f"overall_quality: avg={sum(overall_scores)/len(overall_scores):.2f}")

    # Find most common issues from notes
    print(f"\n=== Common Issues (from notes) ===")

    empty_sessionMemory = 0
    generic_toolContext = 0
    missing_result = 0
    weak_goals = 0
    unnatural_prompt = 0

    for ex in poor_examples:
        notes = ex['quality_scores']['notes'].lower()

        if 'sessionmemory is empty' in notes or 'sessionmemory: ""' in notes or 'sessionmemory: []' in notes:
            empty_sessionMemory += 1

        if 'generic' in notes and 'toolcontext' in notes:
            generic_toolContext += 1

        if 'no result' in notes or 'missing result' in notes:
            missing_result += 1

        if 'primarygoal' in notes and 'subgoal' in notes and ('similar' in notes or 'weak' in notes):
            weak_goals += 1

        if 'prompt' in notes and ('unnatural' in notes or 'result object' in notes):
            unnatural_prompt += 1

    print(f"Empty sessionMemory: {empty_sessionMemory} ({empty_sessionMemory/len(poor_examples)*100:.1f}%)")
    print(f"Generic toolContext: {generic_toolContext} ({generic_toolContext/len(poor_examples)*100:.1f}%)")
    print(f"Missing Result object: {missing_result} ({missing_result/len(poor_examples)*100:.1f}%)")
    print(f"Weak goal hierarchy: {weak_goals} ({weak_goals/len(poor_examples)*100:.1f}%)")
    print(f"Unnatural prompt: {unnatural_prompt} ({unnatural_prompt/len(poor_examples)*100:.1f}%)")

    # Show sample poor examples
    print(f"\n=== Sample Poor Examples ===")
    for i, ex in enumerate(poor_examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"Overall Quality: {ex['quality_scores']['overall_quality']}")
        print(f"User Message: {ex['conversations'][0]['content'][:100]}...")
        print(f"Issues: {ex['quality_scores']['notes'][:200]}...")
        print()

    return poor_examples

def main():
    examples = load_scored_dataset()
    poor_examples = analyze_poor_examples(examples)

    # Save poor examples for improvement
    output_path = Path("Datasets/quality_review/poor_examples.jsonl")
    with open(output_path, 'w') as f:
        for ex in poor_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved {len(poor_examples)} poor examples to {output_path}")

if __name__ == "__main__":
    main()
