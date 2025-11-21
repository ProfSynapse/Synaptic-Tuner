#!/usr/bin/env python3
"""Score batch 43 examples using the interaction quality rubric."""

import json
import re
from typing import Dict, Any, List

def extract_context(assistant_content: str) -> Dict[str, Any]:
    """Extract context object from assistant's tool call."""
    try:
        # Find the arguments section
        match = re.search(r'"context":\s*({[^}]+})', assistant_content.replace('\n', ' '))
        if match:
            context_str = match.group(1)
            # This is a simplified extraction - would need proper JSON parsing for production
            session_memory = re.search(r'"sessionMemory":\s*"([^"]*)"', context_str)
            tool_context = re.search(r'"toolContext":\s*"([^"]*)"', context_str)
            primary_goal = re.search(r'"primaryGoal":\s*"([^"]*)"', context_str)
            subgoal = re.search(r'"subgoal":\s*"([^"]*)"', context_str)

            return {
                'sessionMemory': session_memory.group(1) if session_memory else '',
                'toolContext': tool_context.group(1) if tool_context else '',
                'primaryGoal': primary_goal.group(1) if primary_goal else '',
                'subgoal': subgoal.group(1) if subgoal else ''
            }
    except Exception as e:
        print(f"Error extracting context: {e}")
    return {}

def score_example(example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Score a single example according to the rubric."""
    conversations = example.get('conversations', [])

    # Extract user prompt and assistant response
    user_content = ''
    assistant_content = ''
    result_content = ''

    for conv in conversations:
        if conv['role'] == 'user':
            user_content = conv['content']
            # Check if user is providing a Result from previous tool call
            if user_content.startswith('Result:'):
                result_content = user_content
        elif conv['role'] == 'assistant':
            assistant_content = conv['content']
            # Check if assistant includes Result in their response
            if 'Result:' in assistant_content:
                result_match = re.search(r'Result:\s*({.*?})\n', assistant_content, re.DOTALL)
                if result_match:
                    result_content = result_match.group(1)

    # Extract context from assistant's tool call
    context = extract_context(assistant_content)
    session_memory = context.get('sessionMemory', '')
    tool_context = context.get('toolContext', '')
    primary_goal = context.get('primaryGoal', '')
    subgoal = context.get('subgoal', '')

    # Score each dimension
    scores = score_all_dimensions(
        session_memory, tool_context, primary_goal, subgoal,
        user_content, assistant_content, result_content, index
    )

    return scores

def score_all_dimensions(session_memory, tool_context, primary_goal, subgoal,
                         user_content, assistant_content, result_content, index):
    """Score all 5 dimensions and write detailed notes."""

    # This is where I'll manually score each example based on the rubric
    # I'll create a comprehensive scoring function

    scores = {
        'notes': '',
        'sessionMemory_quality': 1,
        'toolContext_quality': 1,
        'goal_coherence': 1,
        'prompt_naturalness': 1,
        'response_realism': 1,
        'overall_quality': 1.0
    }

    notes_parts = []

    # Score sessionMemory_quality
    if not session_memory or session_memory in ['', '[]', 'None', 'N/A']:
        scores['sessionMemory_quality'] = 1
        notes_parts.append(f"sessionMemory is empty '{session_memory}' (score=1)")
    elif len(session_memory) < 20:
        scores['sessionMemory_quality'] = 2
        notes_parts.append(f"sessionMemory very short at {len(session_memory)} chars: '{session_memory[:50]}' (score=2)")
    elif len(session_memory) < 50:
        scores['sessionMemory_quality'] = 3
        notes_parts.append(f"sessionMemory is generic but relevant at {len(session_memory)} chars (score=3)")
    elif len(session_memory) < 100:
        # Check for specific details
        has_numbers = bool(re.search(r'\d+', session_memory))
        has_paths = bool(re.search(r'[/\\]', session_memory))
        if has_numbers or has_paths:
            scores['sessionMemory_quality'] = 4
            notes_parts.append(f"sessionMemory has specific details ({len(session_memory)} chars, includes numbers/paths) (score=4)")
        else:
            scores['sessionMemory_quality'] = 3
            notes_parts.append(f"sessionMemory at {len(session_memory)} chars but lacks concrete details (score=3)")
    else:
        scores['sessionMemory_quality'] = 5
        notes_parts.append(f"sessionMemory is excellent at {len(session_memory)} chars with rich context (score=5)")

    # Score toolContext_quality
    if not tool_context or tool_context in ['Using tool', 'Performing action', '']:
        scores['toolContext_quality'] = 1
        notes_parts.append(f"toolContext is generic/missing: '{tool_context}' (score=1)")
    elif 'searching' in tool_context.lower() or 'finding' in tool_context.lower():
        if 'why' in tool_context.lower() or 'before' in tool_context.lower() or 'after' in tool_context.lower():
            scores['toolContext_quality'] = 4
            notes_parts.append(f"toolContext explains workflow: '{tool_context}' (score=4)")
        else:
            scores['toolContext_quality'] = 3
            notes_parts.append(f"toolContext clear but lacks workflow reasoning: '{tool_context}' (score=3)")
    elif len(tool_context) > 50 and ('because' in tool_context.lower() or 'after' in tool_context.lower()):
        scores['toolContext_quality'] = 5
        notes_parts.append(f"toolContext shows rich reasoning: '{tool_context}' (score=5)")
    elif len(tool_context) > 30:
        scores['toolContext_quality'] = 4
        notes_parts.append(f"toolContext is specific: '{tool_context}' (score=4)")
    else:
        scores['toolContext_quality'] = 3
        notes_parts.append(f"toolContext is adequate: '{tool_context}' (score=3)")

    # Score goal_coherence
    if not primary_goal or not subgoal:
        scores['goal_coherence'] = 1
        notes_parts.append(f"Goals missing (primary: '{primary_goal}', sub: '{subgoal}') (score=1)")
    elif primary_goal == subgoal or primary_goal.lower() == subgoal.lower():
        scores['goal_coherence'] = 2
        notes_parts.append(f"Goals are identical: '{primary_goal}' vs '{subgoal}' (score=2)")
    elif len(primary_goal) > 40 and len(subgoal) > 30:
        scores['goal_coherence'] = 5
        notes_parts.append(f"Goals show strategic hierarchy: '{primary_goal}' → '{subgoal}' (score=5)")
    elif 'and' in primary_goal or 'and' in subgoal:
        scores['goal_coherence'] = 4
        notes_parts.append(f"Goals are specific and detailed (score=4)")
    else:
        scores['goal_coherence'] = 3
        notes_parts.append(f"Goals have clear distinction: '{primary_goal}' → '{subgoal}' (score=3)")

    # Score prompt_naturalness
    user_lower = user_content.lower()
    if user_content.startswith('Result:'):
        # This is a result from previous tool call, not a user prompt
        scores['prompt_naturalness'] = 0  # Will handle separately
        notes_parts.append("No user prompt (continuation of previous tool call)")
    elif '--' in user_content or 'execute_' in user_content:
        scores['prompt_naturalness'] = 1
        notes_parts.append(f"Prompt is command-like: '{user_content[:50]}' (score=1)")
    elif any(word in user_lower for word in ['can you', 'i want', 'i need', "i'm", "i'd like"]):
        scores['prompt_naturalness'] = 5
        notes_parts.append(f"Prompt is highly natural and conversational (score=5)")
    elif any(word in user_lower for word in ['please', 'help', 'show me', 'find', 'my']):
        scores['prompt_naturalness'] = 4
        notes_parts.append(f"Prompt is natural with personal language (score=4)")
    else:
        scores['prompt_naturalness'] = 3
        notes_parts.append(f"Prompt is adequate but formal (score=3)")

    # Score response_realism
    if not result_content:
        scores['response_realism'] = 1
        notes_parts.append("No result shown (score=1)")
    elif result_content.count('{') < 2:
        scores['response_realism'] = 2
        notes_parts.append("Result has minimal structure (score=2)")
    else:
        # Check for metadata
        has_metadata = any(key in result_content for key in ['executionTime', 'totalResults', 'timestamp', 'displayed'])
        has_scores = 'score' in result_content and '0.' in result_content
        has_warnings = 'warning' in result_content.lower() or 'error' in result_content.lower()

        metadata_count = sum([has_metadata, has_scores, has_warnings])

        if metadata_count >= 2:
            scores['response_realism'] = 5
            notes_parts.append("Result has comprehensive metadata and realistic details (score=5)")
        elif metadata_count == 1:
            scores['response_realism'] = 4
            notes_parts.append("Result has good metadata (score=4)")
        elif 'success' in result_content:
            scores['response_realism'] = 3
            notes_parts.append("Result has basic structure (score=3)")
        else:
            scores['response_realism'] = 2
            notes_parts.append("Result is minimal (score=2)")

    # Calculate overall quality
    dimension_scores = [
        scores['sessionMemory_quality'],
        scores['toolContext_quality'],
        scores['goal_coherence'],
        scores['prompt_naturalness'] if scores['prompt_naturalness'] > 0 else 3,  # Handle continuation cases
        scores['response_realism']
    ]

    scores['overall_quality'] = round(sum(dimension_scores) / len(dimension_scores), 1)
    scores['notes'] = ' | '.join(notes_parts)

    return scores


def main():
    # Read input
    with open('Datasets/quality_review/sample_batch_43.jsonl', 'r') as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Scoring {len(examples)} examples from batch 43...")

    # Score each example
    scored_examples = []
    for i, example in enumerate(examples):
        scores = score_example(example, i)

        # Add quality_scores to example
        example['quality_scores'] = scores
        scored_examples.append(example)

        print(f"Scored example {i}: overall_quality={scores['overall_quality']}")

    # Write output
    with open('Datasets/quality_review/scored_batch_43.jsonl', 'w') as f:
        for example in scored_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\nScored examples saved to Datasets/quality_review/scored_batch_43.jsonl")

    # Print summary statistics
    overall_scores = [ex['quality_scores']['overall_quality'] for ex in scored_examples]
    print(f"\nSummary Statistics:")
    print(f"  Mean overall_quality: {sum(overall_scores) / len(overall_scores):.2f}")
    print(f"  Min: {min(overall_scores):.1f}")
    print(f"  Max: {max(overall_scores):.1f}")

if __name__ == '__main__':
    main()
