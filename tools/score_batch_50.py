#!/usr/bin/env python3
"""
Score examples from sample_batch_50.jsonl according to INTERACTION_QUALITY_RUBRIC.md
"""

import json
import re
from typing import Dict, List, Any

def extract_context(assistant_content: str) -> Dict[str, Any]:
    """Extract context object from assistant tool call."""
    try:
        # Find the arguments section
        if 'arguments:' in assistant_content:
            args_start = assistant_content.index('arguments:') + len('arguments:')
            args_text = assistant_content[args_start:].strip()
            # Parse JSON
            args = json.loads(args_text)
            return args.get('context', {})
    except:
        pass
    return {}

def score_session_memory(session_memory: str) -> tuple[int, str]:
    """Score sessionMemory quality (1-5)."""
    if not session_memory or session_memory in ['', '[]', 'None', 'N/A']:
        return 1, "Empty sessionMemory"

    length = len(session_memory)

    if length < 20:
        return 2, f"Very short sessionMemory ({length} chars), minimal context"
    elif length < 50:
        if any(word in session_memory.lower() for word in ['created', 'searched', 'found', 'listed', 'read']):
            return 3, f"Generic but relevant ({length} chars), mentions some actions"
        return 2, f"Short and generic ({length} chars)"
    elif length < 100:
        # Check for specifics
        has_numbers = bool(re.search(r'\d+', session_memory))
        has_paths = bool(re.search(r'[/\\]', session_memory))
        has_actions = sum(1 for word in ['created', 'searched', 'found', 'listed', 'read', 'updated', 'moved']
                         if word in session_memory.lower())

        if has_numbers or has_paths or has_actions >= 2:
            return 4, f"Good details ({length} chars), includes specific actions/paths/numbers"
        return 3, f"Average detail ({length} chars)"
    else:
        # 100+ chars
        has_rich_context = bool(re.search(r'(searched|found|created|listed|read|updated).*\d+', session_memory.lower()))
        if has_rich_context or session_memory.count(',') >= 2:
            return 5, f"Excellent rich context ({length} chars), multiple specific details"
        return 4, f"Good length ({length} chars) with solid details"

def score_tool_context(tool_context: str) -> tuple[int, str]:
    """Score toolContext quality (1-5)."""
    if not tool_context or tool_context in ['', 'Using tool', 'Performing action']:
        return 1, "Empty or generic toolContext"

    # Check if it just restates tool name
    if len(tool_context) < 20:
        return 2, f"Minimal toolContext ({len(tool_context)} chars), barely more than tool name"

    # Check for reasoning words
    reasoning_words = ['because', 'to confirm', 'to verify', 'need to', 'after', 'before', 'instead of']
    workflow_words = ['first', 'then', 'next', 'final', 'setting up', 'preparing']

    has_reasoning = any(word in tool_context.lower() for word in reasoning_words)
    has_workflow = any(word in tool_context.lower() for word in workflow_words)

    if has_reasoning and has_workflow:
        return 5, f"Excellent toolContext - explains reasoning and workflow position"
    elif has_reasoning or has_workflow:
        return 4, f"Good toolContext - explains {'reasoning' if has_reasoning else 'workflow context'}"
    elif len(tool_context) > 30:
        return 3, f"Average toolContext - clear purpose but missing workflow reasoning"
    else:
        return 2, f"Below average toolContext - describes what, not why"

def score_goal_coherence(primary_goal: str, subgoal: str) -> tuple[int, str]:
    """Score goal hierarchy coherence (1-5)."""
    if not primary_goal or not subgoal:
        return 1, "Missing primary_goal or subgoal"

    if primary_goal == subgoal:
        return 1, "Identical primary_goal and subgoal"

    # Check similarity
    primary_words = set(primary_goal.lower().split())
    subgoal_words = set(subgoal.lower().split())
    overlap = len(primary_words & subgoal_words) / max(len(primary_words), len(subgoal_words))

    if overlap > 0.8:
        return 2, f"Goals overlap significantly ({overlap:.0%} similar)"

    # Check if subgoal is more specific
    is_specific_subgoal = len(subgoal) > len(primary_goal) * 0.7
    has_action_words = any(word in subgoal.lower() for word in ['create', 'find', 'update', 'search', 'list', 'delete', 'move'])

    if is_specific_subgoal and has_action_words:
        return 4, "Good hierarchy - subgoal is specific and actionable"
    elif has_action_words:
        return 3, "Average hierarchy - clear distinction but could be more specific"
    else:
        return 2, "Weak hierarchy - goals are vague"

def score_prompt_naturalness(user_content: str) -> tuple[int, str]:
    """Score how natural the user prompt is (1-5)."""
    # Check if it's a Result object (unnatural)
    if user_content.strip().startswith('Result:') or user_content.strip().startswith('{'):
        return 1, "User message is a Result/JSON object, not natural language"

    # Check for command-style
    if user_content.startswith(('execute_', 'run_', 'call_')) or '--' in user_content:
        return 1, "Pure command syntax, no natural language"

    # Check length
    if len(user_content) < 10:
        return 2, f"Very terse ({len(user_content)} chars), telegraphic"

    # Natural language indicators
    has_pronouns = any(word in user_content.lower() for word in ['i', 'my', 'me', 'we', 'our'])
    has_questions = '?' in user_content
    has_please = 'please' in user_content.lower() or 'can you' in user_content.lower()
    has_casual = any(word in user_content.lower() for word in ['just', 'really', 'kinda', 'gonna', 'wanna'])

    natural_score = sum([has_pronouns, has_questions, has_please, has_casual])

    if natural_score >= 3:
        return 5, "Highly natural - conversational with pronouns, questions, polite language"
    elif natural_score >= 2:
        return 4, "Natural phrasing with personal pronouns and conversational elements"
    elif natural_score >= 1 or len(user_content) > 30:
        return 3, "Mix of natural and formal language"
    else:
        return 2, "Formal or command-like phrasing"

def score_response_realism(assistant_content: str) -> tuple[int, str]:
    """Score how realistic the tool response is (1-5)."""
    # Check if Result is present
    if 'Result:' not in assistant_content:
        return 2, "No Result object shown - incomplete tool execution"

    # Extract Result
    try:
        result_start = assistant_content.index('Result:') + len('Result:')
        result_text = assistant_content[result_start:].strip()
        # Try to find JSON object
        if result_text.startswith('{'):
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(result_text):
                if char == '{': brace_count += 1
                elif char == '}': brace_count -= 1
                if brace_count == 0:
                    result_json = json.loads(result_text[:i+1])
                    break
        else:
            result_json = json.loads(result_text.split('\n')[0])

        # Score based on metadata richness
        metadata_fields = ['success', 'executionTime', 'totalResults', 'displayed',
                          'searchCapabilities', 'timestamp', 'count', 'data']
        present_fields = sum(1 for field in metadata_fields if field in str(result_json))

        if present_fields <= 1:
            return 2, f"Minimal Result structure - only {present_fields} metadata fields"
        elif present_fields <= 2:
            return 3, f"Basic Result structure - {present_fields} metadata fields"
        elif present_fields <= 4:
            return 4, f"Good Result structure - {present_fields} metadata fields with proper data"
        else:
            return 5, f"Excellent Result - rich metadata with {present_fields}+ fields"

    except:
        return 2, "Result present but malformed or unparseable"

def score_example(example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Score a single example and add quality_scores."""
    conversations = example['conversations']

    # Get user prompt and assistant response
    user_msg = conversations[0]['content'] if conversations else ""
    assistant_msg = conversations[1]['content'] if len(conversations) > 1 else ""

    # Extract context from assistant tool call
    context = extract_context(assistant_msg)

    session_memory = context.get('sessionMemory', '')
    tool_context = context.get('toolContext', '')
    primary_goal = context.get('primaryGoal', '')
    subgoal = context.get('subgoal', '')

    # Score each dimension
    sm_score, sm_notes = score_session_memory(session_memory)
    tc_score, tc_notes = score_tool_context(tool_context)
    gc_score, gc_notes = score_goal_coherence(primary_goal, subgoal)
    pn_score, pn_notes = score_prompt_naturalness(user_msg)
    rr_score, rr_notes = score_response_realism(assistant_msg)

    overall = round((sm_score + tc_score + gc_score + pn_score + rr_score) / 5, 1)

    # Compile detailed notes
    notes = (
        f"sessionMemory: {sm_notes}. "
        f"toolContext: {tc_notes}. "
        f"goal_coherence: {gc_notes}. "
        f"prompt_naturalness: {pn_notes}. "
        f"response_realism: {rr_notes}."
    )

    # Add quality_scores to example
    scored_example = example.copy()
    scored_example['quality_scores'] = {
        'notes': notes,
        'sessionMemory_quality': sm_score,
        'toolContext_quality': tc_score,
        'goal_coherence': gc_score,
        'prompt_naturalness': pn_score,
        'response_realism': rr_score,
        'overall_quality': overall
    }

    return scored_example

def main():
    # Load examples
    input_file = '/home/user/Toolset-Training/Datasets/quality_review/sample_batch_50.jsonl'
    output_file = '/home/user/Toolset-Training/Datasets/quality_review/scored_batch_50.jsonl'

    print("Loading examples...")
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Scoring {len(examples)} examples...\n")

    scored_examples = []
    for i, example in enumerate(examples):
        scored = score_example(example, i)
        scored_examples.append(scored)

        # Print summary
        scores = scored['quality_scores']
        print(f"Example {i} (index={example.get('_index', 'N/A')}): "
              f"overall={scores['overall_quality']} "
              f"[sm={scores['sessionMemory_quality']}, "
              f"tc={scores['toolContext_quality']}, "
              f"gc={scores['goal_coherence']}, "
              f"pn={scores['prompt_naturalness']}, "
              f"rr={scores['response_realism']}]")

    # Calculate statistics
    overall_scores = [ex['quality_scores']['overall_quality'] for ex in scored_examples]
    avg_overall = sum(overall_scores) / len(overall_scores)

    print(f"\n{'='*60}")
    print(f"Scoring complete!")
    print(f"Average overall_quality: {avg_overall:.2f}")
    print(f"Range: {min(overall_scores):.1f} - {max(overall_scores):.1f}")
    print(f"{'='*60}\n")

    # Save scored examples
    print(f"Writing scored examples to {output_file}...")
    with open(output_file, 'w') as f:
        for example in scored_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Saved {len(scored_examples)} scored examples")

if __name__ == '__main__':
    main()
