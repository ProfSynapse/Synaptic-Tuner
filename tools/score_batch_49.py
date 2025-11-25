#!/usr/bin/env python3
"""Score batch 49 examples using the interaction quality rubric."""

import json
import re
from typing import Dict, Any, List


def extract_tool_call_context(assistant_content: str) -> Dict[str, Any]:
    """Extract the context object from assistant's tool call."""
    try:
        # Find the arguments JSON
        match = re.search(r'arguments:\s*({.*})', assistant_content, re.DOTALL)
        if not match:
            return {}

        args_str = match.group(1)
        args = json.loads(args_str)
        return args.get('context', {})
    except Exception as e:
        print(f"Error extracting context: {e}")
        return {}


def score_session_memory(session_memory: Any) -> tuple[int, str]:
    """Score sessionMemory quality (1-5)."""
    # Handle different types
    if session_memory == "" or session_memory == [] or session_memory is None:
        return 1, "Empty sessionMemory (empty string/array)"

    if isinstance(session_memory, dict):
        # toolContext should be string, not object
        return 1, "sessionMemory is a dict/object (wrong format - should be string)"

    text = str(session_memory)
    length = len(text)

    # Check for placeholder text
    placeholders = ["none", "n/a", "user working", "null"]
    if text.lower().strip() in placeholders:
        return 1, f"Placeholder text: '{text}'"

    # Score based on length and specificity
    if length < 20:
        return 2, f"Very short ({length} chars): '{text}' - lacks detail"
    elif length < 50:
        # Check for specifics
        has_numbers = bool(re.search(r'\d+', text))
        has_path = bool(re.search(r'/', text))
        has_tool_ref = any(word in text.lower() for word in ['created', 'searched', 'listed', 'moved', 'deleted', 'updated'])

        if has_numbers or has_path or has_tool_ref:
            return 3, f"Some context ({length} chars) with specifics: '{text}'"
        else:
            return 3, f"Generic but relevant ({length} chars): '{text}'"
    elif length < 100:
        # Check for concrete details
        has_numbers = bool(re.search(r'\d+', text))
        has_path = bool(re.search(r'/', text))
        has_specific_action = any(word in text.lower() for word in ['created', 'searched', 'listed', 'moved', 'deleted'])

        if sum([has_numbers, has_path, has_specific_action]) >= 2:
            return 4, f"Specific with concrete details ({length} chars): numbers/paths/actions present"
        else:
            return 3, f"Moderate detail ({length} chars) but could be more specific"
    else:
        # 100+ chars - check for richness
        has_multiple_actions = len(re.findall(r'\b(created|searched|listed|moved|deleted|updated|read|wrote)\b', text.lower())) >= 2
        has_tool_names = bool(re.search(r'(Manager|Librarian|vault|content)', text))
        has_numbers = bool(re.search(r'\d+', text))

        if sum([has_multiple_actions, has_tool_names, has_numbers]) >= 2:
            return 5, f"Rich context ({length} chars) with multiple actions, tool references, and concrete details"
        else:
            return 4, f"Good length ({length} chars) with specific details"


def score_tool_context(tool_context: Any) -> tuple[int, str]:
    """Score toolContext quality (1-5)."""
    if tool_context == "" or tool_context is None:
        return 1, "Empty toolContext"

    if isinstance(tool_context, dict):
        return 1, f"toolContext is a dict/object (wrong format - should be string explaining WHY)"

    text = str(tool_context)
    length = len(text)

    # Generic phrases
    generic_phrases = ["using tool", "performing action", "searching", "creating", "deleting", "updating"]
    if text.lower().strip() in generic_phrases:
        return 1, f"Generic: '{text}' - just restates action"

    # Very short/generic
    if length < 20:
        return 2, f"Short and generic ({length} chars): '{text}' - describes WHAT not WHY"

    # Check for reasoning words
    reasoning_words = ['because', 'to verify', 'to confirm', 'need to', 'before', 'after', 'since', 'so that']
    has_reasoning = any(word in text.lower() for word in reasoning_words)

    # Check for workflow context
    workflow_words = ['first', 'then', 'next', 'after', 'before', 'instead of', 'rather than']
    has_workflow = any(word in text.lower() for word in workflow_words)

    if length < 50:
        if has_reasoning:
            return 3, f"Clear purpose with some reasoning: '{text}'"
        else:
            return 3, f"Clear immediate goal but missing workflow reasoning: '{text}'"
    elif length < 100:
        if has_reasoning and has_workflow:
            return 4, f"Explains reasoning and workflow context ({length} chars)"
        elif has_reasoning or has_workflow:
            return 4, f"Good explanation with reasoning/workflow context: '{text}'"
        else:
            return 3, f"Decent length but lacks deep reasoning"
    else:
        if has_reasoning and has_workflow:
            return 5, f"Rich reasoning ({length} chars) with workflow strategy and decision-making"
        else:
            return 4, f"Good detail ({length} chars) but could show more strategic thinking"


def score_goal_coherence(primary_goal: str, subgoal: str) -> tuple[int, str]:
    """Score goal coherence (1-5)."""
    if not primary_goal or not subgoal:
        return 1, f"Missing goal(s): primary='{primary_goal}', sub='{subgoal}'"

    if primary_goal == subgoal:
        return 1, f"Identical goals: both are '{primary_goal}'"

    # Calculate similarity
    p_words = set(primary_goal.lower().split())
    s_words = set(subgoal.lower().split())
    if len(p_words) > 0:
        overlap = len(p_words & s_words) / len(p_words)
    else:
        overlap = 0

    if overlap > 0.9:
        return 2, f"Goals overlap significantly (90%+): '{primary_goal}' vs '{subgoal}'"

    # Check for clear hierarchy (subgoal should be more specific)
    is_hierarchical = len(subgoal) >= len(primary_goal) * 0.7  # Subgoal somewhat detailed

    # Check for generic goals
    generic_terms = ['work on', 'do task', 'perform action', 'use tool']
    is_generic = any(term in primary_goal.lower() or term in subgoal.lower() for term in generic_terms)

    if is_generic:
        return 2, f"Vague goals: '{primary_goal}' → '{subgoal}'"

    # Check specificity
    has_specifics_primary = bool(re.search(r'[A-Z][a-z]+|\.md|\.txt|/|\d+', primary_goal))
    has_specifics_sub = bool(re.search(r'[A-Z][a-z]+|\.md|\.txt|/|\d+', subgoal))

    if has_specifics_primary and has_specifics_sub:
        return 4, f"Specific and actionable hierarchy: '{primary_goal}' → '{subgoal}'"
    elif has_specifics_primary or has_specifics_sub:
        return 3, f"Clear distinction, proper hierarchy: '{primary_goal}' → '{subgoal}'"
    else:
        return 3, f"Clear but generic: '{primary_goal}' → '{subgoal}'"


def score_prompt_naturalness(user_content: str) -> tuple[int, str]:
    """Score prompt naturalness (1-5)."""
    # Check if this is a Result message (continuation)
    if user_content.strip().startswith("Result:"):
        return 1, "User content is a Result object, not a natural prompt (conversation continuation)"

    # Check for command syntax
    if re.match(r'^[a-z_]+\s*--', user_content):
        return 1, f"Pure command syntax: '{user_content[:50]}...'"

    # Check for unrealistic precision
    if 'exactly' in user_content.lower() and re.search(r'\d{3,}', user_content):
        return 1, f"Unrealistic precision: '{user_content}'"

    # Score based on naturalness features
    features = {
        'pronouns': bool(re.search(r'\b(I|my|me|we|our|you)\b', user_content, re.IGNORECASE)),
        'contractions': bool(re.search(r"(can't|won't|I'm|it's|that's|don't)", user_content, re.IGNORECASE)),
        'questions': bool(re.search(r'\?', user_content)),
        'casual_words': bool(re.search(r'\b(grab|pull up|check out|take a look|find)\b', user_content, re.IGNORECASE)),
        'implicit_ref': bool(re.search(r'\b(that|this|those|these)\b', user_content, re.IGNORECASE)),
        'corrections': bool(re.search(r'\b(actually|instead|rather|or|wait)\b', user_content, re.IGNORECASE)),
        'natural_connectors': bool(re.search(r'\b(and then|after that|before|first|next)\b', user_content, re.IGNORECASE)),
    }

    feature_count = sum(features.values())
    length = len(user_content)

    # Check for command-style terseness
    is_terse = length < 30 and not features['pronouns']

    if is_terse:
        return 2, f"Command-style, terse ({length} chars): '{user_content}'"

    if feature_count == 0:
        return 2, f"Formal, no natural language features: '{user_content}'"
    elif feature_count <= 2:
        return 3, f"Mix of natural and formal ({feature_count} natural features): '{user_content}'"
    elif feature_count <= 4:
        return 4, f"Natural phrasing ({feature_count} natural features): uses {', '.join([k for k,v in features.items() if v])}"
    else:
        return 5, f"Highly natural and conversational ({feature_count} natural features): {', '.join([k for k,v in features.items() if v])}"


def score_response_realism(user_content: str, assistant_content: str) -> tuple[int, str]:
    """Score response realism (1-5)."""
    # Check if there's a Result in user content (from previous tool call)
    result_data = None
    if user_content.strip().startswith("Result:"):
        try:
            result_json = user_content.strip()[7:].strip()
            result_data = json.loads(result_json)
        except:
            pass

    # If no Result shown, check if assistant provides one
    has_result = result_data is not None

    if not has_result:
        # No result shown at all
        return 1, "No Result shown in conversation"

    # Score the Result structure
    if not isinstance(result_data, dict):
        return 1, "Result is not a structured object"

    # Check for minimal structure
    if result_data == {"success": True} or result_data == {"success": False}:
        return 2, "Minimal Result structure: only success field"

    # Check for basic fields
    has_success = 'success' in result_data
    has_metadata = any(k in result_data for k in ['totalResults', 'count', 'executionTime', 'timestamp'])
    has_data = any(k in result_data for k in ['data', 'results', 'agent', 'session', 'workspace', 'content'])
    has_error = 'error' in result_data

    field_count = sum([has_success, has_metadata, has_data])

    if field_count <= 1:
        return 2, f"Basic structure with minimal fields: {list(result_data.keys())}"
    elif field_count == 2:
        if has_metadata:
            return 3, f"Proper structure with basic metadata: {list(result_data.keys())}"
        else:
            return 3, f"Proper structure, basic fields: {list(result_data.keys())}"
    else:
        # Check for rich metadata
        rich_fields = ['executionTime', 'searchCapabilities', 'displayed', 'confidence', 'score', 'warnings', 'timestamp']
        rich_count = sum(k in str(result_data) for k in rich_fields)

        if rich_count >= 2:
            return 5, f"Comprehensive metadata: {list(result_data.keys())} with {rich_count} rich fields"
        elif has_metadata:
            return 4, f"Good structure with metadata: {list(result_data.keys())}"
        else:
            return 3, f"Adequate structure: {list(result_data.keys())}"


def score_example(example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Score a single example."""
    conversations = example['conversations']
    user_msg = conversations[0]['content']
    assistant_msg = conversations[1]['content']

    # Extract context from tool call
    context = extract_tool_call_context(assistant_msg)

    # Score each dimension
    sm_score, sm_notes = score_session_memory(context.get('sessionMemory', ''))
    tc_score, tc_notes = score_tool_context(context.get('toolContext', ''))
    gc_score, gc_notes = score_goal_coherence(
        context.get('primaryGoal', ''),
        context.get('subgoal', '')
    )
    pn_score, pn_notes = score_prompt_naturalness(user_msg)
    rr_score, rr_notes = score_response_realism(user_msg, assistant_msg)

    # Calculate overall
    overall = round((sm_score + tc_score + gc_score + pn_score + rr_score) / 5, 1)

    # Compile detailed notes
    notes = (
        f"sessionMemory: {sm_notes}. "
        f"toolContext: {tc_notes}. "
        f"goal_coherence: {gc_notes}. "
        f"prompt_naturalness: {pn_notes}. "
        f"response_realism: {rr_notes}."
    )

    # Create scored example
    scored = example.copy()
    scored['quality_scores'] = {
        'notes': notes,
        'sessionMemory_quality': sm_score,
        'toolContext_quality': tc_score,
        'goal_coherence': gc_score,
        'prompt_naturalness': pn_score,
        'response_realism': rr_score,
        'overall_quality': overall
    }
    scored['_index'] = index

    return scored


def main():
    """Score all examples in batch 49."""
    # Read input
    input_path = 'Datasets/quality_review/sample_batch_49.jsonl'
    output_path = 'Datasets/quality_review/scored_batch_49.jsonl'

    with open(input_path, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Scoring {len(examples)} examples from batch 49...")

    # Score each example
    scored_examples = []
    for i, example in enumerate(examples):
        scored = score_example(example, i)
        scored_examples.append(scored)

        # Print progress
        scores = scored['quality_scores']
        print(f"[{i}] Overall: {scores['overall_quality']:.1f} | "
              f"SM:{scores['sessionMemory_quality']} TC:{scores['toolContext_quality']} "
              f"GC:{scores['goal_coherence']} PN:{scores['prompt_naturalness']} "
              f"RR:{scores['response_realism']}")

    # Write output
    with open(output_path, 'w') as f:
        for scored in scored_examples:
            f.write(json.dumps(scored) + '\n')

    print(f"\nScored examples saved to: {output_path}")

    # Print summary statistics
    overall_scores = [s['quality_scores']['overall_quality'] for s in scored_examples]
    print(f"\nSummary Statistics:")
    print(f"  Mean: {sum(overall_scores) / len(overall_scores):.2f}")
    print(f"  Min:  {min(overall_scores):.1f}")
    print(f"  Max:  {max(overall_scores):.1f}")
    print(f"  Excellent (4.0+): {sum(1 for s in overall_scores if s >= 4.0)} examples")
    print(f"  Good (3.0-3.9):   {sum(1 for s in overall_scores if 3.0 <= s < 4.0)} examples")
    print(f"  Fair (2.0-2.9):   {sum(1 for s in overall_scores if 2.0 <= s < 3.0)} examples")
    print(f"  Poor (<2.0):      {sum(1 for s in overall_scores if s < 2.0)} examples")


if __name__ == '__main__':
    main()
