#!/usr/bin/env python3
"""
Score synthetic training examples using the Interaction Quality Rubric.
"""

import json
import sys

def score_session_memory(context):
    """Score sessionMemory quality (1-5)."""
    session_memory = context.get("sessionMemory", "")

    # Poor: Empty or placeholder
    if not session_memory or session_memory in ["", "None", "N/A", "[]"]:
        return 1, "Empty sessionMemory"

    length = len(session_memory)

    # Below Average: Very generic, extremely short
    if length < 20 or session_memory in ["User working", "User organizing files"]:
        return 2, f"Very generic and short ({length} chars)"

    # Average: Generic but relevant (20-50 chars)
    if length < 50:
        has_specifics = any(word in session_memory.lower() for word in [
            "created", "listed", "searched", "found", "folder", "file", "path"
        ])
        if not has_specifics:
            return 3, f"Generic but relevant ({length} chars)"

    # Good: Specific actions, concrete details (50-100 chars)
    if length < 100:
        has_numbers = any(char.isdigit() for char in session_memory)
        has_paths = "/" in session_memory or any(word in session_memory for word in ["folder", "file", "directory"])
        if has_numbers or has_paths:
            return 4, f"Specific with concrete details ({length} chars)"
        return 3, f"Relevant but lacks concrete details ({length} chars)"

    # Excellent: Rich context, multiple actions, high info density (100+ chars)
    has_multiple_actions = session_memory.count(",") >= 2 or session_memory.count(";") >= 1
    if has_multiple_actions:
        return 5, f"Rich context with multiple actions ({length} chars)"
    return 4, f"Detailed context ({length} chars)"


def score_tool_context(context):
    """Score toolContext quality (1-5)."""
    tool_context = context.get("toolContext", "")

    # Poor: Empty or generic
    if not tool_context or tool_context in ["Using tool", "Performing action", ""]:
        return 1, "Empty or completely generic"

    # Below Average: Slightly more specific than tool name, but no WHY
    generic_phrases = ["searching", "listing", "creating", "deleting", "updating", "finding"]
    if any(phrase in tool_context.lower() and len(tool_context) < 30 for phrase in generic_phrases):
        return 2, "Describes WHAT but not WHY"

    # Average: Clear purpose, explains immediate goal
    if len(tool_context) < 50:
        if any(word in tool_context.lower() for word in ["finding", "locating", "getting", "checking"]):
            return 3, "Clear purpose but missing workflow reasoning"

    # Good: Explains reasoning, shows decision-making
    reasoning_words = ["confirming", "verifying", "after", "before", "need to", "to ensure"]
    if any(word in tool_context.lower() for word in reasoning_words):
        return 4, "Explains reasoning and workflow context"

    # Excellent: Rich reasoning, explains alternatives
    if len(tool_context) > 70 or "instead of" in tool_context.lower() or "rather than" in tool_context.lower():
        return 5, "Rich reasoning with strategic thinking"

    return 3, "Clear purpose"


def score_goal_coherence(context):
    """Score goal coherence (1-5)."""
    primary = context.get("primaryGoal", "")
    subgoal = context.get("subgoal", "")

    # Poor: Missing, contradictory, or identical
    if not primary or not subgoal:
        return 1, "Missing one or both goals"
    if primary == subgoal:
        return 1, "Goals are identical"

    # Check similarity
    primary_words = set(primary.lower().split())
    subgoal_words = set(subgoal.lower().split())
    overlap = len(primary_words & subgoal_words) / max(len(primary_words), len(subgoal_words))

    # Below Average: Goals overlap significantly
    if overlap > 0.7:
        return 2, f"Goals overlap significantly ({overlap:.0%} similar)"

    # Average: Clear distinction but generic
    if len(primary) < 30 and len(subgoal) < 30:
        return 3, "Clear distinction but generic"

    # Good: Specific and actionable, clear decomposition
    if overlap < 0.5 and len(primary) > 15 and len(subgoal) > 15:
        return 4, "Specific with clear decomposition"

    # Excellent: Strategic hierarchy, multi-step planning evident
    if overlap < 0.3 and len(primary) > 25:
        return 5, "Strategic hierarchy showing multi-step planning"

    return 3, "Proper hierarchy"


def score_prompt_naturalness(prompt):
    """Score prompt naturalness (1-5)."""

    # Poor: Pure command syntax
    if "--" in prompt or prompt.startswith("execute_"):
        return 1, "Pure command syntax"

    # Check for natural language indicators
    natural_indicators = ["my", "I", "can you", "could you", "please", "need to", "want to"]
    has_natural = any(indicator in prompt.lower() for indicator in natural_indicators)

    has_pronouns = any(word in prompt.lower() for word in ["my", "I", "we", "our"])
    has_contractions = "'" in prompt
    has_questions = "?" in prompt

    # Below Average: Command-style with minimal context
    if not has_natural and len(prompt) < 30:
        return 2, "Command-style with minimal context"

    # Average: Mix of natural and command
    if not has_pronouns:
        return 3, "Mix of natural and command style"

    # Good: Natural phrasing with pronouns
    if has_pronouns and len(prompt) > 20:
        return 4, "Natural phrasing with pronouns"

    # Excellent: Highly conversational with ambiguity or implicit context
    if (has_questions or has_contractions or "think" in prompt.lower() or
        "actually" in prompt.lower() or "maybe" in prompt.lower()):
        return 5, "Highly conversational with natural ambiguity"

    return 3, "Moderately natural"


def score_response_realism(assistant_content):
    """Score response realism (1-5)."""

    # Check if there's a Result section
    if "Result:" not in assistant_content and "\nResult:" not in assistant_content:
        return 1, "Missing Result section entirely"

    # Extract result section
    if "Result:" in assistant_content:
        result_start = assistant_content.index("Result:")
        result_section = assistant_content[result_start:]

        # Check for minimal structure
        if result_section.count("{") < 1:
            return 2, "Minimal result structure"

        # Check for metadata
        has_metadata = any(field in result_section for field in [
            "executionTime", "timestamp", "totalResults", "success", "displayed"
        ])

        if not has_metadata:
            return 2, "Basic structure without metadata"

        # Check for rich metadata
        metadata_count = sum(1 for field in [
            "executionTime", "timestamp", "totalResults", "success", "displayed",
            "searchCapabilities", "confidence", "score"
        ] if field in result_section)

        if metadata_count >= 3:
            return 4, f"Rich metadata with {metadata_count} fields"

        return 3, "Basic metadata present"

    return 1, "No result found"


def score_example(example):
    """Score a single example and return quality_scores dict."""
    conversations = example["conversations"]

    # Extract user prompt and assistant response
    user_content = conversations[0]["content"]
    assistant_content = conversations[1]["content"]

    # Parse the context from assistant's tool call
    # Extract JSON from arguments
    if "arguments:" in assistant_content:
        args_start = assistant_content.index("arguments:") + len("arguments:")
        args_str = assistant_content[args_start:].strip()

        # Find the JSON object
        try:
            # Try to parse the arguments
            args = json.loads(args_str)
            context = args.get("context", {})
        except:
            context = {}
    else:
        context = {}

    # Score each dimension
    sm_score, sm_note = score_session_memory(context)
    tc_score, tc_note = score_tool_context(context)
    gc_score, gc_note = score_goal_coherence(context)
    pn_score, pn_note = score_prompt_naturalness(user_content)
    rr_score, rr_note = score_response_realism(assistant_content)

    # Calculate overall quality
    overall = round((sm_score + tc_score + gc_score + pn_score + rr_score) / 5, 1)

    # Construct detailed notes
    notes = (
        f"sessionMemory: {sm_note} (score={sm_score}). "
        f"toolContext: {tc_note} (score={tc_score}). "
        f"goal_coherence: {gc_note} (score={gc_score}). "
        f"prompt_naturalness: {pn_note} (score={pn_score}). "
        f"response_realism: {rr_note} (score={rr_score}). "
        f"Overall: {overall}/5.0"
    )

    return {
        "notes": notes,
        "sessionMemory_quality": sm_score,
        "toolContext_quality": tc_score,
        "goal_coherence": gc_score,
        "prompt_naturalness": pn_score,
        "response_realism": rr_score,
        "overall_quality": overall
    }


def main():
    input_file = "/home/user/Toolset-Training/Datasets/quality_review/sample_batch_100.jsonl"
    output_file = "/home/user/Toolset-Training/Datasets/quality_review/scored_batch_100.jsonl"

    scored_examples = []

    print(f"Scoring examples from {input_file}...")

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            example = json.loads(line.strip())

            # Score the example
            quality_scores = score_example(example)

            # Create output with original data + scores
            scored_example = {
                "conversations": example["conversations"],
                "label": example["label"],
                "quality_scores": quality_scores
            }

            # Preserve original metadata if present
            if "_line_number" in example:
                scored_example["_line_number"] = example["_line_number"]
            if "_index" in example:
                scored_example["_index"] = example["_index"]

            scored_examples.append(scored_example)

            print(f"Scored example {line_num}: overall_quality = {quality_scores['overall_quality']}")

    # Write scored examples
    with open(output_file, 'w') as f:
        for example in scored_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\nWrote {len(scored_examples)} scored examples to {output_file}")

    # Print summary statistics
    scores = [ex["quality_scores"]["overall_quality"] for ex in scored_examples]
    avg_score = sum(scores) / len(scores)
    print(f"\nSummary Statistics:")
    print(f"  Average overall_quality: {avg_score:.2f}")
    print(f"  Min: {min(scores):.1f}")
    print(f"  Max: {max(scores):.1f}")

    # Distribution
    excellent = sum(1 for s in scores if s >= 4.0)
    good = sum(1 for s in scores if 3.0 <= s < 4.0)
    fair = sum(1 for s in scores if 2.0 <= s < 3.0)
    poor = sum(1 for s in scores if s < 2.0)

    print(f"\nDistribution:")
    print(f"  Excellent (4.0-5.0): {excellent} ({excellent/len(scores)*100:.1f}%)")
    print(f"  Good (3.0-3.9): {good} ({good/len(scores)*100:.1f}%)")
    print(f"  Fair (2.0-2.9): {fair} ({fair/len(scores)*100:.1f}%)")
    print(f"  Poor (1.0-1.9): {poor} ({poor/len(scores)*100:.1f}%)")


if __name__ == "__main__":
    main()
