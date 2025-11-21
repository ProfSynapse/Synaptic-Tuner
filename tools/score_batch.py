#!/usr/bin/env python3
"""Score quality review batch according to rubric."""

import json
import re
from pathlib import Path

def extract_context_fields(conversations):
    """Extract context object fields from assistant response."""
    for conv in conversations:
        if conv["role"] == "assistant":
            content = conv["content"]
            # Find arguments: followed by JSON object
            # Match until we find a closing brace at the same level as opening
            match = re.search(r'arguments:\s*({)', content, re.DOTALL)
            if match:
                start = match.start(1)
                # Count braces to find matching closing brace
                depth = 0
                end = start
                for i in range(start, len(content)):
                    if content[i] == '{':
                        depth += 1
                    elif content[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                try:
                    args_str = content[start:end]
                    args = json.loads(args_str)
                    if "context" in args:
                        return args["context"]
                except Exception as e:
                    pass
    return {}

def score_example(example, index):
    """Score a single example according to the rubric."""
    conversations = example["conversations"]
    context = extract_context_fields(conversations)

    # Extract fields
    session_memory = context.get("sessionMemory", "")
    tool_context = context.get("toolContext", "")
    primary_goal = context.get("primaryGoal", "")
    subgoal = context.get("subgoal", "")

    # Get user prompt - handle multi-turn where first message is a Result
    user_prompt = conversations[0]["content"] if conversations else ""
    is_result_prompt = user_prompt.startswith("Result:")

    # For multi-turn, the result is in the user message
    # For single-turn, we expect no result shown (goes straight to response)
    assistant_response = conversations[-1]["content"] if len(conversations) > 1 else ""
    has_result_in_user = is_result_prompt
    has_result_in_assistant = '"success":' in assistant_response or '"results":' in assistant_response

    notes_parts = []

    # 1. sessionMemory_quality
    sm_len = len(session_memory)
    if not session_memory or session_memory in ["", "[]", "None", "N/A"]:
        sm_score = 1
        notes_parts.append(f"sessionMemory is empty/placeholder ('{session_memory}') - automatic score 1")
    elif sm_len < 20:
        sm_score = 2
        notes_parts.append(f"sessionMemory very short ({sm_len} chars): '{session_memory}' - lacks detail, score 2")
    elif sm_len < 50:
        sm_score = 3
        notes_parts.append(f"sessionMemory generic but relevant ({sm_len} chars): '{session_memory}' - score 3")
    elif sm_len < 100:
        sm_score = 4
        notes_parts.append(f"sessionMemory specific with details ({sm_len} chars): '{session_memory}' - score 4")
    else:
        sm_score = 5
        notes_parts.append(f"sessionMemory rich context ({sm_len} chars): '{session_memory}' - high info density, score 5")

    # 2. toolContext_quality
    tc_lower = tool_context.lower()
    if not tool_context or tool_context in ["", "Using tool", "Performing action"]:
        tc_score = 1
        notes_parts.append(f"toolContext empty or generic: '{tool_context}' - score 1")
    elif len(tool_context) < 20 or tc_lower in ["searching", "updating", "creating"]:
        tc_score = 2
        notes_parts.append(f"toolContext restates action without why: '{tool_context}' - score 2")
    elif "before" not in tc_lower and "after" not in tc_lower and "reason" not in tc_lower:
        tc_score = 3
        notes_parts.append(f"toolContext explains purpose but missing workflow reasoning: '{tool_context}' - score 3")
    elif "confirm" in tc_lower or "verify" in tc_lower or "need to" in tc_lower:
        tc_score = 4
        notes_parts.append(f"toolContext shows decision-making: '{tool_context}' - score 4")
    else:
        tc_score = 5
        notes_parts.append(f"toolContext rich reasoning: '{tool_context}' - score 5")

    # 3. goal_coherence
    if not primary_goal or not subgoal:
        gc_score = 1
        notes_parts.append(f"Missing goal fields - primaryGoal: '{primary_goal}', subgoal: '{subgoal}' - score 1")
    elif primary_goal == subgoal or primary_goal.lower() == subgoal.lower():
        gc_score = 2
        notes_parts.append(f"Goals identical: '{primary_goal}' / '{subgoal}' - score 2")
    elif len(primary_goal) < 15 or len(subgoal) < 15:
        gc_score = 3
        notes_parts.append(f"Goals clear but generic: '{primary_goal}' → '{subgoal}' - score 3")
    elif primary_goal and subgoal and len(primary_goal) > 20 and len(subgoal) > 20:
        gc_score = 5
        notes_parts.append(f"Goals strategic with clear hierarchy: '{primary_goal}' → '{subgoal}' - score 5")
    else:
        gc_score = 4
        notes_parts.append(f"Goals specific and actionable: '{primary_goal}' → '{subgoal}' - score 4")

    # 4. prompt_naturalness
    # For multi-turn with Result, score is N/A but we'll give neutral score
    if is_result_prompt:
        pn_score = 3
        notes_parts.append(f"Prompt is Result from prior tool call (multi-turn) - neutral score 3")
    else:
        prompt_lower = user_prompt.lower()
        has_pronouns = any(word in prompt_lower for word in ["i", "my", "me", "you", "can you", "we"])
        has_question = "?" in user_prompt
        is_command = user_prompt.startswith(("Update", "Create", "Search", "Find", "Add", "Replace", "Execute"))
        word_count = len(user_prompt.split())

        if word_count < 5 and not has_pronouns:
            pn_score = 2
            notes_parts.append(f"Prompt terse/command-like: '{user_prompt[:60]}...' - score 2")
        elif is_command and not has_pronouns and not has_question:
            pn_score = 3
            notes_parts.append(f"Prompt command-style: '{user_prompt[:60]}...' - score 3")
        elif has_pronouns or has_question or word_count > 10:
            pn_score = 4
            notes_parts.append(f"Prompt natural with pronouns/question: '{user_prompt[:60]}...' - score 4")
        else:
            pn_score = 3
            notes_parts.append(f"Prompt mix of natural/formal: '{user_prompt[:60]}...' - score 3")

        # Adjust for highly natural prompts
        if ("can you" in prompt_lower or "i'm not sure" in prompt_lower or
            "i think" in prompt_lower or "actually" in prompt_lower or "but " in prompt_lower):
            pn_score = 5
            notes_parts[-1] = f"Prompt highly conversational: '{user_prompt[:60]}...' - score 5"

    # 5. response_realism
    # Check if there's a Result (either in user message for multi-turn, or nowhere for single-turn)
    # Single-turn examples typically don't show the Result
    if has_result_in_user:
        # Multi-turn: Result is shown in user message
        result_text = user_prompt
        if result_text.count('"') < 10:
            rr_score = 2
            notes_parts.append("Result minimal - only success flag, score 2")
        elif "timestamp" in result_text or "relevanceScore" in result_text or "executionTime" in result_text:
            rr_score = 4
            notes_parts.append("Result includes rich metadata (timestamp/relevanceScore/executionTime) - score 4")
        else:
            rr_score = 3
            notes_parts.append("Result proper structure but limited metadata - score 3")
    else:
        # Single-turn: No result shown (normal for training data)
        # Score based on whether this is reasonable
        rr_score = 3
        notes_parts.append("Single-turn example (no Result shown) - typical for training data, score 3")

    # Calculate overall
    overall = round((sm_score + tc_score + gc_score + pn_score + rr_score) / 5, 1)

    notes = ". ".join(notes_parts) + "."

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
    input_file = Path("/home/user/Toolset-Training/Datasets/quality_review/sample_batch_41.jsonl")
    output_file = Path("/home/user/Toolset-Training/Datasets/quality_review/scored_batch_41.jsonl")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    scored_examples = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        example = json.loads(line)
        scores = score_example(example, i)

        # Add quality_scores to example
        example["quality_scores"] = scores
        scored_examples.append(example)

        print(f"Example {i}: overall={scores['overall_quality']}")

    # Write scored examples
    with open(output_file, 'w') as f:
        for example in scored_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\nScored {len(scored_examples)} examples")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
