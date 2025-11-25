#!/usr/bin/env python3
"""Manually score batch 43 examples with detailed analysis."""

import json

def get_scores_for_examples():
    """Return manually scored quality assessments for each example."""

    scores = []

    # Example 0: Rename folder - empty sessionMemory
    scores.append({
        "notes": "sessionMemory is empty '' earning automatic score=1 per rubric. toolContext 'User finalizing project structure' describes user state not why this tool was chosen (score=2). Goals 'Rename to production name' → 'Execute folder rename' show basic hierarchy but very generic (score=3). Prompt 'Rename the project folder to its final production name' is natural and clear with implicit context about 'the project folder' (score=4). No Result shown in response (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 3,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 2.2
    })

    # Example 1: YouTube workspace - excellent sessionMemory
    scores.append({
        "notes": "sessionMemory is excellent with 198 chars including concrete details: '42 videos published, 12.3K subscribers, Revenue: $285 last month, Planning course launch Q2 2026' (score=5). toolContext 'Loading workspace to finish editing video 43 and outline video 44 (TypeScript generics)' shows rich workflow reasoning (score=5). Goals show strategic hierarchy: 'Grow channel to 50K subscribers and launch paid course by Q2 2026' → 'Complete video 43 editing and script video 44 by Friday' with multi-step planning (score=5). Prompt 'I want to continue with my tech tutorial YouTube channel workspace' is highly natural and conversational (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 5,
        "toolContext_quality": 5,
        "goal_coherence": 5,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 4.2
    })

    # Example 2: Append after archive - continuation scenario
    scores.append({
        "notes": "User provides Result showing successful archive operation. sessionMemory 'Calibration note archived' is very short (26 chars) and generic (score=2). toolContext 'Log archive' is minimal, just 11 chars, doesn't explain why appending (score=2). Goals 'Update Logs/Calibrations.md' → 'Record archive' show clear hierarchy with specific file path (score=4). This is a continuation (no user prompt), treating as workflow step (score=3 for naturalness). Result shows basic success structure with paths but no metadata like timestamps or executionTime (score=3).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 3,
        "overall_quality": 2.8
    })

    # Example 3: List models for research agent
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'Checking available models before creating research agent' explains workflow reasoning well (score=4). Goals 'Create research assistant agent' → 'List available Claude models' show clear decomposition of multi-step task (score=4). Prompt 'Can you set up a research assistant agent for me? I want it to help with academic papers using Claude's latest model' is highly natural and conversational with 'Can you' and specific intent (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 3.0
    })

    # Example 4: Marketing workspace - excellent result
    scores.append({
        "notes": "User provides comprehensive Result with 5 active campaigns, ROI data, workflows, directory structure (score=5 - excellent metadata and realistic details). sessionMemory 'Loaded marketing workspace, 5 active campaigns, Content Marketing best at 4.5x ROI' references specific prior tool call with concrete numbers (score=4). toolContext 'Getting detailed metrics for decision' connects to workflow (score=4). Goals 'Plan monthly marketing campaign focus' → 'Review analytics report' show workflow decomposition (score=4). Assistant provides natural analysis before tool call 'Looking at your 5 active campaigns, here's the performance breakdown' (score=4 - natural but lengthy preamble). Overall strong example.",
        "sessionMemory_quality": 4,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 5,
        "overall_quality": 4.2
    })

    # Example 5: Delete test agent
    scores.append({
        "notes": "sessionMemory 'Testing phase complete' is very short (23 chars) and generic (score=2). toolContext 'User confirmed testing is done and agent is no longer needed' explains reasoning well (score=4). Goals 'Clean up test agents' → 'Delete specific test agent by ID' show clear hierarchy (score=4). Prompt 'Delete the 'test-image-agent' since we're done testing. Its ID is agent_1234567890' is natural with explicit reasoning 'since we're done testing' (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 3.2
    })

    # Example 6: Session rename logging - continuation
    scores.append({
        "notes": "User provides Result of session rename. sessionMemory 'Session renamed above' is minimal (20 chars), just references prior action (score=2). toolContext 'Append log' is only 10 chars, extremely terse (score=1). Goals 'Update Halo/Deck/Session Log.md' → 'Record rename' are clear and specific with file path (score=4). Continuation scenario (score=3). Result shows success with session details but minimal metadata (score=3).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 1,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 3,
        "overall_quality": 2.6
    })

    # Example 7: Delete test folder
    scores.append({
        "notes": "sessionMemory 'User created test folder for experimentation and no longer needs it' is specific and explains context (66 chars) (score=4). toolContext 'Deleting test folder with test files' describes what not why (score=2). Goals 'Delete Test folder and contents' → 'Clean up experimental files' show good hierarchy (score=4). Prompt 'Delete the test folder I created earlier, it has a few test files in it' is natural with 'I created earlier' showing continuity (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 3.2
    })

    # Example 8: Rename folder with hyphen
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'Rename LightLeaks folder' just restates the action (score=2). Goals 'Rename Photos/Film/LightLeaks' → 'Apply naming conventions' show purpose but could be stronger (score=3). Prompt 'Rename Photos/Film/LightLeaks to Photos/Film/Light-Leaks' is clear and direct, moderate naturalness (score=3). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 3,
        "prompt_naturalness": 3,
        "response_realism": 1,
        "overall_quality": 2.0
    })

    # Example 9: Generate privacy image
    scores.append({
        "notes": "sessionMemory 'User preparing data protection training presentation' provides relevant context (51 chars) (score=3). toolContext 'Generating data privacy visual' is descriptive but doesn't explain why this tool (score=3). Goals 'Create privacy concept image' → 'Generate educational illustration' show clear hierarchy (score=4). Prompt 'I need an image showing data privacy concepts for a presentation' is natural with personal need 'I need' (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 3.2
    })

    # Example 10: Find and replace APIKey
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'User wants to rename APIKey to ApiKey' explains intent but not why find/replace specifically (score=3). Goals 'Replace APIKey with ApiKey' → 'Execute find and replace' are generic (score=3). Prompt 'Replace all instances of 'APIKey' with 'ApiKey' in my API documentation' is clear and direct, using 'my' for personalization (score=4). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 3,
        "goal_coherence": 3,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 2.4
    })

    # Example 11: Search REST API docs
    scores.append({
        "notes": "sessionMemory 'User is learning REST API design principles' is relevant context (44 chars) (score=3). toolContext 'Finding REST API content with extended snippets' explains the specific requirement (score=4). Goals 'Find REST API documentation' → 'Search with 300-character snippets' show specific decomposition (score=4). Prompt 'Find documentation about REST APIs with longer content snippets of 300 characters' is clear but somewhat formal (score=3). No Result shown (score=1).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 1,
        "overall_quality": 3.0
    })

    # Example 12: Archive brief logging
    scores.append({
        "notes": "User provides Result of archive operation. sessionMemory 'Staging brief moved into Archive' is specific and concrete (37 chars) (score=3). toolContext 'Log archive action' is brief but clear purpose (score=3). Goals 'Update Rigs/Mirelune/Archive Log.md' → 'Capture move note' are specific with file path (score=4). Assistant adds context 'Brief archived—recording the move' before tool call (score=4). Result shows success with old/new paths, basic but complete (score=3).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 3.4
    })

    # Example 13: Bring back Monday work
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'User wants to resume Monday's session' explains intent clearly (score=4). Goals 'Load Monday's work' → 'List sessions' show clear decomposition (score=4). Prompt 'Bring back the work I did on Monday' is highly natural and conversational (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 3.0
    })

    # Example 14: Append completed task
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'User wants to add completed authentication task to sprint note' explains the specific intent well (score=4). Goals 'Track completion' → 'Append task' are somewhat generic but clear (score=3). Prompt 'Add a completed task to my sprint note: 'Implemented user authentication system'' is natural with specific content (score=4). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 4,
        "goal_coherence": 3,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 2.6
    })

    # Example 15: Move third book note
    scores.append({
        "notes": "User provides Result of moving Deep-Work. sessionMemory 'Created Books folder, moved Atomic-Habits and Deep-Work' references specific prior actions with file names (61 chars) (score=4). toolContext 'Relocating third book note' shows sequence in workflow (score=4). Goals 'Consolidate book notes into Books folder' → 'Move Thinking-Fast-and-Slow.md' show strategic intent and specific action (score=5). Continuation (score=3). Result shows success with old/new paths and message, basic structure (score=3).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 4,
        "goal_coherence": 5,
        "prompt_naturalness": 3,
        "response_realism": 3,
        "overall_quality": 3.8
    })

    # Example 16: Create session summary
    scores.append({
        "notes": "User provides Result listing 2 sessions. sessionMemory 'User creating session overview. Found dev work and documentation sessions' is specific with context (75 chars) (score=4). toolContext 'Creating session summary' describes what but not why (score=2). Goals 'Create session summary' → 'Create summary note' are redundant (score=2). Continuation (score=3). Result shows success with session list, basic structure but realistic (score=3).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 3,
        "response_realism": 3,
        "overall_quality": 2.8
    })

    # Example 17: Create auth session after search
    scores.append({
        "notes": "User provides Result of search finding auth guide. sessionMemory 'Found 1 authentication note' is brief but specific (31 chars) (score=3). toolContext 'Creating work session' is generic (score=2). Goals 'Work on authentication' → 'Create session' show basic hierarchy (score=3). Continuation (score=3). Result shows 1 search result with score 0.93 and snippet - good metadata (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 3,
        "prompt_naturalness": 3,
        "response_realism": 4,
        "overall_quality": 3.0
    })

    # Example 18: Execute summarization prompt
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'User wants to summarize research paper' explains intent but not why executePrompt tool (score=3). Goals 'Run summarization on research paper' → 'Execute custom prompt' show basic hierarchy (score=3). Prompt 'Execute my custom summarization prompt on this research paper' is natural with 'my custom' showing ownership (score=4). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 3,
        "goal_coherence": 3,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 2.4
    })

    # Example 19: List workspace sessions
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'User reviewing sessions' is brief and generic (score=2). Goals 'List all sessions' → 'Query workspace sessions' are somewhat redundant (score=2). Prompt 'Show me all sessions from this workspace' is natural with 'Show me' and 'this workspace' showing context (score=4). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 2.0
    })

    # Example 20: Read daily note
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'User wants to review daily note content' explains intent clearly (score=4). Goals 'Read Daily/2024-01-15.md' → 'Retrieve and display file content' show hierarchy but subgoal is very operational (score=3). Prompt 'Can you read the content of my daily note at Daily/2024-01-15.md?' is natural with 'Can you' and 'my' (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 4,
        "goal_coherence": 3,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 2.8
    })

    # Example 21: Halo command logging
    scores.append({
        "notes": "User provides Result of halo:tidal-status command. sessionMemory 'halo:tidal-status output captured' is specific but brief (35 chars) (score=3). toolContext 'Append Halo link note' is brief but clear (score=3). Goals 'Update Ops/Links/Halo.md' → 'Record tidal status' are specific (score=4). Assistant adds context 'Command output ready—logging it for GlowPulse' (score=4). Result shows command output with exitCode 0 - good structure (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.6
    })

    # Example 22: Execute PersonaScribe after get_tools
    scores.append({
        "notes": "User provides Result of get_tools listing agentManager tools. sessionMemory 'Enumerated agentManager tools via get_tools' is specific and references prior tool call (47 chars) (score=4). toolContext 'Use PersonaScribe agent to summarize interviews' is very specific about tool choice and purpose (score=5). Goals 'Generate persona brief' → 'Run agent with interview files' show workflow (score=4). Assistant adds 'Persona tooling confirmed—running PersonaScribe now' (score=4). Result shows 3 tools with descriptions - good structure (score=4).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 5,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 4.2
    })

    # Example 23: Update session after indexing
    scores.append({
        "notes": "User provides Result of archive index creation. sessionMemory 'User indexing archive files. Found archives from 2024 and 2023, created index' is specific with details (82 chars) (score=4). toolContext 'Updating session after indexing' explains workflow sequence (score=4). Goals 'Create archive index' → 'Update session metadata' show clear workflow (score=4). Continuation (score=3). Result shows success with filePath, minimal structure (score=2).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 2,
        "overall_quality": 3.4
    })

    # Example 24: Prepend warning to checklist
    scores.append({
        "notes": "sessionMemory is empty '' (score=1). toolContext 'Adding production warning to checklist' explains the specific intent well (score=4). Goals 'Add warning banner' → 'Prepend warning text' show clear hierarchy (score=4). Prompt 'Add a warning at the top of my Deployment Checklist that says this is the production version' is natural with 'my' and specific requirements (score=5). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 1,
        "overall_quality": 3.0
    })

    # Example 25: List states after finding Python tutorial
    scores.append({
        "notes": "User provides Result of search finding Python tutorial. sessionMemory 'Found 1 Python tutorial' is brief (24 chars) but specific (score=3). toolContext 'Listing saved states' describes action but doesn't explain why after finding tutorial (score=2). Goals 'Review Python resources' → 'View checkpoints' show some connection but checkpoints seem unrelated to tutorial search (score=2). Continuation (score=3). Result shows 1 search result with score 0.92 and snippet - good structure (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 3,
        "response_realism": 4,
        "overall_quality": 2.8
    })

    # Example 26: Find React sessions - malformed context
    scores.append({
        "notes": "sessionMemory is empty array [] earning score=1 per rubric. toolContext is an object with currentPath/openFiles/recentCommands instead of string - malformed (score=1). Goals 'Find React sessions' → 'List all sessions' are okay but second is too generic (score=3). Prompt 'Show me sessions where I was working with React components' is natural (score=4). No Result shown (score=1).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 1,
        "goal_coherence": 3,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 2.0
    })

    # Example 27: Top 3 workspaces
    scores.append({
        "notes": "sessionMemory 'User wants only 3 results' is brief (28 chars) but captures the constraint (score=3). toolContext 'Sort alphabetically with limit 3' explains the specific parameters (score=4). Goals 'List workspaces' → 'Top 3 alphabetically' show clear hierarchy (score=4). Prompt 'Show me just the top 3 workspaces by name' is natural with 'Show me' and 'just' emphasizing constraint (score=4). No Result shown (score=1).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 3.2
    })

    # Example 28: Create investment workspace - excellent
    scores.append({
        "notes": "sessionMemory 'Managing diversified investment portfolio' is relevant but could have more detail (44 chars) (score=3). toolContext 'Organizing financial tracking and analysis' explains the purpose well (score=4). Goals 'Create investment tracking workspace' → 'Set up portfolio management' show clear hierarchy (score=4). Prompt 'Set up a workspace for tracking my investments and portfolio' is natural with 'my' (score=4). Assistant adds helpful preamble 'I'll create an investment tracking workspace for you' before tool call (score=4). No Result shown but tool call is comprehensive with workflows and preferences (score=3 for the call structure).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 1,
        "overall_quality": 3.2
    })

    # Example 29: Share Redwood baseline continuation
    scores.append({
        "notes": "User provides Result showing file read success. sessionMemory 'Snapshot text captured above' is minimal (29 chars) and generic (score=2). toolContext 'Append daily note' is brief but clear (score=3). Goals 'Update Fieldnotes/Trillark/Dailies/2025-11-07.md' → 'Record baseline' are specific with full path (score=4). Continuation (score=3). Result shows success with filePath and contentPreview - good structure (score=4).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 4,
        "overall_quality": 3.2
    })

    return scores


def main():
    # Read input examples
    with open('Datasets/quality_review/sample_batch_43.jsonl', 'r') as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]

    # Get manual scores
    scores = get_scores_for_examples()

    if len(scores) != len(examples):
        print(f"ERROR: {len(scores)} scores but {len(examples)} examples!")
        return

    # Add quality_scores to each example
    scored_examples = []
    for example, score in zip(examples, scores):
        example['quality_scores'] = score
        scored_examples.append(example)

    # Write output
    with open('Datasets/quality_review/scored_batch_43.jsonl', 'w') as f:
        for example in scored_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Scored {len(scored_examples)} examples")

    # Summary statistics
    overall_scores = [ex['quality_scores']['overall_quality'] for ex in scored_examples]
    print(f"\nSummary Statistics:")
    print(f"  Mean: {sum(overall_scores) / len(overall_scores):.2f}")
    print(f"  Min: {min(overall_scores):.1f}")
    print(f"  Max: {max(overall_scores):.1f}")
    print(f"  Range: {min(overall_scores):.1f} - {max(overall_scores):.1f}")

    # Distribution
    print(f"\nQuality Distribution:")
    for threshold in [4.0, 3.0, 2.0]:
        count = sum(1 for s in overall_scores if s >= threshold)
        pct = 100 * count / len(overall_scores)
        print(f"  >= {threshold}: {count} ({pct:.1f}%)")

if __name__ == '__main__':
    main()
