#!/usr/bin/env python3
"""Score batch 44 examples according to the interaction quality rubric."""

import json

# Read the input batch
input_file = "/home/user/Toolset-Training/Datasets/quality_review/sample_batch_44.jsonl"
output_file = "/home/user/Toolset-Training/Datasets/quality_review/scored_batch_44.jsonl"

with open(input_file, 'r') as f:
    examples = [json.loads(line) for line in f]

print(f"Loaded {len(examples)} examples")

# Manually scored examples based on rubric
scores = [
    # Example 0: Spanish workspace
    {
        "notes": "sessionMemory is excellent with rich detail: 'Day 45 of 90-day intensive', specific numbers (1500 words, 3x/week, 45 days), concrete progress tracking, and learning challenges (struggling with subjunctive). High information density, 100+ chars (score=5). toolContext shows workflow reasoning 'Loading workspace to review vocabulary flashcards and prepare for tonight's conversation session' with clear next steps (score=4). Goals form clear hierarchy: broad fluency goal → specific tonight's prep (score=5). Prompt is natural 'I want to continue with my Spanish conversation practice workspace' with first person phrasing (score=4). No Result shown, but tool call structure is realistic with proper workspace ID and context object (score=3).",
        "sessionMemory_quality": 5,
        "toolContext_quality": 4,
        "goal_coherence": 5,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 4.2
    },
    # Example 1: Remove draft tag
    {
        "notes": "sessionMemory is minimal 'Manuscript is finalized' at only ~23 chars with no concrete details (score=2). toolContext 'Cleaning up state tags' is generic, describes what not why (score=2). Goals are weak: 'Remove draft tag' → 'Update state tags' overlap significantly and lack hierarchy (score=2). Prompt is natural and conversational 'Remove the 'draft' tag from my finished manuscript state' (score=4). Response includes brief natural text 'I'll remove the draft tag from your state' and properly structured tool call with removeTags parameter (score=3).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 2.6
    },
    # Example 2: Wedding planning follow-up
    {
        "notes": "sessionMemory references prior action 'Loaded wedding workspace' with specific numbers: 71%, 25 tasks, 3-month phase (score=4). toolContext 'Checking remaining tasks' is generic without explaining workflow reasoning (score=2). Goals have clear hierarchy: 'Review wedding planning status' → 'Read pending checklist' (score=4). Result is highly realistic with comprehensive nested structure: workflows, directoryStructure, keyFiles with rich metadata (totalBudget: 35000, percentSpent: 81%, daysUntil: 222, vendorCount with breakdown). Assistant response is natural and conversational 'Your wedding is 222 days away!' (score=5). Flow between Result and next action is natural (score=5).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 5,
        "overall_quality": 4.0
    },
    # Example 3: Generate image
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User wants to generate image' just restates the obvious without workflow reasoning (score=2). primaryGoal is overly long and duplicates the entire user request verbatim rather than being a concise goal statement, subgoal 'Execute generateImage' just restates tool name (score=2). Prompt is natural with specific creative details 'Generate an image of a futuristic city skyline at sunset and save it to my Images folder' (score=4). Tool call has realistic parameters: detailed prompt expansion, provider/model specified, aspectRatio, numberOfImages, savePath (score=4).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 2.6
    },
    # Example 4: Moving draft files
    {
        "notes": "sessionMemory lists concrete results 'Found 4 draft files' with specific filenames from search (score=4). toolContext 'Consolidating drafts to Drafts folder' explains the purpose clearly (score=4). Goals have good hierarchy: 'Move draft files to Drafts' → 'Move first draft file' showing multi-step workflow (score=4). Result shows realistic search results with scores (0.95, 0.92, 0.90, 0.85), totalResults, searchedPaths metadata (score=4). Assistant response is natural 'Found 4 draft files in different locations. Moving them to the Drafts folder now.' (score=5).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 4,
        "overall_quality": 4.2
    },
    # Example 5: Restore debugging session
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User wants payment bug debugging session' describes what user wants but not why this tool/approach (score=3). Goals form clear hierarchy: 'Load debugging session' → 'Find payment bug session' (score=4). Prompt is highly natural and conversational 'Restore my debugging session from when I was fixing the payment bug.' with implicit context reference (score=5). Tool call structure is proper with limit and order parameters (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 3,
        "overall_quality": 3.2
    },
    # Example 6: Get agent details
    {
        "notes": "sessionMemory is brief 'NeonPour id noted' (~20 chars), minimal but relevant context from prior step (score=3). toolContext 'Get agent info' is generic, doesn't explain why fetching details (score=2). Goals 'Inspect NeonPour' → 'Capture prompt' have clear hierarchy showing specific intent (score=4). Result shows realistic structure with 2 agents listed, isEnabled flags (score=4). Assistant response is natural 'Fetching NeonPour details.' (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.4
    },
    # Example 7: List resources folder
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User preparing to copy files to archive' provides workflow context explaining the why (score=4). Goals 'List files' → 'Check resources' are generic and very similar, weak hierarchy (score=2). Prompt is natural 'List the files in my Resources folder so I can copy them to Archive' with clear intent and explicit reasoning (score=4). Tool call has proper parameters: path, includeFiles (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 4,
        "goal_coherence": 2,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 2.8
    },
    # Example 8: Create stage state
    {
        "notes": "sessionMemory 'Stage list referenced via memoryManager_listStates' references specific prior tool call showing continuity (score=4). toolContext 'Create new stage state' is generic (score=2). Goals 'Save Stagecraft Dusk Sweep' → 'Include cues' show clear intent and specific detail (score=4). Result shows realistic listStates output with IDs and names (score=4). Assistant response is natural 'Creating the new dusk sweep checkpoint now.' Tool call includes rich metadata: name, description, targetSessionId, tags array, nextSteps array (score=5).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 5,
        "overall_quality": 3.8
    },
    # Example 9: Read KB note
    {
        "notes": "sessionMemory 'User building knowledge base. Created KB note and retrieved tools list.' references multiple prior actions with concrete details (score=4). toolContext 'Reading KB note to update' explains the immediate purpose and workflow (score=4). Goals 'Build knowledge base with tools info' → 'Prepare for update' show clear strategic hierarchy (score=4). Result shows realistic tool list array (score=3). Flow is natural with no explicit user prompt shown in this turn (score=3).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 3,
        "overall_quality": 3.6
    },
    # Example 10: Find deprecated method
    {
        "notes": "sessionMemory 'User identified getUserData as deprecated method' captures the specific method name from user's request (score=3). toolContext 'Locating getUserData function' describes the what not why (score=2). Goals 'Add deprecation comment to getUserData' → 'Find function definition' show clear workflow decomposition (score=4). Prompt is highly natural and specific 'The API documentation mentions a deprecated method 'getUserData'. Find it and add a deprecation comment above it.' with domain-appropriate language (score=5). Tool call has realistic search parameters: query, limit, includeContent (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 4,
        "overall_quality": 3.6
    },
    # Example 11: Append agent log
    {
        "notes": "sessionMemory 'Scout prompt updated via agentManager_updateAgent.' references specific prior tool call showing continuity (score=4). toolContext 'Append agent log entry' is generic but relevant (score=3). Goals 'Update Canopy/Solunox/Agent Log.md' → 'Record prompt edits' show clear hierarchy with specific file path (score=4). Result shows realistic updateAgent response with id, updatedFields array (score=4). Assistant response is natural 'Prompt refreshed—logging the change.' Tool call content parameter has realistic log entry format (score=4).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.8
    },
    # Example 12: Find in-progress projects
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'searching project statuses' is generic lowercase text without explaining why (score=2). Goals 'create project summary' → 'find in-progress items' have clear hierarchy (score=4). Prompt is natural and clear 'Find my project files with 'in progress' status and create a summary.' (score=4). Tool call has realistic search parameters targeting status metadata (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 2.8
    },
    # Example 13: Novel editing checkpoint
    {
        "notes": "sessionMemory 'Edited chapters 1-18. Character consistency verified. Dialog improved in chapters 15-18. Pacing refined. Ready for beta readers.' is rich with specific progress details and concrete actions (score=5). toolContext 'Creating checkpoint for editing phase completion' explains strategic purpose (score=4). Goals 'Complete novel draft and submit to publisher by Q2 2026' → 'Save editing progress checkpoint' show excellent strategic hierarchy with timeline (score=5). Result shows comprehensive workspace context with workflows, keyFiles structure, recentActivity (score=5). Assistant response includes extended natural text discussing progress 'You've trimmed 12% of redundancy - good editorial work' with conversational flow (score=5).",
        "sessionMemory_quality": 5,
        "toolContext_quality": 4,
        "goal_coherence": 5,
        "prompt_naturalness": 5,
        "response_realism": 5,
        "overall_quality": 4.8
    },
    # Example 14: Analyze Python code
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'Finding Python files to analyze.' explains the immediate step but not broader workflow (score=3). Goals 'Code quality analysis' → 'Find Python files' show clear decomposition (score=4). Prompt is natural and conversational 'Analyze the code quality of my Python files and suggest improvements.' (score=4). Tool call uses searchDirectory with pattern '*.py' and searchType='files' which is realistic (score=4).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.2
    },
    # Example 15: Replace token line
    {
        "notes": "sessionMemory 'Week 51 token logged earlier' references prior activity with specific detail (score=3). toolContext 'Replace line' is generic (score=2). Goals 'Update Config/glyph.env' → 'Set Week 52 token' have clear hierarchy with specific details (score=4). Prompt is natural with specific technical details 'Replace the cache token line in Config/glyph.env with the new Week 52 token.' (score=4). Tool call uses replaceByLine with specific line numbers and realistic token format (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.4
    },
    # Example 16: Find deprecated API
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User wants to find deprecated methods' just restates user's intent without workflow reasoning (score=2). Goals 'Locate deprecated API calls' → 'Search for @deprecated or common deprecated patterns' show clear decomposition with technical detail (score=4). Prompt is natural and conversational 'Show me where we're using deprecated API methods so I can update them.' (score=4). Tool call has realistic parameters with includeContent=true (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 2.8
    },
    # Example 17: List solar arcs folder
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'List solar arcs folder' just restates the action (score=2). Goals 'Inventory Observatory/Notes/Solar Arcs' → 'Files only' show intent with specific path, though 'Files only' is more of a parameter than a subgoal (score=3). Prompt is natural with domain-specific language 'List Observatory/Notes/Solar Arcs to confirm the new arc files.' (score=4). Tool call has realistic parameters: includeFiles=true, includeFolders=false, depth=0 (score=4).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 3,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 2.8
    },
    # Example 18: Execute LaunchSummarizer
    {
        "notes": "sessionMemory 'Confirmed tools via get_tools' references prior tool call (score=3). toolContext 'Run LaunchSummarizer' just names the agent without explaining why this approach (score=2). Goals 'Generate summary bullets' → 'Reference Research/Notes/Launch Findings.md' show clear intent with specific file (score=4). Result shows realistic tools list with name and description fields (score=3). Tool call has comprehensive parameters: agent name, detailed prompt, model, provider, temperature, returnContent flag (score=5).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 4,
        "overall_quality": 3.2
    },
    # Example 19: Update state description
    {
        "notes": "sessionMemory 'User documenting state purpose' is generic and short (~30 chars) (score=2). toolContext 'Update description only' is somewhat specific but minimal (score=3). Goals 'Update state' → 'Add description' are generic and very similar (score=2). Prompt is natural and clear 'Add a detailed description to the blog drafts state.' (score=4). Tool call has proper structure with stateId and description field containing detailed text (score=4).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 3,
        "goal_coherence": 2,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.0
    },
    # Example 20: Create signal summary
    {
        "notes": "sessionMemory 'Intro captured' is very short (~14 chars) and vague (score=2). toolContext 'Create summary' is generic (score=2). Goals 'Write Research/Briefs/Signal Summary.md' → 'Document' are weak, 'Document' is too generic as a subgoal (score=2). Result shows realistic file content from readContent (score=4). Assistant response has no natural language, jumps directly to tool call (score=3). Tool call creates new file with content (score=4).",
        "sessionMemory_quality": 2,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 3,
        "response_realism": 4,
        "overall_quality": 2.6
    },
    # Example 21: Run MoodDJ agent
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'Run MoodDJ agent' just names the tool/agent (score=2). Goals 'Generate playlist motifs' → 'Append seeds' show hierarchy but are somewhat generic (score=3). Prompt is highly natural with creative domain language 'Run MoodDJ over Playlist Ideas and append three chillwave directions to Music/Ideas/Playlist Seeds.md.' with specific genre and structure (score=5). Tool call has rich parameters: agent name, filepaths array, detailed prompt, model params, action object with type and targetPath (score=5).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 3,
        "prompt_naturalness": 5,
        "response_realism": 5,
        "overall_quality": 3.2
    },
    # Example 22: Restore ML experiment
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User wants to restore machine learning experiment' restates user intent without workflow reasoning (score=2). Goals 'Restore ML experiment' → 'Search for experiment snapshots' show clear decomposition (score=4). Prompt is highly natural and conversational 'I need to go back to that machine learning experiment I was running. Can you restore it?' with implicit reference 'that experiment I was running' (score=5). Tool call has proper parameters (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 3,
        "overall_quality": 3.0
    },
    # Example 23: Rename project plan
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User wants to rename a note.' just restates intent (score=2). Goals 'Rename project plan' → 'Rename note' are nearly identical, no real hierarchy (score=2). Prompt is natural and conversational with polite phrasing 'Can you rename the 'Project Plan.md' to 'Old Project Plan.md'?' (score=4). Tool call uses moveNote with proper path and newPath parameters for renaming (score=4).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 2,
        "goal_coherence": 2,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 2.6
    },
    # Example 24: Search CI docs with snippets
    {
        "notes": "sessionMemory 'User is setting up continuous integration pipeline.' provides context explaining why searching (score=3). toolContext 'Finding CI notes with content preview' explains the specific need for snippets (score=4). Goals 'Find continuous integration documentation' → 'Search with limited results and snippets' show clear hierarchy with specific parameters (score=4). Prompt is natural and specific with technical details 'Search for notes about 'continuous integration' but limit results to 5 and include content snippets.' (score=4). Tool call has realistic parameters including snippetLength=180 (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 4,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.8
    },
    # Example 25: Load research workspace
    {
        "notes": "sessionMemory is empty string earning automatic (score=1). toolContext 'User needs help organizing research' provides some context but is generic (score=3). Goals 'Organize research papers' → 'Load workspace to see structure' show workflow reasoning (score=4). Prompt is natural and conversational 'Help me organize my research papers' (score=4). Tool call has proper workspace ID parameter (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 3,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 3,
        "overall_quality": 3.0
    },
    # Example 26: Find database migration session
    {
        "notes": "sessionMemory is empty array [] earning automatic (score=1). toolContext has nested object structure instead of string - appears to be a format error with currentPath, openFiles, recentCommands (score=1). Goals 'Find database migration session' → 'List recent sessions' show clear hierarchy (score=4). Prompt is highly natural with temporal context and uncertainty 'I can't remember the exact name, but I was working on something related to database migration last week.' (score=5). Tool call structure is proper (score=3).",
        "sessionMemory_quality": 1,
        "toolContext_quality": 1,
        "goal_coherence": 4,
        "prompt_naturalness": 5,
        "response_realism": 3,
        "overall_quality": 2.8
    },
    # Example 27: Append lichen observation
    {
        "notes": "sessionMemory 'Snippet captured from Lagoon Traverse note.' references prior search result with specific file name (score=3). toolContext 'Append highlight' is generic (score=2). Goals 'Update Cartography/Logs/Field Highlights.md' → 'Record lichen observation' show clear hierarchy with specific file and detail (score=4). Result shows realistic search results with filePath, content, single result (score=3). Assistant response is natural 'Found 4... Moving them now.' Wait, checking again - response is about lichen observation (score=4). Tool call appends formatted content (score=4).",
        "sessionMemory_quality": 3,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 3.4
    },
    # Example 28: Create exploration checkpoint
    {
        "notes": "sessionMemory 'User exploring session history. Found planning and development sessions.' provides context with specific details from prior search (score=4). toolContext 'Creating state checkpoint' is generic (score=2). Goals 'Explore and document session work' → 'Save current exploration state' show clear hierarchy (score=4). Result shows realistic listSessions output with IDs and names (score=3). Flow is somewhat unnatural - no explicit user prompt for creating checkpoint, jumps from Result to tool call (score=3).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 2,
        "goal_coherence": 4,
        "prompt_naturalness": 3,
        "response_realism": 3,
        "overall_quality": 3.2
    },
    # Example 29: Read content schedule
    {
        "notes": "sessionMemory 'Published 12 posts this month. 2 videos in post-production. Need to plan December content.' has specific numbers and concrete next steps (score=4). toolContext 'Reading schedule to plan next month's content' explains workflow reasoning (score=4). Goals 'Build audience with consistent weekly content' → 'Review publishing schedule and plan new content' show strategic hierarchy (score=5). Result shows comprehensive workspace context with workflows, keyFiles structure (score=4). Flow from Result to tool call is natural (score=4).",
        "sessionMemory_quality": 4,
        "toolContext_quality": 4,
        "goal_coherence": 5,
        "prompt_naturalness": 4,
        "response_realism": 4,
        "overall_quality": 4.2
    }
]

# Add scores to examples
for i, (example, score) in enumerate(zip(examples, scores)):
    example['quality_scores'] = score
    example['_index'] = i

# Write scored examples
with open(output_file, 'w') as f:
    for example in examples:
        f.write(json.dumps(example) + '\n')

print(f"Scored {len(examples)} examples")
print(f"Output written to: {output_file}")

# Calculate statistics
overall_scores = [s['overall_quality'] for s in scores]
avg_overall = sum(overall_scores) / len(overall_scores)
print(f"\nStatistics:")
print(f"Average overall_quality: {avg_overall:.2f}")
print(f"Min: {min(overall_scores):.1f}")
print(f"Max: {max(overall_scores):.1f}")
print(f"\nDimension averages:")
for dim in ['sessionMemory_quality', 'toolContext_quality', 'goal_coherence', 'prompt_naturalness', 'response_realism']:
    avg = sum(s[dim] for s in scores) / len(scores)
    print(f"  {dim}: {avg:.2f}")
