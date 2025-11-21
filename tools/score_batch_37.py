#!/usr/bin/env python3
"""Score examples in batch 37 according to the quality rubric."""

import json
from typing import Dict, Any

def score_example(example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Score a single example according to the rubric."""

    conversations = example.get("conversations", [])

    # Extract context from assistant message
    context = {}
    assistant_msg = ""
    user_msg = ""

    for conv in conversations:
        if conv["role"] == "user":
            user_msg = conv["content"]
        elif conv["role"] == "assistant":
            assistant_msg = conv["content"]
            # Try to extract context from tool call
            if "arguments:" in assistant_msg:
                try:
                    args_start = assistant_msg.index("arguments:") + 10
                    args_text = assistant_msg[args_start:].strip()
                    # Find the JSON object
                    if args_text.startswith("{"):
                        # Parse the arguments
                        args = json.loads(args_text)
                        context = args.get("context", {})
                except:
                    pass

    session_memory = context.get("sessionMemory", "")
    tool_context = context.get("toolContext", "")
    primary_goal = context.get("primaryGoal", "")
    subgoal = context.get("subgoal", "")

    # Now score each example based on index
    scores = score_by_index(index, user_msg, assistant_msg, session_memory, tool_context, primary_goal, subgoal)

    # Add scores to example
    example["quality_scores"] = scores
    example["_index"] = index

    return example


def score_by_index(idx, user_msg, assistant_msg, session_memory, tool_context, primary_goal, subgoal):
    """Score based on the specific example index."""

    # Example 0
    if idx == 0:
        return {
            "notes": "sessionMemory: 'Snippet pulled from Compass log.' (35 chars) - References specific prior action but lacks detail about what search was performed or why. Score: 3. toolContext: 'Append highlight' (16 chars) - Very generic, just restates action without explaining why this step is needed. Score: 2. goal_coherence: 'Update Archive/Logs/Highlights.md' → 'Record patina note' - Clear hierarchy with specific file path and concrete action. Score: 4. prompt_naturalness: User message is 'Result:' format showing tool output, not natural user language. This is synthetic continuation showing tool execution flow. Score: 2. response_realism: Natural acknowledgment 'Snippet captured—logging it now' followed by proper tool call structure. No result shown to evaluate metadata. Score: 3.",
            "sessionMemory_quality": 3,
            "toolContext_quality": 2,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 3,
            "overall_quality": 2.8
        }

    # Example 1
    elif idx == 1:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic score of 1 per rubric (red flag). toolContext: 'User wants to resume morning work' (35 chars) - Explains the why reasonably well, showing understanding of intent. Score: 3. goal_coherence: 'Resume this morning's session' → 'List recent sessions' - Clear hierarchy showing user's broader goal and immediate step needed. Score: 4. prompt_naturalness: 'Continue from where I stopped this morning.' - Very natural, conversational, uses implicit context. Score: 5. response_realism: Shows only the tool call, no result to evaluate. Tool structure is proper. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 5,
            "response_realism": 3,
            "overall_quality": 3.2
        }

    # Example 2
    elif idx == 2:
        return {
            "notes": "sessionMemory: 'User organizing quarterly OKRs. Created Q1, Q2, and Q3 folders.' (66 chars) - Concrete details with specific folder names and clear narrative. Score: 4. toolContext: 'Creating Q4 OKR folder' (22 chars) - Clear and specific, explains the current step in sequence. Score: 3. goal_coherence: 'Create quarterly OKR folders' → 'Create OKRs/Q4-2026 folder' - Clear hierarchy, shows this is part of series. Score: 4. prompt_naturalness: User provides Result object (tool output), not natural language. Synthetic format. Score: 2. response_realism: Natural acknowledgment 'Created Q3 folder.' with proper tool call. No result metadata shown. Score: 3.",
            "sessionMemory_quality": 4,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 3,
            "overall_quality": 3.2
        }

    # Example 3
    elif idx == 3:
        return {
            "notes": "sessionMemory: 'Created Projects/API-Documentation folder' (42 chars) - References specific prior action with concrete path. Score: 3. toolContext: 'Building comprehensive API docs' (32 chars) - Explains broader purpose but not why batch operation specifically. Score: 3. goal_coherence: 'Create API Documentation structure' → 'Create all documentation files' - Clear hierarchy, second goal shows execution strategy. Score: 4. prompt_naturalness: User provides Result object. Not natural user language. Score: 2. response_realism: Tool call shows batchContent with 4 realistic operations (Overview, Endpoints, Examples, Error-Codes) with appropriate markdown content including code examples. Rich structure. Score: 5.",
            "sessionMemory_quality": 3,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 3.4
        }

    # Example 4
    elif idx == 4:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1 per rubric. toolContext: 'User organizing workspace' (26 chars) - Generic, doesn't explain why Templates folder specifically or what it's for. Score: 2. goal_coherence: 'Create folder' → 'Workspace setup' - Both very generic and almost backwards hierarchy (setup is broader than create folder). Score: 2. prompt_naturalness: 'Create a Templates folder in my workspace.' - Natural language with clear request. Score: 4. response_realism: Shows tool call only, no result. Structure is proper. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 2,
            "goal_coherence": 2,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 2.4
        }

    # Example 5
    elif idx == 5:
        return {
            "notes": "sessionMemory: 'Read research notes about distributed systems' (47 chars) - References specific prior action with topic. Score: 3. toolContext: 'Generating concise AI summary' (30 chars) - Clear immediate action but doesn't explain why summary needed. Score: 3. goal_coherence: 'Generate and save AI summary' → 'Create summary with AI' - Goals overlap significantly (both about creating summary). Unclear distinction. Score: 2. prompt_naturalness: User provides Result object with realistic research note content. Not natural user language. Score: 2. response_realism: Natural acknowledgment 'Read the notes. Generating AI summary.' Tool call to executePrompt with specific instructions and maxTokens. No result shown. Score: 4.",
            "sessionMemory_quality": 3,
            "toolContext_quality": 3,
            "goal_coherence": 2,
            "prompt_naturalness": 2,
            "response_realism": 4,
            "overall_quality": 2.8
        }

    # Example 6
    elif idx == 6:
        return {
            "notes": "sessionMemory: 'Located agent id via agentManager_listAgents' (46 chars) - References specific prior tool call by name. Good. Score: 4. toolContext: 'Toggle SpectrumEcho on' (23 chars) - Clear and specific. Score: 3. goal_coherence: 'Enable SpectrumEcho' → 'Flip agent flag' - Clear hierarchy, second goal describes mechanism. Score: 4. prompt_naturalness: User provides Result object showing agent list. Not natural. Score: 2. response_realism: Natural reasoning 'SpectrumEcho is still disabled, so I'll toggle it back on.' Tool call structure proper with good result metadata (executionTime: 87). Score: 4.",
            "sessionMemory_quality": 4,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 4,
            "overall_quality": 3.4
        }

    # Example 7
    elif idx == 7:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'User preparing reference' (24 chars) - Generic, doesn't explain why opening without focus. Score: 2. goal_coherence: 'Open without focus' → 'Background open' - Both say same thing, no clear hierarchy. Score: 2. prompt_naturalness: 'Open my Product-Roadmap.md in a new tab without focusing on it.' - Natural, specific, includes technical detail about focus behavior. Score: 4. response_realism: Tool call with proper parameters including focus: false. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 2,
            "goal_coherence": 2,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 2.4
        }

    # Example 8
    elif idx == 8:
        return {
            "notes": "sessionMemory: 'Moved 1 file' (13 chars) - Very short, minimal context about what's being done. Score: 2. toolContext: 'Moving second meeting file' (27 chars) - Shows sequence awareness (second of multiple). Score: 3. goal_coherence: 'Archive 2024 meeting files' → 'Move 2024-Q1-meeting-notes.md' - Clear hierarchy with specific file. Score: 4. prompt_naturalness: User provides Result object. Not natural. Score: 2. response_realism: Natural progress update 'Moved 1 of 3. Moving next file.' Tool call with proper paths. No result shown. Score: 4.",
            "sessionMemory_quality": 2,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 4,
            "overall_quality": 3.0
        }

    # Example 9
    elif idx == 9:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'User wants better agent name' (29 chars) - Explains user motivation reasonably. Score: 3. goal_coherence: 'Rename Meeting Summarizer' → 'Update agent name' - Both say same thing, no hierarchy. Score: 2. prompt_naturalness: 'Rename my Meeting Summarizer agent to 'Meeting Notes AI'.' - Natural, clear, specific names provided. Score: 4. response_realism: Tool call with id and name parameters. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 3,
            "goal_coherence": 2,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 2.6
        }

    # Example 10
    elif idx == 10:
        return {
            "notes": "sessionMemory: 'User tracking daily project progress.' (39 chars) - Generic context, lacks specifics about project or prior entries. Score: 3. toolContext: 'User adding today's progress notes' (35 chars) - Clear purpose but redundant with request. Score: 3. goal_coherence: 'Update status document' → 'Append daily progress' - Clear hierarchy. Score: 4. prompt_naturalness: 'Update my project status note with today's progress.' - Natural, conversational, implies file location. Score: 4. response_realism: Tool call with realistic dated entry including bullets. No result metadata shown. Score: 3.",
            "sessionMemory_quality": 3,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 3.4
        }

    # Example 11
    elif idx == 11:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'Using memoryManager to list all sessions' (41 chars) - Describes what tool does, not why it's needed. Score: 2. goal_coherence: 'View active sessions' → 'List all sessions' - Almost identical, minimal hierarchy. Score: 2. prompt_naturalness: 'Show me all my active sessions.' - Natural, conversational. Score: 4. response_realism: Tool call with limit and order parameters. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 2,
            "goal_coherence": 2,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 2.4
        }

    # Example 12
    elif idx == 12:
        return {
            "notes": "sessionMemory: 'Completed authentication module. UI components 60% done. Backend API endpoints 80% complete.' (94 chars) - Rich context with multiple specific progress metrics. Excellent. Score: 5. toolContext: 'Loading workspace to check sprint status and upcoming tasks' (60 chars) - Explains both what and why with workflow reasoning. Score: 4. goal_coherence: 'Release mobile app v2.0 by Q1 2026' → 'Restore workspace context and review development progress' - Strategic hierarchy showing bigger goal and immediate step. Score: 5. prompt_naturalness: 'I want to check my product development workspace' - Natural, conversational. Score: 4. response_realism: Tool call with id parameter. No result shown. Score: 3.",
            "sessionMemory_quality": 5,
            "toolContext_quality": 4,
            "goal_coherence": 5,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 4.2
        }

    # Example 13
    elif idx == 13:
        return {
            "notes": "sessionMemory: 'Found 'Training Materials - Q4' from Nov 2' (45 chars) - References specific search result with date. Good context. Score: 4. toolContext: 'Resuming Q4 training development' (33 chars) - Clear purpose showing continuation. Score: 3. goal_coherence: 'Load training session' → 'Load Q4 training' - Second goal more specific but hierarchy weak (both about loading). Score: 3. prompt_naturalness: User provides Result object with session list. Not natural. Score: 2. response_realism: Natural explanation 'Found your Q4 training materials from November 2nd (last Saturday). Loading it now.' Tool call with createContinuationSession and descriptive name. Rich result object shown with sessions array and metadata. Score: 5.",
            "sessionMemory_quality": 4,
            "toolContext_quality": 3,
            "goal_coherence": 3,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 3.4
        }

    # Example 14
    elif idx == 14:
        return {
            "notes": "sessionMemory: Empty array [] - Automatic 1 per rubric (empty). toolContext: Contains JSON object '{\"currentPath\": \"/home/user/Projects\", \"openFiles\": [], \"recentCommands\": []}' instead of string - This is malformed per schema. The explanation is missing. Score: 1. goal_coherence: 'Delete old test session and load current work' → 'List sessions to find test session' - Clear hierarchy showing multi-step plan. Score: 4. prompt_naturalness: 'Delete the old test session from last month and then load my current project work.' - Natural, includes temporal reference and sequence. Score: 5. response_realism: Tool call with limit and order. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 1,
            "goal_coherence": 4,
            "prompt_naturalness": 5,
            "response_realism": 3,
            "overall_quality": 2.8
        }

    # Example 15
    elif idx == 15:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'User updating release documentation to Beta version' (52 chars) - Explains why the replacement is needed with context. Score: 4. goal_coherence: 'Replace text' → 'Update versions' - Generic goals with minimal hierarchy. Score: 2. prompt_naturalness: 'In the Release-Notes.md file, replace all instances of 'Alpha v1.0' with 'Beta v2.0'' - Natural with specific details. Score: 4. response_realism: Tool call with findReplaceContent including specific strings. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 4,
            "goal_coherence": 2,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 2.8
        }

    # Example 16
    elif idx == 16:
        return {
            "notes": "sessionMemory: 'Created Documentation folder' (29 chars) - References specific prior action. Score: 3. toolContext: 'User organizing documentation' (30 chars) - Generic, doesn't explain why API subfolder needed. Score: 2. goal_coherence: 'Create folders' → 'Create API' - Very terse, second goal incomplete (API what?). Poor. Score: 2. prompt_naturalness: User provides Result object. Not natural. Score: 2. response_realism: Natural continuation of folder creation series. Tool call structure proper. No result shown. Score: 3.",
            "sessionMemory_quality": 3,
            "toolContext_quality": 2,
            "goal_coherence": 2,
            "prompt_naturalness": 2,
            "response_realism": 3,
            "overall_quality": 2.4
        }

    # Example 17
    elif idx == 17:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'User wants markdown file listing' (33 chars) - Clear purpose. Score: 3. goal_coherence: 'Find all .md files in Docs' → 'Search for markdown files' - Clear hierarchy. Score: 4. prompt_naturalness: 'Find all markdown files in my Docs folder and list them.' - Natural, clear request. Score: 4. response_realism: Tool call with query '*.md' and searchType 'files'. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 3.0
        }

    # Example 18
    elif idx == 18:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'User wants to create new note' (30 chars) - Generic. Score: 2. goal_coherence: 'Reorganize project files: create new structure, move notes, update references, and create index' → 'Execute createContent' - First goal is rich multi-step plan, second is just action name. Odd hierarchy. Score: 3. prompt_naturalness: User provides Result object showing batch move results. Not natural. Score: 2. response_realism: Tool call creates INDEX.md with wiki-style links and status note. Good structure. No result shown. Score: 4.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 2,
            "goal_coherence": 3,
            "prompt_naturalness": 2,
            "response_realism": 4,
            "overall_quality": 2.4
        }

    # Example 19
    elif idx == 19:
        return {
            "notes": "sessionMemory: 'Found 4 markdown files in Research folder' (43 chars) - References specific prior result with count. Score: 4. toolContext: 'Appending file list to bibliography' (36 chars) - Clear purpose. Score: 3. goal_coherence: 'Update bibliography note' → 'Append research file references' - Clear hierarchy. Score: 4. prompt_naturalness: User provides Result object with file list. Not natural. Score: 2. response_realism: Natural explanation 'Found 4 markdown files in Research. Adding them to your bibliography now.' Tool call with wiki-style links. Rich result object shown with matches array. Score: 5.",
            "sessionMemory_quality": 4,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 3.6
        }

    # Example 20
    elif idx == 20:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'User adding metadata' (21 chars) - Very generic. Score: 2. goal_coherence: 'Prepend frontmatter to Blog/ai-trends-2025.md' → 'Add YAML frontmatter' - Specific but second goal just restates format. Score: 3. prompt_naturalness: 'Add frontmatter metadata to my blog post about AI trends.' - Natural, clear intent. Score: 4. response_realism: Tool call with realistic YAML frontmatter including title, author, date, tags, category. No result shown. Score: 4.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 2,
            "goal_coherence": 3,
            "prompt_naturalness": 4,
            "response_realism": 4,
            "overall_quality": 2.8
        }

    # Example 21
    elif idx == 21:
        return {
            "notes": "sessionMemory: Empty array [] - Automatic 1. toolContext: Object '{\"currentPath\": \"/home/user/Projects\", \"openFiles\": [], \"recentCommands\": []}' instead of string - Malformed. Score: 1. goal_coherence: 'Load API documentation session' → 'Load existing session' - Redundant, minimal hierarchy. Score: 2. prompt_naturalness: User provides Result object. Not natural. Score: 2. response_realism: Natural explanation 'Found your API documentation session from this morning. Loading it now.' Tool call with createContinuationSession. Rich sessions result shown. Score: 5.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 1,
            "goal_coherence": 2,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 2.2
        }

    # Example 22
    elif idx == 22:
        return {
            "notes": "sessionMemory: 'Updated package.json. Now updating version.ts and app-config.json' (67 chars) - Shows clear workflow sequence with specific files. Score: 4. toolContext: 'Updating version constant' (26 chars) - Clear and specific. Score: 3. goal_coherence: 'Update all versions' → 'Update version.ts' - Clear hierarchy with specific file. Score: 4. prompt_naturalness: User provides Result object. Not natural. Score: 2. response_realism: Tool call with replaceContent showing specific old/new version strings. Good structure. Score: 4.",
            "sessionMemory_quality": 4,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 4,
            "overall_quality": 3.4
        }

    # Example 23
    elif idx == 23:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'Prepend heading' (15 chars) - Extremely short and generic. Score: 1. goal_coherence: 'Add heading to Recipes/Winter Chai.md' → 'Mark tasting' - Specific file path but second goal cryptic. Score: 3. prompt_naturalness: 'Add the winter tasting heading to Recipes/Winter Chai.md and update the spice note.' - Natural with specific file and two actions mentioned. Score: 4. response_realism: Tool call with markdown heading. Only addresses first part of user request (heading), ignores 'update the spice note'. Incomplete. Score: 2.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 1,
            "goal_coherence": 3,
            "prompt_naturalness": 4,
            "response_realism": 2,
            "overall_quality": 2.2
        }

    # Example 24
    elif idx == 24:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'Prepend heading' (15 chars) - Extremely short and generic. Score: 1. goal_coherence: 'Add heading to Recipes/Lunar Tea.md' → 'Mark winter tasting' - Specific file, clear purpose. Score: 4. prompt_naturalness: 'Add the winter tasting heading to Recipes/Lunar Tea.md and update the spice note to mention cinnamon star.' - Natural, specific, mentions two tasks. Score: 4. response_realism: Tool call adds heading but ignores second task ('update the spice note to mention cinnamon star'). Incomplete response. Score: 2.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 1,
            "goal_coherence": 4,
            "prompt_naturalness": 4,
            "response_realism": 2,
            "overall_quality": 2.4
        }

    # Example 25
    elif idx == 25:
        return {
            "notes": "sessionMemory: 'Captured latest states via memoryManager_listStates' (53 chars) - References specific prior tool call. Score: 4. toolContext: 'Log snapshot IDs' (16 chars) - Short but clear. Score: 3. goal_coherence: 'Update Writing/Novel/State Log.md' → 'Record two entries' - Specific file and concrete count. Score: 4. prompt_naturalness: User provides Result object with state list. Not natural. Score: 2. response_realism: Tool call appends formatted log entries with timestamps, IDs, and descriptions. Rich result shown with states array and metadata. Score: 5.",
            "sessionMemory_quality": 4,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 3.6
        }

    # Example 26
    elif idx == 26:
        return {
            "notes": "sessionMemory: 'Backed up 1/3 files' (20 chars) - Shows progress tracking with specific count. Score: 3. toolContext: 'Continuing backup' (18 chars) - Shows sequence awareness. Score: 3. goal_coherence: 'Copy templates to backup' → 'Duplicate file 2 of 3' - Clear hierarchy with progress indicator. Score: 4. prompt_naturalness: User provides Result object. Not natural. Score: 2. response_realism: Tool call continues backup series. Rich result shown with source/target paths and metadata. Score: 5.",
            "sessionMemory_quality": 3,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 3.4
        }

    # Example 27
    elif idx == 27:
        return {
            "notes": "sessionMemory: Empty string '' - Automatic 1. toolContext: 'Identify saved standup state' (29 chars) - Clear immediate purpose. Score: 3. goal_coherence: 'Find Standup Sync Snapshot' → 'List most recent states' - Shows search strategy, clear hierarchy. Score: 4. prompt_naturalness: 'Restore the Standup Sync snapshot and drop today's blockers into Standups/Week-46.md.' - Natural, mentions two tasks with specific file. Score: 4. response_realism: Tool call with includeContext, limit, order parameters. No result shown. Score: 3.",
            "sessionMemory_quality": 1,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 3.0
        }

    # Example 28
    elif idx == 28:
        return {
            "notes": "sessionMemory: 'Found Ops/Metrics/Ops Metrics Summary.md via vaultLibrarian_searchContent' (76 chars) - References specific prior tool call with full file path. Excellent. Score: 5. toolContext: 'Refresh intro block' (20 chars) - Clear and specific. Score: 3. goal_coherence: 'Update headline numbers' → 'Replace lines 1-3' - Clear hierarchy with specific line range. Score: 4. prompt_naturalness: User provides Result object with search results. Not natural. Score: 2. response_realism: Tool call uses replaceByLine with specific line range and realistic metric updates with unicode arrows. Rich result shown with search metadata (score: 0.94, executionTime: 112). Score: 5.",
            "sessionMemory_quality": 5,
            "toolContext_quality": 3,
            "goal_coherence": 4,
            "prompt_naturalness": 2,
            "response_realism": 5,
            "overall_quality": 3.8
        }

    # Example 29
    elif idx == 29:
        return {
            "notes": "sessionMemory: 'User activating docs agent' (27 chars) - Generic, lacks context about why. Score: 2. toolContext: 'Toggle agent on' (16 chars) - Very generic, just restates action. Score: 2. goal_coherence: 'Enable agent' → 'Activate docs' - Both very short and similar, minimal hierarchy. Score: 2. prompt_naturalness: 'Turn on the documentation agent.' - Natural, conversational. Score: 4. response_realism: Tool call with id parameter. No result shown. Score: 3.",
            "sessionMemory_quality": 2,
            "toolContext_quality": 2,
            "goal_coherence": 2,
            "prompt_naturalness": 4,
            "response_realism": 3,
            "overall_quality": 2.6
        }

    return {
        "notes": "No scoring implemented for this index",
        "sessionMemory_quality": 1,
        "toolContext_quality": 1,
        "goal_coherence": 1,
        "prompt_naturalness": 1,
        "response_realism": 1,
        "overall_quality": 1.0
    }


def main():
    input_file = "/home/user/Toolset-Training/Datasets/quality_review/sample_batch_37.jsonl"
    output_file = "/home/user/Toolset-Training/Datasets/quality_review/scored_batch_37.jsonl"

    scored_examples = []

    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():
                example = json.loads(line)
                scored = score_example(example, idx)
                scored_examples.append(scored)

    # Write scored examples
    with open(output_file, 'w') as f:
        for example in scored_examples:
            f.write(json.dumps(example) + '\n')

    print(f"Scored {len(scored_examples)} examples")
    print(f"Output written to: {output_file}")

    # Calculate statistics
    overall_scores = [ex["quality_scores"]["overall_quality"] for ex in scored_examples]
    avg_score = sum(overall_scores) / len(overall_scores)
    print(f"\nAverage overall_quality: {avg_score:.2f}")
    print(f"Min: {min(overall_scores):.1f}, Max: {max(overall_scores):.1f}")

    # Score distribution
    score_ranges = {
        "Excellent (4.0-5.0)": sum(1 for s in overall_scores if s >= 4.0),
        "Good (3.0-3.9)": sum(1 for s in overall_scores if 3.0 <= s < 4.0),
        "Fair (2.0-2.9)": sum(1 for s in overall_scores if 2.0 <= s < 3.0),
        "Poor (1.0-1.9)": sum(1 for s in overall_scores if s < 2.0)
    }

    print("\nScore Distribution:")
    for range_name, count in score_ranges.items():
        print(f"  {range_name}: {count} examples")


if __name__ == "__main__":
    main()
