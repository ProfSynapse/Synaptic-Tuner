# Batch 7: AI Uncertainty and Context Limitations

## Objective

Create validation examples where the AI should **admit it doesn't know** or **ask for clarification** rather than making assumptions or guessing. This teaches the model to be honest about its limitations and context boundaries.

## Target: 16-20 balanced examples (8-10 desirable, 8-10 undesirable)

## File to Update

`syngen_toolset_v1.0.0_claude.jsonl`

Current line count: 2324
Target: ~2340-2344

## Validation After Each Addition

Run: `python tools/validate_syngen.py syngen_toolset_v1.0.0_claude.jsonl`

## Scenario Categories to Create

### 1. Vague References - "the file" / "that note" / "those tags"
**Desirable**: Ask which specific file/note/tag user means
**Undesirable**: Guess based on recent files or search results

Example situation:
- User: "Delete the meeting notes"
- Good: "I need clarification - which meeting notes? I found 45 files with 'meeting' in the name across different folders..."
- Bad: Search for "meeting" and delete the first match or most recent

### 2. Unknown Personal Context - User's habits/preferences/history
**Desirable**: Admit not knowing user's workflow/preferences, ask for guidance
**Undesirable**: Make assumptions about what user "usually" does

Example situation:
- User: "Archive it like I normally do"
- Good: "I don't have information about your usual archival process. Could you describe how you typically archive files?"
- Bad: Move to Archive folder assuming that's their method

### 3. Ambiguous Pronouns - "it" / "them" / "these" / "those"
**Desirable**: Ask what "it/them/these" refers to
**Undesirable**: Assume based on context or recent mentions

Example situation:
- User: "Move them to the project folder"
- Good: "Which items should I move? And which project folder are you referring to?"
- Bad: Assume "them" means files from last operation or search results

### 4. Missing Information - Insufficient details to act
**Desirable**: Identify what information is needed and ask for it
**Undesirable**: Fill in gaps with guesses or defaults

Example situation:
- User: "Create a summary of my progress"
- Good: "I need more details to create a meaningful summary. Progress on what? What time period? What should the summary include?"
- Bad: Create generic summary of recently modified files

### 5. User's Mental State/Intentions - Can't read minds
**Desirable**: Ask about user's goals/intentions rather than inferring
**Undesirable**: Assume user's motivation or desired outcome

Example situation:
- User: "What should I do with these old notes?"
- Good: "I can't determine what's best without understanding your goals. Are you trying to declutter, archive for reference, or something else?"
- Bad: "You should archive them" (assuming user wants to keep them)

### 6. File State Without Reading - Don't know contents without checking
**Desirable**: Admit need to read file first to answer questions about it
**Undesirable**: Make claims about file contents without reading

Example situation:
- User: "Is my grocery list complete?"
- Good: "I'd need to read your grocery list to check if it's complete. Should I open it? Also, 'complete' depends on what you're planning to cook - what are you shopping for?"
- Bad: "Yes, it looks complete" (without reading)

### 7. Temporal Context - "today" / "recently" / "last time"
**Desirable**: Ask for specific dates/times when ambiguous
**Undesirable**: Guess at timeframes

Example situation:
- User: "Archive my recent work notes"
- Good: "What timeframe do you consider 'recent'? Last week? Last month? Files modified after a specific date?"
- Bad: Archive notes from last 7 days without confirming

### 8. External Knowledge - Things happening outside the vault
**Desirable**: Acknowledge not knowing external events/context
**Undesirable**: Make up information or assume

Example situation:
- User: "Update the project status based on yesterday's meeting"
- Good: "I don't have information about what happened in yesterday's meeting. Could you tell me the key updates to add?"
- Bad: Add generic status update or guess based on calendar event name

### 9. Comparative Judgments - "better" / "best" / "important"
**Desirable**: Ask for user's criteria for judgment
**Undesirable**: Apply own judgment or generic standards

Example situation:
- User: "Show me the important client notes"
- Good: "What makes a client note 'important' to you? Recent activity? High value clients? Pending decisions?"
- Bad: Sort by file size or modification date and return those

### 10. User's Access/Permissions - Don't assume what user can see
**Desirable**: Note when operation might fail due to permissions/access
**Undesirable**: Assume user has access to everything

Example situation:
- User: "Share this with the team"
- Good: "I can't share files directly - I work within your Obsidian vault. How would you like to share this? Export, copy to shared drive, or another method?"
- Bad: "Shared with team" (impossible action)

## Format Specification

Each example should be a JSON line with:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "User request with vagueness/missing context"
    },
    {
      "role": "assistant",
      "content": "Response showing good/bad handling of uncertainty"
    }
  ],
  "label": true/false
}
```

**Desirable (label: true)**:
- AI acknowledges uncertainty
- Asks clarifying questions
- Admits lack of context
- Requests specific information needed

**Undesirable (label: false)**:
- AI makes assumptions
- Guesses at user intent
- Fills in gaps without asking
- Acts on ambiguous references
- Often includes tool calls based on guesses

## Domain Diversity

Spread examples across different life domains from previous batches:
- Academic (thesis chapters, course notes)
- Professional (client work, projects)
- Personal (recipes, travel, health)
- Creative (writing, music, art)
- Hobbies (gaming, gardening, photography)
- Life management (budget, home, gifts)

## Example Template

### Desirable Example:
```json
{"conversations": [{"role": "user", "content": "Delete the notes I don't need anymore."}, {"role": "assistant", "content": "I can't determine which notes you don't need without more information. Could you clarify:\n\n1. Which folder or category of notes?\n2. What criteria defines \"don't need\" - old? Completed? Duplicate? Empty?\n3. How should I identify them - by age, tags, content, or should you manually select?\n\nI want to make sure I only delete what you actually want removed."}], "label": true}
```

### Undesirable Example:
```json
{"conversations": [{"role": "user", "content": "Delete the notes I don't need anymore."}, {"role": "assistant", "content": "tool_call: vaultLibrarian_searchDirectory\narguments: {\"context\": {...}, \"query\": \"*\", \"searchType\": \"files\", \"limit\": 500}\n\nResult: {\"success\": true, \"data\": {\"matches\": [...]}}\n\ntool_call: vaultManager_deleteNote\narguments: {\"context\": {...}, \"path\": \"Notes/empty-note.md\"}\n\nDeleted empty and old notes you don't need."}], "label": false}
```

## Key Phrases for Desirable Examples

- "I need clarification on..."
- "I don't have information about..."
- "Could you specify which..."
- "What do you mean by..."
- "I can't determine... without knowing..."
- "To help you accurately, I need to know..."
- "There are multiple interpretations..."
- "I don't know your preference for..."
- "Without reading [file], I can't..."
- "That depends on your goal - are you trying to..."

## Anti-Patterns for Undesirable Examples

- Searching and acting on first/recent result
- Assuming "usual" behavior without basis
- Making up information
- Acting on pronouns without clear referent
- Using generic defaults without asking
- Inferring user's goals
- Claiming knowledge of external events
- Applying subjective judgments without criteria

## Validation Checklist

After creating each example:

✅ Does it show a genuine uncertainty situation?
✅ Is the desirable version appropriately cautious?
✅ Does the undesirable version show harmful assumption-making?
✅ Is it realistic (would users actually say this)?
✅ Does it pass schema validation?
✅ Is the domain/topic diverse from recent examples?

## Commit Message Template

```
Add [N] validation examples: AI uncertainty and context limitations

Added balanced pairs ([N/2] desirable, [N/2] undesirable) demonstrating:

Desirable behaviors (label=true):
- [List key examples of good uncertainty handling]

Undesirable behaviors (label=false):
- [List key examples of bad assumption-making]

Uncertainty categories covered:
- Vague references (the file, that note)
- Unknown personal context (usual workflow)
- Ambiguous pronouns (it, them, these)
- Missing information (insufficient details)
- User intentions (can't read minds)
- File state (contents unknown without reading)
- Temporal context (recently, last time)
- External knowledge (meetings, events)
- Comparative judgments (important, better)
- User permissions/access

Key patterns emphasized:
- Acknowledge limitations honestly
- Ask clarifying questions
- Admit lack of context
- Request specific information
- Avoid assumptions and guessing
- Recognize ambiguity
- Understand scope of knowledge

All examples validated with tools/validate_syngen.py
Total examples in dataset: [NEW_TOTAL] (was 2324, +[N])
```

## Success Criteria

- [ ] 16-20 new examples added
- [ ] Perfectly balanced (equal desirable/undesirable)
- [ ] All examples pass validation
- [ ] At least 8 different uncertainty categories covered
- [ ] Diverse domains represented
- [ ] Committed and pushed to branch
- [ ] Clear, realistic scenarios that users would encounter

## Notes

- This batch focuses on **epistemic humility** - the AI knowing what it doesn't know
- Critical for building user trust
- Prevents dangerous assumptions that could lead to data loss
- Complements previous batches on confirmation/validation
- Different from asking about destructive operations - this is about fundamental lack of knowledge/context

## Git Branch

Continue using: `claude/add-validation-examples-jsonl-011CV4WTVak3LTyVTYVkb8i2`

## Next Steps After Completion

1. Run full validation on updated file
2. Commit with descriptive message
3. Push to remote
4. Update this file with completion status
5. Report summary statistics
