# Execute Prompt Usage Training Dataset

## Purpose

Teach the model WHEN to use `agentManager_executePrompt`, HOW to write effective prompts with proper structure, and WHAT to do with the response.

## Three Core Skills

### 1. WHEN to Use executePrompt

**Appropriate use cases (positive examples):**
- Complex reasoning beyond simple tool operations
- Specialized expertise needed (legal analysis, medical advice, system architecture)
- Creative tasks (writing, design recommendations, naming)
- Research synthesis and analysis
- Strategic planning and recommendations
- Multi-step reasoning requiring deeper context
- Tasks requiring broader knowledge base

**Inappropriate use (anti-patterns):**
- Simple file operations (search, read, write, delete, move)
- Basic organization/cleanup tasks
- Direct tool calls the model can handle itself
- Straightforward text responses within capability
- List/summarize existing data
- Simple transformations or formatting

### 2. HOW to Write Effective Prompts

All prompts to executePrompt should use this structure:

```markdown
# CONTEXT
[Background information, what's happened so far, relevant state]

# MISSION
[Single clear sentence: what needs to be accomplished]

# INSTRUCTIONS
[Step-by-step or numbered list of what to do]

# GUIDELINES
[Constraints, preferences, quality standards, what to avoid]

# OUTPUT FORMAT (optional)
[Specific format requirements if needed]

---

[Any additional context: file contents, notes, data to analyze]
```

**Quality criteria:**
- Clear and specific mission statement
- Sufficient context without overwhelming
- Concrete instructions (not vague)
- Include relevant data/notes after the structure
- Appropriate for the complexity of the task

### 3. WHAT to Do with Response

**Return directly to user:**
- User asked a question → provide answer
- Requested analysis → show results
- Quick recommendations → present them
- Research findings → explain

**Save to note (contentManager_createContent):**
- Documentation that should persist
- Reports for later reference
- Analysis with long-term value
- Structured deliverables
- Templates, guidelines, specs

**Both (explain + save):**
- "I created a comprehensive analysis and saved it to Reports/analysis.md"
- Summarize what was created, then save full version
- Show key points, save complete document

## Dataset Structure

Generate examples covering:

1. **Appropriate delegation (60 examples)**
   - Positive: Complex tasks SHOULD use executePrompt with good prompts
   - Negative: Same task but model does it directly without delegation

2. **Inappropriate delegation (40 examples)**
   - Positive: Simple tasks handled directly WITHOUT executePrompt
   - Negative: Anti-pattern of using executePrompt for trivial tasks

3. **Prompt quality (40 examples)**
   - Positive: Well-structured prompts with all sections
   - Negative: Poor prompts (missing sections, vague, unstructured)

4. **Response handling (40 examples)**
   - Return only: User question → executePrompt → return answer
   - Save only: Generate doc → executePrompt → save to note
   - Both: Complex deliverable → executePrompt → explain + save

**Total: 180 examples (90 pairs)**

## Example Format

### Appropriate Delegation (Positive)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "What's the best architecture for a real-time collaborative editing system?"
    },
    {
      "role": "assistant",
      "content": "tool_call: agentManager_executePrompt\narguments: {\"context\": {...}, \"prompt\": \"# CONTEXT\\nUser building real-time collaborative editing system, needs architectural guidance...\\n\\n# MISSION\\nDesign a scalable architecture for real-time collaborative editing with conflict resolution...\\n\\n# INSTRUCTIONS\\n1. Recommend core architectural patterns\\n2. Explain conflict resolution strategies\\n3. Suggest technology stack\\n4. Address scalability concerns\\n\\n# GUIDELINES\\n- Focus on proven patterns (OT, CRDT)\\n- Consider latency and consistency tradeoffs\\n- Provide concrete examples\\n\\n# OUTPUT FORMAT\\nArchitectural overview with diagrams and implementation guidance\", \"returnContent\": true}"
    }
  ],
  "label": true,
  "behavior": "execute_prompt_usage",
  "pattern": "appropriate_delegation"
}
```

### Inappropriate Delegation (Anti-pattern)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Search for all meeting notes from last week"
    },
    {
      "role": "assistant",
      "content": "tool_call: agentManager_executePrompt\narguments: {\"context\": {...}, \"prompt\": \"# MISSION\\nFind all meeting notes from last week\\n\\n# INSTRUCTIONS\\nSearch the vault for meeting notes\", \"returnContent\": true}"
    }
  ],
  "label": false,
  "behavior": "execute_prompt_usage",
  "pattern": "inappropriate_delegation"
}
```

Should be:
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Search for all meeting notes from last week"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {...}, \"query\": \"meeting notes\", \"limit\": 50}"
    }
  ],
  "label": true,
  "behavior": "execute_prompt_usage",
  "pattern": "direct_handling"
}
```

## Validation

Run standard validator:
```bash
python tools/validate_syngen.py Datasets/execute_prompt_usage/pairs_v1.0.jsonl
```

## Scenarios to Cover

**Complex reasoning (USE executePrompt):**
- System architecture design
- Legal/compliance analysis
- Medical/health guidance
- Strategic business planning
- Creative writing (stories, marketing copy)
- Research synthesis across domains
- Technical deep-dives (security audits, performance optimization)
- Ethical considerations and recommendations

**Simple operations (DON'T use executePrompt):**
- File search, read, write, delete
- List directories or sessions
- Move/organize files
- Simple text responses
- Basic filtering or sorting
- Straightforward summaries of existing data
- Direct tool calls

**Prompt quality examples:**
- Good: All sections, clear mission, specific instructions, relevant context
- Bad: Missing sections, vague mission, no context, unstructured

**Response handling:**
- Return: Q&A, analysis requests, recommendations
- Save: Documentation, reports, templates, specifications
- Both: Complex deliverables with user explanation
