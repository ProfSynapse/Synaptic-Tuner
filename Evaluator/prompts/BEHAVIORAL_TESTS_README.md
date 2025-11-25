# Behavioral Pattern Evaluation Tests

## Overview

This test set evaluates whether the model has learned **meta-behaviors** beyond basic tool usage:
1. **ExecutePrompt Usage** - When to delegate, how to prompt, what to do with responses
2. **Context Efficiency** - Operating within a sliding context window

## Test Categories

### 1. ExecutePrompt Usage (10 tests)

#### Appropriate Delegation (3 tests)
Tests whether model correctly identifies complex tasks that require `agentManager_executePrompt`:
- `execute_prompt_complex_architecture` - Microservices architecture design
- `execute_prompt_legal_analysis` - GDPR compliance analysis
- `execute_prompt_research_synthesis` - Research synthesis across 50 papers

**Expected behavior:**
- ✅ Uses `agentManager_executePrompt` for complex reasoning
- ✅ Creates well-structured prompts with markdown sections
- ✅ Recognizes task is beyond simple tool operations

#### Inappropriate Delegation (2 tests)
Tests whether model avoids unnecessary delegation for simple tasks:
- `no_execute_prompt_simple_search` - Simple file search
- `no_execute_prompt_file_organization` - Simple file move

**Expected behavior:**
- ✅ Does NOT use `agentManager_executePrompt` for simple operations
- ✅ Uses appropriate direct tools (searchContent, moveFolder)
- ✅ Recognizes task is within model's capability

#### Prompt Structure (1 test)
Tests whether model creates well-structured prompts:
- `execute_prompt_prompt_quality_good` - Security audit framework

**Expected behavior:**
- ✅ Prompt includes all sections: CONTEXT, MISSION, INSTRUCTIONS, GUIDELINES, OUTPUT FORMAT
- ✅ Mission statement is clear and specific
- ✅ Instructions are step-by-step and actionable

#### Response Handling (3 tests)
Tests whether model handles executePrompt responses appropriately:
- `execute_prompt_response_return` - PostgreSQL vs MySQL comparison (return directly)
- `execute_prompt_response_save` - API design guide (save to note)
- `execute_prompt_response_both` - Onboarding plan (explain + save)

**Expected behavior:**
- ✅ Return: Shows response directly for Q&A
- ✅ Save: Saves documentation to persistent file
- ✅ Both: Explains to user AND saves for complex deliverables

### 2. Context Efficiency (11 tests)

#### Line-Based Reading (3 tests)
Tests whether model reads specific line ranges instead of entire large files:
- `context_efficiency_large_log` - Read last error from 50k-line log
- `context_efficiency_doc_section` - Read specific section from 5k-line doc
- `context_efficiency_error_context` - Read ±20 lines around error

**Expected behavior:**
- ✅ Uses `offset` and `limit` parameters
- ✅ Reads specific ranges (last 100 lines, lines 800-900, ±20 lines)
- ✅ Avoids reading entire large files

#### Efficient Search (2 tests)
Tests whether model uses `includeContent: false` when appropriate:
- `context_efficiency_find_files_no_content` - Find files for organization
- `context_efficiency_verify_existence` - Check if files exist

**Expected behavior:**
- ✅ Uses `includeContent: false` when only need file paths
- ✅ Saves context by not loading unnecessary content
- ✅ Appropriate limits (20-50, not 500+)

#### SessionMemory Leverage (2 tests)
Tests whether model uses sessionMemory instead of re-reading:
- `context_efficiency_session_memory` - Update config using prior read
- `context_efficiency_continuation` - Add endpoint using prior search

**Expected behavior:**
- ✅ References prior reads from sessionMemory
- ✅ Avoids re-reading files when information already known
- ✅ Uses line numbers/sections from prior context

#### Appropriate Limits (2 tests)
Tests whether model uses reasonable search limits:
- `context_efficiency_appropriate_limit` - Search recent meeting notes
- `context_efficiency_search_limit_excessive` - Avoid excessive limits

**Expected behavior:**
- ✅ Uses limits in range 20-50 for most searches
- ✅ Avoids excessive limits (500+, 1000+)
- ✅ Can search again if more results needed

#### Chunked Operations (1 test)
Tests whether model processes large tasks in chunks:
- `context_efficiency_chunked_processing` - Analyze 100 feedback files

**Expected behavior:**
- ✅ Processes in batches (10-20 items at a time)
- ✅ Tracks progress in sessionMemory
- ✅ Avoids trying to load everything at once

## Running Behavioral Tests

### Run All Behavioral Tests
```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/behavioral_patterns.json \
  --output Evaluator/results/behavioral_$(date +%s).json \
  --markdown Evaluator/results/behavioral_report.md
```

### Run Specific Categories

**ExecutePrompt Usage Only:**
```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/behavioral_patterns.json \
  --filter-tag executePrompt \
  --output Evaluator/results/execute_prompt_$(date +%s).json
```

**Context Efficiency Only:**
```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/behavioral_patterns.json \
  --filter-tag context_efficiency \
  --output Evaluator/results/context_efficiency_$(date +%s).json
```

## Interpreting Results

### ExecutePrompt Usage Scoring

**Appropriate Delegation (3 tests):**
- ✅ Pass: Uses executePrompt with structured prompt
- ⚠️ Partial: Uses executePrompt but poor prompt structure
- ❌ Fail: Tries to handle complex task directly

**Inappropriate Delegation (2 tests):**
- ✅ Pass: Uses appropriate direct tool
- ❌ Fail: Unnecessarily uses executePrompt

**Prompt Structure (1 test):**
- ✅ Pass: All sections present (CONTEXT, MISSION, INSTRUCTIONS, GUIDELINES)
- ⚠️ Partial: Some sections present
- ❌ Fail: Unstructured or minimal prompt

**Response Handling (3 tests):**
- ✅ Pass: Correct handling (return, save, or both as appropriate)
- ❌ Fail: Wrong handling (saves when should return, or vice versa)

### Context Efficiency Scoring

**Line-Based Reading (3 tests):**
- ✅ Pass: Uses offset/limit to read specific ranges
- ⚠️ Partial: Reads more than needed but not entire file
- ❌ Fail: Reads entire large file

**Efficient Search (2 tests):**
- ✅ Pass: Uses includeContent: false and reasonable limit
- ⚠️ Partial: Correct includeContent but excessive limit
- ❌ Fail: Includes content unnecessarily

**SessionMemory Leverage (2 tests):**
- ✅ Pass: Uses prior context, avoids re-reading
- ❌ Fail: Re-reads despite having information

**Appropriate Limits (2 tests):**
- ✅ Pass: Uses limit 20-100
- ⚠️ Partial: Uses limit 100-200
- ❌ Fail: Uses limit 500+

**Chunked Operations (1 test):**
- ✅ Pass: Processes in chunks with progress tracking
- ❌ Fail: Tries to load all items at once

## Success Criteria

### Minimum Passing Scores

**ExecutePrompt Usage:**
- Appropriate delegation: 2/3 (67%)
- Inappropriate delegation: 2/2 (100%)
- Prompt structure: 1/1 (100%)
- Response handling: 2/3 (67%)
- **Overall: 7/10 (70%)**

**Context Efficiency:**
- Line-based reading: 2/3 (67%)
- Efficient search: 2/2 (100%)
- SessionMemory leverage: 2/2 (100%)
- Appropriate limits: 2/2 (100%)
- Chunked operations: 1/1 (100%)
- **Overall: 9/11 (82%)**

**Combined: 16/21 (76%)**

### Production-Ready Scores

**ExecutePrompt Usage: 9/10 (90%)**
**Context Efficiency: 10/11 (91%)**
**Combined: 19/21 (90%)**

## Notes on [Previous: ...] Tests

Some tests use `[Previous: ...]` notation to simulate multi-step scenarios:
- `context_efficiency_session_memory`
- `context_efficiency_continuation`
- `execute_prompt_response_*` tests

For these tests, you may need to:
1. Provide the "previous" context in the test setup
2. Evaluate whether the model correctly references that context
3. Check sessionMemory for proper context tracking

## Comparison with Other Test Sets

**baseline.json** (6 tests)
- General tool usage
- Multi-step workflows
- Clarification handling

**full_coverage.json** (47 tests)
- One test per tool
- Comprehensive tool coverage
- Single-tool operations

**tool_combos.json** (varies)
- Multi-tool workflows
- Complex sequences
- Integration scenarios

**behavioral_patterns.json** (21 tests) ← THIS FILE
- Meta-behaviors
- When/how to use tools
- Context-awareness
- Quality of tool usage

## Expected Model Evolution

**After SFT training:**
- Should pass basic tool usage tests (baseline, full_coverage)
- May fail behavioral pattern tests (not yet learned meta-behaviors)

**After behavioral KTO training:**
- Should pass behavioral pattern tests
- Demonstrates context-awareness
- Uses executePrompt appropriately
- Operates efficiently within context window

**After combined training:**
- Passes all test sets
- Production-ready behavior
- Efficient and appropriate tool usage
