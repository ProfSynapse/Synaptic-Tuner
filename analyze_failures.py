#!/usr/bin/env python3
"""Analyze evaluation run failures in detail."""

import json
from collections import defaultdict
from pathlib import Path

def load_run(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    # Load both runs
    run1 = load_run('Evaluator/results/run_20251208_110128.json')
    run2 = load_run('Evaluator/results/run_20251208_164506.json')

    print("=" * 70)
    print("EVALUATION FAILURE ANALYSIS")
    print("=" * 70)
    print(f"\nModel: {run1['metadata'].get('model_name', 'nexus-tools_sft26')}")
    print(f"Run 1 (Morning): {run1['summary']['passed']}/{run1['summary']['total']} passed")
    print(f"Run 2 (Evening): {run2['summary']['passed']}/{run2['summary']['total']} passed")
    print(f"Pass Rate Change: {run1['summary']['passed']/run1['summary']['total']*100:.1f}% → {run2['summary']['passed']/run2['summary']['total']*100:.1f}%")

    # Critical behavioral failure patterns
    print("\n" + "=" * 70)
    print("CRITICAL BEHAVIORAL FAILURES (Run 2)")
    print("=" * 70)

    behavior_issues = defaultdict(int)
    for record in run2['records']:
        if not record.get('passed') and record.get('behavior'):
            for issue in record['behavior'].get('issues', []):
                check = issue.get('check', 'unknown')
                msg = issue.get('message', 'No message')
                key = f"{check}"
                behavior_issues[key] += 1

    print("\nTop Behavioral Issues:")
    for issue, count in sorted(behavior_issues.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:2d}× {issue}")

    # Tool execution failures
    print("\n" + "=" * 70)
    print("TOOL EXECUTION FAILURES")
    print("=" * 70)

    missing_tools = defaultdict(int)
    for record in run2['records']:
        if not record.get('passed'):
            expected = set(record.get('expected_tools', []))
            called_tools = record.get('validator', {})
            if called_tools:
                called = set(tc.get('name', '') for tc in called_tools.get('tool_calls', []))
            else:
                called = set()

            for tool in expected - called:
                if tool:  # Skip empty strings
                    missing_tools[tool] += 1

    print("\nTools Not Called When Expected:")
    for tool, count in sorted(missing_tools.items(), key=lambda x: -x[1])[:8]:
        print(f"  {count:2d}× {tool}")

    # Specific pattern analysis
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # Over-eager tool calling
    text_only_fails = []
    for r in run2['records']:
        if not r.get('passed'):
            behavior = r.get('behavior', {})
            if behavior.get('response_type_detected') == 'tool_only':
                for issue in behavior.get('issues', []):
                    if issue.get('expected') == 'text_only':
                        text_only_fails.append(r)
                        break

    print(f"\n1. Over-Eager Tool Calling: {len(text_only_fails)} cases")
    print("   Model calls tools when it should ask clarifying questions")
    if text_only_fails:
        example = text_only_fails[0]
        print(f"   Example: {example.get('case_id', 'N/A')}")
        print(f"   Question: \"{example.get('question', 'N/A')[:70]}...\"")

    # Missing clarification
    no_clarification = []
    for r in run2['records']:
        if not r.get('passed') and r.get('behavior'):
            for issue in r['behavior'].get('issues', []):
                if issue.get('check') == 'asks_for_user_input' and not issue.get('passed'):
                    no_clarification.append(r)
                    break

    print(f"\n2. Missing User Clarification: {len(no_clarification)} cases")
    print("   Model doesn't ask for input when ambiguity is present")

    # Immediate tool call anti-pattern
    immediate_calls = []
    for r in run2['records']:
        if not r.get('passed') and r.get('behavior'):
            for issue in r['behavior'].get('issues', []):
                if 'immediate_tool_call' in str(issue.get('check', '')):
                    immediate_calls.append(r)
                    break

    print(f"\n3. Immediate Tool Call Anti-Pattern: {len(immediate_calls)} cases")
    print("   Model calls tools without thinking/analyzing first")

    # Session memory issues
    empty_memory = []
    for r in run2['records']:
        if not r.get('passed') and r.get('behavior'):
            for issue in r['behavior'].get('issues', []):
                if 'sessionMemory' in str(issue.get('message', '')):
                    empty_memory.append(r)
                    break

    print(f"\n4. Session Memory Issues: {len(empty_memory)} cases")
    print("   Empty or insufficient sessionMemory in context")

    # Wrong tool delegation
    wrong_delegation = []
    for r in run2['records']:
        if not r.get('passed') and r.get('behavior'):
            for issue in r['behavior'].get('issues', []):
                check = issue.get('check', '')
                if 'delegates_complex_task' in check or 'uses_execute_prompt' in check:
                    wrong_delegation.append(r)
                    break

    print(f"\n5. Wrong Tool Delegation: {len(wrong_delegation)} cases")
    print("   Should use executePrompt for complex tasks but uses other tools")

    # Category performance
    print("\n" + "=" * 70)
    print("WORST PERFORMING CATEGORIES")
    print("=" * 70)

    category_stats = defaultdict(lambda: {'passed': 0, 'total': 0})
    for record in run2['records']:
        for tag in record.get('tags', []):
            category_stats[tag]['total'] += 1
            if record.get('passed'):
                category_stats[tag]['passed'] += 1

    worst_categories = []
    for cat, stats in category_stats.items():
        if stats['total'] >= 2:  # Only consider categories with 2+ tests
            pass_rate = (stats['passed'] / stats['total']) * 100
            worst_categories.append((cat, pass_rate, stats['passed'], stats['total']))

    worst_categories.sort(key=lambda x: x[1])
    print("\nCategory Performance (Pass Rate):")
    for cat, rate, passed, total in worst_categories[:12]:
        print(f"  {rate:5.1f}% ({passed:2d}/{total:2d}) - {cat}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 70)
    print("""
CRITICAL ISSUES IDENTIFIED:

1. **Over-Eager Tool Calling (9 cases)**
   - Model jumps to tool execution without clarifying ambiguous requests
   - Training needs more examples of asking questions before acting

2. **Missing executePrompt Usage (8+ cases)**
   - Model uses contentManager_createContent for complex generation tasks
   - Should delegate to agentManager_executePrompt instead
   - Training lacks sufficient executePrompt delegation examples

3. **Intellectual Humility Gap (0% pass rate)**
   - Model doesn't acknowledge uncertainty or limitations
   - No examples of "I'm not sure" or "I need more information"
   - Training should include uncertain/ambiguous scenarios

4. **Clarification Blindspot (0% pass rate)**
   - Model assumes user intent rather than verifying
   - Needs examples of asking follow-up questions
   - Should verify destructive operations before executing

5. **Empty SessionMemory (8 cases)**
   - Context objects missing required sessionMemory content
   - Indicates training examples may have placeholder/empty memory
   - Quality issue in training dataset

RECOMMENDED ACTIONS:

1. Add behavioral training examples for:
   - Asking clarifying questions (intellectual_humility)
   - Verifying destructive operations (verification_before_action)
   - Using executePrompt for complex tasks (delegation)
   - Populating sessionMemory with meaningful context

2. Audit existing training data for:
   - Empty or placeholder sessionMemory fields
   - Examples that assume rather than clarify
   - Missing executePrompt delegation patterns

3. Consider synthetic data generation focused on:
   - Ambiguous user requests requiring clarification
   - Complex tasks requiring delegation
   - Multi-step workflows with verification steps
    """)

if __name__ == '__main__':
    main()
