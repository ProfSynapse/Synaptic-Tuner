"""
Rubric Runner - Discovers, composes, and runs modular validation rubrics.

Usage:
    # Interactive selection
    python -m improvement_engine.services.rubric_runner --file dataset.jsonl --line 1

    # Specify rubrics via flag
    python -m improvement_engine.services.rubric_runner --file dataset.jsonl --line 1 \
        --rubrics factuality,context_alignment,confidence_calibration

    # List available rubrics
    python -m improvement_engine.services.rubric_runner --list
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.llm import create_client
from shared.llm.config import LLMConfig
from ..utils.yaml_loader import load_yaml
from ..utils.logger import ImproveLogger
from .schema_validator import SchemaValidator


@dataclass
class RubricInfo:
    """Metadata about a rubric."""
    name: str
    description: str
    filename: str
    scope: str
    pass_threshold: float


@dataclass
class JudgmentResult:
    """Result from judging an example."""
    passed: bool
    scores: Dict[str, float]
    feedback: Dict[str, str]
    raw_judgment: Dict


class RubricRunner:
    """
    Discovers, composes, and runs modular validation rubrics.

    1. Scans rubrics/ folder for YAML files
    2. Lets user select which rubrics to run (interactive or via flag)
    3. Composes selected judge prompts into one system prompt
    4. Runs judge → improve loop until all pass
    """

    def __init__(
        self,
        backend: str = "lmstudio",
        model: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        logger: Optional[ImproveLogger] = None,
    ):
        """
        Initialize rubric runner.

        Args:
            backend: LLM backend (lmstudio, ollama, openrouter)
            model: Model name
            host: Host override for LM Studio/Ollama
            port: Port override
            logger: Logger instance
        """
        self.logger = logger or ImproveLogger()
        self.rubrics_dir = Path(__file__).parent.parent / "rubrics"

        # Setup LLM client
        config = LLMConfig.from_env(env_prefix="IMPROVEMENT")
        config.provider = backend
        if model:
            config.model = model
        if host and backend == "lmstudio":
            config.lmstudio_host = host
        if port and backend == "lmstudio":
            config.lmstudio_port = port
        if host and backend == "ollama":
            config.ollama_host = host
        if port and backend == "ollama":
            config.ollama_port = port

        self.llm_client = create_client(config=config)
        self.backend = backend

        # Discover available rubrics
        self.available_rubrics = self._discover_rubrics()

    def _discover_rubrics(self) -> Dict[str, RubricInfo]:
        """Scan rubrics folder and extract metadata."""
        rubrics = {}

        # Skip these files (legacy or non-rubric configs)
        skip_files = {"hallucination.yaml", "thinking_quality.yaml", "quality_labels.yaml"}

        for yaml_file in self.rubrics_dir.glob("*.yaml"):
            if yaml_file.name in skip_files:
                continue

            try:
                data = load_yaml(yaml_file)
                key = yaml_file.stem  # filename without extension
                rubrics[key] = RubricInfo(
                    name=data.get("name", key),
                    description=data.get("description", ""),
                    filename=yaml_file.name,
                    scope=data.get("scope", "response"),
                    pass_threshold=data.get("pass_threshold", 0.8),
                )
            except Exception as e:
                self.logger.warning(f"Could not load rubric {yaml_file.name}: {e}")

        return rubrics

    def list_rubrics(self) -> List[RubricInfo]:
        """List all available rubrics."""
        return list(self.available_rubrics.values())

    def select_rubrics_interactive(self) -> List[str]:
        """Interactive checkbox selection of rubrics."""
        print("\nAvailable Rubrics:")
        print("-" * 60)

        rubric_keys = list(self.available_rubrics.keys())
        for i, key in enumerate(rubric_keys, 1):
            info = self.available_rubrics[key]
            print(f"  [{i}] {info.name}")
            print(f"      {info.description}")
            print()

        print("Enter numbers separated by commas (e.g., 1,2,3)")
        print("Or 'all' for all rubrics, 'q' to quit")

        while True:
            choice = input("\nSelect rubrics: ").strip().lower()

            if choice == 'q':
                return []

            if choice == 'all':
                return rubric_keys

            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [rubric_keys[i] for i in indices if 0 <= i < len(rubric_keys)]
                if selected:
                    return selected
                print("No valid selections. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter numbers like: 1,2,3")

    def _load_rubric(self, key: str) -> Dict:
        """Load full rubric data from YAML."""
        yaml_file = self.rubrics_dir / f"{key}.yaml"
        return load_yaml(yaml_file)

    def _compose_judge_prompt(
        self,
        rubric_keys: List[str],
        system_prompt: str,
        user_request: str,
        assistant_response: str,
    ) -> str:
        """Compose a single judge prompt from multiple rubrics."""

        parts = [
            "You are evaluating an AI assistant's response against multiple quality criteria.",
            "",
            "=" * 60,
            "EVALUATION CRITERIA",
            "=" * 60,
        ]

        # Add each rubric's judge prompt
        for key in rubric_keys:
            rubric = self._load_rubric(key)
            parts.append("")
            parts.append(rubric.get("judge_prompt", f"# {key} check"))

        # Add the example to evaluate
        parts.extend([
            "",
            "=" * 60,
            "EXAMPLE TO EVALUATE",
            "=" * 60,
            "",
            "**System Prompt:**",
            "```",
            system_prompt,
            "```",
            "",
            "**User Request:**",
            "```",
            user_request,
            "```",
            "",
            "**Assistant Response:**",
            "```",
            assistant_response,
            "```",
            "",
            "=" * 60,
            "YOUR EVALUATION",
            "=" * 60,
            "",
            "Evaluate the response against ALL criteria above.",
            "Output a single JSON object with:",
            "- One score field per rubric (0.0-1.0)",
            "- One `overall_feedback` field explaining all evaluations",
            "Evaluate the response against ALL criteria above.",
            "Output a single JSON object with:",
            "- One score field per rubric (0.0-1.0)",
            "- One `overall_feedback` field explaining all evaluations",
            "",
            "IMPORTANT: Respond with ONLY valid JSON. No markdown, no explanations.",
            "Start with { and end with }",
        ])

        return "\n".join(parts)

    def _build_combined_schema(self, rubric_keys: List[str]) -> Dict:
        """Build combined output schema from all selected rubrics."""
        properties = {}
        required = []

        for key in rubric_keys:
            rubric = self._load_rubric(key)
            schema = rubric.get("output_schema", {})

            # Add all properties from this rubric's schema
            for prop_name, prop_def in schema.get("properties", {}).items():
                properties[prop_name] = prop_def

            # Add required fields
            required.extend(schema.get("required", []))

        # Add combined feedback field
        properties["overall_feedback"] = {
            "type": "string",
            "description": "Combined explanation covering all rubric evaluations"
        }
        required.append("overall_feedback")

        return {
            "type": "object",
            "properties": properties,
            "required": list(set(required)),  # dedupe
            "additionalProperties": False
        }

    def _extract_conversations(self, example: Dict) -> Tuple[str, str, str]:
        """Extract system prompt, user request, assistant response from example."""
        system_prompt = ""
        user_request = ""
        assistant_response = ""

        for conv in example.get("conversations", []):
            role = conv.get("role", "")
            content = conv.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                user_request = content
            elif role == "assistant":
                assistant_response = content

                # Add tool_calls if present
                if "tool_calls" in conv and conv["tool_calls"]:
                    assistant_response += "\n\n[TOOL_CALLS]\n"
                    for tc in conv["tool_calls"]:
                        func = tc.get("function", {})
                        assistant_response += f"\ntool_call: {func.get('name', 'unknown')}\n"
                        assistant_response += f"arguments: {func.get('arguments', '{}')}\n"

        return system_prompt, user_request, assistant_response

    def _validate_schemas_programmatic(
        self,
        example: Dict,
        rubric_keys: List[str]
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Programmatically validate content against all rubric schemas (DRY approach).

        For any rubric with a scope that requires validation (thinking, system_prompt, etc.),
        extracts the content, validates against the rubric's output_schema, and returns errors.

        Args:
            example: Example to validate
            rubric_keys: List of rubric keys to validate

        Returns:
            Dict mapping rubric_key to (is_valid, list_of_errors)
        """
        validation_results = {}

        for key in rubric_keys:
            try:
                rubric = self._load_rubric(key)
                scope = rubric.get("scope", "response")

                # Only validate scopes that have structured content (not plain "response" text)
                if scope in ["thinking", "system_prompt"]:
                    validator = SchemaValidator(key, logger=self.logger)

                    # Extract content based on scope
                    content = validator.extract_content(example)

                    if content is None:
                        validation_results[key] = (False, [f"No {scope} content found"])
                        continue

                    # For system_prompt, validate the structure itself
                    if scope == "system_prompt":
                        # Validate XML structure programmatically
                        system_prompt_text = content.get("system_prompt", "")
                        errors = []

                        required_sections = [
                            '<session_context>',
                            '<vault_structure>',
                            '<available_workspaces>',
                            '<available_agents>',
                            '<selected_workspace'
                        ]

                        for section in required_sections:
                            if section not in system_prompt_text:
                                errors.append(f"Missing required section: {section}")

                        # Validate JSON in <selected_workspace>
                        if '<selected_workspace' in system_prompt_text:
                            match = re.search(r'<selected_workspace[^>]*>(.*?)</selected_workspace>',
                                            system_prompt_text, re.DOTALL)
                            if match:
                                try:
                                    ws_json = json.loads(match.group(1).strip())
                                    # Check for required fields in workspace JSON
                                    required_ws_fields = ['context', 'workspaceStructure', 'workflows']
                                    for field in required_ws_fields:
                                        if field not in ws_json:
                                            errors.append(f"Missing field in selected_workspace: {field}")
                                except json.JSONDecodeError as e:
                                    errors.append(f"Invalid JSON in selected_workspace: {str(e)}")

                        validation_results[key] = (len(errors) == 0, errors)

                    elif scope == "thinking":
                        # Validate thinking block JSON schema
                        is_valid, errors = validator.validate(content)
                        validation_results[key] = (is_valid, errors)

            except Exception as e:
                self.logger.warning(f"Schema validation error for {key}: {e}")
                validation_results[key] = (False, [f"Validation error: {str(e)}"])

        return validation_results

    def judge(
        self,
        example: Dict,
        rubric_keys: List[str],
    ) -> JudgmentResult:
        """
        Judge an example against selected rubrics.

        Args:
            example: Example dict with conversations
            rubric_keys: List of rubric keys to evaluate against

        Returns:
            JudgmentResult with pass/fail, scores, and feedback
        """
        system_prompt, user_request, assistant_response = self._extract_conversations(example)

        # FIRST: Run programmatic schema validation (fast, deterministic, DRY)
        schema_validation_results = self._validate_schemas_programmatic(example, rubric_keys)

        # Build composed prompt and schema
        judge_prompt = self._compose_judge_prompt(
            rubric_keys, system_prompt, user_request, assistant_response
        )
        output_schema = self._build_combined_schema(rubric_keys)

        try:
            # Get judgment from LLM
            judgment = self.llm_client.structured_output(
                messages=[{"role": "user", "content": judge_prompt}],
                schema=output_schema
            )

            # Check if all rubrics pass
            scores = {}
            feedback = {}
            all_passed = True

            # Add programmatic schema validation results for each rubric
            for rubric_key, (is_valid, errors) in schema_validation_results.items():
                schema_key = f"{rubric_key}_schema"
                scores[schema_key] = 1.0 if is_valid else 0.0
                if not is_valid:
                    all_passed = False
                    feedback[schema_key] = f"Schema validation failed: {'; '.join(errors)}"

            for key in rubric_keys:
                rubric = self._load_rubric(key)
                threshold = rubric.get("pass_threshold", 0.8)
                schema = rubric.get("output_schema", {})
                schema_props = schema.get("properties", {})

                # Find score field from the rubric's schema (look for fields ending in _score)
                score_field = None
                for prop_name in schema_props:
                    if prop_name.endswith("_score"):
                        score_field = prop_name
                        break

                # Get score from judgment
                score = None
                if score_field and score_field in judgment:
                    score = judgment[score_field]
                else:
                    # Fallback: If score field not found, default to 0.5
                    self.logger.warning(f"Score field '{score_field}' not found in judgment for {key}")
                    score = 0.5

                scores[key] = score

                if score < threshold:
                    all_passed = False

                # Find feedback field from schema (look for fields ending in _feedback)
                feedback_field = None
                for prop_name in schema_props:
                    if prop_name.endswith("_feedback"):
                        feedback_field = prop_name
                        break

                if feedback_field and feedback_field in judgment:
                    feedback[key] = judgment[feedback_field]

            return JudgmentResult(
                passed=all_passed,
                scores=scores,
                feedback=feedback,
                raw_judgment=judgment
            )

        except Exception as e:
            self.logger.error(f"Judge error: {e}")
            return JudgmentResult(
                passed=False,
                scores={k: 0.0 for k in rubric_keys},
                feedback={k: f"Error: {str(e)}" for k in rubric_keys},
                raw_judgment={"error": str(e)}
            )

    def _build_improve_prompt_by_scope(
        self,
        scope: str,
        example: Dict,
        feedback_text: str,
        rubric_keys: List[str]
    ) -> str:
        """
        Build scope-specific improve prompt.

        Args:
            scope: What to regenerate (system_prompt, thinking, response, user)
            example: Current example
            feedback_text: Judge feedback to fix
            rubric_keys: Rubrics being evaluated

        Returns:
            Improve prompt for that scope
        """
        system_prompt, user_request, assistant_response = self._extract_conversations(example)

        if scope == "system_prompt":
            return f"""Rewrite the system prompt to fix all issues.

## Current System Prompt (has problems)
```
{system_prompt}
```

## User Request (for context)
```
{user_request}
```

## Assistant Response (for extracting tool call paths)
```
{assistant_response}
```

## Problems to Fix
{feedback_text}

## Required Format

Generate a system prompt with ALL 5 required sections:

1. <session_context>
   - Preserve existing sessionId and workspaceId from current system prompt
   - Format: sessionId: "..." and workspaceId: "..."

2. <vault_structure>
   - List root-level folders based on domain (infer from user request and tool calls)
   - For dev: Projects/, Docs/, Config/
   - For notes: Notes/, Articles/, Resources/
   - Include root files: README.md, todo.md

3. <available_workspaces>
   - Generate 1-2 realistic workspaces based on domain
   - Each needs: name, id, description, root folder path

4. <available_agents>
   - Preserve any existing custom agents from current system prompt
   - Add note: "Built-in agents (ContentManager, VaultLibrarian, etc.) are always available via MCP tools."

5. <selected_workspace name="..." id="...">
   - Generate JSON object with ALL fields:
     * context: {{"id": "...", "name": "...", "description": "...", "rootFolder": "..."}}
     * workflows: [{{"id": "...", "name": "...", "steps": [...]}}] (at least 1 workflow)
     * workspaceStructure: [{{"path": "...", "type": "folder", "children": [...]}}] (at least 3 folders)
     * recentFiles: [] (extract from tool calls if present)
     * keyFiles: {{"readme": "README.md", ...}}
     * preferences: "..." (domain-appropriate)
     * sessions: [{{"id": "...", "summary": "..."}}]

   **CRITICAL for workspaceStructure:**
   - Extract file paths from tool calls in assistant response
   - Add those paths to workspaceStructure
   - Add 2-3 realistic neighbor files in same folders
   - Use realistic file names for the domain

## Output

Generate the COMPLETE rewritten system prompt with all 5 sections.
Output ONLY the XML/JSON system prompt, no explanations."""

        elif scope == "thinking":
            return f"""Regenerate ONLY the thinking block to fix schema issues.

## Original Inputs (the ONLY source of truth)

**System Prompt:**
```
{system_prompt}
```

**User Request:**
```
{user_request}
```

## Current Thinking Block (has problems)
{self._extract_thinking_block_text(example)}

## Problems to Fix
{feedback_text}

## Required Format

The thinking block must be VALID JSON inside `<thinking>` tags:

```
<thinking>
{{"goal": "...", "memory": "...", "requirements": [...], "assessment": {{"complex": false, "risky": false}}, "confidence": 0.8, "plan": [...]}}
</thinking>
```

Required fields:
- goal (string): What to do (use ONLY info from user request)
- memory (string): Reference the user's request context (min 20 chars)
- requirements (array): Prerequisites (file exists, etc.) - NOT actions
- assessment (object): {{"complex": bool, "risky": bool}}
- confidence (number): 0.3-0.5 risky, 0.6-0.8 medium, 0.85-0.95 safe
- plan (array): Action steps to take - NOT prerequisites

**CRITICAL**: Use JSON format with double quotes, NOT YAML format.

## Output

Generate ONLY the thinking block in <thinking> tags. No other text."""

        elif scope == "response":
            # Get original tool calls for format reference
            original_tool_calls = None
            for conv in example.get("conversations", []):
                if conv.get("role") == "assistant" and "tool_calls" in conv:
                    original_tool_calls = conv["tool_calls"]
                    break

            tool_format_instructions = ""
            if original_tool_calls:
                original_tool = original_tool_calls[0] if original_tool_calls else {}
                original_func = original_tool.get("function", {})
                original_tool_name = original_func.get("name", "unknown")
                try:
                    original_args = json.loads(original_func.get("arguments", "{}"))
                except:
                    original_args = {}

                tool_format_instructions = f"""
## Tool Call Format

The original response used tool: `{original_tool_name}`
With these parameters: {json.dumps(list(original_args.keys()))}

If making a tool call, use this EXACT format:

```
[TOOL_CALL]
tool_name: {original_tool_name}
arguments:
{{
  {chr(10).join(f'  "{k}": "<value>",' for k in original_args.keys())}
}}
```

**Original tool call for reference:**
```json
{json.dumps(original_args, indent=2)}
```

IMPORTANT for tool calls:
- Use the SAME tool name and parameter names
- sessionId and workspaceId MUST match system prompt exactly
- Only use file paths explicitly mentioned in user request OR reasonably inferable from system prompt context
- If user request is vague but system prompt provides context → tool call is valid
- If both vague → ask for clarification instead"""

            return f"""Regenerate the assistant response (text and/or tool calls) to fix issues.

## Original Inputs (the ONLY source of truth)

**System Prompt:**
```
{system_prompt}
```

**User Request:**
```
{user_request}
```

## Current Response (has problems)
```
{assistant_response}
```

## Problems to Fix
{feedback_text}

## CRITICAL Rules

1. **NO FABRICATION**: Only use information from system prompt or user request
   - sessionId, workspaceId: Use values from <session_context> in system prompt
   - Available agents: Use agents listed in <available_agents> in system prompt
   - File paths: Only if explicitly in user request OR reasonably inferable from context
   - NEVER invent: dates, edit history, fabricated file contents

2. **If user request is vague**:
   - Check system prompt for context (sessionId, workspaceId, workspace structure)
   - If system prompt has context → tool call is VALID
   - If no context → ask for clarification

{tool_format_instructions}

## Output

Generate the assistant response (text and/or tool calls).
Do NOT include <thinking> tags (that's handled separately).
Output ONLY the response content."""

        else:
            # Unknown scope, return generic prompt
            return f"""Fix the following issues:

{feedback_text}

Generate improved content for scope: {scope}"""

    def _extract_thinking_block_text(self, example: Dict) -> str:
        """Extract thinking block text from assistant response."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "assistant":
                content = conv.get("content", "")
                match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
                if match:
                    return f"<thinking>\n{match.group(1).strip()}\n</thinking>"
        return "(No thinking block found)"

    def _apply_improvement_by_scope(
        self,
        scope: str,
        new_content: str,
        example: Dict
    ) -> Dict:
        """
        Apply scope-specific improvement to example.

        Args:
            scope: What was regenerated
            new_content: New content from LLM
            example: Example to update

        Returns:
            Updated example
        """
        improved = json.loads(json.dumps(example))  # Deep copy

        if scope == "system_prompt":
            # Replace system message
            for conv in improved.get("conversations", []):
                if conv.get("role") == "system":
                    conv["content"] = new_content.strip()
                    return improved

            # If no system message exists, add one at the beginning
            improved.setdefault("conversations", []).insert(0, {
                "role": "system",
                "content": new_content.strip()
            })

        elif scope == "thinking":
            # Replace thinking block in assistant message
            for conv in improved.get("conversations", []):
                if conv.get("role") == "assistant":
                    content = conv.get("content", "")

                    # Extract new thinking block
                    match = re.search(r"<thinking>(.*?)</thinking>", new_content, re.DOTALL)
                    if match:
                        new_thinking = f"<thinking>\n{match.group(1).strip()}\n</thinking>"

                        # Replace existing thinking block or prepend if none exists
                        if "<thinking>" in content:
                            content = re.sub(
                                r"<thinking>.*?</thinking>",
                                new_thinking,
                                content,
                                flags=re.DOTALL
                            )
                        else:
                            content = new_thinking + "\n\n" + content

                        conv["content"] = content

        elif scope == "response":
            # Replace assistant text and tool calls
            for conv in improved.get("conversations", []):
                if conv.get("role") == "assistant":
                    # Parse tool calls from new content
                    new_tool_calls = self._parse_tool_calls_from_response(new_content)

                    # Remove tool call block from text content
                    clean_content = self._remove_tool_call_block(new_content)

                    # Keep existing thinking block if present
                    old_thinking_match = re.search(r"<thinking>.*?</thinking>", conv.get("content", ""), re.DOTALL)
                    if old_thinking_match:
                        old_thinking = old_thinking_match.group(0)
                        if "<thinking>" not in clean_content:
                            clean_content = old_thinking + "\n\n" + clean_content

                    conv["content"] = clean_content

                    # Update tool_calls
                    if new_tool_calls:
                        conv["tool_calls"] = new_tool_calls
                    elif "tool_calls" in conv:
                        del conv["tool_calls"]

        return improved

    def improve(
        self,
        example: Dict,
        judgment: JudgmentResult,
        rubric_keys: List[str],
    ) -> Dict:
        """
        Improve an example based on judgment feedback using scope-based regeneration.

        Detects which scopes failed and regenerates them in order:
        1. system_prompt - System message
        2. thinking - Thinking block in assistant response
        3. response - Assistant text and tool calls

        Args:
            example: Original example
            judgment: Judgment result with feedback
            rubric_keys: Rubrics that were evaluated

        Returns:
            Improved example dict
        """
        try:
            # Detect which scopes need regeneration
            scopes_to_regenerate = self._detect_failed_scopes(judgment, rubric_keys)

            if not scopes_to_regenerate:
                self.logger.info("No scopes need regeneration (all passed)")
                return example

            self.logger.info(f"Regenerating scopes in order: {scopes_to_regenerate}")

            # Start with current example
            improved = example

            # Regenerate each scope in order: system_prompt → thinking → response
            for scope in scopes_to_regenerate:
                # Build feedback for this scope only
                feedback_parts = []

                # Include schema validation errors for this scope
                schema_key = f"{scope}_schema"
                for rubric_key in rubric_keys:
                    rubric = self._load_rubric(rubric_key)
                    if rubric.get("scope") == scope:
                        # Add schema feedback
                        if schema_key in judgment.scores and judgment.scores[schema_key] < 1.0:
                            fb = judgment.feedback.get(schema_key, "Schema validation failed")
                            feedback_parts.append(f"**{schema_key}** (score: {judgment.scores[schema_key]:.2f}, needs: 1.0):\n{fb}")

                        # Add rubric feedback
                        if rubric_key in judgment.scores:
                            score = judgment.scores[rubric_key]
                            threshold = rubric.get("pass_threshold", 0.8)
                            if score < threshold:
                                fb = judgment.feedback.get(rubric_key, "No specific feedback")
                                feedback_parts.append(f"**{rubric_key}** (score: {score:.2f}, needs: {threshold}):\n{fb}")

                feedback_text = "\n\n".join(feedback_parts) if feedback_parts else "Fix issues"

                # Build scope-specific improve prompt
                improve_prompt = self._build_improve_prompt_by_scope(
                    scope, improved, feedback_text, rubric_keys
                )

                # Regenerate this scope
                self.logger.info(f"Regenerating scope: {scope}")
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": improve_prompt}],
                    temperature=0.3,
                )

                content = response if isinstance(response, str) else response.get("content", "")

                # Apply improvement for this scope
                improved = self._apply_improvement_by_scope(scope, content, improved)

            # Validate schemas after all improvements
            improved = self._validate_and_fix_schemas(improved, example, rubric_keys)

            return improved

        except Exception as e:
            self.logger.error(f"Improve error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return example

    def _detect_failed_scopes(
        self,
        judgment: JudgmentResult,
        rubric_keys: List[str]
    ) -> List[str]:
        """
        Detect which scopes failed and need regeneration.

        Returns scopes in order: system_prompt → thinking → response

        Args:
            judgment: Judgment result
            rubric_keys: Rubrics evaluated

        Returns:
            List of scopes to regenerate (ordered)
        """
        failed_scopes = set()

        # Check schema validation failures
        for score_key, score_value in judgment.scores.items():
            if score_key.endswith("_schema") and score_value < 1.0:
                # Extract scope from schema key (e.g., "system_prompt_format_schema" → "system_prompt")
                for rubric_key in rubric_keys:
                    if score_key.startswith(rubric_key):
                        rubric = self._load_rubric(rubric_key)
                        scope = rubric.get("scope", "response")
                        failed_scopes.add(scope)
                        break

        # Check rubric failures
        for key in rubric_keys:
            rubric = self._load_rubric(key)
            threshold = rubric.get("pass_threshold", 0.8)
            score = judgment.scores.get(key, 0)

            if score < threshold:
                scope = rubric.get("scope", "response")
                failed_scopes.add(scope)

        # Return in order: system_prompt → thinking → response
        scope_order = ["system_prompt", "user", "thinking", "response"]
        return [s for s in scope_order if s in failed_scopes]

    def _parse_tool_calls_from_response(self, content: str) -> Optional[List[Dict]]:
        """Parse tool calls from improved response text."""
        if "[TOOL_CALL]" not in content and "tool_name:" not in content:
            return None

        tool_calls = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for tool_name: pattern
            if line.startswith("tool_name:"):
                tool_name = line.split("tool_name:")[1].strip()

                # Look for arguments block
                args_str = ""
                if i + 1 < len(lines) and "arguments:" in lines[i + 1]:
                    i += 2  # Skip "arguments:" line
                    # Collect JSON
                    json_lines = []
                    brace_count = 0
                    while i < len(lines):
                        json_line = lines[i]
                        json_lines.append(json_line)
                        brace_count += json_line.count("{") - json_line.count("}")
                        if brace_count <= 0 and "{" in "".join(json_lines):
                            break
                        i += 1
                    args_str = "\n".join(json_lines)

                try:
                    args = json.loads(args_str) if args_str else {}
                    tool_calls.append({
                        "id": f"call_{hash(tool_name) % 10000:04x}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args)
                        }
                    })
                except json.JSONDecodeError:
                    pass

            i += 1

        return tool_calls if tool_calls else None

    def _remove_tool_call_block(self, content: str) -> str:
        """Remove tool call block from content, keeping thinking and text."""
        lines = content.split("\n")
        clean_lines = []
        skip_until_done = False

        for line in lines:
            if "[TOOL_CALL]" in line or line.strip().startswith("tool_name:"):
                skip_until_done = True
                continue
            if skip_until_done:
                # Skip until we're past the JSON block
                if line.strip().startswith("}") or (line.strip() == "" and clean_lines and clean_lines[-1].strip() == "}"):
                    skip_until_done = False
                continue
            clean_lines.append(line)

        return "\n".join(clean_lines).strip()

    def _validate_and_fix_schemas(self, improved: Dict, original: Dict, rubric_keys: List[str]) -> Dict:
        """
        Validate content against all rubric schemas based on their scope.

        Args:
            improved: Improved example to validate
            original: Original example (for fallback)
            rubric_keys: List of rubric keys being evaluated

        Returns:
            Validated and potentially fixed example
        """
        for key in rubric_keys:
            rubric = self._load_rubric(key)
            scope = rubric.get("scope", "response")

            if scope == "thinking":
                # Use existing thinking schema validation logic
                improved = self._validate_and_fix_thinking_schema(improved, original)
            elif scope == "system_prompt":
                # Validate system prompt structure
                # For now, log validation but don't modify
                # Judge will catch issues in next iteration
                try:
                    validator = SchemaValidator(key, logger=self.logger)
                    system_prompt = self._extract_system_prompt_text(improved)

                    # Check for required XML sections
                    required_sections = [
                        '<session_context>',
                        '<vault_structure>',
                        '<available_workspaces>',
                        '<available_agents>',
                        '<selected_workspace'
                    ]

                    missing_sections = [s for s in required_sections if s not in system_prompt]
                    if missing_sections:
                        self.logger.warning(f"System prompt missing sections: {missing_sections}")

                    # Validate JSON in <selected_workspace> if present
                    if '<selected_workspace' in system_prompt:
                        match = re.search(r'<selected_workspace[^>]*>(.*?)</selected_workspace>',
                                        system_prompt, re.DOTALL)
                        if match:
                            try:
                                ws_json = json.loads(match.group(1).strip())
                                # Could validate structure here, but judge handles it
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Invalid JSON in selected_workspace: {e}")
                except Exception as e:
                    self.logger.warning(f"System prompt validation error: {e}")
            # Additional scopes can be added here (tool_calls, response, etc.)

        return improved

    def _extract_system_prompt_text(self, example: Dict) -> str:
        """Extract system prompt text from example."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "system":
                return conv.get("content", "")
        return ""

    def _validate_and_fix_thinking_schema(self, improved: Dict, original: Dict) -> Dict:
        """
        Convert YAML-style thinking to JSON format.

        Does NOT patch missing fields - that's handled by the thinking_schema rubric
        which will force regeneration if schema is invalid.
        """
        import re

        for conv in improved.get("conversations", []):
            if conv.get("role") == "assistant":
                content = conv.get("content", "")

                # Extract thinking block
                match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
                if not match:
                    continue

                thinking_str = match.group(1).strip()

                # Try to parse as JSON first
                try:
                    thinking_block = json.loads(thinking_str)
                    # Already valid JSON - just ensure it's clean
                    new_thinking_str = json.dumps(thinking_block)
                    new_content = content.replace(match.group(0), f"<thinking>\n{new_thinking_str}\n</thinking>")
                    conv["content"] = new_content
                except json.JSONDecodeError:
                    # Not valid JSON - try to convert YAML-style to JSON
                    thinking_block = self._parse_yaml_style_thinking(thinking_str)
                    if thinking_block is not None:
                        # Successfully parsed YAML - convert to JSON
                        new_thinking_str = json.dumps(thinking_block)
                        new_content = content.replace(match.group(0), f"<thinking>\n{new_thinking_str}\n</thinking>")
                        conv["content"] = new_content
                        self.logger.info("Converted YAML-style thinking to JSON")
                    else:
                        # Could not parse - leave as-is, the thinking_schema rubric will fail it
                        self.logger.warning("Could not parse thinking block - will be caught by schema rubric")
                break

        return improved

    def _parse_yaml_style_thinking(self, thinking_str: str) -> Optional[Dict]:
        """
        Parse YAML-style thinking block into a dictionary.

        Handles format like:
            goal: Some goal text
            memory: Some memory text
            requirements: None
            assessment:
              complex: false
              risky: true
            confidence: 0.6
            plan:
            - Step 1
            - Step 2
        """
        import re

        result = {}
        lines = thinking_str.split("\n")
        current_key = None
        current_value = []
        in_array = False
        in_nested = False
        nested_key = None
        nested_obj = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for array item
            if stripped.startswith("- "):
                if current_key:
                    if not isinstance(result.get(current_key), list):
                        result[current_key] = []
                    result[current_key].append(stripped[2:].strip())
                continue

            # Check for nested key (indented key: value)
            if line.startswith("  ") and ":" in stripped and nested_key:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    nested_obj[k] = True
                elif v.lower() == "false":
                    nested_obj[k] = False
                else:
                    nested_obj[k] = v
                continue

            # Check for top-level key
            if ":" in stripped:
                # Save any nested object
                if nested_key and nested_obj:
                    result[nested_key] = nested_obj
                    nested_obj = {}
                    nested_key = None

                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                current_key = k

                if v == "" or v.lower() == "none":
                    # Could be start of nested object or array
                    nested_key = k
                    result[k] = [] if k in ["requirements", "plan"] else {}
                elif v.lower() in ["true", "false"]:
                    result[k] = v.lower() == "true"
                else:
                    try:
                        result[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        result[k] = v

        # Handle final nested object
        if nested_key and nested_obj:
            result[nested_key] = nested_obj

        # Validate we got the required fields
        required = ["goal", "memory", "requirements", "assessment", "confidence", "plan"]
        if not all(k in result for k in required):
            return None

        return result

    def _extract_thinking_from_example(self, example: Dict) -> Optional[Dict]:
        """Extract and parse thinking block from an example."""
        import re

        for conv in example.get("conversations", []):
            if conv.get("role") == "assistant":
                content = conv.get("content", "")
                match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1).strip())
                    except json.JSONDecodeError:
                        return self._parse_yaml_style_thinking(match.group(1).strip())
        return None

    def run(
        self,
        example: Dict,
        rubric_keys: List[str],
        max_iterations: int = 10,
    ) -> Tuple[bool, Dict, List[JudgmentResult]]:
        """
        Run judge/improve loop until all rubrics pass.

        For local providers (lmstudio, ollama), runs unlimited iterations.
        For cloud providers (openrouter), respects max_iterations.

        Args:
            example: Example to validate/improve
            rubric_keys: Rubrics to evaluate against
            max_iterations: Max improvement attempts (ignored for local providers)

        Returns:
            Tuple of (passed, final_example, judgment_history)
        """
        current = example
        history = []

        # Local providers get unlimited retries (no API cost)
        is_local = self.backend.lower() in ["lmstudio", "ollama"]
        effective_max = 999 if is_local else max_iterations

        if is_local:
            self.logger.info(f"Using local provider ({self.backend}) - unlimited iterations")

        iteration = 0
        while iteration < effective_max:
            iteration += 1

            if is_local:
                self.logger.info(f"Iteration {iteration}")
            else:
                self.logger.info(f"Iteration {iteration}/{effective_max}")

            # Judge
            judgment = self.judge(current, rubric_keys)
            history.append(judgment)

            # Log scores
            for key, score in judgment.scores.items():
                # Schema keys are programmatic, not rubric files
                if key.endswith("_schema"):
                    threshold = 1.0
                else:
                    rubric = self._load_rubric(key)
                    threshold = rubric.get("pass_threshold", 0.8)
                status = "✓" if score >= threshold else "✗"
                self.logger.info(f"  {status} {key}: {score:.2f} (threshold: {threshold})")

            if judgment.passed:
                self.logger.success(f"All rubrics passed after {iteration} iteration(s)")
                return True, current, history

            # Improve
            self.logger.info("  Improving...")
            current = self.improve(current, judgment, rubric_keys)

        self.logger.warning(f"Could not pass all rubrics after {effective_max} iterations")
        return False, current, history


def main():
    """CLI entry point."""
    import argparse
    import time
    from concurrent.futures import ThreadPoolExecutor
    from .file_handler import FileHandler

    parser = argparse.ArgumentParser(description="Run modular validation rubrics")
    parser.add_argument("--file", type=str, help="JSONL dataset file")
    parser.add_argument("--line", type=int, help="Line number to process (1-indexed, for single line mode)")
    parser.add_argument("--output", type=str, help="Output file (required for full dataset processing)")
    parser.add_argument("--start-line", type=int, default=1, help="Starting line (for resuming)")
    parser.add_argument("--end-line", type=int, help="Ending line (for processing subset)")
    parser.add_argument("--rubrics", type=str, help="Comma-separated rubric names (or 'all')")
    parser.add_argument("--list", action="store_true", help="List available rubrics")
    parser.add_argument("--backend", default="lmstudio", help="LLM backend")
    parser.add_argument("--host", type=str, help="LLM host (e.g., 192.168.1.236)")
    parser.add_argument("--port", type=int, default=1234, help="LLM port")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max improvement iterations")
    parser.add_argument("--parallel", action="store_true", help="Process batch in parallel (cloud only)")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent workers for parallel mode")

    args = parser.parse_args()

    # Initialize runner
    runner = RubricRunner(
        backend=args.backend,
        host=args.host,
        port=args.port,
    )

    # List mode
    if args.list:
        print("\nAvailable Rubrics:")
        print("-" * 60)
        for info in runner.list_rubrics():
            print(f"  {info.filename.replace('.yaml', '')}")
            print(f"    Name: {info.name}")
            print(f"    Description: {info.description}")
            print(f"    Scope: {info.scope}")
            print(f"    Pass threshold: {info.pass_threshold}")
            print()
        return

    # Validate args
    if not args.file:
        print("Usage:")
        print("  Single line:  python -m improvement_engine.services.rubric_runner --file <dataset.jsonl> --line <N>")
        print("  Full dataset: python -m improvement_engine.services.rubric_runner --file <dataset.jsonl> --output <output.jsonl>")
        print("  List rubrics: python -m improvement_engine.services.rubric_runner --list")
        return

    file_handler = FileHandler()
    total_lines = file_handler.count_lines(args.file)

    # Select rubrics
    if args.rubrics:
        if args.rubrics.lower() == 'all':
            rubric_keys = list(runner.available_rubrics.keys())
        else:
            rubric_keys = [r.strip() for r in args.rubrics.split(",")]
    else:
        rubric_keys = runner.select_rubrics_interactive()

    if not rubric_keys:
        print("No rubrics selected.")
        return

    print(f"\nRunning rubrics: {', '.join(rubric_keys)}")

    # Determine mode: single line or full dataset
    if args.line:
        # SINGLE LINE MODE
        if args.line < 1 or args.line > total_lines:
            print(f"Line {args.line} out of range (1-{total_lines})")
            return

        examples = file_handler.read_jsonl(args.file, start_line=args.line, end_line=args.line)
        example = examples[0]

        print(f"File: {args.file}, Line: {args.line}")
        print("=" * 60)

        # Extract and show context
        for conv in example.get("conversations", []):
            if conv.get("role") == "user":
                print(f"User request: {conv.get('content', '')[:100]}...")
                break

        # Run
        passed, final_example, history = runner.run(
            example,
            rubric_keys,
            max_iterations=args.max_iterations,
        )

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Passed: {passed}")
        print(f"Iterations: {len(history)}")

        if history:
            print("\nFinal scores:")
            for key, score in history[-1].scores.items():
                print(f"  {key}: {score:.2f}")

    else:
        # FULL DATASET MODE
        if not args.output:
            print("Error: --output required for full dataset processing")
            print("  Example: --output Datasets/.../tools_v1.8.jsonl")
            return

        # Validate parallel mode
        if args.parallel and args.backend not in ["openrouter"]:
            print("ERROR: Parallel mode only supports cloud providers (openrouter)")
            return

        print(f"Input: {args.file} ({total_lines} lines)")
        print(f"Output: {args.output}")
        print(f"Starting from line: {args.start_line}")
        if args.parallel:
            print(f"Mode: PARALLEL ({args.workers} workers)")
        else:
            print(f"Mode: SEQUENTIAL")
        print("=" * 60)

        # Stats
        processed = 0
        passed_count = 0
        failed_count = 0
        start_time = time.time()

        # Collect examples to process
        examples_to_process = []
        for line_num, example in file_handler.iterate_jsonl(args.file):
            if line_num < args.start_line:
                continue
            if args.end_line and line_num > args.end_line:
                break
            examples_to_process.append((line_num, example))

        # Process examples
        if args.parallel:
            # PARALLEL MODE
            def process_one(item):
                line_num, example = item
                # Create runner per thread (thread-safe)
                thread_runner = RubricRunner(
                    backend=args.backend,
                    host=args.host,
                    port=args.port,
                )
                passed, final_example, history = thread_runner.run(
                    example,
                    rubric_keys,
                    max_iterations=args.max_iterations,
                )
                return (line_num, passed, final_example, history)

            print(f"\nProcessing {len(examples_to_process)} examples in parallel...")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(process_one, examples_to_process))

            # Process results
            for line_num, passed, final_example, history in results:
                processed += 1
                if passed:
                    passed_count += 1
                else:
                    failed_count += 1
                # Write improved example to output (append mode)
                file_handler.write_jsonl(args.output, [final_example], mode='a')

        else:
            # SEQUENTIAL MODE
            for line_num, example in file_handler.iterate_jsonl(args.file):
                if line_num < args.start_line:
                    continue

                processed += 1

                # Run improvement
                passed, final_example, history = runner.run(
                    example,
                    rubric_keys,
                    max_iterations=args.max_iterations,
                )

                if passed:
                    passed_count += 1
                else:
                    failed_count += 1
                # Write improved example to output
                file_handler.write_jsonl(args.output, [final_example], mode='a')

            # Progress stats every 10 lines
            if processed % 10 == 0:
                elapsed_min = (time.time() - start_time) / 60
                rate_per_min = processed / elapsed_min if elapsed_min > 0 else 0
                remaining = total_lines - line_num
                eta_min = remaining / rate_per_min if rate_per_min > 0 else 0
                print(f"\n--- Progress: {processed} done, {passed_count} passed, {failed_count} failed ---")
                print(f"--- Rate: {rate_per_min:.2f} lines/min, ETA: {eta_min/60:.1f} hours ({eta_min:.0f} min) ---")

        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Total processed: {processed}")
        print(f"Passed: {passed_count} ({passed_count/processed*100:.1f}%)")
        print(f"Failed: {failed_count} ({failed_count/processed*100:.1f}%)")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
