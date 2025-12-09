"""Hallucination validation service using LLM-as-judge."""

import json
import sys
from typing import Dict, Optional, Tuple
from pathlib import Path

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.llm import create_client
from ..core.exceptions import ImprovementEngineError
from ..utils.logger import ImproveLogger
from ..utils.yaml_loader import load_yaml


class HallucinationValidator:
    """
    LLM-as-judge that detects hallucinations in thinking blocks.

    Validates that thinking blocks only use information from:
    - System prompt
    - User request
    - Logical inferences about operations

    Does NOT allow made-up:
    - File paths/names not mentioned
    - Version numbers/dates not provided
    - User history not stated
    - Specific counts/metrics not given
    """

    def __init__(
        self,
        backend: str = "lmstudio",
        model: Optional[str] = None,
        logger: Optional[ImproveLogger] = None,
    ):
        """
        Initialize hallucination validator.

        Args:
            backend: LLM backend (lmstudio, ollama, openrouter)
            model: Model name (optional for lmstudio)
            logger: Logger instance
        """
        # Create LLM client using shared infrastructure
        self.llm_client = create_client(provider=backend, model=model)
        self.logger = logger or ImproveLogger()
        self.backend = backend  # Store for retry logic

        # Load prompts
        config_dir = Path(__file__).parent.parent / "config"
        prompts = load_yaml(config_dir / "system_prompts.yaml")
        self.detection_prompt = prompts["hallucination_detection_prompt"]

    def validate(
        self,
        system_prompt: str,
        user_request: str,
        thinking_block: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Validate that thinking block doesn't hallucinate.

        Args:
            system_prompt: System prompt with session context
            user_request: User's request
            thinking_block: Thinking block to validate

        Returns:
            Tuple of (is_valid, judgment_details)
            - is_valid: True if no hallucination detected
            - judgment_details: {
                "has_hallucination": bool,
                "hallucinated_details": list,
                "confidence": float,
                "explanation": str
              }
        """
        # Format the validation prompt
        full_prompt = self._build_validation_prompt(
            system_prompt, user_request, thinking_block
        )

        try:
            # Get LLM judgment using structured output
            judgment = self.llm_client.structured_output(
                messages=[{"role": "user", "content": full_prompt}],
                schema={
                    "type": "object",
                    "properties": {
                        "has_hallucination": {"type": "boolean"},
                        "hallucinated_details": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "explanation": {"type": "string"}
                    },
                    "required": ["has_hallucination", "hallucinated_details", "confidence", "explanation"]
                }
            )

            # Log if hallucination detected
            if judgment["has_hallucination"]:
                self.logger.warning(
                    f"Hallucination detected (confidence: {judgment['confidence']:.2f})"
                )
                for detail in judgment["hallucinated_details"]:
                    self.logger.warning(f"  - {detail}")

            is_valid = not judgment["has_hallucination"]
            return is_valid, judgment

        except Exception as e:
            self.logger.error(f"Error during hallucination validation: {e}")
            # In case of error, assume invalid to be safe
            return False, {
                "has_hallucination": True,
                "hallucinated_details": [f"Validation error: {str(e)}"],
                "confidence": 0.0,
                "explanation": "Could not validate due to error"
            }

    def _build_validation_prompt(
        self,
        system_prompt: str,
        user_request: str,
        thinking_block: Dict,
    ) -> str:
        """
        Build the full validation prompt.

        Args:
            system_prompt: System prompt
            user_request: User request
            thinking_block: Thinking block to validate

        Returns:
            Complete prompt for LLM judge
        """
        thinking_json = json.dumps(thinking_block, indent=2)

        prompt = f"""{self.detection_prompt}

## Input to Validate

**System Prompt:**
```
{system_prompt}
```

**User Request:**
```
{user_request}
```

**Thinking Block:**
```json
{thinking_json}
```

## Your Analysis

Determine if the thinking block contains hallucinated information.

Output your judgment as JSON following the schema:
```json
{{
  "has_hallucination": true/false,
  "hallucinated_details": ["detail 1", "detail 2", ...],
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation"
}}
```

Be STRICT - any specific detail not in system prompt or user request is a hallucination.
"""

        return prompt

    def validate_batch(
        self,
        examples: list,
    ) -> list:
        """
        Validate multiple examples for hallucination.

        Args:
            examples: List of examples with conversations

        Returns:
            List of validation results with is_valid flag
        """
        results = []

        for i, example in enumerate(examples):
            try:
                # Extract system prompt, user request, and thinking
                conversations = example.get("conversations", [])

                system_prompt = ""
                user_request = ""
                thinking_block = None

                for conv in conversations:
                    role = conv.get("role")
                    content = conv.get("content", "")

                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        user_request = content
                    elif role == "assistant":
                        # Extract thinking block from assistant response
                        thinking_block = self._extract_thinking(content)
                        break

                if not thinking_block:
                    results.append({
                        "index": i,
                        "is_valid": False,
                        "reason": "No thinking block found"
                    })
                    continue

                # Validate
                is_valid, judgment = self.validate(
                    system_prompt, user_request, thinking_block
                )

                results.append({
                    "index": i,
                    "is_valid": is_valid,
                    "judgment": judgment
                })

            except Exception as e:
                self.logger.error(f"Error validating example {i}: {e}")
                results.append({
                    "index": i,
                    "is_valid": False,
                    "reason": str(e)
                })

        return results

    def validate_and_fix(
        self,
        system_prompt: str,
        user_request: str,
        thinking_block: Dict,
        max_retries: int = 3,
    ) -> Tuple[bool, Dict, Optional[Dict]]:
        """
        Validate thinking block and auto-fix if hallucinations detected.

        Uses LLM-as-judge feedback to iteratively fix hallucinations.

        Args:
            system_prompt: System prompt with session context
            user_request: User's request
            thinking_block: Original thinking block
            max_retries: Maximum fix attempts (ignored for local providers)

        Returns:
            Tuple of (is_valid, final_judgment, fixed_thinking_or_None)
            - is_valid: True if fixed or originally valid
            - final_judgment: Last judgment from validator
            - fixed_thinking: Fixed thinking block, or None if couldn't fix
        """
        # Local providers (lmstudio, ollama) can do unlimited retries (no API cost)
        # Cloud providers (openrouter) should have a limit
        is_local_provider = self.backend.lower() in ["lmstudio", "ollama"]
        effective_max_retries = 999 if is_local_provider else max_retries

        if is_local_provider:
            self.logger.info(f"Using local provider ({self.backend}) - unlimited retries enabled")

        current_thinking = thinking_block.copy()

        for attempt in range(effective_max_retries + 1):
            # Validate current version
            is_valid, judgment = self.validate(
                system_prompt, user_request, current_thinking
            )

            if is_valid:
                if attempt > 0:
                    self.logger.success(f"✓ Fixed after {attempt} attempt(s)")
                return True, judgment, current_thinking if attempt > 0 else None

            if attempt < effective_max_retries:
                # Try to fix using judge feedback
                retry_msg = f"Attempt {attempt + 1}"
                if not is_local_provider:
                    retry_msg += f"/{effective_max_retries}"
                retry_msg += ": Fixing hallucinations..."
                self.logger.info(retry_msg)

                try:
                    fixed_thinking = self._fix_hallucinations(
                        system_prompt,
                        user_request,
                        current_thinking,
                        judgment
                    )
                    current_thinking = fixed_thinking
                except Exception as e:
                    self.logger.error(f"Fix attempt failed: {e}")
                    break

        # Could not fix
        self.logger.warning(f"✗ Could not fix after {effective_max_retries} attempts")
        return False, judgment, None

    def _fix_hallucinations(
        self,
        system_prompt: str,
        user_request: str,
        thinking_block: Dict,
        judgment: Dict,
    ) -> Dict:
        """
        Fix hallucinations using judge feedback.

        Args:
            system_prompt: System prompt
            user_request: User request
            thinking_block: Thinking block with hallucinations
            judgment: Judgment with hallucination details

        Returns:
            Fixed thinking block
        """
        # Build fix prompt with judge feedback
        hallucinated_items = "\n".join(
            f"  - {item}" for item in judgment["hallucinated_details"]
        )

        fix_prompt = f"""You are fixing a thinking block that contains hallucinated information.

## Original Inputs

**System Prompt:**
```
{system_prompt}
```

**User Request:**
```
{user_request}
```

## Problematic Thinking Block

```json
{json.dumps(thinking_block, indent=2)}
```

## Hallucinations Detected

The following details were NOT in the inputs and must be removed:
{hallucinated_items}

## Your Task

Generate a FIXED thinking block that:
1. **REMOVES** all hallucinated details listed above
2. **USES ONLY** information from system prompt and user request
3. **STAYS VAGUE** when inputs don't provide specific details
4. **KEEPS** logical inferences (e.g., "delete is risky")

If the user request is vague (e.g., "delete old files"):
- DON'T invent specific paths like "Projects/Archive/Q1-2023/"
- DO say "delete files matching user criteria"

If user provides specifics (e.g., "delete Projects/Archive/Q1-2023/"):
- DO use the exact path provided
- DON'T add numbers, dates, or history not mentioned

Output ONLY the fixed thinking block as valid JSON.
"""

        # Get fixed version from LLM
        response = self.llm_client.chat(
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.3,
        )

        # Parse response
        content = response if isinstance(response, str) else response.get("content", "")

        # Extract JSON from response
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            json_str = content[start:end].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            json_str = content[start:end].strip()
        else:
            json_str = content.strip()

        fixed_thinking = json.loads(json_str)
        return fixed_thinking

    def _extract_thinking(self, assistant_content: str) -> Optional[Dict]:
        """
        Extract thinking block from assistant response.

        Args:
            assistant_content: Full assistant response

        Returns:
            Thinking block as dict, or None if not found
        """
        # Look for <thinking> tags
        if "<thinking>" in assistant_content and "</thinking>" in assistant_content:
            start = assistant_content.index("<thinking>") + len("<thinking>")
            end = assistant_content.index("</thinking>")
            thinking_str = assistant_content[start:end].strip()

            try:
                return json.loads(thinking_str)
            except json.JSONDecodeError:
                return None

        return None


def main():
    """CLI for testing hallucination validator."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate thinking blocks for hallucination")
    parser.add_argument("--file", type=str, help="JSONL file to validate")
    parser.add_argument("--line", type=int, help="Specific line to validate")
    parser.add_argument("--backend", default="lmstudio", help="LLM backend")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--output", type=str, help="Output file for invalid examples (JSONL)")

    args = parser.parse_args()

    # Initialize validator
    validator = HallucinationValidator(
        backend=args.backend,
        model=args.model,
    )

    if args.file:
        # Validate file
        with open(args.file, 'r', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f if line.strip()]

        if args.line:
            examples = [examples[args.line - 1]]

        print(f"Validating {len(examples)} examples...")
        results = validator.validate_batch(examples)

        # Print summary
        valid_count = sum(1 for r in results if r.get("is_valid", False))
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(results)}")
        print(f"Valid: {valid_count}")
        print(f"Invalid: {len(results) - valid_count}")

        # Save invalid examples if output file specified
        if args.output:
            invalid_examples = []
            for r in results:
                if not r.get("is_valid", False):
                    # Get the original example
                    original = examples[r['index']]
                    # Add judgment details
                    original['validation_result'] = r
                    invalid_examples.append(original)

            with open(args.output, 'w', encoding='utf-8') as f:
                for example in invalid_examples:
                    f.write(json.dumps(example) + '\n')

            print(f"\n✓ Saved {len(invalid_examples)} invalid examples to {args.output}")

        # Print details for invalid
        print(f"\n{'='*60}")
        print("INVALID EXAMPLES")
        print(f"{'='*60}")
        for r in results:
            if not r.get("is_valid", False):
                print(f"\nLine {r['index'] + 1}:")
                if "judgment" in r:
                    j = r["judgment"]
                    print(f"  Confidence: {j['confidence']:.2f}")
                    print(f"  Explanation: {j['explanation']}")
                    print(f"  Hallucinated details:")
                    for detail in j["hallucinated_details"]:
                        print(f"    - {detail}")
                else:
                    print(f"  Reason: {r.get('reason', 'Unknown')}")


if __name__ == "__main__":
    main()
