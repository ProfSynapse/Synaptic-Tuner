#!/usr/bin/env python3
"""
Single-Turn Destructive Action Generator

Generates training examples where:
1. System prompt has fake workspace with specific files/folders
2. User explicitly gives permission: "I give you permission to delete X"
3. Model thinks: HIGH RISK + HIGH CONFIDENCE (permission given)
4. Model executes with correct tool schema

Pattern: Explicit Permission → High Confidence (despite high risk)
"""

import json
import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Handle both package and standalone imports
try:
    from .id_utils import generate_ids
except ImportError:
    from id_utils import generate_ids


class SingleTurnDestructiveGenerator:
    """
    Generator for single-turn destructive action examples with explicit permission.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        tool_schemas_path: Optional[Path] = None,
    ):
        """
        Initialize the generator.

        Args:
            config_path: Path to destructive_single_turn.yaml
            tool_schemas_path: Path to Tools/tool_schemas.json
        """
        self.config_path = config_path or (
            Path(__file__).parent / "configs" / "destructive_single_turn.yaml"
        )
        self.tool_schemas_path = tool_schemas_path or (
            Path(__file__).parent.parent / "Tools" / "tool_schemas.json"
        )

        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        # Load tool schemas
        with open(self.tool_schemas_path) as f:
            self.tool_schemas = json.load(f)

    def select_tool(self) -> Tuple[str, str, Dict]:
        """
        Select a destructive tool based on weights.

        Returns:
            Tuple of (agent_name, tool_key, tool_config)
        """
        weights = self.config["generation_flow"]["step_1_select_tool"]["weights"]
        tool_full_name = random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1
        )[0]

        # Parse tool name
        agent_name, tool_key = tool_full_name.split("_", 1)

        # Find tool config
        tool_config = self.config["destructive_tools"][agent_name][tool_key]

        return agent_name, tool_key, tool_config

    def generate_system_prompt(
        self,
        agent_name: str,
        tool_config: Dict,
        session_id: str,
        workspace_id: str,
    ) -> Tuple[str, Dict]:
        """
        Generate minimal system prompt (matching real datasets).

        Args:
            agent_name: Name of agent (vaultManager, contentManager, etc.)
            tool_config: Tool configuration
            session_id: Generated session ID
            workspace_id: Generated workspace ID

        Returns:
            Tuple of (system_prompt, selected_scenario)
        """
        # Select scenario (for generating user request, not system prompt)
        scenario = random.choice(tool_config["scenarios"])

        # Build minimal system prompt (just IDs, like real datasets)
        template = self.config["system_prompt_template"]
        system_prompt = template.format(
            session_id=session_id,
            workspace_id=workspace_id,
        )

        return system_prompt, scenario

    def generate_user_request(
        self,
        tool_config: Dict,
        scenario: Dict,
    ) -> Tuple[str, Any]:
        """
        Generate user request with explicit permission.

        Args:
            tool_config: Tool configuration
            scenario: Selected scenario

        Returns:
            Tuple of (user_request, selected_target)
        """
        # Select permission pattern
        patterns = self.config["permission_patterns"]
        pattern_names = list(patterns.keys())
        pattern_weights = [patterns[name]["weight"] for name in pattern_names]

        pattern_name = random.choices(pattern_names, weights=pattern_weights, k=1)[0]
        pattern = patterns[pattern_name]

        # Select template
        template = random.choice(pattern["templates"])

        # Select target from scenario
        target = random.choice(scenario["examples"])

        # Format user request
        if isinstance(target, dict):
            # Complex target (e.g., agent with id and name)
            if "id" in target:
                target_str = f"the '{target['name']}' agent"
                target_value = target
            elif "content" in target:
                target_str = target["description"]
                target_value = target
            else:
                target_str = str(target)
                target_value = target
        else:
            # Simple string target
            target_str = target
            target_value = target

        user_request = template.format(target=target_str)

        return user_request, target_value

    def generate_thinking_block(
        self,
        tool_config: Dict,
        target: Any,
        user_request: str,
    ) -> str:
        """
        Generate thinking block with HIGH RISK + HIGH CONFIDENCE.

        Args:
            tool_config: Tool configuration
            target: Selected target
            user_request: User's request

        Returns:
            str: Formatted thinking block
        """
        risk_level = tool_config["risk_level"]
        risk_desc = self.config["risk_descriptions"][risk_level]

        # Determine target description
        if isinstance(target, dict):
            if "id" in target:
                target_desc = f"Agent '{target['name']}' ({target['id']})"
                specific_target = target["id"]
                target_location = "Agent registry"
                item_count = "1 agent"
            elif "content" in target:
                target_desc = target["description"]
                specific_target = target["content"][:50] + "..."
                target_location = "Note content"
                item_count = f"{len(target['content'])} characters"
            else:
                target_desc = str(target)
                specific_target = str(target)
                target_location = "Workspace"
                item_count = "1 item"
        else:
            target_desc = target
            specific_target = target
            target_location = "Vault"

            # Estimate item count
            if "/" in str(target) and target.endswith("/"):
                item_count = "multiple files/folders"
            else:
                item_count = "1 item"

        # Fill thinking template
        template = self.config["thinking_template"]

        # Determine confidence level
        if "permission" in user_request.lower():
            confidence = self.config["confidence_levels"]["with_explicit_permission"]
        elif "yes" in user_request.lower() or "confirmed" in user_request.lower():
            confidence = self.config["confidence_levels"]["with_confirmation"]
        else:
            confidence = self.config["confidence_levels"]["with_specific_instruction"]

        thinking = template.format(
            target_description=target_desc,
            specific_target=specific_target,
            operation_type=tool_config["tool_name"],
            risk_level=risk_level,
            impact_description=risk_desc["impact"],
            additional_risks=risk_desc["additional_risks"].format(item_count=item_count),
            user_permission_phrase=user_request,
            target_location=target_location,
            tool_name=tool_config["tool_name"],
            additional_requirements="✓ User understands irreversibility",
            confidence_level=confidence,
            additional_steps="",
        )

        return thinking

    def generate_tool_call(
        self,
        tool_config: Dict,
        target: Any,
        session_id: str,
        workspace_id: str,
    ) -> str:
        """
        Generate tool call with correct schema.

        Args:
            tool_config: Tool configuration
            target: Selected target
            session_id: Session ID
            workspace_id: Workspace ID

        Returns:
            str: Formatted tool call
        """
        tool_name = tool_config["tool_name"]
        schema = self.tool_schemas.get(tool_name, {})

        # Build context object
        context = {
            "sessionId": session_id,
            "workspaceId": workspace_id,
            "sessionDescription": "User authorized workspace cleanup",
            "sessionMemory": f"User explicitly authorized deletion of {target}",
            "toolContext": f"Executing authorized {tool_name}",
            "primaryGoal": "Workspace organization and cleanup",
            "subgoal": f"Delete {target} as authorized"
        }

        # Build arguments based on tool type
        if "deleteFolder" in tool_name or "deleteNote" in tool_name:
            # Path-based deletion
            arguments = {
                "context": context,
                "path": str(target) if isinstance(target, str) else target.get("path", "unknown"),
            }
            if "deleteFolder" in tool_name:
                arguments["recursive"] = True

        elif "deleteAgent" in tool_name:
            # Agent deletion
            arguments = {
                "context": context,
                "id": target["id"] if isinstance(target, dict) else "unknown_id",
            }

        elif "deleteContent" in tool_name:
            # Content deletion
            arguments = {
                "context": context,
                "filePath": "Notes/Active-Note.md",  # Example path
                "content": target["content"] if isinstance(target, dict) else str(target),
            }

        else:
            # Fallback
            arguments = {"context": context}

        # Format tool call
        tool_call = {
            "name": tool_name,
            "arguments": arguments
        }

        return f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>"

    def generate_example(self) -> Dict[str, Any]:
        """
        Generate a complete single-turn destructive action example.

        Returns:
            Dict: Example in ChatML format
        """
        # Generate IDs
        session_id, workspace_id = generate_ids()

        # Step 1: Select tool
        agent_name, tool_key, tool_config = self.select_tool()

        # Step 2: Generate system prompt
        system_prompt, scenario = self.generate_system_prompt(
            agent_name, tool_config, session_id, workspace_id
        )

        # Step 3: Generate user request
        user_request, target = self.generate_user_request(tool_config, scenario)

        # Step 4: Generate response
        thinking = self.generate_thinking_block(tool_config, target, user_request)
        tool_call = self.generate_tool_call(tool_config, target, session_id, workspace_id)

        completion_msg = f"Successfully deleted {target} as authorized. Operation completed."

        assistant_response = f"{thinking}\n\n{tool_call}\n\n{completion_msg}"

        # Build example
        example = {
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "generation_type": "destructive_single_turn",
                "tool_name": tool_config["tool_name"],
                "agent_name": agent_name,
                "risk_level": tool_config["risk_level"],
            }
        }

        return example

    def generate_batch(
        self,
        count: int,
        output_path: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Generate multiple examples.

        Args:
            count: Number of examples to generate
            output_path: Optional output path

        Returns:
            List of examples
        """
        examples = []

        print(f"Generating {count} single-turn destructive action examples...")

        for i in range(count):
            try:
                example = self.generate_example()
                examples.append(example)

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{count}...")

            except Exception as e:
                print(f"  Error generating example {i + 1}: {e}")
                continue

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            print(f"\nSaved {len(examples)} examples to {output_path}")

        return examples


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate single-turn destructive action examples")
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of examples")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file")
    parser.add_argument("--test", action="store_true", help="Test mode (3 examples)")

    args = parser.parse_args()

    if args.test:
        args.count = 3

    generator = SingleTurnDestructiveGenerator()
    examples = generator.generate_batch(args.count, args.output)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total: {len(examples)}")

    tool_counts = {}
    for ex in examples:
        tool = ex["metadata"]["tool_name"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\nBy tool:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tool}: {count}")

    if args.test:
        print("\n" + "="*80)
        print("SAMPLE OUTPUT")
        print("="*80)
        if examples:
            print("\n--- SYSTEM ---")
            print(examples[0]["conversations"][0]["content"][:300] + "...")
            print("\n--- USER ---")
            print(examples[0]["conversations"][1]["content"])
            print("\n--- ASSISTANT ---")
            print(examples[0]["conversations"][2]["content"][:500] + "...")


if __name__ == "__main__":
    main()
