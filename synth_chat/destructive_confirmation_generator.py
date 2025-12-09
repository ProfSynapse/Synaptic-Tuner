"""
Destructive Confirmation Generator

Generates multi-turn training examples showing proper handling of destructive operations:
1. User makes vague destructive request
2. Assistant asks for confirmation with specifics
3. User explicitly confirms
4. Assistant executes with HIGH RISK + HIGH CONFIDENCE thinking

This teaches the model that:
- Never execute destructive operations without confirmation
- High risk + user confirmation = High confidence (correct pattern)
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


class DestructiveConfirmationGenerator:
    """
    Generator for multi-turn destructive confirmation examples.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        model_client=None,
        tool_schemas: Optional[Dict] = None,
    ):
        """
        Initialize the generator.

        Args:
            config_path: Path to destructive_confirmations.yaml
            model_client: LLM client for generation (e.g., LMStudioClient)
            tool_schemas: Tool schema definitions (from Tools/tool_schemas.json)
        """
        self.config_path = config_path or (
            Path(__file__).parent / "configs" / "destructive_confirmations.yaml"
        )
        self.model_client = model_client
        self.tool_schemas = tool_schemas or {}

        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def select_destructive_tool(self) -> Tuple[str, str, Dict]:
        """
        Select a destructive tool based on weights.

        Returns:
            Tuple of (agent_name, tool_name, tool_config)
        """
        weights = self.config["generation_weights"]
        tool_name = random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1
        )[0]

        # Extract agent name from tool_name
        agent_name = tool_name.split("_")[0]

        # Find tool config
        for agent, tools in self.config["destructive_tools"].items():
            for tool_config in tools:
                if tool_config["tool"] == tool_name:
                    return agent_name, tool_name, tool_config

        raise ValueError(f"Tool {tool_name} not found in config")

    def generate_step_1_user_request(
        self,
        tool_name: str,
        tool_config: Dict,
        session_context: str,
    ) -> str:
        """
        Generate Step 1: Vague user request for destructive operation.

        Args:
            tool_name: Name of the destructive tool
            tool_config: Tool configuration
            session_context: Session context with workspace info

        Returns:
            str: User request
        """
        scenario = random.choice(tool_config["scenarios"])

        template = self.config["conversation_flow"]["step_1_user_request"]["template"]
        prompt = template.format(
            tool_name=tool_name,
            scenario=scenario
        )

        # Add session context for realism
        full_prompt = f"{session_context}\n\n{prompt}"

        if self.model_client:
            self.model_client.settings.temperature = 0.8
            self.model_client.settings.max_tokens = 100

            messages = [{"role": "user", "content": full_prompt}]
            response = self.model_client.chat(messages)
            return response.message.strip()

        # Fallback
        return f"Can you delete the old {scenario.lower()}?"

    def generate_step_2_assistant_asks(
        self,
        user_request: str,
        tool_name: str,
        session_context: str,
    ) -> str:
        """
        Generate Step 2: Assistant asks for confirmation.

        The assistant should:
        - List specific items that would be affected
        - Warn about irreversibility
        - Request explicit confirmation

        Args:
            user_request: User's destructive request
            tool_name: Tool that would be used
            session_context: Session context

        Returns:
            str: Assistant's confirmation request
        """
        instructions = self.config["conversation_flow"]["step_2_assistant_asks"]["instructions"]

        prompt = f"""{session_context}

The user requested: "{user_request}"

You need to respond by:
1. Listing the specific items that would be deleted/affected
2. Warning that this is permanent and irreversible
3. Asking for explicit confirmation

{instructions}

Generate the assistant's response asking for confirmation. Include realistic file/folder names
based on the session context. Output ONLY the assistant's response."""

        if self.model_client:
            self.model_client.settings.temperature = 0.6
            self.model_client.settings.max_tokens = 300

            messages = [{"role": "user", "content": prompt}]
            response = self.model_client.chat(messages)
            return response.message.strip()

        # Fallback
        return """I found 3 folders that match "old project stuff":
- Projects/Archive/Q1-2023/
- Projects/Archive/Q2-2023/
- Projects/Completed/Legacy-Project/

These folders contain 47 files total. This action will permanently delete all files and cannot be undone.

Would you like me to proceed with the deletion?"""

    def generate_step_3_user_confirms(self) -> str:
        """
        Generate Step 3: User explicitly confirms.

        Returns:
            str: User confirmation message
        """
        variations = self.config["conversation_flow"]["step_3_user_confirms"]["variations"]
        return random.choice(variations)

    def generate_step_4_assistant_executes(
        self,
        user_request: str,
        assistant_list: str,
        user_confirmation: str,
        tool_name: str,
        session_context: str,
    ) -> str:
        """
        Generate Step 4: Assistant executes with HIGH RISK + HIGH CONFIDENCE.

        This is the key training signal:
        - Thinking block shows HIGH RISK (permanent deletion)
        - Thinking block shows HIGH CONFIDENCE (user confirmed)
        - Tool call is executed correctly

        Args:
            user_request: Original user request
            assistant_list: Assistant's list of items to delete
            user_confirmation: User's confirmation message
            tool_name: Tool to execute
            session_context: Session context with IDs

        Returns:
            str: Assistant's execution response with thinking + tool call
        """
        thinking_pattern = self.config["conversation_flow"]["step_4_assistant_executes"]["thinking_requirements"]["thinking_pattern"]

        prompt = f"""{session_context}

Conversation history:
1. User: "{user_request}"
2. Assistant: {assistant_list}
3. User: "{user_confirmation}"

You are now executing the destructive operation.

CRITICAL REQUIREMENTS:
1. Include <thinking> block with:
   - Risk Assessment: HIGH (permanent deletion, irreversible)
   - Confirmation Verification: User explicitly confirmed
   - Confidence: 9/10 or 10/10 (HIGH confidence due to confirmation)

2. Include proper <tool_call> using: {tool_name}
   - Must include complete context object with sessionId and workspaceId
   - Must target specific items from the list

3. Explain completion after tool calls

{thinking_pattern}

Generate the assistant's execution response with <thinking> and <tool_call> blocks.
Output ONLY the assistant's response."""

        if self.model_client:
            self.model_client.settings.temperature = 0.5  # Lower for tool precision
            self.model_client.settings.max_tokens = 800

            messages = [{"role": "user", "content": prompt}]
            response = self.model_client.chat(messages)
            return response.message.strip()

        # Fallback with proper format
        return """<thinking>
Risk Assessment:
- Operation: Bulk folder deletion
- Risk Level: HIGH (permanent deletion of 47 files)
- Affected Items: 3 folders in Archive/Completed
- Scope: Clear and specific

Confirmation Verification:
- User explicitly confirmed: "Yes, delete them"
- User reviewed the list (3 folders, 47 files)
- User understands this is permanent

Confidence: 9/10
- HIGH confidence to proceed
- User gave explicit permission
- Scope was clear and confirmed
- All paths verified

Proceeding with deletions as confirmed.
</thinking>

<tool_call>
{
  "name": "vaultManager_deleteFolder",
  "arguments": {
    "context": {
      "sessionId": "session_1732300800000_a1b2c3d4e",
      "workspaceId": "ws_1732300800000_f5g6h7i8j",
      "sessionDescription": "Cleaning up old project archives",
      "sessionMemory": "User confirmed deletion of 3 archive folders",
      "toolContext": "Executing confirmed bulk deletion",
      "primaryGoal": "Clean up workspace",
      "subgoal": "Delete archived project folders"
    },
    "path": "Projects/Archive/Q1-2023",
    "recursive": true
  }
}
</tool_call>

Successfully deleted all 3 folders as confirmed. The operation completed successfully."""

    def generate_full_conversation(
        self,
        session_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete 4-turn destructive confirmation conversation.

        Returns:
            Dict: ChatML format conversation with metadata
        """
        # Select tool
        agent_name, tool_name, tool_config = self.select_destructive_tool()

        # Generate session context if not provided
        if not session_context:
            session_id, workspace_id = generate_ids()
            session_context = f"""<session_context>
sessionId: {session_id}
workspaceId: {workspace_id}
sessionDescription: User is organizing and cleaning up their vault
sessionMemory: User is working on workspace cleanup tasks
</session_context>"""

        # Step 1: User makes vague destructive request
        user_request = self.generate_step_1_user_request(
            tool_name, tool_config, session_context
        )

        # Step 2: Assistant asks for confirmation
        assistant_asks = self.generate_step_2_assistant_asks(
            user_request, tool_name, session_context
        )

        # Step 3: User confirms
        user_confirms = self.generate_step_3_user_confirms()

        # Step 4: Assistant executes
        assistant_executes = self.generate_step_4_assistant_executes(
            user_request, assistant_asks, user_confirms, tool_name, session_context
        )

        # Build conversation
        conversation = {
            "conversations": [
                {"role": "user", "content": user_request},
                {"role": "assistant", "content": assistant_asks},
                {"role": "user", "content": user_confirms},
                {"role": "assistant", "content": assistant_executes},
            ],
            "metadata": {
                "generation_type": "destructive_confirmation",
                "tool_name": tool_name,
                "agent_name": agent_name,
                "risk_level": tool_config["risk_level"],
                "turn_count": 4,
            }
        }

        return conversation

    def generate_batch(
        self,
        count: int,
        output_path: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Generate multiple destructive confirmation examples.

        Args:
            count: Number of examples to generate
            output_path: Optional path to save JSONL output

        Returns:
            List of generated conversations
        """
        examples = []

        print(f"Generating {count} destructive confirmation examples...")

        for i in range(count):
            try:
                example = self.generate_full_conversation()
                examples.append(example)

                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{count} examples...")

            except Exception as e:
                print(f"Error generating example {i + 1}: {e}")
                continue

        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            print(f"\nSaved {len(examples)} examples to {output_path}")

        return examples


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate destructive confirmation training examples"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to destructive_confirmations.yaml"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = DestructiveConfirmationGenerator(
        config_path=args.config,
        model_client=None,  # TODO: Add LM Studio client support
    )

    # Generate examples
    examples = generator.generate_batch(
        count=args.count,
        output_path=args.output,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")

    # Count by tool
    tool_counts = {}
    for ex in examples:
        tool = ex["metadata"]["tool_name"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\nBy tool:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
