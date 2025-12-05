"""
SelfPlay Generator - 3-Prompt Pipeline

Generates synthetic training data through a three-prompt pipeline:
1. Generate workspace environment
2. Generate user request
3. Generate assistant response

Supports both tool-based and behavioral generation modes.
"""

import os
import sys
import json
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add tools directory to path for validation
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from .id_utils import generate_ids

try:
    from validate_syngen import validate_example
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: validate_syngen not available, validation disabled")


class SelfPlayGenerator:
    """
    Main generator class for the 3-prompt pipeline.
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        model_client=None,  # LM Studio client or similar
        randomize_params: bool = True,
    ):
        """
        Initialize the generator.

        Args:
            config_dir: Path to configs directory (defaults to ./configs)
            model_client: Client for LLM inference (e.g., LMStudioClient)
            randomize_params: Whether to randomize LLM parameters for diversity
        """
        self.config_dir = config_dir or Path(__file__).parent / "configs"
        self.model_client = model_client
        self.randomize_params = randomize_params

        # Load all configs
        self.load_configs()

    def _randomize_llm_params(self, base_temp: float = 0.7):
        """
        Randomize LLM parameters for diverse generation.

        Args:
            base_temp: Base temperature to randomize around
        """
        if not self.model_client or not self.randomize_params:
            return

        # Temperature: ±0.2 from base
        self.model_client.settings.temperature = base_temp + random.uniform(-0.2, 0.2)

        # Top P: 0.85-0.95 (high quality sampling)
        self.model_client.settings.top_p = random.uniform(0.85, 0.95)

        # Presence penalty: -0.2 to 0.2 (slight variation)
        self.model_client.settings.presence_penalty = random.uniform(-0.2, 0.2)

        # Frequency penalty: 0.0 to 0.3 (encourage diversity)
        self.model_client.settings.frequency_penalty = random.uniform(0.0, 0.3)

    def load_configs(self):
        """Load all YAML configuration files."""
        # Load prompt templates
        prompts_dir = self.config_dir / "prompts"
        with open(prompts_dir / "environment.yaml") as f:
            self.environment_config = yaml.safe_load(f)

        with open(prompts_dir / "user_tool.yaml") as f:
            self.user_tool_config = yaml.safe_load(f)

        with open(prompts_dir / "user_behavioral.yaml") as f:
            self.user_behavioral_config = yaml.safe_load(f)

        # Load agent and behavior configs
        with open(self.config_dir / "agents.yaml") as f:
            self.agents_config = yaml.safe_load(f)

        with open(self.config_dir / "behaviors.yaml") as f:
            self.behaviors_config = yaml.safe_load(f)

    def select_generation_type(self) -> str:
        """
        Select whether to do tool-based or behavioral generation.

        Returns:
            str: "tool" or "behavioral"
        """
        # 70% tool-based, 30% behavioral
        return random.choices(["tool", "behavioral"], weights=[0.7, 0.3], k=1)[0]

    def select_environment_variation(self) -> Dict[str, Any]:
        """
        Select an environment variation based on weights.

        Returns:
            Dict: Selected variation config
        """
        variations = self.environment_config["variations"]
        names = list(variations.keys())
        weights = [variations[name]["weight"] for name in names]

        selected_name = random.choices(names, weights=weights, k=1)[0]
        return {
            "name": selected_name,
            "config": variations[selected_name]
        }

    def prompt_1_generate_environment(self, variation: Dict[str, Any]) -> str:
        """
        Prompt 1: Generate workspace environment.

        Args:
            variation: Environment variation config

        Returns:
            str: Generated workspace environment (from LLM)
        """
        prompt = variation["config"]["prompt"]

        if self.model_client:
            # Randomize parameters for diversity
            self._randomize_llm_params(base_temp=0.9)
            self.model_client.settings.max_tokens = 500

            messages = [{"role": "user", "content": prompt}]
            response = self.model_client.chat(messages)
            return response.message

        # Placeholder response if no client
        return """Workspace Name: Project Atlas
Description: Customer rollout tracking for the Atlas product launch
Root Folder: Projects/Atlas/

Folder Structure:
Projects/Atlas/
├── Planning/
│   ├── Roadmap-Q4.md
│   ├── Customer-List.md
│   └── Risk-Assessment.md
├── Active/
│   ├── Sprint-12-Tasks.md
│   ├── Blockers.md
│   └── Weekly-Status.md
├── Meetings/
│   ├── Standup-2024-01-15.md
│   ├── Planning-Session.md
│   └── Retrospective-Dec.md
└── README.md"""

    def build_session_context(
        self,
        workspace_environment: str,
        session_id: str,
        workspace_id: str
    ) -> str:
        """
        Build the complete session context by injecting IDs into workspace environment.

        Args:
            workspace_environment: Generated workspace description
            session_id: Auto-generated session ID
            workspace_id: Auto-generated workspace ID

        Returns:
            str: Complete session context
        """
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>

<current_workspace>
{workspace_environment}
</current_workspace>"""

        return session_context

    def prompt_2_generate_user_request_tool(
        self,
        session_context: str,
        agent_name: str,
        tools: List[str]
    ) -> str:
        """
        Prompt 2: Generate user request (tool-based).

        Args:
            session_context: Complete session context from Prompt 1
            agent_name: Agent name (e.g., "vaultManager")
            tools: List of tools for this agent

        Returns:
            str: Generated user request
        """
        # Select single tool for single-tool scenarios
        tool = random.choice(tools)

        # Get base template
        base_template = self.user_tool_config["base_template"]

        # Fill in template
        prompt = base_template.format(
            session_context=session_context,
            agent_name=agent_name,
            tool_list=tool
        )

        if self.model_client:
            # Randomize parameters for diversity
            self._randomize_llm_params(base_temp=0.7)
            self.model_client.settings.max_tokens = 150

            messages = [{"role": "user", "content": prompt}]
            response = self.model_client.chat(messages)
            return response.message

        # Placeholder response
        return "Can you create a folder for Q4 planing stuff? I think in the project area"

    def prompt_2_generate_user_request_behavioral(
        self,
        session_context: str,
        behavior_name: str,
        behavior_config: Dict[str, Any]
    ) -> str:
        """
        Prompt 2: Generate user request (behavioral).

        Args:
            session_context: Complete session context from Prompt 1
            behavior_name: Behavior name (e.g., "intellectual_humility")
            behavior_config: Behavior configuration

        Returns:
            str: Generated user request
        """
        # Get base template
        base_template = self.user_behavioral_config["base_template"]

        # Fill in template
        prompt = base_template.format(
            session_context=session_context,
            behavior_name=behavior_name,
            behavior_description=behavior_config["description"],
            behavior_specific_instructions=behavior_config["instructions"]
        )

        if self.model_client:
            # Randomize parameters for diversity
            self._randomize_llm_params(base_temp=0.7)
            self.model_client.settings.max_tokens = 150

            messages = [{"role": "user", "content": prompt}]
            response = self.model_client.chat(messages)
            return response.message

        # Placeholder response
        return "Can you delete the old project files?"

    def prompt_3_generate_response(
        self,
        session_context: str,
        user_request: str
    ) -> str:
        """
        Prompt 3: Generate assistant response.

        IMPORTANT: Include session context as system message!
        This gives the model workspace awareness.

        Args:
            session_context: Session context with workspace structure
            user_request: User request from Prompt 2

        Returns:
            str: Generated assistant response with tool calls
        """
        if self.model_client:
            # Randomize parameters for diversity (lower temp for tool precision)
            self._randomize_llm_params(base_temp=0.5)
            self.model_client.settings.max_tokens = 500

            # INCLUDE session context as system prompt!
            # This gives the model the workspace structure and context
            messages = [
                {"role": "system", "content": session_context},
                {"role": "user", "content": user_request}
            ]
            response = self.model_client.chat(messages)
            return response.message

        # Placeholder response (in Qwen format)
        return """<tool_call>
{
  "name": "vaultManager_createFolder",
  "arguments": {
    "context": {
      "sessionId": "session_1732300800000_a1b2c3d4e",
      "workspaceId": "ws_1732300800000_f5g6h7i8j",
      "sessionDescription": "Working on Project Atlas organization",
      "sessionMemory": "User is organizing Atlas project workspace",
      "toolContext": "Creating folder for Q4 planning materials",
      "primaryGoal": "Organize project structure",
      "subgoal": "Create Q4 planning folder"
    },
    "path": "Projects/Atlas/Planning/Q4-Planning"
  }
}
</tool_call>"""

    def generate_single_example(
        self,
        generation_type: Optional[str] = None,
        specific_tool: Optional[str] = None,
        specific_behavior: Optional[str] = None,
        specific_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a single training example using the 3-prompt pipeline.

        Args:
            generation_type: "tool" or "behavioral" (auto-selected if None)
            specific_tool: Generate example for specific tool (e.g., "vaultManager_createFolder")
            specific_behavior: Generate example for specific behavior (e.g., "intellectual_humility")
            specific_agent: Generate example for specific agent (e.g., "vaultManager")

        Returns:
            Dict: Generated example with metadata
        """
        # Determine generation type from specific parameters
        if specific_tool is not None:
            generation_type = "tool"
        elif specific_behavior is not None:
            generation_type = "behavioral"
        elif generation_type is None:
            generation_type = self.select_generation_type()

        # Generate IDs
        session_id, workspace_id = generate_ids()

        # PROMPT 1: Generate environment
        variation = self.select_environment_variation()
        workspace_environment = self.prompt_1_generate_environment(variation)

        # Build session context
        session_context = self.build_session_context(
            workspace_environment,
            session_id,
            workspace_id
        )

        # PROMPT 2: Generate user request
        if generation_type == "tool":
            # Determine agent and tool
            if specific_tool:
                # Find agent for specific tool
                agent_name = None
                for a_name, a_config in self.agents_config["agents"].items():
                    if specific_tool in a_config["tools"]:
                        agent_name = a_name
                        break
                if not agent_name:
                    raise ValueError(f"Tool {specific_tool} not found in any agent")
                selected_tool = specific_tool
            elif specific_agent:
                # Use specific agent, random tool
                if specific_agent not in self.agents_config["agents"]:
                    raise ValueError(f"Agent {specific_agent} not found")
                agent_name = specific_agent
                agent_config = self.agents_config["agents"][agent_name]
                selected_tool = random.choice(agent_config["tools"])
            else:
                # Random agent and tool
                agent_name = random.choice(list(self.agents_config["agents"].keys()))
                agent_config = self.agents_config["agents"][agent_name]
                tools = agent_config["tools"]
                selected_tool = random.choice(tools)

            # Generate user request for this specific tool
            user_request = self.prompt_2_generate_user_request_tool(
                session_context,
                agent_name,
                [selected_tool]  # Pass only the selected tool
            )

            metadata = {
                "type": "tool",
                "agent": agent_name,
                "tool": selected_tool,  # Track specific tool
                "category": selected_tool,  # Category for dataset splitting
            }

        else:  # behavioral
            # Determine behavior
            behaviors = self.behaviors_config["behaviors"]
            if specific_behavior:
                if specific_behavior not in behaviors:
                    raise ValueError(f"Behavior {specific_behavior} not found")
                behavior_name = specific_behavior
            else:
                behavior_name = random.choice(list(behaviors.keys()))

            behavior_config = behaviors[behavior_name]

            user_request = self.prompt_2_generate_user_request_behavioral(
                session_context,
                behavior_name,
                behavior_config
            )

            metadata = {
                "type": "behavioral",
                "behavior": behavior_name,
                "category": behavior_name,  # Category for dataset splitting
            }

        # PROMPT 3: Generate response
        assistant_response = self.prompt_3_generate_response(
            session_context,
            user_request
        )

        # Build final example
        # ALWAYS include system message with session context
        example = {
            "conversations": [
                {"role": "system", "content": session_context},
                {"role": "user", "content": user_request},
                {"role": "assistant", "content": assistant_response}
            ],
            "metadata": {
                "session_id": session_id,
                "workspace_id": workspace_id,
                "environment_variation": variation["name"],
                **metadata
            }
        }

        # For behavioral examples, add behavior field at top level
        if generation_type == "behavioral":
            example["behavior"] = metadata["behavior"]

        return example

    def validate_example(self, example: Dict[str, Any], index: int = 0) -> Tuple[bool, Optional[Any]]:
        """
        Validate a generated example.

        Args:
            example: Generated example to validate
            index: Index for error reporting

        Returns:
            Tuple[bool, Optional[report]]: (is_valid, validation_report)
        """
        if not VALIDATION_AVAILABLE:
            return True, None

        try:
            report = validate_example(index, example)
            return report.is_valid, report
        except Exception as e:
            print(f"Validation error: {e}")
            return False, None

    def generate_targeted_batch(
        self,
        targets: Dict[str, int],
        output_file: Optional[Path] = None,
        validate: bool = True,
        save_invalid: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a batch with specific targets for tools/behaviors/agents.

        Args:
            targets: Dict mapping category to count
                     e.g., {"vaultManager_createFolder": 50, "intellectual_humility": 25}
            output_file: Optional file to save all examples as JSONL
            validate: Whether to validate examples
            save_invalid: Whether to save invalid examples separately

        Returns:
            Dict with 'valid' and 'invalid' lists
        """
        valid_examples = []
        invalid_examples = []
        total = sum(targets.values())
        current = 0

        for category, count in targets.items():
            print(f"\n{'='*70}")
            print(f"Generating {count} examples for: {category}")
            print(f"{'='*70}")

            # Determine if category is tool, behavior, or agent
            is_tool = any(category in agent["tools"]
                         for agent in self.agents_config["agents"].values())
            is_behavior = category in self.behaviors_config["behaviors"]
            is_agent = category in self.agents_config["agents"]

            for i in range(count):
                current += 1
                print(f"\nExample {current}/{total} ({category} {i+1}/{count})...")

                # Generate with specific target
                if is_tool:
                    example = self.generate_single_example(specific_tool=category)
                elif is_behavior:
                    example = self.generate_single_example(specific_behavior=category)
                elif is_agent:
                    example = self.generate_single_example(specific_agent=category)
                else:
                    print(f"  ⚠️  Unknown category: {category}, skipping")
                    continue

                # Validate if requested
                is_valid = True
                if validate and VALIDATION_AVAILABLE:
                    is_valid, report = self.validate_example(example, current-1)

                    if is_valid:
                        print("  ✅ Valid")
                        example["label"] = True
                        valid_examples.append(example)
                    else:
                        print("  ❌ Invalid")
                        example["label"] = False
                        invalid_examples.append(example)

                        # Show validation issues
                        if report and report.issues:
                            for issue in report.issues[:2]:
                                icon = "❌" if issue.level == "ERROR" else "⚠️ "
                                print(f"    {icon} {issue.message}")
                else:
                    # No validation - assume valid
                    example["label"] = True
                    valid_examples.append(example)

                # Save incrementally if output file specified
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(example) + '\n')

        print(f"\n{'='*70}")
        print(f"📊 Summary: {len(valid_examples)} valid, {len(invalid_examples)} invalid")
        print(f"{'='*70}")

        return {
            "valid": valid_examples,
            "invalid": invalid_examples
        }

    def generate_batch(
        self,
        num_examples: int,
        output_file: Optional[Path] = None,
        validate: bool = True,
        save_invalid: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a batch of random examples with validation.

        Args:
            num_examples: Number of examples to generate
            output_file: Optional file to save valid examples as JSONL
            validate: Whether to validate examples
            save_invalid: Whether to save invalid examples separately

        Returns:
            Dict with 'valid' and 'invalid' lists
        """
        valid_examples = []
        invalid_examples = []

        for i in range(num_examples):
            print(f"\nGenerating example {i+1}/{num_examples}...")
            example = self.generate_single_example()

            # Validate if requested
            is_valid = True
            if validate and VALIDATION_AVAILABLE:
                is_valid, report = self.validate_example(example, i)

                if is_valid:
                    print("  ✅ Valid")
                    example["label"] = True
                    valid_examples.append(example)
                else:
                    print("  ❌ Invalid")
                    example["label"] = False
                    invalid_examples.append(example)

                    # Show validation issues
                    if report and report.issues:
                        for issue in report.issues[:3]:  # Show first 3 issues
                            icon = "❌" if issue.level == "ERROR" else "⚠️ "
                            print(f"    {icon} {issue.message}")
            else:
                # No validation - assume valid
                example["label"] = True
                valid_examples.append(example)

            # Save incrementally if output file specified
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(example) + '\n')

        print(f"\n📊 Summary: {len(valid_examples)} valid, {len(invalid_examples)} invalid")

        return {
            "valid": valid_examples,
            "invalid": invalid_examples
        }


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = SelfPlayGenerator()

    # Generate single example
    print("Generating single example...")
    example = generator.generate_single_example()

    print("\nGenerated Example:")
    print(json.dumps(example, indent=2))

    # Generate batch
    # examples = generator.generate_batch(num_examples=10, output_file=Path("output.jsonl"))
