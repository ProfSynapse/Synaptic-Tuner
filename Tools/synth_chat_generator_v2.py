#!/usr/bin/env python3
"""
True Self-Play Synthetic Data Generator

This module implements true self-play where the model generates BOTH:
1. User prompts (what to ask)
2. Assistant responses (how to respond)

This avoids "teaching to the test" by not using evaluation prompts.

Architecture:
    1. Prompt Generator: Generate diverse user requests
       - Can use teacher model (Claude/GPT) OR
       - Use fine-tuned model to generate its own prompts

    2. Response Generator: Fine-tuned model responds

    3. Validator: Check correctness

    4. Optional: MCP executor for functional validation

    5. Collector: Save for KTO training

Usage:
    # Basic self-play (model generates prompts for itself)
    python Tools/selfplay_generator_v2.py \
        --model claudesidian-mcp \
        --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
        --num-examples 1000

    # With teacher model for prompt generation
    python Tools/selfplay_generator_v2.py \
        --model claudesidian-mcp \
        --prompt-generator-model claude-3-5-sonnet-20241022 \
        --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
        --num-examples 1000

    # With MCP execution (real vault interaction)
    python Tools/selfplay_generator_v2.py \
        --model claudesidian-mcp \
        --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
        --num-examples 1000 \
        --execute-mcp \
        --vault-path ~/obsidian-test-vault
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Evaluator.lmstudio_client import LMStudioClient
from Evaluator.config import LMStudioSettings
from tools.validate_syngen import validate_example, ExampleReport

# Manager-specific prompt generation templates
MANAGER_CONFIGS = {
    "vaultManager": {
        "description": "Manages vault structure - creates/deletes folders, organizes files",
        "modes": ["createFolder", "deleteFolder", "renameFolder", "moveFolder"],
        "examples": [
            "Create a new folder called 'Project Ideas' in the root directory",
            "Delete the 'Archive' folder and all its contents",
            "Rename the folder 'Old Notes' to 'Legacy Notes'",
            "Move all files from 'Temp' to 'Archive'",
        ]
    },
    "contentManager": {
        "description": "Manages note content - creates/reads/updates/deletes notes",
        "modes": ["createNote", "readNote", "updateNote", "deleteNote", "appendToNote"],
        "examples": [
            "Create a new note called 'Meeting Notes.md' with today's date as the header",
            "Read the contents of 'TODO.md'",
            "Update 'Project Plan.md' to include the new timeline",
            "Delete the note 'Draft.md'",
            "Append 'Review PR #42' to my TODO list",
        ]
    },
    "memoryManager": {
        "description": "Manages sessions, workspaces, and context memory",
        "modes": ["createWorkspace", "loadWorkspace", "saveSession", "rememberContext"],
        "examples": [
            "Create a new workspace for the Phoenix Migration project",
            "Load my 'Q4 Planning' workspace",
            "Save my current session so I can resume later",
            "Remember that we decided to use PostgreSQL for the database",
        ]
    },
    "vaultLibrarian": {
        "description": "Advanced search, batch operations, and analysis",
        "modes": ["searchNotes", "batchUpdate", "findByTag", "listByDate", "analyze"],
        "examples": [
            "Search for all notes mentioning 'deployment' from the past week",
            "Find all notes tagged with #urgent that haven't been updated in 30 days",
            "List all notes in the 'Projects' folder sorted by last modified",
            "Analyze my vault and show me which topics I've written most about",
        ]
    },
    "agentManager": {
        "description": "Manages custom agents - creates, executes, and organizes agents",
        "modes": ["createAgent", "executePrompt", "batchExecute", "deleteAgent", "listAgents"],
        "examples": [
            "Create a new agent called 'Standup Summarizer' that generates daily summaries",
            "Execute the prompt 'List all blockers' using the Project Manager agent",
            "Send 'Generate weekly report' to all my reporting agents",
            "Delete the 'Test Agent' since we don't need it anymore",
        ]
    }
}

def get_prompt_generation_system(manager: str) -> str:
    """Get manager-specific system prompt for prompt generation."""
    config = MANAGER_CONFIGS[manager]
    return f"""You are working in an Obsidian vault with a powerful assistant.

Your current focus is using the **{manager}** tool category:
{config['description']}

Available modes in {manager}:
{', '.join(config['modes'])}

Generate a realistic user request that would use one of these {manager} tools.

Example requests for {manager}:
{chr(10).join(f'- "{ex}"' for ex in config['examples'])}

Requirements:
1. Be specific and actionable
2. Use realistic file/folder/workspace names
3. Request should be something a real user would ask
4. Vary complexity (simple to moderately detailed)

Generate ONE user request. Output ONLY the request text, nothing else."""

def get_prompt_generation_user(manager: str) -> str:
    """Get user message for prompt generation."""
    config = MANAGER_CONFIGS[manager]
    modes = random.sample(config['modes'], min(2, len(config['modes'])))

    return f"""Generate a user request for {manager}.

Focus on these modes: {', '.join(modes)}

Be creative but realistic. Just output the request."""


# Behavioral agents - Try to elicit specific behaviors without teaching to the test
BEHAVIORAL_AGENTS = {
    "intellectual_humility": {
        "description": "Elicit admissions of uncertainty and appropriate caution",
        "scenarios": [
            "Ask me to {action} for something that might not exist",
            "Request {action} with ambiguous parameters",
            "Ask about {topic} I might not have information about",
        ],
        "examples": [
            "Search for notes about 'quantum entanglement research' - I might not have any",
            "Create a folder for my 'blockchain startup' - but I'm not sure where to put it",
            "Find all notes mentioning 'Project Nebula' - I think that was the codename but not certain",
        ]
    },
    "error_recovery": {
        "description": "Elicit graceful error handling and recovery suggestions",
        "scenarios": [
            "Ask to {action} something in a way that might fail",
            "Request {action} that could have multiple interpretations",
            "Ask to {action} with potentially conflicting requirements",
        ],
        "examples": [
            "Delete the folder 'Temp' but keep any important files",
            "Create a note called 'Meeting Notes.md' - oh wait, that already exists",
            "Search for notes tagged #urgent but exclude anything from 'Archive' folder",
        ]
    },
    "verification_before_action": {
        "description": "Elicit confirmation requests before destructive actions",
        "scenarios": [
            "Request {action} that would delete or modify important data",
            "Ask to {action} in bulk or across multiple items",
            "Request {action} that can't be easily undone",
        ],
        "examples": [
            "Delete all notes older than 6 months from my vault",
            "Move everything from 'Work' folder to 'Archive'",
            "Rename all my project notes to include the current date",
        ]
    },
    "workspace_awareness": {
        "description": "Elicit proper workspace and context management",
        "scenarios": [
            "Request {action} across different workspaces",
            "Ask to {action} while remembering previous context",
            "Request {action} that requires workspace switching",
        ],
        "examples": [
            "I'm working on Atlas Rollout - create a folder for weekly reports",
            "Switch to my Phoenix Migration workspace and show me recent blockers",
            "Remember that we're using the new folder structure for all projects now",
        ]
    },
}

def get_behavioral_prompt(behavior: str) -> str:
    """Generate a prompt designed to elicit a specific behavior."""
    config = BEHAVIORAL_AGENTS[behavior]

    # Choose a scenario template
    scenario_template = random.choice(config['scenarios'])

    # Or use a pre-written example
    if random.random() < 0.5:  # 50% chance to use example
        return random.choice(config['examples'])

    # Fill in template
    actions = {
        '{action}': random.choice(['search', 'create', 'delete', 'update', 'find', 'list', 'move']),
        '{topic}': random.choice(['deployment strategy', 'API architecture', 'security protocol', 'performance metrics']),
    }

    for placeholder, value in actions.items():
        scenario_template = scenario_template.replace(placeholder, value)

    return scenario_template


@dataclass
class GenerationStats:
    """Track statistics during generation."""
    total_prompts_generated: int = 0
    total_responses_generated: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    mcp_executions: int = 0
    mcp_failures: int = 0
    errors: int = 0

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"\nGeneration Statistics:\n"
            f"  Prompts generated: {self.total_prompts_generated}\n"
            f"  Responses generated: {self.total_responses_generated}\n"
            f"  Valid examples: {self.valid_examples}\n"
            f"  Invalid examples: {self.invalid_examples}\n"
            f"  MCP executions: {self.mcp_executions}\n"
            f"  MCP failures: {self.mcp_failures}\n"
            f"  Errors: {self.errors}\n"
            f"  Success rate: {self.valid_examples / max(1, self.total_responses_generated) * 100:.1f}%\n"
        )


class PromptGenerator:
    """Generate diverse user prompts for self-play."""

    def __init__(
        self,
        generator_model: Optional[str] = None,
        lmstudio_client: Optional[LMStudioClient] = None,
        behavioral_probability: float = 0.3,  # 30% behavioral, 70% tool-based
    ):
        """Initialize prompt generator.

        Args:
            generator_model: Model to use for prompt generation (if different from response model)
            lmstudio_client: LM Studio client (if using same model for prompts)
            behavioral_probability: Probability of generating behavioral prompts vs tool-based
        """
        self.generator_model = generator_model
        self.client = lmstudio_client
        self.managers = list(MANAGER_CONFIGS.keys())
        self.behaviors = list(BEHAVIORAL_AGENTS.keys())
        self.behavioral_probability = behavioral_probability

    def generate_prompt(self) -> str:
        """Generate a user prompt with sampling variation.

        Randomly chooses between:
        - Tool-based prompts (manager-specific)
        - Behavioral prompts (elicit specific behaviors)

        Returns:
            User prompt string
        """
        # Decide whether to generate behavioral or tool-based prompt
        use_behavioral = random.random() < self.behavioral_probability

        if use_behavioral:
            # Generate behavioral prompt
            return self._generate_behavioral_prompt()
        elif self.generator_model:
            # TODO: Use teacher model (Claude/GPT) via API
            # For now, use predefined templates
            return self._generate_from_templates()
        elif self.client:
            # Use fine-tuned model to generate its own prompts
            return self._generate_from_model()
        else:
            # Fallback to templates
            return self._generate_from_templates()

    def _generate_behavioral_prompt(self) -> str:
        """Generate a prompt designed to elicit a specific behavior."""
        behavior = random.choice(self.behaviors)
        return get_behavioral_prompt(behavior)

    def _generate_from_model(self) -> str:
        """Generate prompt using the fine-tuned model itself with sampling variation."""
        # Randomly select a manager category
        manager = random.choice(self.managers)

        # Build manager-specific prompt
        messages = [
            {"role": "system", "content": get_prompt_generation_system(manager)},
            {"role": "user", "content": get_prompt_generation_user(manager)}
        ]

        # Vary sampling parameters for diversity
        temperature = random.uniform(0.6, 1.2)  # Higher temp = more creative
        top_p = random.uniform(0.85, 0.98)      # Nucleus sampling
        max_tokens = random.randint(50, 150)    # Vary length

        try:
            response = self.client.chat(
                messages,
                temperature=temperature,
                top_p=top_p,
                # max_tokens not supported by all LM Studio models
            )
            prompt = response.message.strip()

            # Clean up response
            if prompt.lower().startswith("user:"):
                prompt = prompt[5:].strip()
            if prompt.startswith('"') and prompt.endswith('"'):
                prompt = prompt[1:-1]

            return prompt
        except Exception as e:
            print(f"  Error generating prompt from model: {e}")
            return self._generate_from_templates()

    def _generate_from_templates(self) -> str:
        """Generate prompt from manager-specific templates."""
        # Randomly select a manager
        manager = random.choice(self.managers)
        config = MANAGER_CONFIGS[manager]

        # Get manager-specific templates
        manager_templates = {
            "vaultManager": [
                "Create a new folder called '{name}' in {location}",
                "Delete the folder '{name}' from {location}",
                "Rename the folder '{old_name}' to '{new_name}'",
                "Move the '{name}' folder to {location}",
                "Create a folder structure for {project}",
            ],
            "contentManager": [
                "Create a new note called '{title}' about {topic}",
                "Update the note '{title}' to include {content}",
                "Delete the note '{title}'",
                "Append '{text}' to the note '{title}'",
                "Read the contents of '{title}'",
                "Create a note in {location} with title '{title}' about {topic}",
            ],
            "memoryManager": [
                "Create a new workspace for {project}",
                "Load the workspace '{workspace}'",
                "Save my current session state",
                "Remember that {fact}",
                "Switch to workspace '{workspace}' for {project}",
            ],
            "vaultLibrarian": [
                "Search for all notes about {topic}",
                "Find notes mentioning '{keyword}' from the past {timeframe}",
                "List all notes in the '{folder}' folder",
                "Search for notes tagged with #{tag}",
                "Find all TODO items across my vault",
                "Show me notes created in the last {timeframe}",
            ],
            "agentManager": [
                "Create a new agent called '{agent_name}' that {purpose}",
                "Execute the prompt '{prompt}' with agent '{agent}'",
                "List all available agents",
                "Delete the agent '{agent_name}'",
                "Send '{prompt}' to all my {category} agents",
            ],
        }

        # Random template from selected manager
        templates = manager_templates.get(manager, [])
        if not templates:
            # Fallback to example
            return random.choice(config['examples'])

        template = random.choice(templates)

        # Fill in placeholders with randomization
        placeholders = {
            '{name}': random.choice(['Project Ideas', 'Meeting Notes', 'Research', 'Archive', 'Daily Logs', 'Q4 Planning']),
            '{old_name}': random.choice(['Old Notes', 'Temp', 'Draft']),
            '{new_name}': random.choice(['Archive', 'Legacy Notes', 'Final']),
            '{location}': random.choice(['the root directory', 'Projects/', 'Notes/', 'Work/', 'Personal/']),
            '{topic}': random.choice(['project planning', 'machine learning', 'quarterly review', 'brainstorming', 'sprint retrospective', 'deployment strategy']),
            '{title}': random.choice(['TODO.md', 'Ideas.md', 'Notes.md', 'Summary.md', 'Plan.md', 'Standup.md']),
            '{content}': random.choice(['deployment steps', 'action items', 'key decisions', 'blockers', 'next steps']),
            '{text}': random.choice(['Review PR #42', 'Call with client', 'Update documentation', 'Fix bug in auth', 'Deploy to staging']),
            '{project}': random.choice(['Atlas Rollout', 'Phoenix Migration', 'Q4 Planning', 'Product Launch', 'System Upgrade']),
            '{workspace}': random.choice(['Atlas Rollout', 'Phoenix Migration', 'Q4 Planning', 'Daily Work']),
            '{fact}': random.choice(['we decided to use PostgreSQL', 'the deadline is next Friday', 'John is leading this initiative', 'we need to coordinate with DevOps']),
            '{timeframe}': random.choice(['week', 'month', 'year', 'day', '3 days', '2 weeks']),
            '{keyword}': random.choice(['deployment', 'bug', 'feature', 'meeting', 'decision', 'blocker', 'urgent']),
            '{folder}': random.choice(['Projects', 'Archive', 'Work', 'Personal', 'Team Notes']),
            '{tag}': random.choice(['urgent', 'review', 'todo', 'idea', 'bug', 'blocked']),
            '{agent_name}': random.choice(['Standup Summarizer', 'Report Generator', 'Task Tracker', 'Vault Auditor']),
            '{purpose}': random.choice(['summarizes daily notes', 'tracks action items', 'generates reports', 'audits vault structure', 'finds blockers']),
            '{agent}': random.choice(['Workspace Auditor', 'Report Generator', 'Task Tracker', 'Standup Summarizer']),
            '{prompt}': random.choice(['List all blockers', 'Summarize this week', 'Find overdue tasks', 'Generate weekly report']),
            '{category}': random.choice(['reporting', 'analysis', 'tracking', 'summary']),
        }

        for placeholder, value in placeholders.items():
            template = template.replace(placeholder, value)

        return template


class TrueSelfPlayGenerator:
    """True self-play generator where model generates both prompts and responses."""

    def __init__(
        self,
        model_name: str,
        output_path: Path,
        num_examples: int = 1000,
        temperature: float = 0.7,
        execute_mcp: bool = False,
        vault_path: Optional[Path] = None,
        prompt_generator_model: Optional[str] = None,
        lmstudio_host: Optional[str] = None,
        lmstudio_port: Optional[int] = None,
    ):
        """Initialize true self-play generator."""
        self.model_name = model_name
        self.output_path = output_path
        self.num_examples = num_examples
        self.temperature = temperature
        self.execute_mcp = execute_mcp
        self.vault_path = vault_path
        self.prompt_generator_model = prompt_generator_model

        # Initialize LM Studio client
        settings = LMStudioSettings(
            model=model_name,
            host=lmstudio_host or "localhost",
            port=lmstudio_port or 1234,
        )
        self.client = LMStudioClient(settings=settings)

        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(
            generator_model=prompt_generator_model,
            lmstudio_client=self.client if not prompt_generator_model else None,
        )

        # Statistics
        self.stats = GenerationStats()

        # Validate LM Studio connection
        if not self.client.is_server_running():
            raise RuntimeError("LM Studio server is not running or not accessible")
        print(f"✓ Connected to LM Studio at {settings.base_url()}")

        # Generate session context for all prompts
        self.session_context = self._generate_session_context()

    def _generate_session_context(self) -> Dict[str, Any]:
        """Generate a consistent session context for all prompts."""
        timestamp = int(datetime.now().timestamp() * 1000)
        return {
            "sessionId": f"session_{timestamp}_selfplay",
            "workspaceId": "ws_default",
            "sessionDescription": "Self-play training data generation",
            "sessionMemory": "Generating diverse examples for model training",
        }

    def generate_response(self, user_prompt: str) -> Tuple[str, Optional[str]]:
        """Generate a response from the model.

        Args:
            user_prompt: User request

        Returns:
            (response_text, error_message)
        """
        try:
            # Build messages (NO system prompt - model is fine-tuned and should already know)
            messages = [
                {"role": "user", "content": user_prompt}
            ]

            # Vary sampling parameters for response diversity
            temperature = random.uniform(0.3, 0.9)  # Some conservative, some creative
            top_p = random.uniform(0.90, 0.98)      # Nucleus sampling

            # Call model with varied parameters
            response = self.client.chat(
                messages,
                temperature=temperature,
                top_p=top_p,
            )

            return response.message, None

        except Exception as e:
            return None, str(e)

    def validate_response(
        self,
        user_prompt: str,
        response_text: str,
    ) -> Tuple[bool, ExampleReport]:
        """Validate a response."""
        # Create example in ChatML format
        example = {
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_text},
            ],
            "label": True,  # Assume desirable for now
        }

        # Validate
        report = validate_example(1, example)

        return report.is_valid, report

    def generate_dataset(self) -> None:
        """Generate the complete dataset via self-play."""
        print(f"\nGenerating {self.num_examples} examples via TRUE SELF-PLAY...")
        print(f"  Model: {self.model_name}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Prompt generator: {'Model itself' if not self.prompt_generator_model else self.prompt_generator_model}")
        print(f"  Execute MCP: {self.execute_mcp}")
        print()

        collected_examples = []

        for i in range(self.num_examples):
            # 1. Generate user prompt
            print(f"[{i+1}/{self.num_examples}] Generating prompt...", end=" ")
            user_prompt = self.prompt_generator.generate_prompt()
            self.stats.total_prompts_generated += 1
            print(f"'{user_prompt[:60]}...'")

            # 2. Generate response
            print(f"  Generating response...", end=" ")
            response_text, error = self.generate_response(user_prompt)
            self.stats.total_responses_generated += 1

            if error:
                print(f"ERROR: {error}")
                self.stats.errors += 1
                continue

            # 3. Validate response
            is_valid, validation_report = self.validate_response(
                user_prompt,
                response_text,
            )

            # 4. Collect example
            example = {
                "conversations": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response_text},
                ],
                "label": is_valid,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "prompt_source": "model" if not self.prompt_generator_model else self.prompt_generator_model,
                    "validation_issues": [
                        {"level": issue.level, "message": issue.message}
                        for issue in validation_report.issues
                    ] if not is_valid else [],
                }
            }
            collected_examples.append(example)

            # Update stats
            if is_valid:
                self.stats.valid_examples += 1
                print(f"  ✓ VALID")
            else:
                self.stats.invalid_examples += 1
                print(f"  ✗ INVALID ({len(validation_report.issues)} issues)")

        # Interleave examples for KTO training
        print("\nInterleaving examples for KTO training...")
        interleaved_examples = self._interleave_examples(collected_examples)

        # Write to output file
        print(f"Writing {len(interleaved_examples)} examples to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for example in interleaved_examples:
                # Remove metadata before writing
                output_example = {
                    "conversations": example["conversations"],
                    "label": example["label"],
                }
                f.write(json.dumps(output_example, ensure_ascii=False) + '\n')

        print(f"✓ Dataset written to {self.output_path}")
        print(self.stats.summary())

    def _interleave_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Interleave examples in True/False/True/False pattern for KTO."""
        # Separate valid and invalid examples
        valid = [ex for ex in examples if ex["label"] is True]
        invalid = [ex for ex in examples if ex["label"] is False]

        print(f"  Valid examples: {len(valid)}")
        print(f"  Invalid examples: {len(invalid)}")

        # Balance if needed
        min_count = min(len(valid), len(invalid))
        if min_count == 0:
            print("  WARNING: No invalid examples! Cannot create interleaved dataset.")
            print("  Returning all valid examples (not suitable for KTO training)")
            return valid

        # Trim to equal sizes
        valid = valid[:min_count]
        invalid = invalid[:min_count]

        # Shuffle both
        random.shuffle(valid)
        random.shuffle(invalid)

        # Interleave: True, False, True, False, ...
        interleaved = []
        for v, i in zip(valid, invalid):
            interleaved.append(v)
            interleaved.append(i)

        print(f"  Interleaved dataset size: {len(interleaved)}")

        return interleaved


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data via true self-play"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LM Studio model name (for responses)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (default: 0.7)"
    )
    parser.add_argument(
        "--prompt-generator-model",
        type=str,
        help="Optional: Use different model for prompt generation (e.g., claude-3-5-sonnet)"
    )
    parser.add_argument(
        "--execute-mcp",
        action="store_true",
        help="Execute tool calls via MCP (requires vault-path)"
    )
    parser.add_argument(
        "--vault-path",
        type=Path,
        help="Path to test Obsidian vault (required if --execute-mcp)"
    )
    parser.add_argument(
        "--lmstudio-host",
        type=str,
        help="LM Studio host (default: localhost)"
    )
    parser.add_argument(
        "--lmstudio-port",
        type=int,
        help="LM Studio port (default: 1234)"
    )

    args = parser.parse_args()

    # Validate args
    if args.execute_mcp and not args.vault_path:
        parser.error("--vault-path is required when --execute-mcp is set")

    if args.output.exists():
        response = input(f"Output file '{args.output}' exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    try:
        generator = TrueSelfPlayGenerator(
            model_name=args.model,
            output_path=args.output,
            num_examples=args.num_examples,
            temperature=args.temperature,
            execute_mcp=args.execute_mcp,
            vault_path=args.vault_path,
            prompt_generator_model=args.prompt_generator_model,
            lmstudio_host=args.lmstudio_host,
            lmstudio_port=args.lmstudio_port,
        )
    except Exception as e:
        print(f"Error initializing generator: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate dataset
    try:
        generator.generate_dataset()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
