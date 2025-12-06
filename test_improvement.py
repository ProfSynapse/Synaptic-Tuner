#!/usr/bin/env python3
"""Quick test of improvement engine on a single line."""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

from improvement_engine.core.models import ImprovementConfig
from improvement_engine.services.improvement_service import ImprovementService
from improvement_engine.utils.logger import get_logger

# Load environment variables from root .env (for API keys)
env_path = Path(".env")
load_dotenv(env_path)

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Test improvement engine on a single line")
parser.add_argument("--backend", choices=["openrouter", "lmstudio", "ollama"],
                    default="openrouter", help="LLM backend to use")
parser.add_argument("--model", help="Model name (optional, uses backend defaults)")
parser.add_argument("--line", type=int, default=7, help="Line number to test")
args = parser.parse_args()

config = ImprovementConfig(
    input_file=Path("Datasets/tools_datasets/thinking/contentManager/tools_v1.4.jsonl"),
    output_file=Path("Datasets/tools_datasets/thinking/contentManager/tools_v1.5.jsonl"),
    backend=args.backend,
    model=args.model,
    batch_size=1,
    start_line=args.line,
    end_line=args.line,
    dry_run=True
)

logger = get_logger()

print("\n" + "="*60)
print("Testing Improvement Engine - Line 7 of contentManager v1.4")
print("="*60 + "\n")

print("Configuration:")
print(f"  Input:  {config.input_file}")
print(f"  Line:   {config.start_line}")
print(f"  Dry run: {config.dry_run}")
print(f"  Backend: {os.getenv('IMPROVEMENT_BACKEND', 'default')}")
print(f"  Model:   {os.getenv('IMPROVEMENT_MODEL', 'default')}\n")

# Run improvement
service = ImprovementService(config=config, logger=logger)
results = service.run()

print("\n" + "="*60)
print("Results")
print("="*60)

for batch_result in results:
    print(f"\nBatch {batch_result.batch_number}:")
    print(f"  Success: {batch_result.successful}/{batch_result.total_processed}")
    print(f"  Failed: {batch_result.failed}/{batch_result.total_processed}")

    if batch_result.results:
        result = batch_result.results[0]

        if not result.success:
            print(f"\n  ❌ Error: {result.error}")
            if result.validation_errors:
                print(f"  Validation errors:")
                for error in result.validation_errors:
                    print(f"    - {error}")
        else:
            # Extract thinking blocks from conversations
            orig_thinking = None
            imp_thinking = None

            for conv in result.original.conversations:
                if conv["role"] == "assistant" and "<thinking>" in conv["content"]:
                    # Extract JSON from thinking tags
                    start = conv["content"].find("{")
                    end = conv["content"].rfind("}") + 1
                    if start != -1 and end > start:
                        orig_thinking = json.loads(conv["content"][start:end])
                    break

            if result.improved:
                for conv in result.improved.conversations:
                    if conv["role"] == "assistant" and "<thinking>" in conv["content"]:
                        # Extract JSON from thinking tags
                        start = conv["content"].find("{")
                        end = conv["content"].rfind("}") + 1
                        if start != -1 and end > start:
                            imp_thinking = json.loads(conv["content"][start:end])
                        break

            if orig_thinking and imp_thinking:
                print(f"\n  ORIGINAL:")
                print(f"    Goal: {orig_thinking['goal']}")
                print(f"    Memory: {orig_thinking['memory'][:80]}...")
                print(f"    Requirements[0]: {orig_thinking['requirements'][0]}")
                print(f"    Plan[0]: {orig_thinking['plan'][0]}")
                print(f"    Confidence: {orig_thinking['confidence']}")

                print(f"\n  IMPROVED:")
                print(f"    Goal: {imp_thinking['goal']}")
                print(f"    Memory: {imp_thinking['memory'][:80]}...")
                print(f"    Requirements[0]: {imp_thinking['requirements'][0]}")
                print(f"    Plan[0]: {imp_thinking['plan'][0]}")
                print(f"    Confidence: {imp_thinking['confidence']}")

                print(f"\n  KEY IMPROVEMENTS:")
                if orig_thinking['goal'] != imp_thinking['goal']:
                    print(f"    ✓ Goal improved")
                if orig_thinking['memory'] != imp_thinking['memory']:
                    print(f"    ✓ Memory enhanced")
                if orig_thinking['requirements'] != imp_thinking['requirements']:
                    print(f"    ✓ Requirements distinct from plan")
                if orig_thinking['confidence'] != imp_thinking['confidence']:
                    print(f"    ✓ Confidence calibrated")
