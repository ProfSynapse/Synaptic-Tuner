# Notebook Cells for vLLM Inference + Evaluation

Add these cells after the upload section in `sft_colab_beginner.ipynb`:

---

## Cell: Install vLLM and Evaluator Dependencies

```python
# @title 📦 Install vLLM and Evaluation Tools
# @markdown Install vLLM for fast inference and dependencies for running evaluations.

%%capture
!pip install vllm>=0.6.0
!pip install requests  # For Evaluator

print("✓ vLLM installed")
print("✓ Evaluator dependencies installed")
```

---

## Cell: Download Evaluation Framework

```python
# @title 📥 Download Evaluation Framework
# @markdown Download the Evaluator code and prompt sets from the repository.

import os
import requests
import json
from pathlib import Path

# Create Evaluator directory structure
os.makedirs("Evaluator/prompts", exist_ok=True)
os.makedirs("Evaluator/results", exist_ok=True)

# Base URL for raw files from GitHub
REPO_BASE = "https://raw.githubusercontent.com/ProfSynapse/Toolset-Training/main/Evaluator"

# Files to download
evaluator_files = {
    "runner.py": "Evaluator/runner.py",
    "schema_validator.py": "Evaluator/schema_validator.py",
    "prompt_sets.py": "Evaluator/prompt_sets.py",
    "reporting.py": "Evaluator/reporting.py",
    "config.py": "Evaluator/config.py",
    "__init__.py": "Evaluator/__init__.py",
}

prompt_files = {
    "full_coverage.json": "Evaluator/prompts/full_coverage.json",
    "behavioral_patterns.json": "Evaluator/prompts/behavioral_patterns.json",
    "baseline.json": "Evaluator/prompts/baseline.json",
    "tool_combos.json": "Evaluator/prompts/tool_combos.json",
}

def download_file(url, dest):
    """Download a file from URL to destination."""
    response = requests.get(url)
    response.raise_for_status()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, 'w') as f:
        f.write(response.text)

# Download evaluator files
print("Downloading evaluator framework...")
for name, path in evaluator_files.items():
    url = f"{REPO_BASE.replace('/prompts', '')}/{name}"
    try:
        download_file(url, path)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ⚠️  Failed to download {name}: {e}")

# Download prompt sets
print("\nDownloading prompt sets...")
for name, path in prompt_files.items():
    url = f"{REPO_BASE}/{name}"
    try:
        download_file(url, path)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ⚠️  Failed to download {name}: {e}")

print("\n✓ Evaluation framework ready!")
```

---

## Cell: Load Model with vLLM

```python
# @title 🚀 Load Model for Inference with vLLM
# @markdown Start vLLM server with your fine-tuned model.

from vllm import LLM, SamplingParams
import torch

# Model configuration
# @markdown ### 📍 Model Selection
# @markdown Choose which version to load for inference:
model_source = "HuggingFace Merged Model" # @param ["HuggingFace Merged Model", "Local LoRA Adapters", "Local Final Model"]

if model_source == "HuggingFace Merged Model":
    inference_model_name = f"{hf_user}/{OUTPUT_MODEL_NAME}-merged"
    print(f"Loading merged model from HuggingFace: {inference_model_name}")
elif model_source == "Local LoRA Adapters":
    inference_model_name = MODEL_NAME  # Base model
    enable_lora = True
    lora_path = f"{output_dir}/final_model"
    print(f"Loading base model with LoRA adapters from: {lora_path}")
else:  # Local Final Model
    inference_model_name = f"{output_dir}/final_model"
    print(f"Loading local model from: {inference_model_name}")

# @markdown ### ⚙️ Inference Configuration
tensor_parallel_size = 1 # @param {type:"integer"}
gpu_memory_utilization = 0.8 # @param {type:"slider", min:0.5, max:0.95, step:0.05}

print("\nInitializing vLLM engine...")
print(f"  • Model: {inference_model_name}")
print(f"  • Tensor Parallel: {tensor_parallel_size}")
print(f"  • GPU Memory Utilization: {gpu_memory_utilization}")
print()

# Initialize vLLM
vllm_kwargs = {
    "model": inference_model_name,
    "tensor_parallel_size": tensor_parallel_size,
    "gpu_memory_utilization": gpu_memory_utilization,
    "trust_remote_code": True,
    "dtype": "bfloat16" if is_bfloat16_supported() else "float16",
}

# Add LoRA if using local adapters
if model_source == "Local LoRA Adapters":
    vllm_kwargs["enable_lora"] = True

llm = LLM(**vllm_kwargs)

print("✓ vLLM engine ready!")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  VRAM in use: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

---

## Cell: Create vLLM Client for Evaluator

```python
# @title 🔌 Create vLLM Client Interface
# @markdown Create a client that wraps vLLM for the Evaluator framework.

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence
import time

@dataclass
class VLLMResponse:
    """Response from vLLM inference."""
    message: str
    raw: Dict[str, Any]
    latency_s: float

class VLLMClient:
    """
    vLLM client that implements the same interface as OllamaClient/LMStudioClient.
    This allows it to work seamlessly with the Evaluator framework.
    """

    def __init__(
        self,
        llm: LLM,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        seed: int = None,
    ):
        self.llm = llm
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed

    def chat(self, messages: Sequence[Mapping[str, str]]) -> VLLMResponse:
        """
        Send a chat conversation to vLLM and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            VLLMResponse with the assistant's message, raw output, and latency
        """
        # Format messages into a prompt
        # For Mistral models, use [INST] format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f" {content}</s>")
            elif role == "system":
                prompt_parts.append(f"{content} ")

        prompt = "<s>" + "".join(prompt_parts)

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )

        # Generate
        start = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params)
        latency_s = time.perf_counter() - start

        # Extract response
        output = outputs[0]
        message = output.outputs[0].text.strip()

        # Build raw response dict
        raw = {
            "prompt": prompt,
            "output": message,
            "finish_reason": output.outputs[0].finish_reason,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
        }

        return VLLMResponse(
            message=message,
            raw=raw,
            latency_s=latency_s
        )

# Create client
vllm_client = VLLMClient(
    llm=llm,
    temperature=0.2,
    top_p=0.9,
    max_tokens=1024,
    seed=42,
)

print("✓ vLLM client created and ready for evaluation")
```

---

## Cell: Evaluation Configuration Form

```python
# @title 🎯 Configure Evaluation Run
# @markdown Choose which tests to run and how many samples to evaluate.

# @markdown ### 📋 Test Suite Selection
test_suite = "Full Coverage (47 tools)" # @param ["Full Coverage (47 tools)", "Behavioral Patterns (21 tests)", "Baseline (6 tests)", "Tool Combos (Multi-step)", "All Suites"]

# @markdown ### 🔢 Evaluation Limits
# @markdown Limit the number of prompts to test (useful for quick checks). Leave at 0 for no limit.
max_prompts = 0 # @param {type:"integer"}

# @markdown ### 🎲 Generation Settings
eval_temperature = 0.2 # @param {type:"slider", min:0.0, max:1.0, step:0.1}
eval_top_p = 0.9 # @param {type:"slider", min:0.0, max:1.0, step:0.1}
eval_max_tokens = 1024 # @param {type:"integer"}
eval_seed = 42 # @param {type:"integer"}

# Map test suite to prompt file
suite_map = {
    "Full Coverage (47 tools)": ["Evaluator/prompts/full_coverage.json"],
    "Behavioral Patterns (21 tests)": ["Evaluator/prompts/behavioral_patterns.json"],
    "Baseline (6 tests)": ["Evaluator/prompts/baseline.json"],
    "Tool Combos (Multi-step)": ["Evaluator/prompts/tool_combos.json"],
    "All Suites": [
        "Evaluator/prompts/full_coverage.json",
        "Evaluator/prompts/behavioral_patterns.json",
        "Evaluator/prompts/baseline.json",
        "Evaluator/prompts/tool_combos.json",
    ]
}

prompt_files = suite_map[test_suite]

# Update client settings
vllm_client.temperature = eval_temperature
vllm_client.top_p = eval_top_p
vllm_client.max_tokens = eval_max_tokens
vllm_client.seed = eval_seed

print("✓ Evaluation configured:")
print(f"  • Test Suite: {test_suite}")
print(f"  • Prompt Files: {len(prompt_files)}")
if max_prompts > 0:
    print(f"  • Max Prompts: {max_prompts}")
else:
    print(f"  • Max Prompts: No limit (all prompts)")
print(f"  • Temperature: {eval_temperature}")
print(f"  • Top-p: {eval_top_p}")
print(f"  • Max Tokens: {eval_max_tokens}")
print(f"  • Seed: {eval_seed}")
```

---

## Cell: Run Evaluation

```python
# @title ▶️ Run Evaluation
# @markdown This will run the selected test suite and display results.

import sys
sys.path.insert(0, '/content')  # Add current dir to path

from Evaluator.prompt_sets import load_prompt_cases, filter_prompts
from Evaluator.runner import evaluate_cases
from Evaluator.reporting import build_run_payload, console_summary
from Evaluator.config import PromptFilter
from datetime import datetime
import json

# Results storage
all_results = []
all_records = []

print("=" * 60)
print("STARTING EVALUATION")
print("=" * 60)
print()

for prompt_file in prompt_files:
    print(f"\n📝 Loading prompts from: {prompt_file}")

    # Load and filter prompts
    cases = load_prompt_cases(prompt_file)
    prompt_filter = PromptFilter(tags=None, limit=max_prompts if max_prompts > 0 else None)
    selected_cases = filter_prompts(cases, prompt_filter)

    print(f"   • Loaded {len(cases)} prompts")
    print(f"   • Selected {len(selected_cases)} prompts for evaluation")

    if not selected_cases:
        print("   ⚠️  No prompts matched filters, skipping...")
        continue

    # Progress callback
    def on_record(record):
        status = "✓" if record.passed else "✗"
        time_str = f"{record.latency_s:.2f}s" if record.latency_s else "N/A"
        print(f"   {status} {record.case.id} ({time_str})")

    # Run evaluation
    print(f"\n🔄 Running evaluation...")
    records = evaluate_cases(
        cases=selected_cases,
        client=vllm_client,
        dry_run=False,
        on_record=on_record,
    )

    all_records.extend(records)

    # Calculate stats for this file
    passed = sum(1 for r in records if r.passed)
    failed = sum(1 for r in records if not r.passed)
    avg_latency = sum(r.latency_s for r in records if r.latency_s) / len(records) if records else 0

    print(f"\n   Results: {passed}/{len(records)} passed ({passed/len(records)*100:.1f}%)")
    print(f"   Average latency: {avg_latency:.2f}s")

# Overall summary
print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

total_passed = sum(1 for r in all_records if r.passed)
total_failed = sum(1 for r in all_records if not r.passed)
total_tests = len(all_records)
overall_avg_latency = sum(r.latency_s for r in all_records if r.latency_s) / total_tests if total_tests else 0

print(f"\n📊 Overall Results:")
print(f"   • Total Tests: {total_tests}")
print(f"   • Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
print(f"   • Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
print(f"   • Average Latency: {overall_avg_latency:.2f}s")
print(f"   • Total Time: {sum(r.latency_s for r in all_records if r.latency_s):.2f}s")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"Evaluator/results/eval_{OUTPUT_MODEL_NAME}_{timestamp}.json"

# Build payload
payload = build_run_payload(
    records=all_records,
    model_name=OUTPUT_MODEL_NAME,
    prompts_path=", ".join(prompt_files),
    metadata={
        "test_suite": test_suite,
        "temperature": eval_temperature,
        "top_p": eval_top_p,
        "max_tokens": eval_max_tokens,
        "seed": eval_seed,
        "max_prompts": max_prompts if max_prompts > 0 else "all",
    }
)

with open(results_file, 'w') as f:
    json.dump(payload, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

# Store for analysis
eval_results = {
    "records": all_records,
    "payload": payload,
    "timestamp": timestamp,
    "results_file": results_file,
}
```

---

## Cell: Detailed Results Analysis

```python
# @title 📈 Analyze Results by Category
# @markdown Break down results by tool category and test type.

from collections import defaultdict
import pandas as pd

# Group by tags
results_by_tag = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})

for record in all_records:
    tags = record.case.tags if hasattr(record.case, 'tags') and record.case.tags else ["untagged"]

    for tag in tags:
        results_by_tag[tag]["total"] += 1
        if record.passed:
            results_by_tag[tag]["passed"] += 1
        else:
            results_by_tag[tag]["failed"] += 1

# Convert to DataFrame for nice display
df_data = []
for tag, stats in sorted(results_by_tag.items()):
    pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
    df_data.append({
        "Category": tag,
        "Passed": stats["passed"],
        "Failed": stats["failed"],
        "Total": stats["total"],
        "Pass Rate": f"{pass_rate:.1f}%"
    })

df = pd.DataFrame(df_data)
print("\n📊 Results by Category:")
print(df.to_string(index=False))

# Show failures if any
failures = [r for r in all_records if not r.passed]
if failures:
    print(f"\n❌ Failed Tests ({len(failures)}):")
    for record in failures[:10]:  # Show first 10 failures
        print(f"\n  • {record.case.id}")
        print(f"    Question: {record.case.question[:100]}...")
        if record.error:
            print(f"    Error: {record.error}")
        elif record.validator and record.validator.issues:
            print(f"    Issues:")
            for issue in record.validator.issues[:3]:  # Show first 3 issues
                print(f"      - [{issue.level}] {issue.message}")

    if len(failures) > 10:
        print(f"\n  ... and {len(failures) - 10} more failures")
else:
    print("\n✅ All tests passed!")

# Save to Google Drive if mounted
if os.path.exists(DRIVE_OUTPUT_DIR):
    drive_results_file = f"{DRIVE_OUTPUT_DIR}/eval_results_{timestamp}.json"
    with open(drive_results_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Results also saved to Google Drive: {drive_results_file}")
```

---

## Cell: Generate Markdown Report

```python
# @title 📝 Generate Markdown Report
# @markdown Create a human-readable markdown report of the evaluation.

from Evaluator.reporting import render_markdown

# Generate markdown
markdown_file = f"Evaluator/results/eval_{OUTPUT_MODEL_NAME}_{timestamp}.md"
markdown_content = render_markdown(all_records, OUTPUT_MODEL_NAME, test_suite)

with open(markdown_file, 'w') as f:
    f.write(markdown_content)

print(f"✓ Markdown report saved to: {markdown_file}")

# Display summary
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(markdown_content[:2000])  # Show first 2000 chars
print("\n... (see full report in markdown file)")

# Save to Google Drive if mounted
if os.path.exists(DRIVE_OUTPUT_DIR):
    drive_markdown_file = f"{DRIVE_OUTPUT_DIR}/eval_report_{timestamp}.md"
    with open(drive_markdown_file, 'w') as f:
        f.write(markdown_content)
    print(f"\n💾 Markdown report also saved to Google Drive: {drive_markdown_file}")
```

---

## Cell: Quick Test Interface

```python
# @title 🧪 Quick Test Interface
# @markdown Test your model with a custom prompt to see how it responds.

# @markdown ### Enter your test prompt:
test_prompt = "Can you search for all notes that mention 'Claude Code' and show me the results?" # @param {type:"string"}

# @markdown ### Generation settings:
quick_temperature = 0.2 # @param {type:"slider", min:0.0, max:1.0, step:0.1}
quick_max_tokens = 512 # @param {type:"integer"}

print("🤖 Generating response...\n")

# Create message
messages = [{"role": "user", "content": test_prompt}]

# Update client
vllm_client.temperature = quick_temperature
vllm_client.max_tokens = quick_max_tokens

# Generate
response = vllm_client.chat(messages)

print("=" * 60)
print("RESPONSE")
print("=" * 60)
print(response.message)
print()
print(f"⏱️  Latency: {response.latency_s:.2f}s")
print(f"📊 Tokens: {response.raw.get('completion_tokens', 'N/A')}")

# Validate response
from Evaluator.schema_validator import validate_assistant_response

try:
    validation = validate_assistant_response(response.message)
    print(f"\n✓ Validation: {'PASSED' if validation.passed else 'FAILED'}")

    if validation.tool_calls:
        print(f"\n🔧 Tool Calls Detected ({len(validation.tool_calls)}):")
        for tc in validation.tool_calls:
            print(f"   • {tc.name}")
            print(f"     Arguments: {list(tc.arguments.keys())}")

    if validation.issues:
        print(f"\n⚠️  Issues ({len(validation.issues)}):")
        for issue in validation.issues:
            print(f"   • [{issue.level}] {issue.message}")
except Exception as e:
    print(f"\n❌ Validation error: {e}")
```

---

## Usage Instructions

1. **Install dependencies** (Cell 1-2)
2. **Load model with vLLM** (Cell 3) - Choose HF merged model, local LoRA, or local model
3. **Create vLLM client** (Cell 4) - Wraps vLLM for Evaluator framework
4. **Configure evaluation** (Cell 5) - Use form to select test suite
5. **Run evaluation** (Cell 6) - Executes tests and shows progress
6. **Analyze results** (Cell 7) - Breakdown by category with pass rates
7. **Generate report** (Cell 8) - Create markdown report
8. **Quick test** (Cell 9) - Test individual prompts interactively

## Features

✅ **Interactive forms** - Easy configuration with dropdowns and sliders
✅ **Multiple test suites** - Full coverage, behavioral, baseline, combos, or all
✅ **Real-time progress** - See each test as it runs
✅ **Detailed analysis** - Results by category, failure breakdown
✅ **Markdown reports** - Human-readable summaries
✅ **Quick testing** - Test individual prompts on the fly
✅ **Google Drive integration** - Auto-save results if Drive is mounted

## Example Output

```
📊 Overall Results:
   • Total Tests: 47
   • Passed: 42 (89.4%)
   • Failed: 5 (10.6%)
   • Average Latency: 1.23s
   • Total Time: 57.81s

📊 Results by Category:
Category              Passed  Failed  Total  Pass Rate
agentManager              12       1     13     92.3%
contentManager            11       0     11    100.0%
memoryManager              8       2     10     80.0%
vaultLibrarian             5       1      6     83.3%
vaultManager               6       1      7     85.7%
```
