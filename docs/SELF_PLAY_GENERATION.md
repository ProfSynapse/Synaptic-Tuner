# Self-Play Synthetic Data Generation

This guide explains how to use your fine-tuned model to generate new training data through self-play, creating both correct and incorrect examples for further KTO refinement.

## Overview

The self-play system allows you to:

1. **Generate diverse responses** from your fine-tuned model
2. **Automatically validate** responses using your existing validator
3. **Optionally execute** tool calls against a real Obsidian vault (via MCP)
4. **Collect both correct and incorrect** examples
5. **Build interleaved KTO datasets** ready for training

This creates a **self-improvement loop** where your model generates data to train the next iteration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Self-Play Data Generator                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Prompt Sampler                                            │
│     - Load existing prompt sets                               │
│     - Generate variations with different temperatures         │
│                                                               │
│  2. Model Executor (via LM Studio)                            │
│     - Send prompts to your fine-tuned model                   │
│     - Collect responses                                       │
│                                                               │
│  3. Response Validator                                        │
│     - Use existing validate_syngen.py                         │
│     - Check tool call syntax                                  │
│     - Verify context objects                                  │
│     - Validate against tool schemas                           │
│                                                               │
│  4. MCP Executor (optional)                                   │
│     - Execute tool calls against test Obsidian vault          │
│     - Verify functional correctness                           │
│                                                               │
│  5. Data Collector                                            │
│     - Label examples as correct (label=true) or               │
│       incorrect (label=false)                                 │
│     - Store in ChatML format                                  │
│                                                               │
│  6. Dataset Builder                                           │
│     - Interleave True/False/True/False for KTO                │
│     - Balance dataset                                         │
│     - Write to JSONL                                          │
│                                                               │
│  Output: syngen_selfplay_YYYYMMDD.jsonl                      │
│          - Ready for KTO training                             │
│          - Perfectly interleaved                              │
│          - Balanced positive/negative examples                │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Fine-Tuned Model

You need a working fine-tuned model loaded in LM Studio:

```bash
# Load your model in LM Studio
# From LM Studio UI:
# 1. Load your model (e.g., username/claudesidian-mcp-v1)
# 2. Start the server (Developer > Server > Start Server)
# 3. Enable "Serve on Local Network" if using WSL
```

### 2. LM Studio Configuration (WSL)

If running from WSL and LM Studio is on Windows:

**In LM Studio (Windows):**
1. Click **Developer** → **Server**
2. Toggle ON **Serve on Local Network**
3. Note the IP address (e.g., `192.168.1.104`)

**In WSL `.env` file:**
```bash
LMSTUDIO_HOST=192.168.1.104
```

### 3. Prompt Sets

Use existing prompt sets from `Evaluator/prompts/`:

- `tool_prompts.json` - One prompt per tool (47 prompts)
- `behavior_prompts.json` - Behavior testing prompts

Or create your own following the same format.

## Basic Usage

### Generate 1000 examples (validation only)

This is the simplest approach - no MCP execution, just syntax validation:

```bash
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
  --num-examples 1000 \
  --temperature 0.7 \
  --num-variations 3
```

**What this does:**
1. Loads 47 prompts from `tool_prompts.json`
2. For each prompt, generates 3 variations with temperature 0.7
3. Sends each to your model via LM Studio
4. Validates each response using `validate_syngen.py`
5. Labels as `true` (valid) or `false` (invalid)
6. Interleaves in True/False/True/False pattern
7. Writes 1000 examples to output file

### With custom temperature range

Generate more diverse responses:

```bash
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/syngen_selfplay_diverse.jsonl \
  --num-examples 2000 \
  --temperature 0.9 \
  --num-variations 5
```

**Higher temperature = more diversity, more errors**
- Use `0.3-0.5` for mostly correct examples
- Use `0.7-0.9` for balanced mix
- Use `1.0+` for maximum diversity (many errors)

## Advanced Usage

### With MCP Execution (Future)

When MCP integration is complete, you can execute tool calls against a real vault:

```bash
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/syngen_selfplay_mcp.jsonl \
  --num-examples 1000 \
  --execute-mcp \
  --vault-path /path/to/test/vault
```

**What this adds:**
- Actually executes tool calls against test vault
- Verifies functional correctness (not just syntax)
- Labels examples as invalid if they fail execution
- Provides higher quality training data

**Note:** MCP execution is currently a placeholder. See "MCP Integration" section below.

## Output Format

The generator creates a JSONL file with interleaved examples:

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "tool_call: ..."}], "label": true}
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "tool_call: ..."}], "label": false}
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "tool_call: ..."}], "label": true}
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "tool_call: ..."}], "label": false}
...
```

**Key features:**
- ✅ **Perfectly interleaved** (True/False/True/False)
- ✅ **Balanced** (equal valid/invalid examples)
- ✅ **ChatML format** (ready for KTO training)
- ✅ **No system prompts** (model should internalize patterns)

## Training with Generated Data

Once you have a self-play dataset:

### 1. Validate it

```bash
python tools/validate_syngen.py Datasets/syngen_selfplay_20251204.jsonl
```

### 2. Train with KTO

```bash
cd Trainers/rtx3090_kto

python train_kto.py \
  --model-size 7b \
  --local-file ../../Datasets/syngen_selfplay_20251204.jsonl \
  --num-epochs 1 \
  --batch-size 4
```

### 3. Evaluate

```bash
python -m Evaluator.cli \
  --model your-refined-model \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Evaluator/results/selfplay_eval.json
```

## Self-Improvement Loop

You can create a continuous improvement loop:

```bash
#!/bin/bash
# Self-improvement loop

MODEL="claudesidian-mcp"
ITERATION=1

while true; do
  echo "=== Iteration $ITERATION ==="

  # 1. Generate data
  python Tools/selfplay_generator.py \
    --model "$MODEL" \
    --prompt-set Evaluator/prompts/tool_prompts.json \
    --output "Datasets/syngen_selfplay_iter${ITERATION}.jsonl" \
    --num-examples 1000

  # 2. Train
  cd Trainers/rtx3090_kto
  python train_kto.py \
    --model-size 7b \
    --local-file "../../Datasets/syngen_selfplay_iter${ITERATION}.jsonl" \
    --output-dir "kto_output_rtx3090/iter${ITERATION}"
  cd ../..

  # 3. Upload
  # (Use tuner.py upload or upload_to_hf.py)

  # 4. Evaluate
  # (Use Evaluator/cli.py)

  # 5. Update MODEL for next iteration
  MODEL="username/claudesidian-mcp-iter${ITERATION}"
  ITERATION=$((ITERATION + 1))

  # Ask to continue
  read -p "Continue to iteration $ITERATION? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    break
  fi
done
```

**Warning:** Monitor quality at each iteration to avoid:
- Model degradation (model collapse)
- Overfitting to its own mistakes
- Loss of diversity

## MCP Integration (Coming Soon)

The MCP executor is currently a placeholder. To integrate with Obsidian MCP:

### Architecture

```python
# In selfplay_generator.py

def execute_mcp_tools(self, response_text: str) -> Tuple[bool, Optional[str]]:
    """Execute tool calls against MCP server."""

    # 1. Extract tool calls from response
    tool_calls = extract_tool_calls(response_text)

    # 2. For each tool call:
    for tool_name, arguments in tool_calls:
        # 3. Format for MCP
        mcp_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": generate_request_id()
        }

        # 4. Send to MCP server
        response = requests.post(
            self.mcp_server_url,
            json=mcp_request,
            headers={"Content-Type": "application/json"}
        )

        # 5. Parse response
        result = response.json()

        # 6. Check for errors
        if "error" in result:
            return False, result["error"]["message"]

    return True, None
```

### Setup for MCP

1. **Install Obsidian MCP server**
   ```bash
   # Clone Claudesidian-MCP repo
   git clone https://github.com/yourusername/Claudesidian-MCP.git

   # Follow setup instructions
   cd Claudesidian-MCP
   npm install
   npm run build
   ```

2. **Create test vault**
   ```bash
   # Create a clean test vault
   mkdir -p ~/obsidian-test-vault

   # Initialize with basic structure
   mkdir -p ~/obsidian-test-vault/{Notes,Projects,Archive}
   ```

3. **Start MCP server**
   ```bash
   # Start server pointing to test vault
   cd Claudesidian-MCP
   npm start -- --vault ~/obsidian-test-vault
   ```

4. **Run self-play with MCP**
   ```bash
   python Tools/selfplay_generator.py \
     --model claudesidian-mcp \
     --prompt-set Evaluator/prompts/tool_prompts.json \
     --output Datasets/syngen_selfplay_mcp.jsonl \
     --num-examples 1000 \
     --execute-mcp \
     --vault-path ~/obsidian-test-vault
   ```

### Vault Reset Between Runs

To avoid state accumulation:

```python
# In mcp_executor.py

def reset_vault(self) -> bool:
    """Reset vault to clean state."""
    # 1. Delete all created notes
    # 2. Delete all created folders
    # 3. Clear workspace state
    # 4. Reset session memory

    # Implementation depends on MCP API
    # May need to track created resources during execution
```

## Troubleshooting

### LM Studio connection errors

```bash
# Check LM Studio is running
curl http://localhost:1234/v1/models

# If using WSL, check Windows IP
ip route | grep default
# Use that IP as LMSTUDIO_HOST
```

### Validation failures

```bash
# Run validator directly to debug
python tools/validate_syngen.py Datasets/syngen_selfplay_test.jsonl
```

### Low valid example rate

If you're getting mostly invalid examples:

1. **Check base model quality** - Is it already trained on tools?
2. **Reduce temperature** - Try 0.3-0.5 for higher quality
3. **Check prompts** - Are they clear and well-formed?
4. **Inspect failures** - What validation errors are most common?

### Dataset not interleaving

The generator automatically interleaves, but if you're manually combining datasets:

```bash
# Use the interleave script (if we create one)
python Tools/interleave_dataset.py \
  --input Datasets/raw_selfplay.jsonl \
  --output Datasets/interleaved_selfplay.jsonl
```

## Best Practices

### 1. Start with small batches

Generate 100 examples first to check quality:

```bash
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/test_selfplay.jsonl \
  --num-examples 100
```

Then inspect:
```bash
# Check validation stats
python tools/validate_syngen.py Datasets/test_selfplay.jsonl

# Look at examples
head -5 Datasets/test_selfplay.jsonl | jq
```

### 2. Use temperature strategically

- **Low (0.3-0.5)**: Mostly correct, less diversity
- **Medium (0.6-0.8)**: Balanced mix
- **High (0.9-1.2)**: Maximum diversity, many errors

### 3. Balance your dataset

The generator auto-balances, but verify:

```bash
# Count labels
grep -o '"label": true' Datasets/syngen_selfplay.jsonl | wc -l
grep -o '"label": false' Datasets/syngen_selfplay.jsonl | wc -l
```

Should be equal (or within 1-2 examples).

### 4. Monitor model quality over iterations

Track metrics across iterations:

- **Syntax validation rate** (from selfplay_generator)
- **Tool selection accuracy** (from Evaluator)
- **Context retention** (from Evaluator)
- **Hallucination rate** (from Evaluator)

If quality degrades, stop the loop or increase diversity.

### 5. Mix with human-curated data

Don't rely 100% on self-play:

```bash
# Combine self-play with original dataset
cat Datasets/syngen_tools_11.18.25.jsonl \
    Datasets/syngen_selfplay_20251204.jsonl \
    > Datasets/combined_training.jsonl

# Shuffle and re-interleave
python Tools/shuffle_and_interleave.py \
  --input Datasets/combined_training.jsonl \
  --output Datasets/combined_interleaved.jsonl
```

## Next Steps

1. **Run basic generation** (validation only)
2. **Inspect output quality**
3. **Train with generated data**
4. **Evaluate improvements**
5. **Integrate MCP execution** (when ready)
6. **Set up self-improvement loop**

## References

- [KTO Training Reference](../KTO_TRAINING_REFERENCE.md)
- [Dataset Validation](../tools/validate_syngen.py)
- [Evaluation Guide](../Evaluator/README.md)
- [CLAUDE.md](../CLAUDE.md)
