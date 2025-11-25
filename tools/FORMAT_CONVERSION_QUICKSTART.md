# Dataset Format Conversion - Quick Start

## TL;DR

Your current format is **FINE**. You don't need to convert unless you want MCP/Anthropic API compatibility.

## Quick Comparison

### Your Current Format
```
tool_call: toolName
arguments: {...}
```
- ✅ Simpler for models
- ✅ Works with your pipeline
- ✅ Good for local training

### MCP Format (What criticism suggests)
```json
{
  "type": "tool_use",
  "id": "toolu_xyz",
  "name": "toolName",
  "input": {...}
}
```
- 📊 Industry standard
- 📊 Better compatibility
- 📊 More complex

## One-Line Decision

```bash
# Are you using local models via Ollama/LM Studio for your own use?
# → YES: Keep current format (do nothing)

# Do you need compatibility with external MCP tools or Anthropic API?
# → YES: Convert to MCP format (use script below)
```

## Conversion Commands

### Preview First (Dry Run)
```bash
python tools/convert_to_mcp_format.py \
  Datasets/syngen_tools_sft_11.22.25.jsonl \
  /tmp/preview.jsonl \
  --dry-run
```

### Convert SFT Dataset
```bash
python tools/convert_to_mcp_format.py \
  Datasets/syngen_tools_sft_11.22.25.jsonl \
  Datasets/syngen_tools_sft_11.22.25_mcp.jsonl \
  --validate
```

### Train on Converted Dataset
```bash
cd Trainers/rtx3090_sft
./train.sh --model-size 7b --local-file ../../Datasets/syngen_tools_sft_11.22.25_mcp.jsonl
```

## What Changes After Conversion

**Dataset:**
- ✅ Converted automatically by script

**You Need to Update:**
- ⚠️ `tools/validate_syngen.py` - Update parser to handle JSON blocks
- ⚠️ `Evaluator/schema_validator.py` - Update to extract tool_use blocks
- ⚠️ Inference parsing - Update post-processing logic

**Effort:** ~2-4 hours + re-training time

## My Recommendation

**99% of users:** Keep your current format. It's simpler and works perfectly.

**1% edge case:** You specifically need MCP ecosystem compatibility → Convert.

## Example Before/After

**Before:**
```json
{
  "role": "assistant",
  "content": "tool_call: contentManager_createContent\narguments: {\"filePath\": \"note.md\", ...}"
}
```

**After:**
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_UBW0epbeyQJgo1lsqdrgKoWu",
      "name": "contentManager_createContent",
      "input": {
        "filePath": "note.md",
        ...
      }
    }
  ]
}
```

## Support

- Full documentation: `docs/DATASET_FORMAT_ANALYSIS.md`
- Script: `tools/convert_to_mcp_format.py`
- Help: `python tools/convert_to_mcp_format.py --help`

## Bottom Line

The criticism is technically correct about **Anthropic API standards**, but doesn't apply to your **local training use case**. Your format is fine.

Only convert if you have **specific compatibility requirements** that you can clearly articulate.
