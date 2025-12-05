# SelfPlay Dataset Splitting

## Overview

The SelfPlay generation system now automatically organizes generated examples into the appropriate dataset folders with proper versioning and timestamps.

## File Structure

### Tools
Generated tool examples are placed in:
```
Datasets/tools_datasets/{agent_name}/tools_v{X.X}_{YYYYMMDD_HHMMSS}.jsonl
```

**Example:**
```
Datasets/tools_datasets/vaultManager/tools_v1.4_20251204_160000.jsonl
Datasets/tools_datasets/contentManager/tools_v1.2_20251204_160000.jsonl
Datasets/tools_datasets/memoryManager/tools_v1.3_20251204_160000.jsonl
```

### Behaviors
Generated behavior examples are placed in:
```
Datasets/behavior_datasets/{behavior_name}/pairs_v{X.X}_{YYYYMMDD_HHMMSS}.jsonl
```

**Example:**
```
Datasets/behavior_datasets/intellectual_humility/pairs_v1.6_20251204_160000.jsonl
Datasets/behavior_datasets/error_recovery/pairs_v1.3_20251204_160000.jsonl
Datasets/behavior_datasets/workspace_awareness/pairs_v1.5_20251204_160000.jsonl
```

## Version Numbering

The script automatically:
1. Scans the target directory for existing files
2. Finds the highest version number (e.g., `v1.5`)
3. Increments the minor version (e.g., `v1.6`)
4. Adds timestamp for uniqueness

## Agent Detection

Tool names follow the pattern `{agent}_{toolName}`. The script automatically:
- Extracts the agent name from the tool (e.g., `vaultManager_createFolder` → `vaultManager`)
- Places the file in the correct agent folder

**Agent mappings:**
- `vaultManager_*` → `Datasets/tools_datasets/vaultManager/`
- `contentManager_*` → `Datasets/tools_datasets/contentManager/`
- `memoryManager_*` → `Datasets/tools_datasets/memoryManager/`
- `vaultLibrarian_*` → `Datasets/tools_datasets/vaultLibrarian/`
- `agentManager_*` → `Datasets/tools_datasets/agentManager/`

## Behavior Detection

Behaviors are identified by name. Known behaviors:
- `intellectual_humility`
- `error_recovery`
- `verification_before_action`
- `workspace_awareness`
- `strategic_tool_selection`
- `ask_first`
- `context_continuity`
- `context_efficiency`
- `execute_prompt_usage`
- `response_patterns`

## Usage

### Via CLI (Automatic)

When using the unified CLI, splitting happens automatically:

```bash
python tuner.py generate
# Select "Targeted generation"
# Generate examples
# Answer "Yes" to split prompt
```

### Via Script (Manual)

```bash
python Tools/split_selfplay_dataset.py \
    SelfPlay/selfplay_output.jsonl \
    --datasets-dir Datasets
```

**Options:**
- `--datasets-dir`: Base Datasets directory (default: `Datasets`)
- `--dry-run`: Show what would be done without creating files

### Example Output

```
======================================================================
SELFPLAY DATASET SPLITTER
======================================================================

Input file: SelfPlay/selfplay_output.jsonl
Datasets directory: Datasets
Timestamp: 20251204_160000
Dry run: No

📖 Loading examples from SelfPlay/selfplay_output.jsonl...
   Loaded 150 examples

📊 Grouping by category...
   Found 3 categories:
     - vaultManager_createFolder (tool): 50 examples
     - contentManager_createContent (tool): 50 examples
     - intellectual_humility (behavior): 50 examples

💾 Writing category files...
   ✨ Created tools_v1.4_20251204_160000.jsonl (tool, 50 examples)
   ✨ Created tools_v1.2_20251204_160000.jsonl (tool, 50 examples)
   ✨ Created pairs_v1.6_20251204_160000.jsonl (behavior, 50 examples)

======================================================================
✅ SPLITTING COMPLETE
======================================================================

📊 Statistics:
   Files created: 3
     - Tool datasets: 2
     - Behavior datasets: 1
   Total examples: 150

📁 Base directory: Datasets
   Tools: Datasets/tools_datasets
   Behaviors: Datasets/behavior_datasets
```

## Integration with Existing Datasets

The versioned files integrate seamlessly with existing datasets:

### Before SelfPlay
```
Datasets/tools_datasets/vaultManager/
├── tools_v1.0.jsonl
├── tools_v1.1.jsonl
└── tools_v1.2.jsonl
```

### After SelfPlay
```
Datasets/tools_datasets/vaultManager/
├── tools_v1.0.jsonl
├── tools_v1.1.jsonl
├── tools_v1.2.jsonl
└── tools_v1.3_20251204_160000.jsonl  ← New!
```

## Benefits

1. **Automatic versioning** - No manual version tracking needed
2. **Timestamp uniqueness** - Multiple generations per day don't conflict
3. **Organized structure** - Tools and behaviors in proper folders
4. **Incremental growth** - New data appends to existing collections
5. **Traceability** - Timestamp shows when data was generated
6. **Merge-ready** - Compatible with existing merge scripts

## Notes

- Version numbers are per-folder (each agent/behavior has independent versioning)
- Timestamps use local time in `YYYYMMDD_HHMMSS` format
- Files are append-only (never overwrites existing versions)
- Empty folders are created automatically if needed
