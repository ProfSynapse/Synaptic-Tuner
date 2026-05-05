# Dataset Formats Reference

Dataset requirements for configurable SFT, KTO, and GRPO training data. The pipeline is format-agnostic: wrapper names, argument fields, command strings, and tool schemas come from dataset/config, not from trainer code or generic skill guidance.

---

## SFT Dataset Format

Positive examples only. Tool-calling examples can use OpenAI-style `tool_calls`, text tool blocks, or another configured transport, as long as preprocessing, evaluation, and runtime config all agree on the schema.

```jsonl
{
  "conversations": [
    {"role": "system", "content": "<runtime_context>...</runtime_context>"},
    {"role": "user", "content": "Use the available tools to complete the requested operation."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "CONFIGURED_WRAPPER_NAME",
            "arguments": "{\"FIELD_A\":\"value-a\",\"FIELD_B\":\"value-b\",\"ACTION_FIELD\":\"configured action payload\"}"
          }
        }
      ]
    }
  ]
}
```

Key rules:
- Assistant tool-calling turns must match the active configured response format.
- Wrapper names, context fields, command fields, and action syntax belong in the dataset/config for the run.
- For tool-shape SFT, keep the system prompt lean and aligned with the target configured schema. Remove stale discovery flows, obsolete field instructions, route tables, and tool names from other schemas.
- If individual datasets need a system-prompt refresh, update the individual source files first with a bumped version, spot-check that non-system turns are unchanged, then merge the latest individual versions.

---

## KTO Dataset Format

Interleaved desirable and undesirable examples.

```jsonl
{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"good response"}],"label":true}
{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"bad response"}],"label":false}
```

Key rules:
- `label: true` = desirable
- `label: false` = undesirable
- Keep paired positive/negative coverage where practical

---

## GRPO Dataset Format

Prompts plus ground-truth metadata for reward scoring.

```jsonl
{
  "prompt": [
    {"role": "system", "content": "<runtime_context>...</runtime_context>"},
    {"role": "user", "content": "Use the available tools to complete the requested operation."}
  ],
  "ground_truth_tool": "CONFIGURED_WRAPPER_NAME",
  "ground_truth_args_json": "{\"FIELD_A\":\"value-a\",\"FIELD_B\":\"value-b\",\"ACTION_FIELD\":\"configured action payload\"}"
}
```

Ground-truth fields are examples only. The active GRPO config decides which columns exist, how prompts are built, and how rewards compare generated responses to expected behavior.

---

## Configured Tool Wrapper

The tool-call format is not global. It is defined by the dataset/config for the run. A typical OpenAI-style wrapper looks like:

```json
{
  "tool_calls": [
    {
      "id": "call_0001",
      "type": "function",
      "function": {
        "name": "CONFIGURED_WRAPPER_NAME",
        "arguments": "{\"FIELD_A\":\"value-a\",\"FIELD_B\":\"value-b\",\"ACTION_FIELD\":\"configured action payload\"}"
      }
    }
  ],
  "content": null
}
```

Required and optional fields should be documented in the run's schema/config and validated there. If the schema uses command strings, keep command examples in schema/scenario files rather than in generic skill guidance.

---

## Dataset Locations

Use the dataset folders configured for the current run. For tool-calling corpora, keep individual source datasets versioned separately, then merge them into the SFT input only after validation and spot checks.

Record the active source versions in merged dataset metadata so later SFT, eval, and GRPO runs can be traced back to exact inputs.

---

## Validation

```bash
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/my_dataset.jsonl
```

Use checked-in migration/generation pipelines for corpus refreshes instead of ad hoc rewriting:

```bash
python3 Tools/migrations/05_inventory_cli_schema_datasets.py
python3 Tools/migrations/06_migrate_cli_schema_datasets.py
```

Apply an SFT system-prompt profile to individual datasets before merging:

```bash
python3 Tools/migrations/09_align_sft_system_prompts.py \
  --profile Datasets/tools_datasets/system_prompt_profiles/your_profile.json
python3 Datasets/tools/merge_nonthinking_datasets.py --date MM.DD.YY
```

When rerunning an alignment against a specific previous version, pass explicit `--source name=path/to/file.jsonl` overrides so the script overwrites the intended bumped outputs rather than bumping newly-created files again.

Validation may flag legacy generated IDs or fields when older datasets use formats that differ from the active schema. Treat those as schema-policy findings, not prompt-alignment failures, and confirm with a separate audit that assistant tool calls still parse and include required configured fields.
