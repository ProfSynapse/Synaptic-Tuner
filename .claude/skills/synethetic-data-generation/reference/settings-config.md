# Settings Configuration Reference

**File:** `SynthChat/config/settings.yaml`

---

## LLM Configuration

Two separate LLM configs — generation (raw examples) and improvement (judge/improve loop).

```yaml
llm:
  # Generation model — creates raw examples. Local models work fine.
  generation:
    provider: openrouter          # openrouter | lmstudio | ollama | unsloth
    model: MODEL_ID
    temperature: 0.7              # Higher = more variety in generated examples
    max_tokens: 4096
    provider_routing:
      order: ["PROVIDER_NAME"]    # Optional hosted-provider preference
      allow_fallbacks: true       # Fall back if preferred unavailable
    # LM Studio example:
    # provider: lmstudio
    # model: local-model
    # host: localhost
    # port: 1234
    #
    # Unsloth example (local LoRA):
    # provider: unsloth
    # model: /path/to/lora_adapter
    # max_seq_length: 4096
    # load_in_4bit: true
    # top_p: 0.9

  # Improvement model — judge + improver. Needs strong/capable model.
  improvement:
    provider: openrouter
    model: MODEL_ID
    temperature: 0.1              # Low = deterministic judging
    max_tokens: 2048
    provider_routing:             # OpenRouter-specific
      order: ["PROVIDER_NAME"]   # Optional hosted-provider preference
      allow_fallbacks: true      # Fall back if preferred unavailable
```

**Providers:** `openrouter`, `lmstudio`, `ollama`, `unsloth`

CLI flags `--provider` and `--model` override these at runtime.

---

## Improvement Settings

```yaml
improvement:
  max_iterations: 10              # Max judge->improve loops per example
  on_max_iterations: skip         # skip | fail | save_partial
  default_rubrics:                # Applied when CLI doesn't specify --rubrics
    - system_prompt_format
    - thinking_quality
    - tool_alignment
    - response_quality
```

---

## Output Settings

All generated datasets should be saved to `Datasets/synthchat/`.

```yaml
output:
  default_dir: Datasets/synthchat/    # Where generated datasets go
  include_metadata: true               # Add _meta header line to JSONL
  versioning: datetime                 # Append timestamp to filenames
```

---

## Resilience & Checkpointing

```yaml
resilience:
  checkpoint_every: 10            # Save progress every N examples
  retry_attempts: 3               # Retry on LLM failures
  retry_delay: 5                  # Seconds between retries
  checkpoint_file: .synthchat_checkpoint.json
```

---

## Logging

```yaml
logging:
  save_interactions: true                   # Log judge/improve exchanges
  interactions_dir: SynthChat/interactions/ # Where logs go
  save_failures: true                       # Keep failed examples (KTO negatives)
  log_level: INFO
```

---

## Generation Settings

```yaml
generation:
  random_seed: null               # Set for reproducibility, null for random
  stage_validation: true          # Validate each stage before proceeding

cost_tracking:
  enabled: true                   # Track token usage (OpenRouter)
  include_usage: true
```

## Privacy Preprocess Settings

Privacy preprocessing is opt-in. It is intended for sanitizing raw docs or JSONL before that content is shown to a generation/improvement model.

```yaml
privacy_preprocess:
  enabled: false
  profile: null
  apply_to:
    docs: true
    input_jsonl: false
    generated_output: false
  on_error: fail                  # fail | skip | keep_masked
```

Profiles are defined separately in `SynthChat/config/privacy_profiles.yaml`.

Current built-in profiles:
- `mask_only`
- `realistic_pseudonyms`
- `realistic_pseudonyms_vllm_polish`

Operational notes:
- OPF is the detector runtime, not a chat/completions LLM
- `generated_output` is optional and should stay off unless you explicitly want output-side sanitization
- `input_jsonl` applies to `improve` and `validate`
- `docs` applies to docs-based generation before `{doc_content}` is rendered into prompts
- OPF needs both a local checkpoint path (`OPF_CHECKPOINT`) and a working `tiktoken` cache (`TIKTOKEN_CACHE_DIR`) if auto-download is not reliable

---

## Environment Runtime Validation (Optional)

Enable runtime-backed execution checks during generation.

```yaml
environment:
  enabled: false                  # true to enable by default
  backend: local                  # local | e2b
  template: null                  # E2B template id if backend=e2b
  timeout_seconds: 120.0
  tool_schema_path: null          # Optional custom tool schema YAML
  execution_config_path: null     # Optional custom runtime rules YAML
```

Runtime rules are config-driven through `Evaluator/config/environment_execution.yaml`
(or a custom file via `execution_config_path` / `--env-exec-config`).
This lets any project map its own tool names and argument keys without code changes.

---

## Default Generation Targets

Defines how many examples to generate per scenario when no `--targets-file` is given.

```yaml
defaults:
  targets:
    # Tool scenarios
    workspace_create_folder: 50
    workspace_archive_file: 30
    workspace_move_file: 30
    workspace_list_folder: 40
    workspace_write_note: 40
    workspace_replace_text: 20
    workspace_set_property: 10
    workspace_execute_prompt: 25

    # Content writing
    write_research_note: 50
    write_blog_note: 50
    write_creative_note: 40
    write_documentation_note: 50
    write_journal_note: 40

    # Behavioral
    intellectual_humility: 25
    verification_before_action: 25
    error_recovery: 20
    context_continuity: 20
    strategic_tool_selection: 20
```

For repeatable smoke tests, prefer checked-in target manifests rather than editing `defaults.targets` or passing inline JSON. Current examples in this repo include:

- `SynthChat/config/targets_cli_existing_tools_quickcheck.json`
- `SynthChat/config/targets_cli_existing_tools_representative.json`

---

## Validation Configuration

**File:** `SynthChat/config/validation.yaml`

Controls how the improvement engine extracts content scopes and processes them.

### Scope Definitions

```yaml
scopes:
  system_prompt:
    extraction: { method: role_based, role: system }

  user:
    extraction: { method: role_based, role: user }

  thinking:
    extraction:
      method: pattern
      pattern: "<thinking>(.*?)</thinking>"
    markers: { start: "<thinking>", end: "</thinking>" }
    format: { primary: json, fallback: yaml }

  tool_calls:
    extraction:
      method: pattern
      patterns:
        - name_pattern: "(tool_name|tool_call):\\s*(.+)"
          args_pattern: "arguments:\\s*\\n(.+?)(?=\\n\\S|\\Z)"

  response:
    extraction:
      method: exclusion
      exclude: [thinking, tool_calls]
```

### Processing Order & LLM Settings

```yaml
scope_processing_order:
  - system_prompt
  - user
  - thinking
  - response

llm:
  improvement_temperature: 0.3
  judge_temperature: 0.0
  improvement_max_tokens: 2000
  judge_max_tokens: 1000
```
