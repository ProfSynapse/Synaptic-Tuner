# Testing Protocol

**MANDATORY**: After creating or modifying any scenario or rubric YAML, you MUST follow this protocol before running full generation. Never go straight from YAML authoring to a large run.

---

## The Rule

```
Write/edit YAML → Dry-run 3-5 examples → Show user → Get approval → Full run
```

No exceptions. Even "small" YAML changes can produce thousands of bad examples if not tested first.

For normal scenario development in this repo, a dry-run should exercise the same
quality gates you expect in the real dataset:
- stage rubrics enabled
- judge/final_judge enabled where the scenario design calls for them
- environment validation enabled for tool scenarios when practical
- environment/runtime errors fed back into the judge/improver path when the
  scenario relies on environment-backed validation

If you are running a smoke test without rubrics/judges, call that out explicitly
as a plumbing-only test. Do not mistake a wrapper-format smoke pass for a
quality pass.

For environment-backed multi-turn scenarios, test in this order:
1. Run `env-generate` for the scenario and inspect stage gates before spending
   calls on full rollouts.
2. If the scenario uses stage-specific models, verify the env-generation log
   shows the configured fixture-authoring model, then verify the full rollout
   log shows the intended assistant/turn-generation model.
3. Run a one-scenario smoke from a checked-in targets manifest.
4. Only then run a multi-scenario smoke or dataset pilot.

When a smoke might take more than a minute, launch it in the background with
stdout, stderr, JSONL, and debug JSONL paths. Poll the logs early. If the first
30-60 seconds already show a deterministic gate failure or provider timeout,
stop and fix the config before burning retries.

For privacy preprocessing changes, the first smoke is usually `sanitize`, not a
full dataset run. Prove the OPF checkpoint, tokenizer cache, and replacement
behavior on the checked-in privacy fixtures before you run a larger SynthChat
generation or improvement pass.

---

## Iterative Scaling Ladder

For config-driven environment-backed datasets, use this ladder. Do not jump
from scenario authoring to a large generation run.

1. Author declarative config only: scenario YAML, rubric YAML, tool schema,
   execution config, target manifest, and settings. If a failure is tied to the
   current tool surface or dataset shape, first fix the config, not runtime code.
2. Run an environment-only probe for a few seeds with debug artifacts enabled.
   Confirm the generated fixture is valid, realistic, diverse enough, and
   internally consistent before spending calls on assistant rollouts.
3. Run a tiny full rollout smoke from a checked-in target manifest. Keep the
   worker count low until raw traces show the assistant, tools, judge feedback,
   and final answer are all moving through the expected loop.
4. Analyze the raw JSONL, debug JSONL, stdout, and stderr. Check the generated
   environment, model-facing tool responses, assistant tool calls, judge notes,
   final text, and final assertions. Do not rely only on aggregate pass counts.
   Accepted rows can still be poor training data if they contain recoverable
   syntax errors, unexpected tools, repeated judge repairs, or a different
   command shape from the configured expected trajectory.
5. Make config-only fixes: prompts, schemas, gates, rubrics, target manifests,
   retry counts, worker counts, model routing, or temperature ranges. Runtime
   code changes are only for reusable capabilities that cannot be expressed by
   existing config surfaces.
6. Repeat the environment probe and tiny rollout after each meaningful fix.
7. Scale in stages, for example 3x1 -> 10x2 -> 25x2 -> 50x2. Increase workers
   only after the smaller stage is clean enough to justify more concurrency.
8. For large stages, tolerate a small failed fraction when failures are
   non-systemic. Create a passed-only artifact, then run a make-up slice if the
   target count matters.
9. Record the artifact paths, pass counts, failure categories, and any config
   changes so the next run starts from evidence rather than memory.

When analyzing staged runs, collect at least:
- row count and pass/fail count
- environment pass/fail status
- final text/final judge pass status
- tool sequence distribution and turn count distribution
- environment issue counts, recoverable tool error counts, and unexpected-tool
  counts in accepted rows
- stage review failures by stage
- provider warnings, empty responses, schema errors, and retry counts
- diversity proxies such as unique user prompts, workspace/tool IDs, file paths,
  answer phrases, domains, or task flavors

Classify failures before fixing them:
- Environment structural failure: fix schema, fixture gates, or env-generation
  prompt constraints.
- Fixture semantic mismatch: fix env-generation prompt, environment assertions,
  or env judge wording so the generated files and expected answer agree.
- Assistant rollout/tool-order failure: fix the model-facing system prompt,
  expected trajectory config, response rubric, or in-loop judge feedback.
- Final text failure: fix the terminal answer instruction, final gates, or
  final judge. Multi-turn tool tasks should normally end with a text-only
  assistant answer after tool use is complete.
- Provider or structured-output instability: adjust retries, worker count,
  stage-specific model routing, or temperature ranges. Do not tune scenario
  semantics unless the failure is repeatable.
- Brittle structured CLI arguments: gate the exact expected command shape in
  scenario YAML, verify the executor accepts it, and smoke with the intended
  rollout model. If that model repeatedly produces malformed nested JSON or
  quoting, use scenario-level `assistant_generation` routing for a stronger
  teacher rather than accepting noisy recovery traces as the training source.

If a failure can be checked deterministically from metadata, add a final gate
instead of relying only on an LLM judge. This is especially important for
multi-step tool trajectories where the judge may accept the final state even
though the assistant skipped a required discovery/read step. Use generic gates
whose behavior is supplied by config fields and templates, such as comparing
scenario-defined expected actions with recorded environment actions using
scenario-defined renderers. Keep the gate
configuration declarative; do not add scenario-specific parser or executor
repairs.

It is often useful to test different model/provider choices for environment
generation, assistant turn generation, and judging. Treat model routing as
configuration, and verify in logs that each stage uses the intended route.

---

## Protocol Steps

### Step 1: Dry-Run (3-5 Examples)

After writing or modifying a scenario/rubric, generate a small sample:

```bash
# Dry-run a specific scenario (3 examples)
python -m SynthChat.run generate \
  --scenarios YOUR_SCENARIO_KEY \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_YOUR_SCENARIO.jsonl \
  --targets-file <(echo '{"YOUR_SCENARIO_KEY": 3}')
```

Or use the dry-run helper script:

```bash
.claude/skills/synethetic-data-generation/scripts/dry_run.sh YOUR_SCENARIO_KEY [count]
```

For repeatable multi-scenario smoke tests, prefer a checked-in targets manifest:

```bash
python -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```

**For rubric changes** (testing against existing data):

```bash
# Validate 5 lines with the modified rubric
python -m SynthChat.run validate \
  --input YOUR_DATASET.jsonl \
  --rubrics YOUR_RUBRIC_NAME \
  --start-line 1 --end-line 5
```

**For privacy preprocessing changes** (testing the sanitization path itself):

```powershell
$env:OPF_CHECKPOINT="F:\Code\Toolset-Training\tmp\opf_privacy_filter"
$env:TIKTOKEN_CACHE_DIR="F:\Code\Toolset-Training\tmp\tiktoken_cache"

python -m SynthChat.run sanitize \
  --input tests/fixtures/privacy/raw_seed_docs \
  --output tmp/privacy_mask_only_docs \
  --privacy-profile mask_only

python -m SynthChat.run sanitize \
  --input tests/fixtures/privacy/raw_seed_docs \
  --output tmp/privacy_pseudonyms_docs \
  --privacy-profile realistic_pseudonyms
```

Then run a small docs-based generation smoke:

```powershell
python -m SynthChat.run generate \
  --docs tests/fixtures/privacy/raw_seed_docs \
  --targets-file SynthChat/config/targets_privacy_docs_smoke.json \
  --privacy-profile realistic_pseudonyms \
  --max-iterations 1 \
  --output Datasets/synthchat/privacy_docs_smoke.jsonl
```

### Step 2: Convert to Markdown for Review

Convert the dry-run JSONL to readable Markdown so the user can easily review:

```bash
# Convert to Markdown (creates dryrun_YOUR_SCENARIO_review.md next to the JSONL)
.claude/skills/synethetic-data-generation/scripts/jsonl_to_markdown.sh \
  Datasets/synthchat/dryrun_YOUR_SCENARIO.jsonl
```

This produces a clean Markdown file with:
- Each example as a numbered section
- System prompts collapsed in `<details>` blocks
- Thinking blocks pretty-printed as JSON
- Tool calls formatted with function name + arguments
- Labels shown as ✅ positive / ❌ negative

**What to check:**
- Does the system prompt have all required sections?
- Is the user request natural and varied?
- Does the assistant response use the right tool/behavior?
- Are thinking blocks well-structured (if applicable)?
- Do tool calls have correct parameters?
- Are configured context IDs/fields consistent?
- Is content substantive (not generic filler)?
- If rubrics/judges are configured, did they actually run and gate the example?
- If environment validation ran, were its errors visible to the judge/improver
  and reflected in the saved pass/fail outcome?
- If environment generation failed, distinguish deterministic gate failures
  from LLM judge hallucinations. Trust schema/placeholder/min-fixture gates for
  structural validity and narrow or disable an LLM env judge that contradicts
  normalized JSON.
- If the generated environment is valid but the rollout fails, do not keep
  tuning environment prompts. Inspect the assistant turns, tool feedback, and
  in-loop judge feedback to isolate whether the rollout model missed a tool
  step, the judge pushed after completion, or final text handling was too
  strict.
- If expected action sequences contain stale surfaces such as shell commands
  while the trained surface is a configured tool schema, add scenario gates
  that reject those generated answer keys before the rollout.
- If an agentic rollout failed after recovery, inspect whether final assertions
  passed. For recovery datasets, earlier recoverable tool errors can be useful
  training signal when later actions corrected them.
- If a scenario unexpectedly skips rubrics/judges, stop and fix the scenario
  config before trusting the output.
- For multi-turn tool scenarios, confirm the saved conversation represents the
  full episode rather than a single-response repair artifact.
- Confirm tool feedback shown to the model uses the configured model-facing
  tool names, not unrelated executor or implementation identifiers.
- Confirm successful in-loop judge feedback does not cause another assistant
  turn after a correct final text answer.

The `jsonl_to_markdown.sh` script also supports line ranges for reviewing subsets of larger files:

```bash
# Convert only lines 5-10
.claude/skills/synethetic-data-generation/scripts/jsonl_to_markdown.sh \
  data.jsonl review.md --start 5 --end 10
```

### Step 3: Present to User

Show the user 2-3 representative examples and ask:

> "Here are sample outputs from the new [scenario/rubric]. Do these look right, or should I adjust the YAML?"

**Present concisely** — show the key parts (user request, assistant response, tool calls) not raw JSON walls.

### Step 4: Iterate if Needed

If user has feedback:
1. Adjust the YAML
2. Re-run dry-run (Step 1)
3. Show updated samples (Step 2-3)
4. Repeat until approved

### Step 5: Full Generation (Only After Approval)

Once user approves:

```bash
# Full run with appropriate worker count
python -m SynthChat.run generate \
  --scenarios YOUR_SCENARIO_KEY \
  --workers 4 \
  --output Datasets/synthchat/YOUR_SCENARIO_TIMESTAMP.jsonl
```

---

## Dry-Run Helper Script

**File:** `.claude/skills/synethetic-data-generation/scripts/dry_run.sh`

Quick dry-run for testing new or modified scenarios:

```bash
# Usage: ./dry_run.sh <scenario_key> [count] [extra_args...]
# Default count: 3

./scripts/dry_run.sh workspace_create_folder              # 3 examples
./scripts/dry_run.sh workspace_write_note 5               # 5 examples
./scripts/dry_run.sh essay_outline 2 --docs "essays/"     # 2 examples, docs-based
```

---

## When to Dry-Run

| Change | Dry-Run? | Why |
|--------|----------|-----|
| New scenario YAML | YES | Untested template, could produce garbage |
| Modified scenario prompts | YES | Prompt changes affect all generated examples |
| Added/removed scenario rubrics or judge config | YES | Changes whether quality gates actually run |
| New rubric YAML | YES (validate mode) | Untested judge/improver, could reject good data |
| Modified rubric judge/improver | YES (validate mode) | Scoring may shift, threshold may need adjustment |
| Changed `pass_threshold` | YES (validate mode) | Could mass-pass or mass-fail |
| Changed settings.yaml model/provider | YES | Different model = different output quality |
| Changed privacy preprocess config/profile/runtime | YES | Could leak raw content or break replacements |
| Changed settings.yaml targets only | NO | Just counts, doesn't affect content |
| Changed settings.yaml logging/resilience | NO | Infrastructure, not content |

---

## Validation-Focused Dry-Run

When testing rubric changes against existing data:

```bash
# 1. Validate a small sample
python -m SynthChat.run validate \
  --input Datasets/existing_dataset.jsonl \
  --rubrics YOUR_RUBRIC \
  --start-line 1 --end-line 10

# 2. Check pass/fail rate — does it match expectations?
# Too many failures? → threshold too high or judge too strict
# Everything passes? → threshold too low or judge too lenient

# 3. Try improving a few failures to verify the improver works
python -m SynthChat.run improve \
  --input Datasets/existing_dataset.jsonl \
  --rubrics YOUR_RUBRIC \
  --start-line FAILING_LINE --end-line FAILING_LINE \
  --max-iterations 3
```

---

## Raw Artifact Inspection

When an environment-backed generation fails, inspect raw inputs and outputs
before changing scenarios or rubrics. The important surfaces are:

- generated JSONL row
- `conversations`
- `conversation_trace`
- `metadata.environment`
- `metadata.environment.episode_trace`
- `metadata.stage_reviews`
- latest `SynthChat/interactions/*.jsonl`

Use this checklist:

- Did the rendered system prompt include stale or wrong tool-format prose?
- Did the assistant emit valid structured output before environment execution?
- If the first assistant turn was malformed, did validation feedback stay
  inside the agentic loop and allow a corrected retry?
- Did the environment execute the expected actions?
- If expected actions are stored in task context, did a deterministic final
  gate verify action coverage and order against the executed tool trace?
- Did tool feedback expose the same model-facing tool names as the prompt?
- Did the in-loop judge ask for a correction only when the latest assistant
  action actually failed?
- Did final gates pass while final judge failed because of an extra or malformed
  terminal assistant turn?
- After a final-text request, did the judge accept a grounded text-only answer
  instead of asking for another search/read just to improve wording?
- Did response repair collapse a multi-turn trajectory into one assistant
  message?
- Did post-loop response improvement rewrite or append an assistant message
  after the environment-backed episode already succeeded?
- Are text-only terminal answers accepted as final text even if the structured
  generator includes an empty `tool_calls` field?
- If the dataset is for environment-reward GRPO, does the loop stop once the
  environment passes instead of spending extra turns on brittle final text?

Minimal inspection command:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import json,pathlib; p=pathlib.Path(r'PATH_TO_DRYRUN.jsonl'); \
for i,l in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1): \
    row=json.loads(l); \
    print('\\nLINE', i, row.get('metadata',{}).get('scenario')); \
    env=row.get('metadata',{}).get('environment',{}); \
    print('env', env.get('passed'), 'stop', (env.get('episode_trace') or {}).get('stop_reason')); \
    print('issues', [x.get('message') for x in env.get('issues',[])]); \
    print('tools', [(t.get('name'), t.get('status')) for t in env.get('executed_tools',[])]); \
    print('roles', [(m.get('role'), bool(m.get('tool_calls'))) for m in row.get('conversations',[])])"
```

If the interaction log contains implementation-only tool names that the model
should never see, fix the configured model-facing feedback surface before
tuning prompts.

When validating generated tool/eval data against a trained model, separate
runtime health from model behavior:

- First confirm the serving process loaded and wrote a usable progress or
  summary artifact.
- Treat nonzero eval exits as expected when examples fail; preserve and inspect
  result artifacts before changing scenarios or runtime.
- If every case fails with the same infrastructure warning or token-budget
  error, fix the eval/job config first.
- If failures are varied and trajectory-specific, analyze saved raw responses
  and environment traces before generating more data.

---

## Multi-Turn Environment Smokes

A useful multi-turn smoke should usually prove these separately:

- prompt render: no stale tool names, correct configured action examples, correct
  configured context
- first action: model can make one valid discovery/list/read call
- feedback: environment returns non-empty structured tool results or errors
- continuation: model uses the feedback instead of guessing hidden paths
- terminal answer/action: final text or mutation satisfies assertions
- final review: gates and judge evaluate the whole trajectory, not a flattened
  one-turn repair

For agentic rollouts, keep `--max-iterations` low while debugging prompt shape
and loop behavior, then raise it to the normal default after the raw loop is
healthy. This avoids spending time on repeated improver calls when the problem
is actually prompt rendering or loop control.

For non-contiguous failures, use explicit selectors instead of rerunning a
whole range:

```bash
python -m SynthChat.run improve \
  --input Datasets/existing_dataset.jsonl \
  --rubrics YOUR_RUBRIC \
  --lines 7,12,20-25 \
  --workers 8 \
  --max-iterations 3 \
  --output Datasets/synthchat/regen_slice.jsonl
```

If the same slice needs to be rerun later, keep the selectors in a checked-in
text file and use `--line-file`:

```text
# Datasets/tools_datasets/reports/cli_schema/regen_lines.txt
7
12
20-25
```

```bash
python -m SynthChat.run improve \
  --input Datasets/existing_dataset.jsonl \
  --rubrics YOUR_RUBRIC \
  --line-file Datasets/tools_datasets/reports/cli_schema/regen_lines.txt \
  --workers 8 \
  --max-iterations 3 \
  --output Datasets/synthchat/regen_slice.jsonl
```

The emitted `.improve_report.json` preserves original input `line_number`
values even when only a subset is processed. Use those preserved line numbers
when patching regenerated rows back into the source dataset.

---

## Red Flags in Dry-Run Output

Stop and fix the YAML if you see:

- **Generic/template content** — "Lorem ipsum", "[placeholder]", repeated boilerplate
- **Missing sections** — No `<vault_structure>`, no frontmatter, etc.
- **Hallucinated paths** — File paths that don't exist in the system prompt
- **Wrong tool calls** — Using `delete` when scenario says `create`
- **Context mismatches** — configured context fields don't match the configured prompt/context section
- **All examples identical** — No variety in generated content
- **Empty fields** — Null thinking blocks, empty tool arguments
- **Judge always passes/fails** — Threshold or prompt needs tuning
- **Environment failures ignored by improvement** — runtime issues appear in
  metadata but do not influence judge feedback or the saved pass/fail result
- **Stale environment failures carried into later retries** — if the response
  changes, the next judgment round should only see the new round's environment
  result, not the prior round's failure unless the rerun reproduces it

---

Additional privacy-specific red flags:
- Raw PII survives the sanitize pass. Stop and inspect `OPF_CHECKPOINT` and `TIKTOKEN_CACHE_DIR` before trusting the run.
- Sanitize mutates fields you did not intend to touch. Inspect the emitted privacy reports and metadata before scaling up to larger datasets.

## Gotcha: stale environment feedback in response retries

If a response-stage retry is triggered by environment/runtime errors, rerun the
environment after each improved response and feed only that refreshed result
into the next judgment round.

Check these fields together:
- `metadata.environment.passed`
- `metadata.stage_reviews.final.passed`
- `metadata.labels.filter.stage_failures`

If environment and final judge pass but `stage_failures` still includes
`response`, the retry loop is likely still judging against stale environment
feedback from an earlier round.

---

## Full Workflow Example

Creating a brand-new scenario end-to-end:

```bash
# 1. Write the scenario YAML
#    (add to SynthChat/scenarios/tools.yaml or create new file)

# 2. Add target to settings.yaml defaults.targets
#    my_new_scenario: 50

# 3. Dry-run 3 examples
python -m SynthChat.run generate \
  --scenarios my_new_scenario \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_my_new_scenario.jsonl \
  --targets-file <(echo '{"my_new_scenario": 3}')

# 4. Inspect output, show to user, get feedback

# 5. Iterate on YAML if needed, re-dry-run

# 6. User approves → full generation
python -m SynthChat.run generate \
  --scenarios my_new_scenario \
  --workers 4

# 7. Validate the full output
python -m SynthChat.run validate \
  --input Datasets/synthchat/synthchat_TIMESTAMP.jsonl \
  --rubrics system_prompt_format,thinking_quality

# 8. Improve any failures
python -m SynthChat.run improve \
  --input Datasets/synthchat/synthchat_TIMESTAMP.jsonl \
  --rubrics system_prompt_format,thinking_quality
```
