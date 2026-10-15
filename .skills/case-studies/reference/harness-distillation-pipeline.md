# Case Study: Harness Distillation (Agent-as-Harness) Pipeline

A complete walkthrough of teaching a small model agentic procedures (locate,
read, recover, answer from evidence) by letting it act and having a strong
reviewer correct each step before it runs. The method follows Harness-Zero
(arXiv:2609.24974). Every command and number below comes from the checked-in
pilot, so the walkthrough is reproducible.

---

## Stage 1: Define the Capability

### What we're teaching

A deployable student model (`qwen/qwen3.5-9b` in the pilot) should work a
note workspace through one `useTools` tool:

1. **Locate** a note with `search content` instead of guessing paths.
2. **Recover** from an empty search by changing the query, never by
   repeating it.
3. **Read** the note (`content read "<path>" 1`) before answering.
4. **Answer** in text, grounded in what the read returned.

### Why not plain teacher generation?

Ordinary SynthChat generation has a strong teacher write every turn.
Harness-Zero's ablations show why that transfers poorly:

| Training source | Test pass@1 (paper, USPTO) |
|---|---|
| Base model, untrained | 12% |
| Strong teacher's own trajectories | 12% |
| Teacher running inside the specialized harness | 3% |
| Review with oracle answers | 15% |
| **Student rollout + reviewer guided by procedural rules** | **30%** |

Two lessons drive the design:
- The data must start from the **student's own rollout**, in the student's
  own action space.
- The reviewer gets **procedures, not answers**.

### How this differs from the other case studies

| Dimension | Agentic Search | Harness Distillation |
|---|---|---|
| Who writes assistant turns | Teacher | Student, with reviewer corrections |
| Quality control | Judge after the fact | Review before every step executes |
| Training signal | Teacher's ideal path | Student's path with minimal fixes |
| Extra output | — | DPO pairs: student proposal vs correction, from the same state |

---

## Stage 2: Build the Reference Harness and Scenario

Everything is config. The pilot lives in
`SynthChat/scenarios/workspace_harness_review_pilot.yaml`.

### 1. Student-visible context: no answers

The student sees only what a deployed model would see:
- `workspace_format: cli_student`: the vault structure and session context,
  but no `<note_contents>` and no agent-style tool listing.
- A `<cli_commands>` extra section with the real CLI syntax, taken from the
  command catalog.
- **No answer-key sections.** `task_context` holds the answer, but only the
  user-prompt stage and the final judge see it.

### 2. Student turn config

```yaml
tool_call_format:
  arguments_encoding: object   # the schema makes function.arguments an object
student:
  provider: openrouter
  model: qwen/qwen3.5-9b
  provider_routing: {require_parameters: true}   # only providers that enforce the schema
  schema: use_tools_response
  prompt: "Write the assistant's next message ..."
```

### 3. The private reference harness

The reference harness is a set of **components** (skills and memory written
as procedures) plus **middleware** (deterministic checks that add a private
hint to the review):

```yaml
reference_harness:
  components:
    - id: cli-syntax           # exact commands; quoting matters only for multi-word args
    - id: locate-before-reading
    - id: search-recovery      # search is literal: retry with one distinctive word
    - id: read-before-answering
    - id: grounded-final-answer
    - id: workspace-failure-patterns   # kind: memory
  middleware:
    - id: final-before-read    # final answer but no `content read` yet
    - id: agent-style-command  # searchManager_content inside the CLI string
    - id: shell-syntax
```

Each component must say *how to behave*, never *what the answer is*.

### 4. Reviewer config

The reviewer system prompt is adapted from the paper's harnessing-agent prompt:
- PASS sound steps, harmless style differences, and mistakes the student can
  recover from on its own.
- REPLACE with the smallest coherent correction, written in the student's
  voice, and only for the next step.
- Never mention the review, and never use facts the student hasn't seen.

Set `replace_budget: 3`. Leak patterns reject replacements that mention the
reviewer. Reviewer-voice patterns flag turns for exclusion from training.

---

## Stage 3: Generate and Inspect

Follow the testing protocol: a single smoke first, then a 3-rollout dry run,
reading every trace before scaling.

```bash
# One rollout
python -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_workspace_harness_review_pilot_smoke.json \
  --env-backend local -o scratch/harness_review/smoke.jsonl

# Three rollouts
python -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_workspace_harness_review_pilot_dryrun.json \
  --env-backend local --workers 3 -o scratch/harness_review/dryrun.jsonl

# Read the decisions
python SynthChat/scripts/inspect_rollout_trace.py \
  --input scratch/harness_review/dryrun.jsonl --max-chars 400
```

Export `OPENROUTER_API_KEY` first. `SynthChat.run` does not read `.env` for
you.

### What a good dry run looks like

From the final pilot run (all 3 rollouts passed the final judge):

```
rollout 1: PASS, PASS, PASS              <- the student solved it unaided
rollout 2: t1 REPLACE  search content password reset
                 ->    search content "password reset"   (unquoted multi-word query)
           t2 PASS     search content reset              (student recovered on its own)
           t3 PASS     content read docs/account-access.md 1
           t4 PASS     final answer
```

Check these signals:
- **Structural replacements should be near zero.** If they're high, the
  student can't produce the action format, so it's being trained on reviewer
  turns.
- **Each replacement's `reason` should cite a real procedural fault,** not
  style.
- **`reviewer_voice_turns` should be empty.**
- **Every turn must have a decision; none should be `FAILED`.**

---

## Stage 4: Project to Training Data

```bash
python SynthChat/scripts/project_rollout_datasets.py \
  --input scratch/harness_review/dryrun.jsonl \
  --canonical-output scratch/harness_review/projected/canonical.jsonl \
  --kto-output scratch/harness_review/projected/kto.jsonl \
  --grpo-output scratch/harness_review/projected/grpo.jsonl \
  --sft-output scratch/harness_review/projected/sft_turns.jsonl \
  --sft-projection per_turn \
  --dpo-output scratch/harness_review/projected/dpo.jsonl
```

- **`--sft-projection per_turn`** writes one row per accepted assistant turn,
  ending at that turn. The SFT loss mask trains only a row's final assistant
  turn. With full-trajectory rows, the tool-call turns (where the corrections
  live) would get no loss.
- **`--dpo-output`** turns each REPLACE into
  `{prompt, chosen: correction, rejected: student proposal}` at the same
  state. It skips reviewer-voice turns and empty proposals.

Validate with the trainers' own preprocessing:

```bash
python tuner.py doctor sft-mask \
  --dataset-path scratch/harness_review/projected/sft_turns.jsonl \
  --model Qwen/Qwen3.5-9B --max-seq-length 8192 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

The pilot's 6 rollouts gave 22 SFT rows and 10 DPO pairs. The doctor passed
with no dropped rows. The DPO loader renders tool calls and tool results as
text, the same way the SFT trainer does.
`shared.validation.dataset_validator` rejects the `tool` role. That check is
outdated for multi-turn rows; use the doctor instead.

---

## Stage 5: Train and Evaluate

Training follows the `fine-tuning` skill's cloud paths. This box has no GPU,
and trainer dry-runs need `unsloth`.

1. **SFT on `sft_turns.jsonl`.** Set `validation_group_key: metadata.seed_id`
   so all turns of an episode stay on the same side of the split.
2. **Optional DPO on `dpo.jsonl`, starting from the SFT adapter.** This is the
   comparison signal the paper lists as future work.
3. **Evaluate the student with no reviewer,** using the same CLI scenario
   through the Evaluator. Compare against the untrained student on held-out
   tasks.

---

## Lessons From Building the Pilot

The pilot surfaced each of these. They are worth checking on any new
scenario.

| Symptom | Cause | Fix |
|---|---|---|
| Student answers correctly at turn 1 with no tools | System prompt included `<note_contents>` | `cli_student` workspace format |
| Both models invent `searchManager_content "…"` | Tool listing used agent names; transport said "use CLI syntax from it" | `<cli_commands>` section + `cli-syntax` component + middleware |
| Every student tool call is pseudo-JSON | `function.arguments` was a free-form string | `arguments_encoding: object` |
| Student JSON wrapped in a markdown fence | Provider ignored `json_schema` | `provider_routing.require_parameters: true` |
| HTTP 400 from every OpenAI reviewer | `const` schema nodes lacked `type` | Fixed in `tool_response_schema.py` |
| Reviewer gives up and writes "couldn't find it" | Read-only fixture assertions are true from the start, so the loop sent "completed successfully" too early | `stop_on_environment_pass: false`; let the final judge decide |
| Reviewer spends the budget adding quotes | It didn't know which quoting matters | The `cli-syntax` component says only multi-word arguments need quotes |
| Tool-call turns untrained | Loss mask trains only the final turn | `--sft-projection per_turn` |
| DPO chat template crash (`items` on a string) | DPO loader skipped the shared tool-call renderer | DPO loader now sanitizes like SFT |
| DPO pairs differ in argument key order (student keys alphabetized) | Models emit keys in arbitrary order | Parsing serializes wrapper arguments in the format's declared field order |

---

## Adapting This to a New Capability

1. **Pick a scenario where the student can act in its real action space** and
   success can be judged.
2. **Write the reference harness as procedures, starting from observed
   failures.** Run the student alone first and read its traces.
3. **Strip answers from everything the student and reviewer see.**
4. **Smoke, then a dry run, then read every decision.** Fix the config before
   generating at scale.
5. **Project per-turn SFT plus DPO, validate with the doctor, then train.**
