# Case Study: Agentic Search (RAG Agent) Training Pipeline

A complete walkthrough of how to teach a language model to act as a RAG agent — searching a corpus, selecting relevant documents, and answering questions grounded exclusively in what it found.

---

> **YOUR TOOLS ARE NOT OUR TOOLS**
>
> This case study describes a **three-stage pattern**: Search, Select, Answer. The tool names used below (`YOUR_SEARCH_TOOL`, `YOUR_READ_TOOL`, `YOUR_LIST_TOOL`) are **placeholders**. Your system might use `grep`, `ripgrep`, a vector DB query, `cat`, `readFile`, an HTTP API, or something else entirely.
>
> **Before you proceed past Stage 1, you must know:**
> 1. What tool does "search" in your system?
> 2. What tool does "read" in your system?
> 3. What tool does "list" in your system?
>
> The three-stage behavior is the constant. The tools are variables.

---

## Stage 1: Define the Capability

### What We're Teaching

The model must learn a three-stage behavior loop:

1. **Search** — given a question, generate search terms that would surface target documents in a keyword/grep-style search. The model must reason about vocabulary: what words would the answer contain?
2. **Selective Read** — from the search results, pick the relevant documents and skip distractors. Not everything returned is useful. The model must show judgment.
3. **Grounded Answer** — answer using ONLY content from the documents it read. No hallucination. If the docs don't contain the answer, say so.

### How This Differs from the Other Case Studies

| Dimension | Tool Calling | Essay Style | Agentic Search |
|-----------|-------------|-------------|----------------|
| Goal | Call the right API | Produce creative output | Find and synthesize from sources |
| Output | Structured JSON | Structured prose | Grounded prose with citations |
| Tools are... | The end product | Not involved | The means to an end |
| Validation | Schema match | Rubric scoring | Retrieval recall + groundedness |
| Key challenge | Syntax precision | Voice preservation | Search strategy + no hallucination |

Tool calling trains a model to use tools correctly. Agentic search trains a model to use tools *strategically* — the tools are a means to producing a grounded answer.

### Inspiration: Explore, Verify, Extend

The data generation pattern draws from Chroma's Context-1 approach: generate a seed corpus, then create training examples where the model must explore (search), verify (read and confirm), and extend (synthesize an answer). Our corpus is generated per-seed via the environment backend rather than pre-indexed, and evaluation is YAML-driven through the existing Evaluator.

### The Three Rubrics

Quality is governed by three rubrics, one per stage:

| Rubric | Stage | What It Checks |
|--------|-------|---------------|
| `search_term_quality` | Search | Are the terms specific enough to retrieve target docs? Do they account for vocabulary mismatch? |
| `doc_selection` | Select | Did the model read relevant docs and skip distractors? Precision and recall. |
| `groundedness` | Answer | Is every claim traceable to a document? Does it say "not found" when appropriate? |

---

## Stage 2: Create Training Data

### Prerequisites: Confirm Your Tools

> **STOP.** Before writing any scenarios or generating any data, you must map the three-stage pattern to your actual tools.

Define your tool mapping:

| Stage | Placeholder | Your Tool | Example |
|-------|------------|-----------|---------|
| Search | `YOUR_SEARCH_TOOL` | ___________ | `grep`, `searchContent`, `vector_query` |
| Read | `YOUR_READ_TOOL` | ___________ | `cat`, `read`, `fetchDocument` |
| List | `YOUR_LIST_TOOL` | ___________ | `ls`, `list`, `listDirectory` |

Once you know your tools, create the tool schema and environment execution config that match your system. The scenario YAML references these tools by name — they must match exactly.

### Corpus Design

The environment backend generates the corpus automatically for each training example. A good corpus has:

- **Varied topics** — not all docs about the same thing
- **Overlapping vocabulary** — docs sharing keywords but covering different topics (forces precise search)
- **Deliberate distractors** — docs that look relevant but aren't (forces selective reading)
- **Multi-hop potential** — answers requiring info from 2+ documents

The seed scenario's `environment_generation.prompt` controls corpus shape. Customize it for your domain. Compare to Context-1's domain-specific corpora (web, SEC, patents, email) — same principle: realistic documents with realistic noise.

### Scenario Types

The scenario file defines three scenario types that build on each other:

**1. Seed scenarios** — generate the corpus and a simple single-doc question:
```yaml
scenarios:
  agentic_search_seed:
    type: tool
    environment_mode: generated
    system_template: mocked_workspace_vault
    rubrics:
      response: [search_term_quality, doc_selection, groundedness]
    environment_generation:
      prompt: |
        Generate a workspace with 5-8 documents across 2-3 topics.
        Include 2 distractors per topic that share vocabulary but
        cover different aspects. One document must contain the answer
        to a factual question that requires reading (not guessing).
```

**2. Find-and-answer scenarios** — question requires searching, reading one doc, answering:
The model must: search with good terms, pick the right doc from results, answer from its content only.

**3. Multi-hop scenarios** — answer requires info from 2+ documents:
The model must: search, read doc A, recognize it's incomplete, search again or read doc B, synthesize.

### Running Generation

Copy your scenario file into place and generate:

```bash
# Copy scenario (after replacing YOUR_SEARCH_TOOL etc. with real tool names)
cp .skills/case-studies/configs/agentic_search_scenario.yaml \
   SynthChat/scenarios/agentic_search.yaml

# Dry run — check 3 examples without saving
python -m SynthChat.run generate \
  --scenarios agentic_search \
  --count 3 \
  --dry-run

# Full generation
python -m SynthChat.run generate \
  --scenarios agentic_search \
  --workers 4 \
  --output Datasets/synthchat/agentic_search_raw.jsonl
```

### Example Training Conversation

Tool names below are placeholders — substitute your own:

```
system: [workspace context with corpus file listing]
user:   "What's the maximum retry count for the payment service?"
assistant:
  <thinking>I should search for payment service configuration.</thinking>
  [calls YOUR_SEARCH_TOOL with terms: "payment retry max"]
tool:   [returns: payment_config.md, payment_overview.md, retry_policy.md]
assistant:
  <thinking>payment_config.md and retry_policy.md look relevant.
  payment_overview.md is a high-level summary — skip it.</thinking>
  [calls YOUR_READ_TOOL on payment_config.md]
tool:   [returns file content with "max_retries: 5"]
assistant:
  Based on payment_config.md, the maximum retry count for the payment
  service is **5** (configured via `max_retries: 5` in the service config).
```

All three stages visible: targeted search terms, selective reading (skipped the overview), grounded answer with citation.

---

## Stage 3: Validate & Improve

### Structural Validation

```bash
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py \
  Datasets/synthchat/agentic_search_raw.jsonl
```

### Rubric-Based Quality Validation

Copy rubric files into place, then validate:

```bash
# Copy rubrics (one file per rubric)
cp .skills/case-studies/configs/agentic_search_rubrics/*.yaml \
   SynthChat/rubrics/

# Validate a small batch
python -m SynthChat.run validate \
  -i Datasets/synthchat/agentic_search_raw.jsonl \
  --rubrics search_term_quality,doc_selection,groundedness \
  --start-line 1 --end-line 10
```

**Search term quality:** Specific to the question? Account for vocabulary mismatch? Would plausibly retrieve the target doc? Not too broad or too narrow?

**Document selection:** Read the doc(s) containing the answer? Skipped distractors? Avoided reading every single result?

**Groundedness:** Every claim traceable to a document? No added information? Says "not found" when docs lack the answer? Citations present?

### Improvement Loop

```bash
python -m SynthChat.run improve \
  -i Datasets/synthchat/agentic_search_raw.jsonl \
  --rubrics search_term_quality,doc_selection,groundedness \
  --max-iterations 5
```

### Manual Review

Sample 10-20%: Do search terms feel human? Is reading selective? Can you verify the answer from the cited doc? For multi-hop, does the model actually combine info?

---

## Stage 4: Train

### Training Sequence

```
SFT (learn the loop)  →  KTO (learn judgment)  →  GRPO (optimize retrieval)
      ↓                        ↓                        ↓
  Search→Read→Answer     Good vs bad search      Reward = did you find it?
```

### SFT: Teaching the Search-Read-Answer Loop

**Dataset:** Positive examples only. The model learns WHEN to search, WHAT to read, and HOW to answer.

```bash
cd Trainers/rtx3090_sft

python train_sft.py \
  --model-size 7b \
  --local-file ../../Datasets/synthchat/agentic_search_sft.jsonl \
  --num-epochs 3 \
  --learning-rate 2e-4
```

**What SFT learns:**
- The three-stage loop structure (search → select → answer)
- How to generate effective search terms from a question
- How to decide which results to read (and which to skip)
- How to ground answers in document content
- When to say "I couldn't find that information"

### KTO: Teaching Search Judgment

**Dataset:** Interleaved true/false pairs. Same question, contrasting quality.

Negative examples come naturally from rubric failures during validation:
- **Bad search terms** (too broad, wrong vocabulary) → `label: false`
- **Reading everything** instead of selecting → `label: false`
- **Hallucinated answers** that go beyond what the docs say → `label: false`
- **Missed answers** where the model says "not found" but the doc was there → `label: false`

```bash
cd Trainers/rtx3090_kto

python train_kto.py \
  --model-size 7b \
  --local-file ../../Datasets/synthchat/agentic_search_kto.jsonl \
  --num-epochs 1 \
  --learning-rate 1e-6
```

**What KTO learns:**
- Prefer specific search terms over generic ones
- Prefer selective reading over reading everything
- Prefer grounded answers over hallucinated ones
- Prefer "not found" over fabricated answers

### GRPO: Optimizing Retrieval Reward (Optional)

GRPO is well-suited here because the reward is **verifiable**: did the search terms retrieve the target document? This mirrors Context-1's CISPO approach.

```bash
cd Trainers/rtx3090_grpo
# Set model.lora_path to KTO checkpoint in configs/config.yaml
python train_grpo.py
```

**Reward signal:** Binary retrieval recall. The environment backend re-executes the search and checks whether the target doc appeared. 1.0 if yes, 0.0 if no.

---

## Stage 5: Evaluate

### Setup

Copy evaluation scenarios and add a preset:

```bash
# Copy eval scenarios
cp .skills/case-studies/configs/agentic_search_eval.yaml \
   Evaluator/config/scenarios/agentic_search.yaml
```

Add the preset to `Evaluator/config/eval_run.yaml`:

```yaml
presets:
  agentic_search:
    scenarios:
      - agentic_search.yaml
    tags: [search, grounding, rag]
```

### Two Evaluation Modes

**Static mode** — corpus in the system prompt, quick behavioral check:
```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./Trainers/rtx3090_sft/sft_output_rtx3090/TIMESTAMP/final_model \
  --preset agentic_search \
  --output Evaluator/results/agentic_search_v1.json \
  --markdown Evaluator/results/agentic_search_v1.md
```

**Runtime mode** — real files, actual tool execution via agentic loop:
```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./Trainers/rtx3090_sft/sft_output_rtx3090/TIMESTAMP/final_model \
  --preset agentic_search_runtime \
  --env-backend local \
  --env-tool-schema path/to/your_tool_schema.yaml \
  --env-exec-config path/to/your_exec_rules.yaml \
  --output Evaluator/results/agentic_search_runtime_v1.json \
  --markdown Evaluator/results/agentic_search_runtime_v1.md
```

> **Do not train to the test.** The runtime eval fixtures (documents and questions) must NOT overlap with your training data. If the model saw these exact docs during SynthChat generation, the eval measures memorization, not capability. Use a held-out document set, or generate fresh questions from the fixture environment with a teacher model. The example fixtures in the template are placeholders — replace them with your own held-out content before trusting results.

### Key Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Search hit rate | Did the search terms retrieve the target doc? | > 80% |
| Doc selection precision | Of docs read, how many were relevant? | > 70% |
| Doc selection recall | Of relevant docs, how many were read? | > 90% |
| Groundedness score | Are all claims traceable to sources? | > 85% |
| "Not found" accuracy | When docs lack the answer, does the model say so? | > 75% |

### Status Meanings

| Status | Meaning |
|--------|---------|
| **PASS** | Found the right docs AND answered with grounding |
| **WARN** | Found docs but answer has minor grounding issues (KTO signal) |
| **FAIL** | Wrong docs, hallucinated answer, or missed available answer |

---

## Stage 6: Iterate

### Common Failure Patterns

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Bad search terms (too generic) | Not enough vocabulary-mismatch training | Add scenarios where the question uses different words than the doc |
| Reads everything instead of selecting | Not enough distractor-heavy scenarios | Add corpora with 3-4 distractors per relevant doc |
| Hallucinating beyond docs | Insufficient groundedness training | Add KTO negatives where the answer adds plausible-but-absent facts |
| Missing multi-hop answers | Too few multi-hop scenarios | Add scenarios requiring info from 2+ docs to form a complete answer |
| Never says "not found" | All training examples have answers | Add scenarios where the corpus genuinely lacks the answer |
| Searches when answer is in context | Model always follows the loop mechanically | Add examples where the system prompt already contains enough info |

### The Iteration Loop

```
Evaluate → Identify weakest metric → Generate targeted data → Retrain → Re-evaluate
    ↑                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

**Priority order for iteration:**
1. Groundedness (most important — hallucination is the worst failure)
2. Search hit rate (can't answer if you can't find)
3. Doc selection (efficiency and precision)
4. Multi-hop (hardest capability, iterate last)

---

## Comparison to Context-1

| Aspect | Context-1 (Chroma) | This Pipeline |
|--------|-------------------|---------------|
| Corpus | Pre-indexed domain corpora (web, SEC, patents, email) | Generated per-seed via environment backend |
| Data pattern | Explore → verify → extend | Search → select → answer (same idea, different names) |
| RL reward | Retrieval recall (CISPO) | Retrieval recall via GRPO (same signal) |
| Context pruning | Trained as a separate capability | Not included (train separately if needed) |
| Parallel tool calls | Part of training | Not included (train separately if needed) |
| Evaluation | Custom harness | YAML-driven Evaluator with presets |
| Simplification | Full production pipeline | Focused on the three-stage core |

**What we borrowed:** The explore-verify-extend pattern for data generation, and verifiable retrieval rewards for RL.

**What we simplified:** No context pruning (separate capability, separate training), no parallel tool calls (train via the tool-calling pipeline first).

**What's different:** Corpus generated fresh per seed (easier to control difficulty/coverage). YAML-driven Evaluator shared with every other capability.

---

## File Map

```
.skills/case-studies/configs/agentic_search_*.yaml  ← Template configs to copy
SynthChat/scenarios/agentic_search.yaml             ← Generation scenario (after copy)
SynthChat/rubrics/search_term_quality.yaml          ← Search quality rubric
SynthChat/rubrics/doc_selection.yaml                ← Selection quality rubric
SynthChat/rubrics/groundedness.yaml                 ← Grounding quality rubric
Datasets/synthchat/agentic_search_*.jsonl           ← Generated training data
Evaluator/config/scenarios/agentic_search.yaml      ← Evaluation scenarios
Trainers/rtx3090_sft/                               ← SFT trainer (same as other pipelines)
Trainers/rtx3090_kto/                               ← KTO trainer (same as other pipelines)
Trainers/rtx3090_grpo/                              ← GRPO trainer (same as other pipelines)
```

---

## Cross-References

| Stage | Skill for CLI Details |
|-------|----------------------|
| Scenario authoring | `synethetic-data-generation` |
| Rubric authoring | `synethetic-data-generation` (rubrics section) |
| SFT / KTO / GRPO flags | `fine-tuning` |
| Evaluator presets | `evaluation` |
| Upload trained model | `upload-deployment` |
