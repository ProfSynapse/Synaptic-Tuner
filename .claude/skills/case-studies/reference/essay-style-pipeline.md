# Case Study: Essay-Style Training Pipeline

A complete walkthrough of how the project teaches a language model to transform messy brainstorm notes into structured, voice-aware essay outlines — from document corpus to a fine-tuned model that captures an author's tone, challenges ideas dialectically, and produces publication-ready structure.

---

## Stage 1: Define the Capability

### What We're Teaching

The model must learn to:
1. **Read messy brainstorm text** — fragments, half-thoughts, stream-of-consciousness notes
2. **Extract structure** — identify sections, themes, and rhetorical moves from the raw ideas
3. **Produce a structured outline** — title, overview, per-section summaries, tone/voice notes, key themes
4. **Challenge the author dialectically** — pressure-test ideas, find contradictions, not just affirm
5. **Suggest next steps** — specific, actionable follow-ups tailored to THIS essay, not generic advice

### How This Differs from Tool Calling

| Dimension | Tool Calling | Essay Style |
|-----------|-------------|-------------|
| Output format | Structured JSON (tool calls) | Structured prose (Markdown outline) |
| Validation | Schema match (deterministic) | Rubric scoring (subjective) |
| Source material | Tool schemas | Existing essay corpus |
| "Correct" answer | One right tool, exact params | Many valid outlines, judged on quality |
| Key challenge | Syntax precision | Voice preservation, critical thinking |

### The Source Material: Essay Corpus

Instead of tool schemas, the source of truth is a corpus of existing essays:

```
Meditations on Alignment/_flat_essays/
├── AI Familiars.md
├── Divine Sparks - 04. Quetzalcoatl.md
├── Moloch and WinWin - 008. Uncertainty.md
└── ... (dozens of essays)
```

Each essay is a finished piece with:
- A distinctive voice (scholarly yet conversational, mythic-meets-technical)
- A clear structure (4-6 major sections with thematic headings)
- References and analogies woven throughout
- A specific thesis and argumentative arc

### The Outline Format Specification

The target output follows a precise structure:

```markdown
[Opening reflection — 2 paragraphs challenging the brainstorm dialectically]

# [Essay Title]

## Overview
[2-paragraph summary of thesis and approach]

## [Section Heading — matching essay's actual structure]
- **Key ideas:** [what this section covers]
- **References/analogies:** [specific references used]
- **Rhetorical moves:** [techniques and transitions]

## [Next Section Heading]
...

## Tone & Voice
[Description of writing register, humor, posture]

## Key Themes
- **[Theme]:** [how it manifests in the essay]

[Closing transition + 2-4 specific next steps as bullets]
```

**Critical constraints:**
- Address the author in second person ("your essay argues", not "the essay argues")
- Match the actual section count from the essay (typically 4-6, NOT 10+)
- No "PART 1/2/3" labels — flow naturally
- Keep it concise — fits on one page
- Next steps must be specific to THIS essay, not generic writing advice

### Quality Rubrics

Two rubrics govern quality at generation time:

| Rubric | Applies To | What It Checks |
|--------|-----------|---------------|
| `essay_brainstorm_quality` | User turn | Is the brainstorm messy, short, fragmented? Does it reference specific essay content? |
| `essay_outline_quality` | Assistant turn | Does the outline match the essay's structure? Is the voice description accurate? Are next steps specific? |

---

## Stage 2: Create Training Data

### The Docs-Based Generation Strategy

Unlike tool training (which generates from prompt templates), essay training uses **docs-based generation** — each training example is derived from an actual essay in the corpus.

```
┌──────────────────────────────────────────────────────────┐
│  ESSAY CORPUS                                             │
│  Meditations on Alignment/_flat_essays/                   │
│  Each essay → 1 training example                          │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 1: Generate Brainstorm (User Turn)                 │
│  LLM reads finished essay → reverse-engineers the messy  │
│  thoughts the author would have had BEFORE writing        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 2: Generate Outline (Assistant Turn)               │
│  LLM reads brainstorm + finished essay → produces a      │
│  structured outline that mirrors the essay's real shape   │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  OUTPUT: 2-turn conversation (no system prompt)           │
│  user: messy brainstorm                                   │
│  assistant: structured outline                            │
└──────────────────────────────────────────────────────────┘
```

### The Scenario Definition

**File:** `SynthChat/scenarios/essay_outline.yaml`

This is a `docs_based` scenario — fundamentally different from the `tool` type used for tool calling.

```yaml
scenarios:
  essay_outline:
    type: docs_based
    system: false           # No system prompt in output

    rubrics:
      user: [essay_brainstorm_quality]
      response: [essay_outline_quality]

    prompts:
      user_system: |
        You are simulating the internal brainstorming process of an essay author.
        You have access to the finished essay below. Your job is to imagine what
        the author's messy, unstructured thoughts looked like BEFORE they sat down
        to write.

        <finished_essay>
        {doc_content}
        </finished_essay>

        Guidelines:
        - Keep it SHORT (75-200 words)
        - Stream-of-consciousness, not organized
        - Reference specific ideas from the essay
        - Use sentence fragments, half-formed connections
        - Show intellectual excitement and uncertainty

      user: |
        Generate the messy brainstorm thoughts this author would have had.
        Output ONLY the brainstorm text. No preamble.

      assistant: |
        You are a writing coach. The author shared their brainstorm.
        You also have the reference essay:

        <reference_essay>
        {doc_content}
        </reference_essay>

        Produce: opening reflection → draft outline → next steps
        (flowing naturally, no PART labels)
```

**Key design decisions:**
- `{doc_content}` is injected into BOTH the user-generation prompt AND the assistant-generation prompt, but it does NOT appear in the final training example
- The user turn is generated FIRST, then the assistant turn uses the generated brainstorm as context
- `system: false` means no system prompt in the output — the model must learn to outline from the brainstorm alone

### Running Generation

```bash
python -m SynthChat.run generate \
  --docs "Meditations on Alignment/_flat_essays/" \
  --scenarios essay_outline \
  --per-doc 1 \
  --output Datasets/synthchat/essay_outlines.jsonl
```

**What happens:**
1. SynthChat scans the `--docs` directory for `.md` files
2. For each file, it loads the content as `{doc_content}`
3. It generates a brainstorm (user turn) using the `user_system` + `user` prompts
4. It generates an outline (assistant turn) using the `assistant` prompt + the generated brainstorm
5. It validates both against rubrics
6. It outputs a 2-turn conversation (user brainstorm → assistant outline)

### Example Output

**User turn (generated brainstorm):**

```
familiars—witches' cats, vampire minions in "What We Do…", Pullman's daemons 
as soul-animals. why "companion" feels right? universal longing? bridge worlds, 
digital now? AI as modern familiar? tabula rasa AI that learns your timbre, 
emotions, quirks… adaptive loops, preference recog… but huge players—Gemini, 
Co-Pilot, Claude—feel like corporate familiars, not ours. need democratized, 
personal, consent-first. can we make AI a guide, protector, extension of self?
```

**Assistant turn (generated outline):**

```markdown
Your brainstorm already weaves a rich tapestry: you pull familiar mythic 
figures—witches' cats, Pullman's daemons, patronuses—into a conversation with 
today's generative AI, and you ask what a truly personal, consent-first 
companion might look like. The strongest link I see is the tension between 
the yearning for a magical, loyal partner and the reality of a tool built on 
massive corporate models...

# The Familiar Algorithm: Re-imagining AI Companions

## Overview
Your essay argues that the timeless appeal of mythic familiars can guide the 
design of AI companions that feel intimate, ethical, and under individual 
control...

## Mythic Roots of Companionship
- **Key ideas:** Explore why familiars, daemons, and similar creatures satisfy 
  a universal longing for a bridging partner.
- **References/analogies:** Pullman's daemons, "What We Do in the Shadows" 
  familiars, Harry Potter patronus, Star Wars droids.
- **Rhetorical moves:** Use vivid description to evoke the emotional pull of 
  these myths, then transition to the question of "what if they were digital?"

## From Arcane to Algorithm
...

## Tone & Voice
The essay should feel like a thoughtful technomythologist: scholarly enough to 
reference literature and policy, but conversational and slightly whimsical...

## Key Themes
- **Mythic Ethos as Design Principle:** The loyalty, protection, and intimacy 
  of familiars become criteria for AI behavior, not literal personalities.
- **Consent-First, User-Owned Architecture:** Every data flow, learning loop, 
  and moral rule is anchored in explicit user permission.

A few directions worth exploring as you move forward:
- flesh out a concrete "privacy sandbox" diagram for the personal familiar's 
  local-first architecture.
- add a brief comparative table that pits corporate AI assistants against an 
  open-source familiar on consent, data ownership, and adaptability.
```

---

## Stage 3: Validate & Improve

### Rubric-Based Validation

Unlike tool calling (where validation is mostly deterministic schema checking), essay validation is **rubric-scored by a judge LLM**.

```bash
python -m SynthChat.run validate \
  -i Datasets/synthchat/essay_outlines.jsonl \
  --rubrics essay_brainstorm_quality,essay_outline_quality
```

**Brainstorm quality checks:**
- ✅ Short (75-200 words) — not a full paragraph
- ✅ Fragmentary — sentence fragments, dashes, questions
- ✅ References specific essay content (not generic)
- ✅ Shows intellectual excitement and uncertainty
- ❌ Organized with headers/bullets (should be messy)
- ❌ Clean prose (should be raw)

**Outline quality checks:**
- ✅ Opens with dialectical challenge (not affirmation)
- ✅ Section headings match the actual essay's structure
- ✅ Each section has key ideas, references, rhetorical moves
- ✅ Tone & Voice section captures the essay's register
- ✅ Next steps are specific to THIS essay
- ✅ Uses second person ("your essay")
- ❌ Generic template headings ("Introduction", "Body", "Conclusion")
- ❌ More than 6 sections (should match essay's real count)
- ❌ Generic next steps ("consider adding more detail")

### Improvement Loop

If quality is low, run the improve command:

```bash
python -m SynthChat.run improve \
  -i Datasets/synthchat/essay_outlines.jsonl \
  --rubrics essay_brainstorm_quality,essay_outline_quality \
  --max-iterations 10
```

The improvement LLM (typically GPT-4o at temperature 0.1) acts as a judge:
1. Scores the example against the rubric
2. If it fails, rewrites the failing turn
3. Re-scores the rewrite
4. Repeats up to `max_iterations`

### Manual Review for Voice

Automated rubrics can check structure, but **voice** requires human judgment. Sample 10-20% and ask:

- Does the brainstorm actually sound like messy thinking, or is it too polished?
- Does the outline capture what makes THIS essay unique, or could it apply to any essay?
- Does the dialectical challenge actually push back, or does it just rephrase the brainstorm?
- Are the next steps actionable and specific?

### Converting to Preview Format

For easier manual review, convert JSONL to readable Markdown:

```bash
./scripts/jsonl_to_markdown.sh Datasets/synthchat/essay_outlines.jsonl
```

This creates a preview file like `Datasets/essay_datasets/stage1_thoughts_to_outline_preview.md` with each example formatted for easy reading.

---

## Stage 4: Train

### SFT: Teaching Outline Structure

**Dataset:** Validated essay outlines (positive examples only).

```bash
cd Trainers/sft

python train_sft.py \
  --model-size 7b \
  --local-file ../../Datasets/synthchat/essay_outlines_validated.jsonl \
  --num-epochs 3 \
  --learning-rate 2e-4
```

**What SFT learns:**
- The outline format (title → overview → sections → tone → themes → next steps)
- How to read a messy brainstorm and extract structure
- Section heading style (descriptive, not numbered)
- The dialectical opening pattern (challenge, don't affirm)

### KTO: Teaching Quality Judgment

For KTO, you need paired examples — same brainstorm, one good outline and one bad outline.

**Creating negative examples:**
Unlike tool calling (where bad examples come from high-temperature self-play), essay negatives are typically authored by:
1. Taking a good outline and degrading it systematically:
   - Replace specific section headings with generic ones ("Introduction", "Body")
   - Remove specific references, leaving vague descriptions
   - Change dialectical challenge to bland affirmation
   - Add too many sections (10+ instead of 4-6)
   - Make next steps generic ("consider adding more detail")
2. Using a weaker model to generate outlines (naturally produces lower quality)

```bash
cd Trainers/kto

python train_kto.py \
  --model-size 7b \
  --local-file ../../Datasets/essay_kto_pairs.jsonl \
  --num-epochs 1 \
  --learning-rate 1e-6
```

**What KTO learns:**
- Prefer specific section headings over generic templates
- Prefer dialectical challenge over agreeable affirmation
- Prefer concise outlines (4-6 sections) over bloated ones (10+)
- Prefer specific next steps over generic advice
- Prefer accurate voice descriptions over boilerplate

---

## Stage 5: Evaluate

### Essay Evaluation Challenges

Tool calling evaluation is straightforward: did the model call the right tool with the right arguments? Essay evaluation is harder because there's no single "correct" outline.

**Evaluation approach:** Use a strong judge model (GPT-4o) to score outlines against rubrics.

```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model path/to/essay_lora \
  --scenario essay_prompts.yaml \
  --output Evaluator/results/essay_eval.json
```

**Evaluation dimensions:**

| Dimension | How to Measure |
|-----------|---------------|
| Structure fidelity | Does the outline have all required parts? (Overview, sections, tone, themes, next steps) |
| Section count | 4-6 sections? (Not 10+, not 2) |
| Dialectical quality | Does the opening challenge ideas, or just summarize? |
| Specificity | Are references, analogies, and next steps specific to the input? |
| Voice accuracy | Does the tone description match the essay's actual register? |
| Conciseness | Could it fit on one page? |

### Comparing Iterations

Track metrics across training rounds:

| Metric | Base Model | After SFT | After KTO |
|--------|-----------|-----------|-----------|
| Has all outline parts | 30% | 85% | 90% |
| Correct section count | 20% | 70% | 85% |
| Dialectical opening | 10% | 60% | 80% |
| Specific next steps | 15% | 55% | 75% |

---

## Stage 6: Iterate

### Common Essay Failure Patterns

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Generic headings | Model defaults to template structure | More examples with distinctive headings |
| Affirming instead of challenging | Model avoids conflict | KTO pairs: dialectical (true) vs affirmative (false) |
| Too many sections | Model over-segments | KTO pairs: concise 5-section (true) vs bloated 10-section (false) |
| Generic next steps | Model falls back to boilerplate | More examples with essay-specific suggestions |
| Wrong voice description | Model doesn't analyze register | More examples with varied voice types |
| Missing second person | Model uses third person | Add "your/you" emphasis in more training examples |

### Expanding the Corpus

To improve, add more essays covering:
- Different voice types (formal, whimsical, urgent, contemplative)
- Different lengths (short blog posts vs long-form)
- Different structures (narrative, argumentative, comparative)
- Different domains (technology, philosophy, mythology, policy)

---

## Key Takeaways: How Essay Differs from Tools

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| Data source | Tool schemas → prompt templates | Essay corpus → docs-based generation |
| Validation | Deterministic schema match | LLM-judged rubric scoring |
| Negative examples | High-temperature self-play errors | Systematic degradation of good examples |
| "Correct" answer | Exactly one right tool call | Many valid outlines (quality spectrum) |
| Iteration signal | Schema validation failures | Rubric dimension scores |
| Biggest risk | Wrong tool / bad syntax | Generic / template-y output |

### The Universal Lesson

Despite these differences, the pipeline structure is identical:
1. **Define** what good looks like (schemas OR rubrics)
2. **Generate** training data (templates OR docs-based)
3. **Validate** before training (schema checks OR rubric scoring)
4. **Train** SFT then KTO (same trainers, same commands)
5. **Evaluate** against scenarios (same evaluator, different rubrics)
6. **Iterate** on failures (same loop, different signals)

The tools change. The process doesn't.

---

## File Map

```
Meditations on Alignment/_flat_essays/      ← Source essay corpus
SynthChat/scenarios/essay_outline.yaml       ← Docs-based generation scenario
SynthChat/rubrics/essay_brainstorm_quality/  ← Brainstorm quality rubric
SynthChat/rubrics/essay_outline_quality/     ← Outline quality rubric
Datasets/essay_datasets/                     ← Generated essay training data
Datasets/synthchat/                          ← SynthChat output directory
Trainers/sft/                        ← SFT trainer (same as tools)
Trainers/kto/                        ← KTO trainer (same as tools)
Evaluator/config/scenarios/                  ← Essay evaluation scenarios
```
