# Research Findings: Essay-Style Fine-Tuning Pipeline

> Prepared by: pact-preparer (PLANNING ONLY)
> Date: 2026-02-14
> Scope: Infrastructure research for building a training dataset pipeline from the Meditations on Alignment essays

## Executive Summary

The repository has mature infrastructure that can be substantially reused for building an essay-to-training-data pipeline. The SynthChat system provides a complete generation-improvement loop (scenario YAML, LLM clients, quality engine), and the existing `--docs` mode already supports document-seeded generation. The essay corpus contains approximately 97 usable essays (after filtering research notes, MoC files, outlines, and non-essay content) with rich YAML frontmatter metadata on ~75% of them.

**Key recommendation**: Build the essay pipeline as a new SynthChat scenario type rather than a standalone system. This reuses the LLM client infrastructure, improvement engine, JSONL output conventions, and parallel generation support with minimal new code.

---

## 1. Essay Corpus Analysis

### Location
`/Users/jrosenbaum/Documents/Code/Synthetic Conversations/Meditations on Alignment/`

### File Counts (Accurate)

| Category | Count | Notes |
|----------|-------|-------|
| Total .md files (excluding Research/) | 127 | Includes essays, MoC files, outlines |
| Research reference notes | 94 | In `*/Research/` subdirectories |
| MoC/outline files (`_` prefix) | 4 | `_The Commons.md`, `_On Portability.md`, `_On Noise.md`, `_Archetypical.md` |
| Files WITH YAML frontmatter | 97 | Includes some outlines and kanban boards |
| Files WITHOUT frontmatter | 30 | Older essays, some series entries |

### Estimated Usable Essays: ~90-100

After filtering out: MoC files (4), kanban/outline files (~8), research link files (~3), and empty/stub files.

### Structural Patterns

**Pattern 1: Standalone essays** (root level)
- `The Games we Play.md`, `Meditations on Alignment Preamble.md`, `Memetics and the Mind Virus.md`, etc.
- Generally have YAML frontmatter with: `title`, `subtitle`, `description`, `date`, `tags[]`
- Word count: 800-3000 words typically

**Pattern 2: Numbered series in subdirectories** (most of the corpus)
- `Divine Sparks/01. Prometheus.md` through `10. Walk in Beauty.md`
- `How AI Might (not) Save the World/` series (15 essays)
- `Moloch and WinWin/` series (9 essays)
- `Archetypical/Essays/` series (13 essays)
- `Constructive Rebellion/Essays/` series (10 essays)
- `The Digital Commons/` series (5 essays)
- `The Wizard's Apprentice/` series (6 essays)
- `You Fascist Nazi/Essays/` series (5 essays)
- Additional frontmatter fields: `series`, `essay_number`

**Pattern 3: No frontmatter** (older essays)
- `AI Familiars.md` -- begins with YouTube/LinkedIn social media copy sections before actual essay content
- `An Emulcifier/` -- multiple versions (v1, v2, v2.1, v2.2)
- Several later entries in series (e.g., `Constructive Rebellion/Essays/04-10`)
- These require content-based detection to extract essay body

### YAML Frontmatter Schema

Standard fields found across corpus:
```yaml
---
title: "Essay Title"
subtitle: "Optional Subtitle"
description: "1-2 sentence description"
date: 2025-10-04
tags:
  - tag1
  - tag2
series: "Series Name"        # Optional, for series essays
essay_number: 1              # Optional, for series essays
transcript: "[[reference]]"  # Rare, links to audio transcript
---
```

### Content Preprocessing Requirements

1. **Strip social media copy**: Some essays (e.g., `AI Familiars.md`) begin with `# Youtube Description`, `# Linkedin Post`, `# Image Description(s)`, `# Copy` sections before the actual essay. The pipeline must detect and skip these to extract only the essay body.

2. **YAML frontmatter extraction**: Parse `---` delimited frontmatter for metadata; use content after frontmatter as essay body.

3. **Filter non-essay files**:
   - Skip files in `*/Research/` directories (reference notes, not essays)
   - Skip `_` prefixed files (MoC/index files)
   - Skip files with `tags: [MoC]` in frontmatter
   - Skip kanban boards, series outlines, research link aggregation files
   - Skip non-markdown files (.m4a, .mp4, .png, .jpg, .webp)

4. **Handle mixed MoC/essay files**: `_On Portability.md` contains both an outline section and inline essay content. The pipeline should either skip these or extract only the essay portion.

5. **Version deduplication**: `An Emulcifier/` has v1, v2, v2.1, v2.2 -- only the latest version should be used.

### Series Metadata

| Series | Directory | Essay Count |
|--------|-----------|-------------|
| Meditations on Alignment (standalone) | root | ~8 |
| Divine Sparks | `Divine Sparks/` | 11 (00-10) |
| How AI Might (not) Save the World | `How AI Might (not) Save the World/` | 15 |
| Archetypical | `Archetypical/Essays/` | 13 |
| Moloch and WinWin | `Moloch and WinWin/` | 9 |
| Constructive Rebellion | `Constructive Rebellion/Essays/` | 10 |
| The Digital Commons | `The Digital Commons/` | 5 |
| The Wizard's Apprentice | `The Wizard's Apprentice/` | 6 |
| You Fascist Nazi | `You Fascist Nazi/Essays/` | 5 |
| On Portability | `On Portability/` | 5 |
| Digital Boundaries | `Digital Boundaries/` | 5 |
| The Silicon Zone | `The Silicon Zone/` | 5 |
| On Noise | `On Noise/` | 2 |
| On Politics | `On Politics/` | 3 |

---

## 2. LLM Client Infrastructure

### Location
`/Users/jrosenbaum/Documents/Code/Synthetic Conversations/shared/llm/`

### Architecture

**Factory pattern** with pluggable providers:
```python
from shared.llm import create_client

# Cloud (for outline generation)
client = create_client(provider="openrouter", model="google/gemini-2.0-flash-001")

# Local (for testing)
client = create_client(provider="lmstudio", model="local-model")
```

### BaseLLMClient Interface (`shared/llm/base.py`)

| Method | Signature | Returns | Use Case |
|--------|-----------|---------|----------|
| `chat()` | `(messages, temperature, max_tokens, **kwargs)` | `str` | Free-form text generation (outlines, essays) |
| `structured_output()` | `(messages, schema, temperature, max_tokens)` | `dict` | JSON schema-constrained responses (metadata extraction) |
| `test_connection()` | `()` | `bool` | Verify provider availability |
| `list_models()` | `()` | `list` | Enumerate available models |

### Available Providers (`shared/llm/factory.py`)

| Provider | Class | Config Source | Use Case |
|----------|-------|---------------|----------|
| `openrouter` | `OpenRouterClient` | `OPENROUTER_API_KEY` env var | Cloud inference, strong models |
| `lmstudio` | `LMStudioClient` | Host/port config | Local inference |
| `ollama` | `OllamaClient` | Host/port config | Local inference |
| `unsloth` | `UnslothClient` | Adapter path | LoRA inference |

### Configuration (`shared/llm/config.py`)

`LLMConfig` dataclass with `from_env(env_prefix)` classmethod:
- Reads `{PREFIX}_BACKEND` and `{PREFIX}_MODEL` from environment
- Supports `config_defaults` dict for programmatic configuration
- Supports OpenRouter provider routing (`{"order": ["Groq"], "allow_fallbacks": true}`)
- Auto-loads `.env` file from repo root

### Key for Pipeline

- Use `chat()` for outline generation from essays (free-form text)
- Use `structured_output()` for extracting structured metadata (title, themes, section headers)
- The `create_client()` factory accepts `config_defaults` dict, which is how SynthChat passes settings.yaml config

---

## 3. Existing Dataset Formats

### Location
`/Users/jrosenbaum/Documents/Code/Synthetic Conversations/Datasets/`

### JSONL Format Convention

**Standard format** (from actual dataset files):
```json
{
  "conversations": [
    {"role": "user", "content": "User request text"},
    {"role": "assistant", "content": "<thinking>...</thinking>", "tool_calls": [...]}
  ],
  "label": true,
  "behavior": "context_continuity"
}
```

### Key Rules

| Rule | Detail |
|------|--------|
| **No system message** | Conversations start with `user` role (system context embedded in user message or absent) |
| **label field** | `true` = positive example, `false` = negative (KTO requires both) |
| **behavior field** | Categorizer string for the behavior being trained |
| **Single-turn preferred** | Most examples are single user-assistant exchanges |
| **tool_calls format** | OpenAI-compatible format with `id`, `type`, `function.name`, `function.arguments` (stringified JSON) |

### Dataset Organization

```
Datasets/
├── behavior_datasets/
│   ├── thinking/          # With <thinking> blocks
│   │   ├── context_continuity/pairs_v1.0.jsonl
│   │   └── ... (8 behavior types)
│   └── non_thinking/      # Without thinking blocks
│       ├── context_continuity/pairs_v1.0.jsonl
│       └── ... (10 behavior types)
├── gspo_datasets/         # Group Sparse Policy Optimization
├── archive/legacy_snapshots/ # Previous versions
└── synthchat/             # SynthChat output directory
```

### Implications for Essay Pipeline

The essay pipeline output should follow the same JSONL conventions:
```json
{
  "conversations": [
    {"role": "user", "content": "<detailed outline>"},
    {"role": "assistant", "content": "<full essay text>"}
  ],
  "label": true,
  "behavior": "essay_writing"
}
```

For **real data** (reverse-engineered from existing essays): `label: true` (all positive examples from author's actual work).

For **synthetic data**:
- Good generations: `label: true`
- Failed/low-quality generations (from improvement engine failures): `label: false` (useful for KTO)

---

## 4. SynthChat Infrastructure

### Location
`/Users/jrosenbaum/Documents/Code/Synthetic Conversations/SynthChat/`

### Architecture Overview

```
SynthChat/
├── run.py              # CLI entry: generate | improve | validate
├── generator.py        # SynthChatGenerator - stage-by-stage generation
├── engine.py           # ImprovementEngine - judge/improve loop
├── config/
│   └── settings.yaml   # LLM config, targets, improvement settings
├── scenarios/
│   └── content_writing.yaml  # Self-contained scenario definitions
├── rubrics/            # YAML quality rubrics
├── services/           # Core services (judge, improve, validate)
│   ├── core/           # JudgeService, ImprovementService, etc.
│   ├── data/           # RubricRepository, rubric loading
│   ├── parsing/        # ConversationParser, ScopeExtractor
│   └── scope_handlers/ # Per-scope improvement handlers
└── utils/
    ├── docs_loader.py  # DocsLoader - load markdown files as seed data
    └── prompt_renderer.py
```

### Key Components

#### SynthChatGenerator (`generator.py`)

Stage-by-stage generation pipeline:
1. Generate system prompt -> validate with rubrics
2. Generate user request -> validate
3. Generate thinking (optional) -> validate
4. Generate assistant response -> validate

**Critical feature**: `doc_context` parameter. When `--docs` flag is used, the generator:
- Loads markdown files via `DocsLoader`
- Makes `{doc_content}` and `{doc_path}` available as template variables in scenario prompts
- Generates one example per document

#### DocsLoader (`utils/docs_loader.py`)

```python
@dataclass
class DocFile:
    path: str
    content: str

class DocsLoader:
    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.html', '.htm'}
    DEFAULT_MAX_CHARS = 50_000

    def load(self, path: str) -> List[DocFile]
```

- Loads single file or entire folder
- Strips HTML tags from .html files
- Truncates at 50,000 chars
- Skips binary and empty files
- **Does NOT currently parse YAML frontmatter** -- loads raw content
- **Does NOT currently recurse subdirectories** -- `_load_folder` uses `iterdir()` which is flat

#### ImprovementEngine (`engine.py`)

Judge -> improve loop orchestrator:
- Takes `llm_client` from `shared.llm`
- Processes scopes in order: system_prompt -> thinking -> response
- Supports retry with exponential backoff
- Logs judge and improver interactions
- Returns `ImprovementResult` with scores, pass/fail, and improved example

#### Scenario YAML Format (`scenarios/content_writing.yaml`)

Self-contained scenario definitions. Key structure:
```yaml
scenarios:
  contentManager_write_blog:
    type: tool
    max_tokens: 4096
    rubrics:
      system_prompt: [system_prompt_format]
      response: [tool_alignment, content_writing_quality]
    prompts:
      system: |
        <generation instructions for system prompt>
      user: |
        <generation instructions for user message>
      assistant: |
        <generation instructions for assistant response>
```

Template variables available: `{doc_content}`, `{doc_path}` (when using --docs mode).

#### Settings (`config/settings.yaml`)

Current LLM configuration:
- **Generation**: `openrouter` / `google/gemini-2.0-flash-001` (temp 0.7, max_tokens 4096)
- **Improvement**: `openrouter` / `openai/gpt-oss-120b` (temp 0.1, max_tokens 2048)
- **Max iterations**: 10
- **Checkpointing**: Every 10 examples
- **Parallel workers**: Supported via `--workers` flag

### Docs-Based Generation Mode

The existing `--docs` mode in `run.py` (lines 140-166):
```python
if args.docs:
    docs = DocsLoader().load(args.docs)
    for doc in docs:
        batch_results = generator.generate_batch(
            targets=targets,
            max_iterations=max_iterations,
            doc_context=doc  # Makes {doc_content} available
        )
```

This is the closest existing pattern to what the essay pipeline needs, but it currently:
- Generates tool-call formatted output (not plain essay text)
- Uses tool-based scenarios (contentManager_write_*)
- Does not strip social media copy sections
- Does not extract YAML frontmatter metadata
- Does not load subdirectories recursively

---

## 5. Improvement Engine

### Active Code Location
`/Users/jrosenbaum/Documents/Code/Synthetic Conversations/SynthChat/engine.py` and `SynthChat/services/`

### Legacy Location
`/Users/jrosenbaum/Documents/Code/Synthetic Conversations/improvement_engine/` -- contains only `.pyc` cache files, no active Python source. This is a dead directory.

### How the Engine Works

1. **Load rubrics** by key from `SynthChat/rubrics/` YAML files
2. **Group rubrics by scope** (system_prompt, thinking, response)
3. **For each scope**, run improvement loop:
   a. Validate example with structural validators (XML, JSON, regex, YAML, code)
   b. Build judge prompt from rubrics + validation results
   c. Call LLM judge for scoring
   d. Auto-fail on validation errors (score forced to 0.0)
   e. If failed: build improvement prompt, call LLM improver
   f. Re-evaluate improved version
   g. Repeat up to `max_iterations`
4. Return `ImprovementResult` with final example, scores, and pass/fail

### Relevance to Essay Pipeline

The improvement engine can be reused for quality control on synthetic essays:
- Create new rubrics for essay quality (style matching, coherence, outline fidelity)
- The scope system supports custom scopes beyond the current system_prompt/thinking/response
- The judge -> improve loop works generically on any content type

---

## 6. Recommendations for Pipeline Architecture

### Priority 1: Build as SynthChat Scenarios

Create new scenario YAML files under `SynthChat/scenarios/essay_writing.yaml` for both dataset types:

**Scenario A: Real Data (Reverse-Engineer Outlines)**
- Use `--docs` mode with the essay corpus directory
- Scenario prompts instruct the LLM to generate a detailed outline from `{doc_content}`
- Output format: `user=outline`, `assistant=original essay text` (verbatim from corpus)
- No improvement loop needed on the essay itself (it's the author's real writing)
- Optional quality check on the generated outline only

**Scenario B: Synthetic Data (Generate New Essays)**
- Two-phase generation: first generate outline, then generate essay from outline
- Use improvement engine with essay-quality rubrics to ensure style matching
- Failed generations saved with `label: false` for KTO training

### Priority 2: Enhance DocsLoader

The current `DocsLoader` needs extensions for the essay corpus:
- **Recursive directory loading**: Current `_load_folder` only iterates flat; essays are in subdirectories
- **YAML frontmatter parsing**: Extract `title`, `series`, `tags` for metadata enrichment
- **Content preprocessing**: Strip social media sections, handle MoC files
- **Filtering**: Skip Research/ dirs, `_` prefixed files, non-essay content

### Priority 3: Create Essay-Specific Rubrics

New YAML rubrics for `SynthChat/rubrics/`:
- `essay_outline_quality`: Evaluates whether generated outlines capture the essay's structure, themes, and rhetorical arc
- `essay_style_matching`: For synthetic essays, evaluates alignment with the author's voice, vocabulary, and essay patterns
- `essay_coherence`: Structural coherence, logical flow, conclusion quality

### Priority 4: Output Format

Follow existing JSONL conventions:
```json
{
  "conversations": [
    {"role": "user", "content": "<detailed outline with structure, themes, rhetorical notes>"},
    {"role": "assistant", "content": "<full essay markdown text>"}
  ],
  "label": true,
  "behavior": "essay_writing",
  "metadata": {
    "source": "real|synthetic",
    "series": "Divine Sparks",
    "essay_title": "Prometheus",
    "tags": ["prometheus", "greek-mythology", "AI-development"]
  }
}
```

Save to `Datasets/essay_datasets/` to follow existing organizational patterns.

### Priority 5: Pipeline Commands

Extend SynthChat CLI:
```bash
# Real data: generate outlines from existing essays
python -m SynthChat.run generate --scenarios essay_outline --docs "Meditations on Alignment/"

# Synthetic data: generate new essays from outlines
python -m SynthChat.run generate --scenarios essay_synthetic --workers 4
```

---

## 7. Risks and Constraints

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Outline quality varies by essay complexity | High | Medium | Use stronger model (Opus/Sonnet) for outline generation; add outline rubric |
| Style drift in synthetic essays | Medium | High | Create style-matching rubric with example excerpts from corpus |
| Social media sections contaminate training data | Low | High | Preprocessing filter in enhanced DocsLoader |
| DocsLoader 50K char truncation loses essay content | Low | Low | Most essays are <10K chars; increase limit if needed |
| OpenRouter rate limits with 100+ essays | Medium | Low | Use `--workers 1` and checkpoint; retry logic already built in |
| Series context lost in individual essay processing | Medium | Medium | Include series metadata in outline prompt; consider multi-essay context windows |

---

## 8. Self-Verification Checklist

- [x] All sources are authoritative (direct codebase examination, not secondhand)
- [x] Version/path accuracy verified (actual file paths confirmed via Glob/Read)
- [x] Security implications documented (API keys handled via env vars, no credentials in code)
- [x] Alternative approaches considered (standalone pipeline vs SynthChat extension)
- [x] Documentation organized with clear sections and tables
- [x] All technical terms defined or linked to code
- [x] Recommendations backed by concrete evidence from codebase analysis

---

## Source Files Referenced

| File | Purpose |
|------|---------|
| `shared/llm/__init__.py` | LLM client public API |
| `shared/llm/base.py` | BaseLLMClient ABC interface |
| `shared/llm/factory.py` | Client factory with provider routing |
| `shared/llm/config.py` | LLMConfig dataclass and env loading |
| `SynthChat/run.py` | CLI entry point with generate/improve/validate |
| `SynthChat/generator.py` | Stage-by-stage generation with doc_context |
| `SynthChat/engine.py` | ImprovementEngine judge/improve loop |
| `SynthChat/utils/docs_loader.py` | DocsLoader for markdown file loading |
| `SynthChat/config/settings.yaml` | LLM and generation configuration |
| `SynthChat/scenarios/content_writing.yaml` | Scenario YAML format reference |
| `Datasets/behavior_datasets/thinking/context_continuity/pairs_v1.0.jsonl` | Dataset format reference |
| `Meditations on Alignment/The Games we Play.md` | Essay with full frontmatter |
| `Meditations on Alignment/Divine Sparks/01. Prometheus.md` | Series essay with series/essay_number |
| `Meditations on Alignment/AI Familiars.md` | Essay with social media copy prefix |
| `Meditations on Alignment/On Portability/_On Portability.md` | MoC/outline file with inline essay |
| `Meditations on Alignment/Meditations on Alignment Preamble.md` | Standalone essay with transcript field |
