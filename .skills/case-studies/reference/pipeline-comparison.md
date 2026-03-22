# Pipeline Comparison: Tool Calling vs Essay Style

Side-by-side comparison of how the two training pipelines differ at each stage of the universal pipeline.

---

## Stage 1: Define the Capability

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| **Source of truth** | `tool-schemas.json` — JSON schema per tool | Essay corpus in `Meditations on Alignment/` |
| **Format spec** | OpenAI function calling format with `tool_calls` field | Markdown outline with title, overview, sections, tone, themes |
| **Behavioral spec** | 6 YAML behavior rubrics (intellectual humility, verification, etc.) | 2 quality rubrics (brainstorm quality, outline quality) |
| **"Correct" defined as** | Right tool + right arguments + right context | Accurate structure + dialectical challenge + specific details |
| **Key constraint** | Context object with 4 required fields (`workspaceId`, `sessionId`, `memory`, `goal`), `memory` never empty | Second person address, 4-6 sections matching essay, no generic headings |

---

## Stage 2: Create Training Data

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| **Scenario type** | `type: tool` — template-based generation | `type: docs_based` — derived from source documents |
| **Input to generator** | Tool schema + prompt template | Actual essay content (`{doc_content}`) |
| **User turn** | Natural language request ("Create a folder for Q4") | Reverse-engineered brainstorm (messy, fragmented, 75-200 words) |
| **Assistant turn** | Tool call with `<thinking>` block + arguments | Structured outline with dialectical opening |
| **System prompt** | Yes — provides session/workspace context | No — model learns to outline from brainstorm alone |
| **Data sources** | Handcrafted seeds + SynthChat + self-play (3 sources) | SynthChat docs-based generation (1 primary source) |
| **Scaling strategy** | More prompt templates × temperature variations | More essays in corpus × per-doc variations |
| **Generation command** | `--scenarios tools` | `--docs "path/" --scenarios essay_outline --per-doc 1` |

### Template vs Docs: The Key Difference

**Tool calling** uses prompt templates that can generate infinite variations:
```yaml
# Each run generates different user requests and tool calls
prompts:
  user: "Generate a natural user request that would require creating a folder."
  assistant: "Generate assistant response with useTools call using storageManager.createFolder."
```

**Essay style** is anchored to real documents — each essay produces one training example:
```yaml
# Each essay file → one brainstorm + outline pair
prompts:
  user_system: |
    You have access to the finished essay below.
    <finished_essay>{doc_content}</finished_essay>
    Imagine what the author's messy thoughts looked like BEFORE writing.
```

---

## Stage 3: Validate & Improve

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| **Primary validation** | Deterministic schema checking (`validate_syngen.py`) | LLM-judged rubric scoring (SynthChat validate) |
| **What's checkable automatically** | JSON structure, required fields, ID patterns, tool existence | Format presence (has overview? has tone?), section count |
| **What needs human review** | Edge cases in tool selection logic | Voice accuracy, dialectical quality, specificity |
| **Improvement mechanism** | Schema fix scripts + SynthChat improve | SynthChat improve with quality rubrics |
| **Common structural errors** | Missing context fields, wrong tool name, empty `memory` | Generic headings, too many sections, third-person address |
| **Validation command** | `python3 .skills/synethetic-data-generation/scripts/validate_syngen.py FILE` | `python -m SynthChat.run validate -i FILE --rubrics essay_*` |

### Validation Confidence

```
Tool Calling:   [██████████████████████████████] 95% automated
Essay Style:    [█████████████░░░░░░░░░░░░░░░░░] 45% automated
                └── Structure checks  ─┘└── Voice, quality need human judgment ─┘
```

---

## Stage 4: Train

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| **SFT dataset** | Positive tool call examples (all labels true or absent) | Positive outline examples |
| **SFT learning target** | Tool call syntax, context object, `<thinking>` blocks | Outline structure, dialectical opening, section format |
| **KTO negative sources** | High-temp self-play errors, wrong tool selection, missing fields | Systematic degradation (generic headings, bland affirmation, bloat) |
| **KTO learning target** | Prefer clarification > blind action, prefer complete context > lazy context | Prefer specific > generic, prefer dialectical > affirmative, prefer concise > bloated |
| **Training commands** | Identical — same trainers, same flags, different `--local-file` | Identical — same trainers, same flags, different `--local-file` |
| **Typical dataset size** | 1000-3000 examples | 50-200 examples (limited by corpus size) |

### The Training Commands Are Identical

```bash
# Tool calling SFT
python train_sft.py --model-size 7b --local-file ../../Datasets/tools_sft.jsonl

# Essay style SFT
python train_sft.py --model-size 7b --local-file ../../Datasets/essay_outlines.jsonl

# Same command. Different data. Different capability.
```

---

## Stage 5: Evaluate

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| **Evaluation type** | Schema match + behavior check | Rubric scoring by judge LLM |
| **PASS criteria** | Correct tool + correct arguments + good behavior | All outline parts present + specific + dialectical |
| **WARN criteria** | Right tool but suboptimal behavior | Structure present but generic or bland |
| **FAIL criteria** | Wrong tool, missing tool, or error | Missing major outline sections, completely off-topic |
| **Key metrics** | `schema_pass_rate`, `behavior_pass_rate`, `by_tag` | Rubric dimension scores (structure, specificity, voice, dialectic) |
| **Comparison baseline** | Base model tool calling accuracy | Base model outline quality |

---

## Stage 6: Iterate

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| **Failure signal** | Schema validation errors, wrong tool counts | Low rubric dimension scores |
| **Fix strategy** | Generate more examples for weak tools/behaviors | Add more essays with the failing characteristics |
| **Dataset expansion** | Easy — more prompt templates, more self-play | Harder — need more source essays or essay variations |
| **Convergence speed** | Fast — deterministic validation, clear pass/fail | Slower — subjective quality, harder to measure improvement |

---

## Decision Guide: Which Pattern to Follow?

Use the **tool calling pattern** when:
- Output is structured data (JSON, function calls, API requests)
- "Correct" is binary (right tool or wrong tool)
- You can define a schema that covers all valid outputs
- You need high volume (1000+ examples)
- Validation can be mostly automated

Use the **essay style pattern** when:
- Output is natural language (prose, outlines, summaries, creative text)
- "Correct" is a quality spectrum (good, mediocre, bad)
- Quality is defined by rubrics, not schemas
- You have a reference corpus to learn from
- Voice, tone, and style matter

### Hybrid Capabilities

Some capabilities blend both patterns:
- **Tool call with rich explanation** — tool calling format + natural language quality
- **Structured report generation** — template structure + prose quality
- **Code generation with comments** — syntax correctness + explanation quality

For hybrids, use tool calling validation for structure and essay-style rubrics for prose quality.

---

## Summary Table

| Pipeline Stage | Tool Calling | Essay Style | Shared |
|---------------|-------------|-------------|--------|
| Define | JSON schemas + behavior rubrics | Essay corpus + quality rubrics | Rubric-driven quality definition |
| Generate | Template scenarios + self-play | Docs-based scenarios | SynthChat engine |
| Validate | Schema validation (deterministic) | Rubric scoring (LLM-judged) | SynthChat validate/improve |
| Train | SFT → KTO → GRPO | SFT → KTO | Same trainers, same commands |
| Evaluate | Schema + behavior tests | Rubric dimension scoring | Same evaluator framework |
| Iterate | More templates, more self-play | More essays, more degradation variants | Failure analysis → targeted generation |
