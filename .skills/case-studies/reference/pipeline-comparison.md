# Pipeline Comparison: Tool Calling vs Essay Style

Side-by-side comparison of how two common training pipelines differ at each
stage of the universal pipeline.

---

## Stage 1: Define The Capability

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| Source of truth | Configured tool schemas and response format specs | Reference corpus |
| Format spec | Structured tool-call response shape | Markdown/prose structure |
| Behavioral spec | Scenario rubrics and environment gates | Quality rubrics |
| Correct defined as | Right action, right arguments, right state handling | Accurate structure, voice, and content |
| Key constraint | Schema-owned required fields and action payloads | Corpus-owned tone and structure |

---

## Stage 2: Create Training Data

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| Scenario type | Environment-backed or schema-backed tool scenarios | Docs-based scenarios |
| Input to generator | Tool schema, environment config, and prompt template | Source document content |
| User turn | Natural language task request | Reverse-engineered brainstorm or request |
| Assistant turn | Configured tool-call payload, then text when complete | Structured natural-language response |
| System prompt | Optional and deployment-aligned | Often omitted or minimal |
| Scaling strategy | More scenarios, environments, and action paths | More source documents and variations |

### Template vs Docs

Tool-calling data usually uses templates plus generated environments so the
same behavior appears across many states and surface details. Essay-style data
is usually anchored to reference documents so the model learns the target voice,
structure, and specificity from real examples.

---

## Stage 3: Validate And Improve

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| Primary validation | Deterministic schema/environment checks | LLM-judged rubric scoring |
| Automatically checkable | JSON shape, required fields, action existence, environment state | Section count and required blocks |
| Human review focus | Edge cases in action choice and recovery behavior | Voice, specificity, and judgment |
| Improvement mechanism | Config/rubric-driven repair loops | Rubric-driven improvement |
| Common structural errors | Missing fields, malformed action payloads, wrong action order | Generic headings, bland content, missing sections |

---

## Stage 4: Train

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| SFT target | Structural shape and simple one-turn usage | Target response form and style |
| GRPO target | Multi-step action behavior in an environment | Usually less central unless a verifier exists |
| KTO negative sources | Invalid actions, missing context, unsafe behavior | Generic or low-quality variants |
| Dataset size | Often hundreds to thousands | Often tens to hundreds |

The trainer commands can be identical. The capability changes because the
dataset, reward surface, and evals change.

---

## Stage 5: Evaluate

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| Evaluation type | Schema, environment, and behavior checks | Rubric scoring by judge LLM |
| PASS criteria | Correct configured response shape and successful task completion | Required parts present and high quality |
| WARN criteria | Completed task with inefficient or noisy path | Structure present but generic |
| FAIL criteria | Malformed payload, wrong action, failed environment task | Missing major sections or off-topic |
| Key metrics | Schema pass, environment pass, behavior pass, by-tag rates | Rubric dimension scores |

---

## Stage 6: Iterate

| Aspect | Tool Calling | Essay Style |
|--------|-------------|-------------|
| Failure signal | Schema errors, environment traces, wrong action coverage | Low rubric dimensions |
| Fix strategy | Add scenarios for weak behaviors and tighten config gates | Add source documents or targeted variants |
| Dataset expansion | Usually easy if environments are generated | Limited by corpus and quality review |
| Convergence speed | Faster when deterministic validation is strong | Slower when quality is subjective |

---

## Decision Guide

Use the tool-calling pattern when output is structured, correctness can be
checked against schemas or environment state, and you need high-volume examples.

Use the essay-style pattern when the output is natural language, quality is a
spectrum, and a reference corpus defines the target behavior.

Hybrid capabilities can use deterministic structure validation for the tool or
template portion and rubric scoring for prose quality.
