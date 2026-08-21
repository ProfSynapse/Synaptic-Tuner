# Product Naming Brainstorm

> Status: exploratory. Captured 2026-08-21 on `claude/product-name-brainstorm-85wqvc`.
> Purpose: name the hosted/UI product version of this repo.

---

## 1. What the product actually is

Reading the repo end to end, this is not a fine-tuning tool. Fine-tuning is one
stage of a **closed loop**:

```
define behavior -> SynthChat generates env-backed data -> SFT -> KTO -> env-GRPO
   -> Evaluator (rubrics + LLM judge) -> merge/quantize/publish
   -> hosted inference (vLLM + LoRA hot-swap)
   -> logging proxy captures real traffic
   -> flywheel cleans/tags/stages -> retrain
```

Plus `MechInterp/` — probes, frozen directions, activation steering — which is a
genuinely differentiated layer almost nobody else in this category ships.

So the product's real claim is something like: **"a small model that gets good at
your one thing, and keeps getting better because it's serving your traffic."**
Data generation, training, eval, serving and the feedback loop are one system.

## 2. The problem with "Synaptic Tuner"

Three issues, in order of severity:

1. **"Tuner" names the narrowest stage.** The moat is the loop plus hosted
   inference, not the tuning step. A tuner is a utility; this is a platform.
   The name caps the story you can tell.
2. **"Synapse" is contested in this exact category.** Microsoft shipped Azure
   Synapse Analytics into data/AI. A standalone "Synapse" play fights uphill for
   search and mindshare. "Synaptic" is meaningfully more available.
3. **Two-word compounds age poorly.** Nobody says "Synaptic Tuner" twice — it
   gets clipped to "Synaptic" or "the Tuner" anyway. Pick the clip.

## 3. Brand architecture question (answer this first)

Three assets already exist and they need a deliberate relationship:

| Asset | Current role | Options |
|-------|--------------|---------|
| **Professor Synapse** | creator/persona brand, existing audience | keep as the human brand; product sits under it, or fully separate |
| **Nexus** | model family (Nexus-Quark-L2 etc.) | keep as model family regardless — it works there |
| **Synaptic Tuner** | the repo/toolkit | becomes the product name, or stays as the OSS core under a new product name |

**Recommendation:** keep `Nexus` as the model family, keep `Professor Synapse` as
the human brand, and give the platform a **distinct, ownable name**. The
persona-brand carryover is real but it also anchors you to "prompt guy on the
internet" — a headwind if the buyer is an engineering team. A separate platform
name lets Prof Synapse be the distribution channel without being the ceiling.

A common and effective pattern: OSS core keeps `Synaptic Tuner`, hosted product
gets the new name. You lose nothing and gain a clean commercial surface.

## 4. Naming directions

### Direction A — Neuroscience of *learning* (extends the Synapse lineage)

Not "brain = smart", but the specific mechanism by which use strengthens a
pathway. That mechanism *is* your flywheel.

| Name | Why it works | Watch out |
|------|--------------|-----------|
| **Myelin** | Myelin sheaths the pathways you use most, making that specific signal fast. Perfect metaphor for a specialist model: general models are slow and diffuse; you make one path fast. 6 letters, unambiguous spelling. | slightly clinical |
| **Engram** | The physical trace a learned behavior leaves in tissue. Literally "the encoded skill." Distinct, memorable, no big incumbent. | some sci-fi/Scientology baggage |
| **Potentiate** | Long-term potentiation = repeated activation strengthens the synapse. Your flywheel, verbatim. Verb-able: "potentiate your model." | 3 syllables, a mouthful |
| **Hebb / Hebbian** | "Neurons that fire together wire together." The learning-from-use rule, named. Short, sharp. | obscure outside neuro/ML |
| **Arbor** | Dendritic arbor — the branching structure. Warm, natural, easy. | generic, likely taken |

### Direction B — Metallurgy / making (the loop as craft)

Raw material in, durable specific object out. Reads well to engineers.

| Name | Why it works | Watch out |
|------|--------------|-----------|
| **Anneal** | An actual training-schedule term (LR annealing) *and* controlled heating/cooling to strengthen. Double meaning both of which are literally true. Verb-able. | mild "is that a real word" friction |
| **Temper** | Metallurgy (tempering steel) + **temperament** (behavior — what you're training) + sampling temperature. Triple meaning, all on-target. Strongest verb of the set: "we tempered a Qwen 3B." | common English word; SEO/domain will be hard |
| **Lathe** | Shape a general blank into a precise part. Concrete, tool-like, 5 letters. | less loop-y, more single-step |
| **Kiln** | Fire raw material into something durable. 4 letters. | **likely direct collision** — believe there's an existing OSS fine-tune/eval tool named Kiln AI. Verify before falling in love. |

*Avoid in this direction:* Forge, Foundry, Crucible, Smith — Palantir Foundry and
Azure AI Foundry own "Foundry," and the rest are the most-picked names in dev tooling.

### Direction C — Teacher / student (the distillation story)

| Name | Why it works | Watch out |
|------|--------------|-----------|
| **Understudy** | An understudy learns the role by watching the lead perform, then takes over. That is *exactly* teacher-model distillation plus learning from production traffic. Charming, precise, zero collisions. | 3 syllables, a little soft for enterprise |
| **Protégé** | Teacher -> student, unmistakable. | accent character is a real UX tax in domains/CLIs |
| **Cadre / Cohort** | A trained specialist corps. | generic |

### Direction D — Coined / abstract

The category is saturated with these (Modal, Baseten, Vellum, Tessl, Fireworks).
A coined name here reads as "me-too infra startup" and forces you to spend
marketing dollars teaching people what it means. **Not recommended** given you
have a genuinely ownable metaphor available in A and B.

## 5. Shortlist

Ranked on: metaphor accuracy, verb-ability, availability odds, and how well it
scales from OSS repo to enterprise platform.

1. **Myelin** — best metaphor-to-product fit. Neuro lineage keeps continuity with
   Prof Synapse without being derivative. Says "specialized and fast," which is
   the actual pitch for small trained models. Low collision risk.
2. **Anneal** — the ML pun is real, not decorative. Engineers will get it
   instantly and like that it's technically literate. Verb-able, ownable.
3. **Temper** — the strongest brand *voice* of the group and the best verb, but
   the commonness of the word is a genuine acquisition-cost problem.
4. **Engram** — most distinctive and most "product" sounding of the set. Good if
   you want the memory/knowledge framing over the speed framing.
5. **Understudy** — the most human and most differentiated. A real option if you
   want to feel approachable rather than infrastructural.

**If forced to pick one today: Myelin.** It carries the Synapse lineage forward,
the metaphor is exactly right (make one pathway fast), it's short, spellable,
verb-adjacent, and it doesn't fight an incumbent.

## 6. Naming the surfaces

Whatever the top-level name, the sub-surfaces should stay descriptive so the
product self-documents. Illustrative, using Myelin:

| Surface | Name |
|---------|------|
| Platform | Myelin |
| OSS core | Synaptic Tuner (unchanged) or `myelin-core` |
| Model family | Nexus (unchanged) |
| Data generation | Studio / Generate |
| Training + experiments | Runs / Train |
| Eval harness | Judge / Evals |
| Hosted inference | Serve |
| Retraining loop | Flywheel (already the internal name — keep it, it's good) |
| Interpretability | Lens or Probe (`MechInterp` -> Lens is a nice upgrade) |

## 7. Next steps

- [ ] Decide brand architecture (§3) — separate platform name vs. Synaptic house
- [ ] Narrow to 2-3 candidates
- [ ] Verify: `.com`/`.ai` availability, USPTO/EUIPO word marks in class 9/42,
      GitHub org, npm/PyPI namespace, X and LinkedIn handles
- [ ] Say each finalist out loud in a sentence: "We built it on ___." /
      "___ trained the model." — the one that survives casual speech wins
