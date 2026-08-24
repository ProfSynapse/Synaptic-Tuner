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

---

# Round 2 — Landscape research + revised direction

> Added after ICP clarification (regulated/mission-driven orgs; open-core
> architecture) and feedback that the round-1 shortlist was off-register.

## 8. How this space actually names itself

Six observable registers:

| Register | Examples | What it signals |
|---|---|---|
| **Mythic / literary referent** | Palantir (Tolkien seeing-stone), Anduril (Tolkien reforged sword), Prime Intellect (Roger Williams novel), Sauron | ambition, inevitability, "we are the protagonists" |
| **Historical figure** | Vannevar Labs (Vannevar Bush) | intellectual lineage, institutional seriousness |
| **Greek / philosophical abstract** | Nous Research (*nous* = intellect), EleutherAI (*eleutheria* = freedom), Anthropic | seriousness without swagger; ideas-first |
| **Nature / collective metaphor** | Sakana (Japanese "fish" — school of fish as collective intelligence), Cohere | emergence, systems thinking |
| **Compressed infra-descriptive** | Predibase, Baseten, Together, Fireworks, Anyscale, OpenPipe, Modal, Lamini | commodity tier; interchangeable |
| **Compliance-plain** | Hathr AI, HIPAA Vault | trustworthy, forgettable |

**Round 1 sat in a seventh register — craft/plain/institutional — that basically
nobody in this category uses.** That's why it read wrong.

## 9. The trap in the register you like

Mythic-referent names are now **politically coded**. Palantir and Anduril have
made Tolkien-naming shorthand for surveillance and defense tech. The ICP here is
health systems, nonprofits, government, civil society — the exact audience most
primed to pattern-match a mythic name to Palantir. Prime Intellect gets away
with it because it sells compute to researchers; you would be selling to a
nonprofit ED or a public-health CIO.

**So: take the register, drop the mythos.** The adjacent register with the same
gravity and none of the coding is **Greek/Latin philosophical abstraction** —
Nous, Eleuther, Anthropic. It also opens virtue and civic vocabulary, which the
mission half of the ICP responds to and the defense-coded names can't touch.

## 10. Candidates in-register

### Two-word, Prime Intellect structure ([qualifier] + [faculty of mind])

| Name | Thesis it encodes |
|---|---|
| **Tacit Intellect** | Polanyi's tacit knowledge — "we know more than we can tell." Exactly what you extract from an org and encode into a model. "Tacit" also reads as discreet/unspoken -> privacy, without a compliance word. |
| **Practical Wisdom** | Plain-English *phronesis*. The distinction between knowing-in-general and knowing-what-to-do-here is the whole small-model argument. Accessible; slightly soft. |
| **Latent Reason** | What's already implicit in the org's data, made explicit. Nods at latent space without being jargon. |

### Single classical word

| Name | Thesis it encodes |
|---|---|
| **Phronesis** | Aristotle's *practical wisdom* — right action in a particular situation, explicitly distinguished from *episteme* (abstract theory) and *techne* (craft). This is the frontier-model-vs-domain-specialist distinction, named 2,300 years ago. Best pure concept fit of anything generated. Spelling is the tax. |
| **Paideia** | The formation of a person into their full civic role — training toward a function, with a public/civic charge built in. Lands hard with gov and nonprofit. |
| **Vernacular** | The local tongue of a specific community, as against the official/imperial one. Your model speaks your org's vernacular; the frontier lab's speaks the empire's. Anti-imperial framing that mission orgs will love. English word, easy spelling. |
| **Anamnesis** | Plato: learning as recollection, drawing out what was latent. *Also* the clinical term for the patient history a physician takes. Double meaning lands precisely on the health ICP. |
| **Custos** | Latin: guardian, keeper, custodian (*quis custodiet*). Stewardship + custody, in one classical word, without saying "secure." |
| **Ingenium** | Latin: innate character and natural capacity; the root of both "engine" and "ingenious." Grand, ownable, ambiguous enough to grow into. |

## 11. Revised recommendation

1. **Tacit Intellect** — closest to the register you responded to, and the thesis
   is exactly right. Clips to "Tacit," which is a good standalone.
2. **Vernacular** — best positioning story against frontier labs, easiest to
   spell and say, strongest fit for mission-driven buyers.
3. **Phronesis** — deepest concept fit; highest spelling/pronunciation tax.
4. **Anamnesis** — the health-specific play, if health is the beachhead.

## 12. Open-core naming

Prime Intellect's shape is worth copying: company **Prime Intellect**, OSS stack
`prime`, model family `INTELLECT-1/2`. Direct analogue here:

| Layer | Name |
|---|---|
| Company / product | new name (above) |
| OSS core (submodule) | **Synaptic Tuner** — keep it; "tuner" is *correct* for a library and humble names are an asset in OSS |
| Model family | **Nexus** — keep |
| Creator brand | **Professor Synapse** — distribution channel |

## 13. Sources

- https://en.wikipedia.org/wiki/The_Metamorphosis_of_Prime_Intellect
- https://en.wikipedia.org/wiki/Vannevar_Labs
- https://en.wikipedia.org/wiki/EleutherAI
- https://en.wikipedia.org/wiki/Sakana_AI
- https://research.contrary.com/company/goodfire
- https://www.thebulwark.com/p/how-the-tech-right-learned-to-love-mordor-jrr-tolkien-palantir-thiel
- https://techcrunch.com/2026/01/28/tiny-startup-arcee-ai-built-a-400b-open-source-llm-from-scratch-to-best-metas-llama/

---

# Round 3 — The blacksmith lexicon

## 14. Why this can work where round 1's metallurgy didn't

Round 1 offered Temper, Kiln, Forge, Anneal — the *common* words. Common craft
words read as homey and generic, which is why they landed badly.

But the obscure end of the smithing lexicon behaves completely differently. Words
like *swage*, *stithy*, *billet* have the same property that made **Phronesis**
appealing: **you have to know something to get it.** That's the actual quality
being responded to in the Prime Intellect register — not "mythic," but
"earned." An obscure trade term earns it through craft knowledge instead of
classical education, and it does so without the Palantir/Anduril coding problem.

Note that Anduril is *literally a forged sword*. Mythic smith names (Wayland,
Vulcan, Mjolnir, Ilmarinen) are the single most defense-coded corner available.
Avoid the smith-gods entirely; mine the tools and processes instead.

## 15. The lexicon, mapped to the pipeline

| Smithing term | What it means | What it maps to |
|---|---|---|
| **Swage** | a shaped die that imposes a specific repeatable profile on hot metal; a *swage block* holds many profiles at once | fine-tuning, exactly. And a swage block = a platform of many specialist forms |
| **Wrought** | worked by hand, individually shaped (opposite of *cast*, which is poured into a mold at scale) | the whole positioning: cast = frontier model, wrought = yours |
| **Billet** | the stacked, forge-welded stock drawn out into a blade; in pattern-welding, the accumulated layers | base model + dataset + adapters, accumulated |
| **Stithy** | archaic for smithy (Shakespeare, *Hamlet*: "Vulcan's stithy") | the workshop itself |
| **Heat** | one heating-and-working cycle; smiths count work in heats | an epoch / a training run |
| **Hardy** | the tool seated in the anvil's hardy hole; also the adjective (robust) | a fixed tool the work is shaped against |
| **Fuller** | tool that spreads and grooves metal | shaping |
| **Mandrel** | the form you shape material around | a target spec |
| **Scarf** | the joint shaped before forge-welding two pieces into one | model merging |
| **Welding heat** | the temperature at which two pieces become one metal | LoRA merge |
| **Normalize** | heat and air-cool to relieve stress and refine grain | (real ML homonym, but too generic a word) |
| **Running the colors** | watching oxide run straw -> bronze -> purple -> blue to judge temper | eval / monitoring |
| **Pritchel** | the round hole in the anvil, and the punch for it | — |
| **Tuyere** | the pipe feeding air into the fire | the inference proxy, arguably |

## 16. Picks

1. **Swage** — best conceptual fit in the entire lexicon. A swage imposes a
   specific, repeatable form on general stock; a swage block holds many forms in
   one body. That is a fine-tuning platform, described in one syllable. Obscure,
   ownable, sounds technical rather than cute. Verified: no AI company on it.
2. **Wrought** — carries the strongest *pitch*, not just the strongest name.
   Cast iron is poured into a mold at scale, identical every time, and brittle.
   Wrought iron is worked individually, and it's tough. Cast vs. wrought **is**
   frontier-model vs. trained-for-you. Ordinary English, no spelling tax, and
   "well-wrought" carries quality. Verified clean.
3. **Billet** — short, concrete, physical. The layered stock before it becomes a
   blade. Reads well next to `Nexus` as a model family. Verified clean.
4. **Stithy** — the literary bridge: archaic, Shakespearean, and a workshop.
   Almost certainly available. Highest obscurity, so highest explanation cost.
5. **Hardy** — warm and reassuring for regulated buyers via the adjective, but
   reads as a surname (Thomas/Tom Hardy).

## 17. Dead on arrival

| Name | Why |
|---|---|
| **Anvil** | four tech companies, incl. anvil.ai (defense/public safety — wrong neighbors for this ICP) and useanvil.com (document AI) |
| **Flux** | Black Forest Labs' FLUX image models |
| **Bloom** | BigScience's BLOOM LLM |
| **Wayland** | the Linux display server protocol |
| **Forge / Foundry / Crucible / Smith** | saturated; Palantir and Azure own "Foundry" |
| **Sampo** | Finnish financial group |
| Smith-gods generally | Vulcan/Mjolnir/Ilmarinen — maximally defense-coded |

*Minor flag:* Swagelok (fluid systems, est. 1947) exists and could complicate a
"Swage" word mark. Different class, but worth a real search.

## 18. The system underneath

The lexicon's real value may be as **internal vocabulary** regardless of the
top-level name — it gives the product a coherent voice most competitors lack:

| Surface | Term |
|---|---|
| a training run | a **heat** |
| the workshop / project view | the **stithy** |
| base model + data + adapters | the **billet** |
| the target form being trained toward | the **swage** |
| model merging | **forge weld** / **welding heat** |
| eval dashboard | **running the colors** |
| the finished model | **wrought** |

## 19. Sources (round 3)

- https://ca.linkedin.com/company/anvil-ai
- https://www.useanvil.com/
- https://www.anvilworks.com/about
- https://en.wikipedia.org/wiki/Swagelok
